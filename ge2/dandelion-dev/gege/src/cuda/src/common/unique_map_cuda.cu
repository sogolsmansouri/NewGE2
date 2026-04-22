// unique_map_cuda.cu
#include "common/unique_map_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <limits>
#include <string>

#if defined(__has_include)
#if __has_include(<cuco/static_map.cuh>)
#define GEGE_HAS_CUCO 1
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>
#include <thrust/functional.h>
#else
#define GEGE_HAS_CUCO 0
#endif
#else
#define GEGE_HAS_CUCO 0
#endif

namespace {

struct ByteBufferCache {
    torch::Tensor storage;
    int64_t capacity = 0;
};

struct UniqueMapSortWorkspaceCache {
    torch::Tensor orig_indices;
    torch::Tensor sorted_keys;
    torch::Tensor sorted_orig_indices;
    torch::Tensor unique_keys_full;
    torch::Tensor run_counts;
    torch::Tensor run_offsets;
    torch::Tensor sorted_run_ids;
    int64_t capacity = 0;
    torch::Device device = torch::Device(torch::kCPU);
};

struct UniqueMapSortTempCache {
    ByteBufferCache sort_tmp;
    ByteBufferCache rle_tmp;
    ByteBufferCache scan_tmp;
};

struct UniqueMapHashWorkspaceCache {
    torch::Tensor table_epoch;   // int32
    torch::Tensor table_keys;    // int64
    torch::Tensor table_uids;    // int64
    torch::Tensor unique_values; // int64
    int64_t table_capacity = 0;
    int64_t value_capacity = 0;
    int32_t current_epoch = 2;
    torch::Device device = torch::Device(torch::kCPU);
};

struct UniqueMapBitmapWorkspaceCache {
    torch::Tensor bitset_words;  // int64 storage for 64-bit presence bits
    torch::Tensor word_counts;   // int32 popcount per 64-bit word
    torch::Tensor word_offsets;  // int32 exclusive prefix over word_counts
    int64_t word_capacity = 0;
    torch::Device device = torch::Device(torch::kCPU);
};

struct UniqueMapBitmapTempCache {
    ByteBufferCache scan_tmp;
};

struct UniqueMapBitmapPaddedOutputCache {
    torch::Tensor unique_ids;           // int64
    torch::Tensor inverse;              // int64
    torch::Tensor active_mask;          // uint8
    torch::Tensor invalid_flag;         // int32[1]
    torch::Tensor inverse_invalid_flag; // int32[1]
    int64_t capacity = 0;
    torch::Device device = torch::Device(torch::kCPU);
};

struct BackendTimingAccumulator {
    explicit BackendTimingAccumulator(bool enabled_) : enabled(enabled_) {
        if (!enabled) return;
        AT_CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault));
        AT_CUDA_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));
    }

    ~BackendTimingAccumulator() {
        if (!enabled) return;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start_timing(cudaStream_t stream) {
        if (!enabled) return;
        AT_CUDA_CHECK(cudaEventRecord(start, stream));
    }

    void stop_timing(cudaStream_t stream) {
        if (!enabled) return;
        AT_CUDA_CHECK(cudaEventRecord(stop, stream));
        AT_CUDA_CHECK(cudaEventSynchronize(stop));
        float phase_ms = 0.0f;
        AT_CUDA_CHECK(cudaEventElapsedTime(&phase_ms, start, stop));
        elapsed_ms += static_cast<double>(phase_ms);
    }

    void store(UniqueMapCudaDebugInfo* debug_info) const {
        if (debug_info == nullptr) return;
        debug_info->device_unique_ms = elapsed_ms;
        debug_info->device_unique_timing_valid = enabled;
    }

    bool enabled = false;
    double elapsed_ms = 0.0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

thread_local UniqueMapSortWorkspaceCache sort_workspace_cache;
thread_local UniqueMapSortTempCache sort_temp_cache;
thread_local UniqueMapHashWorkspaceCache hash_workspace_cache;
thread_local UniqueMapBitmapWorkspaceCache bitmap_workspace_cache;
thread_local UniqueMapBitmapTempCache bitmap_temp_cache;
thread_local UniqueMapBitmapPaddedOutputCache bitmap_padded_output_cache;
std::atomic<int64_t> unique_backend_call_counter{0};
std::atomic<int64_t> unique_backend_fallback_counter{0};

enum class UniqueBackend {
    kAuto,
    kSort,
    kHash,
    kBitmap,
    kCuco,
};

bool parse_env_flag(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") return false;
    if (value == "1" || value == "true" || value == "True" || value == "TRUE") return true;
    return default_value;
}

int64_t parse_env_int64(const char* name, int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

bool hash_unique_enabled() {
    static bool enabled = parse_env_flag("GEGE_UNIQUE_HASH", false);
    return enabled;
}

bool fixed_buffer_bitmap_reuse_outputs_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_BITMAP_REUSE_OUTPUTS", false);
    return enabled;
}

UniqueBackend parse_unique_backend(const char* raw_value) {
    if (raw_value == nullptr) return UniqueBackend::kAuto;

    std::string value(raw_value);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (value == "sort") return UniqueBackend::kSort;
    if (value == "hash") return UniqueBackend::kHash;
    if (value == "bitmap") return UniqueBackend::kBitmap;
    if (value == "cuco") return UniqueBackend::kCuco;
    if (value == "auto") return UniqueBackend::kAuto;

    return UniqueBackend::kAuto;
}

UniqueBackend unique_backend() {
    static UniqueBackend backend = parse_unique_backend(std::getenv("GEGE_UNIQUE_BACKEND"));
    return backend;
}

UniqueBackend resolve_unique_backend(const std::string& override_backend) {
    if (!override_backend.empty()) {
        return parse_unique_backend(override_backend.c_str());
    }
    return unique_backend();
}

const char* unique_backend_name(UniqueBackend backend) {
    switch (backend) {
        case UniqueBackend::kSort:
            return "sort";
        case UniqueBackend::kHash:
            return "hash";
        case UniqueBackend::kBitmap:
            return "bitmap";
        case UniqueBackend::kCuco:
            return "cuco";
        case UniqueBackend::kAuto:
        default:
            return "auto";
    }
}

void initialize_debug_info(UniqueMapCudaDebugInfo* debug_info, bool sorted, UniqueBackend requested_backend) {
    int64_t total_calls = unique_backend_call_counter.fetch_add(1) + 1;
    if (debug_info == nullptr) return;

    bool measure_device_timing = debug_info->measure_device_timing;
    debug_info->requested_backend = unique_backend_name(requested_backend);
    debug_info->executed_backend.clear();
    debug_info->fallback_backend.clear();
    debug_info->fallback_reason.clear();
    debug_info->total_calls = total_calls;
    debug_info->total_fallbacks = unique_backend_fallback_counter.load();
    debug_info->device_unique_ms = 0.0;
    debug_info->device_bitmap_zero_ms = 0.0;
    debug_info->device_bitmap_output_init_ms = 0.0;
    debug_info->device_bitmap_mark_ms = 0.0;
    debug_info->device_bitmap_count_ms = 0.0;
    debug_info->device_bitmap_scan_ms = 0.0;
    debug_info->device_bitmap_extract_ms = 0.0;
    debug_info->device_bitmap_mask_ms = 0.0;
    debug_info->device_bitmap_inverse_ms = 0.0;
    debug_info->used_fallback = false;
    debug_info->cuco_compiled = GEGE_HAS_CUCO != 0;
    debug_info->device_unique_timing_valid = false;
    debug_info->measure_device_timing = measure_device_timing;
    debug_info->sorted_request = sorted;
}

void mark_executed_backend(UniqueMapCudaDebugInfo* debug_info, const char* backend_name) {
    if (debug_info == nullptr) return;
    debug_info->executed_backend = backend_name;
    debug_info->total_fallbacks = unique_backend_fallback_counter.load();
}

void mark_fallback(UniqueMapCudaDebugInfo* debug_info, const char* fallback_backend, const char* reason) {
    int64_t total_fallbacks = unique_backend_fallback_counter.fetch_add(1) + 1;
    if (debug_info == nullptr) return;

    debug_info->used_fallback = true;
    debug_info->fallback_backend = fallback_backend;
    debug_info->fallback_reason = reason;
    debug_info->executed_backend = fallback_backend;
    debug_info->total_fallbacks = total_fallbacks;
}

torch::Tensor reserve_byte_buffer(ByteBufferCache& cache, const torch::Device& device, int64_t bytes) {
    if (bytes <= 0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    }
    if (!cache.storage.defined() || cache.storage.device() != device || cache.capacity < bytes) {
        int64_t next_capacity = std::max<int64_t>(bytes, std::max<int64_t>(cache.capacity * 2, int64_t{4096}));
        cache.storage = torch::empty({next_capacity}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
        cache.capacity = next_capacity;
    }
    return cache.storage.narrow(0, 0, bytes);
}

int64_t next_power_of_two(int64_t x) {
    int64_t p = 1;
    while (p < x) {
        TORCH_CHECK(p <= (std::numeric_limits<int64_t>::max() >> 1), "next_power_of_two overflow");
        p <<= 1;
    }
    return p;
}

void ensure_sort_workspace(const torch::Tensor& all_ids, int64_t n) {
    auto device = all_ids.device();
    if (!sort_workspace_cache.orig_indices.defined() || sort_workspace_cache.device != device || sort_workspace_cache.capacity < n) {
        int64_t next_capacity = std::max<int64_t>(n, std::max<int64_t>(sort_workspace_cache.capacity * 2, int64_t{4096}));
        auto key_opts = all_ids.options();
        auto index_opts = all_ids.options().dtype(torch::kInt64);
        auto count_opts = all_ids.options().dtype(torch::kInt32);

        sort_workspace_cache.orig_indices = torch::arange(next_capacity, index_opts);
        sort_workspace_cache.sorted_keys = torch::empty({next_capacity}, key_opts);
        sort_workspace_cache.sorted_orig_indices = torch::empty({next_capacity}, index_opts);
        sort_workspace_cache.unique_keys_full = torch::empty({next_capacity}, key_opts);
        sort_workspace_cache.run_counts = torch::empty({next_capacity}, count_opts);
        sort_workspace_cache.run_offsets = torch::empty({next_capacity}, count_opts);
        sort_workspace_cache.sorted_run_ids = torch::empty({next_capacity}, index_opts);
        sort_workspace_cache.capacity = next_capacity;
        sort_workspace_cache.device = device;
    }
}

void ensure_hash_workspace(const torch::Tensor& all_ids, int64_t table_capacity, int64_t value_capacity) {
    auto device = all_ids.device();
    bool need_realloc = !hash_workspace_cache.table_keys.defined() || hash_workspace_cache.device != device ||
                        hash_workspace_cache.table_capacity < table_capacity || hash_workspace_cache.value_capacity < value_capacity;

    if (need_realloc) {
        auto key_opts = all_ids.options().dtype(torch::kInt64);
        auto epoch_opts = all_ids.options().dtype(torch::kInt32);

        hash_workspace_cache.table_epoch = torch::zeros({table_capacity}, epoch_opts);
        hash_workspace_cache.table_keys = torch::empty({table_capacity}, key_opts);
        hash_workspace_cache.table_uids = torch::empty({table_capacity}, key_opts);
        hash_workspace_cache.unique_values = torch::empty({value_capacity}, key_opts);
        hash_workspace_cache.table_capacity = table_capacity;
        hash_workspace_cache.value_capacity = value_capacity;
        hash_workspace_cache.current_epoch = 2;
        hash_workspace_cache.device = device;
    }
}

void ensure_bitmap_workspace(const torch::Tensor& all_ids, int64_t word_capacity) {
    auto device = all_ids.device();
    if (!bitmap_workspace_cache.bitset_words.defined() || bitmap_workspace_cache.device != device || bitmap_workspace_cache.word_capacity < word_capacity) {
        int64_t next_capacity = std::max<int64_t>(word_capacity, std::max<int64_t>(bitmap_workspace_cache.word_capacity * 2, int64_t{256}));
        auto bitset_opts = all_ids.options().dtype(torch::kInt64);
        auto count_opts = all_ids.options().dtype(torch::kInt32);

        bitmap_workspace_cache.bitset_words = torch::empty({next_capacity}, bitset_opts);
        bitmap_workspace_cache.word_counts = torch::empty({next_capacity}, count_opts);
        bitmap_workspace_cache.word_offsets = torch::empty({next_capacity}, count_opts);
        bitmap_workspace_cache.word_capacity = next_capacity;
        bitmap_workspace_cache.device = device;
    }
}

void ensure_bitmap_padded_output_workspace(const torch::Tensor& all_ids, int64_t n) {
    auto device = all_ids.device();
    if (!bitmap_padded_output_cache.unique_ids.defined() || bitmap_padded_output_cache.device != device ||
        bitmap_padded_output_cache.capacity < n) {
        int64_t next_capacity = std::max<int64_t>(n, std::max<int64_t>(bitmap_padded_output_cache.capacity * 2, int64_t{4096}));
        auto id_opts = all_ids.options().dtype(torch::kInt64);
        auto mask_opts = all_ids.options().dtype(torch::kUInt8);
        auto flag_opts = all_ids.options().dtype(torch::kInt32);

        bitmap_padded_output_cache.unique_ids = torch::empty({next_capacity}, id_opts);
        bitmap_padded_output_cache.inverse = torch::empty({next_capacity}, id_opts);
        bitmap_padded_output_cache.active_mask = torch::empty({next_capacity}, mask_opts);
        bitmap_padded_output_cache.invalid_flag = torch::empty({1}, flag_opts);
        bitmap_padded_output_cache.inverse_invalid_flag = torch::empty({1}, flag_opts);
        bitmap_padded_output_cache.capacity = next_capacity;
        bitmap_padded_output_cache.device = device;
    }
}

int64_t resolve_bitmap_domain_size(int64_t requested_domain_size) {
    static int64_t env_domain_size = parse_env_int64("GEGE_UNIQUE_BITMAP_NUM_NODES", -1);
    if (requested_domain_size > 0) return requested_domain_size;
    if (env_domain_size > 0) return env_domain_size;
    return -1;
}

__device__ __forceinline__ int32_t upper_bound_offsets(const int32_t* offsets, int32_t m, int32_t x) {
    int32_t lo = 0;
    int32_t hi = m;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (offsets[mid] <= x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void build_sorted_run_ids_kernel(const int32_t* run_offsets, int32_t num_runs, int64_t n, int64_t* sorted_run_ids) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;
    int32_t ub = upper_bound_offsets(run_offsets, num_runs, (int32_t)i);
    sorted_run_ids[i] = (int64_t)(ub - 1);
}

__global__ void scatter_inverse_kernel(const int64_t* sorted_orig_indices, const int64_t* sorted_run_ids, int64_t n, int64_t* inverse) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;
    inverse[sorted_orig_indices[i]] = sorted_run_ids[i];
}

__device__ __forceinline__ uint64_t mix_hash64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// Safer atomic load for int64 via atomicAdd(0) on 64-bit word.
__device__ __forceinline__ int64_t atomic_load_i64(int64_t* ptr) {
    auto uptr = reinterpret_cast<unsigned long long*>(ptr);
    return (int64_t)atomicAdd(uptr, 0ULL);
}

__device__ __forceinline__ int32_t atomic_load_i32(int32_t* ptr) {
    return atomicAdd(ptr, 0);
}

// Hash unique with epoch tagging + bounded probing.
// Guarantees progress: each loop iteration advances probes and slot when needed.
__global__ void build_hash_unique_kernel(const int64_t* input_keys,
                                         int64_t n,
                                         int32_t* table_epoch,
                                         int64_t* table_keys,
                                         int64_t* table_uids,
                                         int64_t mask,
                                         int32_t active_epoch,
                                         int32_t initializing_epoch,
                                         int32_t* unique_count,
                                         int64_t* unique_values,
                                         int32_t* overflow_flag,
                                         int64_t* inverse) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;

    int64_t key = input_keys[i];
    uint64_t slot = mix_hash64((uint64_t)key) & (uint64_t)mask;

    // Bounded linear probing: at most (mask+1) slots.
    for (int64_t probes = 0; probes <= mask; ++probes) {
        int32_t epoch = atomic_load_i32(&table_epoch[slot]);

        // If another thread is initializing this slot for this batch, skip to next slot.
        // (Avoid long spins; correctness is preserved by probing.)
        if (epoch == initializing_epoch) {
            slot = (slot + 1ULL) & (uint64_t)mask;
            continue;
        }

        // Active slot for this batch: check key match.
        if (epoch == active_epoch) {
            int64_t existing_key = table_keys[slot];
            if (existing_key == key) {
                // Atomic-load uid to avoid stale reads.
                int64_t uid = atomic_load_i64(&table_uids[slot]);
                // Re-check epoch and key; if still valid, accept.
                int32_t epoch2 = atomic_load_i32(&table_epoch[slot]);
                int64_t key2 = table_keys[slot];
                if (epoch2 == active_epoch && key2 == key && uid >= 0 && uid < n) {
                    inverse[i] = uid;
                    return;
                }
                // If it "should" match but uid invalid, flag overflow to fall back.
                atomicExch(overflow_flag, 1);
                inverse[i] = (int64_t)-1;
                return;
            }
            slot = (slot + 1ULL) & (uint64_t)mask;
            continue;
        }

        // Not active for this batch: try to claim slot by flipping epoch -> initializing_epoch.
        int32_t prev = atomicCAS(&table_epoch[slot], epoch, initializing_epoch);
        if (prev == epoch) {
            // We own this slot initialization.
            int64_t uid = (int64_t)atomicAdd(unique_count, 1);
            if (uid < 0 || uid >= n) {
                // Should never happen; signal fallback.
                atomicExch(overflow_flag, 1);
                // Restore epoch back to previous value (best-effort).
                atomicExch(&table_epoch[slot], epoch);
                inverse[i] = (int64_t)-1;
                return;
            }

            table_keys[slot] = key;
            unique_values[uid] = key;
            table_uids[slot] = uid;

            // Publish: ensure key/uid/value visible before marking active.
            __threadfence();
            atomicExch(&table_epoch[slot], active_epoch);

            inverse[i] = uid;
            return;
        }

        // CAS failed: retry this same slot so identical keys race toward a single published uid.
        continue;
    }

    // Table/probing overflow.
    atomicExch(overflow_flag, 1);
    inverse[i] = (int64_t)-1;
}

__global__ void mark_bitmap_presence_kernel(const int64_t* input_keys, int64_t n, int64_t value_domain_size, uint64_t* bitset_words, int32_t* invalid_flag) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;

    int64_t key = input_keys[i];
    if (key < 0 || key >= value_domain_size) {
        atomicExch(invalid_flag, 1);
        return;
    }

    uint64_t word_idx = static_cast<uint64_t>(key) >> 6;
    uint64_t bit_mask = 1ULL << (static_cast<uint64_t>(key) & 63ULL);
    atomicOr(reinterpret_cast<unsigned long long*>(&bitset_words[word_idx]), static_cast<unsigned long long>(bit_mask));
}

__global__ void mark_bitmap_presence_strided_kernel(const int64_t* input_keys,
                                                    int64_t n,
                                                    int64_t stride0,
                                                    int64_t value_domain_size,
                                                    uint64_t* bitset_words,
                                                    int32_t* invalid_flag) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;

    int64_t key = input_keys[i * stride0];
    if (key < 0 || key >= value_domain_size) {
        atomicExch(invalid_flag, 1);
        return;
    }

    uint64_t word_idx = static_cast<uint64_t>(key) >> 6;
    uint64_t bit_mask = 1ULL << (static_cast<uint64_t>(key) & 63ULL);
    atomicOr(reinterpret_cast<unsigned long long*>(&bitset_words[word_idx]), static_cast<unsigned long long>(bit_mask));
}

__global__ void count_bitmap_words_kernel(const uint64_t* bitset_words, int64_t num_words, int32_t* word_counts) {
    int64_t word_idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (word_idx >= num_words) return;

    word_counts[word_idx] = __popcll(static_cast<unsigned long long>(bitset_words[word_idx]));
}

__global__ void extract_bitmap_unique_ids_kernel(const uint64_t* bitset_words, const int32_t* word_offsets, int64_t num_words, int64_t* unique_ids) {
    int64_t word_idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (word_idx >= num_words) return;

    uint64_t bits = bitset_words[word_idx];
    int32_t out_idx = word_offsets[word_idx];
    while (bits != 0ULL) {
        int bit = __ffsll(static_cast<long long>(bits)) - 1;
        unique_ids[out_idx++] = (word_idx << 6) + bit;
        bits &= (bits - 1ULL);
    }
}

__global__ void mark_bitmap_unique_active_mask_kernel(const int32_t* word_counts, const int32_t* word_offsets, int64_t num_words, uint8_t* active_mask) {
    int64_t word_idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (word_idx >= num_words) return;

    int32_t count = word_counts[word_idx];
    int32_t offset = word_offsets[word_idx];
    for (int32_t idx = 0; idx < count; idx++) {
        active_mask[offset + idx] = uint8_t{1};
    }
}

__global__ void active_masked_adagrad_kernel(float* gradients,
                                             float* state,
                                             float* state_update,
                                             const uint8_t* active_mask,
                                             int64_t rows,
                                             int64_t dim,
                                             float learning_rate) {
    int64_t linear = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = rows * dim;
    if (linear >= total) return;

    int64_t row = linear / dim;
    if (active_mask[row] == 0) {
        gradients[linear] = 0.0f;
        state_update[linear] = 0.0f;
        return;
    }

    float grad = gradients[linear];
    float update = grad * grad;
    float new_state = state[linear] + update;
    state[linear] = new_state;
    state_update[linear] = update;
    gradients[linear] = -learning_rate * (grad / (sqrtf(new_state) + 1e-10f));
}

__global__ void active_masked_index_add_kernel(float* target,
                                               const int64_t* indices,
                                               const float* values,
                                               const uint8_t* active_mask,
                                               int64_t rows,
                                               int64_t dim,
                                               int64_t target_rows) {
    int64_t linear = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t total = rows * dim;
    if (linear >= total) return;

    int64_t row = linear / dim;
    if (active_mask[row] == 0) return;

    int64_t dst_row = indices[row];
    if (dst_row < 0 || dst_row >= target_rows) return;
    int64_t col = linear - row * dim;
    target[dst_row * dim + col] += values[linear];
}

__global__ void build_bitmap_inverse_kernel(const int64_t* input_keys,
                                            int64_t n,
                                            int64_t value_domain_size,
                                            const uint64_t* bitset_words,
                                            const int32_t* word_offsets,
                                            int32_t unique_count,
                                            int32_t* invalid_flag,
                                            int64_t* inverse) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;

    int64_t key = input_keys[i];
    if (key < 0 || key >= value_domain_size) {
        atomicExch(invalid_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    int64_t word_idx = key >> 6;
    int bit = static_cast<int>(key & 63LL);
    uint64_t bits = bitset_words[word_idx];
    if (((bits >> bit) & 1ULL) == 0ULL) {
        atomicExch(invalid_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    uint64_t lower_mask = (bit == 0) ? 0ULL : ((1ULL << bit) - 1ULL);
    int64_t uid = static_cast<int64_t>(word_offsets[word_idx]) + static_cast<int64_t>(__popcll(static_cast<unsigned long long>(bits & lower_mask)));
    if (uid < 0 || uid >= unique_count) {
        atomicExch(invalid_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    inverse[i] = uid;
}

__global__ void build_bitmap_inverse_strided_kernel(const int64_t* input_keys,
                                                    int64_t n,
                                                    int64_t stride0,
                                                    int64_t value_domain_size,
                                                    const uint64_t* bitset_words,
                                                    const int32_t* word_offsets,
                                                    int32_t unique_count,
                                                    int32_t* invalid_flag,
                                                    int64_t* inverse) {
    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;

    int64_t key = input_keys[i * stride0];
    if (key < 0 || key >= value_domain_size) {
        atomicExch(invalid_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    int64_t word_idx = key >> 6;
    int bit = static_cast<int>(key & 63LL);
    uint64_t bits = bitset_words[word_idx];
    if (((bits >> bit) & 1ULL) == 0ULL) {
        atomicExch(invalid_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    uint64_t lower_mask = (bit == 0) ? 0ULL : ((1ULL << bit) - 1ULL);
    int64_t uid = static_cast<int64_t>(word_offsets[word_idx]) + static_cast<int64_t>(__popcll(static_cast<unsigned long long>(bits & lower_mask)));
    if (uid < 0 || uid >= unique_count) {
        atomicExch(invalid_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    inverse[i] = uid;
}

#if GEGE_HAS_CUCO
template <typename MapRef>
__global__ void build_cuco_unique_kernel(MapRef map_ref,
                                         const int64_t* input_keys,
                                         int64_t n,
                                         int32_t* unique_count,
                                         int64_t* unique_values,
                                         int32_t* overflow_flag,
                                         int64_t* inverse) {
    constexpr int64_t kPendingUid = int64_t{-2};
    constexpr int64_t kErrorUid = int64_t{-3};
    constexpr int32_t kMaxPendingSpins = 1 << 20;

    int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (i >= n) return;

    int64_t key = input_keys[i];
    auto [slot, inserted] = map_ref.insert_and_find(cuco::pair<int64_t, int64_t>{key, kPendingUid});
    auto uid_ref = cuda::atomic_ref<int64_t, cuda::thread_scope_device>{slot->second};

    if (inserted) {
        int64_t uid = (int64_t)atomicAdd(unique_count, 1);
        if (uid < 0 || uid >= n) {
            atomicExch(overflow_flag, 1);
            uid_ref.store(kErrorUid, cuda::std::memory_order_release);
            inverse[i] = int64_t{-1};
            return;
        }

        unique_values[uid] = key;
        __threadfence();
        uid_ref.store(uid, cuda::std::memory_order_release);
        inverse[i] = uid;
        return;
    }

    int64_t uid = uid_ref.load(cuda::std::memory_order_acquire);
    int32_t spins = 0;
    while (uid == kPendingUid) {
        if (++spins >= kMaxPendingSpins) {
            atomicExch(overflow_flag, 1);
            inverse[i] = int64_t{-1};
            return;
        }
        uid = uid_ref.load(cuda::std::memory_order_acquire);
    }

    if (uid == kErrorUid || uid < 0 || uid >= n) {
        atomicExch(overflow_flag, 1);
        inverse[i] = int64_t{-1};
        return;
    }

    inverse[i] = uid;
}
#endif

std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_sort(torch::Tensor all_ids, UniqueMapCudaDebugInfo* debug_info) {
    TORCH_CHECK(all_ids.is_cuda(), "map_tensors_unique_inverse_cuda_sort expects a CUDA tensor");
    TORCH_CHECK(all_ids.scalar_type() == torch::kInt64, "map_tensors_unique_inverse_cuda_sort supports int64 tensors");
    TORCH_CHECK(all_ids.dim() == 1, "map_tensors_unique_inverse_cuda_sort expects a 1D tensor");
    mark_executed_backend(debug_info, "sort");

    int64_t n = all_ids.numel();
    auto index_opts = all_ids.options().dtype(torch::kInt64);
    if (n == 0) {
        return std::make_tuple(torch::empty({0}, all_ids.options()), torch::empty({0}, index_opts));
    }
    TORCH_CHECK(n <= std::numeric_limits<int>::max(), "sort unique supports up to INT_MAX elements");

    c10::cuda::CUDAGuard guard(all_ids.device());
    auto stream = at::cuda::getCurrentCUDAStream(all_ids.device().index()).stream();
    BackendTimingAccumulator timing(debug_info != nullptr && debug_info->measure_device_timing);

    ensure_sort_workspace(all_ids, n);

    auto orig_indices = sort_workspace_cache.orig_indices.narrow(0, 0, n);
    auto sorted_keys = sort_workspace_cache.sorted_keys.narrow(0, 0, n);
    auto sorted_orig_indices = sort_workspace_cache.sorted_orig_indices.narrow(0, 0, n);
    auto unique_keys_full = sort_workspace_cache.unique_keys_full.narrow(0, 0, n);
    auto run_counts = sort_workspace_cache.run_counts.narrow(0, 0, n);
    auto run_offsets = sort_workspace_cache.run_offsets.narrow(0, 0, n);
    auto sorted_run_ids = sort_workspace_cache.sorted_run_ids.narrow(0, 0, n);

    // Sort + RLE phase
    timing.start_timing(stream);
    size_t sort_tmp_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_tmp_bytes,
        all_ids.data_ptr<int64_t>(), sorted_keys.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>(), sorted_orig_indices.data_ptr<int64_t>(),
        (int)n, 0, sizeof(int64_t) * 8, stream));

    auto sort_tmp = reserve_byte_buffer(sort_temp_cache.sort_tmp, all_ids.device(), (int64_t)sort_tmp_bytes);
    AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        sort_tmp.data_ptr<uint8_t>(), sort_tmp_bytes,
        all_ids.data_ptr<int64_t>(), sorted_keys.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>(), sorted_orig_indices.data_ptr<int64_t>(),
        (int)n, 0, sizeof(int64_t) * 8, stream));

    // RLE
    auto num_runs = torch::zeros({1}, all_ids.options().dtype(torch::kInt32));
    size_t rle_tmp_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, rle_tmp_bytes,
        sorted_keys.data_ptr<int64_t>(),
        unique_keys_full.data_ptr<int64_t>(),
        run_counts.data_ptr<int32_t>(),
        num_runs.data_ptr<int32_t>(),
        (int)n, stream));

    auto rle_tmp = reserve_byte_buffer(sort_temp_cache.rle_tmp, all_ids.device(), (int64_t)rle_tmp_bytes);
    AT_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        rle_tmp.data_ptr<uint8_t>(), rle_tmp_bytes,
        sorted_keys.data_ptr<int64_t>(),
        unique_keys_full.data_ptr<int64_t>(),
        run_counts.data_ptr<int32_t>(),
        num_runs.data_ptr<int32_t>(),
        (int)n, stream));
    timing.stop_timing(stream);

    // Read num_runs
    int32_t h_num_runs = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_num_runs, num_runs.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(h_num_runs >= 1 && h_num_runs <= n, "invalid run count");

    // Scan + inverse phase
    timing.start_timing(stream);
    size_t scan_tmp_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_tmp_bytes,
        run_counts.data_ptr<int32_t>(),
        run_offsets.data_ptr<int32_t>(),
        h_num_runs, stream));

    auto scan_tmp = reserve_byte_buffer(sort_temp_cache.scan_tmp, all_ids.device(), (int64_t)scan_tmp_bytes);
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_tmp.data_ptr<uint8_t>(), scan_tmp_bytes,
        run_counts.data_ptr<int32_t>(),
        run_offsets.data_ptr<int32_t>(),
        h_num_runs, stream));

    // Build run ids and scatter inverse
    constexpr int kThreads = 256;
    int blocks = (int)((n + kThreads - 1) / kThreads);

    build_sorted_run_ids_kernel<<<blocks, kThreads, 0, stream>>>(
        run_offsets.data_ptr<int32_t>(), h_num_runs, n, sorted_run_ids.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());

    auto inverse = torch::empty({n}, index_opts);
    scatter_inverse_kernel<<<blocks, kThreads, 0, stream>>>(
        sorted_orig_indices.data_ptr<int64_t>(), sorted_run_ids.data_ptr<int64_t>(), n, inverse.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    timing.stop_timing(stream);

    auto unique_ids = unique_keys_full.narrow(0, 0, h_num_runs);
    timing.store(debug_info);
    return std::make_tuple(unique_ids, inverse);
}

std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_hash(torch::Tensor all_ids, UniqueMapCudaDebugInfo* debug_info) {
    TORCH_CHECK(all_ids.is_cuda(), "map_tensors_unique_inverse_cuda_hash expects a CUDA tensor");
    TORCH_CHECK(all_ids.scalar_type() == torch::kInt64, "hash unique supports int64 tensors");
    TORCH_CHECK(all_ids.dim() == 1, "hash unique expects a 1D tensor");
    mark_executed_backend(debug_info, "hash");

    int64_t n = all_ids.numel();
    auto index_opts = all_ids.options().dtype(torch::kInt64);
    if (n == 0) {
        return std::make_tuple(torch::empty({0}, all_ids.options()), torch::empty({0}, index_opts));
    }
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "hash unique supports up to INT32_MAX elements");

    c10::cuda::CUDAGuard guard(all_ids.device());
    auto stream = at::cuda::getCurrentCUDAStream(all_ids.device().index()).stream();
    BackendTimingAccumulator timing(debug_info != nullptr && debug_info->measure_device_timing);

    int64_t target_capacity = std::max<int64_t>(1024, n * 2);
    int64_t table_capacity = next_power_of_two(target_capacity);
    int64_t table_mask = table_capacity - 1;

    ensure_hash_workspace(all_ids, table_capacity, n);

    auto table_epoch = hash_workspace_cache.table_epoch.narrow(0, 0, table_capacity);
    auto table_keys  = hash_workspace_cache.table_keys.narrow(0, 0, table_capacity);
    auto table_uids  = hash_workspace_cache.table_uids.narrow(0, 0, table_capacity);
    auto unique_vals = hash_workspace_cache.unique_values.narrow(0, 0, n);

    auto inverse = torch::empty({n}, index_opts);
    auto unique_count = torch::zeros({1}, all_ids.options().dtype(torch::kInt32));
    auto overflow_flag = torch::zeros({1}, all_ids.options().dtype(torch::kInt32));

    // Epoch management
    if (hash_workspace_cache.current_epoch >= std::numeric_limits<int32_t>::max() - 2) {
        table_epoch.zero_();
        hash_workspace_cache.current_epoch = 2;
    }
    int32_t active_epoch = hash_workspace_cache.current_epoch;
    int32_t initializing_epoch = active_epoch + 1;
    hash_workspace_cache.current_epoch += 2;

    constexpr int kThreads = 256;
    int blocks = (int)((n + kThreads - 1) / kThreads);

    timing.start_timing(stream);
    build_hash_unique_kernel<<<blocks, kThreads, 0, stream>>>(
        all_ids.data_ptr<int64_t>(),
        n,
        table_epoch.data_ptr<int32_t>(),
        table_keys.data_ptr<int64_t>(),
        table_uids.data_ptr<int64_t>(),
        table_mask,
        active_epoch,
        initializing_epoch,
        unique_count.data_ptr<int32_t>(),
        unique_vals.data_ptr<int64_t>(),
        overflow_flag.data_ptr<int32_t>(),
        inverse.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    timing.stop_timing(stream);

    int32_t h_unique_count = 0;
    int32_t h_overflow_flag = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_unique_count, unique_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_overflow_flag, overflow_flag.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_overflow_flag != 0) {
        mark_fallback(debug_info, "sort", "hash_overflow_flag");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }
    TORCH_CHECK(h_unique_count >= 1 && h_unique_count <= n, "invalid hash unique count");

    // Safety net: inverse indices must index into unique_ids [0, h_unique_count).
    timing.start_timing(stream);
    auto inverse_min_tensor = inverse.min();
    auto inverse_max_tensor = inverse.max();
    timing.stop_timing(stream);
    int64_t inverse_min = inverse_min_tensor.item<int64_t>();
    int64_t inverse_max = inverse_max_tensor.item<int64_t>();
    if (inverse_min < 0 || inverse_max >= h_unique_count) {
        mark_fallback(debug_info, "sort", "hash_invalid_inverse");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    auto unique_ids = unique_vals.narrow(0, 0, h_unique_count);
    timing.store(debug_info);
    return std::make_tuple(unique_ids, inverse);
}

std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_bitmap(torch::Tensor all_ids, UniqueMapCudaDebugInfo* debug_info,
                                                                                int64_t value_domain_size) {
    TORCH_CHECK(all_ids.is_cuda(), "map_tensors_unique_inverse_cuda_bitmap expects a CUDA tensor");
    TORCH_CHECK(all_ids.scalar_type() == torch::kInt64, "bitmap unique supports int64 tensors");
    TORCH_CHECK(all_ids.dim() == 1, "bitmap unique expects a 1D tensor");
    mark_executed_backend(debug_info, "bitmap");

    int64_t n = all_ids.numel();
    auto index_opts = all_ids.options().dtype(torch::kInt64);
    if (n == 0) {
        return std::make_tuple(torch::empty({0}, all_ids.options()), torch::empty({0}, index_opts));
    }
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "bitmap unique supports up to INT32_MAX elements");

    int64_t domain_size = resolve_bitmap_domain_size(value_domain_size);
    if (domain_size <= 0) {
        mark_fallback(debug_info, "sort", "bitmap_missing_domain_size");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }
    TORCH_CHECK(domain_size <= std::numeric_limits<int64_t>::max() - 63, "bitmap unique domain size is too large");

    int64_t num_words = (domain_size + 63) >> 6;
    TORCH_CHECK(num_words >= 1, "bitmap unique requires a positive domain size");
    TORCH_CHECK(num_words <= std::numeric_limits<int>::max(), "bitmap unique supports up to INT_MAX 64-bit words");

    c10::cuda::CUDAGuard guard(all_ids.device());
    auto stream = at::cuda::getCurrentCUDAStream(all_ids.device().index()).stream();
    BackendTimingAccumulator timing(debug_info != nullptr && debug_info->measure_device_timing);

    ensure_bitmap_workspace(all_ids, num_words);

    auto bitset_words = bitmap_workspace_cache.bitset_words.narrow(0, 0, num_words);
    auto word_counts = bitmap_workspace_cache.word_counts.narrow(0, 0, num_words);
    auto word_offsets = bitmap_workspace_cache.word_offsets.narrow(0, 0, num_words);

    timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(bitset_words.data_ptr<int64_t>(), 0, static_cast<size_t>(num_words) * sizeof(int64_t), stream));

    auto flag_opts = all_ids.options().dtype(torch::kInt32);
    auto invalid_flag = torch::zeros({1}, flag_opts);

    constexpr int kThreads = 256;
    int input_blocks = (int)((n + kThreads - 1) / kThreads);
    int word_blocks = (int)((num_words + kThreads - 1) / kThreads);

    mark_bitmap_presence_kernel<<<input_blocks, kThreads, 0, stream>>>(
        all_ids.data_ptr<int64_t>(), n, domain_size, reinterpret_cast<uint64_t*>(bitset_words.data_ptr<int64_t>()), invalid_flag.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());

    count_bitmap_words_kernel<<<word_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()), num_words, word_counts.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());

    size_t scan_tmp_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_tmp_bytes, word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), (int)num_words, stream));

    auto scan_tmp = reserve_byte_buffer(bitmap_temp_cache.scan_tmp, all_ids.device(), static_cast<int64_t>(scan_tmp_bytes));
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_tmp.data_ptr<uint8_t>(), scan_tmp_bytes, word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), (int)num_words, stream));
    timing.stop_timing(stream);

    int32_t h_invalid_flag = 0;
    int32_t h_last_count = 0;
    int32_t h_last_offset = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_invalid_flag, invalid_flag.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_last_count, word_counts.data_ptr<int32_t>() + (num_words - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_last_offset, word_offsets.data_ptr<int32_t>() + (num_words - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_invalid_flag != 0) {
        mark_fallback(debug_info, "sort", "bitmap_input_out_of_range");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    int32_t h_unique_count = h_last_offset + h_last_count;
    if (h_unique_count < 1 || h_unique_count > n) {
        mark_fallback(debug_info, "sort", "bitmap_invalid_unique_count");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    auto unique_ids = torch::empty({h_unique_count}, all_ids.options());
    timing.start_timing(stream);
    extract_bitmap_unique_ids_kernel<<<word_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()), word_offsets.data_ptr<int32_t>(), num_words, unique_ids.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());

    auto inverse = torch::empty({n}, index_opts);
    auto inverse_invalid_flag = torch::zeros({1}, flag_opts);
    build_bitmap_inverse_kernel<<<input_blocks, kThreads, 0, stream>>>(
        all_ids.data_ptr<int64_t>(),
        n,
        domain_size,
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()),
        word_offsets.data_ptr<int32_t>(),
        h_unique_count,
        inverse_invalid_flag.data_ptr<int32_t>(),
        inverse.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    timing.stop_timing(stream);

    int32_t h_inverse_invalid_flag = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_inverse_invalid_flag, inverse_invalid_flag.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_inverse_invalid_flag != 0) {
        mark_fallback(debug_info, "sort", "bitmap_invalid_inverse");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    timing.store(debug_info);
    return std::make_tuple(unique_ids, inverse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_bitmap_padded_impl(torch::Tensor all_ids,
                                                                                                           UniqueMapCudaDebugInfo* debug_info,
                                                                                                           int64_t value_domain_size) {
    TORCH_CHECK(all_ids.is_cuda(), "map_tensors_unique_inverse_cuda_bitmap_padded expects a CUDA tensor");
    TORCH_CHECK(all_ids.scalar_type() == torch::kInt64, "bitmap padded unique supports int64 tensors");
    TORCH_CHECK(all_ids.dim() == 1, "bitmap padded unique expects a 1D tensor");
    mark_executed_backend(debug_info, "bitmap_padded");

    int64_t n = all_ids.numel();
    auto index_opts = all_ids.options().dtype(torch::kInt64);
    if (n == 0) {
        return std::make_tuple(torch::empty({0}, all_ids.options()),
                               torch::empty({0}, index_opts),
                               torch::empty({0}, all_ids.options().dtype(torch::kUInt8)));
    }
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "bitmap padded unique supports up to INT32_MAX elements");

    int64_t domain_size = resolve_bitmap_domain_size(value_domain_size);
    TORCH_CHECK(domain_size > 0, "GEGE_FIXED_BUFFER_BITMAP_MAP requires GEGE_UNIQUE_BITMAP_NUM_NODES or a positive value domain");
    TORCH_CHECK(domain_size <= std::numeric_limits<int64_t>::max() - 63, "bitmap padded unique domain size is too large");

    int64_t num_words = (domain_size + 63) >> 6;
    TORCH_CHECK(num_words >= 1, "bitmap padded unique requires a positive domain size");
    TORCH_CHECK(num_words <= std::numeric_limits<int>::max(), "bitmap padded unique supports up to INT_MAX 64-bit words");

    c10::cuda::CUDAGuard guard(all_ids.device());
    auto stream = at::cuda::getCurrentCUDAStream(all_ids.device().index()).stream();
    bool measure_device_timing = debug_info != nullptr && debug_info->measure_device_timing;
    BackendTimingAccumulator zero_timing(measure_device_timing);
    BackendTimingAccumulator output_init_timing(measure_device_timing);
    BackendTimingAccumulator mark_timing(measure_device_timing);
    BackendTimingAccumulator count_timing(measure_device_timing);
    BackendTimingAccumulator scan_timing(measure_device_timing);
    BackendTimingAccumulator extract_timing(measure_device_timing);
    BackendTimingAccumulator mask_timing(measure_device_timing);
    BackendTimingAccumulator inverse_timing(measure_device_timing);

    ensure_bitmap_workspace(all_ids, num_words);

    auto bitset_words = bitmap_workspace_cache.bitset_words.narrow(0, 0, num_words);
    auto word_counts = bitmap_workspace_cache.word_counts.narrow(0, 0, num_words);
    auto word_offsets = bitmap_workspace_cache.word_offsets.narrow(0, 0, num_words);

    zero_timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(bitset_words.data_ptr<int64_t>(), 0, static_cast<size_t>(num_words) * sizeof(int64_t), stream));
    zero_timing.stop_timing(stream);

    torch::Tensor unique_ids;
    torch::Tensor inverse;
    torch::Tensor active_mask;
    torch::Tensor invalid_flag;
    torch::Tensor inverse_invalid_flag;
    if (fixed_buffer_bitmap_reuse_outputs_enabled()) {
        ensure_bitmap_padded_output_workspace(all_ids, n);
        unique_ids = bitmap_padded_output_cache.unique_ids.narrow(0, 0, n);
        inverse = bitmap_padded_output_cache.inverse.narrow(0, 0, n);
        active_mask = bitmap_padded_output_cache.active_mask.narrow(0, 0, n);
        invalid_flag = bitmap_padded_output_cache.invalid_flag;
        inverse_invalid_flag = bitmap_padded_output_cache.inverse_invalid_flag;
        output_init_timing.start_timing(stream);
        AT_CUDA_CHECK(cudaMemsetAsync(invalid_flag.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(inverse_invalid_flag.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
        output_init_timing.stop_timing(stream);
    } else {
        auto flag_opts = all_ids.options().dtype(torch::kInt32);
        unique_ids = torch::empty({n}, all_ids.options());
        inverse = torch::empty({n}, index_opts);
        active_mask = torch::empty({n}, all_ids.options().dtype(torch::kUInt8));
        invalid_flag = torch::zeros({1}, flag_opts);
        inverse_invalid_flag = torch::zeros({1}, flag_opts);
    }

    constexpr int kThreads = 256;
    int input_blocks = (int)((n + kThreads - 1) / kThreads);
    int word_blocks = (int)((num_words + kThreads - 1) / kThreads);

    mark_timing.start_timing(stream);
    mark_bitmap_presence_kernel<<<input_blocks, kThreads, 0, stream>>>(
        all_ids.data_ptr<int64_t>(), n, domain_size, reinterpret_cast<uint64_t*>(bitset_words.data_ptr<int64_t>()), invalid_flag.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    mark_timing.stop_timing(stream);

    count_timing.start_timing(stream);
    count_bitmap_words_kernel<<<word_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()), num_words, word_counts.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    count_timing.stop_timing(stream);

    size_t scan_tmp_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_tmp_bytes, word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), (int)num_words, stream));

    auto scan_tmp = reserve_byte_buffer(bitmap_temp_cache.scan_tmp, all_ids.device(), static_cast<int64_t>(scan_tmp_bytes));
    scan_timing.start_timing(stream);
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_tmp.data_ptr<uint8_t>(), scan_tmp_bytes, word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), (int)num_words, stream));
    scan_timing.stop_timing(stream);

    output_init_timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(unique_ids.data_ptr<int64_t>(), 0, static_cast<size_t>(n) * sizeof(int64_t), stream));
    output_init_timing.stop_timing(stream);
    extract_timing.start_timing(stream);
    extract_bitmap_unique_ids_kernel<<<word_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()), word_offsets.data_ptr<int32_t>(), num_words, unique_ids.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    extract_timing.stop_timing(stream);

    output_init_timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(active_mask.data_ptr<uint8_t>(), 0, static_cast<size_t>(n) * sizeof(uint8_t), stream));
    output_init_timing.stop_timing(stream);
    mask_timing.start_timing(stream);
    mark_bitmap_unique_active_mask_kernel<<<word_blocks, kThreads, 0, stream>>>(
        word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), num_words, active_mask.data_ptr<uint8_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    mask_timing.stop_timing(stream);

    inverse_timing.start_timing(stream);
    build_bitmap_inverse_kernel<<<input_blocks, kThreads, 0, stream>>>(
        all_ids.data_ptr<int64_t>(),
        n,
        domain_size,
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()),
        word_offsets.data_ptr<int32_t>(),
        static_cast<int32_t>(n),
        inverse_invalid_flag.data_ptr<int32_t>(),
        inverse.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    inverse_timing.stop_timing(stream);

    if (debug_info != nullptr) {
        debug_info->device_bitmap_zero_ms = zero_timing.elapsed_ms;
        debug_info->device_bitmap_output_init_ms = output_init_timing.elapsed_ms;
        debug_info->device_bitmap_mark_ms = mark_timing.elapsed_ms;
        debug_info->device_bitmap_count_ms = count_timing.elapsed_ms;
        debug_info->device_bitmap_scan_ms = scan_timing.elapsed_ms;
        debug_info->device_bitmap_extract_ms = extract_timing.elapsed_ms;
        debug_info->device_bitmap_mask_ms = mask_timing.elapsed_ms;
        debug_info->device_bitmap_inverse_ms = inverse_timing.elapsed_ms;
        debug_info->device_unique_ms = zero_timing.elapsed_ms + output_init_timing.elapsed_ms + mark_timing.elapsed_ms +
                                       count_timing.elapsed_ms + scan_timing.elapsed_ms + extract_timing.elapsed_ms +
                                       mask_timing.elapsed_ms + inverse_timing.elapsed_ms;
        debug_info->device_unique_timing_valid = measure_device_timing;
    }
    return std::make_tuple(unique_ids, inverse, active_mask);
}

std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_cuco(torch::Tensor all_ids, UniqueMapCudaDebugInfo* debug_info) {
    TORCH_CHECK(all_ids.is_cuda(), "map_tensors_unique_inverse_cuda_cuco expects a CUDA tensor");
    TORCH_CHECK(all_ids.scalar_type() == torch::kInt64, "cuco unique supports int64 tensors");
    TORCH_CHECK(all_ids.dim() == 1, "cuco unique expects a 1D tensor");
    mark_executed_backend(debug_info, "cuco");

    auto index_opts = all_ids.options().dtype(torch::kInt64);
    int64_t n = all_ids.numel();
    if (n == 0) {
        return std::make_tuple(torch::empty({0}, all_ids.options()), torch::empty({0}, index_opts));
    }

#if GEGE_HAS_CUCO
    c10::cuda::CUDAGuard guard(all_ids.device());
    auto stream = at::cuda::getCurrentCUDAStream(all_ids.device().index()).stream();
    auto cuco_stream = cuco::cuda_stream_ref{stream};
    BackendTimingAccumulator timing(debug_info != nullptr && debug_info->measure_device_timing);

    constexpr int64_t kEmptyKeySentinel = std::numeric_limits<int64_t>::min();
    constexpr int64_t kEmptyValueSentinel = int64_t{-1};

    timing.start_timing(stream);
    auto min_key_tensor = all_ids.min();
    timing.stop_timing(stream);
    int64_t min_key = min_key_tensor.item<int64_t>();
    if (min_key == kEmptyKeySentinel) {
        if (hash_unique_enabled()) {
            mark_fallback(debug_info, "hash", "cuco_empty_key_sentinel_collision");
            return map_tensors_unique_inverse_cuda_hash(all_ids, debug_info);
        }
        mark_fallback(debug_info, "sort", "cuco_empty_key_sentinel_collision");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    using key_type = int64_t;
    using mapped_type = int64_t;
    using probe_type = cuco::linear_probing<1, cuco::default_hash_function<key_type>>;
    using map_type = cuco::static_map<key_type,
                                      mapped_type,
                                      cuco::extent<std::size_t>,
                                      cuda::thread_scope_device,
                                      thrust::equal_to<key_type>,
                                      probe_type>;
    std::size_t map_capacity = static_cast<std::size_t>(std::max<int64_t>(1024, n * 2));

    map_type map{
        map_capacity,
        cuco::empty_key<key_type>{kEmptyKeySentinel},
        cuco::empty_value<mapped_type>{kEmptyValueSentinel},
        thrust::equal_to<key_type>{},
        probe_type{},
        {},
        {},
        {},
        cuco_stream};

    auto unique_vals = torch::empty({n}, all_ids.options().dtype(torch::kInt64));
    auto inverse = torch::empty({n}, index_opts);
    auto unique_count = torch::zeros({1}, all_ids.options().dtype(torch::kInt32));
    auto overflow_flag = torch::zeros({1}, all_ids.options().dtype(torch::kInt32));

    auto map_ref = map.ref(cuco::insert_and_find);

    constexpr int kThreads = 256;
    int blocks = (int)((n + kThreads - 1) / kThreads);
    timing.start_timing(stream);
    build_cuco_unique_kernel<<<blocks, kThreads, 0, stream>>>(
        map_ref,
        all_ids.data_ptr<int64_t>(),
        n,
        unique_count.data_ptr<int32_t>(),
        unique_vals.data_ptr<int64_t>(),
        overflow_flag.data_ptr<int32_t>(),
        inverse.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    timing.stop_timing(stream);

    int32_t h_unique_count = 0;
    int32_t h_overflow_flag = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_unique_count, unique_count.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaMemcpyAsync(&h_overflow_flag, overflow_flag.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (h_overflow_flag != 0) {
        mark_fallback(debug_info, "sort", "cuco_overflow_flag");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }
    if (h_unique_count < 1 || h_unique_count > n) {
        mark_fallback(debug_info, "sort", "cuco_invalid_unique_count");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    timing.start_timing(stream);
    auto inverse_min_tensor = inverse.min();
    auto inverse_max_tensor = inverse.max();
    timing.stop_timing(stream);
    int64_t inverse_min = inverse_min_tensor.item<int64_t>();
    int64_t inverse_max = inverse_max_tensor.item<int64_t>();
    if (inverse_min < 0 || inverse_max >= h_unique_count) {
        mark_fallback(debug_info, "sort", "cuco_invalid_inverse");
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    auto unique_ids = unique_vals.narrow(0, 0, h_unique_count);
    timing.store(debug_info);
    return std::make_tuple(unique_ids, inverse);
#else
    if (hash_unique_enabled()) {
        mark_fallback(debug_info, "hash", "cuco_not_compiled");
        return map_tensors_unique_inverse_cuda_hash(all_ids, debug_info);
    }
    mark_fallback(debug_info, "sort", "cuco_not_compiled");
    return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
#endif
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_bitmap_padded(torch::Tensor all_ids,
                                                                                                      UniqueMapCudaDebugInfo* debug_info,
                                                                                                      int64_t value_domain_size) {
    initialize_debug_info(debug_info, false, UniqueBackend::kBitmap);
    return map_tensors_unique_inverse_cuda_bitmap_padded_impl(all_ids, debug_info, value_domain_size);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> map_tensors_unique_inverse_cuda_bitmap_padded_multi(
    const std::vector<torch::Tensor>& ids,
    UniqueMapCudaDebugInfo* debug_info,
    int64_t value_domain_size) {
    initialize_debug_info(debug_info, false, UniqueBackend::kBitmap);
    mark_executed_backend(debug_info, "bitmap_padded_multi");
    TORCH_CHECK(!ids.empty(), "bitmap padded multi unique expects at least one input tensor");

    const torch::Tensor& base = ids.front();
    TORCH_CHECK(base.is_cuda(), "bitmap padded multi unique expects CUDA tensors");
    TORCH_CHECK(base.scalar_type() == torch::kInt64, "bitmap padded multi unique supports int64 tensors");
    TORCH_CHECK(base.dim() == 1, "bitmap padded multi unique expects 1D tensors");

    int64_t total_n = 0;
    for (const auto& tensor : ids) {
        TORCH_CHECK(tensor.is_cuda(), "bitmap padded multi unique expects CUDA tensors");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt64, "bitmap padded multi unique supports int64 tensors");
        TORCH_CHECK(tensor.dim() == 1, "bitmap padded multi unique expects 1D tensors");
        TORCH_CHECK(tensor.device() == base.device(), "bitmap padded multi unique expects all tensors on the same device");
        TORCH_CHECK(tensor.stride(0) > 0, "bitmap padded multi unique expects positive 1D strides");
        total_n += tensor.numel();
    }

    auto index_opts = base.options().dtype(torch::kInt64);
    if (total_n == 0) {
        std::vector<torch::Tensor> mapped;
        mapped.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            mapped.emplace_back(torch::empty({0}, index_opts));
        }
        return std::make_tuple(torch::empty({0}, base.options()), mapped, torch::empty({0}, base.options().dtype(torch::kUInt8)));
    }
    TORCH_CHECK(total_n <= std::numeric_limits<int32_t>::max(), "bitmap padded multi unique supports up to INT32_MAX elements");

    int64_t domain_size = resolve_bitmap_domain_size(value_domain_size);
    TORCH_CHECK(domain_size > 0, "GEGE_FIXED_BUFFER_BITMAP_MAP requires GEGE_UNIQUE_BITMAP_NUM_NODES or a positive value domain");
    TORCH_CHECK(domain_size <= std::numeric_limits<int64_t>::max() - 63, "bitmap padded multi unique domain size is too large");

    int64_t num_words = (domain_size + 63) >> 6;
    TORCH_CHECK(num_words >= 1, "bitmap padded multi unique requires a positive domain size");
    TORCH_CHECK(num_words <= std::numeric_limits<int>::max(), "bitmap padded multi unique supports up to INT_MAX 64-bit words");

    c10::cuda::CUDAGuard guard(base.device());
    auto stream = at::cuda::getCurrentCUDAStream(base.device().index()).stream();
    bool measure_device_timing = debug_info != nullptr && debug_info->measure_device_timing;
    BackendTimingAccumulator zero_timing(measure_device_timing);
    BackendTimingAccumulator output_init_timing(measure_device_timing);
    BackendTimingAccumulator mark_timing(measure_device_timing);
    BackendTimingAccumulator count_timing(measure_device_timing);
    BackendTimingAccumulator scan_timing(measure_device_timing);
    BackendTimingAccumulator extract_timing(measure_device_timing);
    BackendTimingAccumulator mask_timing(measure_device_timing);
    BackendTimingAccumulator inverse_timing(measure_device_timing);

    ensure_bitmap_workspace(base, num_words);

    auto bitset_words = bitmap_workspace_cache.bitset_words.narrow(0, 0, num_words);
    auto word_counts = bitmap_workspace_cache.word_counts.narrow(0, 0, num_words);
    auto word_offsets = bitmap_workspace_cache.word_offsets.narrow(0, 0, num_words);

    torch::Tensor unique_ids;
    torch::Tensor inverse;
    torch::Tensor active_mask;
    torch::Tensor invalid_flag;
    torch::Tensor inverse_invalid_flag;
    if (fixed_buffer_bitmap_reuse_outputs_enabled()) {
        ensure_bitmap_padded_output_workspace(base, total_n);
        unique_ids = bitmap_padded_output_cache.unique_ids.narrow(0, 0, total_n);
        inverse = bitmap_padded_output_cache.inverse.narrow(0, 0, total_n);
        active_mask = bitmap_padded_output_cache.active_mask.narrow(0, 0, total_n);
        invalid_flag = bitmap_padded_output_cache.invalid_flag;
        inverse_invalid_flag = bitmap_padded_output_cache.inverse_invalid_flag;
        output_init_timing.start_timing(stream);
        AT_CUDA_CHECK(cudaMemsetAsync(invalid_flag.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(inverse_invalid_flag.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
        output_init_timing.stop_timing(stream);
    } else {
        auto flag_opts = base.options().dtype(torch::kInt32);
        unique_ids = torch::empty({total_n}, base.options());
        inverse = torch::empty({total_n}, index_opts);
        active_mask = torch::empty({total_n}, base.options().dtype(torch::kUInt8));
        invalid_flag = torch::zeros({1}, flag_opts);
        inverse_invalid_flag = torch::zeros({1}, flag_opts);
    }

    constexpr int kThreads = 256;
    int word_blocks = (int)((num_words + kThreads - 1) / kThreads);
    zero_timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(bitset_words.data_ptr<int64_t>(), 0, static_cast<size_t>(num_words) * sizeof(int64_t), stream));
    zero_timing.stop_timing(stream);

    mark_timing.start_timing(stream);
    for (const auto& tensor : ids) {
        int64_t n = tensor.numel();
        if (n == 0) continue;
        int input_blocks = (int)((n + kThreads - 1) / kThreads);
        mark_bitmap_presence_strided_kernel<<<input_blocks, kThreads, 0, stream>>>(
            tensor.data_ptr<int64_t>(),
            n,
            tensor.stride(0),
            domain_size,
            reinterpret_cast<uint64_t*>(bitset_words.data_ptr<int64_t>()),
            invalid_flag.data_ptr<int32_t>());
        AT_CUDA_CHECK(cudaGetLastError());
    }
    mark_timing.stop_timing(stream);

    count_timing.start_timing(stream);
    count_bitmap_words_kernel<<<word_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()), num_words, word_counts.data_ptr<int32_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    count_timing.stop_timing(stream);

    size_t scan_tmp_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, scan_tmp_bytes, word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), (int)num_words, stream));
    auto scan_tmp = reserve_byte_buffer(bitmap_temp_cache.scan_tmp, base.device(), static_cast<int64_t>(scan_tmp_bytes));
    scan_timing.start_timing(stream);
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        scan_tmp.data_ptr<uint8_t>(), scan_tmp_bytes, word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), (int)num_words, stream));
    scan_timing.stop_timing(stream);

    output_init_timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(unique_ids.data_ptr<int64_t>(), 0, static_cast<size_t>(total_n) * sizeof(int64_t), stream));
    output_init_timing.stop_timing(stream);
    extract_timing.start_timing(stream);
    extract_bitmap_unique_ids_kernel<<<word_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()), word_offsets.data_ptr<int32_t>(), num_words, unique_ids.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    extract_timing.stop_timing(stream);

    output_init_timing.start_timing(stream);
    AT_CUDA_CHECK(cudaMemsetAsync(active_mask.data_ptr<uint8_t>(), 0, static_cast<size_t>(total_n) * sizeof(uint8_t), stream));
    output_init_timing.stop_timing(stream);
    mask_timing.start_timing(stream);
    mark_bitmap_unique_active_mask_kernel<<<word_blocks, kThreads, 0, stream>>>(
        word_counts.data_ptr<int32_t>(), word_offsets.data_ptr<int32_t>(), num_words, active_mask.data_ptr<uint8_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    mask_timing.stop_timing(stream);

    std::vector<torch::Tensor> mapped;
    mapped.reserve(ids.size());
    int64_t offset = 0;
    inverse_timing.start_timing(stream);
    for (const auto& tensor : ids) {
        int64_t n = tensor.numel();
        torch::Tensor mapped_tensor = inverse.narrow(0, offset, n);
        mapped.emplace_back(mapped_tensor);
        if (n > 0) {
            int input_blocks = (int)((n + kThreads - 1) / kThreads);
            build_bitmap_inverse_strided_kernel<<<input_blocks, kThreads, 0, stream>>>(
                tensor.data_ptr<int64_t>(),
                n,
                tensor.stride(0),
                domain_size,
                reinterpret_cast<const uint64_t*>(bitset_words.data_ptr<int64_t>()),
                word_offsets.data_ptr<int32_t>(),
                static_cast<int32_t>(total_n),
                inverse_invalid_flag.data_ptr<int32_t>(),
                mapped_tensor.data_ptr<int64_t>());
            AT_CUDA_CHECK(cudaGetLastError());
        }
        offset += n;
    }
    inverse_timing.stop_timing(stream);

    if (debug_info != nullptr) {
        debug_info->device_bitmap_zero_ms = zero_timing.elapsed_ms;
        debug_info->device_bitmap_output_init_ms = output_init_timing.elapsed_ms;
        debug_info->device_bitmap_mark_ms = mark_timing.elapsed_ms;
        debug_info->device_bitmap_count_ms = count_timing.elapsed_ms;
        debug_info->device_bitmap_scan_ms = scan_timing.elapsed_ms;
        debug_info->device_bitmap_extract_ms = extract_timing.elapsed_ms;
        debug_info->device_bitmap_mask_ms = mask_timing.elapsed_ms;
        debug_info->device_bitmap_inverse_ms = inverse_timing.elapsed_ms;
        debug_info->device_unique_ms = zero_timing.elapsed_ms + output_init_timing.elapsed_ms + mark_timing.elapsed_ms +
                                       count_timing.elapsed_ms + scan_timing.elapsed_ms + extract_timing.elapsed_ms +
                                       mask_timing.elapsed_ms + inverse_timing.elapsed_ms;
        debug_info->device_unique_timing_valid = measure_device_timing;
    }
    return std::make_tuple(unique_ids, mapped, active_mask);
}

std::tuple<torch::Tensor, torch::Tensor> active_masked_adagrad_cuda(torch::Tensor gradients,
                                                                    torch::Tensor optimizer_state,
                                                                    torch::Tensor active_mask,
                                                                    double learning_rate) {
    TORCH_CHECK(gradients.is_cuda(), "active_masked_adagrad_cuda expects CUDA gradients");
    TORCH_CHECK(optimizer_state.is_cuda(), "active_masked_adagrad_cuda expects CUDA optimizer state");
    TORCH_CHECK(active_mask.is_cuda(), "active_masked_adagrad_cuda expects a CUDA active mask");
    TORCH_CHECK(gradients.scalar_type() == torch::kFloat32, "active_masked_adagrad_cuda supports float32 gradients");
    TORCH_CHECK(optimizer_state.scalar_type() == torch::kFloat32, "active_masked_adagrad_cuda supports float32 optimizer state");
    TORCH_CHECK(active_mask.scalar_type() == torch::kUInt8, "active_masked_adagrad_cuda expects uint8 active mask");
    TORCH_CHECK(gradients.dim() == 2 && optimizer_state.sizes() == gradients.sizes(), "active_masked_adagrad_cuda requires matching 2D gradient/state tensors");
    TORCH_CHECK(active_mask.dim() == 1 && active_mask.size(0) == gradients.size(0), "active_masked_adagrad_cuda active mask must match gradient rows");

    c10::cuda::CUDAGuard guard(gradients.device());
    auto stream = at::cuda::getCurrentCUDAStream(gradients.device().index()).stream();
    auto grad_contig = gradients.contiguous();
    auto state_contig = optimizer_state.contiguous();
    auto mask_contig = active_mask.contiguous();
    auto state_update = torch::empty_like(grad_contig);

    int64_t rows = grad_contig.size(0);
    int64_t dim = grad_contig.size(1);
    int64_t total = rows * dim;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    active_masked_adagrad_kernel<<<blocks, kThreads, 0, stream>>>(
        grad_contig.data_ptr<float>(),
        state_contig.data_ptr<float>(),
        state_update.data_ptr<float>(),
        mask_contig.data_ptr<uint8_t>(),
        rows,
        dim,
        static_cast<float>(learning_rate));
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(grad_contig, state_update);
}

void active_masked_index_add_cuda(torch::Tensor target,
                                  torch::Tensor indices,
                                  torch::Tensor values,
                                  torch::Tensor active_mask) {
    TORCH_CHECK(target.is_cuda(), "active_masked_index_add_cuda expects a CUDA target");
    TORCH_CHECK(indices.is_cuda(), "active_masked_index_add_cuda expects CUDA indices");
    TORCH_CHECK(values.is_cuda(), "active_masked_index_add_cuda expects CUDA values");
    TORCH_CHECK(active_mask.is_cuda(), "active_masked_index_add_cuda expects a CUDA active mask");
    TORCH_CHECK(target.scalar_type() == torch::kFloat32, "active_masked_index_add_cuda supports float32 target");
    TORCH_CHECK(values.scalar_type() == torch::kFloat32, "active_masked_index_add_cuda supports float32 values");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "active_masked_index_add_cuda expects int64 indices");
    TORCH_CHECK(active_mask.scalar_type() == torch::kUInt8, "active_masked_index_add_cuda expects uint8 active mask");
    TORCH_CHECK(target.dim() == 2 && values.dim() == 2, "active_masked_index_add_cuda requires 2D target/values tensors");
    TORCH_CHECK(indices.dim() == 1 && active_mask.dim() == 1, "active_masked_index_add_cuda requires 1D indices/mask tensors");
    TORCH_CHECK(indices.size(0) == values.size(0) && active_mask.size(0) == values.size(0), "active_masked_index_add_cuda row counts must match");
    TORCH_CHECK(target.size(1) == values.size(1), "active_masked_index_add_cuda embedding dimensions must match");

    c10::cuda::CUDAGuard guard(target.device());
    auto stream = at::cuda::getCurrentCUDAStream(target.device().index()).stream();
    auto target_contig = target.contiguous();
    TORCH_CHECK(target_contig.data_ptr<float>() == target.data_ptr<float>(), "active_masked_index_add_cuda target must be contiguous");
    auto indices_contig = indices.contiguous();
    auto values_contig = values.contiguous();
    auto mask_contig = active_mask.contiguous();

    int64_t rows = values_contig.size(0);
    int64_t dim = values_contig.size(1);
    int64_t total = rows * dim;
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    active_masked_index_add_kernel<<<blocks, kThreads, 0, stream>>>(
        target_contig.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        values_contig.data_ptr<float>(),
        mask_contig.data_ptr<uint8_t>(),
        rows,
        dim,
        target_contig.size(0));
    AT_CUDA_CHECK(cudaGetLastError());
}

std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda(torch::Tensor all_ids, bool sorted,
                                                                         UniqueMapCudaDebugInfo* debug_info,
                                                                         int64_t value_domain_size) {
    UniqueBackend requested_backend = resolve_unique_backend(std::string());
    initialize_debug_info(debug_info, sorted, requested_backend);
    if (sorted) {
        return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }

    switch (requested_backend) {
        case UniqueBackend::kSort:
            return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
        case UniqueBackend::kHash:
            return map_tensors_unique_inverse_cuda_hash(all_ids, debug_info);
        case UniqueBackend::kBitmap:
            return map_tensors_unique_inverse_cuda_bitmap(all_ids, debug_info, value_domain_size);
        case UniqueBackend::kCuco:
            return map_tensors_unique_inverse_cuda_cuco(all_ids, debug_info);
        case UniqueBackend::kAuto:
        default:
            if (hash_unique_enabled()) return map_tensors_unique_inverse_cuda_hash(all_ids, debug_info);
            return map_tensors_unique_inverse_cuda_sort(all_ids, debug_info);
    }
}
