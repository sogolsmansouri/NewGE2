#include "storage/buffer.h"

#include <common/util.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include <functional>
#include <future>
#include <shared_mutex>
#ifdef GEGE_CUDA
#include "common/unique_map_cuda.h"
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include "pytorch_scatter/segment_sum.h"
#endif

#include "configuration/constants.h"
#include "reporting/logger.h"

#if defined(GEGE_CUDA)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GEGE_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define GEGE_HAS_NVTX 1
#else
#define GEGE_HAS_NVTX 0
#endif
#else
#define GEGE_HAS_NVTX 0
#endif

namespace {

bool parse_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }

    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }

    return default_value;
}

int64_t parse_env_int(const char *name, int64_t default_value);

bool csr_update_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_UPDATE", false);
    return enabled;
}

bool csr_update_reduce_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_UPDATE_REDUCE", false);
    return enabled;
}

bool csr_nvtx_enabled() {
    static bool enabled = parse_env_flag("GEGE_CSR_NVTX", false);
    return enabled;
}

bool single_gpu_gpu_aware_custom_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM", false);
    return enabled;
}

bool single_gpu_async_admit_preload_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD", false);
    return enabled;
}

bool multi_gpu_async_admit_preload_enabled() {
    static bool enabled = parse_env_flag("GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD", false);
    return enabled;
}

bool single_gpu_async_evict_writeback_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK", false);
    return enabled;
}

bool single_gpu_async_evict_writeback_verify_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_ASYNC_EVICT_VERIFY", false);
    return enabled;
}

bool single_gpu_async_evict_source_verify_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_ASYNC_EVICT_SOURCE_VERIFY", false);
    return enabled;
}

bool mem_partition_buffer_dirty_profile_enabled() {
    static bool enabled = parse_env_flag("GEGE_MEM_PARTITION_BUFFER_DIRTY_PROFILE", false);
    return enabled;
}

bool single_gpu_async_stage_memory_guard_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_ASYNC_STAGE_MEMORY_GUARD", true);
    return enabled;
}

bool single_gpu_async_admit_host_stage_enabled() {
    static bool enabled = parse_env_flag("GEGE_SINGLE_GPU_ASYNC_ADMIT_HOST_STAGE", false);
    return enabled;
}

int64_t single_gpu_async_stage_memory_margin_bytes() {
    static int64_t margin_mib = std::max<int64_t>(parse_env_int("GEGE_SINGLE_GPU_ASYNC_STAGE_MEMORY_MARGIN_MIB", 1024), 0);
    return margin_mib * 1024LL * 1024LL;
}

std::mutex &single_gpu_async_stage_allocation_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool fixed_buffer_masked_update_verify_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY", false);
    return enabled;
}

bool startup_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_STARTUP_TIMING", false);
    return enabled;
}

bool mem_partition_buffer_pinned_host_enabled() {
    static bool enabled = parse_env_flag("GEGE_MEM_PARTITION_BUFFER_PINNED_HOST", false);
    return enabled;
}

bool mem_partition_buffer_host_storage_lock_enabled() {
    static bool enabled = parse_env_flag("GEGE_MEM_PARTITION_BUFFER_HOST_STORAGE_LOCK", false);
    return enabled;
}

bool local_id_validate_enabled_storage() {
    static bool enabled = parse_env_flag("GEGE_LOCAL_ID_VALIDATE", false);
    return enabled;
}

int64_t frame_cache_hidden_frames() {
    static int64_t hidden_frames = std::max<int64_t>(parse_env_int("GEGE_FRAME_CACHE_HIDDEN_FRAMES", 0), 0);
    return hidden_frames;
}

bool frame_cache_hidden_only_preload_enabled() {
    static bool enabled = parse_env_flag("GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD", false);
    return enabled;
}

bool frame_cache_delayed_stale_writeback_enabled() {
    static bool enabled = parse_env_flag("GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK", false);
    return enabled;
}

bool tensor_supports_non_blocking_host_copy(const torch::Tensor &tensor) {
    return tensor.defined() && tensor.is_pinned();
}

int64_t parse_env_int(const char *name, int64_t default_value);

bool partition_buffer_swap_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_SWAP_TIMING", false);
    return enabled;
}

int64_t partition_buffer_swap_timing_max() {
    static int64_t max_swaps = std::max<int64_t>(parse_env_int("GEGE_PARTITION_BUFFER_SWAP_TIMING_MAX", 128), 0);
    return max_swaps;
}

std::atomic<int64_t> &partition_buffer_swap_timing_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

int64_t fixed_buffer_masked_update_verify_max() {
    static int64_t max_checks = std::max<int64_t>(parse_env_int("GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY_MAX", 8), 0);
    return max_checks;
}

std::atomic<int64_t> &fixed_buffer_masked_index_add_verify_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_log_partition_buffer_swap_timing(int64_t &timing_id) {
    if (!partition_buffer_swap_timing_enabled()) {
        return false;
    }
    timing_id = partition_buffer_swap_timing_counter().fetch_add(1);
    return timing_id < partition_buffer_swap_timing_max();
}

int64_t parse_env_int(const char *name, int64_t default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

#ifdef GEGE_CUDA
bool try_allocate_async_cuda_stage(int64_t rows, int64_t embedding_size, torch::Dtype dtype, torch::Device device, int64_t bytes_per_row,
                                   torch::Tensor &stage, int64_t *free_bytes_out, int64_t *required_bytes_out) {
    if (free_bytes_out != nullptr) {
        *free_bytes_out = -1;
    }
    if (required_bytes_out != nullptr) {
        *required_bytes_out = -1;
    }
    if (rows <= 0) {
        return false;
    }

    int64_t allocation_bytes = rows * bytes_per_row;
    int64_t required_bytes = allocation_bytes + single_gpu_async_stage_memory_margin_bytes();
    if (required_bytes_out != nullptr) {
        *required_bytes_out = required_bytes;
    }

    std::lock_guard<std::mutex> allocation_lock(single_gpu_async_stage_allocation_mutex());
    c10::cuda::CUDAGuard device_guard(device);
    if (single_gpu_async_stage_memory_guard_enabled()) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        AT_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes_out != nullptr) {
            *free_bytes_out = static_cast<int64_t>(free_bytes);
        }
        if (static_cast<int64_t>(free_bytes) < required_bytes) {
            return false;
        }
    }

    stage = torch::empty({rows, embedding_size}, torch::TensorOptions().dtype(dtype).device(device));
    return true;
}
#endif

bool eval_finite_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_EVAL_FINITE_DEBUG", false);
    return enabled;
}

int64_t eval_finite_debug_max_logs() {
    static int64_t max_logs = std::max<int64_t>(parse_env_int("GEGE_EVAL_FINITE_DEBUG_MAX_LOGS", 8), 0);
    return max_logs;
}

std::atomic<int64_t> &eval_finite_debug_log_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_log_eval_finite_debug(int64_t &log_id) {
    if (!eval_finite_debug_enabled()) {
        return false;
    }

    int64_t current = eval_finite_debug_log_counter().fetch_add(1);
    if (current >= eval_finite_debug_max_logs()) {
        return false;
    }

    log_id = current;
    return true;
}

std::string tensor_shape_string(const torch::Tensor &tensor) {
    std::ostringstream oss;
    oss << "[";
    for (int64_t dim = 0; dim < tensor.dim(); dim++) {
        if (dim > 0) {
            oss << ", ";
        }
        oss << tensor.size(dim);
    }
    oss << "]";
    return oss.str();
}

std::string tensor_prefix_string(const torch::Tensor &tensor, int64_t limit) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return "[]";
    }

    torch::Tensor flat_cpu = tensor.flatten().to(torch::kCPU).to(torch::kInt64);
    int64_t count = std::min<int64_t>(flat_cpu.numel(), limit);
    std::ostringstream oss;
    oss << "[";
    auto accessor = flat_cpu.accessor<int64_t, 1>();
    for (int64_t i = 0; i < count; i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << accessor[i];
    }
    if (flat_cpu.numel() > count) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

std::string basename_string(const std::string &path) {
    std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

template <typename T>
std::string vector_prefix_string(const std::vector<T> &values, std::size_t limit = 8) {
    std::ostringstream oss;
    oss << "[";
    std::size_t count = std::min<std::size_t>(values.size(), limit);
    for (std::size_t i = 0; i < count; i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    if (values.size() > count) {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

int64_t partition_rows_for_ids(const std::vector<int> &partition_ids, const std::vector<Partition *> &partition_table) {
    int64_t rows = 0;
    for (int partition_id : partition_ids) {
        rows += partition_table[partition_id]->partition_size_;
    }
    return rows;
}

double bytes_to_mib(int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

#ifdef GEGE_CUDA
bool parse_storage_cuda_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }
    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }
    return default_value;
}

void synchronize_cuda_storage_device(const torch::Device &device) {
    if (!device.is_cuda()) {
        return;
    }

    c10::cuda::CUDAGuard device_guard(device);
    AT_CUDA_CHECK(cudaDeviceSynchronize());
}

bool storage_sync_before_swap_enabled() {
    static bool enabled = parse_storage_cuda_env_flag("GEGE_SYNC_BEFORE_SWAP", true);
    return enabled;
}

void empty_cache_for_storage_device(const torch::Device &device) {
    static bool sync_enabled = parse_storage_cuda_env_flag("GEGE_SYNC_BEFORE_SWAP", true);
    // Allocator cache flushes are process-global in PyTorch and can serialize
    // otherwise independent multi-GPU lanes. Keep swap-time flushing opt-in.
    static bool empty_cache_enabled = parse_storage_cuda_env_flag("GEGE_EMPTY_CACHE_AROUND_SWAP", false);
    if (!sync_enabled && !empty_cache_enabled) {
        return;
    }

    if (sync_enabled || empty_cache_enabled) {
        synchronize_cuda_storage_device(device);
    }
    if (empty_cache_enabled) {
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
}
#endif

void log_non_finite_rows_if_any(const char *stage, int64_t log_id, const torch::Device &device, int partition_id, const torch::Tensor &tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return;
    }

    torch::Tensor finite = torch::isfinite(tensor);
    if (finite.all().item<bool>()) {
        return;
    }

    int64_t invalid_values = (~finite).sum().item<int64_t>();
    int64_t invalid_rows = invalid_values;
    if (tensor.dim() >= 2) {
        invalid_rows = (~finite).reshape({tensor.size(0), -1}).any(1).sum().item<int64_t>();
    }

    SPDLOG_ERROR(
        "[eval-finite-debug][buffer {}][{}] device={} partition={} invalid_values={} invalid_rows={} shape={}",
        log_id, stage, device.str(), partition_id, invalid_values, invalid_rows, tensor_shape_string(tensor));
}

void log_non_finite_index_add_rows_if_any(const char *stage, const torch::Device &device, int64_t debug_update_id,
                                          const torch::Tensor &local_rows, const torch::Tensor &tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return;
    }

    torch::Tensor finite = torch::isfinite(tensor);
    if (finite.all().item<bool>()) {
        return;
    }

    int64_t log_id = -1;
    if (!should_log_eval_finite_debug(log_id)) {
        return;
    }

    int64_t invalid_values = (~finite).sum().item<int64_t>();
    torch::Tensor invalid_row_mask = (~finite).reshape({tensor.size(0), -1}).any(1);
    torch::Tensor invalid_row_indices = invalid_row_mask.nonzero().flatten();
    int64_t invalid_rows = invalid_row_indices.numel();
    int64_t sample_count = std::min<int64_t>(invalid_rows, 8);
    torch::Tensor sample_rows = torch::Tensor();
    if (sample_count > 0 && local_rows.defined()) {
        if (local_rows.numel() == tensor.size(0)) {
            sample_rows = local_rows.index_select(0, invalid_row_indices.slice(0, 0, sample_count));
        } else {
            sample_rows = local_rows.slice(0, 0, std::min<int64_t>(local_rows.numel(), sample_count));
        }
    }

    SPDLOG_ERROR(
        "[eval-finite-debug][indexAdd {}][{}] update_id={} device={} invalid_values={} invalid_rows={} tensor_shape={} sample_local_rows={}",
        log_id, stage, debug_update_id, device.str(), invalid_values, invalid_rows, tensor_shape_string(tensor),
        tensor_prefix_string(sample_rows, 8));
}

bool stage_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_STAGE_DEBUG", false);
    return enabled;
}

int64_t stage_debug_max_updates() {
    static int64_t max_updates = parse_env_int("GEGE_STAGE_DEBUG_MAX_UPDATES", 40);
    return std::max<int64_t>(max_updates, 0);
}

std::atomic<int64_t> &stage_debug_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_run_stage_debug(int64_t &debug_update_id) {
    if (!stage_debug_enabled()) {
        return false;
    }
    debug_update_id = stage_debug_counter().fetch_add(1);
    return debug_update_id < stage_debug_max_updates();
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int64_t get_active_slot_count(const torch::Tensor &buffer_state, int capacity, const char *context) {
    if (!buffer_state.defined()) {
        return 0;
    }

    int64_t active_slots = buffer_state.size(0);
    if (active_slots > capacity) {
        throw GegeRuntimeException(fmt::format("{} received {} active slots for capacity {}", context, active_slots, capacity));
    }

    return active_slots;
}

void checked_cpu_index_add_(const char *context, torch::Tensor &target, torch::Tensor indices, torch::Tensor values) {
    if (!target.device().is_cpu() || !indices.device().is_cpu() || !values.device().is_cpu()) {
        throw GegeRuntimeException(std::string(context) + " expects CPU tensors on the CPU update path");
    }

    if (target.scalar_type() != values.scalar_type()) {
        throw GegeRuntimeException(std::string(context) + " requires target and values to share dtype");
    }

    if (indices.scalar_type() != torch::kInt64) {
        indices = indices.to(torch::kInt64);
    }

    target.index_add_(0, indices, values);
}

class ScopedNvtxRange {
   public:
    explicit ScopedNvtxRange(const char *name) {
        active_ = false;
#if GEGE_HAS_NVTX
        if (csr_nvtx_enabled()) {
            nvtxRangePushA(name);
            active_ = true;
        }
#endif
    }

    ~ScopedNvtxRange() {
#if GEGE_HAS_NVTX
        if (active_) {
            nvtxRangePop();
        }
#endif
    }

   private:
    bool active_;
};

#ifdef GEGE_CUDA
std::tuple<torch::Tensor, torch::Tensor> reduce_updates_with_csr(torch::Tensor indices, torch::Tensor values) {
    ScopedNvtxRange nvtx_scope("buffer.reduce_updates_with_csr");

    if (!indices.defined() || !values.defined() || indices.numel() == 0 || values.numel() == 0) {
        return std::forward_as_tuple(indices, values);
    }

    torch::Tensor indices64 = indices.to(torch::kInt64);
    torch::Tensor permutation = torch::argsort(indices64);
    torch::Tensor sorted_indices = indices64.index_select(0, permutation);

    auto unique_tup = torch::unique_consecutive(sorted_indices, false, true);
    torch::Tensor unique_indices = std::get<0>(unique_tup);
    torch::Tensor counts = std::get<2>(unique_tup).to(torch::kInt64);

    if (unique_indices.numel() == sorted_indices.numel()) {
        return std::forward_as_tuple(indices64, values);
    }

    torch::Tensor sorted_values = values.index_select(0, permutation);
    auto indptr_opts = torch::TensorOptions().dtype(torch::kInt64).device(indices.device());
    torch::Tensor indptr = torch::zeros({unique_indices.numel() + 1}, indptr_opts);
    if (counts.numel() > 0) {
        indptr.narrow(0, 1, counts.numel()).copy_(counts.cumsum(0));
    }

    torch::Tensor reduced_values = segment_sum_csr(sorted_values, indptr, torch::nullopt);
    return std::forward_as_tuple(unique_indices, reduced_values);
}
#endif
}  // namespace

Partition::Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset) {
    lock_ = new std::mutex();
    cv_ = new std::condition_variable();
    data_ptr_ = nullptr;
    partition_id_ = partition_id;

    present_ = false;

    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);
    total_size_ = partition_size_ * embedding_size_ * dtype_size_;

    idx_offset_ = idx_offset;
    file_offset_ = file_offset;
    buffer_idx_ = -1;
    physical_frame_idx_ = -1;

    tensor_ = torch::Tensor();

    evicting_ = false;
}

Partition::~Partition() {
    delete lock_;
    delete cv_;
    tensor_ = torch::Tensor();
}

torch::Tensor Partition::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    lock_->lock();

    torch::Tensor ret = tensor_.index_select(0, indices - idx_offset_);

    lock_->unlock();
    cv_->notify_all();

    return ret;
}

PartitionedFile::PartitionedFile(string filename, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                                 torch::Dtype dtype) {
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);

    filename_ = filename;

    int flags = O_RDWR | IO_FLAGS;
    fd_ = open(filename_.c_str(), flags);
    if (fd_ == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void PartitionedFile::readPartition(void *addr, Partition *partition) {
    if (addr == NULL || partition == NULL) {
        // TODO: throw null ptr exception
        throw std::runtime_error("");
    }

    memset_wrapper(addr, 0, partition->total_size_);
    if (pread_wrapper(fd_, addr, partition->total_size_, partition->file_offset_) == -1) {
        SPDLOG_ERROR("Unable to read Block: {}\nError: {}", partition->partition_id_, errno);
        throw std::runtime_error("");
    }
    partition->data_ptr_ = addr;
    partition->tensor_ = torch::from_blob(addr, {partition->partition_size_, embedding_size_}, dtype_);
}

// writePartition accesses data pointed to by p->data_ptr_. Address p->data_ptr_ is expected to contain
// same data as that of p->tensor_.
void PartitionedFile::writePartition(Partition *partition, bool clear_mem) {
    if (partition == NULL || partition->data_ptr_ == nullptr) {
        // TODO: throw null ptr exception
        throw std::runtime_error("");
    }

    if (pwrite_wrapper(fd_, partition->data_ptr_, partition->total_size_, partition->file_offset_) == -1) {
        throw GegeRuntimeException(fmt::format("Unable to write partition: {}\nError: {}", partition->partition_id_, errno));
    }

    if (clear_mem) {
        memset_wrapper(partition->data_ptr_, 0, partition->total_size_);
        partition->data_ptr_ = nullptr;
        partition->tensor_ = torch::Tensor();
    }
}

LookaheadBlock::LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_lookahead) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;
    partitions_ = {};
    lock_ = new std::mutex();

    mems_ = std::vector<void *>(num_per_lookahead);

    for (int i = 0; i < num_per_lookahead; i++) {
        if (posix_memalign(&mems_[i], 4096, total_size_)) {
            SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(mems_[i], 0, total_size_);
    }

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

LookaheadBlock::~LookaheadBlock() {
    delete lock_;

    for (void *mem : mems_) {
        free(mem);
    }
}

void LookaheadBlock::run() {
    while (!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return done_ || present_ == false; });

        if (done_) {
            break;
        }

        if (partitions_.empty()) {
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < partitions_.size(); i++) {
            Partition *partition = partitions_[i];
            std::unique_lock partition_lock(*partition->lock_);
            partition->cv_->wait(partition_lock, [partition] { return partition->evicting_ == false; });
            partitioned_file_->readPartition(mems_[i], partition);
            partition_lock.unlock();
            partition->cv_->notify_all();
        }

        present_ = true;
        lock.unlock();
        cv_.notify_all();
    }
}

void LookaheadBlock::start(std::vector<Partition *> first_partitions) {
    partitions_ = first_partitions;
    if (thread_ == nullptr) {
        thread_ = new std::thread(&LookaheadBlock::run, this);
    }
}

void LookaheadBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            done_ = true;
            present_ = false;
            cv_.notify_all();
            thread_->join();
        }
        delete thread_;
    }
}

void LookaheadBlock::move_to_buffer(std::vector<void *> buff_addrs, std::vector<int64_t> buffer_idxs, std::vector<Partition *> next_partitions) {
    if (partitions_.size() > buff_addrs.size() || partitions_.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function exception
        throw std::runtime_error("");
    }
    // wait until block is populated
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == true; });

#pragma omp parallel for
    for (int i = 0; i < partitions_.size(); i++) {
        Partition *partition = partitions_[i];
        void *addr = buff_addrs[i];
        int64_t buffer_idx = buffer_idxs[i];
        memcpy_wrapper(addr, mems_[i], partition->total_size_);
        memset_wrapper(mems_[i], 0, partition->total_size_);

        partition->data_ptr_ = addr;
        partition->tensor_ = torch::from_blob(partition->data_ptr_, {partition->partition_size_, partition->embedding_size_}, partition->dtype_);
        partition->buffer_idx_ = buffer_idx;
        partition->present_ = true;
    }

    // next partition will be prefetched automatically
    partitions_ = next_partitions;
    present_ = false;
    lock.unlock();
    cv_.notify_all();
}

AsyncWriteBlock::AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_evict) {
    total_size_ = total_size;
    partitioned_file_ = partitioned_file;

    lock_ = new std::mutex();

    mems_ = std::vector<void *>(num_per_evict);

    for (int i = 0; i < num_per_evict; i++) {
        if (posix_memalign(&mems_[i], 4096, total_size_)) {
            SPDLOG_ERROR("Unable to allocate lookahead memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(mems_[i], 0, total_size_);
    }

    done_ = false;
    present_ = false;
    thread_ = nullptr;
}

AsyncWriteBlock::~AsyncWriteBlock() {
    delete lock_;

    for (void *mem : mems_) {
        free(mem);
    }
}

void AsyncWriteBlock::run() {
    while (!done_) {
        // wait until block is empty
        std::unique_lock lock(*lock_);
        cv_.wait(lock, [this] { return present_ == true; });

        if (done_) {
            return;
        }

#pragma omp parallel for
        for (int i = 0; i < partitions_.size(); i++) {
            Partition *partition = partitions_[i];
            partitioned_file_->writePartition(partition);
            partition->present_ = false;
            partition->evicting_ = false;
            partition->cv_->notify_all();
        }

        present_ = false;
        lock.unlock();
        cv_.notify_all();
    }
}

void AsyncWriteBlock::start() {
    if (thread_ == nullptr) {
        thread_ = new std::thread(&AsyncWriteBlock::run, this);
    }
}

void AsyncWriteBlock::stop() {
    if (thread_ != nullptr) {
        if (thread_->joinable()) {
            done_ = true;
            present_ = true;
            cv_.notify_all();
            thread_->join();
        }
        delete thread_;
    }
}

void AsyncWriteBlock::async_write(std::vector<Partition *> partitions) {
    if (partitions.size() > mems_.size()) {
        // TODO: throw invalid inputs for function exception
        throw std::runtime_error("");
    }

    // wait until block is empty
    std::unique_lock lock(*lock_);
    cv_.wait(lock, [this] { return present_ == false; });

    partitions_ = partitions;

#pragma omp parallel for
    for (int i = 0; i < partitions_.size(); i++) {
        void *mem = mems_[i];
        Partition *partition = partitions_[i];

        memcpy_wrapper(mem, partition->data_ptr_, partition->total_size_);
        memset_wrapper(partition->data_ptr_, 0, partition->total_size_);

        partition->data_ptr_ = mem;
        partition->evicting_ = true;
    }

    present_ = true;

    lock.unlock();
    cv_.notify_all();
}

PartitionBuffer::PartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size,
                                 int64_t total_embeddings, torch::Dtype dtype, string filename, bool prefetching) {
    capacity_ = capacity;
    size_ = 0;
    num_partitions_ = num_partitions;
    partition_size_ = partition_size;
    fine_to_coarse_ratio_ = fine_to_coarse_ratio;
    dtype_ = dtype;
    dtype_size_ = get_dtype_size_wrapper(dtype_);
    embedding_size_ = embedding_size;
    total_embeddings_ = total_embeddings;

    filename_ = filename;
    partition_table_ = std::vector<Partition *>();

    prefetching_ = prefetching;

    int64_t curr_idx_offset = 0;
    int64_t curr_file_offset = 0;
    int64_t curr_partition_size = partition_size_;
    int64_t curr_total_size = curr_partition_size * embedding_size_ * dtype_size_;
    for (int64_t i = 0; i < num_partitions_; i++) {
        // the last partition might be slightly smaller
        if (i == num_partitions_ - 1) {
            curr_partition_size = total_embeddings_ - curr_idx_offset;
            curr_total_size = curr_partition_size * embedding_size_ * dtype_size_;
        }
        Partition *curr_part = new Partition(i, curr_partition_size, embedding_size_, dtype_, curr_idx_offset, curr_file_offset);
        partition_table_.push_back(curr_part);

        curr_file_offset += curr_total_size;
        curr_idx_offset += curr_partition_size;
    }

    filename_ = filename;
    partitioned_file_ = new PartitionedFile(filename_, num_partitions_, partition_size_, embedding_size_, total_embeddings_, dtype_);

    tensor_mem_ = nullptr;
    loaded_ = false;
}

PartitionBuffer::~PartitionBuffer() {
    unload(true);

    delete partitioned_file_;
    for (int64_t i = 0; i < num_partitions_; i++) {
        delete partition_table_[i];
    }
}

void PartitionBuffer::load() {
    if (!loaded_) {
        if (posix_memalign(&buff_mem_, 4096, capacity_ * partition_size_ * embedding_size_ * dtype_size_)) {
            SPDLOG_ERROR("Unable to allocate buffer memory\nError: {}", errno);
            throw std::runtime_error("");
        }
        memset_wrapper(buff_mem_, 0, capacity_ * partition_size_ * embedding_size_ * dtype_size_);
        buffer_tensor_view_ = torch::from_blob(buff_mem_, {capacity_ * partition_size_, embedding_size_}, dtype_);

        // initialize buffer
        int partition_id;

        int64_t num_nodes = 0;

        for (int i = 0; i < buffer_state_.size(0); i++) {
            partition_id = buffer_state_[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            void *buff_addr = (char *)buff_mem_ + (i * partition_size_ * embedding_size_ * dtype_size_);
            partitioned_file_->readPartition(buff_addr, partition);
            partition->present_ = true;
            partition->buffer_idx_ = i;
            num_nodes += partition->partition_size_;
        }

        size_.store(num_nodes);
        in_buffer_ids_ = torch::empty({num_nodes}, torch::kInt64);

        if (prefetching_) {
            lookahead_block_ = new LookaheadBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_, fine_to_coarse_ratio_);
            async_write_block_ = new AsyncWriteBlock(partition_size_ * embedding_size_ * dtype_size_, partitioned_file_, fine_to_coarse_ratio_);
            startThreads();
        }

        loaded_ = true;
    }
}

void PartitionBuffer::unload(bool write) {

    if (loaded_) {
        if (write) {
            sync();
        }
        buffer_tensor_view_ = torch::Tensor();
        free(buff_mem_);
        buff_mem_ = nullptr;

        if (prefetching_) {
            stopThreads();
            delete lookahead_block_;
            delete async_write_block_;
        }

        size_ = 0;
        loaded_ = false;
    }
}

torch::Tensor PartitionBuffer::getBufferState() { return buffer_state_; }

// indices a relative to the local node ids
torch::Tensor PartitionBuffer::indexRead(torch::Tensor indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    return buffer_tensor_view_.index_select(0, indices);
}

Indices PartitionBuffer::getRandomIds(int64_t size) { return torch::randint(in_buffer_ids_.size(0), size, torch::kInt64); }

// Duplicate indices are supported via torch::index_add_ on CPU.
void PartitionBuffer::indexAdd(torch::Tensor indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || buffer_tensor_view_.size(1) != values.size(1)) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    checked_cpu_index_add_("PartitionBuffer::indexAdd", buffer_tensor_view_, indices, values);
}

void PartitionBuffer::setBufferOrdering(std::vector<torch::Tensor> buffer_states) {
    buffer_states_ = buffer_states;
    if (buffer_states_.empty()) {
        throw GegeRuntimeException("PartitionBuffer::setBufferOrdering requires at least one buffer state");
    }
    buffer_state_iterator_ = buffer_states_.begin();
    buffer_state_ = *buffer_state_iterator_++;
    
    if (loaded_) {
        SPDLOG_INFO("Buffer ordering changed, unloading buffer");
        unload(true);
        load();
    }
}

bool PartitionBuffer::hasSwap() { return buffer_state_iterator_ != buffer_states_.end(); }

void PartitionBuffer::performNextSwap() {
    if (!buffer_state_.defined() || buffer_state_iterator_ == buffer_states_.end()) {
        return;
    }

    // get evicted and admitted partitions
    std::vector<int> evict_ids = getNextEvict();
    std::vector<int> admit_ids = getNextAdmit();

    std::vector<Partition *> admit_partitions;
    std::vector<Partition *> evict_partitions;
    std::vector<int64_t> evict_buffer_idxs;
    for (int admit_id : admit_ids) {
        admit_partitions.emplace_back(partition_table_[admit_id]);
    }
    for (int evict_id : evict_ids) {
        evict_partitions.emplace_back(partition_table_[evict_id]);
        evict_buffer_idxs.emplace_back(partition_table_[evict_id]->buffer_idx_);
    }

    buffer_state_ = *buffer_state_iterator_++;

    // evict partition
    auto t1 = std::chrono::high_resolution_clock::now();
    evict(evict_partitions);
    auto t2 = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("evict time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    // admit partition
    t1 = std::chrono::high_resolution_clock::now();
    admit(admit_partitions, evict_buffer_idxs);
    t2 = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("admit time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    int64_t num_nodes = 0;

    int partition_id;
    for (int i = 0; i < buffer_state_.size(0); i++) {
        partition_id = buffer_state_[i].item<int>();
        num_nodes += partition_table_[partition_id]->partition_size_;
    }

    in_buffer_ids_ = torch::empty({num_nodes}, torch::kInt64);

    //    int64_t offset = 0;
    //    for (int i = 0; i < buffer_state_.size(0); i++) {
    //        partition_id = buffer_state_[i].item<int>();
    //        Partition *partition = partition_table_[partition_id];
    //        int64_t partition_offset = partition->idx_offset_;
    //
    //        in_buffer_ids_.slice(0, offset, offset + partition->partition_size_) = torch::arange(partition_offset, partition_offset +
    //        partition->partition_size_); offset += partition->partition_size_;
    //    }
}

std::vector<int> PartitionBuffer::getNextAdmit() {
    std::vector<int> admit_ids;
    bool admitted;

    if (buffer_state_iterator_ != buffer_states_.end()) {
        for (int i = 0; i < buffer_state_iterator_->size(0); i++) {
            admitted = true;
            for (int j = 0; j < buffer_state_.size(0); j++) {
                if ((*buffer_state_iterator_)[i].item<int>() == (buffer_state_)[j].item<int>()) {
                    admitted = false;
                }
            }
            if (admitted) {
                admit_ids.emplace_back((*buffer_state_iterator_)[i].item<int>());
            }
        }
    }
    return admit_ids;
}

std::vector<int> PartitionBuffer::getNextEvict() {
    std::vector<int> evict_ids;
    bool evicted;

    for (int i = 0; i < buffer_state_.size(0); i++) {
        evicted = true;
        for (int j = 0; j < buffer_state_iterator_->size(0); j++) {
            if ((*buffer_state_iterator_)[j].item<int>() == buffer_state_[i].item<int>()) {
                evicted = false;
            }
        }
        if (evicted) {
            evict_ids.emplace_back(buffer_state_[i].item<int>());
        }
    }
    return evict_ids;
}

torch::Tensor PartitionBuffer::getGlobalToLocalMap(bool get_current) {

    torch::Tensor buffer_index_map = -torch::ones({total_embeddings_}, torch::kInt64);
    torch::Tensor buffer_state;

    if (get_current) {
        buffer_state = buffer_state_;
#pragma omp parallel for
        for (int i = 0; i < buffer_state.size(0); i++) {
            int partition_id = buffer_state[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            int64_t partition_offset = partition->idx_offset_;
            int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
            buffer_index_map.slice(0, partition_offset, partition_offset + partition->partition_size_) =
                torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
        }
    } else {
        // get mapping for next swap
        buffer_state = *buffer_state_iterator_;

        // get evicted and admitted partitions
        std::vector<int> evict_ids = getNextEvict();
        std::vector<int> admit_ids = getNextAdmit();

        // get mapping for the partitions that will still be in the buffer
#pragma omp parallel for
        for (int i = 0; i < buffer_state.size(0); i++) {
            int partition_id = buffer_state[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            int64_t partition_offset = partition->idx_offset_;

            if (partition->buffer_idx_ != -1) {
                int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
                buffer_index_map.slice(0, partition_offset, partition_offset + partition->partition_size_) =
                    torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
            }
        }

// get mapping for the partitions that will be admitted
#pragma omp parallel for
        for (int i = 0; i < evict_ids.size(); i++) {
            Partition *admit_partition = partition_table_[admit_ids[i]];
            Partition *evict_partition = partition_table_[evict_ids[i]];
            int64_t partition_offset = admit_partition->idx_offset_;
            int64_t buffer_offset = evict_partition->buffer_idx_ * partition_size_;
            buffer_index_map.slice(0, partition_offset, partition_offset + admit_partition->partition_size_) =
                torch::arange(buffer_offset, buffer_offset + admit_partition->partition_size_);
        }
    }
    return buffer_index_map;
}

torch::Tensor PartitionBuffer::getPartitionToBufferSlotMap() {
    torch::Tensor partition_to_buffer_slot = -torch::ones({num_partitions_}, torch::kInt64);
    auto slot_accessor = partition_to_buffer_slot.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int partition_id = 0; partition_id < num_partitions_; partition_id++) {
        Partition *partition = partition_table_[partition_id];
        if (partition->present_) {
            slot_accessor[partition_id] = partition->buffer_idx_;
        }
    }

    return partition_to_buffer_slot;
}

void PartitionBuffer::evict(std::vector<Partition *> evict_partitions) {
    if (prefetching_) {
        async_write_block_->async_write(evict_partitions);
    } else {
#pragma omp parallel for
        for (int i = 0; i < evict_partitions.size(); i++) {
            partitioned_file_->writePartition(evict_partitions[i]);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < evict_partitions.size(); i++) {
        evict_partitions[i]->present_ = false;
    }
}

void PartitionBuffer::admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs) {
    if (admit_partitions.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    std::vector<void *> buff_addrs(buffer_idxs.size());
    
#pragma omp parallel for
    for (int i = 0; i < buffer_idxs.size(); i++) {
        void *buff_addr = (char *)buff_mem_ + (buffer_idxs[i] * partition_size_ * embedding_size_ * dtype_size_);
        buff_addrs[i] = buff_addr;
    }

    if (prefetching_) {
        std::vector<int> next_admit_ids = getNextAdmit();
        std::vector<Partition *> next_partitions;
        if (!next_admit_ids.empty()) {
            for (int admit_id : next_admit_ids) {
                next_partitions.emplace_back(partition_table_[admit_id]);
            }
        }
        lookahead_block_->move_to_buffer(buff_addrs, buffer_idxs, next_partitions);
    } else {
#pragma omp parallel for
        for (int i = 0; i < admit_partitions.size(); i++) {
            Partition *partition = admit_partitions[i];
            partitioned_file_->readPartition(buff_addrs[i], partition);
            partition->present_ = true;
            partition->buffer_idx_ = buffer_idxs[i];
        }
    }
}

void PartitionBuffer::sync() {
    SPDLOG_DEBUG("Synchronizing buffer");
    Partition *curr_partition;
    for (int i = 0; i < num_partitions_; i++) {
        curr_partition = partition_table_[i];
        if (curr_partition->present_) {
            partitioned_file_->writePartition(curr_partition, true);
            curr_partition->present_ = false;
            curr_partition->buffer_idx_ = -1;
        }
    }
}

void PartitionBuffer::startThreads() {
    SPDLOG_DEBUG("Starting prefetching threads");
    std::vector<Partition *> partitions;
    std::vector<int> admit_ids = getNextAdmit();
    for (int admit_id : admit_ids) {
        partitions.emplace_back(partition_table_[admit_id]);
    }
    lookahead_block_->start(partitions);
    async_write_block_->start();
}

void PartitionBuffer::stopThreads() {
    SPDLOG_DEBUG("Stopping prefetching threads");
    lookahead_block_->stop();
    async_write_block_->stop();
}

MemPartitionBuffer::MemPartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                    torch::Dtype dtype, string filename, bool prefetching, torch::Device device, int buffer_sizes)
    :PartitionBuffer(capacity, num_partitions, fine_to_coarse_ratio, partition_size, embedding_size, total_embeddings, dtype, filename, prefetching) {
    buffer_sizes_ = buffer_sizes;
    device_ = device;
    bool log_startup_timing = startup_timing_enabled();
    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBuffer::ctor] begin device={} capacity={} partition_size={} dim={} total_embeddings={} buffer_sizes={}",
                    device_.str(), capacity_, partition_size_, embedding_size_, total_embeddings_, buffer_sizes_);
    }

    // if (posix_memalign(&buff_mem_, 4096, capacity_ * partition_size_ * embedding_size_ * dtype_size_)) {
    //     SPDLOG_ERROR("Unable to allocate buffer memory\nError: {}", errno);
    //     throw std::runtime_error("");
    // }
    // memset_wrapper(buff_mem_, 0, capacity_ * partition_size_ * embedding_size_ * dtype_size_);

    use_pinned_host_buffer_ = mem_partition_buffer_pinned_host_enabled();
    if (!use_pinned_host_buffer_ && buffer_sizes_ > 1 && device_.is_cuda()) {
        static std::once_flag warned_once;
        std::call_once(warned_once, []() {
            SPDLOG_WARN("GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=0 is unsupported for multi-GPU MEM_PARTITION_BUFFER; forcing pinned host buffers");
        });
        use_pinned_host_buffer_ = true;
    }

    hidden_frame_capacity_ =
        ((single_gpu_gpu_aware_custom_enabled() && buffer_sizes_ == 1 && device_.is_cuda()) ||
         (multi_gpu_async_admit_preload_enabled() && buffer_sizes_ > 1 && device_.is_cuda()))
            ? static_cast<int>(frame_cache_hidden_frames())
            : 0;
    physical_frame_capacity_ = capacity_ + hidden_frame_capacity_;
    resetFrameCacheState_();

    buff_mem_ = nullptr;
    buffer_tensor_view_ = torch::Tensor();
    buffer_tensor_gpu_view_ = torch::Tensor();

    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBuffer::ctor] deferred backing allocation device={} visible_rows={} physical_rows={} dim={} pinned={} hidden_frames={}",
                    device_.str(), capacity_ * partition_size_, physical_frame_capacity_ * partition_size_, embedding_size_,
                    use_pinned_host_buffer_, hidden_frame_capacity_);
    }

    // The initial host ordering is identity. Materialize permutation tensors only
    // when repartitioning installs a non-trivial mapping.
    perm_ = torch::Tensor();
    pos_ = torch::Tensor();
    refreshHostPartitionRanges_();
    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBuffer::ctor] end device={}", device_.str());
    }
}

MemPartitionBuffer::~MemPartitionBuffer() {
    joinAsyncAdmitPreload_();
    joinAsyncEvictWriteback_();
    unload(true);
    // free(buff_mem_);
    buffer_tensor_view_ = torch::Tensor();
    buffer_tensor_gpu_view_ = torch::Tensor();
}

bool MemPartitionBuffer::frameCacheEnabled_() const { return hidden_frame_capacity_ > 0; }

bool MemPartitionBuffer::asyncAdmitPreloadEnabled_() const {
    if (!device_.is_cuda()) {
        return false;
    }
    if (buffer_sizes_ == 1) {
        return single_gpu_async_admit_preload_enabled() && single_gpu_gpu_aware_custom_enabled();
    }
    return multi_gpu_async_admit_preload_enabled();
}

void MemPartitionBuffer::refreshFrameCacheTensors_() {
    logical_to_physical_frame_cpu_ = torch::tensor(logical_to_physical_frames_, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (device_.is_cuda()) {
#ifdef GEGE_CUDA
        c10::cuda::CUDAGuard device_guard(device_);
        logical_to_physical_frame_device_ = logical_to_physical_frame_cpu_.to(device_, true);
#else
        logical_to_physical_frame_device_ = logical_to_physical_frame_cpu_;
#endif
    } else {
        logical_to_physical_frame_device_ = logical_to_physical_frame_cpu_;
    }
}

void MemPartitionBuffer::resetFrameCacheState_() {
    logical_to_physical_frames_.resize(static_cast<std::size_t>(capacity_));
    std::iota(logical_to_physical_frames_.begin(), logical_to_physical_frames_.end(), 0);
    {
        std::lock_guard<std::mutex> lock(free_physical_frames_lock_);
        free_physical_frames_.clear();
        for (int64_t frame = capacity_; frame < physical_frame_capacity_; frame++) {
            free_physical_frames_.emplace_back(frame);
        }
    }
    refreshFrameCacheTensors_();
}

int64_t MemPartitionBuffer::logicalSlotToPhysicalFrame_(int64_t logical_slot) const {
    if (logical_slot < 0 || logical_slot >= static_cast<int64_t>(logical_to_physical_frames_.size())) {
        throw GegeRuntimeException(fmt::format("MemPartitionBuffer logical slot {} is out of range 0..{}", logical_slot,
                                               static_cast<int64_t>(logical_to_physical_frames_.size()) - 1));
    }
    return logical_to_physical_frames_[static_cast<std::size_t>(logical_slot)];
}

int64_t MemPartitionBuffer::logicalSlotRowOffset_(int64_t logical_slot) const { return logicalSlotToPhysicalFrame_(logical_slot) * partition_size_; }

int64_t MemPartitionBuffer::partitionRowOffset_(const Partition *partition) const {
    if (partition == nullptr || partition->buffer_idx_ < 0) {
        return -1;
    }
    int64_t physical_frame = partition->physical_frame_idx_ >= 0 ? partition->physical_frame_idx_ : logicalSlotToPhysicalFrame_(partition->buffer_idx_);
    return physical_frame * partition_size_;
}

torch::Tensor MemPartitionBuffer::translateLogicalIndicesToPhysical_(torch::Tensor indices, const char *context) const {
    if (!frameCacheEnabled_() || !indices.defined() || indices.numel() == 0) {
        return indices;
    }

    torch::Tensor logical_indices = indices.to(torch::kInt64);
    torch::Tensor slot_ids = torch::floor_divide(logical_indices, partition_size_);
    if (local_id_validate_enabled_storage()) {
        int64_t min_slot = slot_ids.min().item<int64_t>();
        int64_t max_slot = slot_ids.max().item<int64_t>();
        if (min_slot < 0 || max_slot >= capacity_) {
            throw GegeRuntimeException(
                fmt::format("MemPartitionBuffer::translateLogicalIndicesToPhysical_ ({}) received logical slots [{}, {}] outside visible range [0, {}] for partition_size={} sample_indices={}",
                            context, min_slot, max_slot, capacity_ - 1, partition_size_, tensor_prefix_string(logical_indices, 8)));
        }
    }
    torch::Tensor row_offsets = torch::remainder(logical_indices, partition_size_);
    torch::Tensor slot_map = logical_indices.device().is_cuda() ? logical_to_physical_frame_device_ : logical_to_physical_frame_cpu_;
    torch::Tensor physical_slots = slot_map.index_select(0, slot_ids);
    return physical_slots * partition_size_ + row_offsets;
}

void MemPartitionBuffer::ensureBackingTensorsAllocated_() {
    if (buffer_tensor_view_.defined() && buffer_tensor_gpu_view_.defined()) {
        buff_mem_ = buffer_tensor_view_.data_ptr();
        return;
    }

    bool log_startup_timing = startup_timing_enabled();
    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBuffer::allocate] begin device={} visible_rows={} physical_rows={} dim={} pinned={}",
                    device_.str(), capacity_ * partition_size_, physical_frame_capacity_ * partition_size_, embedding_size_,
                    use_pinned_host_buffer_);
    }

    torch::TensorOptions host_options = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);
    if (use_pinned_host_buffer_) {
        host_options = host_options.pinned_memory(true);
    }
    buffer_tensor_view_ = torch::zeros({physical_frame_capacity_ * partition_size_, embedding_size_}, host_options);
    buff_mem_ = buffer_tensor_view_.data_ptr();

    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBuffer::allocate] host buffer ready device={} physical_rows={} dim={} pinned={}",
                    device_.str(), physical_frame_capacity_ * partition_size_, embedding_size_, use_pinned_host_buffer_);
    }

    torch::TensorOptions device_options = torch::TensorOptions().dtype(dtype_).device(device_);
#ifdef GEGE_CUDA
    if (device_.is_cuda()) {
        c10::cuda::CUDAGuard device_guard(device_);
        buffer_tensor_gpu_view_ = torch::empty({physical_frame_capacity_ * partition_size_, embedding_size_}, device_options);
    } else {
        buffer_tensor_gpu_view_ = torch::empty({physical_frame_capacity_ * partition_size_, embedding_size_}, device_options);
    }
#else
    buffer_tensor_gpu_view_ = torch::empty({physical_frame_capacity_ * partition_size_, embedding_size_}, device_options);
#endif

    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBuffer::allocate] device buffer ready device={} physical_rows={} dim={}",
                    device_.str(), physical_frame_capacity_ * partition_size_, embedding_size_);
    }
}

void MemPartitionBuffer::refreshHostPartitionRanges_() {
    partition_host_start_offsets_.assign(static_cast<std::size_t>(num_partitions_), -1);
    partition_host_contiguous_.assign(static_cast<std::size_t>(num_partitions_), false);

    if (!pos_.defined()) {
        for (int partition_id = 0; partition_id < num_partitions_; partition_id++) {
            Partition *partition = partition_table_[partition_id];
            partition_host_contiguous_[partition_id] = true;
            partition_host_start_offsets_[partition_id] = partition->idx_offset_;
        }
        return;
    }

    torch::Tensor pos_cpu = pos_.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *pos_ptr = pos_cpu.data_ptr<int64_t>();

    for (int partition_id = 0; partition_id < num_partitions_; partition_id++) {
        Partition *partition = partition_table_[partition_id];
        int64_t start_idx = partition->idx_offset_;
        int64_t size = partition->partition_size_;

        if (size <= 0) {
            partition_host_contiguous_[partition_id] = true;
            partition_host_start_offsets_[partition_id] = start_idx;
            continue;
        }

        int64_t first = pos_ptr[start_idx];
        bool contiguous = true;
        for (int64_t row = 1; row < size; row++) {
            if (pos_ptr[start_idx + row] != first + row) {
                contiguous = false;
                break;
            }
        }

        partition_host_contiguous_[partition_id] = contiguous;
        if (contiguous) {
            partition_host_start_offsets_[partition_id] = first;
        }
    }
}

bool MemPartitionBuffer::hasContiguousHostPartitionRange_(int partition_id) const {
    return partition_id >= 0 && partition_id < num_partitions_ && static_cast<std::size_t>(partition_id) < partition_host_contiguous_.size() &&
           partition_host_contiguous_[partition_id];
}

int64_t MemPartitionBuffer::contiguousHostPartitionStart_(int partition_id) const {
    if (!hasContiguousHostPartitionRange_(partition_id)) {
        return -1;
    }
    return partition_host_start_offsets_[partition_id];
}

void MemPartitionBuffer::ensureDirtyRowMaskAllocated_() {
    if (!mem_partition_buffer_dirty_profile_enabled()) {
        return;
    }
    if (dirty_row_mask_.defined()) {
        return;
    }
    torch::TensorOptions mask_options = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
#ifdef GEGE_CUDA
    if (device_.is_cuda()) {
        c10::cuda::CUDAGuard device_guard(device_);
        dirty_row_mask_ = torch::zeros({physical_frame_capacity_ * partition_size_}, mask_options);
    } else {
        dirty_row_mask_ = torch::zeros({physical_frame_capacity_ * partition_size_}, mask_options);
    }
#else
    dirty_row_mask_ = torch::zeros({physical_frame_capacity_ * partition_size_}, mask_options);
#endif
}

void MemPartitionBuffer::markDirtyRows_(torch::Tensor indices, torch::Tensor active_mask) {
    if (!mem_partition_buffer_dirty_profile_enabled()) {
        return;
    }
    if (!indices.defined() || indices.numel() == 0) {
        return;
    }
    ensureDirtyRowMaskAllocated_();
    if (!dirty_row_mask_.defined()) {
        return;
    }

    torch::Tensor dirty_indices = indices.to(torch::kInt64);
    if (active_mask.defined() && active_mask.numel() == dirty_indices.numel()) {
        torch::Tensor mask = active_mask.to(dirty_indices.device()).to(torch::kBool);
        dirty_indices = dirty_indices.masked_select(mask);
        if (dirty_indices.numel() == 0) {
            return;
        }
    }
    dirty_indices = dirty_indices.to(dirty_row_mask_.device());
    dirty_row_mask_.index_fill_(0, dirty_indices, 1);
}

void MemPartitionBuffer::clearDirtyRowsForSlots_(const std::vector<int64_t> &slots, const std::vector<int> &partition_ids) {
    if (!dirty_row_mask_.defined()) {
        return;
    }
    for (std::size_t idx = 0; idx < slots.size() && idx < partition_ids.size(); idx++) {
        Partition *partition = partition_table_[partition_ids[idx]];
        int64_t buffer_offset = partitionRowOffset_(partition);
        dirty_row_mask_.slice(0, buffer_offset, buffer_offset + partition->partition_size_).zero_();
    }
}

int64_t MemPartitionBuffer::countDirtyRowsForSlots_(const std::vector<int64_t> &slots, const std::vector<int> &partition_ids) {
    if (!dirty_row_mask_.defined()) {
        return -1;
    }
    int64_t dirty_rows = 0;
    for (std::size_t idx = 0; idx < slots.size() && idx < partition_ids.size(); idx++) {
        Partition *partition = partition_table_[partition_ids[idx]];
        int64_t buffer_offset = partitionRowOffset_(partition);
        dirty_rows += dirty_row_mask_.slice(0, buffer_offset, buffer_offset + partition->partition_size_).sum().item<int64_t>();
    }
    return dirty_rows;
}

torch::Tensor MemPartitionBuffer::hostPartitionRows_(Partition *partition) {
    if (hasContiguousHostPartitionRange_(partition->partition_id_)) {
        return data_storage_.narrow(0, contiguousHostPartitionStart_(partition->partition_id_), partition->partition_size_);
    }
    return data_storage_.index_select(0, pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_));
}

void MemPartitionBuffer::copyPartitionFromHostToPinned_(Partition *partition, torch::Tensor pinned_view) {
    auto copy_impl = [&]() {
        if (hasContiguousHostPartitionRange_(partition->partition_id_)) {
            pinned_view.copy_(data_storage_.narrow(0, contiguousHostPartitionStart_(partition->partition_id_), partition->partition_size_));
            return;
        }
        pinned_view.copy_(data_storage_.index_select(0, pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_)));
    };
    if (mem_partition_buffer_host_storage_lock_enabled()) {
        std::lock_guard<std::mutex> lock(host_storage_lock_);
        copy_impl();
    } else {
        copy_impl();
    }
}

void MemPartitionBuffer::copyPartitionFromPinnedToHost_(Partition *partition, torch::Tensor pinned_view) {
    auto copy_impl = [&]() {
        if (hasContiguousHostPartitionRange_(partition->partition_id_)) {
            data_storage_.narrow(0, contiguousHostPartitionStart_(partition->partition_id_), partition->partition_size_).copy_(pinned_view);
            return;
        }
        data_storage_.index_put_({pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_)}, pinned_view);
    };
    if (mem_partition_buffer_host_storage_lock_enabled()) {
        std::lock_guard<std::mutex> lock(host_storage_lock_);
        copy_impl();
    } else {
        copy_impl();
    }
}

void MemPartitionBuffer::joinAsyncAdmitPreload_() {
    if (async_admit_preload_thread_.joinable()) {
        async_admit_preload_thread_.join();
    }

    std::exception_ptr preload_exception = nullptr;
    {
        std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
        preload_exception = async_admit_preload_exception_;
        async_admit_preload_exception_ = nullptr;
    }
    if (preload_exception != nullptr) {
        std::rethrow_exception(preload_exception);
    }
}

void MemPartitionBuffer::joinAsyncEvictWriteback_() {
    if (async_evict_writeback_thread_.joinable()) {
        async_evict_writeback_thread_.join();
    }

    std::exception_ptr writeback_exception = nullptr;
    {
        std::lock_guard<std::mutex> lock(async_evict_writeback_lock_);
        writeback_exception = async_evict_writeback_exception_;
        async_evict_writeback_exception_ = nullptr;
        async_evict_writeback_in_flight_ = false;
        async_evict_writeback_partition_ids_.clear();
    }
    if (writeback_exception != nullptr) {
        std::rethrow_exception(writeback_exception);
    }
}

void MemPartitionBuffer::joinAsyncEvictWritebackForPartitions_(const std::vector<int> &partition_ids) {
    bool must_join = false;
    {
        std::lock_guard<std::mutex> lock(async_evict_writeback_lock_);
        if (async_evict_writeback_thread_.joinable()) {
            for (int partition_id : partition_ids) {
                if (std::find(async_evict_writeback_partition_ids_.begin(), async_evict_writeback_partition_ids_.end(), partition_id) !=
                    async_evict_writeback_partition_ids_.end()) {
                    must_join = true;
                    break;
                }
            }
        }
    }

    if (must_join) {
        joinAsyncEvictWriteback_();
    }
}

void MemPartitionBuffer::startAsyncEvictWriteback_(const std::vector<int> &evict_ids, const std::vector<int64_t> &row_offsets, torch::Tensor gpu_stage,
                                                   torch::Tensor expected_host_stage, std::vector<int64_t> release_frames,
                                                   std::vector<int64_t> source_frame_offsets) {
    if (evict_ids.empty() || (gpu_stage.defined() == false && source_frame_offsets.empty())) {
        return;
    }

    joinAsyncEvictWriteback_();
    {
        std::lock_guard<std::mutex> lock(async_evict_writeback_lock_);
        async_evict_writeback_in_flight_ = true;
        async_evict_writeback_exception_ = nullptr;
        async_evict_writeback_partition_ids_ = evict_ids;
    }

    async_evict_writeback_thread_ =
        std::thread([this, evict_ids, row_offsets, gpu_stage, expected_host_stage, release_frames, source_frame_offsets]() {
        auto total_start = std::chrono::high_resolution_clock::now();
        auto phase_start = total_start;
        double host_alloc_ms = 0.0;
        double gpu_to_host_stage_ms = 0.0;
        double verify_host_copy_ms = 0.0;
        double host_storage_write_ms = 0.0;
        double verify_host_storage_ms = 0.0;
        try {
            torch::TensorOptions host_options = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);
            if (use_pinned_host_buffer_) {
                host_options = host_options.pinned_memory(true);
            }
            int64_t total_rows = 0;
            if (!row_offsets.empty()) {
                total_rows = row_offsets.back();
            } else if (gpu_stage.defined()) {
                total_rows = gpu_stage.size(0);
            }
            torch::Tensor host_stage = torch::empty({total_rows, embedding_size_}, host_options);
            auto after_host_alloc = std::chrono::high_resolution_clock::now();
            host_alloc_ms = elapsed_ms(phase_start, after_host_alloc);
            phase_start = after_host_alloc;

#ifdef GEGE_CUDA
            if (!source_frame_offsets.empty()) {
                c10::cuda::CUDAGuard device_guard(device_);
                bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(host_stage);
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                    torch::Tensor host_view = host_stage.slice(0, row_offsets[idx], row_offsets[idx + 1]);
                    int64_t source_start = source_frame_offsets[idx];
                    int64_t source_rows = row_offsets[idx + 1] - row_offsets[idx];
                    torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, source_start, source_start + source_rows);
                    if (non_blocking_host_copy) {
                        host_view.copy_(gpu_view, true);
                    } else {
                        host_view.copy_(gpu_view);
                    }
                }
                cudaStreamSynchronize(copy_stream.stream());
            } else if (gpu_stage.device().is_cuda()) {
                c10::cuda::CUDAGuard device_guard(device_);
                bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(host_stage);
                if (non_blocking_host_copy) {
                    auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                    c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                    host_stage.copy_(gpu_stage, true);
                    cudaStreamSynchronize(copy_stream.stream());
                } else {
                    host_stage.copy_(gpu_stage);
                }
            } else {
                host_stage.copy_(gpu_stage);
            }
#else
            if (gpu_stage.defined()) {
                host_stage.copy_(gpu_stage);
            }
#endif
            auto after_gpu_to_host = std::chrono::high_resolution_clock::now();
            gpu_to_host_stage_ms = elapsed_ms(phase_start, after_gpu_to_host);
            phase_start = after_gpu_to_host;

            if (expected_host_stage.defined()) {
                torch::Tensor expected_cpu = expected_host_stage.to(torch::kCPU).contiguous();
                torch::Tensor actual_cpu = host_stage.to(torch::kCPU).contiguous();
                bool host_copy_match = torch::allclose(actual_cpu, expected_cpu, 1e-5, 1e-6);
                if (!host_copy_match) {
                    double max_abs = (actual_cpu - expected_cpu).abs().max().item<double>();
                    SPDLOG_ERROR(
                        "GEGE_SINGLE_GPU_ASYNC_EVICT_VERIFY failed host-copy storage={} this={} device={} evict_ids={} rows={} max_abs={}",
                        basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(evict_ids), total_rows, max_abs);
                    throw GegeRuntimeException("Async evict host-copy verifier failed");
                }
            }
            auto after_host_copy_verify = std::chrono::high_resolution_clock::now();
            verify_host_copy_ms = elapsed_ms(phase_start, after_host_copy_verify);
            phase_start = after_host_copy_verify;

            for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                Partition *partition = partition_table_[evict_ids[idx]];
                torch::Tensor stage_view = host_stage.slice(0, row_offsets[idx], row_offsets[idx + 1]);
                copyPartitionFromPinnedToHost_(partition, stage_view);
            }
            auto after_host_storage_write = std::chrono::high_resolution_clock::now();
            host_storage_write_ms = elapsed_ms(phase_start, after_host_storage_write);
            phase_start = after_host_storage_write;

            if (expected_host_stage.defined()) {
                torch::Tensor roundtrip_host_stage = torch::empty_like(host_stage);
                for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                    Partition *partition = partition_table_[evict_ids[idx]];
                    torch::Tensor dst_view = roundtrip_host_stage.slice(0, row_offsets[idx], row_offsets[idx + 1]);
                    copyPartitionFromHostToPinned_(partition, dst_view);
                }
                torch::Tensor expected_cpu = expected_host_stage.to(torch::kCPU).contiguous();
                torch::Tensor roundtrip_cpu = roundtrip_host_stage.to(torch::kCPU).contiguous();
                bool host_storage_match = torch::allclose(roundtrip_cpu, expected_cpu, 1e-5, 1e-6);
                if (!host_storage_match) {
                    double max_abs = (roundtrip_cpu - expected_cpu).abs().max().item<double>();
                    SPDLOG_ERROR(
                        "GEGE_SINGLE_GPU_ASYNC_EVICT_VERIFY failed host-storage storage={} this={} device={} evict_ids={} rows={} max_abs={}",
                        basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(evict_ids), total_rows, max_abs);
                    throw GegeRuntimeException("Async evict host-storage verifier failed");
                }
                SPDLOG_INFO("GEGE_SINGLE_GPU_ASYNC_EVICT_VERIFY passed storage={} this={} device={} evict_ids={} rows={}",
                            basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(evict_ids), total_rows);
            }
            auto after_host_storage_verify = std::chrono::high_resolution_clock::now();
            verify_host_storage_ms = elapsed_ms(phase_start, after_host_storage_verify);

            if (!release_frames.empty()) {
                std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
                for (int64_t frame : release_frames) {
                    if (std::find(free_physical_frames_.begin(), free_physical_frames_.end(), frame) == free_physical_frames_.end()) {
                        free_physical_frames_.push_back(frame);
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(async_evict_writeback_lock_);
                async_evict_writeback_in_flight_ = false;
                async_evict_writeback_exception_ = nullptr;
            }
            if (partition_buffer_swap_timing_enabled()) {
                SPDLOG_INFO(
                    "[partition-buffer-async-evict-writeback] storage={} this={} device={} evict_ids={} rows={} release_frames={} host_alloc_ms={:.3f} "
                    "gpu_to_host_stage_ms={:.3f} verify_host_copy_ms={:.3f} host_storage_write_ms={:.3f} verify_host_storage_ms={:.3f} total_ms={:.3f}",
                    basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(evict_ids), total_rows,
                    vector_prefix_string(release_frames), host_alloc_ms,
                    gpu_to_host_stage_ms, verify_host_copy_ms, host_storage_write_ms, verify_host_storage_ms,
                    elapsed_ms(total_start, std::chrono::high_resolution_clock::now()));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(async_evict_writeback_lock_);
            async_evict_writeback_in_flight_ = false;
            async_evict_writeback_exception_ = std::current_exception();
        }
    });
}

void MemPartitionBuffer::startAsyncAdmitPreload() {
    std::vector<int> admit_ids = getNextAdmit();
    std::vector<int> evict_ids = getNextEvict();
    if (admit_ids.empty() || admit_ids.size() != evict_ids.size()) {
        std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
        async_admit_preload_valid_ = false;
        async_admit_preload_in_flight_ = false;
        return;
    }

    std::vector<int64_t> evict_slots;
    evict_slots.reserve(evict_ids.size());
    for (int evict_id : evict_ids) {
        Partition *partition = partition_table_[evict_id];
        if (partition->buffer_idx_ < 0) {
            return;
        }
        evict_slots.emplace_back(partition->buffer_idx_);
    }

    startAsyncAdmitPreloadForPlan_(admit_ids, evict_slots);
}

void MemPartitionBuffer::startAsyncAdmitPreloadForPlan_(const std::vector<int> &admit_ids, const std::vector<int64_t> &evict_slots) {
    if (!asyncAdmitPreloadEnabled_() || !loaded_ || !buffer_state_.defined() || buffer_state_iterator_ == buffer_states_.end() ||
        !buffer_tensor_gpu_view_.defined() || !data_storage_.defined()) {
        return;
    }

    joinAsyncAdmitPreload_();

    if (admit_ids.empty() || admit_ids.size() != evict_slots.size()) {
        std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
        async_admit_preload_valid_ = false;
        async_admit_preload_in_flight_ = false;
        return;
    }

    int64_t admit_rows = partition_rows_for_ids(admit_ids, partition_table_);
    if (admit_rows <= 0) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
        async_admit_preload_in_flight_ = true;
        async_admit_preload_valid_ = false;
        async_admit_preload_exception_ = nullptr;
        async_admit_preload_hidden_publishes_.clear();
        async_admit_preload_admit_ids_.clear();
        async_admit_preload_evict_slots_.clear();
        async_admit_preload_row_offsets_.clear();
        async_admit_preload_gpu_tensor_ = torch::Tensor();
        async_admit_preload_host_load_ms_ = 0.0;
        async_admit_preload_cpu_to_gpu_ms_ = 0.0;
        async_admit_preload_total_ms_ = 0.0;
    }

    async_admit_preload_thread_ = std::thread([this, admit_ids, evict_slots]() {
        auto total_start = std::chrono::high_resolution_clock::now();
        auto phase_start = total_start;
        double host_load_ms = 0.0;
        double cpu_to_gpu_ms = 0.0;

        try {
            joinAsyncEvictWritebackForPartitions_(admit_ids);

            std::vector<HiddenFramePublish> hidden_publishes;
            std::vector<int> stage_admit_ids = admit_ids;
            std::vector<int64_t> stage_evict_slots = evict_slots;
            if (frameCacheEnabled_() && !admit_ids.empty()) {
                const std::size_t target_hidden_count =
                    std::min<std::size_t>(static_cast<std::size_t>(hidden_frame_capacity_), admit_ids.size());
                if (target_hidden_count > 0 && frame_cache_hidden_only_preload_enabled() &&
                    frame_cache_delayed_stale_writeback_enabled() && !single_gpu_async_evict_writeback_enabled()) {
                    std::size_t free_count = 0;
                    {
                        std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
                        free_count = free_physical_frames_.size();
                    }
                    if (free_count < target_hidden_count) {
                        joinAsyncEvictWriteback_();
                    }
                }
                {
                    std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
                    const std::size_t hidden_count = std::min<std::size_t>(free_physical_frames_.size(), target_hidden_count);
                    hidden_publishes.reserve(hidden_count);
                    for (std::size_t idx = 0; idx < hidden_count; idx++) {
                        hidden_publishes.push_back(HiddenFramePublish{
                            admit_ids[idx],
                            evict_slots[idx],
                            free_physical_frames_[free_physical_frames_.size() - 1 - idx],
                        });
                    }
                }
                if (!hidden_publishes.empty()) {
                    if (frame_cache_hidden_only_preload_enabled()) {
                        stage_admit_ids.clear();
                        stage_evict_slots.clear();
                    } else {
                        stage_admit_ids.erase(stage_admit_ids.begin(),
                                              stage_admit_ids.begin() + static_cast<std::ptrdiff_t>(hidden_publishes.size()));
                        stage_evict_slots.erase(stage_evict_slots.begin(),
                                                stage_evict_slots.begin() + static_cast<std::ptrdiff_t>(hidden_publishes.size()));
                    }
                }
            }
            int64_t stage_rows = partition_rows_for_ids(stage_admit_ids, partition_table_);

            std::vector<int64_t> row_offsets;
            row_offsets.reserve(stage_admit_ids.size() + 1);
            row_offsets.emplace_back(0);
            for (int admit_id : stage_admit_ids) {
                row_offsets.emplace_back(row_offsets.back() + partition_table_[admit_id]->partition_size_);
            }

            torch::Tensor host_stage;
            if (stage_rows > 0) {
                torch::TensorOptions host_options = torch::TensorOptions().dtype(dtype_).device(torch::kCPU);
                if (use_pinned_host_buffer_) {
                    host_options = host_options.pinned_memory(true);
                }
                host_stage = torch::empty({stage_rows, embedding_size_}, host_options);
                for (std::size_t idx = 0; idx < stage_admit_ids.size(); idx++) {
                    Partition *partition = partition_table_[stage_admit_ids[idx]];
                    torch::Tensor stage_view = host_stage.slice(0, row_offsets[idx], row_offsets[idx + 1]);
                    copyPartitionFromHostToPinned_(partition, stage_view);
                }
            }

            for (const HiddenFramePublish &hidden_publish : hidden_publishes) {
                Partition *hidden_partition = partition_table_[hidden_publish.partition_id];
                int64_t hidden_offset = hidden_publish.frame * partition_size_;
                torch::Tensor hidden_host_view = buffer_tensor_view_.slice(0, hidden_offset, hidden_offset + hidden_partition->partition_size_);
                copyPartitionFromHostToPinned_(hidden_partition, hidden_host_view);
            }
            auto after_host = std::chrono::high_resolution_clock::now();
            host_load_ms = elapsed_ms(phase_start, after_host);
            phase_start = after_host;

            torch::Tensor gpu_stage;
#ifdef GEGE_CUDA
            {
                c10::cuda::CUDAGuard device_guard(device_);
                int64_t stage_free_bytes = -1;
                int64_t stage_required_bytes = -1;
                bool stage_allocated = true;
                if (stage_rows > 0) {
                    stage_allocated =
                        try_allocate_async_cuda_stage(stage_rows, embedding_size_, dtype_, device_,
                                                      static_cast<int64_t>(embedding_size_) * static_cast<int64_t>(dtype_size_),
                                                      gpu_stage, &stage_free_bytes, &stage_required_bytes);
                }
                if (!stage_allocated) {
                    if (single_gpu_async_admit_host_stage_enabled()) {
                        {
                            std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
                            async_admit_preload_admit_ids_ = stage_admit_ids;
                            async_admit_preload_evict_slots_ = stage_evict_slots;
                            async_admit_preload_row_offsets_ = std::move(row_offsets);
                            async_admit_preload_gpu_tensor_ = host_stage;
                            async_admit_preload_hidden_publishes_ = hidden_publishes;
                            async_admit_preload_host_load_ms_ = host_load_ms;
                            async_admit_preload_cpu_to_gpu_ms_ = 0.0;
                            async_admit_preload_total_ms_ = elapsed_ms(total_start, std::chrono::high_resolution_clock::now());
                            async_admit_preload_valid_ = true;
                            async_admit_preload_in_flight_ = false;
                            async_admit_preload_exception_ = nullptr;
                        }
                            if (partition_buffer_swap_timing_enabled()) {
                                SPDLOG_INFO(
                                "[partition-buffer-preload-host-stage] storage={} this={} device={} admit_ids={} rows={} required_mib={:.3f} free_mib={:.3f}",
                                basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(stage_admit_ids), stage_rows,
                                bytes_to_mib(stage_required_bytes), stage_free_bytes >= 0 ? bytes_to_mib(stage_free_bytes) : -1.0);
                        }
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
                        async_admit_preload_valid_ = false;
                        async_admit_preload_in_flight_ = false;
                        async_admit_preload_exception_ = nullptr;
                    }
                    if (partition_buffer_swap_timing_enabled()) {
                        SPDLOG_INFO(
                            "[partition-buffer-preload-skip] storage={} this={} device={} admit_ids={} rows={} required_mib={:.3f} free_mib={:.3f}",
                            basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(stage_admit_ids), stage_rows,
                            bytes_to_mib(stage_required_bytes), stage_free_bytes >= 0 ? bytes_to_mib(stage_free_bytes) : -1.0);
                    }
                    return;
                }
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                if (stage_rows > 0) {
                    bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(host_stage);
                    if (non_blocking_host_copy) {
                        gpu_stage.copy_(host_stage, true);
                    } else {
                        gpu_stage.copy_(host_stage);
                    }
                }
                for (const HiddenFramePublish &hidden_publish : hidden_publishes) {
                    Partition *hidden_partition = partition_table_[hidden_publish.partition_id];
                    int64_t hidden_offset = hidden_publish.frame * partition_size_;
                    torch::Tensor hidden_host_view = buffer_tensor_view_.slice(0, hidden_offset, hidden_offset + hidden_partition->partition_size_);
                    torch::Tensor hidden_gpu_view = buffer_tensor_gpu_view_.slice(0, hidden_offset, hidden_offset + hidden_partition->partition_size_);
                    bool hidden_non_blocking = tensor_supports_non_blocking_host_copy(hidden_host_view);
                    if (hidden_non_blocking) {
                        hidden_gpu_view.copy_(hidden_host_view, true);
                    } else {
                        hidden_gpu_view.copy_(hidden_host_view);
                    }
                }
                cudaStreamSynchronize(copy_stream.stream());
            }
#else
            gpu_stage = host_stage.to(device_);
#endif
            auto after_gpu = std::chrono::high_resolution_clock::now();
            cpu_to_gpu_ms = elapsed_ms(phase_start, after_gpu);

            {
                std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
                async_admit_preload_admit_ids_ = stage_admit_ids;
                async_admit_preload_evict_slots_ = stage_evict_slots;
                async_admit_preload_row_offsets_ = std::move(row_offsets);
                async_admit_preload_gpu_tensor_ = gpu_stage;
                async_admit_preload_hidden_publishes_ = hidden_publishes;
                async_admit_preload_host_load_ms_ = host_load_ms;
                async_admit_preload_cpu_to_gpu_ms_ = cpu_to_gpu_ms;
                async_admit_preload_total_ms_ = elapsed_ms(total_start, after_gpu);
                async_admit_preload_valid_ = true;
                async_admit_preload_in_flight_ = false;
                async_admit_preload_exception_ = nullptr;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
            async_admit_preload_valid_ = false;
            async_admit_preload_in_flight_ = false;
            async_admit_preload_exception_ = std::current_exception();
        }
    });
}

bool MemPartitionBuffer::consumeAsyncAdmitPreload_(const std::vector<int> &admit_ids, const std::vector<int64_t> &evict_slots, double *wait_ms,
                                                   int64_t *visible_install_rows, int64_t *hidden_publish_rows,
                                                   int64_t *visible_install_parts, int64_t *hidden_publish_parts) {
    if (visible_install_rows != nullptr) {
        *visible_install_rows = 0;
    }
    if (hidden_publish_rows != nullptr) {
        *hidden_publish_rows = 0;
    }
    if (visible_install_parts != nullptr) {
        *visible_install_parts = 0;
    }
    if (hidden_publish_parts != nullptr) {
        *hidden_publish_parts = 0;
    }
    auto wait_start = std::chrono::high_resolution_clock::now();
    joinAsyncAdmitPreload_();
    if (wait_ms != nullptr) {
        *wait_ms = elapsed_ms(wait_start, std::chrono::high_resolution_clock::now());
    }

    std::vector<int64_t> row_offsets;
    std::vector<int> stage_admit_ids;
    std::vector<int64_t> stage_evict_slots;
    torch::Tensor gpu_stage;
    std::vector<HiddenFramePublish> hidden_publishes;
    double preload_host_load_ms = 0.0;
    double preload_cpu_to_gpu_ms = 0.0;
    double preload_total_ms = 0.0;
    {
        std::lock_guard<std::mutex> lock(async_admit_preload_lock_);
        hidden_publishes = async_admit_preload_hidden_publishes_;

        std::vector<int> expected_stage_admit_ids = admit_ids;
        std::vector<int64_t> expected_stage_slots = evict_slots;
        if (!hidden_publishes.empty()) {
            if (hidden_publishes.size() > expected_stage_admit_ids.size() || hidden_publishes.size() > expected_stage_slots.size()) {
                async_admit_preload_valid_ = false;
                async_admit_preload_hidden_publishes_.clear();
                return false;
            }
            for (std::size_t idx = 0; idx < hidden_publishes.size(); idx++) {
                if (expected_stage_admit_ids[idx] != hidden_publishes[idx].partition_id ||
                    expected_stage_slots[idx] != hidden_publishes[idx].logical_slot) {
                    async_admit_preload_valid_ = false;
                    async_admit_preload_hidden_publishes_.clear();
                    return false;
                }
            }
            expected_stage_admit_ids.erase(expected_stage_admit_ids.begin(),
                                           expected_stage_admit_ids.begin() + static_cast<std::ptrdiff_t>(hidden_publishes.size()));
            expected_stage_slots.erase(expected_stage_slots.begin(),
                                       expected_stage_slots.begin() + static_cast<std::ptrdiff_t>(hidden_publishes.size()));
        }

        bool stage_matches = async_admit_preload_valid_ && async_admit_preload_admit_ids_ == expected_stage_admit_ids &&
                             async_admit_preload_evict_slots_ == expected_stage_slots;
        if (!stage_matches && hidden_publishes.empty()) {
            async_admit_preload_valid_ = false;
            async_admit_preload_hidden_publishes_.clear();
            return false;
        }
        row_offsets = async_admit_preload_row_offsets_;
        stage_admit_ids = async_admit_preload_admit_ids_;
        stage_evict_slots = async_admit_preload_evict_slots_;
        gpu_stage = async_admit_preload_gpu_tensor_;
        preload_host_load_ms = async_admit_preload_host_load_ms_;
        preload_cpu_to_gpu_ms = async_admit_preload_cpu_to_gpu_ms_;
        preload_total_ms = async_admit_preload_total_ms_;
        async_admit_preload_valid_ = false;
        async_admit_preload_hidden_publishes_.clear();
        async_admit_preload_gpu_tensor_ = torch::Tensor();
    }

    pending_hidden_publishes_ = hidden_publishes;
    int64_t install_rows = (gpu_stage.defined() && gpu_stage.dim() > 0) ? gpu_stage.size(0) : 0;
    int64_t hidden_rows = 0;
    for (const HiddenFramePublish &hidden_publish : hidden_publishes) {
        hidden_rows += partition_table_[hidden_publish.partition_id]->partition_size_;
    }
    if (visible_install_rows != nullptr) {
        *visible_install_rows = install_rows;
    }
    if (hidden_publish_rows != nullptr) {
        *hidden_publish_rows = hidden_rows;
    }
    if (visible_install_parts != nullptr) {
        *visible_install_parts = static_cast<int64_t>(stage_admit_ids.size());
    }
    if (hidden_publish_parts != nullptr) {
        *hidden_publish_parts = static_cast<int64_t>(hidden_publishes.size());
    }

    auto install_start = std::chrono::high_resolution_clock::now();
    if (gpu_stage.defined() && !row_offsets.empty()) {
#ifdef GEGE_CUDA
        c10::cuda::CUDAGuard device_guard(device_);
        auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
        c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
        for (std::size_t idx = 0; idx < row_offsets.size() - 1; idx++) {
            Partition *partition = partition_table_[stage_admit_ids[idx]];
            int64_t dst_offset = logicalSlotRowOffset_(stage_evict_slots[idx]);
            torch::Tensor dst_view = buffer_tensor_gpu_view_.slice(0, dst_offset, dst_offset + partition->partition_size_);
            torch::Tensor src_view = gpu_stage.slice(0, row_offsets[idx], row_offsets[idx + 1]);
            dst_view.copy_(src_view, true);
        }
        cudaStreamSynchronize(copy_stream.stream());
#else
        for (std::size_t idx = 0; idx < row_offsets.size() - 1; idx++) {
            Partition *partition = partition_table_[stage_admit_ids[idx]];
            int64_t dst_offset = logicalSlotRowOffset_(stage_evict_slots[idx]);
            torch::Tensor dst_view = buffer_tensor_gpu_view_.slice(0, dst_offset, dst_offset + partition->partition_size_);
            torch::Tensor src_view = gpu_stage.slice(0, row_offsets[idx], row_offsets[idx + 1]);
            dst_view.copy_(src_view);
        }
#endif
    }
    double preload_install_ms = elapsed_ms(install_start, std::chrono::high_resolution_clock::now());

    if (partition_buffer_swap_timing_enabled()) {
        std::vector<int> hidden_ids;
        std::vector<int64_t> hidden_slots;
        std::vector<int64_t> hidden_frames;
        hidden_ids.reserve(hidden_publishes.size());
        hidden_slots.reserve(hidden_publishes.size());
        hidden_frames.reserve(hidden_publishes.size());
        for (const HiddenFramePublish &hidden_publish : hidden_publishes) {
            hidden_ids.push_back(hidden_publish.partition_id);
            hidden_slots.push_back(hidden_publish.logical_slot);
            hidden_frames.push_back(hidden_publish.frame);
        }
        SPDLOG_INFO("[partition-buffer-preload-consume] storage={} this={} device={} admit_ids={} rows={} visible_install_parts={} visible_install_rows={} hidden_publish_parts={} hidden_publish_rows={} hidden_publish_partitions={} hidden_publish_slots={} hidden_publish_frames={} preload_host_load_ms={:.3f} preload_cpu_to_gpu_ms={:.3f} preload_total_ms={:.3f} preload_install_ms={:.3f}",
                    basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(admit_ids), gpu_stage.size(0),
                    stage_admit_ids.size(), install_rows, hidden_publishes.size(), hidden_rows,
                    vector_prefix_string(hidden_ids), vector_prefix_string(hidden_slots), vector_prefix_string(hidden_frames),
                    preload_host_load_ms, preload_cpu_to_gpu_ms, preload_total_ms, preload_install_ms);
    }
    return true;
}

void MemPartitionBuffer::setPermutation(torch::Tensor perm, torch::Tensor pos) {
    SPDLOG_INFO("setPermutation");
    joinAsyncEvictWriteback_();
    perm_ = perm;
    pos_ = pos;
    refreshHostPartitionRanges_();
}


torch::Tensor MemPartitionBuffer::getGlobalToLocalMap(bool get_current) {

    torch::Tensor buffer_index_map = -torch::ones({total_embeddings_}, torch::kInt64);
    torch::Tensor buffer_state;

    buffer_state = buffer_state_;
#pragma omp parallel for
    for (int i = 0; i < buffer_state.size(0); i++) {
        int partition_id = buffer_state[i].item<int>();
        Partition *partition = partition_table_[partition_id];
        int64_t partition_offset = partition->idx_offset_;
        int64_t buffer_offset = partition->buffer_idx_ * partition_size_;
        torch::Tensor local_indices = torch::arange(buffer_offset, buffer_offset + partition->partition_size_);
        if (hasContiguousHostPartitionRange_(partition_id)) {
            buffer_index_map.narrow(0, contiguousHostPartitionStart_(partition_id), partition->partition_size_).copy_(local_indices);
        } else {
            buffer_index_map.index_put_({pos_.slice(0, partition_offset, partition_offset + partition->partition_size_)}, local_indices);
        }
    }
    return buffer_index_map;
}

void MemPartitionBuffer::load(torch::Tensor data_storage) {
    if (!loaded_) {
        bool log_startup_timing = startup_timing_enabled();
        auto t1 = std::chrono::high_resolution_clock::now();
        int64_t num_nodes = 0;
        data_storage_ = data_storage;
        ensureBackingTensorsAllocated_();
        resetFrameCacheState_();
        int64_t active_slots = get_active_slot_count(buffer_state_, capacity_, "MemPartitionBuffer::load");
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][MemPartitionBuffer::load] begin device={} active_slots={} capacity={} partition_size={} dim={} total_embeddings={}",
                        device_.str(), active_slots, capacity_, partition_size_, embedding_size_, total_embeddings_);
        }

#pragma omp parallel for
        for (int64_t i = 0; i < active_slots; i++) {
            int partition_id = buffer_state_[i].item<int>();
            Partition *partition = partition_table_[partition_id];
            partition->present_ = true;
            partition->buffer_idx_ = static_cast<int>(i);
            partition->physical_frame_idx_ = static_cast<int>(logicalSlotToPhysicalFrame_(i));
            int64_t buffer_offset = partitionRowOffset_(partition);
            copyPartitionFromHostToPinned_(partition, buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_));
            num_nodes += partition->partition_size_;
        }

        loaded_ = true;
        size_.store(num_nodes);
#ifdef GEGE_CUDA
        if (device_.is_cuda()) {
            c10::cuda::CUDAGuard device_guard(device_);
            bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(buffer_tensor_view_);
            if (non_blocking_host_copy) {
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                buffer_tensor_gpu_view_.copy_(buffer_tensor_view_, true);
                cudaStreamSynchronize(copy_stream.stream());
            } else {
                buffer_tensor_gpu_view_.copy_(buffer_tensor_view_);
            }
        } else {
            buffer_tensor_gpu_view_.copy_(buffer_tensor_view_);
        }
#else
        buffer_tensor_gpu_view_.copy_(buffer_tensor_view_);
#endif
        ensureDirtyRowMaskAllocated_();
        if (dirty_row_mask_.defined()) {
            dirty_row_mask_.zero_();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][MemPartitionBuffer::load] end device={} nodes={} total_ms={:.3f}",
                        device_.str(), num_nodes, elapsed_ms(t1, t2));
        }
    }
}

void MemPartitionBuffer::indexAdd(torch::Tensor indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || buffer_tensor_gpu_view_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    torch::Tensor storage_indices = indices.to(torch::kInt64);
    markDirtyRows_(storage_indices);
    int64_t debug_update_id = -1;
    bool run_stage_debug = should_run_stage_debug(debug_update_id);
    bool run_eval_finite_debug = eval_finite_debug_enabled();
    auto index_add_start = std::chrono::high_resolution_clock::now();
    auto step_start = index_add_start;
    torch::Tensor touched_rows;

    if (run_eval_finite_debug) {
        torch::Tensor values_finite = torch::isfinite(values);
        if (!values_finite.all().item<bool>()) {
            torch::Tensor invalid_update_rows = (~values_finite).reshape({values.size(0), -1}).any(1).nonzero().flatten();
            torch::Tensor sample_rows = torch::Tensor();
            if (invalid_update_rows.numel() > 0) {
                sample_rows = storage_indices.index_select(0, invalid_update_rows.slice(0, 0, std::min<int64_t>(invalid_update_rows.numel(), 8)));
            }
            log_non_finite_index_add_rows_if_any("values", device_, debug_update_id, sample_rows,
                                                 values.index_select(0, invalid_update_rows.numel() > 0 ? invalid_update_rows : torch::arange(0, 0, indices.options())));
        }

        touched_rows = std::get<0>(torch::_unique(storage_indices));
    }
    
    if (values.device().is_cuda()) {
#ifdef GEGE_CUDA
        if (csr_update_enabled()) {
            static bool logged = false;
            if (!logged) {
                SPDLOG_INFO("MemPartitionBuffer::indexAdd using direct CSR update path");
                logged = true;
            }
            torch::Tensor update_indices = storage_indices;
            torch::Tensor update_values = values;
            if (csr_update_reduce_enabled()) {
                std::tie(update_indices, update_values) = reduce_updates_with_csr(storage_indices, values);
                if (run_stage_debug) {
                    auto now = std::chrono::high_resolution_clock::now();
                    SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 1] csr_reduce ms={:.3f} in_rows={} out_rows={}",
                                debug_update_id, elapsed_ms(step_start, now), indices.numel(), update_indices.numel());
                    step_start = now;
                }
            }
            buffer_tensor_gpu_view_.index_add_(0, update_indices, update_values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 2] index_add_cuda ms={:.3f} rows={} dim={}",
                            debug_update_id, elapsed_ms(step_start, now), update_indices.numel(), update_values.size(1));
            }
        } else {
            buffer_tensor_gpu_view_.index_add_(0, storage_indices, values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 1] index_add_cuda ms={:.3f} rows={} dim={} csr_update={}",
                            debug_update_id, elapsed_ms(step_start, now), storage_indices.numel(), values.size(1), false);
            }
        }
#else
        buffer_tensor_gpu_view_.index_add_(0, storage_indices, values);
#endif
    } else {
        int d = values.size(1);
        int64_t size = storage_indices.size(0);
        checked_cpu_index_add_("MemPartitionBuffer::indexAdd", buffer_tensor_view_, storage_indices.to(buffer_tensor_view_.device()), values);
        if (run_stage_debug) {
            auto now = std::chrono::high_resolution_clock::now();
            SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 1] index_add_cpu ms={:.3f} rows={} dim={}",
                        debug_update_id, elapsed_ms(step_start, now), size, d);
        }
    }

    if (run_eval_finite_debug && touched_rows.defined() && touched_rows.numel() > 0) {
        torch::Tensor touched_values;
        if (values.device().is_cuda()) {
            touched_values = buffer_tensor_gpu_view_.index_select(0, touched_rows);
        } else {
            touched_values = buffer_tensor_view_.index_select(0, touched_rows.to(buffer_tensor_view_.device()));
        }
        log_non_finite_index_add_rows_if_any("buffer_rows", device_, debug_update_id, touched_rows, touched_values);
    }

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][buffer.indexAdd][update {}][step 9] total_ms={:.3f} device={}",
                    debug_update_id, elapsed_ms(index_add_start, now), values.device().str());
    }
}

void MemPartitionBuffer::indexAddMasked(torch::Tensor indices, torch::Tensor values, torch::Tensor active_mask) {
    if (!active_mask.defined() || active_mask.numel() != indices.numel()) {
        indexAdd(indices, values);
        return;
    }

#ifdef GEGE_CUDA
    if (values.device().is_cuda() && indices.device().is_cuda() && active_mask.device().is_cuda() &&
        buffer_tensor_gpu_view_.device().is_cuda() && values.scalar_type() == torch::kFloat32 &&
        buffer_tensor_gpu_view_.scalar_type() == torch::kFloat32 && active_mask.scalar_type() == torch::kUInt8) {
        bool verify_index_add = false;
        int64_t verify_id = -1;
        torch::Tensor active_rows;
        torch::Tensor active_indices;
        torch::Tensor active_values;
        torch::Tensor expected_values;
        if (fixed_buffer_masked_update_verify_enabled()) {
            verify_id = fixed_buffer_masked_index_add_verify_counter().fetch_add(1);
            verify_index_add = verify_id < fixed_buffer_masked_update_verify_max();
        }
        torch::Tensor storage_indices = indices.to(torch::kInt64);
        if (verify_index_add) {
            torch::NoGradGuard no_grad;
            active_rows = torch::nonzero(active_mask).flatten();
            if (active_rows.numel() > 0) {
                active_indices = storage_indices.index_select(0, active_rows);
                active_values = values.index_select(0, active_rows).clone();
                expected_values = buffer_tensor_gpu_view_.index_select(0, active_indices).clone() + active_values;
            }
        }
        markDirtyRows_(storage_indices, active_mask);
        active_masked_index_add_cuda(buffer_tensor_gpu_view_, storage_indices, values, active_mask);
        if (verify_index_add && active_rows.numel() > 0) {
            torch::Tensor actual_values = buffer_tensor_gpu_view_.index_select(0, active_indices);
            bool values_match = torch::allclose(actual_values, expected_values, 1e-5, 1e-6);
            if (!values_match) {
                double max_abs = (actual_values - expected_values).abs().max().item<double>();
                SPDLOG_ERROR("GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY failed in indexAdd check {} rows={} max_abs={}",
                             verify_id, active_rows.numel(), max_abs);
                throw GegeRuntimeException("Masked indexAdd verifier failed");
            }
            SPDLOG_INFO("GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY indexAdd check {} passed active_rows={}",
                        verify_id, active_rows.numel());
        }
        return;
    }
#endif

    torch::Tensor masked_values = values;
    if (values.device().is_cuda() || active_mask.device().is_cuda()) {
        masked_values = values * active_mask.to(values.device()).to(values.dtype()).reshape({-1, 1});
    }
    indexAdd(indices, masked_values);
}

void MemPartitionBuffer::performNextSwapLegacy_(std::uintptr_t swap_ready_event) {
    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
    auto total_start = std::chrono::high_resolution_clock::now();
    auto phase_start = total_start;
    double gpu_to_cpu_ms = 0.0;
    double host_sync_ms = 0.0;
    double host_load_ms = 0.0;
    double cpu_to_gpu_ms = 0.0;
    double preload_wait_ms = 0.0;
    double setup_ms = 0.0;
    double evict_stage_alloc_ms = 0.0;
    double evict_stage_copy_submit_ms = 0.0;
    double evict_stage_sync_ms = 0.0;
    double evict_verify_ms = 0.0;
    double evict_writeback_submit_ms = 0.0;
    double evict_metadata_ms = 0.0;
    double admit_preload_consume_ms = 0.0;
    double admit_fallback_evict_join_ms = 0.0;
    double admit_fallback_host_load_ms = 0.0;
    double admit_gpu_copy_submit_ms = 0.0;
    double admit_gpu_copy_sync_ms = 0.0;
    double state_bookkeeping_ms = 0.0;
    bool preloaded_admit = false;

    std::vector<int> evict_ids = getNextEvict();
    std::vector<int> admit_ids = getNextAdmit();
    if (evict_ids.size() != admit_ids.size()) {
        throw GegeRuntimeException("MemPartitionBuffer::performNextSwap expected matched evict/admit counts for single-GPU GPU-aware CUSTOM");
    }

    std::vector<int> retained_ids;
    std::unordered_map<int, int64_t> retained_slots_before;
    {
        auto current_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
        auto *state_ptr = current_state.data_ptr<int64_t>();
        for (int64_t slot = 0; slot < current_state.numel(); slot++) {
            int partition_id = static_cast<int>(state_ptr[slot]);
            if (std::find(evict_ids.begin(), evict_ids.end(), partition_id) == evict_ids.end()) {
                retained_ids.emplace_back(partition_id);
                retained_slots_before[partition_id] = partition_table_[partition_id]->buffer_idx_;
            }
        }
    }

    std::vector<int64_t> evict_slots;
    evict_slots.reserve(evict_ids.size());
    for (auto evict_id : evict_ids) {
        Partition *partition = partition_table_[evict_id];
        if (partition->buffer_idx_ < 0) {
            throw GegeRuntimeException("MemPartitionBuffer::performNextSwap encountered an evict partition without a resident buffer slot");
        }
        evict_slots.emplace_back(partition->buffer_idx_);
    }

    int64_t retained_rows = partition_rows_for_ids(retained_ids, partition_table_);
    int64_t evict_rows = partition_rows_for_ids(evict_ids, partition_table_);
    int64_t admit_rows = partition_rows_for_ids(admit_ids, partition_table_);
    int64_t bytes_per_row = static_cast<int64_t>(embedding_size_) * static_cast<int64_t>(dtype_size_);
    int64_t retained_bytes = retained_rows * bytes_per_row;
    int64_t evict_bytes = evict_rows * bytes_per_row;
    int64_t admit_bytes = admit_rows * bytes_per_row;
    int64_t full_async_stage_bytes = evict_bytes + admit_bytes;
    int64_t evict_dirty_rows = -1;
    double evict_dirty_ratio = -1.0;
    int64_t evict_dirty_bytes = -1;
    double dirty_profile_ms = 0.0;
    int64_t cuda_free_bytes = -1;
    int64_t cuda_total_bytes = -1;
    if (log_timing) {
#ifdef GEGE_CUDA
        if (device_.is_cuda()) {
            c10::cuda::CUDAGuard device_guard(device_);
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            AT_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            cuda_free_bytes = static_cast<int64_t>(free_bytes);
            cuda_total_bytes = static_cast<int64_t>(total_bytes);
        }
#endif
        if (mem_partition_buffer_dirty_profile_enabled()) {
            auto dirty_start = std::chrono::high_resolution_clock::now();
            evict_dirty_rows = countDirtyRowsForSlots_(evict_slots, evict_ids);
            if (evict_dirty_rows >= 0 && evict_rows > 0) {
                evict_dirty_ratio = static_cast<double>(evict_dirty_rows) / static_cast<double>(evict_rows);
                evict_dirty_bytes = evict_dirty_rows * bytes_per_row;
            }
            dirty_profile_ms = elapsed_ms(dirty_start, std::chrono::high_resolution_clock::now());
        }
    }

    bool async_evict_writeback = single_gpu_async_evict_writeback_enabled() && evict_rows > 0;
    bool async_evict_memory_fallback = false;
    int64_t async_evict_stage_free_bytes = -1;
    int64_t async_evict_stage_required_bytes = -1;
    std::vector<int64_t> evict_row_offsets;
    torch::Tensor async_evict_gpu_stage;
    torch::Tensor async_evict_expected_host_stage;
    if (async_evict_writeback) {
        evict_row_offsets.reserve(evict_ids.size() + 1);
        evict_row_offsets.emplace_back(0);
        for (int evict_id : evict_ids) {
            evict_row_offsets.emplace_back(evict_row_offsets.back() + partition_table_[evict_id]->partition_size_);
        }
    }
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        setup_ms = elapsed_ms(phase_start, now);
        phase_start = now;
    }

    {
#ifdef GEGE_CUDA
        c10::cuda::CUDAGuard device_guard(device_);
        if (async_evict_writeback) {
            auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
            c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
            if (swap_ready_event != 0) {
                auto ready_event = reinterpret_cast<cudaEvent_t>(swap_ready_event);
                AT_CUDA_CHECK(cudaStreamWaitEvent(copy_stream.stream(), ready_event, 0));
            }
            bool stage_allocated =
                try_allocate_async_cuda_stage(evict_rows, embedding_size_, dtype_, device_, bytes_per_row, async_evict_gpu_stage,
                                              &async_evict_stage_free_bytes, &async_evict_stage_required_bytes);
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                evict_stage_alloc_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
            if (!stage_allocated) {
                async_evict_writeback = false;
                async_evict_memory_fallback = true;
            } else {
                for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                    Partition *partition = partition_table_[evict_ids[idx]];
                    int64_t buffer_offset = evict_slots[idx] * partition_size_;
                    torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                    torch::Tensor stage_view = async_evict_gpu_stage.slice(0, evict_row_offsets[idx], evict_row_offsets[idx + 1]);
                    stage_view.copy_(gpu_view.detach(), true);
                }
                if (log_timing) {
                    auto now = std::chrono::high_resolution_clock::now();
                    evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                    phase_start = now;
                }
                cudaStreamSynchronize(copy_stream.stream());
                if (log_timing) {
                    auto now = std::chrono::high_resolution_clock::now();
                    evict_stage_sync_ms = elapsed_ms(phase_start, now);
                    phase_start = now;
                }
            }
        }
        if (!async_evict_writeback) {
            bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(buffer_tensor_view_);
            if (non_blocking_host_copy) {
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                if (swap_ready_event != 0) {
                    auto ready_event = reinterpret_cast<cudaEvent_t>(swap_ready_event);
                    AT_CUDA_CHECK(cudaStreamWaitEvent(copy_stream.stream(), ready_event, 0));
                }
                for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                    Partition *partition = partition_table_[evict_ids[idx]];
                    int64_t buffer_offset = evict_slots[idx] * partition_size_;
                    torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                    torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                    cpu_view.copy_(gpu_view.detach(), true);
                }
                if (log_timing) {
                    auto now = std::chrono::high_resolution_clock::now();
                    evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                    phase_start = now;
                }
                cudaStreamSynchronize(copy_stream.stream());
                if (log_timing) {
                    auto now = std::chrono::high_resolution_clock::now();
                    evict_stage_sync_ms = elapsed_ms(phase_start, now);
                    phase_start = now;
                }
            } else {
                for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                    Partition *partition = partition_table_[evict_ids[idx]];
                    int64_t buffer_offset = evict_slots[idx] * partition_size_;
                    torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                    torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                    cpu_view.copy_(gpu_view.detach());
                }
                if (log_timing) {
                    auto now = std::chrono::high_resolution_clock::now();
                    evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                    phase_start = now;
                }
            }
        }
#else
        for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
            Partition *partition = partition_table_[evict_ids[idx]];
            int64_t buffer_offset = evict_slots[idx] * partition_size_;
            torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            cpu_view.copy_(gpu_view.detach());
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }
#endif
    }

    if (async_evict_writeback && single_gpu_async_evict_writeback_verify_enabled()) {
        async_evict_expected_host_stage = async_evict_gpu_stage.detach().to(torch::kCPU).contiguous();
    }
    if (async_evict_writeback && single_gpu_async_evict_source_verify_enabled()) {
#ifdef GEGE_CUDA
        synchronize_cuda_storage_device(device_);
#endif
        torch::Tensor source_expected_host_stage =
            torch::empty({evict_rows, embedding_size_}, torch::TensorOptions().dtype(dtype_).device(torch::kCPU));
        for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
            Partition *partition = partition_table_[evict_ids[idx]];
            int64_t buffer_offset = evict_slots[idx] * partition_size_;
            torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            torch::Tensor expected_view = source_expected_host_stage.slice(0, evict_row_offsets[idx], evict_row_offsets[idx + 1]);
            expected_view.copy_(gpu_view.detach());
        }
        torch::Tensor staged_host_stage = async_evict_gpu_stage.detach().to(torch::kCPU).contiguous();
        source_expected_host_stage = source_expected_host_stage.contiguous();
        bool source_match = torch::allclose(staged_host_stage, source_expected_host_stage, 1e-5, 1e-6);
        if (!source_match) {
            double max_abs = (staged_host_stage - source_expected_host_stage).abs().max().item<double>();
            SPDLOG_ERROR("GEGE_SINGLE_GPU_ASYNC_EVICT_SOURCE_VERIFY failed storage={} this={} device={} evict_ids={} rows={} max_abs={}",
                         basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(evict_ids), evict_rows, max_abs);
            throw GegeRuntimeException("Async evict source verifier failed");
        }
        SPDLOG_INFO("GEGE_SINGLE_GPU_ASYNC_EVICT_SOURCE_VERIFY passed storage={} this={} device={} evict_ids={} rows={}",
                    basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(evict_ids), evict_rows);
    }
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        evict_verify_ms = elapsed_ms(phase_start, now);
        gpu_to_cpu_ms = setup_ms + evict_stage_alloc_ms + evict_stage_copy_submit_ms + evict_stage_sync_ms + evict_verify_ms;
        phase_start = now;
    }

    if (async_evict_writeback) {
        startAsyncEvictWriteback_(evict_ids, evict_row_offsets, async_evict_gpu_stage, async_evict_expected_host_stage);
    }
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        evict_writeback_submit_ms = elapsed_ms(phase_start, now);
        phase_start = now;
    }
    for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
        Partition *partition = partition_table_[evict_ids[idx]];
        int64_t buffer_offset = evict_slots[idx] * partition_size_;
        if (!async_evict_writeback) {
            torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            copyPartitionFromPinnedToHost_(partition, cpu_view);
        }
        partition->data_ptr_ = nullptr;
        partition->present_ = false;
        partition->buffer_idx_ = -1;
        partition->physical_frame_idx_ = -1;
    }
    clearDirtyRowsForSlots_(evict_slots, evict_ids);
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        evict_metadata_ms = elapsed_ms(phase_start, now);
        host_sync_ms = evict_writeback_submit_ms + evict_metadata_ms;
        phase_start = now;
    }

    if (asyncAdmitPreloadEnabled_()) {
        auto preload_consume_start = std::chrono::high_resolution_clock::now();
        preloaded_admit = consumeAsyncAdmitPreload_(admit_ids, evict_slots, &preload_wait_ms);
        if (log_timing) {
            admit_preload_consume_ms = elapsed_ms(preload_consume_start, std::chrono::high_resolution_clock::now());
        }
    }
    if (!preloaded_admit) {
        auto join_start = std::chrono::high_resolution_clock::now();
        joinAsyncEvictWritebackForPartitions_(admit_ids);
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            admit_fallback_evict_join_ms = elapsed_ms(join_start, now);
        }
        auto host_load_start = std::chrono::high_resolution_clock::now();
        for (std::size_t idx = 0; idx < admit_ids.size(); idx++) {
            Partition *partition = partition_table_[admit_ids[idx]];
            int64_t buffer_offset = evict_slots[idx] * partition_size_;
            torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            copyPartitionFromHostToPinned_(partition, cpu_view);
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            admit_fallback_host_load_ms = elapsed_ms(host_load_start, now);
        }
    }
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        host_load_ms = preloaded_admit ? admit_preload_consume_ms : elapsed_ms(phase_start, now);
        phase_start = now;
    }

    if (!preloaded_admit) {
#ifdef GEGE_CUDA
        c10::cuda::CUDAGuard device_guard(device_);
        bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(buffer_tensor_view_);
        if (non_blocking_host_copy) {
            auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
            c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
            for (std::size_t idx = 0; idx < admit_ids.size(); idx++) {
                Partition *partition = partition_table_[admit_ids[idx]];
                int64_t buffer_offset = evict_slots[idx] * partition_size_;
                torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                gpu_view.copy_(cpu_view, true);
            }
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                admit_gpu_copy_submit_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
            cudaStreamSynchronize(copy_stream.stream());
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                admit_gpu_copy_sync_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
        } else {
            for (std::size_t idx = 0; idx < admit_ids.size(); idx++) {
                Partition *partition = partition_table_[admit_ids[idx]];
                int64_t buffer_offset = evict_slots[idx] * partition_size_;
                torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                gpu_view.copy_(cpu_view);
            }
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                admit_gpu_copy_submit_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
        }
#else
        for (std::size_t idx = 0; idx < admit_ids.size(); idx++) {
            Partition *partition = partition_table_[admit_ids[idx]];
            int64_t buffer_offset = evict_slots[idx] * partition_size_;
            torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            gpu_view.copy_(cpu_view);
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            admit_gpu_copy_submit_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }
#endif
    }
    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        cpu_to_gpu_ms = admit_gpu_copy_submit_ms + admit_gpu_copy_sync_ms;
        phase_start = now;
    }

    buffer_state_ = *buffer_state_iterator_;
    for (int i = 0; i < buffer_sizes_; i++) {
        if (buffer_states_.end() != buffer_state_iterator_) {
            buffer_state_iterator_++;
        }
    }

    int64_t num_nodes = 0;
    auto next_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *next_state_ptr = next_state.data_ptr<int64_t>();
    for (int64_t slot = 0; slot < next_state.numel(); slot++) {
        int partition_id = static_cast<int>(next_state_ptr[slot]);
        Partition *partition = partition_table_[partition_id];
        if (!partition->present_ && partition->buffer_idx_ == -1) {
            auto admit_it = std::find(admit_ids.begin(), admit_ids.end(), partition_id);
            if (admit_it == admit_ids.end()) {
                throw GegeRuntimeException("MemPartitionBuffer::performNextSwap lost a resident survivor partition during single-GPU GPU-aware CUSTOM swap");
            }
            std::size_t admit_offset = static_cast<std::size_t>(std::distance(admit_ids.begin(), admit_it));
            partition->buffer_idx_ = static_cast<int>(evict_slots[admit_offset]);
            partition->physical_frame_idx_ = static_cast<int>(partition->buffer_idx_);
            partition->present_ = true;
            partition->data_ptr_ = nullptr;
        }
        num_nodes += partition->partition_size_;
    }

    for (int retained_id : retained_ids) {
        Partition *partition = partition_table_[retained_id];
        auto prev_slot_it = retained_slots_before.find(retained_id);
        if (prev_slot_it == retained_slots_before.end()) {
            throw GegeRuntimeException("MemPartitionBuffer::performNextSwap lost retained partition slot metadata");
        }
        if (!partition->present_ || partition->buffer_idx_ != prev_slot_it->second) {
            throw GegeRuntimeException(
                fmt::format("MemPartitionBuffer::performNextSwap changed retained partition {} from slot {} to {}",
                            retained_id, prev_slot_it->second, partition->buffer_idx_));
        }
        partition->physical_frame_idx_ = static_cast<int>(partition->buffer_idx_);
    }

    size_.store(num_nodes);
    loaded_ = true;

    if (log_timing) {
        auto total_end = std::chrono::high_resolution_clock::now();
        int64_t retained_parts = buffer_state_.size(0) - static_cast<int64_t>(admit_ids.size());
        SPDLOG_INFO(
            "[partition-buffer-swap][swap {}] storage={} this={} device={} active_parts={} retained_parts={} evict_parts={} admit_parts={} nodes={} "
            "retained_ids={} evict_ids={} admit_ids={} evict_slots={} retained_mib={:.3f} gpu_to_cpu_mib={:.3f} cpu_to_gpu_mib={:.3f} "
            "full_async_stage_mib={:.3f} cuda_free_mib={:.3f} cuda_total_mib={:.3f} full_async_stage_fits={} "
            "async_evict_required_mib={:.3f} async_evict_free_mib={:.3f} async_evict_memory_fallback={} "
            "evict_dirty_rows={} evict_dirty_ratio={:.6f} dirty_writeback_mib={:.3f} dirty_profile_ms={:.3f} "
            "gpu_to_cpu_ms={:.3f} host_sync_ms={:.3f} host_load_ms={:.3f} cpu_to_gpu_ms={:.3f} preload_wait_ms={:.3f} preloaded_admit={} async_evict_writeback={} "
            "setup_ms={:.3f} evict_stage_alloc_ms={:.3f} evict_stage_copy_submit_ms={:.3f} evict_stage_sync_ms={:.3f} evict_verify_ms={:.3f} "
            "evict_writeback_submit_ms={:.3f} evict_metadata_ms={:.3f} admit_preload_consume_ms={:.3f} admit_fallback_evict_join_ms={:.3f} "
            "admit_fallback_host_load_ms={:.3f} admit_gpu_copy_submit_ms={:.3f} admit_gpu_copy_sync_ms={:.3f} state_bookkeeping_ms={:.3f} total_ms={:.3f}",
            timing_id, basename_string(filename_), fmt::ptr(this), device_.str(), buffer_state_.size(0), retained_parts, evict_ids.size(),
            admit_ids.size(), num_nodes, vector_prefix_string(retained_ids), vector_prefix_string(evict_ids), vector_prefix_string(admit_ids),
            vector_prefix_string(evict_slots), bytes_to_mib(retained_bytes), bytes_to_mib(evict_bytes), bytes_to_mib(admit_bytes),
            bytes_to_mib(full_async_stage_bytes), cuda_free_bytes >= 0 ? bytes_to_mib(cuda_free_bytes) : -1.0,
            cuda_total_bytes >= 0 ? bytes_to_mib(cuda_total_bytes) : -1.0,
            cuda_free_bytes >= 0 ? full_async_stage_bytes < cuda_free_bytes : false,
            async_evict_stage_required_bytes >= 0 ? bytes_to_mib(async_evict_stage_required_bytes) : -1.0,
            async_evict_stage_free_bytes >= 0 ? bytes_to_mib(async_evict_stage_free_bytes) : -1.0,
            async_evict_memory_fallback, evict_dirty_rows, evict_dirty_ratio, evict_dirty_bytes >= 0 ? bytes_to_mib(evict_dirty_bytes) : -1.0,
            dirty_profile_ms, gpu_to_cpu_ms, host_sync_ms, host_load_ms, cpu_to_gpu_ms, preload_wait_ms, preloaded_admit, async_evict_writeback,
            setup_ms, evict_stage_alloc_ms, evict_stage_copy_submit_ms, evict_stage_sync_ms, evict_verify_ms, evict_writeback_submit_ms,
            evict_metadata_ms, admit_preload_consume_ms, admit_fallback_evict_join_ms, admit_fallback_host_load_ms, admit_gpu_copy_submit_ms,
            admit_gpu_copy_sync_ms, state_bookkeeping_ms, elapsed_ms(total_start, total_end));
    }
}


void MemPartitionBuffer::performNextSwap(std::uintptr_t swap_ready_event) {
    if (!buffer_state_.defined() || buffer_state_iterator_ == buffer_states_.end()) {
        return;
    }

#ifdef GEGE_CUDA
    if (storage_sync_before_swap_enabled()) {
        synchronize_cuda_storage_device(device_);
    }
#endif

    if (single_gpu_gpu_aware_custom_enabled() && buffer_sizes_ == 1 && device_.is_cuda() && !frameCacheEnabled_()) {
        performNextSwapLegacy_(swap_ready_event);
        return;
    }

    if (frameCacheEnabled_() && device_.is_cuda()) {
        pending_hidden_publishes_.clear();
        int64_t timing_id = -1;
        bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
        auto total_start = std::chrono::high_resolution_clock::now();
        auto phase_start = total_start;
        double gpu_to_cpu_ms = 0.0;
        double host_sync_ms = 0.0;
        double host_load_ms = 0.0;
        double cpu_to_gpu_ms = 0.0;
        double preload_wait_ms = 0.0;
        double setup_ms = 0.0;
        double evict_stage_alloc_ms = 0.0;
        double evict_stage_copy_submit_ms = 0.0;
        double evict_stage_sync_ms = 0.0;
        double evict_verify_ms = 0.0;
        double evict_writeback_submit_ms = 0.0;
        double evict_metadata_ms = 0.0;
        double admit_preload_consume_ms = 0.0;
        double admit_fallback_evict_join_ms = 0.0;
        double admit_fallback_host_load_ms = 0.0;
        double admit_gpu_copy_submit_ms = 0.0;
        double admit_gpu_copy_sync_ms = 0.0;
        double state_bookkeeping_ms = 0.0;
        bool preloaded_admit = false;
        int64_t preload_visible_install_rows = 0;
        int64_t preload_visible_install_parts = 0;
        int64_t hidden_publish_rows = 0;
        int64_t hidden_publish_parts = 0;
        bool async_admit_preload_in_flight_before_swap = false;
        bool async_admit_preload_valid_before_swap = false;
        int64_t reserved_preload_frames_before_swap = 0;
        bool async_evict_in_flight_before_swap = false;
        int64_t async_evict_in_flight_partitions_before_swap = 0;
        int64_t free_frames_before_swap = -1;
        int64_t stale_backlog_frames_before_swap = -1;
        int64_t free_frames_before_publish = -1;
        int64_t stale_backlog_frames_before_publish = -1;
        int64_t free_frames_after_publish = -1;
        int64_t stale_backlog_frames_after_publish = -1;
        int64_t fallback_visible_admit_parts = 0;
        int64_t fallback_visible_admit_rows = 0;
        bool fallback_due_to_preload_miss = false;
        bool fallback_due_to_partial_preload = false;

        std::vector<int> evict_ids = getNextEvict();
        std::vector<int> admit_ids = getNextAdmit();
        if (evict_ids.size() != admit_ids.size()) {
            throw GegeRuntimeException("MemPartitionBuffer::performNextSwap expected matched evict/admit counts for single-GPU GPU-aware CUSTOM");
        }

        std::vector<int> retained_ids;
        std::unordered_map<int, int64_t> retained_slots_before;
        {
            auto current_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
            auto *state_ptr = current_state.data_ptr<int64_t>();
            for (int64_t slot = 0; slot < current_state.numel(); slot++) {
                int partition_id = static_cast<int>(state_ptr[slot]);
                if (std::find(evict_ids.begin(), evict_ids.end(), partition_id) == evict_ids.end()) {
                    retained_ids.emplace_back(partition_id);
                    retained_slots_before[partition_id] = partition_table_[partition_id]->buffer_idx_;
                }
            }
        }

        std::vector<int64_t> evict_slots;
        evict_slots.reserve(evict_ids.size());
        for (auto evict_id : evict_ids) {
            Partition *partition = partition_table_[evict_id];
            if (partition->buffer_idx_ < 0) {
                throw GegeRuntimeException("MemPartitionBuffer::performNextSwap encountered an evict partition without a resident buffer slot");
            }
            evict_slots.emplace_back(partition->buffer_idx_);
        }

        bool delayed_stale_writeback = false;
        std::vector<int> delayed_stale_partition_ids;
        std::vector<int64_t> delayed_stale_logical_slots;
        std::vector<int64_t> delayed_stale_old_frames;
        std::vector<int64_t> delayed_stale_source_offsets;
        std::vector<int64_t> delayed_stale_row_offsets;
        int64_t delayed_stale_rows = 0;
        if (frame_cache_delayed_stale_writeback_enabled() && !single_gpu_async_evict_writeback_enabled() &&
            frame_cache_hidden_only_preload_enabled() &&
            asyncAdmitPreloadEnabled_() && !admit_ids.empty() && !evict_slots.empty()) {
            joinAsyncAdmitPreload_();
            std::lock_guard<std::mutex> preload_lock(async_admit_preload_lock_);
            if (async_admit_preload_valid_ && !async_admit_preload_hidden_publishes_.empty() && async_admit_preload_admit_ids_.empty() &&
                async_admit_preload_evict_slots_.empty()) {
                delayed_stale_row_offsets.emplace_back(0);
                delayed_stale_writeback = true;
                for (const HiddenFramePublish &hidden_publish : async_admit_preload_hidden_publishes_) {
                    auto slot_it = std::find(evict_slots.begin(), evict_slots.end(), hidden_publish.logical_slot);
                    if (slot_it == evict_slots.end()) {
                        delayed_stale_writeback = false;
                        break;
                    }
                    std::size_t idx = static_cast<std::size_t>(std::distance(evict_slots.begin(), slot_it));
                    delayed_stale_partition_ids.push_back(evict_ids[idx]);
                    delayed_stale_logical_slots.push_back(hidden_publish.logical_slot);
                    int64_t old_frame = logicalSlotToPhysicalFrame_(hidden_publish.logical_slot);
                    delayed_stale_old_frames.push_back(old_frame);
                    delayed_stale_source_offsets.push_back(old_frame * partition_size_);
                    delayed_stale_rows += partition_table_[evict_ids[idx]]->partition_size_;
                    delayed_stale_row_offsets.push_back(delayed_stale_row_offsets.back() + partition_table_[evict_ids[idx]]->partition_size_);
                }
                if (!delayed_stale_writeback) {
                    delayed_stale_partition_ids.clear();
                    delayed_stale_logical_slots.clear();
                    delayed_stale_old_frames.clear();
                    delayed_stale_source_offsets.clear();
                    delayed_stale_row_offsets.clear();
                    delayed_stale_rows = 0;
                }
            }
        }

        std::vector<int> boundary_evict_ids = evict_ids;
        std::vector<int64_t> boundary_evict_slots = evict_slots;
        if (delayed_stale_writeback) {
            for (std::size_t idx = 0; idx < delayed_stale_logical_slots.size(); idx++) {
                auto slot_it = std::find(boundary_evict_slots.begin(), boundary_evict_slots.end(), delayed_stale_logical_slots[idx]);
                if (slot_it != boundary_evict_slots.end()) {
                    std::size_t slot_idx = static_cast<std::size_t>(std::distance(boundary_evict_slots.begin(), slot_it));
                    boundary_evict_slots.erase(slot_it);
                    boundary_evict_ids.erase(boundary_evict_ids.begin() + static_cast<std::ptrdiff_t>(slot_idx));
                }
            }
        }

        int64_t retained_rows = partition_rows_for_ids(retained_ids, partition_table_);
        int64_t evict_rows = partition_rows_for_ids(boundary_evict_ids, partition_table_);
        int64_t admit_rows = partition_rows_for_ids(admit_ids, partition_table_);
        int64_t bytes_per_row = static_cast<int64_t>(embedding_size_) * static_cast<int64_t>(dtype_size_);
        int64_t retained_bytes = retained_rows * bytes_per_row;
        int64_t evict_bytes = evict_rows * bytes_per_row;
        int64_t admit_bytes = admit_rows * bytes_per_row;
        int64_t full_async_stage_bytes = evict_bytes + admit_bytes;
        int64_t evict_dirty_rows = -1;
        double evict_dirty_ratio = -1.0;
        int64_t evict_dirty_bytes = -1;
        double dirty_profile_ms = 0.0;
        int64_t cuda_free_bytes = -1;
        int64_t cuda_total_bytes = -1;
        if (frameCacheEnabled_()) {
            std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
            free_frames_before_swap = static_cast<int64_t>(free_physical_frames_.size());
            stale_backlog_frames_before_swap = static_cast<int64_t>(hidden_frame_capacity_) - free_frames_before_swap;
        }
        {
            std::lock_guard<std::mutex> preload_lock(async_admit_preload_lock_);
            async_admit_preload_in_flight_before_swap = async_admit_preload_in_flight_;
            async_admit_preload_valid_before_swap = async_admit_preload_valid_;
            reserved_preload_frames_before_swap = static_cast<int64_t>(async_admit_preload_hidden_publishes_.size());
        }
        {
            std::lock_guard<std::mutex> evict_lock(async_evict_writeback_lock_);
            async_evict_in_flight_before_swap = async_evict_writeback_in_flight_;
            async_evict_in_flight_partitions_before_swap = static_cast<int64_t>(async_evict_writeback_partition_ids_.size());
        }
        if (log_timing) {
#ifdef GEGE_CUDA
            if (device_.is_cuda()) {
                c10::cuda::CUDAGuard device_guard(device_);
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                AT_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                cuda_free_bytes = static_cast<int64_t>(free_bytes);
                cuda_total_bytes = static_cast<int64_t>(total_bytes);
            }
#endif
            if (mem_partition_buffer_dirty_profile_enabled()) {
                auto dirty_start = std::chrono::high_resolution_clock::now();
                evict_dirty_rows = countDirtyRowsForSlots_(boundary_evict_slots, boundary_evict_ids);
                if (evict_dirty_rows >= 0 && evict_rows > 0) {
                    evict_dirty_ratio = static_cast<double>(evict_dirty_rows) / static_cast<double>(evict_rows);
                    evict_dirty_bytes = evict_dirty_rows * bytes_per_row;
                }
                dirty_profile_ms = elapsed_ms(dirty_start, std::chrono::high_resolution_clock::now());
            }
        }
        bool async_evict_writeback = single_gpu_async_evict_writeback_enabled() && evict_rows > 0;
        bool async_evict_memory_fallback = false;
        int64_t async_evict_stage_free_bytes = -1;
        int64_t async_evict_stage_required_bytes = -1;
        std::vector<int64_t> evict_row_offsets;
        torch::Tensor async_evict_gpu_stage;
        torch::Tensor async_evict_expected_host_stage;
        if (async_evict_writeback) {
            evict_row_offsets.reserve(boundary_evict_ids.size() + 1);
            evict_row_offsets.emplace_back(0);
            for (int evict_id : boundary_evict_ids) {
                evict_row_offsets.emplace_back(evict_row_offsets.back() + partition_table_[evict_id]->partition_size_);
            }
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            setup_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }

        {
#ifdef GEGE_CUDA
            c10::cuda::CUDAGuard device_guard(device_);
            if (async_evict_writeback) {
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                if (swap_ready_event != 0) {
                    auto ready_event = reinterpret_cast<cudaEvent_t>(swap_ready_event);
                    AT_CUDA_CHECK(cudaStreamWaitEvent(copy_stream.stream(), ready_event, 0));
                }
                bool stage_allocated =
                    try_allocate_async_cuda_stage(evict_rows, embedding_size_, dtype_, device_, bytes_per_row, async_evict_gpu_stage,
                                                  &async_evict_stage_free_bytes, &async_evict_stage_required_bytes);
                if (log_timing) {
                    auto now = std::chrono::high_resolution_clock::now();
                    evict_stage_alloc_ms = elapsed_ms(phase_start, now);
                    phase_start = now;
                }
                if (!stage_allocated) {
                    async_evict_writeback = false;
                    async_evict_memory_fallback = true;
                } else {
                    for (std::size_t idx = 0; idx < boundary_evict_ids.size(); idx++) {
                        Partition *partition = partition_table_[boundary_evict_ids[idx]];
                        int64_t buffer_offset = partitionRowOffset_(partition);
                        torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        torch::Tensor stage_view = async_evict_gpu_stage.slice(0, evict_row_offsets[idx], evict_row_offsets[idx + 1]);
                        stage_view.copy_(gpu_view.detach(), true);
                    }
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                    cudaStreamSynchronize(copy_stream.stream());
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        evict_stage_sync_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                }
            }
            if (!async_evict_writeback) {
                bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(buffer_tensor_view_);
                if (non_blocking_host_copy) {
                    auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                    c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                    if (swap_ready_event != 0) {
                        auto ready_event = reinterpret_cast<cudaEvent_t>(swap_ready_event);
                        AT_CUDA_CHECK(cudaStreamWaitEvent(copy_stream.stream(), ready_event, 0));
                    }
                    for (std::size_t idx = 0; idx < boundary_evict_ids.size(); idx++) {
                        Partition *partition = partition_table_[boundary_evict_ids[idx]];
                        int64_t buffer_offset = partitionRowOffset_(partition);
                        torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        cpu_view.copy_(gpu_view.detach(), true);
                    }
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                    cudaStreamSynchronize(copy_stream.stream());
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        evict_stage_sync_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                } else {
                    for (std::size_t idx = 0; idx < boundary_evict_ids.size(); idx++) {
                        Partition *partition = partition_table_[boundary_evict_ids[idx]];
                        int64_t buffer_offset = partitionRowOffset_(partition);
                        torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        cpu_view.copy_(gpu_view.detach());
                    }
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                }
            }
#else
            for (std::size_t idx = 0; idx < boundary_evict_ids.size(); idx++) {
                Partition *partition = partition_table_[boundary_evict_ids[idx]];
                int64_t buffer_offset = partitionRowOffset_(partition);
                torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                cpu_view.copy_(gpu_view.detach());
            }
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                evict_stage_copy_submit_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
#endif
        }
        if (async_evict_writeback && single_gpu_async_evict_writeback_verify_enabled()) {
            async_evict_expected_host_stage = async_evict_gpu_stage.detach().to(torch::kCPU).contiguous();
        }
        if (async_evict_writeback && single_gpu_async_evict_source_verify_enabled()) {
#ifdef GEGE_CUDA
            synchronize_cuda_storage_device(device_);
#endif
            torch::Tensor source_expected_host_stage =
                torch::empty({evict_rows, embedding_size_}, torch::TensorOptions().dtype(dtype_).device(torch::kCPU));
            for (std::size_t idx = 0; idx < boundary_evict_ids.size(); idx++) {
                Partition *partition = partition_table_[boundary_evict_ids[idx]];
                int64_t buffer_offset = partitionRowOffset_(partition);
                torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                torch::Tensor expected_view = source_expected_host_stage.slice(0, evict_row_offsets[idx], evict_row_offsets[idx + 1]);
                expected_view.copy_(gpu_view.detach());
            }
            torch::Tensor staged_host_stage = async_evict_gpu_stage.detach().to(torch::kCPU).contiguous();
            source_expected_host_stage = source_expected_host_stage.contiguous();
            bool source_match = torch::allclose(staged_host_stage, source_expected_host_stage, 1e-5, 1e-6);
            if (!source_match) {
                double max_abs = (staged_host_stage - source_expected_host_stage).abs().max().item<double>();
                SPDLOG_ERROR(
                    "GEGE_SINGLE_GPU_ASYNC_EVICT_SOURCE_VERIFY failed storage={} this={} device={} evict_ids={} rows={} max_abs={}",
                    basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(boundary_evict_ids), evict_rows, max_abs);
                throw GegeRuntimeException("Async evict source verifier failed");
            }
            SPDLOG_INFO("GEGE_SINGLE_GPU_ASYNC_EVICT_SOURCE_VERIFY passed storage={} this={} device={} evict_ids={} rows={}",
                        basename_string(filename_), fmt::ptr(this), device_.str(), vector_prefix_string(boundary_evict_ids), evict_rows);
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            evict_verify_ms = elapsed_ms(phase_start, now);
            gpu_to_cpu_ms = setup_ms + evict_stage_alloc_ms + evict_stage_copy_submit_ms + evict_stage_sync_ms + evict_verify_ms;
            phase_start = now;
        }

        if (async_evict_writeback) {
            startAsyncEvictWriteback_(boundary_evict_ids, evict_row_offsets, async_evict_gpu_stage, async_evict_expected_host_stage);
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            evict_writeback_submit_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }
        for (std::size_t idx = 0; idx < boundary_evict_ids.size(); idx++) {
            Partition *partition = partition_table_[boundary_evict_ids[idx]];
            int64_t buffer_offset = partitionRowOffset_(partition);
            if (!async_evict_writeback) {
                torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                copyPartitionFromPinnedToHost_(partition, cpu_view);
            }
            partition->data_ptr_ = nullptr;
            partition->present_ = false;
            partition->buffer_idx_ = -1;
            partition->physical_frame_idx_ = -1;
        }
        if (delayed_stale_writeback) {
            for (int delayed_partition_id : delayed_stale_partition_ids) {
                Partition *stale_partition = partition_table_[delayed_partition_id];
                stale_partition->data_ptr_ = nullptr;
                stale_partition->present_ = false;
                stale_partition->buffer_idx_ = -1;
                stale_partition->physical_frame_idx_ = -1;
            }
        }
        clearDirtyRowsForSlots_(evict_slots, evict_ids);
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            evict_metadata_ms = elapsed_ms(phase_start, now);
            host_sync_ms = evict_writeback_submit_ms + evict_metadata_ms;
            phase_start = now;
        }

        if (asyncAdmitPreloadEnabled_()) {
            auto preload_consume_start = std::chrono::high_resolution_clock::now();
            preloaded_admit = consumeAsyncAdmitPreload_(admit_ids, evict_slots, &preload_wait_ms,
                                                        &preload_visible_install_rows, &hidden_publish_rows,
                                                        &preload_visible_install_parts, &hidden_publish_parts);
            if (log_timing) {
                admit_preload_consume_ms = elapsed_ms(preload_consume_start, std::chrono::high_resolution_clock::now());
            }
        }
        if (frameCacheEnabled_()) {
            std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
            free_frames_before_publish = static_cast<int64_t>(free_physical_frames_.size());
            stale_backlog_frames_before_publish = static_cast<int64_t>(hidden_frame_capacity_) - free_frames_before_publish;
        }
        std::vector<int> fallback_admit_ids = admit_ids;
        std::vector<int64_t> fallback_evict_slots = evict_slots;
        if (!pending_hidden_publishes_.empty()) {
            std::vector<int> filtered_admit_ids;
            std::vector<int64_t> filtered_evict_slots;
            filtered_admit_ids.reserve(fallback_admit_ids.size());
            filtered_evict_slots.reserve(fallback_evict_slots.size());
            for (std::size_t idx = 0; idx < fallback_admit_ids.size(); idx++) {
                bool is_hidden_publish = false;
                for (const HiddenFramePublish &hidden_publish : pending_hidden_publishes_) {
                    if (fallback_admit_ids[idx] == hidden_publish.partition_id) {
                        is_hidden_publish = true;
                        break;
                    }
                }
                if (!is_hidden_publish) {
                    filtered_admit_ids.push_back(fallback_admit_ids[idx]);
                    filtered_evict_slots.push_back(fallback_evict_slots[idx]);
                }
            }
            fallback_admit_ids = std::move(filtered_admit_ids);
            fallback_evict_slots = std::move(filtered_evict_slots);
        }
        bool fallback_remaining_admits =
            !preloaded_admit ||
            (preloaded_admit && (preload_visible_install_rows + hidden_publish_rows) < admit_rows && !fallback_admit_ids.empty());
        if (fallback_remaining_admits) {
            fallback_due_to_preload_miss = !preloaded_admit;
            fallback_due_to_partial_preload = preloaded_admit;
            if (!preloaded_admit) {
                preload_visible_install_rows = admit_rows;
                preload_visible_install_parts = static_cast<int64_t>(admit_ids.size());
                hidden_publish_rows = 0;
                hidden_publish_parts = 0;
                fallback_admit_ids = admit_ids;
                fallback_evict_slots = evict_slots;
            }
            fallback_visible_admit_parts = static_cast<int64_t>(fallback_admit_ids.size());
            fallback_visible_admit_rows = partition_rows_for_ids(fallback_admit_ids, partition_table_);
            auto join_start = std::chrono::high_resolution_clock::now();
            joinAsyncEvictWritebackForPartitions_(fallback_admit_ids);
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                admit_fallback_evict_join_ms = elapsed_ms(join_start, now);
            }
            auto host_load_start = std::chrono::high_resolution_clock::now();
            for (std::size_t idx = 0; idx < fallback_admit_ids.size(); idx++) {
                Partition *partition = partition_table_[fallback_admit_ids[idx]];
                int64_t buffer_offset = logicalSlotRowOffset_(fallback_evict_slots[idx]);
                torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                copyPartitionFromHostToPinned_(partition, cpu_view);
            }
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                admit_fallback_host_load_ms = elapsed_ms(host_load_start, now);
            }
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            host_load_ms = preloaded_admit ? admit_preload_consume_ms : elapsed_ms(phase_start, now);
            phase_start = now;
        }

        if (fallback_remaining_admits) {
#ifdef GEGE_CUDA
            c10::cuda::CUDAGuard device_guard(device_);
            bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(buffer_tensor_view_);
            if (non_blocking_host_copy) {
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                for (std::size_t idx = 0; idx < fallback_admit_ids.size(); idx++) {
                    Partition *partition = partition_table_[fallback_admit_ids[idx]];
                    int64_t buffer_offset = logicalSlotRowOffset_(fallback_evict_slots[idx]);
                        torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        gpu_view.copy_(cpu_view, true);
                    }
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        admit_gpu_copy_submit_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                    cudaStreamSynchronize(copy_stream.stream());
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        admit_gpu_copy_sync_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                } else {
                    for (std::size_t idx = 0; idx < fallback_admit_ids.size(); idx++) {
                        Partition *partition = partition_table_[fallback_admit_ids[idx]];
                        int64_t buffer_offset = logicalSlotRowOffset_(fallback_evict_slots[idx]);
                        torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                        gpu_view.copy_(cpu_view);
                    }
                    if (log_timing) {
                        auto now = std::chrono::high_resolution_clock::now();
                        admit_gpu_copy_submit_ms = elapsed_ms(phase_start, now);
                        phase_start = now;
                    }
                }
#else
            for (std::size_t idx = 0; idx < fallback_admit_ids.size(); idx++) {
                Partition *partition = partition_table_[fallback_admit_ids[idx]];
                int64_t buffer_offset = logicalSlotRowOffset_(fallback_evict_slots[idx]);
                torch::Tensor cpu_view = buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                torch::Tensor gpu_view = buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                gpu_view.copy_(cpu_view);
            }
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                admit_gpu_copy_submit_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
#endif
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            cpu_to_gpu_ms = admit_gpu_copy_submit_ms + admit_gpu_copy_sync_ms;
            phase_start = now;
        }

        buffer_state_ = *buffer_state_iterator_;
        for (int i = 0; i < buffer_sizes_; i++) {
            if (buffer_states_.end() != buffer_state_iterator_) {
                buffer_state_iterator_++;
            }
        }

        int64_t num_nodes = 0;
        auto next_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
        auto *next_state_ptr = next_state.data_ptr<int64_t>();
        std::unordered_map<int, int64_t> hidden_publish_frames_by_partition;
        std::vector<int> logged_hidden_publish_partition_ids;
        std::vector<int64_t> logged_hidden_publish_logical_slots;
        std::vector<int64_t> logged_hidden_publish_frames;
        if (!pending_hidden_publishes_.empty()) {
            {
                std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
                for (const HiddenFramePublish &hidden_publish : pending_hidden_publishes_) {
                    auto free_it = std::find(free_physical_frames_.begin(), free_physical_frames_.end(), hidden_publish.frame);
                    if (free_it != free_physical_frames_.end()) {
                        free_physical_frames_.erase(free_it);
                    }
                }
            }
            {
                std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
                for (const HiddenFramePublish &hidden_publish : pending_hidden_publishes_) {
                    int64_t published_old_frame = logicalSlotToPhysicalFrame_(hidden_publish.logical_slot);
                    logical_to_physical_frames_[static_cast<std::size_t>(hidden_publish.logical_slot)] = hidden_publish.frame;
                    hidden_publish_frames_by_partition[hidden_publish.partition_id] = hidden_publish.frame;
                    logged_hidden_publish_partition_ids.push_back(hidden_publish.partition_id);
                    logged_hidden_publish_logical_slots.push_back(hidden_publish.logical_slot);
                    logged_hidden_publish_frames.push_back(hidden_publish.frame);
                    bool defer_old_frame_release = delayed_stale_writeback &&
                                                   std::find(delayed_stale_logical_slots.begin(), delayed_stale_logical_slots.end(),
                                                             hidden_publish.logical_slot) != delayed_stale_logical_slots.end();
                    if (!defer_old_frame_release &&
                        std::find(free_physical_frames_.begin(), free_physical_frames_.end(), published_old_frame) == free_physical_frames_.end()) {
                        free_physical_frames_.push_back(published_old_frame);
                    }
                }
            }
            refreshFrameCacheTensors_();
        }
        for (int64_t slot = 0; slot < next_state.numel(); slot++) {
            int partition_id = static_cast<int>(next_state_ptr[slot]);
            Partition *partition = partition_table_[partition_id];
            if (!partition->present_ && partition->buffer_idx_ == -1) {
                auto admit_it = std::find(admit_ids.begin(), admit_ids.end(), partition_id);
                if (admit_it == admit_ids.end()) {
                    throw GegeRuntimeException("MemPartitionBuffer::performNextSwap lost a resident survivor partition during single-GPU GPU-aware CUSTOM swap");
                }
                std::size_t admit_offset = static_cast<std::size_t>(std::distance(admit_ids.begin(), admit_it));
                partition->buffer_idx_ = static_cast<int>(evict_slots[admit_offset]);
                auto hidden_frame_it = hidden_publish_frames_by_partition.find(partition_id);
                if (hidden_frame_it != hidden_publish_frames_by_partition.end()) {
                    partition->physical_frame_idx_ = static_cast<int>(hidden_frame_it->second);
                } else {
                    partition->physical_frame_idx_ = static_cast<int>(logicalSlotToPhysicalFrame_(partition->buffer_idx_));
                }
                partition->present_ = true;
                partition->data_ptr_ = nullptr;
            } else if (partition->present_) {
                partition->physical_frame_idx_ = static_cast<int>(logicalSlotToPhysicalFrame_(partition->buffer_idx_));
            }
            num_nodes += partition->partition_size_;
        }

        if (delayed_stale_writeback && !delayed_stale_partition_ids.empty()) {
            startAsyncEvictWriteback_(delayed_stale_partition_ids, delayed_stale_row_offsets, torch::Tensor(), torch::Tensor(),
                                      delayed_stale_old_frames, delayed_stale_source_offsets);
        }
        if (frameCacheEnabled_()) {
            std::lock_guard<std::mutex> frame_lock(free_physical_frames_lock_);
            free_frames_after_publish = static_cast<int64_t>(free_physical_frames_.size());
            stale_backlog_frames_after_publish = static_cast<int64_t>(hidden_frame_capacity_) - free_frames_after_publish;
        }

        frame_cache_perf_stats_.swap_samples += 1;
        frame_cache_perf_stats_.visible_install_parts += preload_visible_install_parts;
        frame_cache_perf_stats_.visible_install_rows += preload_visible_install_rows;
        frame_cache_perf_stats_.hidden_publish_parts += hidden_publish_parts;
        frame_cache_perf_stats_.hidden_publish_rows += hidden_publish_rows;
        frame_cache_perf_stats_.fallback_visible_admit_parts += fallback_visible_admit_parts;
        frame_cache_perf_stats_.fallback_visible_admit_rows += fallback_visible_admit_rows;
        frame_cache_perf_stats_.preload_miss_swap_count += fallback_due_to_preload_miss ? 1 : 0;
        frame_cache_perf_stats_.partial_preload_swap_count += fallback_due_to_partial_preload ? 1 : 0;
        frame_cache_perf_stats_.delayed_stale_writeback_swap_count += delayed_stale_writeback ? 1 : 0;
        frame_cache_perf_stats_.async_admit_valid_before_swap_count += async_admit_preload_valid_before_swap ? 1 : 0;
        frame_cache_perf_stats_.async_evict_in_flight_before_swap_count += async_evict_in_flight_before_swap ? 1 : 0;
        frame_cache_perf_stats_.reserved_preload_frames_sum += std::max<int64_t>(reserved_preload_frames_before_swap, 0);
        frame_cache_perf_stats_.free_frames_before_swap_sum += std::max<int64_t>(free_frames_before_swap, 0);
        frame_cache_perf_stats_.free_frames_after_publish_sum += std::max<int64_t>(free_frames_after_publish, 0);
        frame_cache_perf_stats_.stale_backlog_before_swap_max =
            std::max(frame_cache_perf_stats_.stale_backlog_before_swap_max, std::max<int64_t>(stale_backlog_frames_before_swap, 0));
        frame_cache_perf_stats_.stale_backlog_after_publish_max =
            std::max(frame_cache_perf_stats_.stale_backlog_after_publish_max, std::max<int64_t>(stale_backlog_frames_after_publish, 0));
        pending_hidden_publishes_.clear();

        for (int retained_id : retained_ids) {
            Partition *partition = partition_table_[retained_id];
            auto prev_slot_it = retained_slots_before.find(retained_id);
            if (prev_slot_it == retained_slots_before.end()) {
                throw GegeRuntimeException("MemPartitionBuffer::performNextSwap lost retained partition slot metadata");
            }
            if (!partition->present_ || partition->buffer_idx_ != prev_slot_it->second) {
                throw GegeRuntimeException(
                    fmt::format("MemPartitionBuffer::performNextSwap changed retained partition {} from slot {} to {}",
                                retained_id, prev_slot_it->second, partition->buffer_idx_));
            }
        }

        size_.store(num_nodes);
        loaded_ = true;

        if (log_timing) {
            auto total_end = std::chrono::high_resolution_clock::now();
            state_bookkeeping_ms = elapsed_ms(phase_start, total_end);
            int64_t retained_parts = buffer_state_.size(0) - static_cast<int64_t>(admit_ids.size());
            SPDLOG_INFO(
                "[partition-buffer-swap][swap {}] storage={} this={} device={} active_parts={} retained_parts={} evict_parts={} admit_parts={} nodes={} "
                "retained_ids={} evict_ids={} admit_ids={} evict_slots={} retained_mib={:.3f} gpu_to_cpu_mib={:.3f} cpu_to_gpu_mib={:.3f} "
                "visible_install_parts={} visible_install_rows={} visible_install_mib={:.3f} hidden_publish_parts={} hidden_publish_rows={} hidden_publish_mib={:.3f} "
                "fallback_visible_admit_parts={} fallback_visible_admit_mib={:.3f} fallback_due_to_preload_miss={} fallback_due_to_partial_preload={} "
                "frame_cache_free_frames_before_swap={} reserved_preload_frames_before_swap={} stale_backlog_frames_before_swap={} "
                "frame_cache_free_frames_before_publish={} stale_backlog_frames_before_publish={} frame_cache_free_frames_after_publish={} stale_backlog_frames_after_publish={} "
                "async_admit_preload_in_flight_before_swap={} async_admit_preload_valid_before_swap={} async_evict_in_flight_before_swap={} async_evict_in_flight_partitions_before_swap={} "
                "hidden_publish_partitions={} hidden_publish_slots={} hidden_publish_frames={} "
                "deferred_stale_writeback={} deferred_stale_partitions={} deferred_stale_slots={} deferred_stale_frames={} deferred_stale_mib={:.3f} "
                "full_async_stage_mib={:.3f} cuda_free_mib={:.3f} cuda_total_mib={:.3f} full_async_stage_fits={} "
                "async_evict_required_mib={:.3f} async_evict_free_mib={:.3f} async_evict_memory_fallback={} "
                "evict_dirty_rows={} evict_dirty_ratio={:.6f} dirty_writeback_mib={:.3f} dirty_profile_ms={:.3f} "
                "gpu_to_cpu_ms={:.3f} host_sync_ms={:.3f} host_load_ms={:.3f} cpu_to_gpu_ms={:.3f} preload_wait_ms={:.3f} preloaded_admit={} async_evict_writeback={} "
                "setup_ms={:.3f} evict_stage_alloc_ms={:.3f} evict_stage_copy_submit_ms={:.3f} evict_stage_sync_ms={:.3f} evict_verify_ms={:.3f} "
                "evict_writeback_submit_ms={:.3f} evict_metadata_ms={:.3f} admit_preload_consume_ms={:.3f} admit_fallback_evict_join_ms={:.3f} "
                "admit_fallback_host_load_ms={:.3f} admit_gpu_copy_submit_ms={:.3f} admit_gpu_copy_sync_ms={:.3f} state_bookkeeping_ms={:.3f} total_ms={:.3f}",
                timing_id, basename_string(filename_), fmt::ptr(this), device_.str(), buffer_state_.size(0), retained_parts, evict_ids.size(), admit_ids.size(),
                num_nodes, vector_prefix_string(retained_ids), vector_prefix_string(evict_ids), vector_prefix_string(admit_ids),
                vector_prefix_string(evict_slots), bytes_to_mib(retained_bytes), bytes_to_mib(evict_bytes), bytes_to_mib(admit_bytes),
                preload_visible_install_parts, preload_visible_install_rows, bytes_to_mib(preload_visible_install_rows * bytes_per_row),
                hidden_publish_parts, hidden_publish_rows, bytes_to_mib(hidden_publish_rows * bytes_per_row),
                fallback_visible_admit_parts, bytes_to_mib(fallback_visible_admit_rows * bytes_per_row),
                fallback_due_to_preload_miss, fallback_due_to_partial_preload,
                free_frames_before_swap, reserved_preload_frames_before_swap, stale_backlog_frames_before_swap,
                free_frames_before_publish, stale_backlog_frames_before_publish, free_frames_after_publish, stale_backlog_frames_after_publish,
                async_admit_preload_in_flight_before_swap, async_admit_preload_valid_before_swap,
                async_evict_in_flight_before_swap, async_evict_in_flight_partitions_before_swap,
                vector_prefix_string(logged_hidden_publish_partition_ids), vector_prefix_string(logged_hidden_publish_logical_slots),
                vector_prefix_string(logged_hidden_publish_frames), delayed_stale_writeback,
                vector_prefix_string(delayed_stale_partition_ids), vector_prefix_string(delayed_stale_logical_slots),
                vector_prefix_string(delayed_stale_old_frames),
                bytes_to_mib(delayed_stale_rows * bytes_per_row),
                bytes_to_mib(full_async_stage_bytes), cuda_free_bytes >= 0 ? bytes_to_mib(cuda_free_bytes) : -1.0,
                cuda_total_bytes >= 0 ? bytes_to_mib(cuda_total_bytes) : -1.0,
                cuda_free_bytes >= 0 ? full_async_stage_bytes < cuda_free_bytes : false,
                async_evict_stage_required_bytes >= 0 ? bytes_to_mib(async_evict_stage_required_bytes) : -1.0,
                async_evict_stage_free_bytes >= 0 ? bytes_to_mib(async_evict_stage_free_bytes) : -1.0,
                async_evict_memory_fallback, evict_dirty_rows, evict_dirty_ratio, evict_dirty_bytes >= 0 ? bytes_to_mib(evict_dirty_bytes) : -1.0,
                dirty_profile_ms, gpu_to_cpu_ms, host_sync_ms, host_load_ms, cpu_to_gpu_ms, preload_wait_ms, preloaded_admit,
                async_evict_writeback, setup_ms, evict_stage_alloc_ms, evict_stage_copy_submit_ms,
                evict_stage_sync_ms, evict_verify_ms, evict_writeback_submit_ms, evict_metadata_ms, admit_preload_consume_ms,
                admit_fallback_evict_join_ms, admit_fallback_host_load_ms, admit_gpu_copy_submit_ms, admit_gpu_copy_sync_ms,
                state_bookkeeping_ms, elapsed_ms(total_start, total_end));
        }
        return;
    }

    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
    auto total_start = std::chrono::high_resolution_clock::now();
    auto t1 = total_start;
    double unload_ms = 0.0;
    double load_ms = 0.0;

    // evict partition
    unload(true);
    auto t2 = std::chrono::high_resolution_clock::now();
    unload_ms = elapsed_ms(t1, t2);
#ifdef GEGE_CUDA
    empty_cache_for_storage_device(device_);
#endif
    
    buffer_state_ = *buffer_state_iterator_;

    for(int i = 0; i < buffer_sizes_; i ++) {
        if (buffer_states_.end() != buffer_state_iterator_)
            buffer_state_iterator_++;
    }

    // admit partition
    t1 = std::chrono::high_resolution_clock::now();
    load(data_storage_);
    t2 = std::chrono::high_resolution_clock::now();
    load_ms = elapsed_ms(t1, t2);
#ifdef GEGE_CUDA
    empty_cache_for_storage_device(device_);
#endif

    int64_t num_nodes = 0;

    int partition_id;
    for (int i = 0; i < buffer_state_.size(0); i++) {
        partition_id = buffer_state_[i].item<int>();
        num_nodes += partition_table_[partition_id]->partition_size_;
    }

    if (log_timing) {
        auto total_end = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO(
            "[partition-buffer-swap][swap {}] storage={} this={} device={} active_parts={} nodes={} unload_ms={:.3f} load_ms={:.3f} total_ms={:.3f}",
            timing_id, basename_string(filename_), fmt::ptr(this), device_.str(), buffer_state_.size(0), num_nodes, unload_ms, load_ms,
            elapsed_ms(total_start, total_end));
    }
}

void MemPartitionBuffer::sync() {
    joinAsyncEvictWriteback_();
    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t active_slots = get_active_slot_count(buffer_state_, capacity_, "MemPartitionBuffer::sync");
    std::vector<int> active_partition_ids;
    active_partition_ids.reserve(static_cast<std::size_t>(active_slots));
    for (int64_t i = 0; i < active_slots; i++) {
        active_partition_ids.emplace_back(buffer_state_[i].item<int>());
    }
#pragma omp parallel for
    for (int64_t i = 0; i < active_slots; i++) {
        int partition_id = buffer_state_[i].item<int>();
        Partition *partition = partition_table_[partition_id];
        int64_t buffer_offset = partitionRowOffset_(partition);
        copyPartitionFromPinnedToHost_(partition, buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_));
        partition->data_ptr_ = nullptr;
        partition->present_ = false;
        partition->buffer_idx_ = -1;
        partition->physical_frame_idx_ = -1;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    if (log_timing) {
        int64_t rows = partition_rows_for_ids(active_partition_ids, partition_table_);
        SPDLOG_INFO("[partition-buffer-swap][sync {}] storage={} this={} device={} parts={} part_ids={} host_rows={} host_mib={:.3f} sync_ms={:.3f}",
                    timing_id, basename_string(filename_), fmt::ptr(this), device_.str(), buffer_state_.defined() ? buffer_state_.size(0) : 0,
                    vector_prefix_string(active_partition_ids), rows, bytes_to_mib(rows * static_cast<int64_t>(embedding_size_) * dtype_size_),
                    elapsed_ms(t1, t2));
    }
}



void MemPartitionBuffer::setBufferOrdering(std::vector<torch::Tensor> buffer_states) {
    buffer_states_ = buffer_states;
    if (buffer_states_.empty()) {
        throw GegeRuntimeException("MemPartitionBuffer::setBufferOrdering requires at least one buffer state");
    }

    buffer_state_iterator_ = buffer_states_.begin();

    int device_index = device_.index();
    if (device_index < 0) {
        throw GegeRuntimeException("MemPartitionBuffer::setBufferOrdering requires a concrete CUDA device index");
    }

    for (int i = 0; i < device_index; i++) {
        ++buffer_state_iterator_;
        if (buffer_state_iterator_ == buffer_states_.end()) {
            throw GegeRuntimeException("MemPartitionBuffer::setBufferOrdering ran out of states before reaching the device offset");
        }
    }

    if (buffer_state_iterator_ == buffer_states_.end()) {
        throw GegeRuntimeException("MemPartitionBuffer::setBufferOrdering could not assign an initial state");
    }

    buffer_state_ = *buffer_state_iterator_;

    for (int i = 0; i < buffer_sizes_ && buffer_state_iterator_ != buffer_states_.end(); i++) {
        ++buffer_state_iterator_;
    }
    loaded_ = false;
}

void MemPartitionBuffer::unload(bool write) {

    if (loaded_) {
        joinAsyncEvictWriteback_();
#ifdef GEGE_CUDA
        if (storage_sync_before_swap_enabled()) {
            synchronize_cuda_storage_device(device_);
        }
#endif
        int64_t timing_id = -1;
        bool log_timing = should_log_partition_buffer_swap_timing(timing_id);
        int64_t eval_finite_log_id = -1;
        bool log_eval_finite = write && should_log_eval_finite_debug(eval_finite_log_id);
        auto total_start = std::chrono::high_resolution_clock::now();
        auto phase_start = total_start;
        double gpu_to_cpu_ms = 0.0;
        if (write && buffer_tensor_gpu_view_.device().is_cuda()) {
#ifdef GEGE_CUDA
            c10::cuda::CUDAGuard device_guard(device_);
            bool non_blocking_host_copy = tensor_supports_non_blocking_host_copy(buffer_tensor_view_);
            if (non_blocking_host_copy) {
                auto copy_stream = c10::cuda::getStreamFromPool(false, device_.index());
                c10::cuda::CUDAStreamGuard stream_guard(copy_stream);
                buffer_tensor_view_.copy_(buffer_tensor_gpu_view_.detach(), true);
                cudaStreamSynchronize(copy_stream.stream());
            } else {
                buffer_tensor_view_.copy_(buffer_tensor_gpu_view_.detach());
            }
#else
            buffer_tensor_view_.copy_(buffer_tensor_gpu_view_.detach());
#endif
            if (log_timing) {
                auto now = std::chrono::high_resolution_clock::now();
                gpu_to_cpu_ms = elapsed_ms(phase_start, now);
                phase_start = now;
            }
            buff_mem_unload_ = buffer_tensor_view_.data_ptr();

        }

        if (write && log_eval_finite) {
            auto active_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
            auto *state_ptr = active_state.data_ptr<int64_t>();
            for (int64_t slot = 0; slot < active_state.numel(); slot++) {
                int partition_id = static_cast<int>(state_ptr[slot]);
                Partition *partition = partition_table_[partition_id];
                int64_t buffer_offset = partitionRowOffset_(partition);
                torch::Tensor partition_view =
                    buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                log_non_finite_rows_if_any("gpu_copy", eval_finite_log_id, device_, partition_id, partition_view);
            }
        }

        if (write) {
            sync();
        } else if (buffer_state_.defined()) {
            auto active_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
            auto *state_ptr = active_state.data_ptr<int64_t>();
            for (int64_t slot = 0; slot < active_state.numel(); slot++) {
                int partition_id = static_cast<int>(state_ptr[slot]);
                Partition *partition = partition_table_[partition_id];
                partition->data_ptr_ = nullptr;
                partition->present_ = false;
                partition->buffer_idx_ = -1;
                partition->physical_frame_idx_ = -1;
            }
        }
        double sync_ms = 0.0;
        if (write && log_eval_finite) {
            auto active_state = buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
            auto *state_ptr = active_state.data_ptr<int64_t>();
            for (int64_t slot = 0; slot < active_state.numel(); slot++) {
                int partition_id = static_cast<int>(state_ptr[slot]);
                Partition *partition = partition_table_[partition_id];
                torch::Tensor host_snapshot;
                if (hasContiguousHostPartitionRange_(partition_id)) {
                    host_snapshot = data_storage_.narrow(0, contiguousHostPartitionStart_(partition_id), partition->partition_size_);
                } else {
                    torch::Tensor global_ids =
                        pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_);
                    host_snapshot = data_storage_.index_select(0, global_ids);
                }
                log_non_finite_rows_if_any("host_sync", eval_finite_log_id, device_, partition_id, host_snapshot);
            }
        }
        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            sync_ms = elapsed_ms(phase_start, now);
            SPDLOG_INFO(
                "[partition-buffer-swap][unload {}] storage={} this={} device={} loaded_parts={} gpu_to_cpu_ms={:.3f} sync_ms={:.3f} total_ms={:.3f}",
                timing_id, basename_string(filename_), fmt::ptr(this), device_.str(), buffer_state_.defined() ? buffer_state_.size(0) : 0, gpu_to_cpu_ms,
                sync_ms, elapsed_ms(total_start, now));
        }
        // buffer_tensor_view_ = torch::Tensor();
        // buff_mem_unload_ = nullptr;

        size_ = 0;
        loaded_ = false;
    }
}

void MemPartitionBuffer::evict(std::vector<Partition *> evict_partitions) {
    if (prefetching_) {
        async_write_block_->async_write(evict_partitions);
    } else {
        unload(true);
    }
}

void MemPartitionBuffer::admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs) {
    if (admit_partitions.size() > buffer_idxs.size()) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    if (prefetching_) {
        std::vector<void *> buff_addrs(buffer_idxs.size());

#pragma omp parallel for
        for (int i = 0; i < buffer_idxs.size(); i++) {
            void *buff_addr = (char *)buff_mem_ + (buffer_idxs[i] * partition_size_ * embedding_size_ * dtype_size_);
            buff_addrs[i] = buff_addr;
        }

        std::vector<int> next_admit_ids = getNextAdmit();
        std::vector<Partition *> next_partitions;
        if (!next_admit_ids.empty()) {
            for (int admit_id : next_admit_ids) {
                next_partitions.emplace_back(partition_table_[admit_id]);
            }
        }
        lookahead_block_->move_to_buffer(buff_addrs, buffer_idxs, next_partitions);
    } else {
        load(data_storage_);
    }
}

torch::Tensor MemPartitionBuffer::indexRead(torch::Tensor indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    return buffer_tensor_gpu_view_.index_select(0, indices.to(torch::kInt64));
}
