#include "storage/storage.h"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <string>

#include "common/util.h"
#include "configuration/constants.h"
#include "reporting/logger.h"

#if defined(GEGE_CUDA)
#include "pytorch_scatter/segment_sum.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
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

using std::ios;
using std::ios_base;

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

bool partition_buffer_peer_relay_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_PEER_RELAY", false);
    return enabled;
}

bool multi_gpu_async_admit_preload_enabled() {
    static bool enabled = parse_env_flag("GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD", false);
    return enabled;
}

enum class StateflowPeerRuntimeMode {
    AUTO = 0,
    ON = 1,
    OFF = 2,
};

enum class StateflowPeerRuntimeScope {
    ALL = 0,
    EMBEDDINGS = 1,
};

StateflowPeerRuntimeMode stateflow_peer_runtime_mode() {
    static StateflowPeerRuntimeMode mode = []() {
        const char *raw = std::getenv("GEGE_STATEFLOW_PEER_RUNTIME");
        if (raw == nullptr) {
            return StateflowPeerRuntimeMode::AUTO;
        }
        std::string value(raw);
        if (value == "off" || value == "OFF" || value == "Off" || value == "0") {
            return StateflowPeerRuntimeMode::OFF;
        }
        if (value == "on" || value == "ON" || value == "On" || value == "1") {
            return StateflowPeerRuntimeMode::ON;
        }
        return StateflowPeerRuntimeMode::AUTO;
    }();
    return mode;
}

const char *stateflow_peer_runtime_mode_name(StateflowPeerRuntimeMode mode) {
    switch (mode) {
        case StateflowPeerRuntimeMode::AUTO:
            return "auto";
        case StateflowPeerRuntimeMode::ON:
            return "on";
        case StateflowPeerRuntimeMode::OFF:
            return "off";
    }
    return "auto";
}

StateflowPeerRuntimeScope stateflow_peer_runtime_scope() {
    static StateflowPeerRuntimeScope scope = []() {
        const char *raw = std::getenv("GEGE_STATEFLOW_PEER_RUNTIME_SCOPE");
        if (raw == nullptr) {
            return StateflowPeerRuntimeScope::ALL;
        }
        std::string value(raw);
        if (value == "embeddings" || value == "EMBEDDINGS" || value == "embedding" || value == "EMBEDDING") {
            return StateflowPeerRuntimeScope::EMBEDDINGS;
        }
        return StateflowPeerRuntimeScope::ALL;
    }();
    return scope;
}

const char *stateflow_peer_runtime_scope_name(StateflowPeerRuntimeScope scope) {
    switch (scope) {
        case StateflowPeerRuntimeScope::ALL:
            return "all";
        case StateflowPeerRuntimeScope::EMBEDDINGS:
            return "embeddings";
    }
    return "all";
}

bool stateflow_optimizer_state_storage_filename(const std::string &filename) {
    return filename.find(PathConstants::embeddings_state_file) != std::string::npos ||
           filename.find(PathConstants::embeddings_g_state_file) != std::string::npos ||
           filename.find("qual_embeddings_state") != std::string::npos ||
           filename.find("optimizer_state") != std::string::npos ||
           filename.find("_state.bin") != std::string::npos;
}

const char *stateflow_storage_scope_skip_name(const std::string &filename) {
    return stateflow_optimizer_state_storage_filename(filename) ? "optimizer-state storage" : "storage";
}

int64_t peer_handoff_lookup_key(int64_t round_idx, int partition_id) {
    constexpr uint64_t kRoundMix = 0x9E3779B185EBCA87ULL;
    uint64_t round_bits = static_cast<uint64_t>(round_idx);
    uint64_t partition_bits = static_cast<uint32_t>(partition_id);
    return static_cast<int64_t>((round_bits * kRoundMix) ^ partition_bits);
}

bool startup_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_STARTUP_TIMING", false);
    return enabled;
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

bool eval_finite_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_EVAL_FINITE_DEBUG", false);
    return enabled;
}

int64_t eval_finite_debug_max_logs() {
    static int64_t max_logs = std::max<int64_t>(parse_env_int("GEGE_EVAL_FINITE_DEBUG_MAX_LOGS", 32), 0);
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

void log_non_finite_rows_if_any(const char *stage, int partition_id, int src_dev, int dst_dev, const torch::Tensor &tensor) {
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
    int64_t invalid_rows = invalid_values;
    if (tensor.dim() >= 2) {
        invalid_rows = (~finite).reshape({tensor.size(0), -1}).any(1).sum().item<int64_t>();
    }

    SPDLOG_ERROR("[eval-finite-debug][peer-relay {}][{}] partition={} src_dev={} dst_dev={} invalid_values={} invalid_rows={} shape={}",
                 log_id, stage, partition_id, src_dev, dst_dev, invalid_values, invalid_rows, tensor_shape_string(tensor));
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

void checked_cpu_index_put_(const char *context, torch::Tensor &target, torch::Tensor indices, torch::Tensor values) {
    if (!target.device().is_cpu() || !indices.device().is_cpu() || !values.device().is_cpu()) {
        throw GegeRuntimeException(std::string(context) + " expects CPU tensors on the CPU update path");
    }

    if (target.scalar_type() != values.scalar_type()) {
        throw GegeRuntimeException(std::string(context) + " requires target and values to share dtype");
    }

    if (indices.scalar_type() != torch::kInt64) {
        indices = indices.to(torch::kInt64);
    }

    torch::Tensor unique_indices = std::get<0>(torch::_unique(indices));
    if (unique_indices.numel() != indices.numel()) {
        throw GegeRuntimeException(std::string(context) + " does not support duplicate indices on the CPU write path");
    }

    target.index_put_({indices}, values);
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
    ScopedNvtxRange nvtx_scope("storage.reduce_updates_with_csr");

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

void renameFile(string old_filename, string new_filename) {
    int result = rename(old_filename.c_str(), new_filename.c_str());
    if (result != 0) {
        SPDLOG_ERROR("Unable to rename {}\nError: {}", old_filename, errno);
        throw std::runtime_error("");
    }
}

void copyFile(string src_file, string dst_file) {
    std::ifstream src;
    std::ofstream dst;

    src.open(src_file, ios::in | ios::binary);
    dst.open(dst_file, ios::out | ios::binary);

    dst << src.rdbuf();

    src.close();
    dst.close();
}

bool fileExists(string filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

void createDir(string path, bool exist_ok) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        if (errno == EEXIST) {
            if (exist_ok) {
                SPDLOG_DEBUG("{} directory already exists", path);
            } else {
                SPDLOG_ERROR("{} directory already exists", path);
                throw std::runtime_error("");
            }
        } else {
            SPDLOG_ERROR("Failed to create {}\nError: {}", path, errno);
            throw std::runtime_error("");
        }
    }
}

Storage::Storage() : device_(torch::kCPU) {}

PartitionBufferStorage::PartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, torch::Tensor data, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    options_ = options;
    dtype_ = options_->dtype;
    append(data);
    initialized_ = true;
    loaded_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

PartitionBufferStorage::PartitionBufferStorage(string filename, shared_ptr<PartitionBufferOptions> options) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    options_ = options;
    dtype_ = options_->dtype;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCPU;

    buffer_ = new PartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching);
}

void PartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    close(fd);
}

void PartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

PartitionBufferStorage::~PartitionBufferStorage() { delete buffer_; }

void PartitionBufferStorage::load() {
    if (!loaded_ && initialized_) {
        buffer_->load();
        loaded_ = true;
    }
}

void PartitionBufferStorage::write() {
    if (loaded_) {
        buffer_->sync();
    }
}

void PartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        buffer_->unload(perform_write);
        loaded_ = false;
    }
}

torch::Tensor PartitionBufferStorage::indexRead(Indices indices) { return buffer_->indexRead(indices); }

void PartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { return buffer_->indexAdd(indices, values); }

torch::Tensor PartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for PartitionBufferStorage");
    throw std::runtime_error("");
}

void PartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Shuffle not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

void PartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for PartitionBufferStorage");
    throw std::runtime_error("");
};

MemPartitionBufferStorage::MemPartitionBufferStorage(string filename, int64_t dim0_size, int64_t dim1_size, shared_ptr<PartitionBufferOptions> options, std::vector<torch::Device> devices) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    options_ = options;
    dtype_ = options_->dtype;
    initialized_ = true;
    loaded_ = false;
    peer_relay_runtime_enabled_ = false;
    peer_relay_init_attempted_ = false;
    int64_t partition_size = ceil((double)dim0_size_ / options_->num_partitions);
    device_ = torch::kCUDA;
    devices_ = devices;
    stateflow_peer_handoff_index_per_device_.resize(devices_.size());
    stateflow_transition_counts_.assign(devices_.size(), 0);
    device_peer_bytes_executed_.assign(devices_.size(), 0);
    device_host_fallback_bytes_.assign(devices_.size(), 0);
    device_peer_copy_count_.assign(devices_.size(), 0);
    device_host_fallback_count_.assign(devices_.size(), 0);
    device_descriptor_mismatch_count_.assign(devices_.size(), 0);
    device_peer_sync_wait_ns_.assign(devices_.size(), 0);
    stateflow_peer_mismatch_warned_keys_.resize(devices_.size());
    bool log_startup_timing = startup_timing_enabled();
    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::ctor] begin filename={} dim0={} dim1={} devices={} partition_size={} capacity={}",
                    filename_, dim0_size_, dim1_size_, devices_.size(), partition_size, options_->buffer_capacity);
    }
    for (int i = 0; i < devices_.size(); i ++) {
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::ctor] begin buffer_idx={} device={}", i, devices_[i].str());
        }
        MemPartitionBuffer* buffer = new MemPartitionBuffer(options_->buffer_capacity, options_->num_partitions, options_->fine_to_coarse_ratio, partition_size, dim1_size_, dim0_size_,
                                  dtype_, filename_, options_->prefetching, devices_[i], devices_.size());
        buffers_.emplace_back(buffer);
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::ctor] end buffer_idx={} device={}", i, devices_[i].str());
        }
    }
    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::ctor] end filename={} buffers={}", filename_, buffers_.size());
    }
}

void MemPartitionBufferStorage::setStateflowPeerHandoffs(const std::vector<PeerHandoffDescriptor> &peer_handoffs) {
    bool buffers_loaded = std::any_of(buffers_.begin(), buffers_.end(), [](const MemPartitionBuffer *buffer) { return buffer->loaded_; });
    if (buffers_loaded) {
        throw GegeRuntimeException(
            "MemPartitionBufferStorage::setStateflowPeerHandoffs does not support mid-training schedule updates after buffers are loaded");
    }

    for (auto &index : stateflow_peer_handoff_index_per_device_) {
        index.clear();
    }
    stateflow_peer_schedule_active_ = !peer_handoffs.empty();
    resetStateflowTransitionCounts();

    for (const auto &handoff : peer_handoffs) {
        if (handoff.dst_lane_id < 0 || static_cast<std::size_t>(handoff.dst_lane_id) >= stateflow_peer_handoff_index_per_device_.size()) {
            throw GegeRuntimeException(fmt::format("Stateflow peer handoff has invalid dst_lane_id={}", handoff.dst_lane_id));
        }
        if (handoff.round_idx < 0 || handoff.partition_id < 0) {
            throw GegeRuntimeException("Stateflow peer handoff is missing round_idx or partition_id");
        }
        int64_t key = peer_handoff_lookup_key(handoff.round_idx, handoff.partition_id);
        auto &index = stateflow_peer_handoff_index_per_device_[handoff.dst_lane_id];
        if (!index.emplace(key, handoff).second) {
            throw GegeRuntimeException(fmt::format(
                "Duplicate Stateflow peer handoff for dst_lane={} round={} partition={}",
                handoff.dst_lane_id, handoff.round_idx, handoff.partition_id));
        }
    }

    resetPeerRelayPerfStats();
    std::lock_guard<std::mutex> lock(peer_relay_init_lock_);
    peer_relay_init_attempted_ = false;
    peer_relay_runtime_enabled_ = false;
}

PeerRelayPerfStats MemPartitionBufferStorage::getPeerRelayPerfStats() const {
    PeerRelayPerfStats stats;
    stats.device_peer_bytes_executed = device_peer_bytes_executed_;
    stats.device_host_fallback_bytes = device_host_fallback_bytes_;
    stats.device_peer_copy_count = device_peer_copy_count_;
    stats.device_host_fallback_count = device_host_fallback_count_;
    stats.device_descriptor_mismatch_count = device_descriptor_mismatch_count_;
    stats.device_peer_sync_wait_ns = device_peer_sync_wait_ns_;
    for (std::size_t idx = 0; idx < device_peer_bytes_executed_.size(); idx++) {
        stats.peer_bytes_executed += device_peer_bytes_executed_[idx];
        stats.host_fallback_bytes += idx < device_host_fallback_bytes_.size() ? device_host_fallback_bytes_[idx] : 0;
        stats.peer_copy_count += idx < device_peer_copy_count_.size() ? device_peer_copy_count_[idx] : 0;
        stats.host_fallback_count += idx < device_host_fallback_count_.size() ? device_host_fallback_count_[idx] : 0;
        stats.descriptor_mismatch_count += idx < device_descriptor_mismatch_count_.size() ? device_descriptor_mismatch_count_[idx] : 0;
        stats.peer_sync_wait_ns += idx < device_peer_sync_wait_ns_.size() ? device_peer_sync_wait_ns_[idx] : 0;
    }
    return stats;
}

void MemPartitionBufferStorage::resetPeerRelayPerfStats() {
    std::fill(device_peer_bytes_executed_.begin(), device_peer_bytes_executed_.end(), 0);
    std::fill(device_host_fallback_bytes_.begin(), device_host_fallback_bytes_.end(), 0);
    std::fill(device_peer_copy_count_.begin(), device_peer_copy_count_.end(), 0);
    std::fill(device_host_fallback_count_.begin(), device_host_fallback_count_.end(), 0);
    std::fill(device_descriptor_mismatch_count_.begin(), device_descriptor_mismatch_count_.end(), 0);
    std::fill(device_peer_sync_wait_ns_.begin(), device_peer_sync_wait_ns_.end(), 0);
    for (auto &warned_keys : stateflow_peer_mismatch_warned_keys_) {
        warned_keys.clear();
    }
}

void MemPartitionBufferStorage::resetStateflowTransitionCounts() {
    std::fill(stateflow_transition_counts_.begin(), stateflow_transition_counts_.end(), 0);
}

void MemPartitionBufferStorage::rebuildStateflowGlobalNextRequired_(const std::vector<torch::Tensor> &buffer_states) {
    stateflow_global_next_required_by_round_.clear();
    stateflow_global_next_required_by_round_.emplace_back();

    if (devices_.empty() || options_ == nullptr || buffer_states.empty()) {
        return;
    }

    std::size_t lane_count = devices_.size();
    for (std::size_t round_idx = 1;; round_idx++) {
        std::vector<uint8_t> next_required(static_cast<std::size_t>(options_->num_partitions), 0);
        bool found_round = false;
        for (std::size_t lane = 0; lane < lane_count; lane++) {
            std::size_t state_idx = round_idx * lane_count + lane;
            if (state_idx >= buffer_states.size()) {
                continue;
            }
            found_round = true;
            auto lane_next_state = buffer_states[state_idx].to(torch::kCPU).to(torch::kInt64).contiguous();
            auto *lane_next_ptr = lane_next_state.data_ptr<int64_t>();
            for (int64_t i = 0; i < lane_next_state.numel(); i++) {
                int64_t partition_id = lane_next_ptr[i];
                if (partition_id >= 0 && partition_id < static_cast<int64_t>(next_required.size())) {
                    next_required[static_cast<std::size_t>(partition_id)] = 1;
                }
            }
        }

        if (!found_round) {
            break;
        }
        stateflow_global_next_required_by_round_.emplace_back(std::move(next_required));
    }
}

void MemPartitionBufferStorage::initializePeerRelay_() {
    peer_relay_runtime_enabled_ = false;
    peer_relay_source_scratch_tensors_.clear();
    peer_relay_source_scratch_pool_tensors_.clear();
    peer_relay_source_scratch_frames_.clear();
    peer_relay_source_slot_snapshots_.clear();
    peer_relay_source_scratch_rounds_.clear();
#if defined(GEGE_CUDA)
    for (auto &ready_event_handle : peer_relay_source_ready_events_) {
        if (ready_event_handle != 0) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ready_event_handle));
            ready_event_handle = 0;
        }
    }
#endif
    peer_relay_source_ready_events_.clear();
    peer_relay_source_publish_mutexes_.clear();
    peer_relay_source_publish_cvs_.clear();
    peer_relay_source_published_rounds_.clear();

    StateflowPeerRuntimeMode stateflow_mode = stateflow_peer_runtime_mode();
    StateflowPeerRuntimeScope stateflow_scope = stateflow_peer_runtime_scope();
    bool stateflow_runtime_scope_allows_storage =
        stateflow_scope == StateflowPeerRuntimeScope::ALL || !stateflow_optimizer_state_storage_filename(filename_);
    bool stateflow_runtime_requested =
        stateflow_peer_schedule_active_ && stateflow_mode != StateflowPeerRuntimeMode::OFF && stateflow_runtime_scope_allows_storage;
    bool low_level_runtime_requested = partition_buffer_peer_relay_enabled();
    bool force_peer_runtime =
        stateflow_peer_schedule_active_ && stateflow_mode == StateflowPeerRuntimeMode::ON && stateflow_runtime_scope_allows_storage;

    if (stateflow_peer_schedule_active_ && stateflow_mode != StateflowPeerRuntimeMode::OFF && !stateflow_runtime_scope_allows_storage &&
        !low_level_runtime_requested) {
        SPDLOG_INFO("Skipping Stateflow peer relay runtime for {}={} because GEGE_STATEFLOW_PEER_RUNTIME_SCOPE={}",
                    stateflow_storage_scope_skip_name(filename_), filename_, stateflow_peer_runtime_scope_name(stateflow_scope));
    }

    auto fail_or_fallback = [&](const std::string &message, bool warn = true) {
#if defined(GEGE_CUDA)
        // Clear any sticky CUDA runtime status before falling back so later
        // unrelated launch checks do not observe stale peer-relay failures.
        cudaGetLastError();
#endif
        if (force_peer_runtime) {
            throw GegeRuntimeException(message);
        }
        if (warn) {
            SPDLOG_WARN("{}", message);
        } else {
            SPDLOG_INFO("{}", message);
        }
    };

    if (!low_level_runtime_requested && !stateflow_runtime_requested) {
        return;
    }

#if !defined(GEGE_CUDA)
    fail_or_fallback("Stateflow/partition peer relay requested but GEGE_CUDA is not enabled; falling back to CPU-backed swaps");
    return;
#else
    if (devices_.size() <= 1) {
        fail_or_fallback(fmt::format("Stateflow/partition peer relay requested but only {} physical CUDA device is active; falling back to CPU-backed swaps",
                                     devices_.size()),
                         false);
        return;
    }

    if (options_ != nullptr && options_->prefetching) {
        fail_or_fallback("Stateflow/partition peer relay does not support partition prefetching; falling back to CPU-backed swaps");
        return;
    }

    if (options_ != nullptr && options_->edge_bucket_ordering != EdgeBucketOrdering::CUSTOM) {
        fail_or_fallback("Stateflow/partition peer relay currently supports CUSTOM edge-bucket ordering only; falling back to CPU-backed swaps");
        return;
    }

    for (std::size_t src = 0; src < devices_.size(); src++) {
        if (!devices_[src].is_cuda()) {
            fail_or_fallback("Stateflow/partition peer relay requires CUDA devices only; falling back to CPU-backed swaps");
            return;
        }
        for (std::size_t dst = 0; dst < devices_.size(); dst++) {
            if (src == dst) {
                continue;
            }
            int can_access = 0;
            cudaError_t status = cudaDeviceCanAccessPeer(&can_access, devices_[src].index(), devices_[dst].index());
            if (status != cudaSuccess || can_access == 0) {
                fail_or_fallback(fmt::format("Stateflow/partition peer relay disabled: CUDA peer access unavailable between device {} and {}",
                                             devices_[src].index(), devices_[dst].index()));
                return;
            }
        }
    }

    for (std::size_t src = 0; src < devices_.size(); src++) {
        c10::cuda::CUDAGuard device_guard(devices_[src]);
        for (std::size_t dst = 0; dst < devices_.size(); dst++) {
            if (src == dst) {
                continue;
            }
            cudaError_t status = cudaDeviceEnablePeerAccess(devices_[dst].index(), 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled) {
                // cudaDeviceEnablePeerAccess may leave a sticky runtime error even
                // though repeated enable attempts are semantically harmless.
                // Clear it here so later kernel launch checks do not trip on this
                // stale status.
                cudaGetLastError();
                continue;
            }
            if (status != cudaSuccess) {
                fail_or_fallback(fmt::format("Stateflow/partition peer relay disabled: failed to enable peer access from device {} to {} ({})",
                                             devices_[src].index(), devices_[dst].index(), cudaGetErrorString(status)));
                return;
            }
        }
    }

    // Ensure no sticky peer-access status leaks into unrelated CUDA work.
    cudaGetLastError();

    peer_relay_source_scratch_tensors_.resize(devices_.size());
    peer_relay_source_scratch_pool_tensors_.resize(devices_.size());
    peer_relay_source_scratch_frames_.resize(devices_.size());
    peer_relay_source_slot_snapshots_.assign(devices_.size(),
                                             std::vector<int64_t>(static_cast<std::size_t>(options_->num_partitions), -1));
    peer_relay_source_scratch_rounds_.assign(devices_.size(),
                                             std::vector<int64_t>(static_cast<std::size_t>(options_->num_partitions), -1));
    peer_relay_source_ready_events_.resize(devices_.size(), 0);
    peer_relay_source_publish_mutexes_.reserve(devices_.size());
    peer_relay_source_publish_cvs_.reserve(devices_.size());
    peer_relay_source_published_rounds_.assign(devices_.size(), -1);
    for (std::size_t src = 0; src < devices_.size(); src++) {
        peer_relay_source_publish_mutexes_.emplace_back(std::make_shared<std::mutex>());
        peer_relay_source_publish_cvs_.emplace_back(std::make_shared<std::condition_variable>());
    }

    for (std::size_t src = 0; src < devices_.size(); src++) {
        c10::cuda::CUDAGuard device_guard(devices_[src]);
        cudaEvent_t ready_event = nullptr;
        cudaError_t status = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            fail_or_fallback(fmt::format("Stateflow/partition peer relay disabled: failed to create source-ready event on device {} ({})",
                                         devices_[src].index(), cudaGetErrorString(status)));
            return;
        }
        peer_relay_source_ready_events_[src] = reinterpret_cast<std::uintptr_t>(ready_event);
    }

    peer_relay_runtime_enabled_ = true;
    SPDLOG_INFO(
        "Enabled peer relay runtime for {} CUDA devices (stateflow_schedule_active={} stateflow_mode={} stateflow_scope={} low_level_requested={}); source scratch will be allocated lazily for outgoing handoffs and CPU backing store will be synchronized at unload/eval boundaries",
        devices_.size(), stateflow_peer_schedule_active_, stateflow_peer_runtime_mode_name(stateflow_mode),
        stateflow_peer_runtime_scope_name(stateflow_scope), low_level_runtime_requested);
#endif
}

bool MemPartitionBufferStorage::peerRelayEnabled_() {
    std::lock_guard<std::mutex> lock(peer_relay_init_lock_);
    if (!peer_relay_init_attempted_) {
        peer_relay_init_attempted_ = true;
        initializePeerRelay_();
    }
    return peer_relay_runtime_enabled_;
}


void MemPartitionBufferStorage::rangePut(int64_t offset, torch::Tensor values) {
    int fd = open(filename_.c_str(), O_RDWR | IO_FLAGS);
    if (fd == -1) {
        SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);
    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }

    close(fd);
}

void MemPartitionBufferStorage::append(torch::Tensor values) {
    ios::openmode flags;

    if (dim0_size_ == 0) {
        flags = ios::trunc | ios::binary;
    } else {
        flags = ios::binary | ios_base::app;
    }

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);

    outfile.close();
}

MemPartitionBufferStorage::~MemPartitionBufferStorage() { 
#if defined(GEGE_CUDA)
    for (auto &ready_event_handle : peer_relay_source_ready_events_) {
        if (ready_event_handle != 0) {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ready_event_handle));
            ready_event_handle = 0;
        }
    }
#endif
    for(int i = 0; i < devices_.size(); i ++) {
        delete buffers_[i];
    }
}

void MemPartitionBufferStorage::ensureHostLoaded_() {
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        void* data_ptr_ = data_.data_ptr();
        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
        loaded_ = true;
    }
}

void MemPartitionBufferStorage::load() {
    // SPDLOG_INFO("MemPartitionBufferStorage Loading {}", filename_);
    ensureHostLoaded_();
    bool log_startup_timing = startup_timing_enabled();

    if (device_ != torch::kCUDA) {
        for (int i = 0; i < buffers_.size(); i ++) {
            if (buffers_[i]->loaded_) {
                buffers_[i]->unload(false);
            }
        }
        return;
    }

    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::load] filename={} buffers={} dim0={} dim1={} host_loaded={}",
                    filename_, buffers_.size(), dim0_size_, dim1_size_, loaded_);
    }

    for (int i = 0; i < buffers_.size(); i ++) {
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::load] begin buffer_idx={} device={}", i, buffers_[i]->device_.str());
        }
        buffers_[i]->load(data_);
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][MemPartitionBufferStorage::load] end buffer_idx={} device={}", i, buffers_[i]->device_.str());
        }
    }
}

void MemPartitionBufferStorage::write() {
    if (loaded_ && !filename_.empty()) { 
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        data = data_.to(torch::kCPU);


        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void MemPartitionBufferStorage::unload(bool perform_write) {
    if (loaded_) {
        for (int i = 0; i < buffers_.size(); i ++)
            buffers_[i]->unload(perform_write);

        if (perform_write) {
            write();
            close(fd_);
            data_ = torch::Tensor();
            loaded_ = false;
        }
    }
}

void MemPartitionBufferStorage::syncToHostWithoutDiskWrite() {
    if (!loaded_) {
        return;
    }

    for (int i = 0; i < buffers_.size(); i++) {
        if (buffers_[i]->loaded_) {
            buffers_[i]->unload(true);
        }
    }
}

void MemPartitionBufferStorage::unload(bool perform_write, int32_t device_idx) {
    if (!loaded_) {
        return;
    }

    if (device_idx < 0 || device_idx >= static_cast<int32_t>(buffers_.size())) {
        throw GegeRuntimeException("MemPartitionBufferStorage::unload received an invalid device index");
    }

    if (perform_write) {
        throw GegeRuntimeException("Per-device MemPartitionBufferStorage unload with write-back is unsupported");
    }

    buffers_[device_idx]->unload(false);
}

void MemPartitionBufferStorage::performNextSwap(int32_t device_idx, std::uintptr_t swap_ready_event) {
    if (!peerRelayEnabled_()) {
        buffers_[device_idx]->performNextSwap(swap_ready_event);
        return;
    }

#if !defined(GEGE_CUDA)
    buffers_[device_idx]->performNextSwap();
#else
    auto *buffer = buffers_[device_idx];
    if (!buffer->buffer_state_.defined() || buffer->buffer_state_iterator_ == buffer->buffer_states_.end()) {
        return;
    }
    if (!stateflow_peer_schedule_active_) {
        buffers_[device_idx]->performNextSwap(swap_ready_event);
        return;
    }

    auto previous_state = buffer->buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto next_state = (*buffer->buffer_state_iterator_).to(torch::kCPU).to(torch::kInt64).contiguous();
    std::vector<int> evict_ids = buffer->getNextEvict();
    std::vector<int> admit_ids = buffer->getNextAdmit();
    if (evict_ids.size() != admit_ids.size()) {
        throw GegeRuntimeException("MemPartitionBufferStorage::performNextSwap expected matched evict/admit counts for Stateflow peer relay");
    }

    bool stateflow_runtime_controlling =
        device_idx >= 0 && static_cast<std::size_t>(device_idx) < stateflow_peer_handoff_index_per_device_.size();
    int64_t transition_round_idx =
        stateflow_runtime_controlling && static_cast<std::size_t>(device_idx) < stateflow_transition_counts_.size()
            ? stateflow_transition_counts_[device_idx] + 1
            : -1;
    const std::vector<uint8_t> *next_required = nullptr;
    if (transition_round_idx > 0 && static_cast<std::size_t>(transition_round_idx) < stateflow_global_next_required_by_round_.size()) {
        next_required = &stateflow_global_next_required_by_round_[static_cast<std::size_t>(transition_round_idx)];
    }
    bool frame_cache_mapping = buffer->frameCacheEnabled_();

    const int64_t partition_count = static_cast<int64_t>(buffer->partition_table_.size());
    std::vector<uint8_t> evict_partition_mask(static_cast<std::size_t>(partition_count), 0);
    for (int evict_id : evict_ids) {
        if (evict_id < 0 || evict_id >= partition_count) {
            throw GegeRuntimeException(fmt::format("MemPartitionBufferStorage::performNextSwap encountered invalid evict partition {}", evict_id));
        }
        evict_partition_mask[static_cast<std::size_t>(evict_id)] = 1;
    }
    std::vector<int64_t> next_slot_by_partition(static_cast<std::size_t>(partition_count), -1);
    auto *next_state_ptr = next_state.data_ptr<int64_t>();
    for (int64_t slot = 0; slot < next_state.numel(); slot++) {
        int64_t partition_id = next_state_ptr[slot];
        if (partition_id < 0 || partition_id >= partition_count) {
            throw GegeRuntimeException(
                fmt::format("MemPartitionBufferStorage::performNextSwap encountered invalid next-state partition {}", partition_id));
        }
        next_slot_by_partition[static_cast<std::size_t>(partition_id)] = slot;
    }

    std::vector<int64_t> evict_slots;
    evict_slots.reserve(evict_ids.size());
    for (int evict_id : evict_ids) {
        Partition *partition = buffer->partition_table_[evict_id];
        if (partition->buffer_idx_ < 0) {
            throw GegeRuntimeException("MemPartitionBufferStorage::performNextSwap encountered an evict partition without a resident buffer slot");
        }
        evict_slots.emplace_back(partition->buffer_idx_);
    }

    auto load_partition_from_host = [&](Partition *partition, const torch::Tensor &cpu_view) {
        if (!buffer->pos_.defined()) {
            cpu_view.copy_(data_.narrow(0, partition->idx_offset_, partition->partition_size_));
            return;
        }
        torch::Tensor host_indices = buffer->pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_);
        cpu_view.copy_(data_.index_select(0, host_indices));
    };

    auto write_partition_to_host = [&](Partition *partition, const torch::Tensor &cpu_view) {
        if (!buffer->pos_.defined()) {
            data_.narrow(0, partition->idx_offset_, partition->partition_size_).copy_(cpu_view);
            return;
        }
        torch::Tensor host_indices = buffer->pos_.slice(0, partition->idx_offset_, partition->idx_offset_ + partition->partition_size_);
        data_.index_put_({host_indices}, cpu_view);
    };

    auto warn_stateflow_fallback = [&](int64_t handoff_key, const std::string &message) {
        if (device_idx < 0 || static_cast<std::size_t>(device_idx) >= stateflow_peer_mismatch_warned_keys_.size()) {
            SPDLOG_WARN("{}", message);
            return;
        }
        if (stateflow_peer_mismatch_warned_keys_[device_idx].emplace(handoff_key).second) {
            SPDLOG_WARN("{}", message);
        }
    };

    std::vector<PeerHandoffDescriptor> outgoing_handoffs;
    if (stateflow_runtime_controlling) {
        for (const auto &handoff_index : stateflow_peer_handoff_index_per_device_) {
            for (const auto &[handoff_key, handoff] : handoff_index) {
                if (handoff.round_idx == transition_round_idx && handoff.src_lane_id == device_idx) {
                    outgoing_handoffs.emplace_back(handoff);
                }
            }
        }
        std::sort(outgoing_handoffs.begin(), outgoing_handoffs.end(),
                  [](const PeerHandoffDescriptor &lhs, const PeerHandoffDescriptor &rhs) {
                      if (lhs.partition_id != rhs.partition_id) {
                          return lhs.partition_id < rhs.partition_id;
                      }
                      if (lhs.dst_lane_id != rhs.dst_lane_id) {
                          return lhs.dst_lane_id < rhs.dst_lane_id;
                      }
                      return lhs.dst_slot_id < rhs.dst_slot_id;
                  });
    }

    struct RelayDebugSlot {
        int partition_id;
        int src_dev;
        int64_t dst_slot;
        int64_t dst_offset;
    };

    auto &published_source_scratch = peer_relay_source_scratch_tensors_[device_idx];
    auto &published_source_scratch_frames = peer_relay_source_scratch_frames_[device_idx];

    {
        c10::cuda::CUDAGuard device_guard(buffer->device_);
        auto comm_stream = c10::cuda::getStreamFromPool(false, buffer->device_.index());
        auto peer_stream = c10::cuda::getStreamFromPool(false, buffer->device_.index());
        c10::cuda::CUDAStreamGuard stream_guard(comm_stream);
        if (swap_ready_event != 0) {
            auto ready_event = reinterpret_cast<cudaEvent_t>(swap_ready_event);
            AT_CUDA_CHECK(cudaStreamWaitEvent(comm_stream.stream(), ready_event, 0));
        }

        auto &source_scratch_pool = peer_relay_source_scratch_pool_tensors_[device_idx];
        auto publish_mutex = peer_relay_source_publish_mutexes_[device_idx];
        auto publish_cv = peer_relay_source_publish_cvs_[device_idx];
        {
            std::lock_guard<std::mutex> publish_lock(*publish_mutex);
            // Scratch tensors are only needed for the currently published round.
            // Retaining one tensor per partition across rounds quickly accumulates
            // tens of gigabytes on 24 GB boards, and any destination that trails
            // far enough behind to need an older round already falls back because
            // scratch_rounds no longer matches the requested transition round.
            published_source_scratch.clear();
            source_scratch_pool.clear();
            if (frame_cache_mapping && !published_source_scratch_frames.empty()) {
                std::lock_guard<std::mutex> frame_lock(buffer->free_physical_frames_lock_);
                for (const auto &[partition_id, frame] : published_source_scratch_frames) {
                    if (frame < buffer->capacity_ || frame >= buffer->physical_frame_capacity_) {
                        continue;
                    }
                    bool frame_is_visible = false;
                    for (int64_t mapped_frame : buffer->logical_to_physical_frames_) {
                        if (mapped_frame == frame) {
                            frame_is_visible = true;
                            break;
                        }
                    }
                    if (!frame_is_visible &&
                        std::find(buffer->free_physical_frames_.begin(), buffer->free_physical_frames_.end(), frame) ==
                            buffer->free_physical_frames_.end()) {
                        buffer->free_physical_frames_.push_back(frame);
                    }
                }
                published_source_scratch_frames.clear();
            }

            auto &slot_snapshot = peer_relay_source_slot_snapshots_[device_idx];
            std::fill(slot_snapshot.begin(), slot_snapshot.end(), -1);
            auto *previous_state_ptr = previous_state.data_ptr<int64_t>();
            for (int64_t slot = 0; slot < previous_state.numel(); slot++) {
                int64_t partition_id = previous_state_ptr[slot];
                if (partition_id >= 0 && partition_id < static_cast<int64_t>(slot_snapshot.size())) {
                    slot_snapshot[static_cast<std::size_t>(partition_id)] = slot;
                }
            }

            auto &scratch_rounds = peer_relay_source_scratch_rounds_[device_idx];
            for (const auto &handoff : outgoing_handoffs) {
                if (handoff.partition_id >= 0 && handoff.partition_id < static_cast<int>(scratch_rounds.size())) {
                    scratch_rounds[static_cast<std::size_t>(handoff.partition_id)] = -1;
                }
            }

            for (const auto &handoff : outgoing_handoffs) {
                if (handoff.partition_id < 0 || handoff.partition_id >= static_cast<int>(buffer->partition_table_.size())) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer handoff references invalid partition {} on source lane {}", handoff.partition_id, device_idx));
                }
                if (handoff.partition_id < static_cast<int>(scratch_rounds.size()) &&
                    scratch_rounds[static_cast<std::size_t>(handoff.partition_id)] == transition_round_idx) {
                    continue;
                }

                Partition *src_partition = buffer->partition_table_[handoff.partition_id];
                if (!src_partition->present_ || src_partition->buffer_idx_ < 0) {
                    continue;
                }
                if (src_partition->buffer_idx_ != handoff.src_slot_id) {
                    continue;
                }

                int64_t rows = src_partition->partition_size_;
                int64_t src_offset = buffer->logicalSlotRowOffset_(src_partition->buffer_idx_);
                torch::Tensor src_view = buffer->buffer_tensor_gpu_view_.slice(0, src_offset, src_offset + rows);
                try {
                    torch::Tensor source_scratch;
                    if (frame_cache_mapping) {
                        int64_t scratch_frame = -1;
                        {
                            std::lock_guard<std::mutex> frame_lock(buffer->free_physical_frames_lock_);
                            if (!buffer->free_physical_frames_.empty()) {
                                scratch_frame = buffer->free_physical_frames_.back();
                                buffer->free_physical_frames_.pop_back();
                            }
                        }
                        if (scratch_frame < 0) {
                            continue;
                        }
                        int64_t scratch_offset = scratch_frame * buffer->partition_size_;
                        source_scratch =
                            buffer->buffer_tensor_gpu_view_.slice(0, scratch_offset, scratch_offset + rows);
                        published_source_scratch_frames[handoff.partition_id] = scratch_frame;
                    } else {
                        auto pool_it = source_scratch_pool.find(handoff.partition_id);
                        if (pool_it != source_scratch_pool.end() && pool_it->second.defined() && pool_it->second.size(0) == rows &&
                            pool_it->second.size(1) == dim1_size_ && pool_it->second.scalar_type() == src_view.scalar_type()) {
                            source_scratch = pool_it->second;
                        } else {
                            source_scratch = torch::empty({rows, dim1_size_}, src_view.options());
                            source_scratch_pool[handoff.partition_id] = source_scratch;
                        }
                    }
                    source_scratch.copy_(src_view, true);
                    published_source_scratch[handoff.partition_id] = source_scratch;
                    scratch_rounds[static_cast<std::size_t>(handoff.partition_id)] = transition_round_idx;
                } catch (const c10::Error &e) {
                    cudaGetLastError();
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay unable to allocate source scratch for partition {} on device {} during swap ({})",
                                    handoff.partition_id, buffer->device_.index(), e.what()));
                }
            }

            if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < peer_relay_source_ready_events_.size() &&
                peer_relay_source_ready_events_[device_idx] != 0) {
                auto ready_event = reinterpret_cast<cudaEvent_t>(peer_relay_source_ready_events_[device_idx]);
                AT_CUDA_CHECK(cudaEventRecord(ready_event, comm_stream.stream()));
            }
            // The destination lane can observe published_rounds_ without a new
            // condition-variable notification if its wait predicate is already
            // true. Keep the round unpublished until the source scratch stream
            // has completed, otherwise the peer copy can wait on a stale event.
            AT_CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));
            peer_relay_source_published_rounds_[device_idx] = transition_round_idx;
        }
        publish_cv->notify_all();

        std::vector<RelayDebugSlot> relay_debug_slots;
        std::vector<int> host_evict_ids;
        int64_t peer_bytes_executed = 0;
        int64_t host_fallback_bytes = 0;
        int64_t peer_copy_count = 0;
        int64_t host_fallback_count = 0;
        int64_t descriptor_mismatch_count = 0;
        int64_t preload_visible_install_rows = 0;
        int64_t hidden_publish_rows = 0;
        int64_t preload_visible_install_parts = 0;
        int64_t hidden_publish_parts = 0;
        int64_t fallback_visible_admit_parts = 0;
        int64_t fallback_visible_admit_rows = 0;
        int64_t free_frames_before_swap = 0;
        int64_t stale_backlog_frames_before_swap = 0;
        int64_t free_frames_after_publish = 0;
        int64_t stale_backlog_frames_after_publish = 0;
        bool submitted_peer_copy = false;
        bool peer_stream_armed = false;
        std::vector<bool> waited_on_source_ready(buffers_.size(), false);
        std::vector<torch::Tensor> peer_copy_lifetime_holds;

        if (frame_cache_mapping) {
            std::lock_guard<std::mutex> frame_lock(buffer->free_physical_frames_lock_);
            free_frames_before_swap = static_cast<int64_t>(buffer->free_physical_frames_.size());
            stale_backlog_frames_before_swap =
                static_cast<int64_t>(buffer->hidden_frame_capacity_) - free_frames_before_swap;
        }

        std::unordered_set<int> preloaded_host_admit_ids;
        if (multi_gpu_async_admit_preload_enabled()) {
            std::vector<int> host_preload_admit_ids;
            std::vector<int64_t> host_preload_slots;
            host_preload_admit_ids.reserve(admit_ids.size());
            host_preload_slots.reserve(evict_slots.size());
            for (std::size_t idx = 0; idx < admit_ids.size(); idx++) {
                int partition_id = admit_ids[idx];
                int64_t handoff_key = transition_round_idx >= 0 ? peer_handoff_lookup_key(transition_round_idx, partition_id) : -1;
                bool has_scheduled_peer_handoff = false;
                if (stateflow_runtime_controlling) {
                    auto handoff_it = stateflow_peer_handoff_index_per_device_[device_idx].find(handoff_key);
                    has_scheduled_peer_handoff = handoff_it != stateflow_peer_handoff_index_per_device_[device_idx].end();
                }
                if (!has_scheduled_peer_handoff) {
                    int64_t target_slot =
                        (partition_id >= 0 && partition_id < static_cast<int>(next_slot_by_partition.size()))
                            ? next_slot_by_partition[static_cast<std::size_t>(partition_id)]
                            : -1;
                    if (target_slot < 0) {
                        target_slot = idx < evict_slots.size() ? evict_slots[idx] : -1;
                    }
                    if (target_slot < 0) {
                        throw GegeRuntimeException(
                            fmt::format("Stateflow peer relay could not determine preload target slot for partition {} on device {}",
                                        partition_id, buffer->device_.index()));
                    }
                    host_preload_admit_ids.push_back(partition_id);
                    host_preload_slots.push_back(target_slot);
                }
            }

            if (!host_preload_admit_ids.empty()) {
                double preload_wait_ms = 0.0;
                if (buffer->consumeAsyncAdmitPreload_(host_preload_admit_ids, host_preload_slots, &preload_wait_ms,
                                                      &preload_visible_install_rows, &hidden_publish_rows,
                                                      &preload_visible_install_parts, &hidden_publish_parts)) {
                    preloaded_host_admit_ids.insert(host_preload_admit_ids.begin(), host_preload_admit_ids.end());
                }
            }
        }

        std::unordered_set<int64_t> hidden_publish_slots;
        hidden_publish_slots.reserve(buffer->pending_hidden_publishes_.size());
        for (const auto &hidden_publish : buffer->pending_hidden_publishes_) {
            hidden_publish_slots.insert(hidden_publish.logical_slot);
        }

        if (frame_cache_mapping) {
            stale_backlog_frames_before_swap =
                std::max<int64_t>(static_cast<int64_t>(buffer->hidden_frame_capacity_) - free_frames_before_swap, 0);
        }

        const bool delayed_stale_enabled =
            frame_cache_mapping &&
            multi_gpu_async_admit_preload_enabled() &&
            parse_env_flag("GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD", false) &&
            parse_env_flag("GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK", false) &&
            !parse_env_flag("GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK", false);
        std::unordered_set<int64_t> delayed_stale_logical_slots;
        std::vector<int> delayed_stale_partition_ids;
        std::vector<int64_t> delayed_stale_row_offsets;
        std::vector<int64_t> delayed_stale_release_frames;
        std::vector<int64_t> delayed_stale_source_offsets;
        struct DelayedStaleWritebackEntry {
            int64_t logical_slot = -1;
            int partition_id = -1;
            int64_t rows = 0;
            int64_t release_frame = -1;
            int64_t source_offset = -1;
        };
        std::vector<DelayedStaleWritebackEntry> delayed_stale_entries;
        int64_t delayed_stale_rows = 0;
        delayed_stale_row_offsets.emplace_back(0);
        auto rebuild_delayed_stale_vectors = [&]() {
            delayed_stale_partition_ids.clear();
            delayed_stale_row_offsets.clear();
            delayed_stale_release_frames.clear();
            delayed_stale_source_offsets.clear();
            delayed_stale_rows = 0;
            delayed_stale_row_offsets.emplace_back(0);
            for (const auto &entry : delayed_stale_entries) {
                delayed_stale_partition_ids.push_back(entry.partition_id);
                delayed_stale_release_frames.push_back(entry.release_frame);
                delayed_stale_source_offsets.push_back(entry.source_offset);
                delayed_stale_rows += entry.rows;
                delayed_stale_row_offsets.push_back(delayed_stale_rows);
            }
        };
        if (delayed_stale_enabled && !hidden_publish_slots.empty()) {
            const int64_t max_stale_backlog = buffer->frameCacheMaxStaleBacklog_();
            const int64_t stale_backlog_before_delay =
                frame_cache_mapping ? std::max<int64_t>(
                                          static_cast<int64_t>(buffer->hidden_frame_capacity_) - free_frames_before_swap, 0)
                                    : 0;
            int64_t remaining_delayed_stale_slots =
                std::max<int64_t>(max_stale_backlog - stale_backlog_before_delay, 0);
            remaining_delayed_stale_slots =
                std::min<int64_t>(remaining_delayed_stale_slots, static_cast<int64_t>(hidden_publish_slots.size()));
            for (std::size_t idx = 0; idx < evict_ids.size(); idx++) {
                int evict_id = evict_ids[idx];
                int64_t evict_slot = evict_slots[idx];
                if (next_required != nullptr && evict_id >= 0 && evict_id < static_cast<int>(next_required->size()) &&
                    (*next_required)[static_cast<std::size_t>(evict_id)] != 0) {
                    continue;
                }
                if (remaining_delayed_stale_slots <= 0) {
                    continue;
                }
                Partition *partition = buffer->partition_table_[evict_id];
                int64_t old_frame = buffer->logicalSlotToPhysicalFrame_(evict_slot);
                delayed_stale_logical_slots.insert(evict_slot);
                delayed_stale_entries.push_back(
                    {evict_slot, evict_id, partition->partition_size_, old_frame, old_frame * buffer->partition_size_});
                remaining_delayed_stale_slots -= 1;
            }
            rebuild_delayed_stale_vectors();
        }

        for (int evict_id : evict_ids) {
            if (next_required != nullptr && evict_id >= 0 && evict_id < static_cast<int>(next_required->size()) &&
                (*next_required)[static_cast<std::size_t>(evict_id)] != 0) {
                continue;
            }

            Partition *partition = buffer->partition_table_[evict_id];
            int64_t src_slot = partition->buffer_idx_;
            if (delayed_stale_logical_slots.find(src_slot) != delayed_stale_logical_slots.end()) {
                continue;
            }
            int64_t buffer_offset = buffer->logicalSlotRowOffset_(src_slot);
            torch::Tensor cpu_view = buffer->buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            torch::Tensor gpu_view = buffer->buffer_tensor_gpu_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
            cpu_view.copy_(gpu_view.detach(), true);
            host_evict_ids.emplace_back(evict_id);
        }

        if (!host_evict_ids.empty()) {
            AT_CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));
            for (int partition_id : host_evict_ids) {
                Partition *partition = buffer->partition_table_[partition_id];
                int64_t src_slot = partition->buffer_idx_;
                int64_t buffer_offset = buffer->logicalSlotRowOffset_(src_slot);
                torch::Tensor cpu_view = buffer->buffer_tensor_view_.slice(0, buffer_offset, buffer_offset + partition->partition_size_);
                write_partition_to_host(partition, cpu_view);
            }
        }

        auto flush_delayed_stale_slot_for_scratch = [&](int64_t logical_slot) {
            auto entry_it = std::find_if(delayed_stale_entries.begin(), delayed_stale_entries.end(),
                                         [&](const DelayedStaleWritebackEntry &entry) {
                                             return entry.logical_slot == logical_slot;
                                         });
            if (entry_it == delayed_stale_entries.end()) {
                return false;
            }

            const DelayedStaleWritebackEntry entry = *entry_it;
            Partition *partition = buffer->partition_table_[entry.partition_id];
            torch::Tensor cpu_view =
                buffer->buffer_tensor_view_.slice(0, entry.source_offset, entry.source_offset + entry.rows);
            torch::Tensor gpu_view =
                buffer->buffer_tensor_gpu_view_.slice(0, entry.source_offset, entry.source_offset + entry.rows);
            cpu_view.copy_(gpu_view.detach(), true);
            AT_CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));
            write_partition_to_host(partition, cpu_view);

            delayed_stale_logical_slots.erase(logical_slot);
            delayed_stale_entries.erase(entry_it);
            rebuild_delayed_stale_vectors();
            return true;
        };

        auto move_partition_between_slots = [&](int partition_id, int64_t src_slot, int64_t dst_slot) {
            if (src_slot == dst_slot) {
                return;
            }
            if (frame_cache_mapping) {
                int64_t src_frame = buffer->logicalSlotToPhysicalFrame_(src_slot);
                int64_t dst_frame = buffer->logicalSlotToPhysicalFrame_(dst_slot);
                buffer->logical_to_physical_frames_[static_cast<std::size_t>(dst_slot)] = src_frame;
                buffer->logical_to_physical_frames_[static_cast<std::size_t>(src_slot)] = dst_frame;
                return;
            }
            Partition *partition = buffer->partition_table_[partition_id];
            int64_t bytes = partition->partition_size_ * dim1_size_ * get_dtype_size_wrapper(dtype_);
            void *dst_ptr =
                static_cast<char *>(buffer->buffer_tensor_gpu_view_.data_ptr()) +
                (dst_slot * buffer->partition_size_ * dim1_size_ * get_dtype_size_wrapper(dtype_));
            void *src_ptr =
                static_cast<char *>(buffer->buffer_tensor_gpu_view_.data_ptr()) +
                (src_slot * buffer->partition_size_ * dim1_size_ * get_dtype_size_wrapper(dtype_));
            cudaError_t status = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, comm_stream.stream());
            if (status != cudaSuccess) {
                throw GegeRuntimeException(
                    fmt::format("Stateflow peer relay local slot move failed for partition {} on device {} (src_slot={} dst_slot={}): {}",
                                partition_id, buffer->device_.index(), src_slot, dst_slot, cudaGetErrorString(status)));
            }
        };

        auto remove_free_slot = [](std::vector<int64_t> &free_slots, int64_t slot) {
            auto it = std::find(free_slots.begin(), free_slots.end(), slot);
            if (it == free_slots.end()) {
                throw GegeRuntimeException(fmt::format("Stateflow peer relay lost track of free slot {}", slot));
            }
            free_slots.erase(it);
        };

        std::vector<int64_t> current_slot_by_partition(static_cast<std::size_t>(partition_count), -1);
        if (frame_cache_mapping) {
            std::vector<int64_t> retained_frame_by_partition(static_cast<std::size_t>(partition_count), -1);
            std::vector<int64_t> reusable_evicted_frames;
            auto *previous_state_ptr = previous_state.data_ptr<int64_t>();
            for (int64_t slot = 0; slot < previous_state.numel(); slot++) {
                int partition_id = static_cast<int>(previous_state_ptr[slot]);
                if (partition_id < 0 || partition_id >= partition_count) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay encountered invalid previous-state partition {} on device {}",
                                    partition_id, buffer->device_.index()));
                }
                int64_t frame = buffer->logicalSlotToPhysicalFrame_(slot);
                if (evict_partition_mask[static_cast<std::size_t>(partition_id)] == 0) {
                    retained_frame_by_partition[static_cast<std::size_t>(partition_id)] = frame;
                } else if (delayed_stale_logical_slots.find(slot) == delayed_stale_logical_slots.end()) {
                    reusable_evicted_frames.push_back(frame);
                }
            }
            std::size_t reusable_frame_cursor = 0;
            std::size_t delayed_stale_frame_cursor = 0;
            auto take_reusable_frame = [&]() -> int64_t {
                if (reusable_frame_cursor < reusable_evicted_frames.size()) {
                    return reusable_evicted_frames[reusable_frame_cursor++];
                }
                while (!delayed_stale_entries.empty()) {
                    int64_t logical_slot = delayed_stale_entries.front().logical_slot;
                    int64_t release_frame = delayed_stale_entries.front().release_frame;
                    if (flush_delayed_stale_slot_for_scratch(logical_slot)) {
                        return release_frame;
                    }
                }
                throw GegeRuntimeException(
                    fmt::format("Stateflow peer relay has no reusable evicted frame for admitted partition on device {}",
                                buffer->device_.index()));
            };
            for (int64_t slot = 0; slot < next_state.numel(); slot++) {
                int partition_id = static_cast<int>(next_state_ptr[slot]);
                int64_t retained_frame = retained_frame_by_partition[static_cast<std::size_t>(partition_id)];
                if (retained_frame >= 0) {
                    buffer->logical_to_physical_frames_[static_cast<std::size_t>(slot)] = retained_frame;
                    current_slot_by_partition[static_cast<std::size_t>(partition_id)] = slot;
                } else {
                    if (hidden_publish_slots.find(slot) != hidden_publish_slots.end() &&
                        delayed_stale_frame_cursor < delayed_stale_entries.size()) {
                        buffer->logical_to_physical_frames_[static_cast<std::size_t>(slot)] =
                            delayed_stale_entries[delayed_stale_frame_cursor++].release_frame;
                        continue;
                    }
                    // Retained partitions may move logical slots. Therefore an admitted logical
                    // slot cannot safely inherit the frame that used to live at the same slot: that
                    // frame may now be owned by a retained partition. Assign each admitted slot a
                    // frame from an actually evicted partition; hidden publishes will replace and
                    // release this frame later, while visible host/peer admits copy into it.
                    buffer->logical_to_physical_frames_[static_cast<std::size_t>(slot)] = take_reusable_frame();
                }
            }
            if (reusable_frame_cursor < reusable_evicted_frames.size()) {
                std::lock_guard<std::mutex> frame_lock(buffer->free_physical_frames_lock_);
                for (std::size_t idx = reusable_frame_cursor; idx < reusable_evicted_frames.size(); idx++) {
                    int64_t frame = reusable_evicted_frames[idx];
                    bool frame_is_visible = false;
                    for (int64_t mapped_frame : buffer->logical_to_physical_frames_) {
                        if (mapped_frame == frame) {
                            frame_is_visible = true;
                            break;
                        }
                    }
                    if (!frame_is_visible &&
                        std::find(buffer->free_physical_frames_.begin(), buffer->free_physical_frames_.end(), frame) ==
                            buffer->free_physical_frames_.end()) {
                        buffer->free_physical_frames_.push_back(frame);
                    }
                }
            }
        } else {
            std::vector<int> slot_to_partition(previous_state.numel(), -1);
            std::vector<int64_t> free_slots;
            std::vector<int> retained_partitions;
            retained_partitions.reserve(previous_state.numel());
            auto *previous_state_ptr = previous_state.data_ptr<int64_t>();
            for (int64_t idx = 0; idx < previous_state.numel(); idx++) {
                int partition_id = static_cast<int>(previous_state_ptr[idx]);
                if (partition_id < 0 || partition_id >= partition_count) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay encountered invalid previous-state partition {} on device {}",
                                    partition_id, buffer->device_.index()));
                }
                Partition *partition = buffer->partition_table_[partition_id];
                int64_t current_slot = partition->buffer_idx_;
                if (current_slot < 0 || current_slot >= previous_state.numel()) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay encountered invalid current slot {} for partition {} on device {}",
                                    current_slot, partition_id, buffer->device_.index()));
                }
                if (evict_partition_mask[static_cast<std::size_t>(partition_id)] != 0) {
                    free_slots.emplace_back(current_slot);
                } else {
                    slot_to_partition[current_slot] = partition_id;
                    current_slot_by_partition[static_cast<std::size_t>(partition_id)] = current_slot;
                    retained_partitions.push_back(partition_id);
                }
            }

            std::vector<int> pending_moves;
            pending_moves.reserve(retained_partitions.size());
            for (int partition_id : retained_partitions) {
                int64_t current_slot = current_slot_by_partition[static_cast<std::size_t>(partition_id)];
                int64_t target_slot = next_slot_by_partition[static_cast<std::size_t>(partition_id)];
                if (target_slot < 0) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay lost retained partition {} from next-state layout on device {}",
                                    partition_id, buffer->device_.index()));
                }
                if (target_slot != current_slot) {
                    pending_moves.push_back(partition_id);
                }
            }

            while (!pending_moves.empty()) {
                bool progressed = false;
                for (auto it = pending_moves.begin(); it != pending_moves.end();) {
                    int partition_id = *it;
                    int64_t current_slot = current_slot_by_partition[static_cast<std::size_t>(partition_id)];
                    int64_t target_slot = next_slot_by_partition[static_cast<std::size_t>(partition_id)];
                    if (std::find(free_slots.begin(), free_slots.end(), target_slot) != free_slots.end()) {
                        move_partition_between_slots(partition_id, current_slot, target_slot);
                        slot_to_partition[target_slot] = partition_id;
                        remove_free_slot(free_slots, target_slot);
                        slot_to_partition[current_slot] = -1;
                        free_slots.emplace_back(current_slot);
                        current_slot_by_partition[static_cast<std::size_t>(partition_id)] = target_slot;
                        it = pending_moves.erase(it);
                        progressed = true;
                    } else {
                        ++it;
                    }
                }

                if (pending_moves.empty()) {
                    break;
                }

                if (!progressed) {
                    if (free_slots.empty()) {
                        int64_t recovered_slot = -1;
                        for (int partition_id : pending_moves) {
                            int64_t target_slot = next_slot_by_partition[static_cast<std::size_t>(partition_id)];
                            if (delayed_stale_logical_slots.find(target_slot) != delayed_stale_logical_slots.end()) {
                                recovered_slot = target_slot;
                                break;
                            }
                        }
                        if (recovered_slot < 0 && !delayed_stale_entries.empty()) {
                            recovered_slot = delayed_stale_entries.front().logical_slot;
                        }
                        if (recovered_slot >= 0 && flush_delayed_stale_slot_for_scratch(recovered_slot)) {
                            free_slots.emplace_back(recovered_slot);
                            continue;
                        }
                        throw GegeRuntimeException(
                            fmt::format("Stateflow peer relay has no free slot available to realize retained-slot permutation on device {}",
                                        buffer->device_.index()));
                    }
                    int partition_id = *pending_moves.begin();
                    int64_t current_slot = current_slot_by_partition[static_cast<std::size_t>(partition_id)];
                    int64_t scratch_slot = free_slots.back();
                    free_slots.pop_back();
                    move_partition_between_slots(partition_id, current_slot, scratch_slot);
                    slot_to_partition[scratch_slot] = partition_id;
                    slot_to_partition[current_slot] = -1;
                    free_slots.emplace_back(current_slot);
                    current_slot_by_partition[static_cast<std::size_t>(partition_id)] = scratch_slot;
                }
            }
        }

        struct ScopedCudaEvent {
            cudaEvent_t handle = nullptr;

            ~ScopedCudaEvent() {
                if (handle != nullptr) {
                    cudaEventDestroy(handle);
                }
            }
        } layout_ready_event;
        AT_CUDA_CHECK(cudaEventCreateWithFlags(&layout_ready_event.handle, cudaEventDisableTiming));
        AT_CUDA_CHECK(cudaEventRecord(layout_ready_event.handle, comm_stream.stream()));

        for (int64_t slot = 0; slot < next_state.numel(); slot++) {
            int partition_id = static_cast<int>(next_state_ptr[slot]);
            int64_t retained_slot =
                (partition_id >= 0 && partition_id < partition_count)
                    ? current_slot_by_partition[static_cast<std::size_t>(partition_id)]
                    : -1;
            if (retained_slot >= 0) {
                if (retained_slot != slot) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay failed to place retained partition {} into slot {} on device {} (actual slot={})",
                                    partition_id, slot, buffer->device_.index(), retained_slot));
                }
                continue;
            }
            if (preloaded_host_admit_ids.find(partition_id) != preloaded_host_admit_ids.end()) {
                continue;
            }

            Partition *dst_partition = buffer->partition_table_[partition_id];
            int64_t rows = dst_partition->partition_size_;
            int64_t bytes = rows * dim1_size_ * get_dtype_size_wrapper(dtype_);
            int64_t dst_slot = slot;
            int64_t dst_offset = buffer->logicalSlotRowOffset_(dst_slot);
            int64_t handoff_key = transition_round_idx >= 0 ? peer_handoff_lookup_key(transition_round_idx, partition_id) : -1;

            const PeerHandoffDescriptor *scheduled_handoff = nullptr;
            if (stateflow_runtime_controlling) {
                auto handoff_it = stateflow_peer_handoff_index_per_device_[device_idx].find(handoff_key);
                if (handoff_it != stateflow_peer_handoff_index_per_device_[device_idx].end()) {
                    scheduled_handoff = &handoff_it->second;
                }
            }

            bool used_peer_copy = false;
            if (scheduled_handoff != nullptr) {
                if (scheduled_handoff->dst_lane_id != device_idx || scheduled_handoff->dst_slot_id != dst_slot) {
                    descriptor_mismatch_count += 1;
                    warn_stateflow_fallback(
                        handoff_key,
                        fmt::format(
                            "Stateflow peer relay falling back to host for dst_lane={} round={} partition={} because the projected destination slot does not match the runtime evict slot (expected dst_lane={} dst_slot={}, actual dst_lane={} dst_slot={})",
                            device_idx, transition_round_idx, partition_id, scheduled_handoff->dst_lane_id,
                            scheduled_handoff->dst_slot_id, device_idx, dst_slot));
                } else if (scheduled_handoff->src_lane_id == device_idx) {
                    throw GegeRuntimeException(
                        fmt::format("Stateflow peer relay encountered an unexpected local source for dst_lane={} round={} partition={}",
                                    device_idx, transition_round_idx, partition_id));
                } else if (scheduled_handoff->src_lane_id < 0 ||
                           scheduled_handoff->src_lane_id >= static_cast<int>(buffers_.size())) {
                    descriptor_mismatch_count += 1;
                    warn_stateflow_fallback(
                        handoff_key,
                        fmt::format(
                            "Stateflow peer relay falling back to host for dst_lane={} round={} partition={} because the projected source lane={} is invalid",
                            device_idx, transition_round_idx, partition_id, scheduled_handoff->src_lane_id));
                } else {
                    auto *src_buffer = buffers_[scheduled_handoff->src_lane_id];
                    torch::Tensor src_scratch;
                    int64_t published_src_slot = -1;
                    int64_t published_scratch_round = -1;
                    auto publish_mutex = peer_relay_source_publish_mutexes_[scheduled_handoff->src_lane_id];
                    auto publish_cv = peer_relay_source_publish_cvs_[scheduled_handoff->src_lane_id];
                    {
                        std::unique_lock<std::mutex> publish_lock(*publish_mutex);
                        publish_cv->wait(publish_lock, [&] {
                            return peer_relay_source_published_rounds_[scheduled_handoff->src_lane_id] >= transition_round_idx;
                        });

                        const auto &slot_snapshot = peer_relay_source_slot_snapshots_[scheduled_handoff->src_lane_id];
                        if (partition_id >= 0 && partition_id < static_cast<int>(slot_snapshot.size())) {
                            published_src_slot = slot_snapshot[static_cast<std::size_t>(partition_id)];
                        }

                        const auto &scratch_rounds = peer_relay_source_scratch_rounds_[scheduled_handoff->src_lane_id];
                        if (partition_id >= 0 && partition_id < static_cast<int>(scratch_rounds.size())) {
                            published_scratch_round = scratch_rounds[static_cast<std::size_t>(partition_id)];
                        }

                        auto &source_scratch_map = peer_relay_source_scratch_tensors_[scheduled_handoff->src_lane_id];
                        auto scratch_it = source_scratch_map.find(partition_id);
                        if (scratch_it != source_scratch_map.end() && scratch_it->second.defined()) {
                            src_scratch = scratch_it->second;
                        }
                    }

                    if (published_src_slot < 0) {
                        descriptor_mismatch_count += 1;
                        warn_stateflow_fallback(
                            handoff_key,
                            fmt::format(
                                "Stateflow peer relay falling back to host for dst_lane={} round={} partition={} because the projected source lane={} is not resident at runtime",
                                device_idx, transition_round_idx, partition_id, scheduled_handoff->src_lane_id));
                    } else if (published_src_slot != scheduled_handoff->src_slot_id) {
                        descriptor_mismatch_count += 1;
                        warn_stateflow_fallback(
                            handoff_key,
                            fmt::format(
                                "Stateflow peer relay falling back to host for dst_lane={} round={} partition={} because the projected source slot does not match the runtime slot (expected src_lane={} src_slot={}, actual src_lane={} src_slot={})",
                                device_idx, transition_round_idx, partition_id, scheduled_handoff->src_lane_id,
                                scheduled_handoff->src_slot_id, scheduled_handoff->src_lane_id, published_src_slot));
                    } else if (published_scratch_round != transition_round_idx || !src_scratch.defined()) {
                        descriptor_mismatch_count += 1;
                        warn_stateflow_fallback(
                            handoff_key,
                            fmt::format(
                                "Stateflow peer relay falling back to host for dst_lane={} round={} partition={} because no source scratch was prepared for src_lane={} src_slot={}",
                                device_idx, transition_round_idx, partition_id, scheduled_handoff->src_lane_id,
                                scheduled_handoff->src_slot_id));
                    } else {
                        void *dst_ptr =
                            static_cast<char *>(buffer->buffer_tensor_gpu_view_.data_ptr()) +
                            (dst_offset * dim1_size_ * get_dtype_size_wrapper(dtype_));
                        void *src_ptr = src_scratch.data_ptr();
                        if (!peer_stream_armed) {
                            AT_CUDA_CHECK(cudaStreamWaitEvent(peer_stream.stream(), layout_ready_event.handle, 0));
                            peer_stream_armed = true;
                        }
                        if (scheduled_handoff->src_lane_id >= 0 &&
                            static_cast<std::size_t>(scheduled_handoff->src_lane_id) < peer_relay_source_ready_events_.size() &&
                            !waited_on_source_ready[scheduled_handoff->src_lane_id]) {
                            auto source_ready_event =
                                reinterpret_cast<cudaEvent_t>(peer_relay_source_ready_events_[scheduled_handoff->src_lane_id]);
                            if (source_ready_event != nullptr) {
                                AT_CUDA_CHECK(cudaStreamWaitEvent(peer_stream.stream(), source_ready_event, 0));
                            }
                            waited_on_source_ready[scheduled_handoff->src_lane_id] = true;
                        }
                        cudaError_t status = cudaMemcpyPeerAsync(
                            dst_ptr, buffer->device_.index(), src_ptr, src_buffer->device_.index(), bytes, peer_stream.stream());
                        if (status != cudaSuccess) {
                            throw GegeRuntimeException(
                                fmt::format("Peer relay copy failed for partition {} from device {} to {}: {}",
                                            partition_id, src_buffer->device_.index(), buffer->device_.index(),
                                            cudaGetErrorString(status)));
                        }
                        // The source lane may advance and clear its published scratch map before
                        // this destination stream is synchronized. Hold an owning Tensor reference
                        // locally so the async peer copy cannot observe freed source storage.
                        peer_copy_lifetime_holds.emplace_back(src_scratch);
                        used_peer_copy = true;
                        submitted_peer_copy = true;
                        peer_bytes_executed += bytes;
                        peer_copy_count += 1;
                        if (eval_finite_debug_enabled()) {
                            relay_debug_slots.push_back(
                                {partition_id, static_cast<int>(scheduled_handoff->src_lane_id), dst_slot, dst_offset});
                        }
                    }
                }
            }

            if (!used_peer_copy) {
                torch::Tensor cpu_view = buffer->buffer_tensor_view_.slice(0, dst_offset, dst_offset + rows);
                load_partition_from_host(dst_partition, cpu_view);
                torch::Tensor gpu_view = buffer->buffer_tensor_gpu_view_.slice(0, dst_offset, dst_offset + rows);
                gpu_view.copy_(cpu_view, true);
                fallback_visible_admit_parts += 1;
                fallback_visible_admit_rows += rows;
                if (scheduled_handoff != nullptr) {
                    host_fallback_bytes += bytes;
                    host_fallback_count += 1;
                }
            }
        }

        AT_CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));
        if (submitted_peer_copy && device_idx >= 0 && static_cast<std::size_t>(device_idx) < device_peer_sync_wait_ns_.size()) {
            auto peer_sync_start = std::chrono::high_resolution_clock::now();
            AT_CUDA_CHECK(cudaStreamSynchronize(peer_stream.stream()));
            device_peer_sync_wait_ns_[device_idx] +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - peer_sync_start).count();
        }
        AT_CUDA_CHECK(cudaEventDestroy(layout_ready_event.handle));
        layout_ready_event.handle = nullptr;
        bool published_hidden_frames = false;
        if (!buffer->pending_hidden_publishes_.empty()) {
            std::unordered_set<int64_t> delayed_release_frames(
                delayed_stale_release_frames.begin(), delayed_stale_release_frames.end());
            std::lock_guard<std::mutex> frame_lock(buffer->free_physical_frames_lock_);
            for (const auto &hidden_publish : buffer->pending_hidden_publishes_) {
                auto free_it = std::find(buffer->free_physical_frames_.begin(), buffer->free_physical_frames_.end(), hidden_publish.frame);
                if (free_it != buffer->free_physical_frames_.end()) {
                    buffer->free_physical_frames_.erase(free_it);
                }
            }
            for (const auto &hidden_publish : buffer->pending_hidden_publishes_) {
                int64_t published_old_frame =
                    buffer->logical_to_physical_frames_[static_cast<std::size_t>(hidden_publish.logical_slot)];
                buffer->logical_to_physical_frames_[static_cast<std::size_t>(hidden_publish.logical_slot)] = hidden_publish.frame;
                const bool delay_old_frame_release =
                    delayed_release_frames.find(published_old_frame) != delayed_release_frames.end();
                if (!delay_old_frame_release && published_old_frame != hidden_publish.frame &&
                    std::find(buffer->free_physical_frames_.begin(), buffer->free_physical_frames_.end(), published_old_frame) ==
                        buffer->free_physical_frames_.end()) {
                    buffer->free_physical_frames_.push_back(published_old_frame);
                }
            }
            published_hidden_frames = true;
            buffer->pending_hidden_publishes_.clear();
        }
        if (frame_cache_mapping || published_hidden_frames) {
            buffer->refreshFrameCacheTensors_();
        }
        if (!delayed_stale_partition_ids.empty()) {
            buffer->startAsyncEvictWriteback_(delayed_stale_partition_ids, delayed_stale_row_offsets, torch::Tensor(), torch::Tensor(),
                                              delayed_stale_release_frames, delayed_stale_source_offsets);
        }
        if (frame_cache_mapping) {
            std::lock_guard<std::mutex> frame_lock(buffer->free_physical_frames_lock_);
            free_frames_after_publish = static_cast<int64_t>(buffer->free_physical_frames_.size());
            stale_backlog_frames_after_publish =
                static_cast<int64_t>(buffer->hidden_frame_capacity_) - free_frames_after_publish;
        }
        if (frame_cache_mapping || multi_gpu_async_admit_preload_enabled()) {
            buffer->frame_cache_perf_stats_.swap_samples += 1;
            buffer->frame_cache_perf_stats_.visible_install_parts += preload_visible_install_parts;
            buffer->frame_cache_perf_stats_.visible_install_rows += preload_visible_install_rows;
            buffer->frame_cache_perf_stats_.hidden_publish_parts += hidden_publish_parts;
            buffer->frame_cache_perf_stats_.hidden_publish_rows += hidden_publish_rows;
            buffer->frame_cache_perf_stats_.fallback_visible_admit_parts += fallback_visible_admit_parts;
            buffer->frame_cache_perf_stats_.fallback_visible_admit_rows += fallback_visible_admit_rows;
            buffer->frame_cache_perf_stats_.partial_preload_swap_count +=
                (fallback_visible_admit_parts > 0 && hidden_publish_parts > 0) ? 1 : 0;
            buffer->frame_cache_perf_stats_.delayed_stale_writeback_swap_count +=
                delayed_stale_partition_ids.empty() ? 0 : 1;
            buffer->frame_cache_perf_stats_.async_admit_valid_before_swap_count +=
                preloaded_host_admit_ids.empty() ? 0 : 1;
            buffer->frame_cache_perf_stats_.reserved_preload_frames_sum +=
                std::max<int64_t>(static_cast<int64_t>(buffer->hidden_frame_capacity_), 0);
            buffer->frame_cache_perf_stats_.free_frames_before_swap_sum += std::max<int64_t>(free_frames_before_swap, 0);
            buffer->frame_cache_perf_stats_.free_frames_after_publish_sum += std::max<int64_t>(free_frames_after_publish, 0);
            buffer->frame_cache_perf_stats_.stale_backlog_before_swap_max =
                std::max(buffer->frame_cache_perf_stats_.stale_backlog_before_swap_max,
                         std::max<int64_t>(stale_backlog_frames_before_swap, 0));
            buffer->frame_cache_perf_stats_.stale_backlog_after_publish_max =
                std::max(buffer->frame_cache_perf_stats_.stale_backlog_after_publish_max,
                         std::max<int64_t>(stale_backlog_frames_after_publish, 0));
        }
        if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < device_peer_bytes_executed_.size()) {
            device_peer_bytes_executed_[device_idx] += peer_bytes_executed;
            device_host_fallback_bytes_[device_idx] += host_fallback_bytes;
            device_peer_copy_count_[device_idx] += peer_copy_count;
            device_host_fallback_count_[device_idx] += host_fallback_count;
            device_descriptor_mismatch_count_[device_idx] += descriptor_mismatch_count;
        }

        if (eval_finite_debug_enabled()) {
            for (const auto &debug_slot : relay_debug_slots) {
                auto &source_scratch_map = peer_relay_source_scratch_tensors_[debug_slot.src_dev];
                auto scratch_it = source_scratch_map.find(debug_slot.partition_id);
                if (scratch_it == source_scratch_map.end() || !scratch_it->second.defined()) {
                    continue;
                }
                Partition *dst_partition = buffer->partition_table_[debug_slot.partition_id];
                int64_t rows = dst_partition->partition_size_;
                int64_t dst_offset = debug_slot.dst_offset;
                torch::Tensor dst_view = buffer->buffer_tensor_gpu_view_.slice(0, dst_offset, dst_offset + rows);
                log_non_finite_rows_if_any("src_scratch", debug_slot.partition_id, buffers_[debug_slot.src_dev]->device_.index(),
                                           buffer->device_.index(), scratch_it->second);
                log_non_finite_rows_if_any("dst", debug_slot.partition_id, buffers_[debug_slot.src_dev]->device_.index(),
                                           buffer->device_.index(), dst_view);
            }
        }
    }

    auto *previous_state_ptr = previous_state.data_ptr<int64_t>();
    for (int64_t i = 0; i < previous_state.numel(); i++) {
        int evict_id = static_cast<int>(previous_state_ptr[i]);
        Partition *partition = buffer->partition_table_[evict_id];
        partition->present_ = false;
        partition->buffer_idx_ = -1;
        partition->data_ptr_ = nullptr;
        partition->physical_frame_idx_ = -1;
    }

    buffer->buffer_state_ = *buffer->buffer_state_iterator_;
    for (int i = 0; i < buffer->buffer_sizes_; i++) {
        if (buffer->buffer_state_iterator_ != buffer->buffer_states_.end()) {
            buffer->buffer_state_iterator_++;
        }
    }

    int64_t num_rows = 0;
    auto current_state = buffer->buffer_state_.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *current_state_ptr = current_state.data_ptr<int64_t>();
    for (int64_t i = 0; i < current_state.numel(); i++) {
        int partition_id = static_cast<int>(current_state_ptr[i]);
        Partition *partition = buffer->partition_table_[partition_id];
        partition->present_ = true;
        partition->buffer_idx_ = static_cast<int>(i);
        partition->data_ptr_ = nullptr;
        partition->physical_frame_idx_ = static_cast<int>(buffer->logicalSlotToPhysicalFrame_(i));
        num_rows += partition->partition_size_;
    }

    buffer->size_.store(num_rows);
    buffer->loaded_ = true;
    if (stateflow_runtime_controlling && device_idx >= 0 && static_cast<std::size_t>(device_idx) < stateflow_transition_counts_.size()) {
        stateflow_transition_counts_[device_idx] = transition_round_idx;
    }
#endif
}

void MemPartitionBufferStorage::startAsyncAdmitPreload(int32_t device_idx) {
    if (device_idx < 0 || device_idx >= static_cast<int32_t>(buffers_.size())) {
        throw GegeRuntimeException("MemPartitionBufferStorage::startAsyncAdmitPreload received an invalid device index");
    }
    if (!peerRelayEnabled_()) {
        buffers_[device_idx]->startAsyncAdmitPreload();
        return;
    }
    if (!stateflow_peer_schedule_active_ || !multi_gpu_async_admit_preload_enabled()) {
        return;
    }

    auto *buffer = buffers_[device_idx];
    if (!buffer->buffer_state_.defined() || buffer->buffer_state_iterator_ == buffer->buffer_states_.end()) {
        return;
    }

    std::vector<int> evict_ids = buffer->getNextEvict();
    std::vector<int> admit_ids = buffer->getNextAdmit();
    if (evict_ids.size() != admit_ids.size()) {
        throw GegeRuntimeException("MemPartitionBufferStorage::startAsyncAdmitPreload expected matched evict/admit counts");
    }

    int64_t transition_round_idx =
        static_cast<std::size_t>(device_idx) < stateflow_transition_counts_.size() ? stateflow_transition_counts_[device_idx] + 1 : -1;

    auto next_state = (*buffer->buffer_state_iterator_).to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t partition_count = static_cast<int64_t>(buffer->partition_table_.size());
    std::vector<int64_t> next_slot_by_partition(static_cast<std::size_t>(partition_count), -1);
    auto *next_state_ptr = next_state.data_ptr<int64_t>();
    for (int64_t slot = 0; slot < next_state.numel(); slot++) {
        int64_t partition_id = next_state_ptr[slot];
        if (partition_id < 0 || partition_id >= partition_count) {
            throw GegeRuntimeException(
                fmt::format("MemPartitionBufferStorage::startAsyncAdmitPreload encountered invalid next-state partition {}", partition_id));
        }
        next_slot_by_partition[static_cast<std::size_t>(partition_id)] = slot;
    }

    std::vector<int64_t> evict_slots;
    evict_slots.reserve(evict_ids.size());
    for (int evict_id : evict_ids) {
        Partition *partition = buffer->partition_table_[evict_id];
        if (partition->buffer_idx_ < 0) {
            return;
        }
        evict_slots.emplace_back(partition->buffer_idx_);
    }

    std::vector<int> host_preload_admit_ids;
    std::vector<int64_t> host_preload_slots;
    host_preload_admit_ids.reserve(admit_ids.size());
    host_preload_slots.reserve(evict_slots.size());
    for (std::size_t idx = 0; idx < admit_ids.size(); idx++) {
        int partition_id = admit_ids[idx];
        int64_t handoff_key = transition_round_idx >= 0 ? peer_handoff_lookup_key(transition_round_idx, partition_id) : -1;
        bool has_scheduled_peer_handoff = false;
        if (static_cast<std::size_t>(device_idx) < stateflow_peer_handoff_index_per_device_.size()) {
            auto handoff_it = stateflow_peer_handoff_index_per_device_[device_idx].find(handoff_key);
            has_scheduled_peer_handoff = handoff_it != stateflow_peer_handoff_index_per_device_[device_idx].end();
        }
        if (!has_scheduled_peer_handoff) {
            int64_t target_slot =
                (partition_id >= 0 && partition_id < static_cast<int>(next_slot_by_partition.size()))
                    ? next_slot_by_partition[static_cast<std::size_t>(partition_id)]
                    : -1;
            if (target_slot < 0) {
                target_slot = idx < evict_slots.size() ? evict_slots[idx] : -1;
            }
            if (target_slot < 0) {
                throw GegeRuntimeException(
                    fmt::format("Stateflow peer relay could not determine preload target slot for partition {} on device {}",
                                partition_id, buffer->device_.index()));
            }
            host_preload_admit_ids.push_back(partition_id);
            host_preload_slots.push_back(target_slot);
        }
    }

    std::unordered_set<int> outgoing_peer_scratch_partitions;
    if (transition_round_idx >= 0) {
        for (const auto &handoff_index : stateflow_peer_handoff_index_per_device_) {
            for (const auto &[handoff_key, handoff] : handoff_index) {
                if (handoff.round_idx == transition_round_idx && handoff.src_lane_id == device_idx) {
                    outgoing_peer_scratch_partitions.insert(handoff.partition_id);
                }
            }
        }
    }

    buffer->startAsyncAdmitPreloadForPlan_(
        host_preload_admit_ids, host_preload_slots,
        static_cast<int64_t>(outgoing_peer_scratch_partitions.size()));
}

torch::Tensor MemPartitionBufferStorage::indexRead(Indices indices) { 
    if(device_ == torch::kCUDA) {
        return buffers_[0]->indexRead(indices);
    } else { 
        if (indices.sizes().size() != 1) {
            // TODO: throw invalid input to func exception
            throw std::runtime_error("");
        }

        if (data_.defined()) {
            return data_.index_select(0, indices.to(devices_[0]));
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor MemPartitionBufferStorage::indexRead(Indices indices, int32_t device_idx) { 
    if (device_idx >= 0 && device_idx < static_cast<int32_t>(buffers_.size()) && buffers_[device_idx]->hasDeviceResidentFrames()) {
        return buffers_[device_idx]->indexRead(indices);
    } else { 
        if (indices.sizes().size() != 1) {
            // TODO: throw invalid input to func exception
            throw std::runtime_error("");
        }
        // std::cout << data_.device() << std::endl;
        if (data_.defined()) {
            return data_.index_select(0, indices);
        } else {
            return torch::Tensor();
        }
    }
}

bool MemPartitionBufferStorage::hasDeviceResidentFrames(int32_t device_idx) const {
    if (device_idx < 0 || device_idx >= static_cast<int32_t>(buffers_.size()) || buffers_[device_idx] == nullptr) {
        return false;
    }
    return buffers_[device_idx]->hasDeviceResidentFrames();
}

void MemPartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values) { 
    return buffers_[0]->indexAdd(indices, values); 
}

void MemPartitionBufferStorage::indexAdd(Indices indices, torch::Tensor values, int32_t device_idx) { 
    return buffers_[device_idx]->indexAdd(indices, values); 
}

void MemPartitionBufferStorage::indexAddMasked(Indices indices, torch::Tensor values, torch::Tensor active_mask, int32_t device_idx) {
    return buffers_[device_idx]->indexAddMasked(indices, values, active_mask);
}

void MemPartitionBufferStorage::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

torch::Tensor MemPartitionBufferStorage::range(int64_t offset, int64_t n) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

void MemPartitionBufferStorage::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
}

void MemPartitionBufferStorage::shuffle() {
    SPDLOG_ERROR("Unsupported operation for MemPartitionBufferStorage");
    throw std::runtime_error("");
};

void MemPartitionBufferStorage::sort(bool src) {
    SPDLOG_ERROR("Sort not supported for MemPartitionBufferStorage");
    throw std::runtime_error("");
};

FlatFile::FlatFile(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, bool alloc) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = torch::kCPU;

    if (alloc) {
        int64_t dtype_size = 0;

        if (dtype_ == torch::kFloat64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kFloat32) {
            dtype_size = 4;
        } else if (dtype_ == torch::kFloat16) {
            dtype_size = 2;
        } else if (dtype_ == torch::kInt64) {
            dtype_size = 8;
        } else if (dtype_ == torch::kInt32) {
            dtype_size = 4;
        }

        std::ofstream ofs(filename_, std::ios::binary | std::ios::out);
        ofs.seekp(dim0_size_ * dim1_size_ * dtype_size - 1);
        ofs.write("", 1);
        ofs.close();
    }
}

FlatFile::FlatFile(string filename, torch::Tensor data) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    loaded_ = false;
    append(data);
    initialized_ = true;
    device_ = torch::kCPU;
}

FlatFile::FlatFile(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    initialized_ = false;
    loaded_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
}

void FlatFile::rangePut(int64_t offset, torch::Tensor values) {
    if (!values.defined() || (dim0_size_ != 0 && (values.size(0) + offset > dim0_size_ || values.size(1) != dim1_size_))) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), values.size(0) * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::append(torch::Tensor values) {
    ios::openmode flags = dim0_size_ == 0 ? ios::trunc | ios::binary : ios::binary | ios_base::app;

    dim0_size_ += values.size(0);
    dim1_size_ = values.size(1);
    dtype_ = values.scalar_type();

    std::ofstream outfile(filename_, flags);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)values.data_ptr(), values.size(0) * values.size(1) * dtype_size);
    outfile.close();
}

void FlatFile::load() {
    if (!loaded_ && initialized_) {
        fd_ = open(filename_.c_str(), O_RDWR | IO_FLAGS);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        loaded_ = true;
    }
}

void FlatFile::write() { return; }

void FlatFile::unload(bool perform_write) {
    (void)perform_write;
    if (loaded_) {
        close(fd_);
        loaded_ = false;
    }
}

torch::Tensor FlatFile::indexRead(Indices indices) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexAdd(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::indexPut(Indices indices, torch::Tensor values) {
    SPDLOG_ERROR("Unsupported operation for FlatFile, only sequential access is supported");
    throw std::runtime_error("");
}

void FlatFile::move(string new_filename) {
    unload(false);

    renameFile(filename_, new_filename);

    load();
}

void FlatFile::copy(string new_filename, bool rename) {
    unload(false);

    copyFile(filename_, new_filename);

    if (rename) {
        filename_ = new_filename;
    }
    load();
}

torch::Tensor FlatFile::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    torch::Tensor output_tensor = torch::empty({n, dim1_size_}, dtype_);
    if (pread_wrapper(fd_, output_tensor.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
    return output_tensor;
}

void FlatFile::rangePut(int64_t offset, int64_t n, torch::Tensor values) {
    int dtype_size = get_dtype_size_wrapper(dtype_);

    int64_t ptr_offset = offset * dim1_size_ * dtype_size;

    if (pwrite_wrapper(fd_, values.data_ptr(), n * dim1_size_ * dtype_size, ptr_offset) == -1) {
        SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
        throw std::runtime_error("");
    }
}

void FlatFile::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SHUFFLE_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SHUFFLE_SIZE;
            }
            torch::Tensor chunk = range(offset, curr_size);
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::randperm(chunk.size(0), opts)));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::sort(bool src) {
    // function for sorting flat file storing edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();
    }
    if (edge_bucket_sizes_.empty()) {
        int64_t offset = 0;
        int64_t curr_size = 0;
        while (offset < dim0_size_) {
            if (dim0_size_ - offset < MAX_SORT_SIZE) {
                curr_size = dim0_size_ - offset;
            } else {
                curr_size = MAX_SORT_SIZE;
            }

            torch::Tensor chunk = range(offset, curr_size);
            // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            chunk.copy_(chunk.index_select(0, torch::argsort(chunk.select(1, sort_dim))));
            rangePut(offset, chunk);
            offset += curr_size;
        }
    } else {
        int64_t offset = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = range(offset, *itr);
            edge_bucket.copy_(edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            rangePut(offset, edge_bucket);
            offset += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}

void FlatFile::mem_load() {
    if (!loaded_) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_ERROR("Unable to open {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);
        SPDLOG_DEBUG("Initialized memory edges");
        process_mem_usage();

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        SPDLOG_DEBUG("Read edges from disk");
        process_mem_usage();

        loaded_ = true;
    }
}

void FlatFile::mem_unload(bool write) {
    if (loaded_) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (write) {
            if (pwrite_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
                SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
                throw std::runtime_error("");
            }
        }

        close(fd_);

        SPDLOG_DEBUG("Edges written");
        process_mem_usage();
        loaded_ = false;
        process_mem_usage();
        data_ = torch::Tensor();
        SPDLOG_DEBUG("Nulled tensor and pointer");
        process_mem_usage();
    }
}

InMemory::InMemory(string filename, int64_t dim0_size, int64_t dim1_size, torch::Dtype dtype, torch::Device device) {
    filename_ = filename;
    dim0_size_ = dim0_size;
    dim1_size_ = dim1_size;
    dtype_ = dtype;
    initialized_ = true;
    loaded_ = false;
    device_ = device;
}

InMemory::InMemory(string filename, torch::Tensor data, torch::Device device) {
    filename_ = filename;
    dim0_size_ = data.size(0);
    dim1_size_ = data.size(1);
    dtype_ = data.scalar_type();
    device_ = device;
    loaded_ = false;

    torch::Tensor temp = data.to(torch::kCPU);

    std::ofstream outfile(filename_, ios::out | ios::binary);

    int64_t dtype_size = get_dtype_size_wrapper(dtype_);

    outfile.write((char *)temp.data_ptr(), data.size(0) * data.size(1) * dtype_size);

    outfile.close();
}

InMemory::InMemory(string filename, torch::Dtype dtype) {
    filename_ = filename;
    dim0_size_ = 0;
    dim1_size_ = 0;
    initialized_ = false;
    dtype_ = dtype;
    device_ = torch::kCPU;
    loaded_ = false;
}

InMemory::InMemory(torch::Tensor data) {
    if (data.sizes().size() == 2) {
        dim0_size_ = data.size(0);
        dim1_size_ = data.size(1); 
    } else if (data.sizes().size() == 1) {
        dim0_size_ = data.size(0);
        dim1_size_ = 1;
    } else {
        throw GegeRuntimeException("Tensor must have 1 or two dimensions");
    }

    filename_ = "";
    data_ = data.reshape({dim0_size_, dim1_size_});

    initialized_ = true;
    dtype_ = data.scalar_type();
    device_ = data.device();
    loaded_ = true;
}

void InMemory::load() {
    if (!loaded_ && !filename_.empty()) {
        fd_ = open((filename_).c_str(), O_RDWR);
        if (fd_ == -1) {
            SPDLOG_DEBUG("Unable to open {}\nError: {}", filename_, errno);
            return;
        }
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        data_ = torch::empty({dim0_size_, dim1_size_}, dtype_);

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pread_wrapper(fd_, data_.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to read {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }

        if (device_ == torch::kCUDA) {
            data_ = data_.to(device_);
        }

        loaded_ = true;
    }
}

void InMemory::write() {
    if (loaded_ && !filename_.empty()) {
        int64_t dtype_size = get_dtype_size_wrapper(dtype_);

        torch::Tensor data = data_;
        if (device_ == torch::kCUDA) {
            data = data_.to(torch::kCPU);
        }

        int64_t offset = 0;
        int64_t read_size = dim0_size_ * dim1_size_ * dtype_size;

        if (pwrite_wrapper(fd_, data.data_ptr(), read_size, offset) == -1) {
            SPDLOG_ERROR("Unable to write {}\nError: {}", filename_, errno);
            throw std::runtime_error("");
        }
    }
}

void InMemory::unload(bool perform_write) {
    if (loaded_ && !filename_.empty()) {
        if (perform_write) {
            write();
        }

        close(fd_);
        fd_ = -1;
        loaded_ = false;
        data_ = torch::Tensor();
    }
}

torch::Tensor InMemory::indexRead(Indices indices) {
    if (indices.sizes().size() != 1) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }

    if (data_.defined()) {
        return data_.index_select(0, indices.to(device_));
    } else {
        return torch::Tensor();
    }
}

void InMemory::indexAdd(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    int64_t debug_update_id = -1;
    bool run_stage_debug = should_run_stage_debug(debug_update_id);
    auto index_add_start = std::chrono::high_resolution_clock::now();
    auto step_start = index_add_start;

    if (values.device().is_cuda()) {
#ifdef GEGE_CUDA
        if (csr_update_enabled()) {
            ScopedNvtxRange nvtx_scope("storage.InMemory.indexAdd.csr");
            static bool logged = false;
            if (!logged) {
                SPDLOG_INFO("InMemory::indexAdd using direct CSR update path");
                logged = true;
            }
            torch::Tensor update_indices = indices;
            torch::Tensor update_values = values;
            if (csr_update_reduce_enabled()) {
                std::tie(update_indices, update_values) = reduce_updates_with_csr(indices, values);
                if (run_stage_debug) {
                    auto now = std::chrono::high_resolution_clock::now();
                    SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 1] csr_reduce ms={:.3f} in_rows={} out_rows={}",
                                debug_update_id, elapsed_ms(step_start, now), indices.numel(), update_indices.numel());
                    step_start = now;
                }
            }
            data_.index_add_(0, update_indices, update_values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 2] index_add_cuda ms={:.3f} rows={} dim={}",
                            debug_update_id, elapsed_ms(step_start, now), update_indices.numel(), update_values.size(1));
            }
        } else {
            data_.index_add_(0, indices, values);
            if (run_stage_debug) {
                auto now = std::chrono::high_resolution_clock::now();
                SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 1] index_add_cuda ms={:.3f} rows={} dim={} csr_update={}",
                            debug_update_id, elapsed_ms(step_start, now), indices.numel(), values.size(1), false);
            }
        }
#else
        data_.index_add_(0, indices, values);
#endif
    } else {
        int64_t size = indices.size(0);
        int d = values.size(1);
        checked_cpu_index_add_("InMemory::indexAdd", data_, indices, values);
        if (run_stage_debug) {
            auto now = std::chrono::high_resolution_clock::now();
            SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 1] index_add_cpu ms={:.3f} rows={} dim={}",
                        debug_update_id, elapsed_ms(step_start, now), size, d);
        }
    }

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][storage.indexAdd][update {}][step 9] total_ms={:.3f} device={}",
                    debug_update_id, elapsed_ms(index_add_start, now), values.device().str());
    }
}

void InMemory::indexPut(Indices indices, torch::Tensor values) {
    if (!values.defined() || indices.sizes().size() != 1 || indices.size(0) != values.size(0) || data_.size(1) != values.size(1)) {
        // TODO: throw invalid input to func exception
        throw std::runtime_error("");
    }
    if (values.device().is_cuda()) {
        data_[indices] = values;
    } else {
        checked_cpu_index_put_("InMemory::indexPut", data_, indices, values);
    }
}

torch::Tensor InMemory::range(int64_t offset, int64_t n) {
    if (n + offset > dim0_size_) {
        // TODO: throw invalid inputs for function error
        throw std::runtime_error("");
    }
    return data_.narrow(0, offset, n);
}

void InMemory::rangePut(int64_t offset, int64_t n, torch::Tensor values) { data_.narrow(0, offset, n).copy_(values); }

void InMemory::shuffle() {
    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full shuffle
    if (edge_bucket_sizes_.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::randperm(dim0_size_, opts)));
    }
    // shuffle within edge buckets
    else {
        int64_t start = 0;
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::randperm(edge_bucket.size(0), opts)));
            start += *itr;
        }
    }
    // if (!loaded) {
    //     unload(true);
    // }
}

// void InMemory::shuffle() {
//     auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
//     torch::Tenosr perm = torch::randperm(dim0_size_, opts);


// }

void InMemory::sort(bool src) {
    // function for sorting in memory edges
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }

    bool loaded = loaded_;
    if (!loaded) {
        load();

        // may cause silent failures
        if (!loaded_) {
            return;
        }
    }

    // full sort
    if (edge_bucket_sizes_.empty()) {
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        data_.copy_(data_.index_select(0, torch::argsort(data_.select(1, sort_dim))));
    }
    // sort within edge buckets
    else {
        int64_t start = 0;
        // auto opts = torch::TensorOptions().dtype(torch::kInt64).device(data_.device());
        for (auto itr = edge_bucket_sizes_.begin(); itr != edge_bucket_sizes_.end(); itr++) {
            torch::Tensor edge_bucket = data_.narrow(0, start, *itr);
            data_.narrow(0, start, *itr) = (edge_bucket.index_select(0, torch::argsort(edge_bucket.select(1, sort_dim))));
            start += *itr;
        }
    }
    if (!loaded) {
        unload(true);
    }
}
