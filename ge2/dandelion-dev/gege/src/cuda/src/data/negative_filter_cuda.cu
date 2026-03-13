#include "data/samplers/negative_filter_cuda.h"

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>

namespace {

struct DegFilterCudaWorkspaceCache {
    torch::Tensor flags;
    torch::Tensor offsets;
    torch::Tensor temp_storage;
    int64_t capacity = 0;
    int64_t temp_storage_bytes = 0;
    torch::Device device = torch::Device(torch::kCPU);
};

thread_local DegFilterCudaWorkspaceCache deg_filter_cuda_workspace_cache;

void ensure_deg_filter_workspace(const torch::Tensor &deg_sample_indices, int64_t num_entries, size_t temp_storage_bytes) {
    auto device = deg_sample_indices.device();
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    if (!deg_filter_cuda_workspace_cache.flags.defined() || deg_filter_cuda_workspace_cache.device != device ||
        deg_filter_cuda_workspace_cache.capacity < num_entries) {
        int64_t next_capacity =
            std::max<int64_t>(num_entries, std::max<int64_t>(deg_filter_cuda_workspace_cache.capacity * 2, int64_t{4096}));
        deg_filter_cuda_workspace_cache.flags = torch::empty({next_capacity}, int_opts);
        deg_filter_cuda_workspace_cache.offsets = torch::empty({next_capacity}, int_opts);
        deg_filter_cuda_workspace_cache.capacity = next_capacity;
        deg_filter_cuda_workspace_cache.device = device;
    }

    if (!deg_filter_cuda_workspace_cache.temp_storage.defined() || deg_filter_cuda_workspace_cache.device != device ||
        deg_filter_cuda_workspace_cache.temp_storage_bytes < static_cast<int64_t>(temp_storage_bytes)) {
        int64_t next_temp_bytes = std::max<int64_t>(static_cast<int64_t>(temp_storage_bytes),
                                                    std::max<int64_t>(deg_filter_cuda_workspace_cache.temp_storage_bytes * 2,
                                                                      int64_t{4096}));
        deg_filter_cuda_workspace_cache.temp_storage = torch::empty({next_temp_bytes}, byte_opts);
        deg_filter_cuda_workspace_cache.temp_storage_bytes = next_temp_bytes;
        deg_filter_cuda_workspace_cache.device = device;
    }
}

__global__ void mark_deg_filter_matches_kernel(const int64_t *deg_sample_indices,
                                               int32_t *flags,
                                               int64_t num_entries,
                                               int64_t num_deg_negs,
                                               int64_t chunk_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_entries) {
        return;
    }

    int64_t chunk_id = idx / num_deg_negs;
    int64_t edge_id = deg_sample_indices[idx];
    flags[idx] = (edge_id / chunk_size == chunk_id) ? 1 : 0;
}

__global__ void scatter_deg_filter_matches_kernel(const int64_t *deg_sample_indices,
                                                  const int32_t *flags,
                                                  const int32_t *offsets,
                                                  int64_t *filter_out,
                                                  int64_t num_entries,
                                                  int64_t num_deg_negs) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_entries || flags[idx] == 0) {
        return;
    }

    int64_t out_row = static_cast<int64_t>(offsets[idx]);
    filter_out[2 * out_row] = deg_sample_indices[idx];
    filter_out[2 * out_row + 1] = idx % num_deg_negs;
}

}  // namespace

std::tuple<torch::Tensor, DegNegativeLocalFilterCudaStats> deg_negative_local_filter_cuda(torch::Tensor deg_sample_indices,
                                                                                          int64_t chunk_size) {
    TORCH_CHECK(deg_sample_indices.is_cuda(), "deg_negative_local_filter_cuda expects a CUDA tensor");
    TORCH_CHECK(deg_sample_indices.scalar_type() == torch::kInt64, "deg_negative_local_filter_cuda expects int64 input");
    TORCH_CHECK(deg_sample_indices.dim() == 2, "deg_negative_local_filter_cuda expects a 2D tensor");
    TORCH_CHECK(chunk_size > 0, "deg_negative_local_filter_cuda requires positive chunk_size");

    auto input = deg_sample_indices.contiguous();
    int64_t num_chunks = input.size(0);
    int64_t num_deg_negs = input.size(1);
    int64_t num_entries = num_chunks * num_deg_negs;
    DegNegativeLocalFilterCudaStats stats;

    auto out_opts = torch::TensorOptions().dtype(torch::kInt64).device(input.device());
    if (num_entries == 0) {
        return std::forward_as_tuple(torch::empty({0, 2}, out_opts), stats);
    }

    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();

    size_t temp_storage_bytes = 0;
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, static_cast<const int32_t *>(nullptr),
                                                static_cast<int32_t *>(nullptr), num_entries, stream));
    ensure_deg_filter_workspace(input, num_entries, temp_storage_bytes);

    auto flags = deg_filter_cuda_workspace_cache.flags.narrow(0, 0, num_entries);
    auto offsets = deg_filter_cuda_workspace_cache.offsets.narrow(0, 0, num_entries);
    auto temp_storage = deg_filter_cuda_workspace_cache.temp_storage.narrow(0, 0, temp_storage_bytes);

    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((num_entries + threads_per_block - 1) / threads_per_block);

    auto match_start = std::chrono::high_resolution_clock::now();
    mark_deg_filter_matches_kernel<<<blocks, threads_per_block, 0, stream>>>(input.data_ptr<int64_t>(), flags.data_ptr<int32_t>(),
                                                                              num_entries, num_deg_negs, chunk_size);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    stats.match_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - match_start).count();

    auto compact_start = std::chrono::high_resolution_clock::now();
    AT_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes, flags.data_ptr<int32_t>(),
                                                offsets.data_ptr<int32_t>(), num_entries, stream));
    AT_CUDA_CHECK(cudaGetLastError());

    int32_t last_flag = 0;
    int32_t last_offset = 0;
    AT_CUDA_CHECK(cudaMemcpyAsync(&last_flag, flags.data_ptr<int32_t>() + (num_entries - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaMemcpyAsync(&last_offset, offsets.data_ptr<int32_t>() + (num_entries - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t match_count = static_cast<int64_t>(last_offset) + static_cast<int64_t>(last_flag);
    stats.compact_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - compact_start).count();

    if (match_count == 0) {
        return std::forward_as_tuple(torch::empty({0, 2}, out_opts), stats);
    }

    torch::Tensor filter = torch::empty({match_count, 2}, out_opts);
    auto scatter_start = std::chrono::high_resolution_clock::now();
    scatter_deg_filter_matches_kernel<<<blocks, threads_per_block, 0, stream>>>(
        input.data_ptr<int64_t>(), flags.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(), filter.data_ptr<int64_t>(), num_entries,
        num_deg_negs);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
    stats.scatter_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - scatter_start).count();

    return std::forward_as_tuple(filter, stats);
}
