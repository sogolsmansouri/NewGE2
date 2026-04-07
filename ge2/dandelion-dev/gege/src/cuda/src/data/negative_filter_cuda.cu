#include "data/samplers/negative_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime_api.h>

namespace {

__global__ void build_deg_padded_filter_kernel(const int64_t *sample_indices,
                                               int64_t n,
                                               int64_t num_degree,
                                               int64_t chunk_size,
                                               int64_t *filter) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int64_t chunk_id = idx / num_degree;
    int64_t neg_id = idx - chunk_id * num_degree;
    int64_t edge_id = sample_indices[idx];
    int64_t chunk_start = chunk_id * chunk_size;
    int64_t chunk_end = chunk_start + chunk_size;

    filter[idx * 2] = (edge_id >= chunk_start && edge_id < chunk_end) ? edge_id : -1;
    filter[idx * 2 + 1] = neg_id;
}

template <typename scalar_t>
__global__ void apply_score_filter_kernel(scalar_t *scores,
                                          int64_t score_rows,
                                          int64_t score_cols,
                                          const int64_t *filter,
                                          int64_t filter_rows) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= filter_rows) {
        return;
    }

    int64_t row = filter[idx * 2];
    int64_t col = filter[idx * 2 + 1];
    if (row < 0 || row >= score_rows || col < 0 || col >= score_cols) {
        return;
    }

    scores[row * score_cols + col] = static_cast<scalar_t>(-1e9);
}

}  // namespace

torch::Tensor deg_negative_local_filter_padded_cuda(torch::Tensor deg_sample_indices, int64_t num_edges) {
    TORCH_CHECK(deg_sample_indices.is_cuda(), "deg_negative_local_filter_padded_cuda expects CUDA sample indices");
    TORCH_CHECK(deg_sample_indices.scalar_type() == torch::kInt64, "deg_negative_local_filter_padded_cuda expects int64 sample indices");
    TORCH_CHECK(deg_sample_indices.dim() == 2, "deg_negative_local_filter_padded_cuda expects [num_chunks, num_degree]");

    c10::cuda::CUDAGuard guard(deg_sample_indices.device());
    torch::Tensor sample = deg_sample_indices.contiguous();
    int64_t num_chunks = sample.size(0);
    int64_t num_degree = sample.size(1);
    int64_t n = sample.numel();
    auto filter = torch::empty({n, 2}, sample.options());
    if (n == 0) {
        return filter;
    }

    int64_t chunk_size = (num_edges + num_chunks - 1) / num_chunks;
    auto stream = at::cuda::getCurrentCUDAStream(sample.device().index()).stream();
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    build_deg_padded_filter_kernel<<<blocks, threads, 0, stream>>>(
        sample.data_ptr<int64_t>(), n, num_degree, chunk_size, filter.data_ptr<int64_t>());
    AT_CUDA_CHECK(cudaGetLastError());
    return filter;
}

void apply_score_filter_cuda(torch::Tensor scores, torch::Tensor filter) {
    TORCH_CHECK(scores.is_cuda(), "apply_score_filter_cuda expects CUDA scores");
    TORCH_CHECK(filter.is_cuda(), "apply_score_filter_cuda expects CUDA filter");
    TORCH_CHECK(filter.scalar_type() == torch::kInt64, "apply_score_filter_cuda expects int64 filter");
    TORCH_CHECK(scores.dim() == 2, "apply_score_filter_cuda expects 2D scores");
    TORCH_CHECK(filter.dim() == 2 && filter.size(1) == 2, "apply_score_filter_cuda expects [N, 2] filter");
    TORCH_CHECK(scores.is_contiguous(), "apply_score_filter_cuda expects contiguous scores");
    TORCH_CHECK(filter.is_contiguous(), "apply_score_filter_cuda expects contiguous filter");

    if (filter.size(0) == 0 || scores.size(0) == 0 || scores.size(1) == 0) {
        return;
    }

    c10::cuda::CUDAGuard guard(scores.device());
    auto stream = at::cuda::getCurrentCUDAStream(scores.device().index()).stream();
    int64_t filter_rows = filter.size(0);
    int threads = 256;
    int blocks = static_cast<int>((filter_rows + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scores.scalar_type(), "apply_score_filter_cuda", [&] {
        apply_score_filter_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            scores.data_ptr<scalar_t>(), scores.size(0), scores.size(1), filter.data_ptr<int64_t>(), filter_rows);
    });
    AT_CUDA_CHECK(cudaGetLastError());
}
