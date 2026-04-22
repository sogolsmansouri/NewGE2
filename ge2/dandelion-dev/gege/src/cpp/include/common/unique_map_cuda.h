#pragma once

#include "common/datatypes.h"

struct UniqueMapCudaDebugInfo {
    std::string requested_backend;
    std::string executed_backend;
    std::string fallback_backend;
    std::string fallback_reason;
    int64_t total_calls = 0;
    int64_t total_fallbacks = 0;
    double device_unique_ms = 0.0;
    double device_bitmap_zero_ms = 0.0;
    double device_bitmap_output_init_ms = 0.0;
    double device_bitmap_mark_ms = 0.0;
    double device_bitmap_count_ms = 0.0;
    double device_bitmap_scan_ms = 0.0;
    double device_bitmap_extract_ms = 0.0;
    double device_bitmap_mask_ms = 0.0;
    double device_bitmap_inverse_ms = 0.0;
    bool used_fallback = false;
    bool cuco_compiled = false;
    bool device_unique_timing_valid = false;
    bool measure_device_timing = false;
    bool sorted_request = false;
};

// Returns {unique_ids, inverse_indices} for a 1D CUDA int64 tensor.
// Ordering of unique_ids depends on the selected backend.
std::tuple<torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda(torch::Tensor all_ids, bool sorted,
                                                                         UniqueMapCudaDebugInfo *debug_info = nullptr,
                                                                         int64_t value_domain_size = -1);

// Returns {padded_unique_ids, inverse_indices, active_mask} for a 1D CUDA int64 tensor using
// the bitmap backend. padded_unique_ids has all_ids.numel() rows; valid unique
// ids are written at the front, active_mask marks those rows, and unused rows
// are zero-filled. This avoids the host-side unique-count read needed by the
// variable-length API.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> map_tensors_unique_inverse_cuda_bitmap_padded(torch::Tensor all_ids,
                                                                                                      UniqueMapCudaDebugInfo *debug_info = nullptr,
                                                                                                      int64_t value_domain_size = -1);

// Same padded bitmap contract, but consumes the input tensors separately. This
// avoids the hot-path concat/pack step before bitmap mapping.
std::tuple<torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> map_tensors_unique_inverse_cuda_bitmap_padded_multi(
    const std::vector<torch::Tensor> &ids,
    UniqueMapCudaDebugInfo *debug_info = nullptr,
    int64_t value_domain_size = -1);

// Fused CUDA helpers for the fixed-buffer bitmap experiment. These consume the
// active-row mask on device, avoiding dynamic compaction before optimizer/update.
std::tuple<torch::Tensor, torch::Tensor> active_masked_adagrad_cuda(torch::Tensor gradients,
                                                                    torch::Tensor optimizer_state,
                                                                    torch::Tensor active_mask,
                                                                    double learning_rate);

void active_masked_index_add_cuda(torch::Tensor target,
                                  torch::Tensor indices,
                                  torch::Tensor values,
                                  torch::Tensor active_mask);
