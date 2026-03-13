#pragma once

#include "common/datatypes.h"

struct DegNegativeLocalFilterCudaStats {
    int64_t match_ns = 0;
    int64_t compact_ns = 0;
    int64_t scatter_ns = 0;
};

std::tuple<torch::Tensor, DegNegativeLocalFilterCudaStats> deg_negative_local_filter_cuda(torch::Tensor deg_sample_indices,
                                                                                          int64_t chunk_size);
