#pragma once

#include "common/datatypes.h"

torch::Tensor deg_negative_local_filter_padded_cuda(torch::Tensor deg_sample_indices, int64_t num_edges);

void apply_score_filter_cuda(torch::Tensor scores, torch::Tensor filter);
