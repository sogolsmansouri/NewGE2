#pragma once

#include <functional>
#include <tuple>
#include "nn/decoders/edge/edge_decoder.h"

// Opt-in compute workers are separate from the logical storage/training lanes.
// GEGE_FOUR_WORKER_COMPLEX_DEVICES must contain four distinct visible CUDA IDs.
// Optional GEGE_FOUR_WORKER_COMPLEX_TRACE records actual forward/backward work.
bool four_worker_complex_requested();

// Returns native-shaped, unfiltered (pos, neg, inverse_pos, inverse_neg) scores
// on the owner device. Sampling, filters, SUM loss, backward, and optimizer steps
// remain with the caller. Differentiable device copies retain the original
// owner embedding/relation leaves; no worker model or optimizer is created.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
four_worker_complex_forward(
    std::shared_ptr<EdgeDecoder> decoder,
    torch::Tensor positive_edges,
    torch::Tensor node_embeddings,
    torch::Tensor dst_negative_mapping,
    torch::Tensor src_negative_mapping,
    int64_t batch_id,
    std::function<void()> post_gather_callback = nullptr);
