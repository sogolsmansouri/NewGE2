#pragma once

#include "nn/decoders/edge/edge_decoder.h"

/**
 * TRing3: GETD-style Tensor Ring scorer for arity-3 facts (src, rel, dst, qval).
 *
 * The contraction order is rel -> src -> dst(open) -> qval, returning a partial
 * vector in entity space for ranking the corrupted dst slot.
 */
class TRing3 : public EdgeDecoder, public torch::nn::Cloneable<TRing3> {
   public:
    int ring_rank_;

    torch::Tensor rel_core_;
    torch::Tensor src_core_;
    torch::Tensor dst_core_;
    torch::Tensor qval_core_;

    TRing3(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(),
           int ring_rank = 50, EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;

    torch::Tensor tring3_partial(torch::Tensor e_s, torch::Tensor e_r, torch::Tensor e_qv);
};
