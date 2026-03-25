#pragma once

#include "nn/decoders/edge/edge_decoder.h"

/**
 * TRing4: GETD-style Tensor Ring scorer for arity-4 facts (src, rel, dst, qrel, qval).
 *
 * The contraction order is rel -> src -> dst(open) -> qrel -> qval, returning a
 * partial vector in entity space for ranking the corrupted dst slot.
 */
class TRing4 : public EdgeDecoder, public torch::nn::Cloneable<TRing4> {
   public:
    int ring_rank_;

    torch::Tensor rel_core_;
    torch::Tensor src_core_;
    torch::Tensor dst_core_;
    torch::Tensor qrel_core_;
    torch::Tensor qval_core_;

    TRing4(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(),
           int ring_rank = 40, EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;

    torch::Tensor tring4_partial(torch::Tensor e_s, torch::Tensor e_r,
                                 torch::Tensor e_qr, torch::Tensor e_qv);
};
