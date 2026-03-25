#pragma once

#include "nn/decoders/edge/edge_decoder.h"

/**
 * TRing2: GETD-style Tensor Ring scorer for binary facts (src, rel, dst).
 *
 * The decoder performs a streamed tensor-ring contraction over relation and src,
 * keeping the dst mode open, and returns a partial vector in entity space.
 * Scores are obtained by a dot product between the partial vector and dst.
 */
class TRing2 : public EdgeDecoder, public torch::nn::Cloneable<TRing2> {
   public:
    int ring_rank_;

    torch::Tensor rel_core_;
    torch::Tensor src_core_;
    torch::Tensor dst_core_;

    TRing2(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(),
           bool use_inverse_relations = true, int ring_rank = 50,
           EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;

    torch::Tensor tring2_partial(torch::Tensor e_s, torch::Tensor e_r);
};
