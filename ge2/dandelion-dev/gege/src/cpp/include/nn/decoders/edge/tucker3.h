#pragma once

#include "nn/decoders/edge/edge_decoder.h"

/**
 * TuckER3: Tucker decomposition scorer for arity-3 facts (src, rel, dst, qval).
 *
 * Score(s, r, o, qv) = W ×₁ (e_s A_e) ×₂ (e_r A_r) ×₃ (e_o A_e) ×₄ (e_qv A_e)
 * where W ∈ R^{k_e × k_r × k_e × k_e} is the 4-mode Tucker core.
 *
 * tucker3_partial() contracts all modes except the dst mode (mode 2), returning
 * shape [B, embedding_dim] which is then dot-producted with the dst embedding.
 */
class TuckER3 : public EdgeDecoder, public torch::nn::Cloneable<TuckER3> {
   public:
    int core_dim_e_;  /**< Tucker core entity dimension (k_e) */
    int core_dim_r_;  /**< Tucker core relation dimension (k_r) */

    torch::Tensor W_core_;  /**< Tucker core tensor [k_e, k_r, k_e, k_e] */
    torch::Tensor A_e_;     /**< Entity projection matrix [embedding_dim, k_e] */
    torch::Tensor A_r_;     /**< Relation projection matrix [embedding_dim, k_r] */

    TuckER3(int num_relations, int embedding_dim, torch::TensorOptions tensor_options = torch::TensorOptions(),
            int core_dim_e = 10, int core_dim_r = 10,
            EdgeDecoderMethod decoder_method = EdgeDecoderMethod::CORRUPT_NODE);

    void reset() override;

    /**
     * Tucker-3 partial contraction leaving the dst mode open.
     *
     * @param e_s   Source entity embeddings [B, embedding_dim]
     * @param e_r   Relation embeddings [B, embedding_dim]
     * @param e_qv  Qualifier value entity embeddings [B, embedding_dim]
     * @return      Partial score vector [B, embedding_dim]; dot with e_dst gives scalar score.
     */
    torch::Tensor tucker3_partial(torch::Tensor e_s, torch::Tensor e_r, torch::Tensor e_qv);
};
