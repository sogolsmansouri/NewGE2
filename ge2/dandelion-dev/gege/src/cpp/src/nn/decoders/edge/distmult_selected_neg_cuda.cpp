#include <torch/script.h>

#include "nn/decoders/edge/distmult_selected_neg_cuda.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

namespace {

class DistMultSelectedNegScores : public torch::autograd::Function<DistMultSelectedNegScores> {
   public:
    static variable_list forward(AutogradContext *ctx, Variable chunked_adjusted_embeddings, Variable negative_embeddings, Variable selected_neg_indices) {
        auto out = distmult_selected_neg_scores_cuda_forward(chunked_adjusted_embeddings, negative_embeddings, selected_neg_indices);
        ctx->save_for_backward({chunked_adjusted_embeddings, negative_embeddings, selected_neg_indices});
        return {out};
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
        auto grad_out = grad_outs[0];
        auto saved = ctx->get_saved_variables();
        auto grads = distmult_selected_neg_scores_cuda_backward(grad_out, saved[0], saved[1], saved[2]);
        return {std::get<0>(grads), std::get<1>(grads), Variable()};
    }
};

}  // namespace

torch::Tensor distmult_selected_neg_scores(torch::Tensor chunked_adjusted_embeddings,
                                           torch::Tensor negative_embeddings,
                                           torch::Tensor selected_neg_indices) {
    auto result = DistMultSelectedNegScores::apply(chunked_adjusted_embeddings, negative_embeddings, selected_neg_indices);
    return result[0];
}
