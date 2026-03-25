#include "nn/decoders/edge/tring2.h"

namespace {

void init_tr_core(torch::Tensor &tensor) {
    torch::nn::init::uniform_(tensor, -1e-1, 1e-1);
}

torch::Tensor init_relation_embeddings(int64_t rows, int64_t cols, torch::TensorOptions options) {
    return 1e-3 * torch::randn({rows, cols}, options);
}

}  // namespace

TRing2::TRing2(int num_relations, int embedding_dim, torch::TensorOptions tensor_options,
               bool use_inverse_relations, int ring_rank, EdgeDecoderMethod decoder_method) {
    comparator_ = std::make_shared<DotCompare>();
    relation_operator_ = std::make_shared<HadamardOperator>();  // unused for Tensor Ring
    num_relations_ = num_relations;
    embedding_size_ = embedding_dim;
    use_inverse_relations_ = use_inverse_relations;
    tensor_options_ = tensor_options;
    decoder_method_ = decoder_method;
    ring_rank_ = ring_rank;

    learning_task_ = LearningTask::LINK_PREDICTION;

    TRing2::reset();
}

void TRing2::reset() {
    int r = ring_rank_;
    int d = embedding_size_;

    rel_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(rel_core_);
    rel_core_ = register_parameter("tring2_rel_core", rel_core_);

    src_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(src_core_);
    src_core_ = register_parameter("tring2_src_core", src_core_);

    dst_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(dst_core_);
    dst_core_ = register_parameter("tring2_dst_core", dst_core_);

    relations_ = init_relation_embeddings(num_relations_, d, tensor_options_).set_requires_grad(true);
    relations_ = register_parameter("relation_embeddings", relations_);
    if (use_inverse_relations_) {
        inverse_relations_ = init_relation_embeddings(num_relations_, d, tensor_options_).set_requires_grad(true);
        inverse_relations_ = register_parameter("inverse_relation_embeddings", inverse_relations_);
    }
}

torch::Tensor TRing2::tring2_partial(torch::Tensor e_s, torch::Tensor e_r) {
    torch::Tensor state = torch::einsum("bn,anc->bac", {e_r, rel_core_});
    state = torch::einsum("blc,cnd,bn->bld", {state, src_core_, e_s});
    state = torch::einsum("blc,cnd->blnd", {state, dst_core_});
    torch::Tensor diag = state.diagonal(0, 1, 3);
    return diag.sum(-1);
}
