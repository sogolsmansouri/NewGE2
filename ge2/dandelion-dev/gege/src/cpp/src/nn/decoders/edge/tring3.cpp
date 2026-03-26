#include "nn/decoders/edge/tring3.h"

namespace {

void init_tr_core(torch::Tensor &tensor) {
    torch::nn::init::uniform_(tensor, -1e-1, 1e-1);
}

torch::Tensor init_relation_embeddings(int64_t rows, int64_t cols, torch::TensorOptions options) {
    return 1e-3 * torch::randn({rows, cols}, options);
}

torch::Tensor contract_core_with_vector(torch::Tensor core, torch::Tensor vec) {
    int64_t ring_rank = core.size(0);
    torch::Tensor flattened = core.permute({1, 0, 2}).contiguous().view({core.size(1), ring_rank * ring_rank});
    return torch::matmul(vec, flattened).view({vec.size(0), ring_rank, ring_rank});
}

}  // namespace

TRing3::TRing3(int num_relations, int embedding_dim, torch::TensorOptions tensor_options,
               int ring_rank, EdgeDecoderMethod decoder_method) {
    comparator_ = std::make_shared<DotCompare>();
    relation_operator_ = std::make_shared<HadamardOperator>();  // unused for Tensor Ring
    num_relations_ = num_relations;
    embedding_size_ = embedding_dim;
    use_inverse_relations_ = false;
    tensor_options_ = tensor_options;
    decoder_method_ = decoder_method;
    ring_rank_ = ring_rank;

    learning_task_ = LearningTask::LINK_PREDICTION;

    TRing3::reset();
}

void TRing3::reset() {
    int r = ring_rank_;
    int d = embedding_size_;

    rel_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(rel_core_);
    rel_core_ = register_parameter("tring3_rel_core", rel_core_);

    src_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(src_core_);
    src_core_ = register_parameter("tring3_src_core", src_core_);

    dst_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(dst_core_);
    dst_core_ = register_parameter("tring3_dst_core", dst_core_);

    qval_core_ = torch::zeros({r, d, r}, tensor_options_).set_requires_grad(true);
    init_tr_core(qval_core_);
    qval_core_ = register_parameter("tring3_qval_core", qval_core_);

    relations_ = init_relation_embeddings(num_relations_, d, tensor_options_).set_requires_grad(true);
    relations_ = register_parameter("relation_embeddings", relations_);
}

torch::Tensor TRing3::tring3_partial(torch::Tensor e_s, torch::Tensor e_r, torch::Tensor e_qv) {
    torch::Tensor rel_state = contract_core_with_vector(rel_core_, e_r);
    torch::Tensor src_state = contract_core_with_vector(src_core_, e_s);
    torch::Tensor qval_state = contract_core_with_vector(qval_core_, e_qv);
    torch::Tensor left_state = torch::bmm(rel_state, src_state);
    torch::Tensor trace_state = torch::bmm(qval_state, left_state);
    return torch::einsum("bcd,cnd->bn", {trace_state.transpose(1, 2), dst_core_});
}
