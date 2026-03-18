#include "nn/decoders/edge/tucker3.h"

TuckER3::TuckER3(int num_relations, int embedding_dim, torch::TensorOptions tensor_options,
                 int core_dim_e, int core_dim_r,
                 EdgeDecoderMethod decoder_method) {
    comparator_ = std::make_shared<DotCompare>();
    relation_operator_ = std::make_shared<HadamardOperator>();  // unused for arity-3 path
    num_relations_ = num_relations;
    embedding_size_ = embedding_dim;
    use_inverse_relations_ = false;  // arity-3 disables inverse relations
    tensor_options_ = tensor_options;
    decoder_method_ = decoder_method;
    core_dim_e_ = core_dim_e;
    core_dim_r_ = core_dim_r;

    learning_task_ = LearningTask::LINK_PREDICTION;

    TuckER3::reset();
}

void TuckER3::reset() {
    int ke = core_dim_e_, kr = core_dim_r_, d = embedding_size_;

    // Tucker core W [k_e, k_r, k_e, k_e] — Xavier-uniform initialisation
    W_core_ = torch::zeros({ke, kr, ke, ke}, tensor_options_).set_requires_grad(true);
    torch::nn::init::xavier_uniform_(W_core_.view({ke, kr * ke * ke}));
    W_core_ = register_parameter("tucker3_core", W_core_);

    // Entity projection [d_e, k_e]
    A_e_ = torch::zeros({d, ke}, tensor_options_).set_requires_grad(true);
    torch::nn::init::xavier_uniform_(A_e_);
    A_e_ = register_parameter("tucker3_proj_entity", A_e_);

    // Relation projection [d_r, k_r]
    A_r_ = torch::zeros({d, kr}, tensor_options_).set_requires_grad(true);
    torch::nn::init::xavier_uniform_(A_r_);
    A_r_ = register_parameter("tucker3_proj_relation", A_r_);

    // Relation embedding table
    relations_ = torch::ones({num_relations_, d}, tensor_options_).set_requires_grad(true);
    relations_ = register_parameter("relation_embeddings", relations_);
}

torch::Tensor TuckER3::tucker3_partial(torch::Tensor e_s, torch::Tensor e_r, torch::Tensor e_qv) {
    // Project all inputs into core space
    torch::Tensor es  = torch::mm(e_s,  A_e_);   // [B, k_e]
    torch::Tensor er  = torch::mm(e_r,  A_r_);   // [B, k_r]
    torch::Tensor eqv = torch::mm(e_qv, A_e_);   // [B, k_e]

    // 4-mode Tucker contraction, leaving mode 2 (dst) open.
    // W_core_ has shape [k_e, k_r, k_e, k_e] with named modes p, q, r, s.
    // partial_core[b, r] = Σ_{p,q,s} W[p,q,r,s] * es[b,p] * er[b,q] * eqv[b,s]
    torch::Tensor partial_core = torch::einsum("pqrs,bp,bq,bs->br",
                                               {W_core_, es, er, eqv});  // [B, k_e]

    // Project partial back to entity embedding dimension
    return torch::mm(partial_core, A_e_.t());  // [B, embedding_dim]
}
