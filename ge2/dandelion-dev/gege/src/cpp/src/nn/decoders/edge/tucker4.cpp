#include "nn/decoders/edge/tucker4.h"

TuckER4::TuckER4(int num_relations, int embedding_dim, torch::TensorOptions tensor_options,
                 int core_dim_e, int core_dim_r,
                 EdgeDecoderMethod decoder_method) {
    // TuckER4 uses DotCompare for scoring (partial ⋅ e_dst)
    // No relation operator needed — Tucker contraction replaces apply_relation
    comparator_ = std::make_shared<DotCompare>();
    relation_operator_ = std::make_shared<HadamardOperator>();  // unused for arity-4 path
    num_relations_ = num_relations;
    embedding_size_ = embedding_dim;
    use_inverse_relations_ = false;  // arity-4 disables inverse relations
    tensor_options_ = tensor_options;
    decoder_method_ = decoder_method;
    core_dim_e_ = core_dim_e;
    core_dim_r_ = core_dim_r;

    learning_task_ = LearningTask::LINK_PREDICTION;

    TuckER4::reset();
}

void TuckER4::reset() {
    int ke = core_dim_e_, kr = core_dim_r_, d = embedding_size_;

    // Tucker core W [k_e, k_r, k_e, k_r, k_e] — Xavier-uniform initialisation
    W_core_ = torch::zeros({ke, kr, ke, kr, ke}, tensor_options_).set_requires_grad(true);
    torch::nn::init::xavier_uniform_(W_core_.view({ke, kr * ke * kr * ke}));
    W_core_ = register_parameter("tucker4_core", W_core_);

    // Entity projection [d_e, k_e]
    A_e_ = torch::zeros({d, ke}, tensor_options_).set_requires_grad(true);
    torch::nn::init::xavier_uniform_(A_e_);
    A_e_ = register_parameter("tucker4_proj_entity", A_e_);

    // Relation projection [d_r, k_r]  (d_r == embedding_size_ for shared embedding dim)
    A_r_ = torch::zeros({d, kr}, tensor_options_).set_requires_grad(true);
    torch::nn::init::xavier_uniform_(A_r_);
    A_r_ = register_parameter("tucker4_proj_relation", A_r_);

    // Relation embedding table (shared between rel and qrel columns)
    relations_ = torch::ones({num_relations_, d}, tensor_options_).set_requires_grad(true);
    relations_ = register_parameter("relation_embeddings", relations_);
}

torch::Tensor TuckER4::tucker4_partial(torch::Tensor e_s, torch::Tensor e_r,
                                       torch::Tensor e_qr, torch::Tensor e_qv) {
    // Project all inputs into core space
    torch::Tensor es  = torch::mm(e_s,  A_e_);   // [B, k_e]
    torch::Tensor er  = torch::mm(e_r,  A_r_);   // [B, k_r]
    torch::Tensor eqr = torch::mm(e_qr, A_r_);   // [B, k_r]
    torch::Tensor eqv = torch::mm(e_qv, A_e_);   // [B, k_e]

    // 5-mode Tucker contraction, leaving mode 2 (dst) open.
    // W_core_ has shape [k_e, k_r, k_e, k_r, k_e] with named modes p, q, r, s, t.
    // partial_core[b, r] = Σ_{p,q,s,t} W[p,q,r,s,t] * es[b,p] * er[b,q] * eqr[b,s] * eqv[b,t]
    torch::Tensor partial_core = torch::einsum("pqrst,bp,bq,bs,bt->br",
                                               {W_core_, es, er, eqr, eqv});  // [B, k_e]

    // Project partial back to entity embedding dimension
    return torch::mm(partial_core, A_e_.t());  // [B, embedding_dim]
}
