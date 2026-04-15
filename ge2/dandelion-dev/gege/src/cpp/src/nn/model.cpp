#include "nn/model.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#ifdef GEGE_CUDA
#include "common/unique_map_cuda.h"
#include <torch/csrc/cuda/nccl.h>
#endif

#include "configuration/constants.h"
#include "configuration/options.h"
#include "data/samplers/negative.h"
#include "nn/decoders/edge/decoder_methods.h"
#include "nn/layers/embedding/embedding.h"
#include "nn/initialization.h"
#include "nn/model_helpers.h"
#include "reporting/logger.h"

namespace {

bool parse_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }
    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }
    return default_value;
}

int64_t parse_env_int(const char *name, int64_t default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

bool eval_finite_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_EVAL_FINITE_DEBUG", false);
    return enabled;
}

bool eval_chunked_ranks_enabled() {
    static bool enabled = parse_env_flag("GEGE_EVAL_CHUNKED_RANKS", false);
    return enabled;
}

bool emulate_dot_single_relation_enabled() {
    static bool enabled = parse_env_flag("GEGE_EMULATE_DOT_SINGLE_RELATION", false);
    return enabled;
}

bool fixed_buffer_manual_dot_rns_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS", false);
    return enabled;
}

bool fixed_buffer_manual_dot_rns_sanity_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_SANITY", false);
    return enabled;
}

bool fixed_buffer_manual_dot_rns_torch_adagrad_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_TORCH_ADAGRAD", false);
    return enabled;
}

bool fixed_buffer_manual_dot_rns_verify_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_VERIFY", false);
    return enabled;
}

bool fixed_buffer_padded_forward_verify_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_PADDED_FORWARD_VERIFY", false);
    return enabled;
}

bool fixed_buffer_padded_backward_verify_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_PADDED_BACKWARD_VERIFY", false);
    return enabled;
}

bool fixed_buffer_padded_backward_verify_strict_enabled() {
    static bool enabled = parse_env_flag("GEGE_FIXED_BUFFER_PADDED_BACKWARD_VERIFY_STRICT", false);
    return enabled;
}

int64_t fixed_buffer_manual_dot_rns_verify_max() {
    static int64_t max_batches = std::max<int64_t>(parse_env_int("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_VERIFY_MAX", 8), 0);
    return max_batches;
}

int64_t fixed_buffer_padded_forward_verify_max() {
    static int64_t max_batches = std::max<int64_t>(parse_env_int("GEGE_FIXED_BUFFER_PADDED_FORWARD_VERIFY_MAX", 8), 0);
    return max_batches;
}

int64_t fixed_buffer_padded_backward_verify_max() {
    static int64_t max_batches = std::max<int64_t>(parse_env_int("GEGE_FIXED_BUFFER_PADDED_BACKWARD_VERIFY_MAX", 8), 0);
    return max_batches;
}

std::atomic<int64_t> &fixed_buffer_manual_dot_rns_verify_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

std::atomic<int64_t> &fixed_buffer_padded_forward_verify_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

std::atomic<int64_t> &fixed_buffer_padded_backward_verify_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool startup_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_STARTUP_TIMING", false);
    return enabled;
}

int64_t eval_negative_chunk_size() {
    static int64_t chunk_size = std::max<int64_t>(parse_env_int("GEGE_EVAL_NEGATIVE_CHUNK_SIZE", 131072), 1);
    return chunk_size;
}

bool negative_sampler_filtered_for_eval(const shared_ptr<NegativeSampler> &negative_sampler) {
    if (negative_sampler == nullptr) {
        return false;
    }

    if (instance_of<NegativeSampler, NegativeSamplingBase>(negative_sampler)) {
        return std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler)->filtered_;
    }

    if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler)) {
        return std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler)->filtered_;
    }

    return false;
}

int64_t eval_finite_debug_max_logs() {
    static int64_t max_logs = std::max<int64_t>(parse_env_int("GEGE_EVAL_FINITE_DEBUG_MAX_LOGS", 16), 0);
    return max_logs;
}

std::atomic<int64_t> &eval_finite_debug_log_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_log_eval_finite_debug(int64_t &log_id) {
    if (!eval_finite_debug_enabled()) {
        return false;
    }

    int64_t current = eval_finite_debug_log_counter().fetch_add(1);
    if (current >= eval_finite_debug_max_logs()) {
        return false;
    }

    log_id = current;
    return true;
}

bool should_log_eval_chunked_path() {
    static std::atomic<bool> logged{false};
    bool expected = false;
    return logged.compare_exchange_strong(expected, true);
}

std::string tensor_shape_string(const torch::Tensor &tensor) {
    std::ostringstream oss;
    oss << "[";
    for (int64_t dim = 0; dim < tensor.dim(); dim++) {
        if (dim > 0) {
            oss << ", ";
        }
        oss << tensor.size(dim);
    }
    oss << "]";
    return oss.str();
}

shared_ptr<InitConfig> default_relation_init_config(const shared_ptr<ModelConfig> &model_config) {
    if (model_config == nullptr || model_config->encoder == nullptr || model_config->encoder->layers.empty() ||
        model_config->encoder->layers[0].empty() || model_config->encoder->layers[0][0] == nullptr) {
        return nullptr;
    }

    auto first_layer = model_config->encoder->layers[0][0];
    if (first_layer->type != LayerType::EMBEDDING) {
        return nullptr;
    }

    return first_layer->init;
}

void maybe_apply_distmult_relation_init(const shared_ptr<Decoder> &decoder, const shared_ptr<InitConfig> &init_config) {
    auto distmult = std::dynamic_pointer_cast<DistMult>(decoder);
    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder);
    if (distmult == nullptr || edge_decoder == nullptr || !edge_decoder->relations_.defined()) {
        return;
    }

    torch::NoGradGuard no_grad;
    if (emulate_dot_single_relation_enabled() && edge_decoder->relations_.size(0) == 1) {
        edge_decoder->relations_.fill_(1.0f);
        if (edge_decoder->use_inverse_relations_ && edge_decoder->inverse_relations_.defined()) {
            edge_decoder->inverse_relations_.fill_(1.0f);
        }
        SPDLOG_INFO("[relation-init] decoder=DISTMULT single_relation_dot_emulation=1 forcing relation embeddings to ones");
        return;
    }

    if (init_config == nullptr) {
        return;
    }

    auto relation_init = initialize_tensor(init_config,
                                           {edge_decoder->relations_.size(0), edge_decoder->relations_.size(1)},
                                           edge_decoder->relations_.options());
    edge_decoder->relations_.copy_(relation_init);

    if (edge_decoder->use_inverse_relations_ && edge_decoder->inverse_relations_.defined()) {
        auto inverse_relation_init = initialize_tensor(init_config,
                                                       {edge_decoder->inverse_relations_.size(0), edge_decoder->inverse_relations_.size(1)},
                                                       edge_decoder->inverse_relations_.options());
        edge_decoder->inverse_relations_.copy_(inverse_relation_init);
    }
}

std::string tensor_stats_string(const torch::Tensor &tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return "undefined";
    }

    torch::Tensor cpu = tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "shape=" << tensor_shape_string(cpu)
        << " mean=" << cpu.mean().item<double>()
        << " std=" << cpu.std(/*unbiased=*/false).item<double>()
        << " min=" << cpu.min().item<double>()
        << " max=" << cpu.max().item<double>();
    return oss.str();
}

void log_edge_decoder_relation_stats(const char *stage, const shared_ptr<Decoder> &decoder) {
    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder);
    if (edge_decoder == nullptr || !edge_decoder->relations_.defined()) {
        return;
    }

    const char *decoder_name = "EDGE";
    if (std::dynamic_pointer_cast<DistMult>(decoder) != nullptr) {
        decoder_name = "DISTMULT";
    } else if (std::dynamic_pointer_cast<ComplEx>(decoder) != nullptr) {
        decoder_name = "COMPLEX";
    } else if (std::dynamic_pointer_cast<TransE>(decoder) != nullptr) {
        decoder_name = "TRANSE";
    }

    SPDLOG_INFO("[relation-stats][{}] decoder={} relations: {}", stage, decoder_name, tensor_stats_string(edge_decoder->relations_));
    if (edge_decoder->use_inverse_relations_ && edge_decoder->inverse_relations_.defined()) {
        SPDLOG_INFO("[relation-stats][{}] decoder={} inverse_relations: {}",
                    stage, decoder_name, tensor_stats_string(edge_decoder->inverse_relations_));
    }
}

void log_eval_tensor_if_non_finite(const char *stage, int batch_id, const torch::Tensor &tensor) {
    if (!eval_finite_debug_enabled() || !tensor.defined() || tensor.numel() == 0) {
        return;
    }

    torch::Tensor finite = torch::isfinite(tensor);
    if (finite.all().item<bool>()) {
        return;
    }

    int64_t log_id = -1;
    if (!should_log_eval_finite_debug(log_id)) {
        return;
    }

    int64_t invalid_values = (~finite).sum().item<int64_t>();
    int64_t invalid_rows = invalid_values;
    if (tensor.dim() >= 2) {
        invalid_rows = (~finite).reshape({tensor.size(0), -1}).any(1).sum().item<int64_t>();
    }

    SPDLOG_ERROR("[eval-finite-debug][model {}][{}] batch_id={} invalid_values={} invalid_rows={} shape={} device={}",
                 log_id, stage, batch_id, invalid_values, invalid_rows, tensor_shape_string(tensor), tensor.device().str());
}

void copy_named_tensors_to_device(torch::OrderedDict<std::string, torch::Tensor> src, torch::OrderedDict<std::string, torch::Tensor> dst,
                                  torch::Device device) {
    torch::NoGradGuard no_grad;
    for (auto &item : src) {
        dst[item.key()].copy_(item.value().to(device));
    }
}

shared_ptr<GeneralEncoder> clone_encoder_for_device(const shared_ptr<GeneralEncoder> &encoder, torch::Device device) {
    auto cloned = std::make_shared<GeneralEncoder>(encoder->encoder_config_, device, encoder->num_relations_);
    copy_named_tensors_to_device(std::dynamic_pointer_cast<torch::nn::Module>(encoder)->named_parameters(/*recurse=*/true),
                                 std::dynamic_pointer_cast<torch::nn::Module>(cloned)->named_parameters(/*recurse=*/true), device);
    copy_named_tensors_to_device(std::dynamic_pointer_cast<torch::nn::Module>(encoder)->named_buffers(/*recurse=*/true),
                                 std::dynamic_pointer_cast<torch::nn::Module>(cloned)->named_buffers(/*recurse=*/true), device);
    return cloned;
}

void align_broadcast_model_to_device(const shared_ptr<GeneralEncoder> &encoder, const shared_ptr<Decoder> &decoder, torch::Device device) {
    if (encoder != nullptr) {
        encoder->to(device);
        encoder->device_ = device;
        for (auto &stage : encoder->layers_) {
            for (auto &layer : stage) {
                if (layer != nullptr) {
                    layer->device_ = device;
                }
            }
        }
    }

    auto decoder_module = std::dynamic_pointer_cast<torch::nn::Module>(decoder);
    if (decoder_module != nullptr) {
        decoder_module->to(device);
    }

    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder);
    if (edge_decoder != nullptr) {
        edge_decoder->tensor_options_ = edge_decoder->tensor_options_.device(device);
    }
}

torch::Tensor pad_rows(torch::Tensor input, int64_t rows) {
    if (input.size(0) >= rows) {
        return input;
    }
    auto pad = torch::zeros({rows - input.size(0), input.size(1)}, input.options());
    return torch::cat({input, pad}, 0);
}

void accumulate_dot_softmax_side(torch::Tensor grad_unique,
                                 torch::Tensor node_embeddings,
                                 torch::Tensor anchor_ids,
                                 torch::Tensor other_ids,
                                 torch::Tensor neg_ids,
                                 torch::Tensor neg_filter) {
    if (!neg_ids.defined() || neg_ids.numel() == 0) {
        return;
    }

    anchor_ids = anchor_ids.to(torch::kInt64);
    other_ids = other_ids.to(torch::kInt64);
    neg_ids = neg_ids.to(torch::kInt64);

    int64_t batch_size = anchor_ids.numel();
    int64_t num_chunks = neg_ids.size(0);
    int64_t num_negatives = neg_ids.size(1);
    if (batch_size == 0 || num_chunks == 0 || num_negatives == 0) {
        return;
    }

    int64_t dim = node_embeddings.size(1);
    int64_t per_chunk = (batch_size + num_chunks - 1) / num_chunks;
    int64_t padded_batch_size = per_chunk * num_chunks;

    torch::Tensor anchor = node_embeddings.index_select(0, anchor_ids);
    torch::Tensor other = node_embeddings.index_select(0, other_ids);
    torch::Tensor neg_flat = neg_ids.flatten(0, 1);
    torch::Tensor neg_embeddings = node_embeddings.index_select(0, neg_flat).reshape({num_chunks, num_negatives, dim});

    torch::Tensor anchor_padded = pad_rows(anchor, padded_batch_size);
    torch::Tensor anchor_chunks = anchor_padded.reshape({num_chunks, per_chunk, dim});
    torch::Tensor neg_scores = torch::bmm(anchor_chunks, neg_embeddings.transpose(1, 2))
                                   .reshape({padded_batch_size, num_negatives})
                                   .narrow(0, 0, batch_size);
    neg_scores = apply_score_filter(neg_scores, neg_filter);

    torch::Tensor pos_scores = (anchor * other).sum(1, true);
    torch::Tensor max_neg = std::get<0>(neg_scores.max(1, true));
    torch::Tensor max_logits = torch::maximum(pos_scores, max_neg);
    torch::Tensor exp_pos = (pos_scores - max_logits).exp();
    torch::Tensor exp_neg = (neg_scores - max_logits).exp();
    torch::Tensor denom = exp_pos + exp_neg.sum(1, true);
    torch::Tensor grad_pos = exp_pos / denom - 1.0;
    torch::Tensor grad_neg = exp_neg / denom;

    torch::Tensor grad_neg_padded = pad_rows(grad_neg, padded_batch_size);
    torch::Tensor grad_neg_chunks = grad_neg_padded.reshape({num_chunks, per_chunk, num_negatives});
    torch::Tensor anchor_grad_from_neg = torch::bmm(grad_neg_chunks, neg_embeddings)
                                             .reshape({padded_batch_size, dim})
                                             .narrow(0, 0, batch_size);
    torch::Tensor grad_anchor = grad_pos * other + anchor_grad_from_neg;
    torch::Tensor grad_other = grad_pos * anchor;
    torch::Tensor grad_neg_embeddings = torch::bmm(grad_neg_chunks.transpose(1, 2), anchor_chunks)
                                            .reshape({num_chunks * num_negatives, dim});

    grad_unique.index_add_(0, anchor_ids, grad_anchor);
    grad_unique.index_add_(0, other_ids, grad_other);
    grad_unique.index_add_(0, neg_flat, grad_neg_embeddings);
}

bool can_use_manual_dot_rns_update(const shared_ptr<EdgeDecoder> &edge_decoder, const shared_ptr<Batch> &batch,
                                   NegativeSamplingMethod negative_sampling_method, LearningTask learning_task) {
#ifndef GEGE_CUDA
    return false;
#else
    if (!fixed_buffer_manual_dot_rns_enabled()) {
        return false;
    }
    if (learning_task != LearningTask::LINK_PREDICTION || negative_sampling_method != NegativeSamplingMethod::RNS) {
        return false;
    }
    if (!emulate_dot_single_relation_enabled()) {
        return false;
    }
    if (std::dynamic_pointer_cast<DistMult>(edge_decoder) == nullptr) {
        return false;
    }
    if (!batch->edges_.defined() || !(batch->edges_.size(1) == 2 || batch->edges_.size(1) == 3)) {
        return false;
    }
    if (!batch->node_embeddings_.defined() || !batch->node_embeddings_state_.defined() ||
        !batch->unique_node_indices_.defined()) {
        return false;
    }
    if (!batch->node_embeddings_.device().is_cuda() || !batch->node_embeddings_state_.device().is_cuda() ||
        !batch->unique_node_indices_.device().is_cuda()) {
        return false;
    }
    if (batch->node_embeddings_.scalar_type() != torch::kFloat32 || batch->node_embeddings_state_.scalar_type() != torch::kFloat32 ||
        batch->unique_node_indices_.scalar_type() != torch::kInt64) {
        return false;
    }
    if (batch->unique_node_active_mask_.defined() &&
        (batch->unique_node_active_mask_.scalar_type() != torch::kUInt8 || !batch->unique_node_active_mask_.device().is_cuda() ||
         batch->unique_node_active_mask_.numel() != batch->node_embeddings_.size(0))) {
        return false;
    }
    if (!batch->dst_neg_indices_mapping_.defined() || batch->dst_neg_indices_mapping_.numel() == 0) {
        return false;
    }
    if (!batch->dst_neg_indices_mapping_.device().is_cuda()) {
        return false;
    }
    return true;
#endif
}

torch::Tensor remap_fixed_buffer_ids(torch::Tensor ids, torch::Tensor old_to_compact) {
    if (!ids.defined() || ids.numel() == 0) {
        return ids;
    }
    auto original_sizes = ids.sizes().vec();
    torch::Tensor flat = ids.flatten(0, -1).to(torch::kInt64);
    return old_to_compact.index_select(0, flat).reshape(original_sizes).to(ids.scalar_type());
}

std::shared_ptr<Batch> make_compact_active_shadow_batch(const std::shared_ptr<Batch> &batch) {
    torch::Tensor active_mask = batch->unique_node_active_mask_.to(batch->unique_node_indices_.device()).to(torch::kBool);
    torch::Tensor active_rows = torch::nonzero(active_mask).flatten();
    torch::Tensor old_to_compact = active_mask.to(torch::kInt64).cumsum(0) - 1;

    auto shadow = std::make_shared<Batch>(batch->train_);
    shadow->batch_id_ = batch->batch_id_;
    shadow->start_idx_ = batch->start_idx_;
    shadow->batch_size_ = batch->batch_size_;
    shadow->streamed_edge_start_ = batch->streamed_edge_start_;
    shadow->streamed_edge_size_ = batch->streamed_edge_size_;
    shadow->resident_local_lp_direct_ = batch->resident_local_lp_direct_;
    shadow->task_ = batch->task_;
    shadow->device_id_ = batch->device_id_;
    shadow->dense_graph_ = batch->dense_graph_;
    shadow->encoded_uniques_ = batch->encoded_uniques_;
    shadow->node_embeddings_g_ = batch->node_embeddings_g_;
    shadow->qual_embeddings_ = batch->qual_embeddings_;
    shadow->qual_indices_ = batch->qual_indices_;
    shadow->neg_edges_ = batch->neg_edges_;
    shadow->rel_neg_indices_ = batch->rel_neg_indices_;
    shadow->src_neg_indices_ = batch->src_neg_indices_;
    shadow->dst_neg_indices_ = batch->dst_neg_indices_;
    shadow->src_neg_filter_ = batch->src_neg_filter_;
    shadow->dst_neg_filter_ = batch->dst_neg_filter_;
    shadow->unique_node_indices_ = batch->unique_node_indices_.index_select(0, active_rows);
    shadow->unique_node_active_mask_ = torch::Tensor();
    shadow->node_embeddings_ = batch->node_embeddings_.index_select(0, active_rows);
    if (batch->node_embeddings_state_.defined() && batch->node_embeddings_state_.numel() > 0) {
        shadow->node_embeddings_state_ = batch->node_embeddings_state_.index_select(0, active_rows);
    }
    if (batch->node_features_.defined() && batch->node_features_.dim() > 0 &&
        batch->node_features_.size(0) == batch->node_embeddings_.size(0)) {
        shadow->node_features_ = batch->node_features_.index_select(0, active_rows);
    }

    if (batch->edges_.defined() && batch->edges_.dim() == 2 && batch->edges_.size(1) >= 2) {
        torch::Tensor compact_edges = batch->edges_.clone();
        compact_edges.select(1, 0).copy_(old_to_compact.index_select(0, batch->edges_.select(1, 0).to(torch::kInt64)).to(batch->edges_.scalar_type()));
        int64_t dst_col = batch->edges_.size(1) - 1;
        compact_edges.select(1, dst_col).copy_(old_to_compact.index_select(0, batch->edges_.select(1, dst_col).to(torch::kInt64)).to(batch->edges_.scalar_type()));
        shadow->edges_ = compact_edges;
    }

    shadow->dst_neg_indices_mapping_ = remap_fixed_buffer_ids(batch->dst_neg_indices_mapping_, old_to_compact);
    shadow->src_neg_indices_mapping_ = remap_fixed_buffer_ids(batch->src_neg_indices_mapping_, old_to_compact);
    return shadow;
}

double max_abs_diff_or_zero(torch::Tensor lhs, torch::Tensor rhs) {
    if (!lhs.defined() && !rhs.defined()) {
        return 0.0;
    }
    if (!lhs.defined() || !rhs.defined()) {
        return std::numeric_limits<double>::infinity();
    }
    if (lhs.numel() == 0 && rhs.numel() == 0) {
        return 0.0;
    }
    if (lhs.sizes() != rhs.sizes()) {
        return std::numeric_limits<double>::infinity();
    }
    return (lhs - rhs).abs().max().item<double>();
}

void verify_padded_forward_equivalence(Model *model, const std::shared_ptr<Batch> &batch,
                                       const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> &padded_scores) {
    if (!fixed_buffer_padded_forward_verify_enabled() || !batch->unique_node_active_mask_.defined() ||
        !batch->node_embeddings_.defined() || !batch->node_embeddings_.device().is_cuda()) {
        return;
    }
    int64_t verify_id = fixed_buffer_padded_forward_verify_counter().fetch_add(1);
    if (verify_id >= fixed_buffer_padded_forward_verify_max()) {
        return;
    }

    torch::NoGradGuard no_grad;
    auto shadow = make_compact_active_shadow_batch(batch);
    auto compact_scores = model->forward_lp(shadow, true);
    double pos_diff = max_abs_diff_or_zero(std::get<0>(padded_scores), std::get<0>(compact_scores));
    double neg_diff = max_abs_diff_or_zero(std::get<1>(padded_scores), std::get<1>(compact_scores));
    double inv_pos_diff = max_abs_diff_or_zero(std::get<2>(padded_scores), std::get<2>(compact_scores));
    double inv_neg_diff = max_abs_diff_or_zero(std::get<3>(padded_scores), std::get<3>(compact_scores));
    bool match = pos_diff <= 1e-6 && neg_diff <= 1e-6 && inv_pos_diff <= 1e-6 && inv_neg_diff <= 1e-6;
    SPDLOG_INFO("GEGE_FIXED_BUFFER_PADDED_FORWARD_VERIFY check={} batch={} match={} rows={} active_rows={} pos_max_abs={} neg_max_abs={} inv_pos_max_abs={} inv_neg_max_abs={}",
                verify_id, batch->batch_id_, match, batch->node_embeddings_.size(0), shadow->node_embeddings_.size(0),
                pos_diff, neg_diff, inv_pos_diff, inv_neg_diff);
    if (!match) {
        throw GegeRuntimeException("Padded fixed-buffer forward verifier failed");
    }
}

void verify_padded_backward_equivalence(Model *model, const std::shared_ptr<Batch> &batch) {
    if (!fixed_buffer_padded_backward_verify_enabled() || !batch->unique_node_active_mask_.defined() ||
        !batch->node_embeddings_.defined() || !batch->node_embeddings_.grad().defined() ||
        !batch->node_embeddings_.device().is_cuda()) {
        return;
    }
    int64_t verify_id = fixed_buffer_padded_backward_verify_counter().fetch_add(1);
    if (verify_id >= fixed_buffer_padded_backward_verify_max()) {
        return;
    }

    torch::Tensor active_mask = batch->unique_node_active_mask_.to(batch->unique_node_indices_.device()).to(torch::kBool);
    torch::Tensor active_rows = torch::nonzero(active_mask).flatten();
    auto shadow = make_compact_active_shadow_batch(batch);
    shadow->node_embeddings_ = shadow->node_embeddings_.detach().clone();
    shadow->node_embeddings_.requires_grad_();

    auto compact_scores = model->forward_lp(shadow, true);
    torch::Tensor compact_loss;
    if (std::get<3>(compact_scores).defined()) {
        torch::Tensor rhs_loss = model->loss_function_->operator()(std::get<0>(compact_scores), std::get<1>(compact_scores), true);
        torch::Tensor lhs_loss = model->loss_function_->operator()(std::get<2>(compact_scores), std::get<3>(compact_scores), true);
        compact_loss = lhs_loss + rhs_loss;
    } else {
        compact_loss = (*model->loss_function_)(std::get<0>(compact_scores), std::get<1>(compact_scores), true);
    }

    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(model->decoder_);
    std::vector<torch::Tensor> grad_inputs = {shadow->node_embeddings_};
    bool check_rel = edge_decoder != nullptr && edge_decoder->relations_.defined() && edge_decoder->relations_.requires_grad() &&
                     edge_decoder->relations_.grad().defined();
    bool check_inv_rel = edge_decoder != nullptr && edge_decoder->inverse_relations_.defined() && edge_decoder->inverse_relations_.requires_grad() &&
                         edge_decoder->inverse_relations_.grad().defined();
    if (check_rel) {
        grad_inputs.emplace_back(edge_decoder->relations_);
    }
    if (check_inv_rel) {
        grad_inputs.emplace_back(edge_decoder->inverse_relations_);
    }

    std::vector<torch::Tensor> compact_grads =
        torch::autograd::grad({compact_loss}, grad_inputs, {}, /*retain_graph=*/false, /*create_graph=*/false, /*allow_unused=*/false);
    torch::Tensor padded_active_grad = batch->node_embeddings_.grad().index_select(0, active_rows);
    torch::Tensor compact_grad = compact_grads.at(0);
    bool match = torch::allclose(padded_active_grad, compact_grad, 1e-4, 1e-5);
    double max_abs = (padded_active_grad - compact_grad).abs().max().item<double>();
    double rel_max_abs = 0.0;
    double inv_rel_max_abs = 0.0;
    bool rel_match = true;
    bool inv_rel_match = true;
    int64_t grad_idx = 1;
    if (check_rel) {
        torch::Tensor actual_rel_grad = edge_decoder->relations_.grad();
        if (actual_rel_grad.defined()) {
            rel_match = torch::allclose(actual_rel_grad, compact_grads.at(grad_idx), 1e-4, 1e-5);
            rel_max_abs = (actual_rel_grad - compact_grads.at(grad_idx)).abs().max().item<double>();
        } else {
            rel_match = false;
            rel_max_abs = std::numeric_limits<double>::infinity();
        }
        grad_idx++;
    }
    if (check_inv_rel) {
        torch::Tensor actual_inv_rel_grad = edge_decoder->inverse_relations_.grad();
        if (actual_inv_rel_grad.defined()) {
            inv_rel_match = torch::allclose(actual_inv_rel_grad, compact_grads.at(grad_idx), 1e-4, 1e-5);
            inv_rel_max_abs = (actual_inv_rel_grad - compact_grads.at(grad_idx)).abs().max().item<double>();
        } else {
            inv_rel_match = false;
            inv_rel_max_abs = std::numeric_limits<double>::infinity();
        }
    }
    bool all_match = match && rel_match && inv_rel_match;
    SPDLOG_INFO("GEGE_FIXED_BUFFER_PADDED_BACKWARD_VERIFY check={} batch={} match={} active_rows={} grad_max_abs={} rel_match={} rel_max_abs={} inv_rel_match={} inv_rel_max_abs={}",
                verify_id, batch->batch_id_, all_match, active_rows.numel(), max_abs, rel_match, rel_max_abs, inv_rel_match, inv_rel_max_abs);
    if (!all_match && fixed_buffer_padded_backward_verify_strict_enabled()) {
        throw GegeRuntimeException("Padded fixed-buffer backward verifier failed");
    }
}

void manual_dot_rns_update(shared_ptr<Batch> batch, float learning_rate, bool include_src_negatives, torch::Tensor *raw_gradient_out = nullptr) {
#ifdef GEGE_CUDA
    torch::NoGradGuard no_grad;
    torch::Tensor node_embeddings = batch->node_embeddings_.detach();
    torch::Tensor optimizer_state = batch->node_embeddings_state_.detach();
    torch::Tensor grad_unique = torch::zeros_like(node_embeddings);

    torch::Tensor src_ids = batch->edges_.select(1, 0).to(torch::kInt64);
    torch::Tensor dst_ids = batch->edges_.select(1, -1).to(torch::kInt64);

    accumulate_dot_softmax_side(grad_unique, node_embeddings, src_ids, dst_ids,
                                batch->dst_neg_indices_mapping_, batch->dst_neg_filter_);

    if (include_src_negatives && batch->src_neg_indices_mapping_.defined() && batch->src_neg_indices_mapping_.numel() > 0) {
        accumulate_dot_softmax_side(grad_unique, node_embeddings, dst_ids, src_ids,
                                    batch->src_neg_indices_mapping_, batch->src_neg_filter_);
    }

    if (fixed_buffer_manual_dot_rns_sanity_enabled()) {
        if (!torch::isfinite(grad_unique).all().item<bool>()) {
            throw GegeRuntimeException("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS produced non-finite gradients");
        }
    }

    if (raw_gradient_out != nullptr) {
        *raw_gradient_out = grad_unique.detach().clone();
    }

    if (fixed_buffer_manual_dot_rns_torch_adagrad_enabled()) {
        if (batch->unique_node_active_mask_.defined() && batch->unique_node_active_mask_.numel() == grad_unique.size(0)) {
            torch::Tensor active_mask = batch->unique_node_active_mask_.to(grad_unique.device()).to(grad_unique.dtype()).reshape({-1, 1});
            grad_unique = grad_unique * active_mask;
        }
        batch->node_state_update_ = grad_unique.pow(2);
        optimizer_state.add_(batch->node_state_update_);
        batch->node_gradients_ = -learning_rate * (grad_unique / (optimizer_state.sqrt().add_(1e-10)));
    } else if (batch->unique_node_active_mask_.defined() && batch->unique_node_active_mask_.numel() == grad_unique.size(0)) {
        std::tie(batch->node_gradients_, batch->node_state_update_) =
            active_masked_adagrad_cuda(grad_unique, optimizer_state, batch->unique_node_active_mask_, learning_rate);
    } else {
        batch->node_state_update_ = grad_unique.pow(2);
        optimizer_state.add_(batch->node_state_update_);
        batch->node_gradients_ = -learning_rate * (grad_unique / (optimizer_state.sqrt().add_(1e-10)));
    }
    batch->node_embeddings_state_ = torch::Tensor();
#else
    (void)batch;
    (void)learning_rate;
#endif
}

shared_ptr<Batch> clone_manual_dot_rns_verify_batch(const shared_ptr<Batch> &batch) {
    auto shadow = std::make_shared<Batch>(batch->train_);
    shadow->batch_id_ = batch->batch_id_;
    shadow->batch_size_ = batch->batch_size_;
    shadow->task_ = batch->task_;
    shadow->device_id_ = batch->device_id_;
    shadow->resident_local_lp_direct_ = false;

    shadow->edges_ = batch->edges_;
    shadow->unique_node_indices_ = batch->unique_node_indices_;
    shadow->unique_node_active_mask_ = batch->unique_node_active_mask_;
    shadow->src_neg_indices_mapping_ = batch->src_neg_indices_mapping_;
    shadow->dst_neg_indices_mapping_ = batch->dst_neg_indices_mapping_;
    shadow->src_neg_filter_ = batch->src_neg_filter_;
    shadow->dst_neg_filter_ = batch->dst_neg_filter_;
    shadow->qual_embeddings_ = batch->qual_embeddings_.defined() ? batch->qual_embeddings_.detach().clone() : torch::Tensor();
    shadow->qual_embeddings_state_ = batch->qual_embeddings_state_.defined() ? batch->qual_embeddings_state_.detach().clone() : torch::Tensor();
    shadow->qual_indices_ = batch->qual_indices_;
    shadow->node_features_ = batch->node_features_;
    shadow->dense_graph_ = batch->dense_graph_;

    shadow->node_embeddings_ = batch->node_embeddings_.detach().clone();
    shadow->node_embeddings_state_ = batch->node_embeddings_state_.detach().clone();
    return shadow;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reference_dot_rns_autograd_update(Model *model, const shared_ptr<Batch> &batch) {
    auto shadow = clone_manual_dot_rns_verify_batch(batch);
    shadow->node_embeddings_.requires_grad_();

    auto all_scores = model->forward_lp(shadow, true);
    torch::Tensor pos_scores = std::get<0>(all_scores);
    torch::Tensor neg_scores = std::get<1>(all_scores);
    torch::Tensor inv_pos_scores = std::get<2>(all_scores);
    torch::Tensor inv_neg_scores = std::get<3>(all_scores);

    torch::Tensor loss;
    if (inv_neg_scores.defined()) {
        torch::Tensor rhs_loss = model->loss_function_->operator()(pos_scores, neg_scores, true);
        torch::Tensor lhs_loss = model->loss_function_->operator()(inv_pos_scores, inv_neg_scores, true);
        loss = lhs_loss + rhs_loss;
    } else {
        loss = (*model->loss_function_)(pos_scores, neg_scores, true);
    }
    loss.backward();

    torch::NoGradGuard no_grad;
    torch::Tensor raw_gradients = shadow->node_embeddings_.grad().detach().clone();
    torch::Tensor gradients = raw_gradients;
    if (shadow->unique_node_active_mask_.defined() && shadow->unique_node_active_mask_.numel() == gradients.size(0)) {
        torch::Tensor active_mask = shadow->unique_node_active_mask_.to(gradients.device()).to(gradients.dtype()).reshape({-1, 1});
        gradients = gradients * active_mask;
        raw_gradients = raw_gradients * active_mask;
    }

    torch::Tensor optimizer_state = shadow->node_embeddings_state_.detach().clone();
    torch::Tensor state_update = gradients.pow(2);
    optimizer_state.add_(state_update);
    torch::Tensor adjusted_gradients = -model->sparse_lr_ * (gradients / (optimizer_state.sqrt().add_(1e-10)));
    model->clear_grad();
    return std::make_tuple(raw_gradients, adjusted_gradients, state_update);
}

void verify_manual_dot_rns_update(Model *model, const shared_ptr<Batch> &batch, bool include_src_negatives) {
#ifdef GEGE_CUDA
    if (!fixed_buffer_manual_dot_rns_verify_enabled()) {
        return;
    }
    int64_t verify_id = fixed_buffer_manual_dot_rns_verify_counter().fetch_add(1);
    if (verify_id >= fixed_buffer_manual_dot_rns_verify_max()) {
        return;
    }

    auto manual_shadow = clone_manual_dot_rns_verify_batch(batch);
    torch::Tensor ref_raw_gradients;
    torch::Tensor ref_gradients;
    torch::Tensor ref_state_update;
    std::tie(ref_raw_gradients, ref_gradients, ref_state_update) = reference_dot_rns_autograd_update(model, batch);
    torch::Tensor manual_raw_gradients;
    manual_dot_rns_update(manual_shadow, model->sparse_lr_, include_src_negatives, &manual_raw_gradients);

    bool raw_gradients_match = torch::allclose(manual_raw_gradients, ref_raw_gradients, 1e-4, 1e-7);
    bool gradients_match = torch::allclose(manual_shadow->node_gradients_, ref_gradients, 1e-4, 1e-5);
    bool state_update_match = torch::allclose(manual_shadow->node_state_update_, ref_state_update, 1e-4, 1e-5);
    if (!raw_gradients_match || !gradients_match || !state_update_match) {
        torch::Tensor raw_abs_diff = (manual_raw_gradients - ref_raw_gradients).abs();
        torch::Tensor grad_abs_diff = (manual_shadow->node_gradients_ - ref_gradients).abs();
        torch::Tensor state_abs_diff = (manual_shadow->node_state_update_ - ref_state_update).abs();
        double raw_grad_max_abs = raw_abs_diff.max().item<double>();
        double grad_max_abs = grad_abs_diff.max().item<double>();
        double state_max_abs = state_abs_diff.max().item<double>();
        double grad_negated_max_abs = (manual_shadow->node_gradients_ + ref_gradients).abs().max().item<double>();
        double ref_raw_grad_max = ref_raw_gradients.abs().max().item<double>();
        double manual_raw_grad_max = manual_raw_gradients.abs().max().item<double>();
        double ref_grad_max = ref_gradients.abs().max().item<double>();
        double manual_grad_max = manual_shadow->node_gradients_.abs().max().item<double>();
        int64_t flat_idx = raw_abs_diff.flatten().argmax().item<int64_t>();
        int64_t diff_row = flat_idx / ref_raw_gradients.size(1);
        int64_t diff_col = flat_idx - diff_row * ref_raw_gradients.size(1);
        int64_t unique_id = batch->unique_node_indices_.defined() && diff_row < batch->unique_node_indices_.numel()
                                ? batch->unique_node_indices_.select(0, diff_row).item<int64_t>()
                                : -1;
        uint8_t active = batch->unique_node_active_mask_.defined() && diff_row < batch->unique_node_active_mask_.numel()
                             ? batch->unique_node_active_mask_.select(0, diff_row).item<uint8_t>()
                             : uint8_t{255};
        double ref_raw_value = ref_raw_gradients.index({diff_row, diff_col}).item<double>();
        double manual_raw_value = manual_raw_gradients.index({diff_row, diff_col}).item<double>();
        double ref_value = ref_gradients.index({diff_row, diff_col}).item<double>();
        double manual_value = manual_shadow->node_gradients_.index({diff_row, diff_col}).item<double>();
        double ref_state_value = ref_state_update.index({diff_row, diff_col}).item<double>();
        double manual_state_value = manual_shadow->node_state_update_.index({diff_row, diff_col}).item<double>();
        SPDLOG_ERROR("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_VERIFY failed check={} batch={} include_src_negatives={} raw_match={} gradients_match={} state_update_match={} raw_grad_max_abs={} grad_max_abs={} grad_negated_max_abs={} state_max_abs={} ref_raw_grad_max={} manual_raw_grad_max={} ref_grad_max={} manual_grad_max={} rows={} first_row={} first_col={} unique_id={} active={} ref_raw_grad={} manual_raw_grad={} ref_grad={} manual_grad={} ref_state_update={} manual_state_update={}",
                     verify_id, batch->batch_id_, include_src_negatives, raw_gradients_match, gradients_match, state_update_match,
                     raw_grad_max_abs, grad_max_abs, grad_negated_max_abs, state_max_abs, ref_raw_grad_max, manual_raw_grad_max,
                     ref_grad_max, manual_grad_max, ref_gradients.size(0), diff_row, diff_col, unique_id, static_cast<int>(active),
                     ref_raw_value, manual_raw_value, ref_value, manual_value, ref_state_value, manual_state_value);
        throw GegeRuntimeException("Manual Dot/RNS verifier failed");
    }

    SPDLOG_INFO("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS_VERIFY check={} batch={} passed rows={} include_src_negatives={}",
                verify_id, batch->batch_id_, ref_gradients.size(0), include_src_negatives);
#else
    (void)model;
    (void)batch;
    (void)include_src_negatives;
#endif
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
resident_local_forward_lp(shared_ptr<EdgeDecoder> edge_decoder,
                          NegativeSamplingMethod negative_sampling_method,
                          shared_ptr<Batch> batch) {
    if (negative_sampling_method != NegativeSamplingMethod::RNS) {
        throw GegeRuntimeException("Resident-local LP direct path currently supports RNS only");
    }

    bool is_nary = (batch->edges_.size(1) == 4 || batch->edges_.size(1) == 5);
    bool has_relations;
    if (batch->edges_.size(1) == 3 || is_nary) {
        has_relations = true;
    } else if (batch->edges_.size(1) == 2) {
        has_relations = false;
    } else {
        throw TensorSizeMismatchException(batch->edges_, "Edge list must be a 2, 3, 4, or 5 column tensor");
    }

    torch::Tensor pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor inv_neg_scores;

    auto all_pos_embeddings = prepare_pos_embeddings(edge_decoder, batch->edges_, batch->resident_src_embeddings_,
                                                     batch->resident_dst_embeddings_, has_relations, batch->qual_embeddings_);
    torch::Tensor adjusted_src_embeddings = std::get<0>(all_pos_embeddings);
    torch::Tensor adjusted_dst_embeddings = std::get<1>(all_pos_embeddings);

    if (has_relations) {
        pos_scores = edge_decoder->compute_scores(adjusted_src_embeddings, batch->resident_dst_embeddings_);
        neg_scores = edge_decoder->compute_scores(adjusted_src_embeddings, batch->resident_dst_neg_embeddings_);
        if (!is_nary && edge_decoder->use_inverse_relations_) {
            inv_pos_scores = edge_decoder->compute_scores(adjusted_dst_embeddings, batch->resident_src_embeddings_);
            inv_neg_scores = edge_decoder->compute_scores(adjusted_dst_embeddings, batch->resident_src_neg_embeddings_);
        }
    } else {
        pos_scores = edge_decoder->compute_scores(adjusted_src_embeddings, batch->resident_dst_embeddings_);
        neg_scores = edge_decoder->compute_scores(adjusted_src_embeddings, batch->resident_dst_neg_embeddings_);
    }

    if (pos_scores.defined() && neg_scores.defined() && pos_scores.size(0) != neg_scores.size(0)) {
        int64_t new_size = neg_scores.size(0) - pos_scores.size(0);
        torch::nn::functional::PadFuncOptions options({0, new_size});
        pos_scores = torch::nn::functional::pad(pos_scores, options);
        if (inv_pos_scores.defined()) {
            inv_pos_scores = torch::nn::functional::pad(inv_pos_scores, options);
        }
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

}  // namespace

Model::Model(shared_ptr<GeneralEncoder> encoder, shared_ptr<Decoder> decoder, shared_ptr<LossFunction> loss, shared_ptr<Reporter> reporter,
             std::vector<shared_ptr<Optimizer>> optimizers)
    : device_(torch::Device(torch::kCPU)) {
    encoder_ = encoder;
    decoder_ = decoder;
    loss_function_ = loss;
    reporter_ = reporter;
    optimizers_ = optimizers;
    learning_task_ = decoder_->learning_task_;

    if (reporter_ == nullptr) {
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            reporter_ = std::make_shared<LinkPredictionReporter>();
            reporter_->addMetric(std::make_shared<MeanRankMetric>());
            reporter_->addMetric(std::make_shared<MeanReciprocalRankMetric>());
            reporter_->addMetric(std::make_shared<HitskMetric>(1));
            reporter_->addMetric(std::make_shared<HitskMetric>(3));
            reporter_->addMetric(std::make_shared<HitskMetric>(5));
            reporter_->addMetric(std::make_shared<HitskMetric>(10));
            reporter_->addMetric(std::make_shared<HitskMetric>(50));
            reporter_->addMetric(std::make_shared<HitskMetric>(100));
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            reporter_ = std::make_shared<NodeClassificationReporter>();
            reporter_->addMetric(std::make_shared<CategoricalAccuracyMetric>());
        } else {
            throw GegeRuntimeException("Reporter must be specified for this learning task.");
        }
    }

    if (encoder_ != nullptr) {
        register_module("encoder", std::dynamic_pointer_cast<torch::nn::Module>(encoder_));
    }

    if (decoder_ != nullptr) {
        register_module("decoder", std::dynamic_pointer_cast<torch::nn::Module>(decoder_));
    }
}

void Model::clear_grad() {
#pragma omp parallel for
    for (int i = 0; i < optimizers_.size(); i++) {
        optimizers_[i]->clear_grad();
    }

    if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
#pragma omp parallel for
        for (int i = 0; i < optimizers_g_.size(); i++) {
            optimizers_g_[i]->clear_grad();
        }
    }
}

void Model::clear_grad_all() {
    for (int i = 0; i < device_models_.size(); i++) {
        device_models_[i]->clear_grad();
    }
}

void Model::step() {
#pragma omp parallel for
    for (int i = 0; i < optimizers_.size(); i++) {
        optimizers_[i]->step();
    }
}

void Model::step_g() {
#pragma omp parallel for
    for (int i = 0; i < optimizers_g_.size(); i++) {
        optimizers_g_[i]->step();
    }
}

void Model::step_all() {
    for (int i = 0; i < device_models_.size(); i++) {
        // SPDLOG_INFO("Model::step_all device {}", i);
        device_models_[i]->step();
    }
}

void Model::save(std::string directory) {
    SPDLOG_INFO("Model::save");
    log_edge_decoder_relation_stats("save_pre", decoder_);
    string model_filename = directory + PathConstants::model_file;
    string model_state_filename = directory + PathConstants::model_state_file;
    string model_meta_filename = directory + PathConstants::model_config_file;

    torch::serialize::OutputArchive model_archive;
    torch::serialize::OutputArchive state_archive;

    std::dynamic_pointer_cast<torch::nn::Module>(encoder_)->save(model_archive);

    if (decoder_ != nullptr) {
        for(auto& model : device_models_) {
            torch::serialize::OutputArchive model_archive;
            std::dynamic_pointer_cast<torch::nn::Module>(model->decoder_)->save(model_archive);
            model_archive.save_to(model_filename + "_" + std::to_string(model->device_.index()));
        }
    }

    // Outputs each optimizer as a <K, V> pair, where key is the loop counter and value
    // is the optimizer itself. in Model::load, Optimizer::load is called on each key.
    int32_t count = 0;
    for(auto& model : device_models_) {
        for (int i = 0; i < model->optimizers_.size(); i++) {
            torch::serialize::OutputArchive optim_archive;
            model->optimizers_[i]->save(optim_archive);
            state_archive.write(std::to_string(count ++), optim_archive);
        }
    }
    state_archive.save_to(model_state_filename);
}

void Model::load(std::string directory, bool train) {
    string model_filename = directory + PathConstants::model_file;
    string model_state_filename = directory + PathConstants::model_state_file;
    string per_device_model_filename =
            model_filename + "_" + std::to_string(device_.has_index() ? device_.index() : 0);

    torch::serialize::InputArchive model_archive;
    torch::serialize::InputArchive state_archive;

    const bool has_combined_model_file = static_cast<bool>(std::ifstream(model_filename));
    const bool has_per_device_model_file = static_cast<bool>(std::ifstream(per_device_model_filename));

    if (has_combined_model_file) {
        model_archive.load_from(model_filename);
    } else if (decoder_ != nullptr && has_per_device_model_file) {
        model_archive.load_from(per_device_model_filename);
    } else {
        throw std::runtime_error(
                "Unable to find compatible model checkpoint. Expected " + model_filename +
                " or " + per_device_model_filename);
    }

    if (train) {
        state_archive.load_from(model_state_filename);
    }

    int optimizer_idx = 0;
    for (auto key : state_archive.keys()) {
        torch::serialize::InputArchive tmp_state_archive;
        state_archive.read(key, tmp_state_archive);
        // optimizers have already been created as part of initModelFromConfig
        optimizers_[optimizer_idx++]->load(tmp_state_archive);
    }

    if (has_combined_model_file) {
        std::dynamic_pointer_cast<torch::nn::Module>(encoder_)->load(model_archive);
    }

    if (decoder_ != nullptr) {
        std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->load(model_archive);
    }

    log_edge_decoder_relation_stats(train ? "load_train" : "load_eval", decoder_);
}

void Model::all_reduce_rel() {
    SPDLOG_INFO("all_reduce_rel");
    torch::NoGradGuard no_grad;
    int num_gpus = device_models_.size();
    for (int i = 0; i < named_parameters().keys().size(); i++) {
        string key = named_parameters().keys()[i];

        std::vector<torch::Tensor> input_rels(num_gpus);
        for (int j = 0; j < num_gpus; j++) {
            input_rels[j] = (device_models_[j]->named_parameters()[key]);
            input_rels[j].copy_(input_rels[j].divide(num_gpus));
        }

#ifdef GEGE_CUDA
        torch::cuda::nccl::all_reduce(input_rels, input_rels);
#endif
    }

}

void Model::all_reduce(const std::vector<int64_t> &grad_scales) {
    torch::NoGradGuard no_grad;
    int num_gpus = device_models_.size();
    for (int i = 0; i < named_parameters().keys().size(); i++) {
        string key = named_parameters().keys()[i];

        int32_t device_finish = 0;
        std::vector<torch::Tensor> input_gradients(num_gpus);
        for (int j = 0; j < num_gpus; j++) {
            if (!device_models_[j]->named_parameters()[key].mutable_grad().defined()) {
                device_finish ++;
                device_models_[j]->named_parameters()[key].mutable_grad() = torch::zeros_like(device_models_[j]->named_parameters()[key]);
            }
            input_gradients[j] = (device_models_[j]->named_parameters()[key].mutable_grad());
            double grad_scale = static_cast<double>(num_gpus);
            if (!grad_scales.empty() && j < grad_scales.size()) {
                grad_scale *= std::max<int64_t>(grad_scales[j], 1);
            }
            input_gradients[j].copy_(input_gradients[j].divide(grad_scale));

        }

#ifdef GEGE_CUDA
        torch::cuda::nccl::all_reduce(input_gradients, input_gradients);
#endif
        
    }

    step_all();
    clear_grad_all();
}

void Model::setup_optimizers(shared_ptr<ModelConfig> model_config) {
    if (model_config->dense_optimizer == nullptr) {
        throw UnexpectedNullPtrException();
    }

    // need to assign named parameters to each optimizer
    torch::OrderedDict<shared_ptr<OptimizerConfig>, torch::OrderedDict<std::string, torch::Tensor>> param_map;

    {
        torch::OrderedDict<std::string, torch::Tensor> empty_dict;
        param_map.insert(model_config->dense_optimizer, empty_dict);
    }

    // get optimizers we need to keep track of for the encoder
    for (auto module_name : encoder_->named_modules().keys()) {
        if (module_name.empty()) {
            continue;
        }
        auto layer = std::dynamic_pointer_cast<Layer>(encoder_->named_modules()[module_name]);
        if (layer->config_->optimizer == nullptr) {
            for (auto param_name : layer->named_parameters().keys()) {
                param_map[model_config->dense_optimizer].insert(module_name + "_" + param_name, layer->named_parameters()[param_name]);
            }
        } else {
            if (!param_map.contains(layer->config_->optimizer)) {
                torch::OrderedDict<std::string, torch::Tensor> empty_dict;
                param_map.insert(layer->config_->optimizer, empty_dict);
            }

            for (auto param_name : layer->named_parameters().keys()) {
                param_map[layer->config_->optimizer].insert(module_name + "_" + param_name, layer->named_parameters()[param_name]);
            }
        }
    }

    for (auto key : std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->named_parameters().keys()) {
        param_map[model_config->dense_optimizer].insert(key, std::dynamic_pointer_cast<torch::nn::Module>(decoder_)->named_parameters()[key]);
    }

    for (auto key : param_map.keys()) {
        switch (key->type) {
            case OptimizerType::SGD: {
                optimizers_.emplace_back(std::make_shared<SGDOptimizer>(param_map[key], key->options->learning_rate));
                break;
            }
            case OptimizerType::ADAGRAD: {
                optimizers_.emplace_back(std::make_shared<AdagradOptimizer>(param_map[key], std::dynamic_pointer_cast<AdagradOptions>(key->options)));
                break;
            }
            case OptimizerType::ADAM: {
                optimizers_.emplace_back(std::make_shared<AdamOptimizer>(param_map[key], std::dynamic_pointer_cast<AdamOptions>(key->options)));
                break;
            }
            default:
                throw std::invalid_argument("Unrecognized optimizer type");
        }
    }

    if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
        for (auto key : param_map.keys()) {
            switch (key->type) {
                case OptimizerType::SGD: {
                    optimizers_g_.emplace_back(std::make_shared<SGDOptimizer>(param_map[key], key->options->learning_rate));
                    break;
                }
                case OptimizerType::ADAGRAD: {
                    optimizers_g_.emplace_back(std::make_shared<AdagradOptimizer>(param_map[key], std::dynamic_pointer_cast<AdagradOptions>(key->options)));
                    break;
                }
                case OptimizerType::ADAM: {
                    optimizers_g_.emplace_back(std::make_shared<AdamOptimizer>(param_map[key], std::dynamic_pointer_cast<AdamOptions>(key->options)));
                    break;
                }
                default:
                    throw std::invalid_argument("Unrecognized optimizer type");
            }
        }
    }
}

int64_t Model::get_base_embedding_dim() {
    int max_offset = 0;
    int size = 0;

    for (auto stage : encoder_->layers_) {
        for (auto layer : stage) {
            if (layer->config_->type == LayerType::EMBEDDING) {
                int offset = std::dynamic_pointer_cast<EmbeddingLayer>(layer)->offset_;

                if (size == 0) {
                    size = layer->config_->output_dim;
                }

                if (offset > max_offset) {
                    max_offset = offset;
                    size = layer->config_->output_dim;
                }
            }
        }
    }

    return max_offset + size;
}

bool Model::has_embeddings() { return encoder_->has_embeddings_; }

torch::Tensor Model::forward_nc(at::optional<torch::Tensor> node_embeddings, at::optional<torch::Tensor> node_features, DENSEGraph dense_graph, bool train) {
    torch::Tensor encoded_nodes = encoder_->forward(node_embeddings, node_features, dense_graph, train);
    torch::Tensor y_pred = std::dynamic_pointer_cast<NodeDecoder>(decoder_)->forward(encoded_nodes);
    return y_pred;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Model::forward_lp(shared_ptr<Batch> batch, bool train) {
    if (!train) {
        log_eval_tensor_if_non_finite("node_embeddings", batch->batch_id_, batch->node_embeddings_);
        log_eval_tensor_if_non_finite("node_features", batch->batch_id_, batch->node_features_);
    }

    // call proper decoder
    torch::Tensor pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor inv_neg_scores;

    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder_);
    if (batch->resident_local_lp_direct_) {
        std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) =
            resident_local_forward_lp(edge_decoder, negative_sampling_method_, batch);
        if (neg_scores.defined()) {
            neg_scores = apply_score_filter(neg_scores, batch->dst_neg_filter_);
        }
        if (inv_neg_scores.defined()) {
            inv_neg_scores = apply_score_filter(inv_neg_scores, batch->src_neg_filter_);
        }
        return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
    }

    torch::Tensor encoded_nodes = encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, train);
    if (!train) {
        log_eval_tensor_if_non_finite("encoded_nodes", batch->batch_id_, encoded_nodes);
    }

    if (edge_decoder->decoder_method_ == EdgeDecoderMethod::ONLY_POS) {
        std::tie(pos_scores, inv_pos_scores) = only_pos_forward(edge_decoder, batch->edges_, encoded_nodes, batch->qual_embeddings_);
    } else if (edge_decoder->decoder_method_ == EdgeDecoderMethod::POS_AND_NEG) {
        throw GegeRuntimeException("Decoder method currently unsupported.");
        std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) = neg_and_pos_forward(edge_decoder, batch->edges_, batch->neg_edges_, encoded_nodes);
    } else if (edge_decoder->decoder_method_ == EdgeDecoderMethod::CORRUPT_NODE) {
        if (train) {
            std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) =
                mod_node_corrupt_forward(negative_sampling_method_, negative_sampling_selected_ratio_, negative_sampler_, edge_decoder, batch->edges_, encoded_nodes,
                                         batch->dst_neg_indices_mapping_, batch->src_neg_indices_mapping_, batch->node_embeddings_g_,
                                         batch->qual_embeddings_);
        } else {  // evalutate
            std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) =
                node_corrupt_forward(edge_decoder, batch->edges_, encoded_nodes, batch->dst_neg_indices_mapping_, batch->src_neg_indices_mapping_,
                                     batch->qual_embeddings_);
        }
    } else if (edge_decoder->decoder_method_ == EdgeDecoderMethod::CORRUPT_REL) {
        throw GegeRuntimeException("Decoder method currently unsupported.");
        std::tie(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores) =
            rel_corrupt_forward(edge_decoder, batch->edges_, encoded_nodes, batch->rel_neg_indices_);
    } else {
        throw GegeRuntimeException("Unsupported encoder method");
    }

    if (neg_scores.defined()) {
        neg_scores = apply_score_filter(neg_scores, batch->dst_neg_filter_);
    }

    if (inv_neg_scores.defined()) {
        inv_neg_scores = apply_score_filter(inv_neg_scores, batch->src_neg_filter_);
    }

    if (!train) {
        log_eval_tensor_if_non_finite("pos_scores", batch->batch_id_, pos_scores);
        log_eval_tensor_if_non_finite("neg_scores", batch->batch_id_, neg_scores);
        log_eval_tensor_if_non_finite("inv_pos_scores", batch->batch_id_, inv_pos_scores);
        log_eval_tensor_if_non_finite("inv_neg_scores", batch->batch_id_, inv_neg_scores);
    }

    return std::forward_as_tuple(pos_scores, neg_scores, inv_pos_scores, inv_neg_scores);
}

void Model::train_batch(shared_ptr<Batch> batch, bool call_step) {
    if (call_step) {
        clear_grad();
    }

    auto manual_edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder_);
    if (can_use_manual_dot_rns_update(manual_edge_decoder, batch, negative_sampling_method_, learning_task_)) {
        bool has_relations = batch->edges_.size(1) == 3;
        bool include_src_negatives = has_relations && manual_edge_decoder->use_inverse_relations_;
        verify_manual_dot_rns_update(this, batch, include_src_negatives);
        manual_dot_rns_update(batch, sparse_lr_, include_src_negatives);
        return;
    }

    if (batch->node_embeddings_.defined()) {
        batch->node_embeddings_.requires_grad_();
    }
    if (batch->resident_local_lp_direct_) {
        if (batch->resident_src_embeddings_.defined()) batch->resident_src_embeddings_.requires_grad_();
        if (batch->resident_dst_embeddings_.defined()) batch->resident_dst_embeddings_.requires_grad_();
        if (batch->resident_src_neg_embeddings_.defined()) batch->resident_src_neg_embeddings_.requires_grad_();
        if (batch->resident_dst_neg_embeddings_.defined()) batch->resident_dst_neg_embeddings_.requires_grad_();
    }

    // Arity-4: enable gradient tracking for qualifier value embeddings
    if (batch->qual_embeddings_.defined()) {
        batch->qual_embeddings_.requires_grad_();
    }

    torch::Tensor loss;

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        auto all_scores = forward_lp(batch, true);
        verify_padded_forward_equivalence(this, batch, all_scores);

        torch::Tensor pos_scores = std::get<0>(all_scores);
        torch::Tensor neg_scores = std::get<1>(all_scores);
        torch::Tensor inv_pos_scores = std::get<2>(all_scores);
        torch::Tensor inv_neg_scores = std::get<3>(all_scores);

        if (inv_neg_scores.defined()) {
            torch::Tensor rhs_loss = loss_function_->operator()(pos_scores, neg_scores, true);
            torch::Tensor lhs_loss = loss_function_->operator()(inv_pos_scores, inv_neg_scores, true);
            loss = lhs_loss + rhs_loss;
        } else {
            loss = (*loss_function_)(pos_scores, neg_scores, true);
        }
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        torch::Tensor y_pred = forward_nc(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, true);
        loss = (*loss_function_)(y_pred, batch->node_labels_.to(torch::kInt64), false);
    } else {
        throw GegeRuntimeException("Unsupported learning task for training");
    }

    loss.backward();
    verify_padded_backward_equivalence(this, batch);

    if (call_step) {
        step();
    }

    if (batch->resident_local_lp_direct_) {
        batch->accumulateResidentLocalGradients(sparse_lr_);
    } else if (batch->node_embeddings_.defined()) {
        batch->accumulateGradients(sparse_lr_);
    }

    if (negative_sampling_method_ == NegativeSamplingMethod::GAN && learning_task_ == LearningTask::LINK_PREDICTION) {
        if (batch->edges_.defined() && batch->edges_.dim() == 2 && batch->edges_.size(1) >= 4) {
            throw GegeRuntimeException("GAN negative sampling is not yet supported for n-ary edge tensors.");
        }
        if (batch->node_embeddings_g_.defined()) {
            batch->node_embeddings_g_.requires_grad_();
        }

        auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder_);
        torch::Tensor encoded_nodes = encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, true);
        auto rewards = get_rewards(edge_decoder, batch->edges_, encoded_nodes, batch->dst_neg_indices_mapping_, batch->src_neg_indices_mapping_, batch->qual_embeddings_);
        torch::Tensor reward = std::get<0>(rewards);
        torch::Tensor inv_reward = std::get<1>(rewards);
        torch::Tensor loss_g = forward_g(edge_decoder, batch->edges_, batch->node_embeddings_g_, batch->dst_neg_indices_mapping_, batch->src_neg_indices_mapping_, reward, inv_reward, batch->qual_embeddings_);

        loss_g.backward();
        step_g();

        if (batch->node_embeddings_g_.defined()) {
            batch->accumulateGradientsG(sparse_lr_);
        }
    }
}

void Model::evaluate_batch(shared_ptr<Batch> batch) {
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder_);
        bool filtered_eval = negative_sampler_filtered_for_eval(negative_sampler_);
        if (edge_decoder != nullptr && edge_decoder->decoder_method_ == EdgeDecoderMethod::CORRUPT_NODE && filtered_eval &&
            eval_chunked_ranks_enabled() && batch->dst_neg_indices_mapping_.defined()) {
            if (should_log_eval_chunked_path()) {
                SPDLOG_INFO("Link prediction evaluation is using chunked exact filtered ranks; batch_size={} negative_chunk_size={}",
                            batch->edges_.size(0), eval_negative_chunk_size());
            }
            torch::Tensor encoded_nodes = encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
            auto chunked_ranks = node_corrupt_ranks_chunked(edge_decoder, batch->edges_, encoded_nodes, batch->dst_neg_indices_mapping_, batch->dst_neg_filter_,
                                                            batch->src_neg_indices_mapping_, batch->src_neg_filter_, batch->qual_embeddings_,
                                                            eval_negative_chunk_size());
            auto reporter = std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_);
            reporter->addRanks(std::get<0>(chunked_ranks));
            if (std::get<1>(chunked_ranks).defined()) {
                reporter->addRanks(std::get<1>(chunked_ranks));
            }
            return;
        }

        auto all_scores = forward_lp(batch, false);
        torch::Tensor pos_scores = std::get<0>(all_scores);
        torch::Tensor neg_scores = std::get<1>(all_scores);
        torch::Tensor inv_pos_scores = std::get<2>(all_scores);
        torch::Tensor inv_neg_scores = std::get<3>(all_scores);

        if (neg_scores.defined()) {
            std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(pos_scores, neg_scores, batch->edges_);
        }

        if (inv_neg_scores.defined()) {
            std::dynamic_pointer_cast<LinkPredictionReporter>(reporter_)->addResult(inv_pos_scores, inv_neg_scores, batch->edges_);
        }
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        torch::Tensor y_pred = forward_nc(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, false);
        torch::Tensor labels = batch->node_labels_;

        std::dynamic_pointer_cast<NodeClassificationReporter>(reporter_)->addResult(labels, y_pred);

    } else {
        throw GegeRuntimeException("Unsupported learning task for evaluation");
    }
}

void Model::broadcast(std::vector<torch::Device> devices, shared_ptr<ModelConfig> model_config) {
    bool log_startup_timing = startup_timing_enabled();
    int i = 0;
    for (auto device : devices) {
        SPDLOG_INFO("Broadcast to GPU {}", device.index());
        if (device != device_) {
            if (log_startup_timing) {
                SPDLOG_INFO("[startup-timing][Model::broadcast] begin clone slot={} device={}", i, device.str());
            }
            shared_ptr<GeneralEncoder> encoder = clone_encoder_for_device(encoder_, device);
            shared_ptr<Decoder> decoder = decoder_clone_helper(decoder_, device);
            align_broadcast_model_to_device(encoder, decoder, device);
            if (log_startup_timing) {
                SPDLOG_INFO("[startup-timing][Model::broadcast] cloned encoder/decoder slot={} device={}", i, device.str());
            }
            device_models_[i] = std::make_shared<Model>(encoder, decoder, loss_function_, reporter_);
            device_models_[i]->device_ = device;
            device_models_[i]->negative_sampling_method_ = negative_sampling_method_;
            device_models_[i]->negative_sampling_selected_ratio_ = negative_sampling_selected_ratio_;
            if (log_startup_timing) {
                SPDLOG_INFO("[startup-timing][Model::broadcast] model wrapper created slot={} device={}", i, device.str());
            }
            if (!optimizers_.empty()) {
                if (log_startup_timing) {
                    SPDLOG_INFO("[startup-timing][Model::broadcast] setup_optimizers begin slot={} device={}", i, device.str());
                }
                device_models_[i]->setup_optimizers(model_config);
                if (log_startup_timing) {
                    SPDLOG_INFO("[startup-timing][Model::broadcast] setup_optimizers end slot={} device={} optimizers={}",
                                i, device.str(), device_models_[i]->optimizers_.size());
                }
            }
            device_models_[i]->sparse_lr_ = sparse_lr_;
            if (log_startup_timing) {
                SPDLOG_INFO("[startup-timing][Model::broadcast] clone ready slot={} device={}", i, device.str());
            }
        } else {
            device_models_[i] = std::dynamic_pointer_cast<Model>(shared_from_this());
            if (log_startup_timing) {
                SPDLOG_INFO("[startup-timing][Model::broadcast] reused primary model slot={} device={}", i, device.str());
            }
        }
        // SPDLOG_INFO("Broadcast: device_models_[{}]->optimizers_ size {}, devices {}", i, device_models_[i]->optimizers_.size(), device.index());
        i++;
    }
    SPDLOG_INFO("Broadcast finished");
}

shared_ptr<Model> initModelFromConfig(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations, bool train, NegativeSamplingMethod nsm, float nsmr) {
    bool log_startup_timing = startup_timing_enabled();
    shared_ptr<GeneralEncoder> encoder = nullptr;
    shared_ptr<Decoder> decoder = nullptr;
    shared_ptr<LossFunction> loss = nullptr;
    shared_ptr<Model> model;

    if (model_config->encoder == nullptr) {
        throw UnexpectedNullPtrException("Encoder config undefined");
    }

    if (model_config->decoder == nullptr) {
        throw UnexpectedNullPtrException("Decoder config undefined");
    }

    if (model_config->loss == nullptr) {
        throw UnexpectedNullPtrException("Loss config undefined");
    }

    auto tensor_options = torch::TensorOptions().device(devices[0]).dtype(torch::kFloat32);

    encoder = std::make_shared<GeneralEncoder>(model_config->encoder, devices[0], num_relations);

    if (model_config->learning_task == LearningTask::LINK_PREDICTION) {
        shared_ptr<EdgeDecoderOptions> decoder_options = std::dynamic_pointer_cast<EdgeDecoderOptions>(model_config->decoder->options);

        int last_stage = model_config->encoder->layers.size() - 1;
        int last_layer = model_config->encoder->layers[last_stage].size() - 1;
        int64_t dim = model_config->encoder->layers[last_stage][last_layer]->output_dim;

        decoder = get_edge_decoder(model_config->decoder->type, decoder_options->edge_decoder_method, num_relations, dim, tensor_options,
                                   decoder_options->inverse_edges);
        maybe_apply_distmult_relation_init(decoder, default_relation_init_config(model_config));
        log_edge_decoder_relation_stats("init", decoder);
    } else {
        decoder = get_node_decoder(model_config->decoder->type);
    }

    loss = getLossFunction(model_config->loss);

    model = std::make_shared<Model>(encoder, decoder, loss);
    model->device_ = devices[0];
    model->device_models_ = std::vector<shared_ptr<Model>>(devices.size());

    model->negative_sampling_method_ = nsm;
    model->negative_sampling_selected_ratio_ = nsmr;
    if (nsm == NegativeSamplingMethod::RNS) {
        SPDLOG_INFO("NegativeSamplingMethod: RNS");
    } else if (nsm == NegativeSamplingMethod::DNS) {
        SPDLOG_INFO("NegativeSamplingMethod: DNS");
        SPDLOG_INFO("NegativeSamplingSelectedRatio: {}", nsmr);
    } else if (nsm == NegativeSamplingMethod::GAN) {
        SPDLOG_INFO("NegativeSamplingMethod: GAN");
        SPDLOG_INFO("NegativeSamplingSelectedRatio: {}", nsmr);
    } else if (nsm == NegativeSamplingMethod::OTHER) {
        SPDLOG_INFO("NegativeSamplingMethod: OTHER");
    } else {
        throw GegeRuntimeException("Invalid NegativeSamplingMethod");
    }

    if (train) {
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][initModelFromConfig] base setup_optimizers begin device={}", devices[0].str());
        }
        model->setup_optimizers(model_config);
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][initModelFromConfig] base setup_optimizers end device={} optimizers={}",
                        devices[0].str(), model->optimizers_.size());
        }

        if (model_config->sparse_optimizer != nullptr) {
            model->sparse_lr_ = model_config->sparse_optimizer->options->learning_rate;
        } else {
            model->sparse_lr_ = model_config->dense_optimizer->options->learning_rate;
        }
    }

    if (devices.size() > 1) {
        SPDLOG_INFO("Broadcasting model to: {} GPUs", devices.size());
        model->broadcast(devices, model_config);
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][initModelFromConfig] broadcast returned devices={}", devices.size());
        }

        for (int i = 1; i < devices.size(); i++) {
          model->device_models_[i]->negative_sampling_method_ = nsm;
          model->device_models_[i]->negative_sampling_selected_ratio_ = nsmr;
        }
        if (log_startup_timing) {
            SPDLOG_INFO("[startup-timing][initModelFromConfig] device negative sampling metadata assigned devices={}", devices.size());
        }
    } else {
        model->device_models_[0] = model;
    }

    if (log_startup_timing) {
        SPDLOG_INFO("[startup-timing][initModelFromConfig] return devices={} train={}", devices.size(), train);
    }

    return model;
}
