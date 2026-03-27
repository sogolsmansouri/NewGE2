#include "nn/model.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef GEGE_CUDA
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
    static bool enabled = parse_env_flag("GEGE_EVAL_CHUNKED_RANKS", true);
    return enabled;
}

bool emulate_dot_single_relation_enabled() {
    static bool enabled = parse_env_flag("GEGE_EMULATE_DOT_SINGLE_RELATION", false);
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

    torch::Tensor encoded_nodes = encoder_->forward(batch->node_embeddings_, batch->node_features_, batch->dense_graph_, train);
    if (!train) {
        log_eval_tensor_if_non_finite("encoded_nodes", batch->batch_id_, encoded_nodes);
    }
    // call proper decoder
    torch::Tensor pos_scores;
    torch::Tensor neg_scores;
    torch::Tensor inv_pos_scores;
    torch::Tensor inv_neg_scores;

    auto edge_decoder = std::dynamic_pointer_cast<EdgeDecoder>(decoder_);

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

    if (batch->node_embeddings_.defined()) {
        batch->node_embeddings_.requires_grad_();
    }

    // Arity-4: enable gradient tracking for qualifier value embeddings
    if (batch->qual_embeddings_.defined()) {
        batch->qual_embeddings_.requires_grad_();
    }

    torch::Tensor loss;

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        auto all_scores = forward_lp(batch, true);

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

    if (call_step) {
        step();
    }

    if (batch->node_embeddings_.defined()) {
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
    int i = 0;
    for (auto device : devices) {
        SPDLOG_INFO("Broadcast to GPU {}", device.index());
        if (device != device_) {
            shared_ptr<GeneralEncoder> encoder = encoder_clone_helper(encoder_, device);
            shared_ptr<Decoder> decoder = decoder_clone_helper(decoder_, device);
            device_models_[i] = std::make_shared<Model>(encoder, decoder, loss_function_, reporter_);
            device_models_[i]->device_ = device;
            device_models_[i]->negative_sampling_method_ = negative_sampling_method_;
            device_models_[i]->negative_sampling_selected_ratio_ = negative_sampling_selected_ratio_;
            if (!optimizers_.empty()) {
                device_models_[i]->setup_optimizers(model_config);
            }
            device_models_[i]->sparse_lr_ = sparse_lr_;
        } else {
            device_models_[i] = std::dynamic_pointer_cast<Model>(shared_from_this());
        }
        // SPDLOG_INFO("Broadcast: device_models_[{}]->optimizers_ size {}, devices {}", i, device_models_[i]->optimizers_.size(), device.index());
        i++;
    }
    SPDLOG_INFO("Broadcast finished");
}

shared_ptr<Model> initModelFromConfig(shared_ptr<ModelConfig> model_config, std::vector<torch::Device> devices, int num_relations, bool train, NegativeSamplingMethod nsm, float nsmr) {
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
        model->setup_optimizers(model_config);

        if (model_config->sparse_optimizer != nullptr) {
            model->sparse_lr_ = model_config->sparse_optimizer->options->learning_rate;
        } else {
            model->sparse_lr_ = model_config->dense_optimizer->options->learning_rate;
        }
    }

    if (devices.size() > 1) {
        SPDLOG_INFO("Broadcasting model to: {} GPUs", devices.size());
        model->broadcast(devices, model_config);

        for (int i = 1; i < devices.size(); i++) {
          model->device_models_[i]->negative_sampling_method_ = nsm;
          model->device_models_[i]->negative_sampling_selected_ratio_ = nsmr;
        }
    } else {
        model->device_models_[0] = model;
    }

    return model;
}
