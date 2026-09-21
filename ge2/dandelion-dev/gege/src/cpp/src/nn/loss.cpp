#include "nn/loss.h"
#ifdef GEGE_CUDA
#include "nn/manual_backward_cuda.h"
#endif

#include <cmath>
#include <cstdlib>
#include <string>

namespace {

void require_unweighted_softmax() {
    // Reject obsolete tuning flags rather than silently mislabel old recipes.
    static const bool checked = []() {
        for (const auto &setting : {std::make_pair("GEGE_SOFTMAX_NEGATIVE_MASS_SCALE", 1.0),
                                   std::make_pair("GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS", 0.0)}) {
            const char *raw = std::getenv(setting.first);
            if (raw == nullptr || raw[0] == '\0') continue;
            bool valid = false;
            try {
                size_t consumed = 0;
                std::string text(raw);
                double parsed = std::stod(text, &consumed);
                valid = consumed == text.size() && std::isfinite(parsed) && parsed == setting.second;
            } catch (...) {}
            if (!valid) {
                throw GegeRuntimeException(std::string("Weighted negative softmax was removed; unset ") + setting.first);
            }
        }
        return true;
    }();
    (void)checked;
}

}  // namespace

void check_score_shapes(torch::Tensor pos_scores, torch::Tensor neg_scores) {
    if (!pos_scores.defined()) {
        throw UndefinedTensorException();
    }

    if (!neg_scores.defined()) {
        throw UndefinedTensorException();
    }

    if (pos_scores.sizes().size() != 1) {
        throw TensorSizeMismatchException(pos_scores, "Positive scores should be 1-dimensional");
    }

    if (neg_scores.sizes().size() != 2) {
        throw TensorSizeMismatchException(neg_scores, "Negative scores should be 2-dimensional");
    }

    if (pos_scores.size(0) != neg_scores.size(0)) {
        //        throw TensorSizeMismatchException(pos_scores, (std::stringstream("Size: ") << neg_scores.size(1) << " First dimension of pos_scores and
        //        neg_scores should match.").str());
        throw TensorSizeMismatchException(pos_scores, "First dimension of pos_scores and neg_scores should match.");
    }
}

torch::Tensor to_one_hot(torch::Tensor labels, int num_classes) {
    torch::Tensor one_hot_encodings = torch::zeros({labels.size(0), num_classes}, torch::kInt64);
    one_hot_encodings.index_fill_(1, labels.to(torch::kInt64), 1);
    return one_hot_encodings.to(torch::kFloat32);
}

std::tuple<torch::Tensor, torch::Tensor> scores_to_labels(torch::Tensor pos_scores, torch::Tensor neg_scores, bool one_hot) {
    torch::Tensor y_pred = torch::cat({pos_scores, neg_scores}, -1);
    torch::Tensor labels;
    if (one_hot) {
        labels = torch::cat({torch::ones_like(pos_scores), torch::zeros_like(neg_scores)}, -1);
    } else {
        auto options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
        labels = torch::zeros({pos_scores.size(0)}, options);
    }

    return std::forward_as_tuple(y_pred, labels);
}

torch::Tensor SoftmaxCrossEntropy::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (!scores) {
        throw GegeRuntimeException(
            "Input to SoftmaxCrossEntropy loss function must be scores. SoftmaxCrossEntropy is currently unsupported for classification.");
    }

    check_score_shapes(y_pred, labels);
    require_unweighted_softmax();
    torch::Tensor negative_log_mass = labels.logsumexp(1, true);
    std::tie(y_pred, labels) = scores_to_labels(y_pred.unsqueeze(1), negative_log_mass, false);

    torch::nn::functional::CrossEntropyFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::cross_entropy(y_pred, labels, options);
}

std::tuple<torch::Tensor, torch::Tensor> SoftmaxCrossEntropy::score_gradients(
    torch::Tensor pos_scores, torch::Tensor neg_scores) const {
    torch::NoGradGuard no_grad;
    check_score_shapes(pos_scores, neg_scores);
    require_unweighted_softmax();
    auto negative_log_mass = neg_scores.logsumexp(1, true);
    auto logits = torch::cat({pos_scores.unsqueeze(1), negative_log_mass}, 1);
    auto log_probs = logits.log_softmax(1);
    auto targets = torch::zeros({pos_scores.size(0)}, pos_scores.options().dtype(torch::kInt64));
    int64_t reduction = reduction_type_ == LossReduction::MEAN ? at::Reduction::Mean : at::Reduction::Sum;
    // Use the same native reverse kernels and reduction scaling as forward loss,
    // without constructing an autograd graph. Algebraic softmax reassociation
    // changes near-zero gradients enough to alter first-step Adagrad updates.
    auto grad_log_probs = at::nll_loss_backward(torch::ones({}, pos_scores.options()), log_probs, targets,
        c10::nullopt, reduction, -100, torch::full({}, pos_scores.size(0), pos_scores.options()));
    auto grad_logits = at::_log_softmax_backward_data(grad_log_probs, log_probs, 1, logits.scalar_type());
    torch::Tensor grad_neg;
#ifdef GEGE_CUDA
    if (neg_scores.is_cuda() && neg_scores.scalar_type() == torch::kFloat32) {
        grad_neg = negative_log_mass_backward_cuda(neg_scores, negative_log_mass, grad_logits.narrow(1, 1, 1));
    } else
#endif
    {
        grad_neg = grad_logits.narrow(1, 1, 1) * (neg_scores - negative_log_mass).exp();
    }
    return {grad_logits.narrow(1, 0, 1), grad_neg};
}

torch::Tensor RankingLoss::operator()(torch::Tensor pos_scores, torch::Tensor neg_scores, bool scores) {
    // does this loss make sense?

    if (!scores) {
        throw GegeRuntimeException("Input to ranking loss function must be scores. This loss function is unsupported for classification.");
    }

    auto device_options = torch::TensorOptions().dtype(torch::kInt64).device(pos_scores.device());
    torch::nn::functional::MarginRankingLossFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }
    options.margin(margin_);

    return torch::nn::functional::margin_ranking_loss(neg_scores, pos_scores.unsqueeze(1), pos_scores.new_full({1, 1}, -1, device_options), options);
}

torch::Tensor CrossEntropyLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred.unsqueeze(1), labels, false);
    }

    torch::nn::functional::CrossEntropyFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::cross_entropy(y_pred, labels, options);
}

torch::Tensor BCEAfterSigmoidLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    torch::nn::functional::BinaryCrossEntropyFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::binary_cross_entropy(y_pred.sigmoid(), labels, options);
}

torch::Tensor BCEWithLogitsLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::binary_cross_entropy_with_logits(y_pred, labels, options);
}

torch::Tensor MSELoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    torch::nn::functional::MSELossFuncOptions options;
    if (reduction_type_ == LossReduction::MEAN) {
        options.reduction(torch::kMean);
    } else if (reduction_type_ == LossReduction::SUM) {
        options.reduction(torch::kSum);
    }

    return torch::nn::functional::mse_loss(y_pred, labels, options);
}

torch::Tensor SoftPlusLoss::operator()(torch::Tensor y_pred, torch::Tensor labels, bool scores) {
    if (scores) {
        check_score_shapes(y_pred, labels);
        std::tie(y_pred, labels) = scores_to_labels(y_pred, labels.flatten(0, 1), true);
    } else {
        labels = to_one_hot(labels, y_pred.size(-1));
    }

    labels = 2 * labels - 1;
    auto loss = torch::nn::functional::softplus(((-1) * labels * y_pred));
    if (reduction_type_ == LossReduction::MEAN) {
        loss = loss.mean();
    } else if (reduction_type_ == LossReduction::SUM) {
        loss = loss.sum();
    }

    return loss;
}

std::shared_ptr<LossFunction> getLossFunction(shared_ptr<LossConfig> config) {
    if (config == nullptr) {
        throw UnexpectedNullPtrException();
    }

    if (config->type == LossFunctionType::SOFTMAX_CE) {
        return std::make_shared<SoftmaxCrossEntropy>(config->options);
    } else if (config->type == LossFunctionType::RANKING) {
        return std::make_shared<RankingLoss>(std::dynamic_pointer_cast<RankingLossOptions>(config->options));
    } else if (config->type == LossFunctionType::CROSS_ENTROPY) {
        return std::make_shared<CrossEntropyLoss>(config->options);
    } else if (config->type == LossFunctionType::BCE_AFTER_SIGMOID) {
        return std::make_shared<BCEAfterSigmoidLoss>(config->options);
    } else if (config->type == LossFunctionType::BCE_WITH_LOGITS) {
        return std::make_shared<BCEWithLogitsLoss>(config->options);
    } else if (config->type == LossFunctionType::MSE) {
        return std::make_shared<MSELoss>(config->options);
    } else if (config->type == LossFunctionType::SOFTPLUS) {
        return std::make_shared<SoftPlusLoss>(config->options);
    } else {
        throw std::runtime_error("Unsupported loss function type");
    }
}
