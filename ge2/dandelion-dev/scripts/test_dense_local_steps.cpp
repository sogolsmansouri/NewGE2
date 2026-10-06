// Independent CPU arithmetic for the diagnostic separate-gradient update.
#include <torch/cuda.h>
#include <iostream>
#include "nn/model.h"
#include "nn/decoders/edge/complex.h"

int main() {
    if (torch::cuda::device_count() < 2) return 2;
    torch::set_num_threads(2);
    torch::manual_seed(17);
    const auto initial = torch::randn({7, 100}) * .02f;
    std::vector<std::shared_ptr<Model>> replicas;
    for (int lane = 0; lane < 2; ++lane) {
        auto device = torch::Device(torch::kCUDA, lane);
        auto decoder = std::make_shared<ComplEx>(7, 100, initial.options().device(device), true,
                                                EdgeDecoderMethod::CORRUPT_NODE);
        torch::NoGradGuard guard;
        decoder->relations_.copy_(initial);
        decoder->inverse_relations_.copy_(initial * .5f);
        auto model = std::make_shared<Model>(nullptr, decoder, nullptr);
        model->device_ = device;
        model->negative_sampling_method_ = NegativeSamplingMethod::RNS;
        torch::OrderedDict<std::string, torch::Tensor> parameters;
        parameters.insert("relations", decoder->relations_);
        parameters.insert("inverse_relations", decoder->inverse_relations_);
        auto options = std::make_shared<AdagradOptions>();
        options->learning_rate = .1f;
        options->eps = 1.e-10f;
        options->init_value = 0;
        options->lr_decay = 0;
        options->weight_decay = 0;
        model->optimizers_.push_back(std::make_shared<AdagradOptimizer>(parameters, options));
        replicas.push_back(model);
    }
    // The coordinator's keys match the replica decoders, as in production.
    auto leader = std::make_shared<Model>(nullptr, replicas[0]->decoder_, nullptr);
    leader->negative_sampling_method_ = NegativeSamplingMethod::RNS;
    leader->device_models_ = replicas;
    std::vector<torch::Tensor> expected = {initial.clone(), initial * .5f};
    std::vector<torch::Tensor> sums = {torch::zeros_like(initial), torch::zeros_like(initial)};
    double weight_error = 0, state_error = 0;
    for (int round = 0; round < 24; ++round) {
        std::vector<std::vector<torch::Tensor>> gradients(2);
        const bool inactive = round % 4 == 1;
        for (int lane = 0; lane < 2; ++lane) {
            auto decoder = std::dynamic_pointer_cast<ComplEx>(leader->device_models_[lane]->decoder_);
            for (int table = 0; table < 2; ++table) {
                auto parameter = table == 0 ? decoder->relations_ : decoder->inverse_relations_;
                auto gradient = torch::randn_like(initial) * (lane == 0 ? .3f : .017f);
                gradients[lane].push_back(gradient);
                parameter.mutable_grad() = lane == 1 && inactive ? torch::Tensor() : gradient.to(parameter.device());
            }
        }
        for (int lane = 0; lane < 2; ++lane) {
            if (lane == 1 && inactive) continue;
            for (int table = 0; table < 2; ++table) {
                const auto &gradient = gradients[lane][table];
                sums[table].addcmul_(gradient, gradient);
                expected[table].addcdiv_(gradient, sums[table].sqrt().add_(1.e-10f), -.1f);
            }
        }
        leader->all_reduce({1, 1});
        for (int lane = 0; lane < 2; ++lane) {
            auto model = leader->device_models_[lane];
            auto decoder = std::dynamic_pointer_cast<ComplEx>(model->decoder_);
            auto optimizer = std::dynamic_pointer_cast<AdagradOptimizer>(model->optimizers_[0]);
            for (int table = 0; table < 2; ++table) {
                const auto key = table == 0 ? "relations" : "inverse_relations";
                auto parameter = table == 0 ? decoder->relations_ : decoder->inverse_relations_;
                TORCH_CHECK(torch::isfinite(parameter).all().item<bool>() &&
                    torch::isfinite(optimizer->state_dict_[key]["sum"]).all().item<bool>(),
                    "Nonfinite relation weights or optimizer state");
                weight_error = std::max(weight_error, (parameter.cpu() - expected[table]).abs().max().item<double>());
                state_error = std::max(state_error,
                    (optimizer->state_dict_[key]["sum"].cpu() - sums[table]).abs().max().item<double>());
                TORCH_CHECK(!parameter.grad().defined(), "Relation gradients were not cleared");
            }
        }
    }
    bool passed = weight_error < 2.e-6 && state_error < 2.e-6;
    std::cout << "separate_local_steps rounds=24 weight_max_error=" << weight_error
              << " state_max_error=" << state_error << " passed=" << passed << '\n';
    return passed ? 0 : 1;
}
