// Exercise production relation-gradient reduction and optimizer state on two GPUs.
#include <ATen/Context.h>
#include <torch/cuda.h>

#include <cmath>
#include <iostream>

#include "nn/model.h"
#include "nn/decoders/edge/complex.h"

namespace {

std::shared_ptr<AdagradOptions> adagrad_options() {
    auto options = std::make_shared<AdagradOptions>();
    options->learning_rate = .1f;
    options->eps = 1.e-10f;
    options->init_value = 0;
    options->lr_decay = 0;
    options->weight_decay = 0;
    return options;
}

std::shared_ptr<Model> make_model(const torch::Tensor &weights, int device, bool adagrad) {
    auto options = weights.options().device(torch::Device(torch::kCUDA, device));
    auto decoder = std::make_shared<ComplEx>(7, weights.size(1), options, true, EdgeDecoderMethod::CORRUPT_NODE);
    {
        torch::NoGradGuard guard;
        decoder->relations_.copy_(weights);
        decoder->inverse_relations_.copy_(weights * .5f);
    }
    auto model = std::make_shared<Model>(nullptr, decoder, nullptr);
    model->device_ = torch::Device(torch::kCUDA, device);
    torch::OrderedDict<std::string, torch::Tensor> params;
    params.insert("relations", decoder->relations_);
    params.insert("inverse_relations", decoder->inverse_relations_);
    if (adagrad) {
        model->optimizers_.push_back(std::make_shared<AdagradOptimizer>(params, adagrad_options()));
    } else {
        model->optimizers_.push_back(std::make_shared<SGDOptimizer>(params, .1f));
    }
    return model;
}

double max_error(const torch::Tensor &actual, const torch::Tensor &expected) {
    return (actual.cpu() - expected).abs().max().item<double>();
}

bool run_case(bool adagrad, const std::string &scenario) {
    torch::manual_seed(20261004);
    auto weights = torch::randn({7, 100}) * .02f;
    auto leader = make_model(weights, 0, adagrad);
    leader->device_models_ = {make_model(weights, 0, adagrad), make_model(weights, 1, adagrad)};
    auto expected = std::vector<torch::Tensor>{weights.clone(), weights * .5f};
    auto sums = std::vector<torch::Tensor>{torch::zeros_like(weights), torch::zeros_like(weights)};
    double weight_error = 0, state_error = 0, replica_error = 0;
    for (int step = 0; step < 24; ++step) {
        std::vector<int64_t> counts = {1, 1};
        if (scenario == "accumulated") counts = {3, 2};
        if (scenario == "inactive" && step % 3 == 1) counts[1] = 0;
        for (int table = 0; table < 2; ++table) {
            auto reduced = torch::zeros_like(weights);
            for (int lane = 0; lane < 2; ++lane) {
                auto decoder = std::dynamic_pointer_cast<ComplEx>(leader->device_models_[lane]->decoder_);
                auto parameter = table == 0 ? decoder->relations_ : decoder->inverse_relations_;
                if (counts[lane] == 0) {
                    parameter.mutable_grad() = torch::Tensor();
                    continue;
                }
                auto gradient = torch::randn_like(weights) * (step % 2 == 0 ? .03f : .3f);
                // Partial batches produce smaller SUM-reduced gradients, not a new averaging rule.
                if (scenario == "partial" && lane == 1) gradient *= .17f;
                gradient *= counts[lane];
                parameter.mutable_grad() = gradient.to(parameter.device());
                reduced.add_(gradient / (2.0 * counts[lane]));
            }
            if (adagrad) {
                sums[table].addcmul_(reduced, reduced);
                expected[table].addcdiv_(reduced, sums[table].sqrt().add_(1.e-10f), -.1f);
            } else {
                expected[table].add_(reduced, -.1f);
            }
        }
        leader->all_reduce(counts);
        for (int lane = 0; lane < 2; ++lane) {
            auto model = leader->device_models_[lane];
            auto decoder = std::dynamic_pointer_cast<ComplEx>(model->decoder_);
            weight_error = std::max(weight_error, max_error(decoder->relations_, expected[0]));
            weight_error = std::max(weight_error, max_error(decoder->inverse_relations_, expected[1]));
            if (adagrad) {
                auto optimizer = std::dynamic_pointer_cast<AdagradOptimizer>(model->optimizers_[0]);
                state_error = std::max(state_error, max_error(optimizer->state_dict_["relations"]["sum"], sums[0]));
                state_error = std::max(state_error, max_error(optimizer->state_dict_["inverse_relations"]["sum"], sums[1]));
            }
        }
        auto first = std::dynamic_pointer_cast<ComplEx>(leader->device_models_[0]->decoder_);
        auto second = std::dynamic_pointer_cast<ComplEx>(leader->device_models_[1]->decoder_);
        replica_error = std::max(replica_error, max_error(first->relations_, second->relations_.cpu()));
        replica_error = std::max(replica_error, max_error(first->inverse_relations_, second->inverse_relations_.cpu()));
    }
    // CPU/GPU division and fused Adagrad operations need not be bitwise identical.
    bool passed = std::isfinite(weight_error) && weight_error < 2.e-6 &&
                  std::isfinite(state_error) && state_error < 2.e-6 && replica_error == 0;
    std::cout << "optimizer=" << (adagrad ? "adagrad" : "sgd") << " scenario=" << scenario
              << " steps=24 weight_max_error=" << weight_error << " state_max_error=" << state_error
              << " replica_max_error=" << replica_error << " passed=" << passed << '\n';
    return passed;
}

}  // namespace

int main() {
    if (torch::cuda::device_count() < 2) {
        std::cerr << "Two CUDA devices are required; this is not a CPU fallback test.\n";
        return 2;
    }
    torch::set_num_threads(2);
    at::globalContext().setAllowTF32CuBLAS(false);
    bool passed = true;
    for (bool adagrad : {false, true}) {
        for (const std::string &scenario : {"balanced", "partial", "accumulated", "inactive"}) {
            passed = run_case(adagrad, scenario) && passed;
        }
    }
    return passed ? 0 : 1;
}
