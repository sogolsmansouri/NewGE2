#include "common/training_contract.h"
#include "data/samplers/negative.h"
#include <iostream>

void check(bool value, const char *name) {
    TORCH_CHECK(value, name);
    std::cout << "PASS " << name << std::endl;
}

int main() {
    setenv("GEGE_TRAINING_REPLAY_SEED", "741135446461071584", 1);
    setenv("GEGE_DEG_CHUNK_EXCLUSION", "0", 1);
    setenv("GEGE_GLOBAL_DEGREE_SAMPLING", "0", 1);
    setenv("GEGE_BASELINE_TRAINING_SEMANTICS", "1", 1);
    auto graph = std::make_shared<GegeGraph>();
    graph->num_nodes_in_memory_ = 100003;
    for (auto device : {torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0)}) {
        auto options = torch::TensorOptions().device(device).dtype(torch::kInt64);
        auto edges = torch::randint(100003, {1003, 3}, options);
        NegativeSamplingBase sampler(50, 1000, 0.5, false, 0, LocalFilterMode::DEG);
        auto sample = [&](int epoch, int state, int batch) {
            training_contract::NegativeScope scope(training_contract::generator(device, 2, epoch, state, batch, 0));
            return sampler.getNodeCorruptNegatives(graph, edges, true, 0);
        };
        auto a = sample(0, 3, 1);
        torch::manual_seed(91);
        auto noise = torch::randint(100003, {12345}, options);
        auto b = sample(0, 3, 1);
        check(torch::equal(std::get<0>(a), std::get<0>(b)) && torch::equal(std::get<2>(a), std::get<2>(b)),
              "negative replay is unaffected by default RNG draws");
        check(!torch::equal(std::get<0>(a).slice(1, 500), std::get<2>(a).slice(1, 500)),
              "baseline head and tail uniforms are independent");
        {
            training_contract::NegativeScope scope(training_contract::generator(device, 2, 0, 3, 1, 0));
            auto src = sampler.getNegatives(graph, edges, true, 0);
            auto dst = sampler.getNegatives(graph, edges, false, 0);
            check(torch::equal(std::get<0>(a), std::get<0>(src)) && torch::equal(std::get<2>(a), std::get<0>(dst)),
                  "baseline paired API equals two separate original-style draws");
        }
        check(!training_contract::negative_generator, "negative scope restores RNG context");
        check(!torch::equal(std::get<0>(a), std::get<0>(sample(1, 3, 1))), "epoch changes negative draws");
        check(!torch::equal(std::get<0>(a), std::get<0>(sample(0, 4, 1))), "state changes negative draws");
        check(!torch::equal(std::get<0>(a), std::get<0>(sample(0, 3, 2))), "batch changes negative draws");
        auto perm = [&]() { return at::randperm(1003, training_contract::generator(device, 1, 0, 3, 0, 0), options); };
        auto p = perm();
        sample(4, 4, 4);
        check(torch::equal(p, perm()), "edge shuffle stream is independent of negative sampling");
        setenv("GEGE_BASELINE_TRAINING_SEMANTICS", "0", 1);
        auto legacy = sample(0, 3, 1);
        check(torch::equal(std::get<0>(legacy).slice(1, 500), std::get<2>(legacy).slice(1, 500)),
              "legacy paired sampling remains unchanged when compatibility is disabled");
        setenv("GEGE_BASELINE_TRAINING_SEMANTICS", "1", 1);
    }
}
