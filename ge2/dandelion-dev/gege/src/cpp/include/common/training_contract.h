#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <torch/torch.h>
#include <cstdlib>
#include <string>
#ifdef GEGE_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

namespace training_contract {

inline bool baseline_semantics() {
    const char *value = std::getenv("GEGE_BASELINE_TRAINING_SEMANTICS");
    return value && std::string(value) == "1";
}

inline c10::optional<uint64_t> replay_seed() {
    static const auto seed = []() -> c10::optional<uint64_t> {
        const char *value = std::getenv("GEGE_TRAINING_REPLAY_SEED");
        if (!value) return c10::nullopt;
        std::string text(value);
        TORCH_CHECK(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
                    "GEGE_TRAINING_REPLAY_SEED must be an unsigned integer");
        return std::stoull(text);
    }();
    return seed;
}

inline uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Domain-separated generators leave model initialization and background workers' RNG untouched.
inline c10::optional<at::Generator> generator(torch::Device device, uint64_t domain,
                                             int64_t epoch, int64_t state, int64_t batch, int64_t lane) {
    auto seed = replay_seed();
    if (!seed) return c10::nullopt;
    uint64_t key = mix(*seed ^ domain);
    for (auto part : {epoch, state, batch, lane}) key = mix(key ^ mix(static_cast<uint64_t>(part)));
    if (device.is_cpu()) return at::make_generator<at::CPUGeneratorImpl>(key);
#ifdef GEGE_CUDA
    if (device.is_cuda()) {
        auto gen = at::cuda::detail::createCUDAGenerator(device.index());
        gen.set_current_seed(key);
        return gen;
    }
#endif
    TORCH_CHECK(false, "Training replay only supports CPU and CUDA generators");
}

inline thread_local c10::optional<at::Generator> negative_generator;

inline uint64_t tensor_fingerprint(const torch::Tensor &tensor) {
    if (!tensor.defined()) return 0;
    auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    const auto *bytes = static_cast<const unsigned char *>(cpu.data_ptr());
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < cpu.nbytes(); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}

class NegativeScope {
    c10::optional<at::Generator> previous_;
public:
    explicit NegativeScope(c10::optional<at::Generator> gen) : previous_(negative_generator) {
        negative_generator = std::move(gen);
    }
    ~NegativeScope() { negative_generator = previous_; }
    NegativeScope(const NegativeScope &) = delete;
    NegativeScope &operator=(const NegativeScope &) = delete;
};

}  // namespace training_contract
