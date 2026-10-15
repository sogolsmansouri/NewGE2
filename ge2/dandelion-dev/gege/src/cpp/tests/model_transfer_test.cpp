// Exercise production broadcast and transport gates without training a model.
#include <torch/cuda.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include "nn/model.h"
#include "nn/decoders/edge/distmult.h"

class BufferedTransferDecoder : public EdgeDecoder, public torch::nn::Cloneable<BufferedTransferDecoder> {
 public:
    torch::Tensor probe_;
    explicit BufferedTransferDecoder(torch::Device device) {
        tensor_options_ = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        learning_task_ = LearningTask::LINK_PREDICTION;
        num_relations_ = 3;
        embedding_size_ = 100;
        use_inverse_relations_ = false;
        reset();
    }
    void reset() override {
        relations_ = register_parameter("relation_embeddings", torch::zeros({3, 100}, tensor_options_));
        probe_ = register_buffer("transport_probe", torch::arange(257, tensor_options_.dtype(torch::kInt64)) + 123);
    }
};

std::shared_ptr<Model> make_model(bool with_buffer) {
    const torch::Device device(torch::kCUDA, 0);
    auto layer = std::make_shared<LayerConfig>();
    layer->type = LayerType::EMBEDDING;
    layer->input_dim = -1;
    layer->output_dim = 100;
    layer->bias = true;
    layer->activation = ActivationFunction::NONE;
    layer->init = std::make_shared<InitConfig>(InitDistribution::GLOROT_UNIFORM, nullptr);
    layer->bias_init = std::make_shared<InitConfig>(InitDistribution::ZEROS, nullptr);
    auto config = std::make_shared<EncoderConfig>();
    config->layers = {{layer}};
    auto encoder = std::make_shared<GeneralEncoder>(config, device, 3);
    std::shared_ptr<Decoder> decoder;
    if (with_buffer) decoder = std::make_shared<BufferedTransferDecoder>(device);
    else decoder = std::make_shared<DistMult>(3, 100, torch::TensorOptions().dtype(torch::kFloat32).device(device), true);
    auto model = std::make_shared<Model>(encoder, decoder, nullptr);
    model->device_ = device;
    model->device_models_.resize(2);
    model->negative_sampling_method_ = NegativeSamplingMethod::RNS;
    model->negative_sampling_selected_ratio_ = 1;
    model->sparse_lr_ = .1;
    torch::NoGradGuard guard;
    int offset = 1;
    for (auto &parameter : model->named_parameters()) {
        auto pattern = torch::arange(parameter.value().numel(), torch::kFloat32).div(64).add(offset++);
        parameter.value().copy_(pattern.reshape(parameter.value().sizes()));
    }
    return model;
}

bool run_host_case(bool with_buffer) {
    auto leader = make_model(with_buffer);
    torch::OrderedDict<std::string, torch::Tensor> original;
    for (const auto &p : leader->named_parameters()) original.insert(p.key(), p.value().detach().cpu().clone());
    leader->broadcast({torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)}, nullptr);
    for (int lane = 0; lane < 2; ++lane) {
        const auto &replica = leader->device_models_[lane];
        for (const auto &p : replica->named_parameters()) {
            if (!torch::equal(p.value().detach().cpu(), original[p.key()])) return false;
        }
        if (with_buffer) {
            const auto buffers = replica->named_buffers();
            if (buffers.size() != 1 || !torch::equal(buffers["decoder.transport_probe"].cpu(), torch::arange(257, torch::kInt64) + 123)) return false;
        }
    }
    // Both entry points would execute NCCL for this nonempty parameter set.
    // Rejection must happen before any division, gradient creation, or update.
    int rejected = 0;
    try { leader->all_reduce_rel(); } catch (const GegeRuntimeException &error) {
        if (std::string(error.what()).find("NCCL collective") != std::string::npos) ++rejected;
    }
    try { leader->all_reduce({1, 1}); } catch (const GegeRuntimeException &error) {
        if (std::string(error.what()).find("NCCL collective") != std::string::npos) ++rejected;
    }
    for (const auto &replica : leader->device_models_) {
        for (const auto &p : replica->named_parameters()) {
            if (p.value().grad().defined() || !torch::equal(p.value().detach().cpu(), original[p.key()])) return false;
        }
    }
    std::cout << "case=" << (with_buffer ? "decoder_buffer" : "distmult_encoder_bias")
              << " exact_broadcast=1 source_unchanged=1 collective_rejections=" << rejected << '\n';
    return rejected == 2;
}

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "host";
    if (mode != "host" && mode != "direct" && mode != "expect-direct-failure") return 2;
    if (torch::cuda::device_count() < 2) { std::cerr << "Two CUDA devices required\n"; return 2; }
    setenv("GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK", mode == "host" ? "1" : "0", 1);
    torch::set_num_threads(8);
    try {
        if (mode == "host") return run_host_case(false) && run_host_case(true) ? 0 : 1;
        auto model = make_model(false);
        model->broadcast({torch::Device(torch::kCUDA, 0), torch::Device(torch::kCUDA, 1)}, nullptr);
        if (mode == "expect-direct-failure") { std::cerr << "Expected failing direct-peer payload gate\n"; return 1; }
        std::cout << "direct_transport_and_broadcast_passed=1\n";
        return 0;
    } catch (const std::exception &error) {
        const std::string message = error.what();
        if (mode == "expect-direct-failure" && message.find("direct_peer payload validation failed") != std::string::npos &&
            message.find("GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK=1") != std::string::npos) {
            std::cout << "expected_direct_payload_rejection=1 reason=" << message << '\n';
            return 0;
        }
        std::cerr << message << '\n';
        return 1;
    }
}
