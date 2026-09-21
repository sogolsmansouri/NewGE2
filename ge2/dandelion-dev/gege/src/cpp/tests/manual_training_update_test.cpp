// Compare the explicit native backward with the configured autograd procedure.
// Supported manual cases must execute without autograd and match bitwise.
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <chrono>
#include <iostream>
#include "nn/model.h"
#include "nn/layers/embedding/embedding.h"
#include "nn/decoders/edge/distmult.h"
#include "nn/decoders/edge/complex.h"

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "manual";
    if (mode != "safe_manual" && mode != "autograd" && mode != "manual") {
        std::cerr << "mode must be safe_manual, autograd, or manual\n";
        return 2;
    }
    const char *enabled = mode == "autograd" ? "0" : "1";
    int64_t seed = 20260919, width = 100;
    std::string mass = "1";
    std::string log_mass = "0";
    bool benchmark = false, expect_rejection = false;
    for (int arg = 2; arg < argc; ++arg) {
        const std::string key = argv[arg];
        if (key == "--benchmark") benchmark = true;
        else if (key == "--seed" && arg+1 < argc) seed = std::stoll(argv[++arg]);
        else if (key == "--width" && arg+1 < argc) width = std::stoll(argv[++arg]);
        else if (key == "--mass" && arg+1 < argc) mass = argv[++arg];
        else if (key == "--log-mass" && arg+1 < argc) log_mass = argv[++arg];
        else if (key == "--expect-unweighted-rejection") expect_rejection = true;
        else { std::cerr << "Unknown/incomplete argument: " << key << "\n"; return 2; }
    }
    setenv("GEGE_EMULATE_DOT_SINGLE_RELATION", "1", 1);
    setenv("GEGE_FIXED_BUFFER_MANUAL_DOT_RNS", enabled, 1);
    setenv("GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS", enabled, 1);
    setenv("GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS", enabled, 1);
    setenv("GEGE_FIXED_BUFFER_MASKED_UPDATE", "1", 1);
    setenv("GEGE_CSR_GATHER", "0", 1);
    setenv("GEGE_SCORE_FILTER_CUDA", "1", 1);
    setenv("GEGE_SOFTMAX_NEGATIVE_MASS_SCALE", mass.c_str(), 1);
    setenv("GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS", log_mass.c_str(), 1);
    if (expect_rejection) {
        auto options = std::make_shared<LossOptions>();
        options->loss_reduction = LossReduction::SUM;
        SoftmaxCrossEntropy loss(options);
        auto pos = torch::zeros({2});
        auto neg = torch::zeros({2, 3});
        int rejected = 0;
        try { loss(pos, neg, true); } catch (const GegeRuntimeException &) { ++rejected; }
        try { loss.score_gradients(pos, neg); } catch (const GegeRuntimeException &) { ++rejected; }
        std::cout << "unweighted_guard_rejections=" << rejected << std::endl;
        return rejected == 2 ? 0 : 1;
    }
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
    at::globalContext().setDeterministicAlgorithms(!benchmark, false);
    at::globalContext().setAllowTF32CuBLAS(false);
    torch::set_num_threads(4);
    torch::Device device(torch::kCUDA, 0);
    auto f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto i = torch::TensorOptions().dtype(torch::kInt64).device(device);
    bool passed = true;
    for (const std::string &decoder_name : {"dot", "distmult", "complex"}) {
      for (const std::string &scenario : {"sum", "partial", "mean", "cross_entropy", "bce", "bias", "activation", "zero_state", "paper_batch", "learned_relations", "paper_learned", "paper_mean", "no_mask", "mean_no_mask"}) {
        if (benchmark && scenario != "paper_batch") continue;
        torch::manual_seed(seed);
        bool paper = scenario.rfind("paper_", 0) == 0;
        int64_t rows = paper ? 150000 : 4096;
        int64_t n = paper ? 50000 : scenario == "partial" ? 1003 : 1000;
        int64_t chunks = 50, negatives = paper ? 1000 : 32;
        auto layer_cfg = std::make_shared<LayerConfig>();
        layer_cfg->type = LayerType::EMBEDDING;
        layer_cfg->output_dim = width;
        layer_cfg->input_dim = -1;
        layer_cfg->bias = scenario == "bias";
        layer_cfg->activation = scenario == "activation" ? ActivationFunction::RELU : ActivationFunction::NONE;
        layer_cfg->init = std::make_shared<InitConfig>(InitDistribution::GLOROT_UNIFORM, nullptr);
        layer_cfg->bias_init = std::make_shared<InitConfig>(InitDistribution::ZEROS, nullptr);
        auto layer = std::make_shared<EmbeddingLayer>(layer_cfg, device);
        if (scenario == "bias") {
            torch::NoGradGuard guard;
            layer->bias_.fill_(.1);
        }
        auto encoder = std::make_shared<GeneralEncoder>(std::vector<std::vector<std::shared_ptr<Layer>>>{{layer}});
        std::shared_ptr<EdgeDecoder> decoder;
        if (decoder_name == "complex") decoder = std::make_shared<ComplEx>(7, width, f, true, EdgeDecoderMethod::CORRUPT_NODE);
        else decoder = std::make_shared<DistMult>(7, width, f, true, EdgeDecoderMethod::CORRUPT_NODE);
        if (scenario == "learned_relations" || scenario == "paper_learned") {
            torch::NoGradGuard guard;
            decoder->relations_.copy_(torch::randn_like(decoder->relations_));
            decoder->inverse_relations_.copy_(torch::randn_like(decoder->inverse_relations_));
        }
        auto options = std::make_shared<LossOptions>();
        options->loss_reduction = (scenario == "mean" || scenario == "paper_mean" || scenario == "mean_no_mask") ? LossReduction::MEAN : LossReduction::SUM;
        std::shared_ptr<LossFunction> loss;
        if (scenario == "cross_entropy") loss = std::make_shared<CrossEntropyLoss>(options);
        else if (scenario == "bce") loss = std::make_shared<BCEWithLogitsLoss>(options);
        else loss = std::make_shared<SoftmaxCrossEntropy>(options);
        Model model(encoder, decoder, loss);
        model.negative_sampling_method_ = NegativeSamplingMethod::RNS;
        model.negative_sampling_selected_ratio_ = 1;
        model.sparse_lr_ = .1;
        auto embeddings = torch::randn({rows, width}, f) * .02;
        auto state = (scenario == "zero_state" || paper)
            ? torch::zeros_like(embeddings) : torch::ones_like(embeddings) * .1;
        auto edges = torch::randint(rows-10, {n, decoder_name == "dot" ? 2 : 3}, i);
        if (decoder_name != "dot") edges.select(1, 1).copy_(torch::randint(7, {n}, i));
        auto head = torch::randint(rows-10, {chunks, negatives}, i);
        auto tail = torch::randint(rows-10, {chunks, negatives}, i);
        // Include real filters and padding sentinels used by bitmap batching.
        auto filter = torch::tensor({{0, 1}, {1, 2}, {-1, -1}}, i);
        auto make_batch = [&]() {
            auto b = std::make_shared<Batch>(true);
            b->batch_size_ = n;
            b->edges_ = edges.clone();
            b->node_embeddings_ = embeddings.clone();
            b->node_embeddings_state_ = state.clone();
            b->unique_node_indices_ = torch::arange(rows, i);
            if (scenario != "no_mask" && scenario != "mean_no_mask") {
                b->unique_node_active_mask_ = torch::ones({rows}, f.dtype(torch::kUInt8));
                b->unique_node_active_mask_.narrow(0, rows-10, 10).zero_();
            }
            b->src_neg_indices_mapping_ = head.clone();
            b->dst_neg_indices_mapping_ = tail.clone();
            b->src_neg_filter_ = filter.clone();
            b->dst_neg_filter_ = filter.clone();
            return b;
        };
        if (benchmark) {
            std::vector<double> elapsed;
            for (int step = 0; step < 35; ++step) {
                auto b = make_batch();
                decoder->relations_.mutable_grad() = torch::Tensor();
                decoder->inverse_relations_.mutable_grad() = torch::Tensor();
                torch::cuda::synchronize();
                auto begin = std::chrono::steady_clock::now();
                model.train_batch(b, false);
                torch::cuda::synchronize();
                if (step >= 5) elapsed.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count());
                if ((!b->node_embeddings_.requires_grad()) != (mode != "autograd")) return 3;
            }
            std::sort(elapsed.begin(), elapsed.end());
            double sum = 0;
            for (double ms : elapsed) sum += ms;
            std::cout << "{\"benchmark\":true,\"mode\":\"" << mode << "\",\"decoder\":\"" << decoder_name
                      << "\",\"width\":" << width << ",\"batch\":" << n << ",\"steps\":" << elapsed.size()
                      << ",\"median_ms\":" << (elapsed[14]+elapsed[15])/2 << ",\"mean_ms\":" << sum/elapsed.size()
                      << ",\"min_ms\":" << elapsed.front() << ",\"max_ms\":" << elapsed.back() << "}" << std::endl;
            continue;
        }
        auto reference = make_batch();
        reference->node_embeddings_.requires_grad_();
        auto scores = model.forward_lp(reference, true);
        auto objective = (*loss)(std::get<0>(scores), std::get<1>(scores), true);
        if (std::get<3>(scores).defined()) objective = objective + (*loss)(std::get<2>(scores), std::get<3>(scores), true);
        objective.backward();
        auto raw = reference->node_embeddings_.grad().clone();
        reference->accumulateGradients(.1);
        auto ref_rel = decoder->relations_.grad().defined() ? decoder->relations_.grad().clone() : torch::Tensor();
        auto ref_inv = decoder->inverse_relations_.grad().defined() ? decoder->inverse_relations_.grad().clone() : torch::Tensor();
        auto ref_bias = layer->bias_.defined() && layer->bias_.grad().defined() ? layer->bias_.grad().clone() : torch::Tensor();
        decoder->relations_.mutable_grad() = torch::Tensor();
        decoder->inverse_relations_.mutable_grad() = torch::Tensor();
        if (layer->bias_.defined()) layer->bias_.mutable_grad() = torch::Tensor();
        auto actual = make_batch();
        model.train_batch(actual, false);
        bool manual_executed = !actual->node_embeddings_.requires_grad();
        bool manual_expected = mode != "autograd" && scenario != "cross_entropy" && scenario != "bce" &&
            scenario != "bias" && scenario != "activation";
        bool deltas = torch::allclose(actual->node_gradients_, reference->node_gradients_, 1e-4, 1e-5);
        bool accumulators = torch::allclose(actual->node_state_update_, reference->node_state_update_, 1e-4, 1e-7);
        bool relations = !ref_rel.defined() || (decoder->relations_.grad().defined() &&
            torch::allclose(decoder->relations_.grad(), ref_rel, 1e-4, 1e-6) &&
            torch::allclose(decoder->inverse_relations_.grad(), ref_inv, 1e-4, 1e-6));
        auto difference = (actual->node_gradients_ - reference->node_gradients_).abs();
        auto worst = difference.flatten().argmax().item<int64_t>();
        bool bias = !ref_bias.defined() || (layer->bias_.grad().defined() && torch::allclose(layer->bias_.grad(), ref_bias, 1e-4, 1e-6));
        bool exact = torch::equal(actual->node_gradients_, reference->node_gradients_) &&
            torch::equal(actual->node_state_update_, reference->node_state_update_) &&
            (!ref_rel.defined() || (decoder->relations_.grad().defined() && decoder->inverse_relations_.grad().defined() &&
             torch::equal(decoder->relations_.grad(), ref_rel) && torch::equal(decoder->inverse_relations_.grad(), ref_inv))) &&
            (!ref_bias.defined() || (layer->bias_.grad().defined() && torch::equal(layer->bias_.grad(), ref_bias)));
        bool ok = deltas && accumulators && relations && bias && exact &&
            manual_executed == manual_expected;
        passed &= ok;
        std::cout << "{\"mode\":\"" << mode << "\",\"decoder\":\"" << decoder_name
                  << "\",\"scenario\":\"" << scenario << "\",\"pass\":" << (ok ? "true" : "false")
                  << ",\"exact\":" << (exact ? "true" : "false")
                  << ",\"manual_executed\":" << (manual_executed ? "true" : "false")
                  << ",\"seed\":" << seed << ",\"width\":" << width << ",\"negative_mass\":" << mass
                  << ",\"delta_max_abs\":" << difference.max().item<double>()
                  << ",\"state_max_abs\":" << (actual->node_state_update_-reference->node_state_update_).abs().max().item<double>()
                  << ",\"worst_ref_raw_gradient\":" << raw.flatten()[worst].item<double>()
                  << ",\"worst_ref_delta\":" << reference->node_gradients_.flatten()[worst].item<double>()
                  << ",\"worst_actual_delta\":" << actual->node_gradients_.flatten()[worst].item<double>()
                  << ",\"delta_mismatch_count\":" << (difference > (1e-5 + 1e-4 * reference->node_gradients_.abs())).sum().item<int64_t>()
                  << ",\"delta_elements\":" << difference.numel()
                  << ",\"opposite_sign_count\":" << ((actual->node_gradients_ * reference->node_gradients_) < 0).sum().item<int64_t>()
                  << ",\"bias_gradients_match\":" << (bias ? "true" : "false")
                  << ",\"relation_gradients_match\":" << (relations ? "true" : "false") << "}" << std::endl;
      }
    }
    return passed ? 0 : 1;
}
