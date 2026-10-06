// Diagnostic only: keep local relation gradients separate in Adagrad.
#ifndef GEGE_CUDA
#error "Build this intervention with the native engine's CUDA and ABI settings."
#endif
#include <atomic>
#include "nn/model.h"

void Model::all_reduce(const std::vector<int64_t> &grad_scales) {
    torch::NoGradGuard guard;
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true)) {
        SPDLOG_INFO("[diagnostic-dense-steps] separate_local_gradients=1 averaged_update=0");
    }
    const auto lanes = device_models_.size();
    TORCH_CHECK(lanes > 1, "Separate local steps require multiple lanes");
    TORCH_CHECK(grad_scales.empty() || grad_scales.size() == lanes,
                "Dense intervention scale count differs from lane count");
    const auto keys = named_parameters().keys();
    std::vector<std::vector<torch::Tensor>> gradients(lanes);
    std::vector<bool> active(lanes, false);
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        TORCH_CHECK(grad_scales.empty() || grad_scales[lane] == 1,
                    "Separate local steps require dense_sync_batches=1");
        for (const auto &key : keys) {
            auto gradient = device_models_[lane]->named_parameters()[key].grad();
            active[lane] = active[lane] || gradient.defined();
            gradients[lane].push_back(gradient.defined() ? gradient.detach().clone() : torch::Tensor());
        }
    }
    // These gradients use pre-round weights. This is not serial SGD replay.
    for (std::size_t source = 0; source < lanes; ++source) {
        if (!active[source]) continue;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            for (std::size_t table = 0; table < keys.size(); ++table) {
                auto parameter = device_models_[lane]->named_parameters()[keys[table]];
                parameter.mutable_grad() = gradients[source][table].defined()
                    ? gradients[source][table].to(parameter.device(), false, true)
                    : torch::zeros_like(parameter);
            }
        }
        step_all();
        clear_grad_all();
    }
}
