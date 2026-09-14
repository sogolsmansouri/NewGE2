#include <fstream>
#include <iostream>
#include <sstream>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "storage/buffer.h"

// Exact integer updates test retention, partial admits, delayed host freshness,
// frame reuse, and the shortened final partition independently of training.
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const int p = std::stoi(argv[1]);
    const int rows = 4096;
    const int nodes = p * rows - 2;
    std::ifstream input(argv[2]);
    std::vector<torch::Tensor> states;
    std::string line;
    while (std::getline(input, line)) {
        const auto begin = line.find('[');
        const auto end = line.find(']');
        if (begin == std::string::npos || end == std::string::npos) return 2;
        line = line.substr(begin + 1, end - begin - 1);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream stream(line);
        std::vector<int64_t> state;
        int64_t id;
        while (stream >> id) state.push_back(id);
        if (!state.empty()) states.push_back(torch::tensor(state, torch::kInt64));
    }
    if (states.empty()) return 2;
    const int q = states.front().numel();
    TORCH_CHECK(q >= 2 && q <= p, "Invalid visible capacity");
    for (const auto &state : states) TORCH_CHECK(state.numel() == q, "Mixed state capacities");
    std::ofstream(argv[3]).close();
    auto host = torch::zeros({nodes, 8});
    auto expected = host.clone();
    MemPartitionBuffer buffer(q, p, 1, rows, 8, nodes, torch::kFloat32, argv[3], false, torch::Device(torch::kCUDA, 0));
    const int k = q + std::stoi(std::getenv("GEGE_FRAME_CACHE_HIDDEN_FRAMES"));
    const int hs = std::stoi(std::getenv("GEGE_FRAME_CACHE_MAX_STALE_BACKLOG"));
    for (int epoch = 0; epoch < 5; ++epoch) {
        buffer.setBufferOrdering(states);
        buffer.load(host);
        TORCH_CHECK(buffer.buffer_tensor_gpu_view_.size(0) == k * rows, "Physical allocation grew");
        for (std::size_t step = 0; step < states.size(); ++step) {
            auto map = buffer.getGlobalToLocalMap(true);
            auto global = torch::nonzero(map >= 0).flatten();
            auto local = map.index_select(0, global).to(torch::kCUDA);
            TORCH_CHECK(torch::equal(buffer.indexRead(local).cpu(), expected.index_select(0, global)), "Stale visible data");
            if (step + 1 < states.size()) buffer.startAsyncAdmitPreload();
            auto values = torch::full({global.size(0), 8}, epoch + step + 1.0f);
            expected.index_add_(0, global, values);
            buffer.indexAdd(local, values.to(torch::kCUDA));
            cudaEvent_t ready;
            TORCH_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) == cudaSuccess);
            TORCH_CHECK(cudaEventRecord(ready, c10::cuda::getCurrentCUDAStream().stream()) == cudaSuccess);
            if (step + 1 < states.size()) buffer.performNextSwap(reinterpret_cast<std::uintptr_t>(ready));
            TORCH_CHECK(cudaEventSynchronize(ready) == cudaSuccess);
            TORCH_CHECK(cudaEventDestroy(ready) == cudaSuccess);
        }
        buffer.unload(true);
        TORCH_CHECK(torch::equal(host, expected), "Final host state lost an update");
        TORCH_CHECK(buffer.getFrameCachePerfStats().stale_backlog_after_publish_max <= hs, "Stale quota exceeded");
    }
    std::cout << "fixed_frame_value_test=pass p=" << p << " q=" << q << " k=" << k << " epochs=5\n";
}
