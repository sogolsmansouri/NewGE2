#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "data/ordering.h"

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    std::ifstream input(std::string(argv[1]) + "/edges/train_partition_offsets.txt");
    std::vector<int64_t> weights;
    for (int64_t value; input >> value;) weights.push_back(value);
    const int partitions = std::stoi(argv[2]);
    if (weights.size() != static_cast<std::size_t>(partitions * partitions)) return 2;
    spdlog::set_level(spdlog::level::off);
    PlanEmbeddingLayout layout;
    layout.embedding_dim = 100;
    layout.dtype_size = 4;
    layout.optimizer_state_multiplier = 2;
    auto rows = computePartitionRowCounts(std::stoll(argv[3]), partitions);
    std::ostringstream output;
    output << "[";
    bool first = true;
    for (int64_t epoch = 0; epoch < 3; ++epoch) {
        auto ordering = getEpochRelabeledBoundedCoverOrdering(partitions, 4, weights, 17, epoch);
        for (int window : {1, 2, 4, 8, 12, 16}) {
            setenv("GEGE_STATEFLOW_ORDER_WINDOW", std::to_string(window).c_str(), 1);
            if (!first) output << ",";
            first = false;
            output << "{\"epoch\":" << epoch << ",\"window\":" << window;
            try {
                auto plan = compileMultiGpuStateflowPlan(std::get<0>(ordering), std::get<1>(ordering), 2,
                                                        weights, rows, layout);
                output << ",\"status\":\"passed\",\"plan\":" << stateflowPlanToJson(plan, true);
            } catch (const std::exception &error) {
                std::cerr << "epoch=" << epoch << " window=" << window << " " << error.what() << '\n';
                output << ",\"status\":\"search_failed\"";
            }
            output << "}";
        }
    }
    output << "]";
    std::cout << "\nSCHEDULE_PREFLIGHT_JSON=" << output.str() << '\n';
}
