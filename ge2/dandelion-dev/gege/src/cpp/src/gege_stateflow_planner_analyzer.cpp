#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"
#include "data/ordering.h"

namespace {

struct Args {
    std::string dataset_dir;
    int num_partitions = 16;
    int buffer_capacity = 4;
    bool randomly_assign_edge_buckets = false;
    bool allow_hybrid_cover = true;
    bool all_candidates = false;
    bool include_microstates = false;
    bool emit_json = false;
    int active_devices = 1;
    int64_t embedding_dim = 100;
    int64_t dtype_size = 4;
    int64_t optimizer_state_multiplier = 2;
    std::string force_family;
    std::string force_variant;
    double alpha = 1.0;
    double beta = 1e-7;
    double gamma = 0.95;
    double delta = 0.25;
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <num_partitions> <buffer_capacity>"
                 " [--random] [--no-hybrid] [--all-candidates] [--include-microstates] [--json]"
                 " [--multi-gpu <devices>] [--embedding-dim <int>] [--dtype-size <int>] [--optimizer-state-multiplier <int>]"
                 " [--force-family <name>] [--force-variant <name>]"
                 " [--alpha <double>] [--beta <double>] [--gamma <double>] [--delta <double>]\n";
}

Args parse_args(int argc, char **argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        throw std::runtime_error("Not enough arguments");
    }

    Args args;
    args.dataset_dir = argv[1];
    args.num_partitions = std::stoi(argv[2]);
    args.buffer_capacity = std::stoi(argv[3]);

    for (int idx = 4; idx < argc; idx++) {
        std::string arg = argv[idx];
        if (arg == "--random") {
            args.randomly_assign_edge_buckets = true;
        } else if (arg == "--no-hybrid") {
            args.allow_hybrid_cover = false;
        } else if (arg == "--all-candidates") {
            args.all_candidates = true;
        } else if (arg == "--include-microstates") {
            args.include_microstates = true;
        } else if (arg == "--json") {
            args.emit_json = true;
        } else if (arg == "--multi-gpu" && idx + 1 < argc) {
            args.active_devices = std::stoi(argv[++idx]);
        } else if (arg == "--embedding-dim" && idx + 1 < argc) {
            args.embedding_dim = std::max<int64_t>(1, std::stoll(argv[++idx]));
        } else if (arg == "--dtype-size" && idx + 1 < argc) {
            args.dtype_size = std::max<int64_t>(1, std::stoll(argv[++idx]));
        } else if (arg == "--optimizer-state-multiplier" && idx + 1 < argc) {
            args.optimizer_state_multiplier = std::max<int64_t>(1, std::stoll(argv[++idx]));
        } else if (arg == "--force-family" && idx + 1 < argc) {
            args.force_family = argv[++idx];
        } else if (arg == "--force-variant" && idx + 1 < argc) {
            args.force_variant = argv[++idx];
        } else if (arg == "--alpha" && idx + 1 < argc) {
            args.alpha = std::stod(argv[++idx]);
        } else if (arg == "--beta" && idx + 1 < argc) {
            args.beta = std::stod(argv[++idx]);
        } else if (arg == "--gamma" && idx + 1 < argc) {
            args.gamma = std::stod(argv[++idx]);
        } else if (arg == "--delta" && idx + 1 < argc) {
            args.delta = std::stod(argv[++idx]);
        } else {
            print_usage(argv[0]);
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return args;
}

std::vector<int64_t> read_bucket_sizes(const std::string &dataset_dir, int num_partitions) {
    std::string offsets_path = dataset_dir + "/edges/train_partition_offsets.txt";
    std::ifstream in(offsets_path);
    if (!in) {
        throw std::runtime_error("Failed to open offsets file: " + offsets_path);
    }

    std::vector<int64_t> values;
    int64_t value = 0;
    while (in >> value) {
        values.emplace_back(value);
    }

    const std::size_t expected = static_cast<std::size_t>(num_partitions) * static_cast<std::size_t>(num_partitions);
    if (values.size() != expected) {
        throw std::runtime_error("Expected " + std::to_string(expected) + " bucket sizes in " + offsets_path + ", found " +
                                 std::to_string(values.size()));
    }

    return values;
}

int64_t read_num_nodes(const std::string &dataset_dir) {
    std::string dataset_yaml_path = dataset_dir + "/dataset.yaml";
    std::ifstream in(dataset_yaml_path);
    if (!in) {
        return -1;
    }

    std::string line;
    while (std::getline(in, line)) {
        const std::string prefix = "num_nodes:";
        if (line.rfind(prefix, 0) != 0) {
            continue;
        }
        std::string value = line.substr(prefix.size());
        std::size_t first_non_space = value.find_first_not_of(" \t");
        if (first_non_space == std::string::npos) {
            return -1;
        }
        value = value.substr(first_non_space);
        try {
            return std::stoll(value);
        } catch (...) {
            return -1;
        }
    }

    return -1;
}

void set_weight_env(const char *name, double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    std::string encoded = oss.str();
    if (setenv(name, encoded.c_str(), 1) != 0) {
        throw std::runtime_error("Failed to set environment variable " + std::string(name));
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.emit_json) {
            spdlog::set_level(spdlog::level::off);
        }
        auto edge_bucket_sizes = read_bucket_sizes(args.dataset_dir, args.num_partitions);
        auto partition_row_counts = computePartitionRowCounts(read_num_nodes(args.dataset_dir), args.num_partitions);
        PlanEmbeddingLayout embedding_layout;
        embedding_layout.embedding_dim = args.embedding_dim;
        embedding_layout.dtype_size = args.dtype_size;
        embedding_layout.optimizer_state_multiplier = args.optimizer_state_multiplier;

        set_weight_env("GEGE_STATEFLOW_COST_ALPHA", args.alpha);
        set_weight_env("GEGE_STATEFLOW_COST_BETA", args.beta);
        set_weight_env("GEGE_STATEFLOW_COST_GAMMA", args.gamma);
        set_weight_env("GEGE_STATEFLOW_COST_DELTA", args.delta);
        if (!args.force_family.empty()) {
            if (setenv("GEGE_STATEFLOW_FORCE_FAMILY", args.force_family.c_str(), 1) != 0) {
                throw std::runtime_error("Failed to set GEGE_STATEFLOW_FORCE_FAMILY");
            }
        }
        if (!args.force_variant.empty()) {
            if (setenv("GEGE_STATEFLOW_FORCE_VARIANT", args.force_variant.c_str(), 1) != 0) {
                throw std::runtime_error("Failed to set GEGE_STATEFLOW_FORCE_VARIANT");
            }
        }

        if (args.active_devices > 1) {
            auto ordering = getBoundedGreedyCoverMultiGpuEdgeBucketOrdering(args.num_partitions, args.buffer_capacity,
                                                                            args.active_devices, edge_bucket_sizes);
            auto buffer_states = std::get<0>(ordering);
            auto edge_buckets_per_buffer = std::get<1>(ordering);
            if (buffer_states.empty() || edge_buckets_per_buffer.empty()) {
                std::cerr << "No multi-GPU buffer states produced\n";
                return 1;
            }
            if (args.all_candidates) {
                auto plans = enumerateMultiGpuStateflowPlans(buffer_states, edge_buckets_per_buffer,
                                                             args.active_devices, edge_bucket_sizes,
                                                             partition_row_counts, embedding_layout);
                if (args.emit_json) {
                    std::cout << "[";
                    for (std::size_t idx = 0; idx < plans.size(); idx++) {
                        if (idx > 0) {
                            std::cout << ",";
                        }
                        std::cout << stateflowPlanToJson(plans[idx], args.include_microstates);
                    }
                    std::cout << "]\n";
                    return 0;
                }

                std::cout << "dataset_dir=" << args.dataset_dir << "\n";
                std::cout << "num_partitions=" << args.num_partitions << "\n";
                std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
                std::cout << "active_devices=" << args.active_devices << "\n";
                std::cout << "weights={alpha:" << args.alpha << ",beta:" << args.beta
                          << ",gamma:" << args.gamma << ",delta:" << args.delta << "}\n";
                std::cout << "candidate_count=" << plans.size() << "\n";
                for (std::size_t idx = 0; idx < plans.size(); idx++) {
                    std::cout << "\n[candidate " << idx << "]\n";
                    std::cout << stateflowPlanToText(plans[idx], args.include_microstates);
                }
                return 0;
            }

            StateflowPlan plan = compileMultiGpuStateflowPlan(buffer_states, edge_buckets_per_buffer,
                                                              args.active_devices, edge_bucket_sizes,
                                                              partition_row_counts, embedding_layout);
            if (plan.family == PlanFamily::UNKNOWN) {
                std::cerr << "No valid multi-GPU Stateflow plan produced\n";
                return 1;
            }

            if (args.emit_json) {
                std::cout << stateflowPlanToJson(plan, args.include_microstates) << "\n";
                return 0;
            }

            std::cout << "dataset_dir=" << args.dataset_dir << "\n";
            std::cout << "num_partitions=" << args.num_partitions << "\n";
            std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
            std::cout << "active_devices=" << args.active_devices << "\n";
            std::cout << "weights={alpha:" << args.alpha << ",beta:" << args.beta
                      << ",gamma:" << args.gamma << ",delta:" << args.delta << "}\n";
            std::cout << stateflowPlanToText(plan, args.include_microstates);
            return 0;
        }

        if (args.all_candidates) {
            auto plans = enumerateSingleGpuStateflowPlans(args.num_partitions, args.buffer_capacity,
                                                          args.randomly_assign_edge_buckets, edge_bucket_sizes,
                                                          args.allow_hybrid_cover, partition_row_counts);
            if (args.emit_json) {
                std::cout << "[";
                for (std::size_t idx = 0; idx < plans.size(); idx++) {
                    if (idx > 0) {
                        std::cout << ",";
                    }
                    std::cout << stateflowPlanToJson(plans[idx], args.include_microstates);
                }
                std::cout << "]\n";
                return 0;
            }

            std::cout << "dataset_dir=" << args.dataset_dir << "\n";
            std::cout << "num_partitions=" << args.num_partitions << "\n";
            std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
            std::cout << "weights={alpha:" << args.alpha << ",beta:" << args.beta
                      << ",gamma:" << args.gamma << ",delta:" << args.delta << "}\n";
            std::cout << "candidate_count=" << plans.size() << "\n";
            for (std::size_t idx = 0; idx < plans.size(); idx++) {
                std::cout << "\n[candidate " << idx << "]\n";
                std::cout << stateflowPlanToText(plans[idx], args.include_microstates);
            }
            return 0;
        }

        StateflowPlan plan = compileSingleGpuStateflowPlan(args.num_partitions, args.buffer_capacity,
                                                           args.randomly_assign_edge_buckets, edge_bucket_sizes,
                                                           args.allow_hybrid_cover, partition_row_counts);
        if (plan.family == PlanFamily::UNKNOWN) {
            std::cerr << "No valid Stateflow plan produced\n";
            return 1;
        }

        if (args.emit_json) {
            std::cout << stateflowPlanToJson(plan, args.include_microstates) << "\n";
            return 0;
        }

        std::cout << "dataset_dir=" << args.dataset_dir << "\n";
        std::cout << "num_partitions=" << args.num_partitions << "\n";
        std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
        std::cout << "weights={alpha:" << args.alpha << ",beta:" << args.beta
                  << ",gamma:" << args.gamma << ",delta:" << args.delta << "}\n";
        std::cout << stateflowPlanToText(plan, args.include_microstates);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
