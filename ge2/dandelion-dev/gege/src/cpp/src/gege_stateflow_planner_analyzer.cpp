#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "data/ordering.h"

namespace {

struct Args {
    std::string dataset_dir;
    int num_partitions = 16;
    int buffer_capacity = 4;
    bool randomly_assign_edge_buckets = false;
    bool allow_hybrid_cover = true;
    bool include_microstates = false;
    bool emit_json = false;
    double alpha = 1.0;
    double beta = 1e-7;
    double gamma = 0.95;
    double delta = 0.25;
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <num_partitions> <buffer_capacity>"
                 " [--random] [--no-hybrid] [--include-microstates] [--json]"
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
        } else if (arg == "--include-microstates") {
            args.include_microstates = true;
        } else if (arg == "--json") {
            args.emit_json = true;
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
        auto edge_bucket_sizes = read_bucket_sizes(args.dataset_dir, args.num_partitions);
        auto partition_row_counts = computePartitionRowCounts(read_num_nodes(args.dataset_dir), args.num_partitions);

        set_weight_env("GEGE_STATEFLOW_COST_ALPHA", args.alpha);
        set_weight_env("GEGE_STATEFLOW_COST_BETA", args.beta);
        set_weight_env("GEGE_STATEFLOW_COST_GAMMA", args.gamma);
        set_weight_env("GEGE_STATEFLOW_COST_DELTA", args.delta);

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
