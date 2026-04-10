#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "configuration/options.h"
#include "data/ordering.h"

namespace {

struct Args {
    std::string dataset_dir;
    EdgeBucketOrdering ordering;
    int num_partitions;
    int buffer_capacity;
    int fine_to_coarse_ratio;
    int num_cache_partitions;
    bool randomly_assign_edge_buckets;
    int active_devices;
    bool regroup = false;
    bool access_aware = false;
    bool access_aware_generate = false;
    int64_t seed = 12345;
    int64_t batch_size = 50000;
    int64_t num_chunks = 50;
    int64_t negatives_per_positive = 1000;
    double degree_fraction = 0.5;
    int64_t state_idx = 0;
    bool analyze_all_states = false;
};

struct StateDescriptorStats {
    int64_t state_idx = -1;
    int64_t active_buckets = 0;
    int64_t active_edges = 0;
    int64_t batches = 0;
    int64_t avg_edges_per_batch = 0;
    int64_t max_edges_per_batch = 0;
    double slice_plan_ms = 0.0;
    double descriptor_plan_ms = 0.0;
    int64_t batch_descriptor_bytes = 0;
    int64_t negative_descriptor_bytes = 0;
    int64_t total_descriptor_bytes = 0;
};

constexpr int64_t kBatchDescriptorBytes = 32;
constexpr int64_t kNegativeDescriptorBytes = 40;

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]"
                 " [--seed <int64>] [--batch-size <int64>] [--num-chunks <int64>]"
                 " [--negatives-per-positive <int64>] [--degree-fraction <float>]"
                 " [--state-idx <int64>] [--all-states]\n";
}

Args parse_args(int argc, char **argv) {
    if (argc < 9) {
        print_usage(argv[0]);
        throw std::runtime_error("Not enough arguments");
    }

    Args args;
    args.dataset_dir = argv[1];
    args.ordering = getEdgeBucketOrderingEnum(argv[2]);
    args.num_partitions = std::stoi(argv[3]);
    args.buffer_capacity = std::stoi(argv[4]);
    args.fine_to_coarse_ratio = std::stoi(argv[5]);
    args.num_cache_partitions = std::stoi(argv[6]);
    args.randomly_assign_edge_buckets = std::stoi(argv[7]) != 0;
    args.active_devices = std::stoi(argv[8]);

    for (int i = 9; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--regroup") {
            args.regroup = true;
        } else if (arg == "--access-aware") {
            args.access_aware = true;
        } else if (arg == "--access-aware-generate") {
            args.access_aware_generate = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoll(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batch_size = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--num-chunks" && i + 1 < argc) {
            args.num_chunks = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--negatives-per-positive" && i + 1 < argc) {
            args.negatives_per_positive = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--degree-fraction" && i + 1 < argc) {
            args.degree_fraction = std::stod(argv[++i]);
        } else if (arg == "--state-idx" && i + 1 < argc) {
            args.state_idx = std::stoll(argv[++i]);
        } else if (arg == "--all-states") {
            args.analyze_all_states = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return args;
}

std::vector<int64_t> read_offsets(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open offsets file: " + path);
    }
    std::vector<int64_t> offsets;
    int64_t value = 0;
    while (in >> value) {
        offsets.emplace_back(value);
    }
    return offsets;
}

double bytes_to_gib(int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1024ll * 1024ll * 1024ll);
}

StateDescriptorStats analyze_state(const Args &args,
                                   int64_t state_idx,
                                   const std::vector<int64_t> &bucket_sizes,
                                   const std::vector<torch::Tensor> &buffer_states,
                                   const std::vector<torch::Tensor> &edge_buckets_per_buffer) {
    if (state_idx < 0 || static_cast<std::size_t>(state_idx) >= edge_buckets_per_buffer.size()) {
        throw std::runtime_error("state_idx out of range");
    }

    auto total_start = std::chrono::high_resolution_clock::now();
    auto slice_start = total_start;

    torch::Tensor buckets_tensor = edge_buckets_per_buffer[state_idx].to(torch::kCPU).to(torch::kInt64).contiguous();
    auto buckets = buckets_tensor.accessor<int64_t, 2>();

    std::vector<int64_t> slice_sizes;
    slice_sizes.reserve(static_cast<std::size_t>(buckets_tensor.size(0)) * 2);

    int64_t active_edges = 0;
    int64_t max_edges_per_batch = 0;
    for (int64_t bucket_pos = 0; bucket_pos < buckets_tensor.size(0); bucket_pos++) {
        int64_t src_part = buckets[bucket_pos][0];
        int64_t dst_part = buckets[bucket_pos][1];
        int64_t bucket_idx = src_part * args.num_partitions + dst_part;
        int64_t bucket_size = bucket_sizes.at(static_cast<std::size_t>(bucket_idx));
        active_edges += bucket_size;
        for (int64_t offset = 0; offset < bucket_size; offset += args.batch_size) {
            int64_t slice_size = std::min<int64_t>(args.batch_size, bucket_size - offset);
            slice_sizes.emplace_back(slice_size);
            max_edges_per_batch = std::max(max_edges_per_batch, slice_size);
        }
    }

    auto slice_end = std::chrono::high_resolution_clock::now();

    std::mt19937_64 rng(static_cast<uint64_t>(args.seed) ^ static_cast<uint64_t>(state_idx * 0x9e3779b97f4a7c15ULL));
    std::shuffle(slice_sizes.begin(), slice_sizes.end(), rng);

    std::vector<uint64_t> batch_seeds;
    batch_seeds.reserve(slice_sizes.size());
    for (std::size_t batch_idx = 0; batch_idx < slice_sizes.size(); batch_idx++) {
        uint64_t seed = rng();
        batch_seeds.emplace_back(seed);
    }
    auto total_end = std::chrono::high_resolution_clock::now();

    StateDescriptorStats stats;
    stats.state_idx = state_idx;
    stats.active_buckets = buckets_tensor.size(0);
    stats.active_edges = active_edges;
    stats.batches = static_cast<int64_t>(slice_sizes.size());
    stats.avg_edges_per_batch = stats.batches > 0 ? active_edges / stats.batches : 0;
    stats.max_edges_per_batch = max_edges_per_batch;
    stats.slice_plan_ms = std::chrono::duration<double, std::milli>(slice_end - slice_start).count();
    stats.descriptor_plan_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    stats.batch_descriptor_bytes = stats.batches * kBatchDescriptorBytes;
    stats.negative_descriptor_bytes = stats.batches * kNegativeDescriptorBytes;
    stats.total_descriptor_bytes = stats.batch_descriptor_bytes + stats.negative_descriptor_bytes;
    return stats;
}

void print_state_report(const Args &args,
                        const StateDescriptorStats &stats,
                        int64_t num_states) {
    std::cout << "state_idx=" << stats.state_idx << "\n";
    std::cout << "num_states=" << num_states << "\n";
    std::cout << "active_buckets=" << stats.active_buckets << "\n";
    std::cout << "active_edges=" << stats.active_edges << "\n";
    std::cout << "batches=" << stats.batches << "\n";
    std::cout << "avg_edges_per_batch=" << stats.avg_edges_per_batch << "\n";
    std::cout << "max_edges_per_batch=" << stats.max_edges_per_batch << "\n";
    std::cout << "assumptions.batch_size=" << args.batch_size << "\n";
    std::cout << "assumptions.num_chunks=" << args.num_chunks << "\n";
    std::cout << "assumptions.negatives_per_positive=" << args.negatives_per_positive << "\n";
    std::cout << "assumptions.degree_fraction=" << std::fixed << std::setprecision(3) << args.degree_fraction << "\n";
    std::cout << "assumptions.batch_descriptor_bytes=" << kBatchDescriptorBytes << "\n";
    std::cout << "assumptions.negative_descriptor_bytes=" << kNegativeDescriptorBytes << "\n";
    std::cout << "descriptor.slice_plan_ms=" << std::fixed << std::setprecision(3) << stats.slice_plan_ms << "\n";
    std::cout << "descriptor.total_plan_ms=" << std::fixed << std::setprecision(3) << stats.descriptor_plan_ms << "\n";
    std::cout << "descriptor.batch_descriptor_gib=" << std::fixed << std::setprecision(6) << bytes_to_gib(stats.batch_descriptor_bytes) << "\n";
    std::cout << "descriptor.negative_descriptor_gib=" << std::fixed << std::setprecision(6) << bytes_to_gib(stats.negative_descriptor_bytes) << "\n";
    std::cout << "descriptor.total_gib=" << std::fixed << std::setprecision(6) << bytes_to_gib(stats.total_descriptor_bytes) << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        std::srand(static_cast<unsigned int>(args.seed));
        torch::manual_seed(args.seed);

        std::string offsets_path = args.dataset_dir + "/edges/train_partition_offsets.txt";
        auto bucket_sizes = read_offsets(offsets_path);
        if (bucket_sizes.size() != static_cast<std::size_t>(args.num_partitions * args.num_partitions)) {
            throw std::runtime_error("Unexpected number of edge buckets in offsets file");
        }

        auto ordering_result = args.access_aware_generate
            ? getAccessAwareCustomEdgeBucketOrdering(args.num_partitions, args.buffer_capacity, args.active_devices)
            : getEdgeBucketOrdering(args.ordering, args.num_partitions, args.buffer_capacity, args.fine_to_coarse_ratio,
                                    args.num_cache_partitions, args.randomly_assign_edge_buckets);
        auto buffer_states = std::get<0>(ordering_result);
        auto edge_buckets_per_buffer = std::get<1>(ordering_result);

        if (!args.access_aware_generate && (args.access_aware || args.regroup)) {
            auto permutation = args.access_aware
                ? getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, args.active_devices)
                : getDisjointBufferStatePermutation(buffer_states, args.active_devices);
            std::vector<torch::Tensor> reordered_states;
            std::vector<torch::Tensor> reordered_buckets;
            reordered_states.reserve(buffer_states.size());
            reordered_buckets.reserve(edge_buckets_per_buffer.size());
            for (auto idx : permutation) {
                reordered_states.emplace_back(buffer_states[idx]);
                reordered_buckets.emplace_back(edge_buckets_per_buffer[idx]);
            }
            buffer_states = std::move(reordered_states);
            edge_buckets_per_buffer = std::move(reordered_buckets);
        }

        if (buffer_states.empty() || edge_buckets_per_buffer.size() != buffer_states.size()) {
            throw std::runtime_error("Invalid buffer ordering");
        }

        std::cout << "dataset_dir=" << args.dataset_dir << "\n";
        std::cout << "num_states=" << buffer_states.size() << "\n";
        std::cout << "num_partitions=" << args.num_partitions << "\n";
        std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
        std::cout << "ordering_enum=" << static_cast<int>(args.ordering) << "\n";

        if (args.analyze_all_states) {
            for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
                auto stats = analyze_state(args, static_cast<int64_t>(state_idx), bucket_sizes, buffer_states, edge_buckets_per_buffer);
                std::cout << "=== state " << state_idx << " ===\n";
                print_state_report(args, stats, static_cast<int64_t>(buffer_states.size()));
            }
        } else {
            auto stats = analyze_state(args, args.state_idx, bucket_sizes, buffer_states, edge_buckets_per_buffer);
            print_state_report(args, stats, static_cast<int64_t>(buffer_states.size()));
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
