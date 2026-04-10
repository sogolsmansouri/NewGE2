#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/cuda/CUDAGuard.h>

#include "common/util.h"
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
    bool use_inverse_relations = true;
    int64_t embedding_dim = 100;
    int64_t microbench_batches = 8;
    int64_t microbench_warmup = 3;
    int64_t microbench_iters = 10;
    bool microbench = true;
};

struct BatchSample {
    torch::Tensor edge_src;
    torch::Tensor edge_dst;
    torch::Tensor src_neg;
    torch::Tensor dst_neg;
    int64_t direct_ids = 0;
    int64_t compact_unique = 0;
    int64_t positive_unique = 0;
};

struct BenchStats {
    int64_t batches = 0;
    double direct_mean_ms = 0.0;
    double compact_mean_ms = 0.0;
    double direct_p50_ms = 0.0;
    double direct_p90_ms = 0.0;
    double compact_p50_ms = 0.0;
    double compact_p90_ms = 0.0;
    double compact_over_direct = 0.0;
};

struct StateStats {
    int64_t state_idx = -1;
    int64_t active_rows = 0;
    int64_t positive_edges = 0;
    int64_t batches = 0;
    int64_t avg_edges_per_batch = 0;
    int64_t avg_positive_unique_per_batch = 0;
    int64_t max_positive_unique_per_batch = 0;
    int64_t avg_compact_unique_per_batch = 0;
    int64_t max_compact_unique_per_batch = 0;
    int64_t avg_direct_ids_per_batch = 0;
    int64_t max_direct_ids_per_batch = 0;
    double avg_positive_read_amplification = 0.0;
    double avg_total_read_amplification = 0.0;
    int64_t shared_uniform_ids_per_batch = 0;
    int64_t degree_ids_per_side_per_batch = 0;
    int64_t total_negative_ids_per_batch = 0;
    double analyze_ms = 0.0;
    BenchStats bench;
};

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]"
                 " [--seed <int64>] [--batch-size <int64>] [--num-chunks <int64>]"
                 " [--negatives-per-positive <int64>] [--degree-fraction <float>]"
                 " [--state-idx <int64>] [--all-states] [--use-inverse-relations 0|1]"
                 " [--embedding-dim <int64>] [--microbench-batches <int64>]"
                 " [--microbench-warmup <int64>] [--microbench-iters <int64>] [--microbench 0|1]\n";
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
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--regroup") {
            args.regroup = true;
        } else if (arg == "--access-aware") {
            args.access_aware = true;
        } else if (arg == "--access-aware-generate") {
            args.access_aware_generate = true;
        } else if (arg == "--seed") {
            args.seed = std::stoll(require_value("--seed"));
        } else if (arg == "--batch-size") {
            args.batch_size = std::max<int64_t>(1, std::stoll(require_value("--batch-size")));
        } else if (arg == "--num-chunks") {
            args.num_chunks = std::max<int64_t>(1, std::stoll(require_value("--num-chunks")));
        } else if (arg == "--negatives-per-positive") {
            args.negatives_per_positive = std::max<int64_t>(1, std::stoll(require_value("--negatives-per-positive")));
        } else if (arg == "--degree-fraction") {
            args.degree_fraction = std::stod(require_value("--degree-fraction"));
        } else if (arg == "--state-idx") {
            args.state_idx = std::stoll(require_value("--state-idx"));
        } else if (arg == "--all-states") {
            args.analyze_all_states = true;
        } else if (arg == "--use-inverse-relations") {
            args.use_inverse_relations = std::stoll(require_value("--use-inverse-relations")) != 0;
        } else if (arg == "--embedding-dim") {
            args.embedding_dim = std::max<int64_t>(1, std::stoll(require_value("--embedding-dim")));
        } else if (arg == "--microbench-batches") {
            args.microbench_batches = std::max<int64_t>(0, std::stoll(require_value("--microbench-batches")));
        } else if (arg == "--microbench-warmup") {
            args.microbench_warmup = std::max<int64_t>(0, std::stoll(require_value("--microbench-warmup")));
        } else if (arg == "--microbench-iters") {
            args.microbench_iters = std::max<int64_t>(1, std::stoll(require_value("--microbench-iters")));
        } else if (arg == "--microbench") {
            args.microbench = std::stoll(require_value("--microbench")) != 0;
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

int64_t read_num_nodes(const std::string &dataset_yaml_path) {
    std::ifstream in(dataset_yaml_path);
    if (!in) {
        throw std::runtime_error("Failed to open dataset yaml: " + dataset_yaml_path);
    }
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find("num_nodes:");
        if (pos != std::string::npos) {
            std::string value = line.substr(pos + std::string("num_nodes:").size());
            value.erase(0, value.find_first_not_of(" \t"));
            return std::stoll(value);
        }
    }
    throw std::runtime_error("num_nodes not found in dataset yaml");
}

int infer_edge_columns(const std::string &path, int64_t num_edges) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open edges file: " + path);
    }
    int64_t bytes = static_cast<int64_t>(in.tellg());
    if (num_edges <= 0) {
        throw std::runtime_error("num_edges must be positive");
    }
    int64_t bytes_per_edge = bytes / num_edges;
    if (bytes % num_edges != 0) {
        throw std::runtime_error("Edge file size does not divide evenly by number of edges");
    }
    if (bytes_per_edge == 8 || bytes_per_edge == 16) return 2;
    if (bytes_per_edge == 12 || bytes_per_edge == 24) return 3;
    throw std::runtime_error("Unsupported bytes per edge: " + std::to_string(bytes_per_edge));
}

int infer_edge_dtype_bytes(const std::string &path, int64_t num_edges, int edge_cols) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open edges file: " + path);
    }
    int64_t bytes = static_cast<int64_t>(in.tellg());
    int64_t bytes_per_value = bytes / (num_edges * edge_cols);
    if (bytes_per_value != 4 && bytes_per_value != 8) {
        throw std::runtime_error("Unsupported edge value width: " + std::to_string(bytes_per_value));
    }
    return static_cast<int>(bytes_per_value);
}

int64_t decode_value(const char *ptr, int dtype_bytes) {
    if (dtype_bytes == 4) {
        int32_t value = 0;
        std::memcpy(&value, ptr, sizeof(int32_t));
        return static_cast<int64_t>(value);
    }
    int64_t value = 0;
    std::memcpy(&value, ptr, sizeof(int64_t));
    return value;
}

double bytes_to_gib(int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1024ll * 1024ll * 1024ll);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    double pos = (values.size() - 1) * p;
    auto lo = static_cast<size_t>(pos);
    auto hi = std::min(lo + 1, values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return values[lo] + (values[hi] - values[lo]) * frac;
}

int64_t partition_row_count(int64_t partition_id, int64_t partition_size, int64_t num_nodes) {
    int64_t start = partition_id * partition_size;
    if (start >= num_nodes) {
        return 0;
    }
    return std::min<int64_t>(partition_size, num_nodes - start);
}

torch::Tensor build_random_uniform_ids(std::mt19937_64 &rng, int64_t domain_rows, int64_t rows, int64_t cols) {
    if (cols <= 0) {
        return torch::Tensor();
    }
    auto out = torch::empty({rows, cols}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = out.accessor<int64_t, 2>();
    std::uniform_int_distribution<int64_t> dist(0, domain_rows - 1);
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            acc[i][j] = dist(rng);
        }
    }
    return out;
}

torch::Tensor build_random_edge_ids(std::mt19937_64 &rng, int64_t batch_edges, int64_t num_chunks, int64_t num_degree) {
    if (num_degree <= 0) {
        return torch::Tensor();
    }
    auto out = torch::empty({num_chunks, num_degree}, torch::TensorOptions().dtype(torch::kInt64));
    auto acc = out.accessor<int64_t, 2>();
    std::uniform_int_distribution<int64_t> dist(0, batch_edges - 1);
    for (int64_t chunk = 0; chunk < num_chunks; chunk++) {
        for (int64_t col = 0; col < num_degree; col++) {
            acc[chunk][col] = dist(rng);
        }
    }
    return out;
}

torch::Tensor materialize_edge_endpoint(torch::Tensor edge_nodes, torch::Tensor sample_edge_ids) {
    if (!sample_edge_ids.defined() || sample_edge_ids.numel() == 0) {
        return torch::Tensor();
    }
    return edge_nodes.index_select(0, sample_edge_ids.reshape({-1})).view_as(sample_edge_ids);
}

void mark_batch_tensor_rows(torch::Tensor tensor, std::vector<uint32_t> &seen, uint32_t generation, int64_t &unique_count) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return;
    }
    torch::Tensor flat = tensor.flatten().to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = flat.data_ptr<int64_t>();
    for (int64_t i = 0; i < flat.numel(); i++) {
        int64_t row = data[i];
        if (row < 0 || static_cast<std::size_t>(row) >= seen.size()) {
            throw std::runtime_error("Local row out of range during direct-state-local analysis");
        }
        auto idx = static_cast<std::size_t>(row);
        if (seen[idx] != generation) {
            seen[idx] = generation;
            unique_count++;
        }
    }
}

BenchStats run_microbench(const Args &args, const std::vector<BatchSample> &samples, int64_t active_rows) {
    BenchStats stats;
    if (!args.microbench || args.microbench_batches <= 0 || samples.empty()) {
        return stats;
    }
    TORCH_CHECK(torch::cuda::is_available(), "CUDA required for direct-state-local microbench");

    c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
    torch::NoGradGuard no_grad;
    auto device = torch::Device(torch::kCUDA, 0);
    auto emb_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto embedding = torch::randn({active_rows, args.embedding_dim}, emb_opts);

    auto bench_batches = std::min<std::size_t>(samples.size(), static_cast<std::size_t>(args.microbench_batches));
    std::vector<torch::Tensor> edge_src;
    std::vector<torch::Tensor> edge_dst;
    std::vector<torch::Tensor> src_neg;
    std::vector<torch::Tensor> dst_neg;
    edge_src.reserve(bench_batches);
    edge_dst.reserve(bench_batches);
    src_neg.reserve(bench_batches);
    dst_neg.reserve(bench_batches);
    for (std::size_t i = 0; i < bench_batches; i++) {
        edge_src.emplace_back(samples[i].edge_src.to(device));
        edge_dst.emplace_back(samples[i].edge_dst.to(device));
        src_neg.emplace_back(samples[i].src_neg.defined() ? samples[i].src_neg.to(device) : torch::Tensor());
        dst_neg.emplace_back(samples[i].dst_neg.defined() ? samples[i].dst_neg.to(device) : torch::Tensor());
    }

    auto run_direct = [&](std::vector<double> *latencies) {
        for (std::size_t i = 0; i < bench_batches; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor acc = embedding.index_select(0, edge_src[i]);
            acc = acc + embedding.index_select(0, edge_dst[i]);
            if (src_neg[i].defined()) {
                acc = acc + embedding.index_select(0, src_neg[i].reshape({-1})).sum(0, false);
            }
            if (dst_neg[i].defined()) {
                acc = acc + embedding.index_select(0, dst_neg[i].reshape({-1})).sum(0, false);
            }
            auto sink = acc.sum().item<float>();
            (void)sink;
            AT_CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            if (latencies != nullptr) {
                latencies->emplace_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
    };

    auto compact_active = [](torch::Tensor unique, torch::Tensor active_mask) {
        if (!active_mask.defined() || active_mask.numel() != unique.numel()) {
            return unique;
        }
        return unique.masked_select(active_mask.to(unique.device()).to(torch::kBool));
    };

    auto run_compact = [&](std::vector<double> *latencies) {
        for (std::size_t i = 0; i < bench_batches; i++) {
            std::vector<torch::Tensor> ids = {edge_src[i], edge_dst[i]};
            if (src_neg[i].defined()) {
                ids.emplace_back(src_neg[i].flatten());
            }
            if (dst_neg[i].defined()) {
                ids.emplace_back(dst_neg[i].flatten());
            }

            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor active_mask;
            auto map_result = map_tensors(ids, true, nullptr, &active_mask, active_rows);
            torch::Tensor unique = compact_active(std::get<0>(map_result), active_mask);
            auto mapped = std::get<1>(map_result);
            torch::Tensor loaded = embedding.index_select(0, unique);
            torch::Tensor acc = loaded.index_select(0, mapped[0]);
            acc = acc + loaded.index_select(0, mapped[1]);
            std::size_t idx = 2;
            if (src_neg[i].defined()) {
                acc = acc + loaded.index_select(0, mapped[idx++]).sum(0, false);
            }
            if (dst_neg[i].defined()) {
                acc = acc + loaded.index_select(0, mapped[idx++]).sum(0, false);
            }
            auto sink = acc.sum().item<float>();
            (void)sink;
            AT_CUDA_CHECK(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            if (latencies != nullptr) {
                latencies->emplace_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
    };

    for (int64_t i = 0; i < args.microbench_warmup; i++) {
        run_direct(nullptr);
        run_compact(nullptr);
    }

    std::vector<double> direct_latencies;
    std::vector<double> compact_latencies;
    direct_latencies.reserve(static_cast<std::size_t>(args.microbench_iters * static_cast<int64_t>(bench_batches)));
    compact_latencies.reserve(static_cast<std::size_t>(args.microbench_iters * static_cast<int64_t>(bench_batches)));

    for (int64_t iter = 0; iter < args.microbench_iters; iter++) {
        run_direct(&direct_latencies);
        run_compact(&compact_latencies);
    }

    stats.batches = static_cast<int64_t>(bench_batches);
    stats.direct_mean_ms = direct_latencies.empty() ? 0.0
                                                    : std::accumulate(direct_latencies.begin(), direct_latencies.end(), 0.0) /
                                                          static_cast<double>(direct_latencies.size());
    stats.compact_mean_ms = compact_latencies.empty() ? 0.0
                                                      : std::accumulate(compact_latencies.begin(), compact_latencies.end(), 0.0) /
                                                            static_cast<double>(compact_latencies.size());
    stats.direct_p50_ms = percentile(direct_latencies, 0.50);
    stats.direct_p90_ms = percentile(direct_latencies, 0.90);
    stats.compact_p50_ms = percentile(compact_latencies, 0.50);
    stats.compact_p90_ms = percentile(compact_latencies, 0.90);
    stats.compact_over_direct = stats.direct_mean_ms > 0.0 ? stats.compact_mean_ms / stats.direct_mean_ms : 0.0;
    return stats;
}

StateStats analyze_state(const Args &args,
                         int64_t state_idx,
                         const std::vector<int64_t> &bucket_sizes,
                         int64_t num_nodes,
                         int edge_cols,
                         int dtype_bytes,
                         const std::string &edge_path,
                         const std::vector<torch::Tensor> &buffer_states,
                         const std::vector<torch::Tensor> &edge_buckets_per_buffer) {
    if (state_idx < 0 || static_cast<std::size_t>(state_idx) >= buffer_states.size()) {
        throw std::runtime_error("state_idx out of range");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> state_partitions = tensor_to_vector(buffer_states[state_idx]);
    int64_t partition_size = static_cast<int64_t>(std::ceil(static_cast<double>(num_nodes) / static_cast<double>(args.num_partitions)));
    std::unordered_map<int64_t, int64_t> partition_to_local_base;
    partition_to_local_base.reserve(state_partitions.size());
    int64_t active_rows = 0;
    for (auto part : state_partitions) {
        partition_to_local_base[part] = active_rows;
        active_rows += partition_row_count(part, partition_size, num_nodes);
    }

    std::vector<int64_t> bucket_offsets(bucket_sizes.size(), 0);
    {
        int64_t running = 0;
        for (std::size_t i = 0; i < bucket_sizes.size(); i++) {
            bucket_offsets[i] = running;
            running += bucket_sizes[i];
        }
    }

    torch::Tensor buckets_tensor = edge_buckets_per_buffer[state_idx].to(torch::kCPU).to(torch::kInt64).contiguous();
    auto buckets = buckets_tensor.accessor<int64_t, 2>();

    std::ifstream edge_in(edge_path, std::ios::binary);
    if (!edge_in) {
        throw std::runtime_error("Failed to open edge file");
    }

    constexpr int64_t rows_per_chunk = 1 << 16;
    std::vector<char> raw(static_cast<std::size_t>(rows_per_chunk * edge_cols * dtype_bytes));

    auto localize = [&](int64_t global_id) -> int64_t {
        int64_t partition_id = global_id / partition_size;
        auto it = partition_to_local_base.find(partition_id);
        if (it == partition_to_local_base.end()) {
            throw std::runtime_error("Edge node outside selected state");
        }
        int64_t partition_start = partition_id * partition_size;
        int64_t local_offset = global_id - partition_start;
        int64_t rows_in_partition = partition_row_count(partition_id, partition_size, num_nodes);
        if (local_offset < 0 || local_offset >= rows_in_partition) {
            throw std::runtime_error("Local offset outside active partition rows");
        }
        return it->second + local_offset;
    };

    int64_t num_degree = static_cast<int64_t>(std::llround(static_cast<double>(args.negatives_per_positive) * args.degree_fraction));
    num_degree = std::max<int64_t>(0, std::min<int64_t>(args.negatives_per_positive, num_degree));
    int64_t num_uniform = args.negatives_per_positive - num_degree;
    bool need_src_negatives = edge_cols == 3 && args.use_inverse_relations;
    int64_t shared_uniform_ids_per_batch = args.num_chunks * num_uniform;
    int64_t degree_ids_per_side_per_batch = args.num_chunks * num_degree;
    int64_t total_negative_ids_per_batch =
        (need_src_negatives ? 2 : 1) * args.num_chunks * (num_uniform + num_degree);

    std::mt19937_64 rng(static_cast<uint64_t>(args.seed) ^ static_cast<uint64_t>(state_idx * 0x9e3779b97f4a7c15ULL));

    std::vector<uint32_t> seen(static_cast<std::size_t>(active_rows), 0);
    uint32_t generation = 1;

    std::vector<int64_t> positive_unique_counts;
    std::vector<int64_t> compact_unique_counts;
    std::vector<int64_t> direct_id_counts;
    std::vector<BatchSample> bench_samples;

    std::vector<int64_t> current_src;
    std::vector<int64_t> current_dst;
    current_src.reserve(static_cast<std::size_t>(args.batch_size));
    current_dst.reserve(static_cast<std::size_t>(args.batch_size));

    auto finalize_batch = [&]() {
        if (current_src.empty()) {
            return;
        }
        auto src_tensor = torch::from_blob(current_src.data(), {static_cast<int64_t>(current_src.size())},
                                           torch::TensorOptions().dtype(torch::kInt64))
                              .clone();
        auto dst_tensor = torch::from_blob(current_dst.data(), {static_cast<int64_t>(current_dst.size())},
                                           torch::TensorOptions().dtype(torch::kInt64))
                              .clone();

        int64_t positive_unique = 0;
        mark_batch_tensor_rows(src_tensor, seen, generation, positive_unique);
        mark_batch_tensor_rows(dst_tensor, seen, generation, positive_unique);
        generation++;

        torch::Tensor uniform_ids = build_random_uniform_ids(rng, active_rows, args.num_chunks, num_uniform);
        torch::Tensor dst_edge_ids = build_random_edge_ids(rng, src_tensor.size(0), args.num_chunks, num_degree);
        torch::Tensor dst_degree = materialize_edge_endpoint(dst_tensor, dst_edge_ids);
        torch::Tensor dst_neg =
            dst_degree.defined() && uniform_ids.defined() ? torch::cat({dst_degree, uniform_ids}, 1)
                                                          : (dst_degree.defined() ? dst_degree : uniform_ids);

        torch::Tensor src_neg;
        if (need_src_negatives) {
            torch::Tensor src_edge_ids = build_random_edge_ids(rng, src_tensor.size(0), args.num_chunks, num_degree);
            torch::Tensor src_degree = materialize_edge_endpoint(src_tensor, src_edge_ids);
            src_neg = src_degree.defined() && uniform_ids.defined() ? torch::cat({src_degree, uniform_ids}, 1)
                                                                    : (src_degree.defined() ? src_degree : uniform_ids);
        }

        int64_t compact_unique = 0;
        mark_batch_tensor_rows(src_tensor, seen, generation, compact_unique);
        mark_batch_tensor_rows(dst_tensor, seen, generation, compact_unique);
        mark_batch_tensor_rows(dst_neg, seen, generation, compact_unique);
        if (src_neg.defined()) {
            mark_batch_tensor_rows(src_neg, seen, generation, compact_unique);
        }
        generation++;
        if (generation == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            generation = 1;
        }

        int64_t direct_ids = src_tensor.numel() + dst_tensor.numel() + dst_neg.numel() + (src_neg.defined() ? src_neg.numel() : 0);
        positive_unique_counts.emplace_back(positive_unique);
        compact_unique_counts.emplace_back(compact_unique);
        direct_id_counts.emplace_back(direct_ids);

        if (static_cast<int64_t>(bench_samples.size()) < args.microbench_batches) {
            BatchSample sample;
            sample.edge_src = src_tensor;
            sample.edge_dst = dst_tensor;
            sample.src_neg = src_neg;
            sample.dst_neg = dst_neg;
            sample.direct_ids = direct_ids;
            sample.compact_unique = compact_unique;
            sample.positive_unique = positive_unique;
            bench_samples.emplace_back(std::move(sample));
        }

        current_src.clear();
        current_dst.clear();
    };

    int64_t positive_edges = 0;
    for (int64_t bucket_pos = 0; bucket_pos < buckets_tensor.size(0); bucket_pos++) {
        int64_t src_part = buckets[bucket_pos][0];
        int64_t dst_part = buckets[bucket_pos][1];
        int64_t bucket_idx = src_part * args.num_partitions + dst_part;
        int64_t bucket_size = bucket_sizes.at(static_cast<std::size_t>(bucket_idx));
        int64_t bucket_offset = bucket_offsets.at(static_cast<std::size_t>(bucket_idx));
        int64_t rows_remaining = bucket_size;
        int64_t consumed = 0;
        while (rows_remaining > 0) {
            int64_t rows_to_read = std::min<int64_t>(rows_remaining, rows_per_chunk);
            std::streamoff byte_offset = static_cast<std::streamoff>((bucket_offset + consumed) * edge_cols * dtype_bytes);
            edge_in.seekg(byte_offset, std::ios::beg);
            edge_in.read(raw.data(), static_cast<std::streamsize>(rows_to_read * edge_cols * dtype_bytes));
            if (!edge_in) {
                throw std::runtime_error("Failed to read edge chunk");
            }
            for (int64_t row = 0; row < rows_to_read; row++) {
                const char *base = raw.data() + row * edge_cols * dtype_bytes;
                int64_t src = decode_value(base, dtype_bytes);
                int64_t dst = decode_value(base + (edge_cols - 1) * dtype_bytes, dtype_bytes);
                current_src.emplace_back(localize(src));
                current_dst.emplace_back(localize(dst));
                positive_edges++;
                if (static_cast<int64_t>(current_src.size()) == args.batch_size) {
                    finalize_batch();
                }
            }
            consumed += rows_to_read;
            rows_remaining -= rows_to_read;
        }
    }
    finalize_batch();

    StateStats stats;
    stats.state_idx = state_idx;
    stats.active_rows = active_rows;
    stats.positive_edges = positive_edges;
    stats.batches = static_cast<int64_t>(positive_unique_counts.size());
    stats.avg_edges_per_batch = stats.batches > 0 ? positive_edges / stats.batches : 0;
    stats.avg_positive_unique_per_batch =
        stats.batches > 0 ? std::accumulate(positive_unique_counts.begin(), positive_unique_counts.end(), int64_t{0}) / stats.batches : 0;
    stats.max_positive_unique_per_batch =
        positive_unique_counts.empty() ? 0 : *std::max_element(positive_unique_counts.begin(), positive_unique_counts.end());
    stats.avg_compact_unique_per_batch =
        stats.batches > 0 ? std::accumulate(compact_unique_counts.begin(), compact_unique_counts.end(), int64_t{0}) / stats.batches : 0;
    stats.max_compact_unique_per_batch =
        compact_unique_counts.empty() ? 0 : *std::max_element(compact_unique_counts.begin(), compact_unique_counts.end());
    stats.avg_direct_ids_per_batch =
        stats.batches > 0 ? std::accumulate(direct_id_counts.begin(), direct_id_counts.end(), int64_t{0}) / stats.batches : 0;
    stats.max_direct_ids_per_batch =
        direct_id_counts.empty() ? 0 : *std::max_element(direct_id_counts.begin(), direct_id_counts.end());
    stats.avg_positive_read_amplification =
        stats.avg_positive_unique_per_batch > 0 ? static_cast<double>(2 * stats.avg_edges_per_batch) / static_cast<double>(stats.avg_positive_unique_per_batch)
                                                : 0.0;
    stats.avg_total_read_amplification =
        stats.avg_compact_unique_per_batch > 0 ? static_cast<double>(stats.avg_direct_ids_per_batch) / static_cast<double>(stats.avg_compact_unique_per_batch)
                                               : 0.0;
    stats.shared_uniform_ids_per_batch = shared_uniform_ids_per_batch;
    stats.degree_ids_per_side_per_batch = degree_ids_per_side_per_batch;
    stats.total_negative_ids_per_batch = total_negative_ids_per_batch;
    stats.analyze_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    stats.bench = run_microbench(args, bench_samples, active_rows);
    return stats;
}

void print_state_report(const Args &args, const StateStats &stats) {
    std::cout << "state_idx=" << stats.state_idx << "\n";
    std::cout << "active_rows=" << stats.active_rows << "\n";
    std::cout << "positive_edges=" << stats.positive_edges << "\n";
    std::cout << "batches=" << stats.batches << "\n";
    std::cout << "avg_edges_per_batch=" << stats.avg_edges_per_batch << "\n";
    std::cout << "avg_positive_unique_per_batch=" << stats.avg_positive_unique_per_batch << "\n";
    std::cout << "max_positive_unique_per_batch=" << stats.max_positive_unique_per_batch << "\n";
    std::cout << "avg_compact_unique_per_batch=" << stats.avg_compact_unique_per_batch << "\n";
    std::cout << "max_compact_unique_per_batch=" << stats.max_compact_unique_per_batch << "\n";
    std::cout << "avg_direct_ids_per_batch=" << stats.avg_direct_ids_per_batch << "\n";
    std::cout << "max_direct_ids_per_batch=" << stats.max_direct_ids_per_batch << "\n";
    std::cout << "avg_positive_read_amplification=" << std::fixed << std::setprecision(6) << stats.avg_positive_read_amplification << "\n";
    std::cout << "avg_total_read_amplification=" << std::fixed << std::setprecision(6) << stats.avg_total_read_amplification << "\n";
    std::cout << "assumptions.num_chunks=" << args.num_chunks << "\n";
    std::cout << "assumptions.negatives_per_positive=" << args.negatives_per_positive << "\n";
    std::cout << "assumptions.degree_fraction=" << std::fixed << std::setprecision(3) << args.degree_fraction << "\n";
    std::cout << "assumptions.use_inverse_relations=" << args.use_inverse_relations << "\n";
    std::cout << "assumptions.shared_uniform_ids_per_batch=" << stats.shared_uniform_ids_per_batch << "\n";
    std::cout << "assumptions.degree_ids_per_side_per_batch=" << stats.degree_ids_per_side_per_batch << "\n";
    std::cout << "assumptions.total_negative_ids_per_batch=" << stats.total_negative_ids_per_batch << "\n";
    std::cout << "analyze_ms=" << std::fixed << std::setprecision(3) << stats.analyze_ms << "\n";
    if (stats.bench.batches > 0) {
        std::cout << "microbench.batches=" << stats.bench.batches << "\n";
        std::cout << "microbench.direct_mean_ms=" << std::fixed << std::setprecision(6) << stats.bench.direct_mean_ms << "\n";
        std::cout << "microbench.direct_p50_ms=" << stats.bench.direct_p50_ms << "\n";
        std::cout << "microbench.direct_p90_ms=" << stats.bench.direct_p90_ms << "\n";
        std::cout << "microbench.compact_mean_ms=" << stats.bench.compact_mean_ms << "\n";
        std::cout << "microbench.compact_p50_ms=" << stats.bench.compact_p50_ms << "\n";
        std::cout << "microbench.compact_p90_ms=" << stats.bench.compact_p90_ms << "\n";
        std::cout << "microbench.compact_over_direct=" << stats.bench.compact_over_direct << "\n";
    }
    std::cout << "workspace.embedding_gib=" << std::fixed << std::setprecision(6)
              << bytes_to_gib(stats.active_rows * args.embedding_dim * static_cast<int64_t>(sizeof(float))) << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);
        std::srand(static_cast<unsigned int>(args.seed));
        torch::manual_seed(args.seed);

        std::string edge_path = args.dataset_dir + "/edges/train_edges.bin";
        std::string offsets_path = args.dataset_dir + "/edges/train_partition_offsets.txt";
        std::string dataset_yaml_path = args.dataset_dir + "/dataset.yaml";
        auto bucket_sizes = read_offsets(offsets_path);
        if (bucket_sizes.size() != static_cast<std::size_t>(args.num_partitions * args.num_partitions)) {
            throw std::runtime_error("Unexpected number of edge buckets in offsets file");
        }
        int64_t num_edges = std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), static_cast<int64_t>(0));
        int64_t num_nodes = read_num_nodes(dataset_yaml_path);
        int edge_cols = infer_edge_columns(edge_path, num_edges);
        int dtype_bytes = infer_edge_dtype_bytes(edge_path, num_edges, edge_cols);

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
        std::cout << "num_nodes=" << num_nodes << "\n";
        std::cout << "num_edges=" << num_edges << "\n";
        std::cout << "num_states=" << buffer_states.size() << "\n";
        std::cout << "num_partitions=" << args.num_partitions << "\n";
        std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
        std::cout << "ordering_enum=" << static_cast<int>(args.ordering) << "\n";
        std::cout << "edge_cols=" << edge_cols << "\n";

        if (args.analyze_all_states) {
            for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
                auto stats = analyze_state(args, static_cast<int64_t>(state_idx), bucket_sizes, num_nodes, edge_cols, dtype_bytes,
                                           edge_path, buffer_states, edge_buckets_per_buffer);
                std::cout << "=== state " << state_idx << " ===\n";
                print_state_report(args, stats);
            }
        } else {
            auto stats = analyze_state(args, args.state_idx, bucket_sizes, num_nodes, edge_cols, dtype_bytes, edge_path,
                                       buffer_states, edge_buckets_per_buffer);
            print_state_report(args, stats);
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
