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
};

struct StateAccumulatorStats {
    int64_t state_idx = -1;
    int64_t active_rows = 0;
    int64_t positive_edges = 0;
    int64_t batches = 0;
    int64_t avg_total_updates_per_batch = 0;
    int64_t p50_total_updates_per_batch = 0;
    int64_t p90_total_updates_per_batch = 0;
    int64_t max_total_updates_per_batch = 0;
    int64_t avg_unique_rows_per_batch = 0;
    int64_t p50_unique_rows_per_batch = 0;
    int64_t p90_unique_rows_per_batch = 0;
    int64_t max_unique_rows_per_batch = 0;
    double avg_duplicate_ratio = 0.0;
    double p50_duplicate_ratio = 0.0;
    double p90_duplicate_ratio = 0.0;
    int64_t avg_max_row_repeats = 0;
    int64_t p50_max_row_repeats = 0;
    int64_t p90_max_row_repeats = 0;
    int64_t max_max_row_repeats = 0;
    int64_t avg_rows_ge2 = 0;
    int64_t avg_rows_ge4 = 0;
    int64_t avg_rows_ge8 = 0;
    int64_t avg_rows_ge16 = 0;
    double avg_accumulator_mb_lf50 = 0.0;
    double avg_accumulator_mb_lf67 = 0.0;
    double avg_accumulator_mb_lf80 = 0.0;
    double p90_accumulator_mb_lf67 = 0.0;
    double max_accumulator_mb_lf67 = 0.0;
    double dense_grad_gib = 0.0;
    double analyze_ms = 0.0;
};

template <typename T>
T percentile(std::vector<T> values, double p) {
    if (values.empty()) {
        return T{};
    }
    std::sort(values.begin(), values.end());
    double pos = (values.size() - 1) * p;
    auto lo = static_cast<size_t>(pos);
    auto hi = std::min(lo + 1, values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return static_cast<T>(values[lo] + (values[hi] - values[lo]) * frac);
}

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
                 " [--embedding-dim <int64>]\n";
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

int64_t partition_row_count(int64_t partition_id, int64_t partition_size, int64_t num_nodes) {
    int64_t start = partition_id * partition_size;
    if (start >= num_nodes) {
        return 0;
    }
    return std::min<int64_t>(partition_size, num_nodes - start);
}

int64_t next_power_of_two(int64_t value) {
    int64_t capacity = 1;
    while (capacity < value) {
        capacity <<= 1;
    }
    return capacity;
}

double bytes_to_gib(int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1024ll * 1024ll * 1024ll);
}

double bytes_to_mib(int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1024ll * 1024ll);
}

double estimate_accumulator_mib(int64_t unique_rows, int64_t embedding_dim, double load_factor) {
    if (unique_rows <= 0) {
        return 0.0;
    }
    int64_t capacity = next_power_of_two(static_cast<int64_t>(std::ceil(static_cast<double>(unique_rows) / load_factor)));
    constexpr int64_t entry_metadata_bytes = 8;  // int32 key + occupancy/padding
    int64_t bytes = capacity * (entry_metadata_bytes + embedding_dim * static_cast<int64_t>(sizeof(float)));
    return bytes_to_mib(bytes);
}

StateAccumulatorStats analyze_state(const Args &args,
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
    int64_t running = 0;
    for (std::size_t i = 0; i < bucket_sizes.size(); i++) {
        bucket_offsets[i] = running;
        running += bucket_sizes[i];
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

    std::mt19937_64 rng(static_cast<uint64_t>(args.seed) ^ static_cast<uint64_t>(state_idx * 0x9e3779b97f4a7c15ULL));
    std::uniform_int_distribution<int64_t> uniform_row_dist(0, active_rows - 1);

    std::vector<int32_t> counts(static_cast<std::size_t>(active_rows), 0);
    std::vector<int64_t> touched_rows;
    touched_rows.reserve(static_cast<std::size_t>(args.batch_size * 4));

    std::vector<int64_t> total_updates_per_batch;
    std::vector<int64_t> unique_rows_per_batch;
    std::vector<double> duplicate_ratio_per_batch;
    std::vector<int64_t> max_row_repeats_per_batch;
    std::vector<int64_t> rows_ge2_per_batch;
    std::vector<int64_t> rows_ge4_per_batch;
    std::vector<int64_t> rows_ge8_per_batch;
    std::vector<int64_t> rows_ge16_per_batch;
    std::vector<double> accumulator_mib_lf50_per_batch;
    std::vector<double> accumulator_mib_lf67_per_batch;
    std::vector<double> accumulator_mib_lf80_per_batch;

    std::vector<int64_t> current_src;
    std::vector<int64_t> current_dst;
    current_src.reserve(static_cast<std::size_t>(args.batch_size));
    current_dst.reserve(static_cast<std::size_t>(args.batch_size));

    auto mark_row = [&](int64_t row) {
        auto idx = static_cast<std::size_t>(row);
        if (counts[idx] == 0) {
            touched_rows.emplace_back(row);
        }
        counts[idx] += 1;
    };

    auto finalize_batch = [&]() {
        if (current_src.empty()) {
            return;
        }

        touched_rows.clear();
        int64_t batch_edges = static_cast<int64_t>(current_src.size());

        for (int64_t row : current_src) mark_row(row);
        for (int64_t row : current_dst) mark_row(row);

        for (int64_t chunk = 0; chunk < args.num_chunks; chunk++) {
            for (int64_t j = 0; j < num_uniform; j++) {
                mark_row(uniform_row_dist(rng));
            }
        }

        if (num_degree > 0) {
            std::uniform_int_distribution<int64_t> edge_dist(0, batch_edges - 1);
            for (int64_t chunk = 0; chunk < args.num_chunks; chunk++) {
                for (int64_t j = 0; j < num_degree; j++) {
                    int64_t picked = edge_dist(rng);
                    mark_row(current_dst[static_cast<std::size_t>(picked)]);
                    if (need_src_negatives) {
                        mark_row(current_src[static_cast<std::size_t>(picked)]);
                    }
                }
            }
        }

        int64_t unique_rows = static_cast<int64_t>(touched_rows.size());
        int64_t total_updates = 2 * batch_edges + args.num_chunks * num_uniform + args.num_chunks * num_degree * (need_src_negatives ? 2 : 1);
        int64_t max_repeat = 0;
        int64_t rows_ge2 = 0;
        int64_t rows_ge4 = 0;
        int64_t rows_ge8 = 0;
        int64_t rows_ge16 = 0;
        for (int64_t row : touched_rows) {
            int32_t count = counts[static_cast<std::size_t>(row)];
            max_repeat = std::max<int64_t>(max_repeat, static_cast<int64_t>(count));
            rows_ge2 += count >= 2;
            rows_ge4 += count >= 4;
            rows_ge8 += count >= 8;
            rows_ge16 += count >= 16;
            counts[static_cast<std::size_t>(row)] = 0;
        }

        total_updates_per_batch.emplace_back(total_updates);
        unique_rows_per_batch.emplace_back(unique_rows);
        duplicate_ratio_per_batch.emplace_back(total_updates > 0 ? 1.0 - (static_cast<double>(unique_rows) / static_cast<double>(total_updates)) : 0.0);
        max_row_repeats_per_batch.emplace_back(max_repeat);
        rows_ge2_per_batch.emplace_back(rows_ge2);
        rows_ge4_per_batch.emplace_back(rows_ge4);
        rows_ge8_per_batch.emplace_back(rows_ge8);
        rows_ge16_per_batch.emplace_back(rows_ge16);
        accumulator_mib_lf50_per_batch.emplace_back(estimate_accumulator_mib(unique_rows, args.embedding_dim, 0.50));
        accumulator_mib_lf67_per_batch.emplace_back(estimate_accumulator_mib(unique_rows, args.embedding_dim, 0.67));
        accumulator_mib_lf80_per_batch.emplace_back(estimate_accumulator_mib(unique_rows, args.embedding_dim, 0.80));

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

    StateAccumulatorStats stats;
    stats.state_idx = state_idx;
    stats.active_rows = active_rows;
    stats.positive_edges = positive_edges;
    stats.batches = static_cast<int64_t>(unique_rows_per_batch.size());
    stats.avg_total_updates_per_batch = stats.batches > 0 ? std::accumulate(total_updates_per_batch.begin(), total_updates_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.p50_total_updates_per_batch = percentile(total_updates_per_batch, 0.50);
    stats.p90_total_updates_per_batch = percentile(total_updates_per_batch, 0.90);
    stats.max_total_updates_per_batch = total_updates_per_batch.empty() ? 0 : *std::max_element(total_updates_per_batch.begin(), total_updates_per_batch.end());
    stats.avg_unique_rows_per_batch = stats.batches > 0 ? std::accumulate(unique_rows_per_batch.begin(), unique_rows_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.p50_unique_rows_per_batch = percentile(unique_rows_per_batch, 0.50);
    stats.p90_unique_rows_per_batch = percentile(unique_rows_per_batch, 0.90);
    stats.max_unique_rows_per_batch = unique_rows_per_batch.empty() ? 0 : *std::max_element(unique_rows_per_batch.begin(), unique_rows_per_batch.end());
    stats.avg_duplicate_ratio = duplicate_ratio_per_batch.empty() ? 0.0
                                                                  : std::accumulate(duplicate_ratio_per_batch.begin(), duplicate_ratio_per_batch.end(), 0.0) /
                                                                        static_cast<double>(duplicate_ratio_per_batch.size());
    stats.p50_duplicate_ratio = percentile(duplicate_ratio_per_batch, 0.50);
    stats.p90_duplicate_ratio = percentile(duplicate_ratio_per_batch, 0.90);
    stats.avg_max_row_repeats = stats.batches > 0 ? std::accumulate(max_row_repeats_per_batch.begin(), max_row_repeats_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.p50_max_row_repeats = percentile(max_row_repeats_per_batch, 0.50);
    stats.p90_max_row_repeats = percentile(max_row_repeats_per_batch, 0.90);
    stats.max_max_row_repeats = max_row_repeats_per_batch.empty() ? 0 : *std::max_element(max_row_repeats_per_batch.begin(), max_row_repeats_per_batch.end());
    stats.avg_rows_ge2 = stats.batches > 0 ? std::accumulate(rows_ge2_per_batch.begin(), rows_ge2_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.avg_rows_ge4 = stats.batches > 0 ? std::accumulate(rows_ge4_per_batch.begin(), rows_ge4_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.avg_rows_ge8 = stats.batches > 0 ? std::accumulate(rows_ge8_per_batch.begin(), rows_ge8_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.avg_rows_ge16 = stats.batches > 0 ? std::accumulate(rows_ge16_per_batch.begin(), rows_ge16_per_batch.end(), int64_t{0}) / stats.batches : 0;
    stats.avg_accumulator_mb_lf50 = accumulator_mib_lf50_per_batch.empty() ? 0.0
                                                                           : std::accumulate(accumulator_mib_lf50_per_batch.begin(), accumulator_mib_lf50_per_batch.end(), 0.0) /
                                                                                 static_cast<double>(accumulator_mib_lf50_per_batch.size());
    stats.avg_accumulator_mb_lf67 = accumulator_mib_lf67_per_batch.empty() ? 0.0
                                                                           : std::accumulate(accumulator_mib_lf67_per_batch.begin(), accumulator_mib_lf67_per_batch.end(), 0.0) /
                                                                                 static_cast<double>(accumulator_mib_lf67_per_batch.size());
    stats.avg_accumulator_mb_lf80 = accumulator_mib_lf80_per_batch.empty() ? 0.0
                                                                           : std::accumulate(accumulator_mib_lf80_per_batch.begin(), accumulator_mib_lf80_per_batch.end(), 0.0) /
                                                                                 static_cast<double>(accumulator_mib_lf80_per_batch.size());
    stats.p90_accumulator_mb_lf67 = percentile(accumulator_mib_lf67_per_batch, 0.90);
    stats.max_accumulator_mb_lf67 = accumulator_mib_lf67_per_batch.empty() ? 0.0 : *std::max_element(accumulator_mib_lf67_per_batch.begin(), accumulator_mib_lf67_per_batch.end());
    stats.dense_grad_gib = bytes_to_gib(active_rows * args.embedding_dim * static_cast<int64_t>(sizeof(float)));
    stats.analyze_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    return stats;
}

void print_state_report(const Args &args, const StateAccumulatorStats &stats) {
    std::cout << "state_idx=" << stats.state_idx << "\n";
    std::cout << "active_rows=" << stats.active_rows << "\n";
    std::cout << "positive_edges=" << stats.positive_edges << "\n";
    std::cout << "batches=" << stats.batches << "\n";
    std::cout << "avg_total_updates_per_batch=" << stats.avg_total_updates_per_batch << "\n";
    std::cout << "p50_total_updates_per_batch=" << stats.p50_total_updates_per_batch << "\n";
    std::cout << "p90_total_updates_per_batch=" << stats.p90_total_updates_per_batch << "\n";
    std::cout << "max_total_updates_per_batch=" << stats.max_total_updates_per_batch << "\n";
    std::cout << "avg_unique_rows_per_batch=" << stats.avg_unique_rows_per_batch << "\n";
    std::cout << "p50_unique_rows_per_batch=" << stats.p50_unique_rows_per_batch << "\n";
    std::cout << "p90_unique_rows_per_batch=" << stats.p90_unique_rows_per_batch << "\n";
    std::cout << "max_unique_rows_per_batch=" << stats.max_unique_rows_per_batch << "\n";
    std::cout << "avg_duplicate_ratio=" << std::fixed << std::setprecision(6) << stats.avg_duplicate_ratio << "\n";
    std::cout << "p50_duplicate_ratio=" << stats.p50_duplicate_ratio << "\n";
    std::cout << "p90_duplicate_ratio=" << stats.p90_duplicate_ratio << "\n";
    std::cout << "avg_max_row_repeats=" << stats.avg_max_row_repeats << "\n";
    std::cout << "p50_max_row_repeats=" << stats.p50_max_row_repeats << "\n";
    std::cout << "p90_max_row_repeats=" << stats.p90_max_row_repeats << "\n";
    std::cout << "max_max_row_repeats=" << stats.max_max_row_repeats << "\n";
    std::cout << "avg_rows_ge2=" << stats.avg_rows_ge2 << "\n";
    std::cout << "avg_rows_ge4=" << stats.avg_rows_ge4 << "\n";
    std::cout << "avg_rows_ge8=" << stats.avg_rows_ge8 << "\n";
    std::cout << "avg_rows_ge16=" << stats.avg_rows_ge16 << "\n";
    std::cout << "avg_accumulator_mib_lf50=" << std::fixed << std::setprecision(3) << stats.avg_accumulator_mb_lf50 << "\n";
    std::cout << "avg_accumulator_mib_lf67=" << stats.avg_accumulator_mb_lf67 << "\n";
    std::cout << "avg_accumulator_mib_lf80=" << stats.avg_accumulator_mb_lf80 << "\n";
    std::cout << "p90_accumulator_mib_lf67=" << stats.p90_accumulator_mb_lf67 << "\n";
    std::cout << "max_accumulator_mib_lf67=" << stats.max_accumulator_mb_lf67 << "\n";
    std::cout << "dense_grad_gib=" << std::fixed << std::setprecision(6) << stats.dense_grad_gib << "\n";
    std::cout << "assumptions.embedding_dim=" << args.embedding_dim << "\n";
    std::cout << "assumptions.num_chunks=" << args.num_chunks << "\n";
    std::cout << "assumptions.negatives_per_positive=" << args.negatives_per_positive << "\n";
    std::cout << "assumptions.degree_fraction=" << std::fixed << std::setprecision(3) << args.degree_fraction << "\n";
    std::cout << "assumptions.use_inverse_relations=" << args.use_inverse_relations << "\n";
    std::cout << "analyze_ms=" << std::fixed << std::setprecision(3) << stats.analyze_ms << "\n";
}

void print_all_state_summary(const std::vector<StateAccumulatorStats> &stats) {
    if (stats.empty()) {
        return;
    }
    auto mean_i64 = [&](auto accessor) {
        int64_t total = 0;
        for (const auto &s : stats) total += accessor(s);
        return total / static_cast<int64_t>(stats.size());
    };
    auto mean_f64 = [&](auto accessor) {
        double total = 0.0;
        for (const auto &s : stats) total += accessor(s);
        return total / static_cast<double>(stats.size());
    };
    auto max_f64 = [&](auto accessor) {
        double value = 0.0;
        for (const auto &s : stats) value = std::max(value, accessor(s));
        return value;
    };
    std::cout << "=== summary ===\n";
    std::cout << "states=" << stats.size() << "\n";
    std::cout << "mean_active_rows=" << mean_i64([](const auto &s) { return s.active_rows; }) << "\n";
    std::cout << "mean_avg_total_updates_per_batch=" << mean_i64([](const auto &s) { return s.avg_total_updates_per_batch; }) << "\n";
    std::cout << "mean_avg_unique_rows_per_batch=" << mean_i64([](const auto &s) { return s.avg_unique_rows_per_batch; }) << "\n";
    std::cout << "mean_avg_duplicate_ratio=" << std::fixed << std::setprecision(6)
              << mean_f64([](const auto &s) { return s.avg_duplicate_ratio; }) << "\n";
    std::cout << "mean_avg_max_row_repeats=" << mean_i64([](const auto &s) { return s.avg_max_row_repeats; }) << "\n";
    std::cout << "mean_avg_accumulator_mib_lf67=" << std::fixed << std::setprecision(3)
              << mean_f64([](const auto &s) { return s.avg_accumulator_mb_lf67; }) << "\n";
    std::cout << "max_accumulator_mib_lf67=" << max_f64([](const auto &s) { return s.max_accumulator_mb_lf67; }) << "\n";
    std::cout << "mean_dense_grad_gib=" << std::fixed << std::setprecision(6)
              << mean_f64([](const auto &s) { return s.dense_grad_gib; }) << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args args = parse_args(argc, argv);

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

        std::cout << "dataset_dir=" << args.dataset_dir << "\n";
        std::cout << "num_nodes=" << num_nodes << "\n";
        std::cout << "num_edges=" << num_edges << "\n";
        std::cout << "num_states=" << buffer_states.size() << "\n";
        std::cout << "num_partitions=" << args.num_partitions << "\n";
        std::cout << "buffer_capacity=" << args.buffer_capacity << "\n";
        std::cout << "ordering_enum=" << static_cast<int>(args.ordering) << "\n";
        std::cout << "edge_cols=" << edge_cols << "\n";

        if (args.analyze_all_states) {
            std::vector<StateAccumulatorStats> all_stats;
            all_stats.reserve(buffer_states.size());
            for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
                auto stats = analyze_state(args, static_cast<int64_t>(state_idx), bucket_sizes, num_nodes, edge_cols, dtype_bytes,
                                           edge_path, buffer_states, edge_buckets_per_buffer);
                all_stats.emplace_back(stats);
                std::cout << "=== state " << state_idx << " ===\n";
                print_state_report(args, stats);
            }
            print_all_state_summary(all_stats);
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
