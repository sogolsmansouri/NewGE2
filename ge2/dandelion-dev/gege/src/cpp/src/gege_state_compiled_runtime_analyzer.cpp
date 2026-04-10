#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

struct StateCompileStats {
    int64_t state_idx = -1;
    int64_t batches = 0;
    int64_t positive_edges = 0;
    int64_t state_domain_rows = 0;
    int64_t active_partition_rows = 0;
    int64_t positive_unique_rows = 0;
    int64_t total_batch_unique_touches = 0;
    int64_t avg_positive_unique_per_batch = 0;
    int64_t max_positive_unique_per_batch = 0;
    int64_t total_positive_unique_arena_rows = 0;
    double compile_ms = 0.0;
    struct WindowOverlapSummary {
        int64_t window_batches = 0;
        double avg_fraction = 0.0;
        double min_fraction = 0.0;
        double max_fraction = 0.0;
    };
    struct CoverageSummary {
        double row_fraction = 0.0;
        int64_t rows = 0;
        double touch_fraction = 0.0;
    };
    std::vector<WindowOverlapSummary> overlap_windows;
    std::vector<CoverageSummary> hotset_coverages;
    int64_t rows_for_50pct_touches = 0;
    int64_t rows_for_70pct_touches = 0;
    int64_t rows_for_90pct_touches = 0;
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]"
                 " [--seed <int64>] [--batch-size <int64>] [--num-chunks <int64>]"
                 " [--negatives-per-positive <int64>] [--degree-fraction <float>]"
                 " [--state-idx <int64>] [--all-states]\n";
}

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
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

double bytes_to_gib(int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1024ll * 1024ll * 1024ll);
}

int64_t partition_row_count(int64_t partition_id, int64_t partition_size, int64_t num_nodes) {
    int64_t start = partition_id * partition_size;
    if (start >= num_nodes) {
        return 0;
    }
    return std::min<int64_t>(partition_size, num_nodes - start);
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

StateCompileStats analyze_state(const Args &args,
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

    auto state_start = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> state_partitions = tensor_to_vector(buffer_states[state_idx]);
    std::unordered_map<int64_t, int64_t> partition_to_slot;
    partition_to_slot.reserve(state_partitions.size());
    int64_t partition_size = static_cast<int64_t>(std::ceil(static_cast<double>(num_nodes) / static_cast<double>(args.num_partitions)));
    int64_t state_domain_rows = static_cast<int64_t>(state_partitions.size()) * partition_size;
    int64_t active_partition_rows = 0;
    for (std::size_t slot = 0; slot < state_partitions.size(); slot++) {
        partition_to_slot[state_partitions[slot]] = static_cast<int64_t>(slot);
        active_partition_rows += partition_row_count(state_partitions[slot], partition_size, num_nodes);
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

    std::vector<uint8_t> state_seen(static_cast<std::size_t>(state_domain_rows), 0);
    std::vector<uint32_t> batch_seen(static_cast<std::size_t>(state_domain_rows), 0);
    uint32_t batch_generation = 1;
    std::vector<uint32_t> batch_touch_counts(static_cast<std::size_t>(state_domain_rows), 0);
    std::vector<std::vector<int32_t>> batch_unique_rows;
    std::vector<int32_t> current_batch_rows;

    int64_t positive_edges = 0;
    int64_t positive_unique_rows = 0;
    int64_t total_batch_unique_touches = 0;
    int64_t batches = 0;
    int64_t total_positive_unique_arena_rows = 0;
    int64_t max_positive_unique_per_batch = 0;
    int64_t current_batch_edges = 0;
    int64_t current_batch_unique = 0;

    auto flush_batch = [&]() {
        if (current_batch_edges == 0) {
            return;
        }
        batches++;
        std::sort(current_batch_rows.begin(), current_batch_rows.end());
        current_batch_rows.erase(std::unique(current_batch_rows.begin(), current_batch_rows.end()), current_batch_rows.end());
        for (auto row : current_batch_rows) {
            batch_touch_counts[static_cast<std::size_t>(row)]++;
        }
        total_batch_unique_touches += static_cast<int64_t>(current_batch_rows.size());
        batch_unique_rows.emplace_back(std::move(current_batch_rows));
        current_batch_rows.clear();
        total_positive_unique_arena_rows += current_batch_unique;
        max_positive_unique_per_batch = std::max<int64_t>(max_positive_unique_per_batch, current_batch_unique);
        current_batch_edges = 0;
        current_batch_unique = 0;
        batch_generation++;
        if (batch_generation == 0) {
            std::fill(batch_seen.begin(), batch_seen.end(), 0);
            batch_generation = 1;
        }
    };

    std::ifstream edge_in(edge_path, std::ios::binary);
    if (!edge_in) {
        throw std::runtime_error("Failed to open edge file");
    }

    constexpr int64_t rows_per_chunk = 1 << 16;
    std::vector<char> raw;
    raw.resize(static_cast<std::size_t>(rows_per_chunk * edge_cols * dtype_bytes));

    auto register_row = [&](int64_t global_id) {
        int64_t partition_id = global_id / partition_size;
        auto slot_it = partition_to_slot.find(partition_id);
        if (slot_it == partition_to_slot.end()) {
            throw std::runtime_error("Edge row does not belong to the selected state");
        }
        int64_t slot = slot_it->second;
        int64_t local_offset = global_id - partition_id * partition_size;
        int64_t local_row = slot * partition_size + local_offset;
        if (local_row < 0 || local_row >= state_domain_rows) {
            throw std::runtime_error("Computed local row out of state domain");
        }
        std::size_t row_idx = static_cast<std::size_t>(local_row);
        if (!state_seen[row_idx]) {
            state_seen[row_idx] = 1;
            positive_unique_rows++;
        }
        if (batch_seen[row_idx] != batch_generation) {
            batch_seen[row_idx] = batch_generation;
            current_batch_unique++;
            current_batch_rows.emplace_back(static_cast<int32_t>(local_row));
        }
    };

    for (int64_t bucket_pos = 0; bucket_pos < buckets_tensor.size(0); bucket_pos++) {
        int64_t src_part = buckets[bucket_pos][0];
        int64_t dst_part = buckets[bucket_pos][1];
        int64_t bucket_idx = src_part * args.num_partitions + dst_part;
        int64_t bucket_size = bucket_sizes[static_cast<std::size_t>(bucket_idx)];
        int64_t bucket_offset = bucket_offsets[static_cast<std::size_t>(bucket_idx)];
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
                register_row(src);
                register_row(dst);
                positive_edges++;
                current_batch_edges++;
                if (current_batch_edges == args.batch_size) {
                    flush_batch();
                }
            }
            consumed += rows_to_read;
            rows_remaining -= rows_to_read;
        }
    }
    flush_batch();

    StateCompileStats stats;
    stats.state_idx = state_idx;
    stats.batches = batches;
    stats.positive_edges = positive_edges;
    stats.state_domain_rows = state_domain_rows;
    stats.active_partition_rows = active_partition_rows;
    stats.positive_unique_rows = positive_unique_rows;
    stats.total_batch_unique_touches = total_batch_unique_touches;
    stats.avg_positive_unique_per_batch = batches > 0 ? total_positive_unique_arena_rows / batches : 0;
    stats.max_positive_unique_per_batch = max_positive_unique_per_batch;
    stats.total_positive_unique_arena_rows = total_positive_unique_arena_rows;
    stats.compile_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - state_start).count();

    if (!batch_unique_rows.empty()) {
        std::vector<uint32_t> union_seen(static_cast<std::size_t>(state_domain_rows), 0);
        uint32_t union_generation = 1;
        for (auto window : {1, 2, 4, 8}) {
            StateCompileStats::WindowOverlapSummary summary;
            summary.window_batches = window;
            double overlap_sum = 0.0;
            int64_t overlap_samples = 0;
            double min_overlap = std::numeric_limits<double>::infinity();
            double max_overlap = 0.0;
            for (std::size_t batch_idx = 1; batch_idx < batch_unique_rows.size(); batch_idx++) {
                if (union_generation == 0) {
                    std::fill(union_seen.begin(), union_seen.end(), 0);
                    union_generation = 1;
                }
                std::size_t start_idx = batch_idx > static_cast<std::size_t>(window) ? batch_idx - static_cast<std::size_t>(window) : 0;
                for (std::size_t prev_idx = start_idx; prev_idx < batch_idx; prev_idx++) {
                    for (auto row : batch_unique_rows[prev_idx]) {
                        union_seen[static_cast<std::size_t>(row)] = union_generation;
                    }
                }
                int64_t hits = 0;
                for (auto row : batch_unique_rows[batch_idx]) {
                    hits += union_seen[static_cast<std::size_t>(row)] == union_generation ? 1 : 0;
                }
                double overlap = batch_unique_rows[batch_idx].empty()
                    ? 0.0
                    : static_cast<double>(hits) / static_cast<double>(batch_unique_rows[batch_idx].size());
                overlap_sum += overlap;
                overlap_samples++;
                min_overlap = std::min(min_overlap, overlap);
                max_overlap = std::max(max_overlap, overlap);
                union_generation++;
            }
            if (overlap_samples > 0) {
                summary.avg_fraction = overlap_sum / static_cast<double>(overlap_samples);
                summary.min_fraction = std::isfinite(min_overlap) ? min_overlap : 0.0;
                summary.max_fraction = max_overlap;
            }
            stats.overlap_windows.emplace_back(summary);
        }
    }

    std::vector<uint32_t> positive_touch_hist;
    positive_touch_hist.reserve(static_cast<std::size_t>(positive_unique_rows));
    for (auto count : batch_touch_counts) {
        if (count > 0) {
            positive_touch_hist.emplace_back(count);
        }
    }
    std::sort(positive_touch_hist.begin(), positive_touch_hist.end(), std::greater<uint32_t>());

    int64_t total_positive_touch_weight = std::accumulate(
        positive_touch_hist.begin(), positive_touch_hist.end(), static_cast<int64_t>(0));
    std::vector<int64_t> sorted_touch_prefix(positive_touch_hist.size() + 1, 0);
    for (std::size_t i = 0; i < positive_touch_hist.size(); i++) {
        sorted_touch_prefix[i + 1] = sorted_touch_prefix[i] + static_cast<int64_t>(positive_touch_hist[i]);
    }

    for (auto row_fraction : {0.001, 0.005, 0.01, 0.02, 0.05, 0.10}) {
        StateCompileStats::CoverageSummary summary;
        summary.row_fraction = row_fraction;
        summary.rows = std::min<int64_t>(
            static_cast<int64_t>(std::ceil(row_fraction * static_cast<double>(active_partition_rows))),
            static_cast<int64_t>(positive_touch_hist.size()));
        int64_t covered_touches = sorted_touch_prefix[static_cast<std::size_t>(summary.rows)];
        summary.touch_fraction = total_positive_touch_weight > 0
            ? static_cast<double>(covered_touches) / static_cast<double>(total_positive_touch_weight)
            : 0.0;
        stats.hotset_coverages.emplace_back(summary);
    }

    auto rows_for_touch_target = [&](double target_fraction) {
        if (total_positive_touch_weight <= 0) {
            return int64_t{0};
        }
        int64_t target = static_cast<int64_t>(std::ceil(target_fraction * static_cast<double>(total_positive_touch_weight)));
        auto it = std::lower_bound(sorted_touch_prefix.begin(), sorted_touch_prefix.end(), target);
        return static_cast<int64_t>(std::distance(sorted_touch_prefix.begin(), it));
    };
    stats.rows_for_50pct_touches = rows_for_touch_target(0.50);
    stats.rows_for_70pct_touches = rows_for_touch_target(0.70);
    stats.rows_for_90pct_touches = rows_for_touch_target(0.90);
    return stats;
}

void print_state_report(const Args &args, const StateCompileStats &stats, int64_t num_nodes) {
    int64_t num_uniform = static_cast<int64_t>(std::llround(
        static_cast<double>(args.negatives_per_positive) * (1.0 - args.degree_fraction)));
    int64_t num_degree = std::max<int64_t>(args.negatives_per_positive - num_uniform, 0);
    int64_t negative_ids_per_batch = 2 * args.num_chunks * (num_uniform + num_degree);
    int64_t max_positive_input_ids = 2 * args.batch_size;
    int64_t positive_only_unique_buf_rows = stats.max_positive_unique_per_batch;
    int64_t positive_only_inverse_rows = max_positive_input_ids;
    int64_t full_replay_input_ids = max_positive_input_ids + negative_ids_per_batch;
    int64_t full_replay_unique_upper_bound = std::min<int64_t>(stats.state_domain_rows, stats.max_positive_unique_per_batch + negative_ids_per_batch);

    constexpr int64_t batch_tape_entry_bytes = 48;
    constexpr int64_t negative_seed_descriptor_bytes = 40;

    int64_t seen_generation_bytes = stats.state_domain_rows * 4;
    int64_t compact_index_bytes = stats.state_domain_rows * 4;
    int64_t positive_only_unique_buf_bytes = positive_only_unique_buf_rows * 4;
    int64_t positive_only_inverse_bytes = positive_only_inverse_rows * 4;
    int64_t full_replay_unique_buf_bytes = full_replay_unique_upper_bound * 4;
    int64_t full_replay_inverse_bytes = full_replay_input_ids * 4;

    int64_t batch_tape_bytes = stats.batches * batch_tape_entry_bytes;
    int64_t positive_unique_arena_bytes = stats.total_positive_unique_arena_rows * 4;
    int64_t positive_offsets_bytes = (stats.batches + 1) * 8;
    int64_t negative_seed_tape_bytes = stats.batches * negative_seed_descriptor_bytes;
    int64_t materialized_negative_ids_bytes = stats.batches * negative_ids_per_batch * 8;

    std::cout << "state_idx=" << stats.state_idx << "\n";
    std::cout << "positive_edges=" << stats.positive_edges << "\n";
    std::cout << "batches=" << stats.batches << "\n";
    std::cout << "state_domain_rows=" << stats.state_domain_rows << "\n";
    std::cout << "active_partition_rows=" << stats.active_partition_rows << "\n";
    std::cout << "positive_unique_rows=" << stats.positive_unique_rows << "\n";
    std::cout << "positive_density=" << std::fixed << std::setprecision(6)
              << (stats.state_domain_rows > 0 ? static_cast<double>(stats.positive_unique_rows) / static_cast<double>(stats.state_domain_rows) : 0.0)
              << "\n";
    std::cout << "total_batch_unique_touches=" << stats.total_batch_unique_touches << "\n";
    std::cout << "positive_batch_touch_reuse="
              << std::fixed << std::setprecision(6)
              << (stats.positive_unique_rows > 0 ? static_cast<double>(stats.total_batch_unique_touches) / static_cast<double>(stats.positive_unique_rows) : 0.0)
              << "\n";
    std::cout << "avg_positive_unique_per_batch=" << stats.avg_positive_unique_per_batch << "\n";
    std::cout << "max_positive_unique_per_batch=" << stats.max_positive_unique_per_batch << "\n";
    std::cout << "compile_ms=" << std::fixed << std::setprecision(3) << stats.compile_ms << "\n";
    std::cout << "assumptions.batch_size=" << args.batch_size << "\n";
    std::cout << "assumptions.num_chunks=" << args.num_chunks << "\n";
    std::cout << "assumptions.negatives_per_positive=" << args.negatives_per_positive << "\n";
    std::cout << "assumptions.negative_ids_per_batch=" << negative_ids_per_batch << "\n";
    std::cout << "workspace.positive_only.seen_generation_gib=" << std::fixed << std::setprecision(3) << bytes_to_gib(seen_generation_bytes) << "\n";
    std::cout << "workspace.positive_only.compact_index_gib=" << bytes_to_gib(compact_index_bytes) << "\n";
    std::cout << "workspace.positive_only.unique_buf_gib=" << bytes_to_gib(positive_only_unique_buf_bytes) << "\n";
    std::cout << "workspace.positive_only.inverse_buf_gib=" << bytes_to_gib(positive_only_inverse_bytes) << "\n";
    std::cout << "workspace.positive_only.total_gib="
              << bytes_to_gib(seen_generation_bytes + compact_index_bytes + positive_only_unique_buf_bytes + positive_only_inverse_bytes) << "\n";
    std::cout << "workspace.full_replay.unique_buf_upper_gib=" << bytes_to_gib(full_replay_unique_buf_bytes) << "\n";
    std::cout << "workspace.full_replay.inverse_buf_gib=" << bytes_to_gib(full_replay_inverse_bytes) << "\n";
    std::cout << "workspace.full_replay.total_upper_gib="
              << bytes_to_gib(seen_generation_bytes + compact_index_bytes + full_replay_unique_buf_bytes + full_replay_inverse_bytes) << "\n";
    std::cout << "tape.batch_entry_gib=" << bytes_to_gib(batch_tape_bytes) << "\n";
    std::cout << "tape.positive_unique_arena_gib=" << bytes_to_gib(positive_unique_arena_bytes) << "\n";
    std::cout << "tape.positive_offsets_gib=" << bytes_to_gib(positive_offsets_bytes) << "\n";
    std::cout << "tape.negative_seed_descriptor_gib=" << bytes_to_gib(negative_seed_tape_bytes) << "\n";
    std::cout << "tape.materialized_negative_ids_gib=" << bytes_to_gib(materialized_negative_ids_bytes) << "\n";
    std::cout << "dataset_num_nodes=" << num_nodes << "\n";
    for (const auto &summary : stats.overlap_windows) {
        std::cout << "overlap.window_" << summary.window_batches << ".avg="
                  << std::fixed << std::setprecision(6) << summary.avg_fraction << "\n";
        std::cout << "overlap.window_" << summary.window_batches << ".min="
                  << std::fixed << std::setprecision(6) << summary.min_fraction << "\n";
        std::cout << "overlap.window_" << summary.window_batches << ".max="
                  << std::fixed << std::setprecision(6) << summary.max_fraction << "\n";
    }
    for (const auto &summary : stats.hotset_coverages) {
        int64_t pct = static_cast<int64_t>(std::llround(summary.row_fraction * 1000.0));
        std::cout << "hotset.top_" << pct << "_permille.rows=" << summary.rows << "\n";
        std::cout << "hotset.top_" << pct << "_permille.touch_fraction="
                  << std::fixed << std::setprecision(6) << summary.touch_fraction << "\n";
    }
    std::cout << "hotset.rows_for_50pct_touches=" << stats.rows_for_50pct_touches << "\n";
    std::cout << "hotset.rows_for_70pct_touches=" << stats.rows_for_70pct_touches << "\n";
    std::cout << "hotset.rows_for_90pct_touches=" << stats.rows_for_90pct_touches << "\n";
    std::cout << "hotset.rows_for_50pct_touches_fraction="
              << std::fixed << std::setprecision(6)
              << (stats.active_partition_rows > 0 ? static_cast<double>(stats.rows_for_50pct_touches) / static_cast<double>(stats.active_partition_rows) : 0.0)
              << "\n";
    std::cout << "hotset.rows_for_70pct_touches_fraction="
              << std::fixed << std::setprecision(6)
              << (stats.active_partition_rows > 0 ? static_cast<double>(stats.rows_for_70pct_touches) / static_cast<double>(stats.active_partition_rows) : 0.0)
              << "\n";
    std::cout << "hotset.rows_for_90pct_touches_fraction="
              << std::fixed << std::setprecision(6)
              << (stats.active_partition_rows > 0 ? static_cast<double>(stats.rows_for_90pct_touches) / static_cast<double>(stats.active_partition_rows) : 0.0)
              << "\n";
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

        if (args.analyze_all_states) {
            for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
                auto stats = analyze_state(args, static_cast<int64_t>(state_idx), bucket_sizes, num_nodes, edge_cols, dtype_bytes,
                                           edge_path, buffer_states, edge_buckets_per_buffer);
                std::cout << "=== state " << state_idx << " ===\n";
                print_state_report(args, stats, num_nodes);
            }
        } else {
            auto stats = analyze_state(args, args.state_idx, bucket_sizes, num_nodes, edge_cols, dtype_bytes, edge_path,
                                       buffer_states, edge_buckets_per_buffer);
            print_state_report(args, stats, num_nodes);
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
