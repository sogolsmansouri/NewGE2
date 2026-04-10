#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    int64_t state_idx = 0;
    int64_t window_batches = 4;
    double hot_touch_target = 0.50;
};

struct HotWindowSummary {
    int64_t window_batches = 0;
    double target_touch_fraction = 0.0;
    double avg_hot_rows_fraction = 0.0;
    double min_hot_rows_fraction = 0.0;
    double max_hot_rows_fraction = 0.0;
    double avg_covered_touch_fraction = 0.0;
    double avg_hot_rows = 0.0;
    double avg_batch_hot_unique_fraction = 0.0;
    double avg_batch_cold_unique_fraction = 0.0;
    double avg_batch_cold_unique_rows = 0.0;
    double max_batch_cold_unique_rows = 0.0;
};

struct ReplayValidationStats {
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
    int64_t reconstruction_failures = 0;
    int64_t unique_reference_mismatches = 0;
    int64_t empty_batches = 0;
    int64_t src_map_max = 0;
    int64_t dst_map_max = 0;
    double compile_validate_ms = 0.0;
    double positive_unique_arena_gib = 0.0;
    double src_map_arena_gib = 0.0;
    double dst_map_arena_gib = 0.0;
    double positive_offsets_gib = 0.0;
    double replay_plan_total_gib = 0.0;
    double workspace_total_gib = 0.0;
    int64_t replay_map_index_bytes = 4;
    HotWindowSummary hot_window;
};

void print_usage(const char *prog) {
    std::cerr << "Usage: " << prog
              << " <dataset_dir> <ordering> <num_partitions> <buffer_capacity> <fine_to_coarse_ratio> <num_cache_partitions>"
                 " <randomly_assign_edge_buckets:0|1> <active_devices>"
                 " [--regroup] [--access-aware] [--access-aware-generate]"
                 " [--seed <int64>] [--batch-size <int64>] [--state-idx <int64>]"
                 " [--window-batches <int64>] [--hot-touch-target <float>]\n";
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
        } else if (arg == "--state-idx" && i + 1 < argc) {
            args.state_idx = std::stoll(argv[++i]);
        } else if (arg == "--window-batches" && i + 1 < argc) {
            args.window_batches = std::max<int64_t>(1, std::stoll(argv[++i]));
        } else if (arg == "--hot-touch-target" && i + 1 < argc) {
            args.hot_touch_target = std::stod(argv[++i]);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return args;
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

double bytes_to_gib(int64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1024ll * 1024ll * 1024ll);
}

torch::Tensor tensor_from_vector(const std::vector<int64_t> &values) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (values.empty()) {
        return torch::empty({0}, opts);
    }
    return torch::from_blob(const_cast<int64_t *>(values.data()), {static_cast<int64_t>(values.size())}, opts).clone();
}

torch::Tensor compact_fixed_buffer_unique_prefix(torch::Tensor unique_ids, torch::Tensor active_mask) {
    if (!unique_ids.defined() || !active_mask.defined() || active_mask.numel() != unique_ids.numel()) {
        return unique_ids;
    }
    int64_t active_rows = active_mask.sum().item<int64_t>();
    active_rows = std::max<int64_t>(active_rows, 0);
    active_rows = std::min<int64_t>(active_rows, unique_ids.numel());
    return unique_ids.narrow(0, 0, active_rows);
}

std::vector<int64_t> sorted_unique_reference(std::vector<int32_t> rows) {
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::vector<int64_t> out;
    out.reserve(rows.size());
    for (auto row : rows) {
        out.emplace_back(static_cast<int64_t>(row));
    }
    return out;
}

HotWindowSummary summarize_hot_windows(const std::vector<std::vector<int32_t>> &batch_unique_rows,
                                       int64_t active_partition_rows,
                                       int64_t window_batches,
                                       double target_touch_fraction) {
    HotWindowSummary summary;
    summary.window_batches = window_batches;
    summary.target_touch_fraction = target_touch_fraction;
    if (batch_unique_rows.empty() || active_partition_rows <= 0) {
        return summary;
    }

    double row_fraction_sum = 0.0;
    double covered_touch_sum = 0.0;
    double hot_rows_sum = 0.0;
    double batch_hot_fraction_sum = 0.0;
    double batch_cold_fraction_sum = 0.0;
    double batch_cold_rows_sum = 0.0;
    double max_batch_cold_rows = 0.0;
    int64_t samples = 0;
    int64_t batch_samples = 0;
    double min_row_fraction = std::numeric_limits<double>::infinity();
    double max_row_fraction = 0.0;

    for (std::size_t start = 0; start < batch_unique_rows.size(); start++) {
        std::size_t end = std::min<std::size_t>(batch_unique_rows.size(), start + static_cast<std::size_t>(window_batches));
        std::unordered_map<int32_t, int32_t> touch_counts;
        int64_t total_touches = 0;
        for (std::size_t batch_idx = start; batch_idx < end; batch_idx++) {
            total_touches += static_cast<int64_t>(batch_unique_rows[batch_idx].size());
            for (auto row : batch_unique_rows[batch_idx]) {
                touch_counts[row] += 1;
            }
        }
        if (touch_counts.empty() || total_touches <= 0) {
            continue;
        }

        std::vector<std::pair<int32_t, int32_t>> counts;
        counts.reserve(touch_counts.size());
        for (const auto &kv : touch_counts) {
            counts.emplace_back(kv.first, kv.second);
        }
        std::sort(counts.begin(), counts.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
            }
            return lhs.first < rhs.first;
        });

        int64_t target_touches = static_cast<int64_t>(std::ceil(target_touch_fraction * static_cast<double>(total_touches)));
        int64_t covered_touches = 0;
        int64_t rows_needed = 0;
        std::unordered_set<int32_t> hot_rows;
        hot_rows.reserve(counts.size());
        for (const auto &entry : counts) {
            covered_touches += static_cast<int64_t>(entry.second);
            rows_needed++;
            hot_rows.insert(entry.first);
            if (covered_touches >= target_touches) {
                break;
            }
        }

        double row_fraction = static_cast<double>(rows_needed) / static_cast<double>(active_partition_rows);
        double covered_fraction = static_cast<double>(covered_touches) / static_cast<double>(total_touches);

        row_fraction_sum += row_fraction;
        covered_touch_sum += covered_fraction;
        hot_rows_sum += static_cast<double>(rows_needed);
        min_row_fraction = std::min(min_row_fraction, row_fraction);
        max_row_fraction = std::max(max_row_fraction, row_fraction);
        samples++;

        for (std::size_t batch_idx = start; batch_idx < end; batch_idx++) {
            const auto &rows = batch_unique_rows[batch_idx];
            if (rows.empty()) {
                continue;
            }
            int64_t hot_hits = 0;
            for (auto row : rows) {
                if (hot_rows.find(row) != hot_rows.end()) {
                    hot_hits++;
                }
            }
            int64_t cold_rows = static_cast<int64_t>(rows.size()) - hot_hits;
            batch_hot_fraction_sum += static_cast<double>(hot_hits) / static_cast<double>(rows.size());
            batch_cold_fraction_sum += static_cast<double>(cold_rows) / static_cast<double>(rows.size());
            batch_cold_rows_sum += static_cast<double>(cold_rows);
            max_batch_cold_rows = std::max(max_batch_cold_rows, static_cast<double>(cold_rows));
            batch_samples++;
        }
    }

    if (samples > 0) {
        summary.avg_hot_rows_fraction = row_fraction_sum / static_cast<double>(samples);
        summary.min_hot_rows_fraction = std::isfinite(min_row_fraction) ? min_row_fraction : 0.0;
        summary.max_hot_rows_fraction = max_row_fraction;
        summary.avg_covered_touch_fraction = covered_touch_sum / static_cast<double>(samples);
        summary.avg_hot_rows = hot_rows_sum / static_cast<double>(samples);
    }
    if (batch_samples > 0) {
        summary.avg_batch_hot_unique_fraction = batch_hot_fraction_sum / static_cast<double>(batch_samples);
        summary.avg_batch_cold_unique_fraction = batch_cold_fraction_sum / static_cast<double>(batch_samples);
        summary.avg_batch_cold_unique_rows = batch_cold_rows_sum / static_cast<double>(batch_samples);
        summary.max_batch_cold_unique_rows = max_batch_cold_rows;
    }
    return summary;
}

ReplayValidationStats validate_state(const Args &args,
                                     const std::vector<int64_t> &bucket_sizes,
                                     int64_t num_nodes,
                                     int edge_cols,
                                     int dtype_bytes,
                                     const std::string &edge_path,
                                     const std::vector<torch::Tensor> &buffer_states,
                                     const std::vector<torch::Tensor> &edge_buckets_per_buffer) {
    if (args.state_idx < 0 || static_cast<std::size_t>(args.state_idx) >= buffer_states.size()) {
        throw std::runtime_error("state_idx out of range");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> state_partitions = tensor_to_vector(buffer_states[static_cast<std::size_t>(args.state_idx)]);
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

    torch::Tensor buckets_tensor =
        edge_buckets_per_buffer[static_cast<std::size_t>(args.state_idx)].to(torch::kCPU).to(torch::kInt64).contiguous();
    auto buckets = buckets_tensor.accessor<int64_t, 2>();

    std::vector<uint8_t> state_seen(static_cast<std::size_t>(state_domain_rows), 0);
    std::vector<uint32_t> batch_seen(static_cast<std::size_t>(state_domain_rows), 0);
    std::vector<uint32_t> batch_touch_counts(static_cast<std::size_t>(state_domain_rows), 0);
    std::vector<std::vector<int32_t>> batch_unique_rows;
    uint32_t batch_generation = 1;

    std::vector<int64_t> current_edge_src;
    std::vector<int64_t> current_edge_dst;
    std::vector<int32_t> current_batch_rows;

    int64_t positive_edges = 0;
    int64_t positive_unique_rows = 0;
    int64_t total_batch_unique_touches = 0;
    int64_t batches = 0;
    int64_t total_positive_unique_arena_rows = 0;
    int64_t max_positive_unique_per_batch = 0;
    int64_t current_batch_edges = 0;
    int64_t current_batch_unique = 0;
    int64_t reconstruction_failures = 0;
    int64_t unique_reference_mismatches = 0;
    int64_t empty_batches = 0;
    int64_t src_map_max = 0;
    int64_t dst_map_max = 0;

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
        return local_row;
    };

    auto flush_batch = [&]() {
        if (current_batch_edges == 0) {
            return;
        }

        batches++;
        auto reference_unique_rows = sorted_unique_reference(current_batch_rows);
        for (auto row : reference_unique_rows) {
            batch_touch_counts[static_cast<std::size_t>(row)]++;
        }
        total_batch_unique_touches += static_cast<int64_t>(reference_unique_rows.size());
        batch_unique_rows.emplace_back(current_batch_rows.begin(), current_batch_rows.end());

        torch::Tensor edge_src = tensor_from_vector(current_edge_src);
        torch::Tensor edge_dst = tensor_from_vector(current_edge_dst);
        torch::Tensor active_mask;
        auto mapped_tup = map_tensors({edge_src, edge_dst}, true, nullptr, &active_mask, state_domain_rows);
        torch::Tensor positive_unique = compact_fixed_buffer_unique_prefix(std::get<0>(mapped_tup), active_mask)
                                            .to(torch::kCPU)
                                            .to(torch::kInt64)
                                            .contiguous();
        auto mapped = std::get<1>(mapped_tup);
        torch::Tensor src_map = mapped[0].to(torch::kCPU).to(torch::kInt64).contiguous();
        torch::Tensor dst_map = mapped[1].to(torch::kCPU).to(torch::kInt64).contiguous();

        if (!positive_unique.index_select(0, src_map).equal(edge_src)) {
            reconstruction_failures++;
        }
        if (!positive_unique.index_select(0, dst_map).equal(edge_dst)) {
            reconstruction_failures++;
        }
        if (!positive_unique.equal(tensor_from_vector(reference_unique_rows))) {
            unique_reference_mismatches++;
        }

        total_positive_unique_arena_rows += positive_unique.numel();
        max_positive_unique_per_batch = std::max<int64_t>(max_positive_unique_per_batch, positive_unique.numel());
        if (src_map.numel() > 0) {
            src_map_max = std::max<int64_t>(src_map_max, src_map.max().item<int64_t>());
        }
        if (dst_map.numel() > 0) {
            dst_map_max = std::max<int64_t>(dst_map_max, dst_map.max().item<int64_t>());
        }
        if (positive_unique.numel() == 0) {
            empty_batches++;
        }

        current_edge_src.clear();
        current_edge_dst.clear();
        current_batch_rows.clear();
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
                int64_t src_local = register_row(src);
                int64_t dst_local = register_row(dst);
                current_edge_src.emplace_back(src_local);
                current_edge_dst.emplace_back(dst_local);
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

    std::vector<std::vector<int32_t>> deduped_batch_unique_rows;
    deduped_batch_unique_rows.reserve(batch_unique_rows.size());
    for (auto rows : batch_unique_rows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        deduped_batch_unique_rows.emplace_back(std::move(rows));
    }

    int64_t map_index_bytes = max_positive_unique_per_batch <= std::numeric_limits<uint16_t>::max() ? 2 : 4;
    int64_t positive_unique_arena_bytes = total_positive_unique_arena_rows * 4;
    int64_t src_map_arena_bytes = positive_edges * map_index_bytes;
    int64_t dst_map_arena_bytes = positive_edges * map_index_bytes;
    int64_t positive_offsets_bytes = (batches + 1) * 8 * 2;
    int64_t workspace_total_bytes = state_domain_rows * 4 + state_domain_rows * 4 +
                                    max_positive_unique_per_batch * 4 + (2 * args.batch_size) * 4;

    ReplayValidationStats stats;
    stats.state_idx = args.state_idx;
    stats.batches = batches;
    stats.positive_edges = positive_edges;
    stats.state_domain_rows = state_domain_rows;
    stats.active_partition_rows = active_partition_rows;
    stats.positive_unique_rows = positive_unique_rows;
    stats.total_batch_unique_touches = total_batch_unique_touches;
    stats.avg_positive_unique_per_batch = batches > 0 ? total_positive_unique_arena_rows / batches : 0;
    stats.max_positive_unique_per_batch = max_positive_unique_per_batch;
    stats.total_positive_unique_arena_rows = total_positive_unique_arena_rows;
    stats.reconstruction_failures = reconstruction_failures;
    stats.unique_reference_mismatches = unique_reference_mismatches;
    stats.empty_batches = empty_batches;
    stats.src_map_max = src_map_max;
    stats.dst_map_max = dst_map_max;
    stats.compile_validate_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
    stats.positive_unique_arena_gib = bytes_to_gib(positive_unique_arena_bytes);
    stats.src_map_arena_gib = bytes_to_gib(src_map_arena_bytes);
    stats.dst_map_arena_gib = bytes_to_gib(dst_map_arena_bytes);
    stats.positive_offsets_gib = bytes_to_gib(positive_offsets_bytes);
    stats.replay_plan_total_gib = bytes_to_gib(positive_unique_arena_bytes + src_map_arena_bytes + dst_map_arena_bytes + positive_offsets_bytes);
    stats.workspace_total_gib = bytes_to_gib(workspace_total_bytes);
    stats.replay_map_index_bytes = map_index_bytes;
    stats.hot_window = summarize_hot_windows(deduped_batch_unique_rows, active_partition_rows, args.window_batches, args.hot_touch_target);
    return stats;
}

void print_report(const ReplayValidationStats &stats) {
    std::cout << "state_idx=" << stats.state_idx << "\n";
    std::cout << "batches=" << stats.batches << "\n";
    std::cout << "positive_edges=" << stats.positive_edges << "\n";
    std::cout << "state_domain_rows=" << stats.state_domain_rows << "\n";
    std::cout << "active_partition_rows=" << stats.active_partition_rows << "\n";
    std::cout << "positive_unique_rows=" << stats.positive_unique_rows << "\n";
    std::cout << "positive_density=" << std::fixed << std::setprecision(6)
              << (stats.state_domain_rows > 0 ? static_cast<double>(stats.positive_unique_rows) / static_cast<double>(stats.state_domain_rows) : 0.0)
              << "\n";
    std::cout << "total_batch_unique_touches=" << stats.total_batch_unique_touches << "\n";
    std::cout << "positive_batch_touch_reuse=" << std::fixed << std::setprecision(6)
              << (stats.positive_unique_rows > 0 ? static_cast<double>(stats.total_batch_unique_touches) / static_cast<double>(stats.positive_unique_rows) : 0.0)
              << "\n";
    std::cout << "avg_positive_unique_per_batch=" << stats.avg_positive_unique_per_batch << "\n";
    std::cout << "max_positive_unique_per_batch=" << stats.max_positive_unique_per_batch << "\n";
    std::cout << "replay_map_index_bytes=" << stats.replay_map_index_bytes << "\n";
    std::cout << "src_map_max=" << stats.src_map_max << "\n";
    std::cout << "dst_map_max=" << stats.dst_map_max << "\n";
    std::cout << "validation.reconstruction_failures=" << stats.reconstruction_failures << "\n";
    std::cout << "validation.unique_reference_mismatches=" << stats.unique_reference_mismatches << "\n";
    std::cout << "validation.empty_batches=" << stats.empty_batches << "\n";
    std::cout << "compile_validate_ms=" << std::fixed << std::setprecision(3) << stats.compile_validate_ms << "\n";
    std::cout << "plan.positive_unique_arena_gib=" << std::fixed << std::setprecision(3) << stats.positive_unique_arena_gib << "\n";
    std::cout << "plan.src_map_arena_gib=" << stats.src_map_arena_gib << "\n";
    std::cout << "plan.dst_map_arena_gib=" << stats.dst_map_arena_gib << "\n";
    std::cout << "plan.positive_offsets_gib=" << stats.positive_offsets_gib << "\n";
    std::cout << "plan.total_gib=" << stats.replay_plan_total_gib << "\n";
    std::cout << "workspace.positive_only.total_gib=" << stats.workspace_total_gib << "\n";
    std::cout << "hot_window.window_batches=" << stats.hot_window.window_batches << "\n";
    std::cout << "hot_window.target_touch_fraction=" << std::fixed << std::setprecision(6) << stats.hot_window.target_touch_fraction << "\n";
    std::cout << "hot_window.avg_hot_rows_fraction=" << stats.hot_window.avg_hot_rows_fraction << "\n";
    std::cout << "hot_window.min_hot_rows_fraction=" << stats.hot_window.min_hot_rows_fraction << "\n";
    std::cout << "hot_window.max_hot_rows_fraction=" << stats.hot_window.max_hot_rows_fraction << "\n";
    std::cout << "hot_window.avg_covered_touch_fraction=" << stats.hot_window.avg_covered_touch_fraction << "\n";
    std::cout << "hot_window.avg_hot_rows=" << stats.hot_window.avg_hot_rows << "\n";
    std::cout << "hot_window.avg_batch_hot_unique_fraction=" << stats.hot_window.avg_batch_hot_unique_fraction << "\n";
    std::cout << "hot_window.avg_batch_cold_unique_fraction=" << stats.hot_window.avg_batch_cold_unique_fraction << "\n";
    std::cout << "hot_window.avg_batch_cold_unique_rows=" << stats.hot_window.avg_batch_cold_unique_rows << "\n";
    std::cout << "hot_window.max_batch_cold_unique_rows=" << stats.hot_window.max_batch_cold_unique_rows << "\n";
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
        std::cout << "window_batches=" << args.window_batches << "\n";
        std::cout << "hot_touch_target=" << std::fixed << std::setprecision(6) << args.hot_touch_target << "\n";

        auto stats = validate_state(args, bucket_sizes, num_nodes, edge_cols, dtype_bytes, edge_path, buffer_states, edge_buckets_per_buffer);
        print_report(stats);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
