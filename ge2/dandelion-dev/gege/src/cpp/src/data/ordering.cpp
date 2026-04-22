#include "common/datatypes.h"
#include "data/ordering.h"
#include "reporting/logger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef GEGE_OMP
#include "omp.h"
#endif

namespace {

std::tuple<torch::Tensor, torch::Tensor> unique_with_counts_sorted(torch::Tensor values) {
    auto sort_tup = torch::sort(values.to(torch::kInt64), 0, false);
    torch::Tensor sorted_values = std::get<0>(sort_tup);
    auto unique_tup = torch::unique_consecutive(sorted_values, false, true);
    return std::forward_as_tuple(std::get<0>(unique_tup), std::get<2>(unique_tup));
}

struct StateAccessSummary {
    std::vector<int64_t> partitions;
    std::unordered_map<int64_t, int64_t> incident_bucket_counts;
    int64_t total_bucket_edges = 0;
};

enum class HybridCoverVariant {
    LEGACY_ROTATED = 0,
    NATURAL = 1,
    REVERSED = 2,
};

std::vector<int64_t> tensor_to_partitions(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

bool optimized_custom_schedule_enabled() {
    const char *raw = std::getenv("GEGE_OPTIMIZED_CUSTOM_SCHEDULE");
    if (raw == nullptr) {
        return false;
    }

    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

bool contrastive_greedy_cover_ordering_enabled() {
    const char *raw = std::getenv("GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING");
    if (raw == nullptr) {
        return false;
    }

    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

bool hybrid_cover_ordering_enabled() {
    const char *raw = std::getenv("GEGE_HYBRID_COVER");
    if (raw == nullptr) {
        return false;
    }

    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

double stateflow_cost_env(const char *name, double default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stod(std::string(raw));
    } catch (...) {
        SPDLOG_WARN("Ignoring invalid {}='{}'; using default {}", name, raw, default_value);
        return default_value;
    }
}

const char *stateflow_env_value(const char *primary, const char *secondary = nullptr) {
    const char *raw = std::getenv(primary);
    if (raw != nullptr) {
        return raw;
    }
    if (secondary != nullptr) {
        return std::getenv(secondary);
    }
    return nullptr;
}

int64_t stateflow_env_int64(const char *primary, const char *secondary, int64_t default_value) {
    const char *raw = stateflow_env_value(primary, secondary);
    if (raw == nullptr) {
        return default_value;
    }
    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        SPDLOG_WARN("Ignoring invalid {}='{}'; using default {}", primary, raw, default_value);
        return default_value;
    }
}

bool stateflow_env_bool(const char *primary, const char *secondary, bool default_value) {
    const char *raw = stateflow_env_value(primary, secondary);
    if (raw == nullptr) {
        return default_value;
    }
    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

enum class LaneMatchSolver {
    GREEDY = 0,
    OPTIMAL2 = 1,
};

LaneMatchSolver stateflow_lane_match_solver() {
    const char *raw = stateflow_env_value("GEGE_STATEFLOW_LANE_MATCH_SOLVER", "STATEFLOW_LANE_MATCH_SOLVER");
    if (raw == nullptr) {
        return LaneMatchSolver::OPTIMAL2;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "greedy") {
        return LaneMatchSolver::GREEDY;
    }
    if (value == "optimal2" || value == "hungarian") {
        return LaneMatchSolver::OPTIMAL2;
    }
    SPDLOG_WARN("Ignoring invalid lane match solver '{}'; using optimal2", value);
    return LaneMatchSolver::OPTIMAL2;
}

LaneMatchCostConfig lane_match_cost_config_from_env() {
    LaneMatchCostConfig cfg;
    cfg.peer_bandwidth_bps =
        std::max<int64_t>(1, stateflow_env_int64("GEGE_STATEFLOW_PEER_BANDWIDTH_BPS", "STATEFLOW_PEER_BANDWIDTH_BPS", 32000000000LL));
    cfg.host_bandwidth_bps =
        std::max<int64_t>(1, stateflow_env_int64("GEGE_STATEFLOW_HOST_BANDWIDTH_BPS", "STATEFLOW_HOST_BANDWIDTH_BPS", 16000000000LL));
    cfg.imbalance_weight = stateflow_cost_env("GEGE_STATEFLOW_LANE_IMBALANCE_WEIGHT", 1.0);
    cfg.boundary_weight = stateflow_cost_env("GEGE_STATEFLOW_LANE_BOUNDARY_WEIGHT", 1.0);
    cfg.allow_peer_relay = stateflow_env_bool("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "STATEFLOW_ALLOW_PEER_RELAY", false);
    return cfg;
}

int64_t plan_embedding_bytes_per_row(const PlanEmbeddingLayout &layout) {
    if (layout.embedding_dim <= 0 || layout.dtype_size <= 0) {
        return 0;
    }
    return std::max<int64_t>(layout.embedding_dim, 1) *
           std::max<int64_t>(layout.dtype_size, 1) *
           std::max<int64_t>(layout.optimizer_state_multiplier, 1);
}

int64_t partition_transfer_bytes(int64_t partition_id,
                                 const std::vector<int64_t> &partition_row_counts,
                                 int64_t bytes_per_row) {
    if (bytes_per_row <= 0) {
        return 1;
    }
    if (partition_id < 0 || partition_id >= static_cast<int64_t>(partition_row_counts.size())) {
        return bytes_per_row;
    }
    return std::max<int64_t>(partition_row_counts[partition_id], 1) * bytes_per_row;
}

int64_t resident_overlap_bytes(const std::vector<int64_t> &lhs_partitions,
                               const std::vector<int64_t> &rhs_partitions,
                               const std::vector<int64_t> &partition_row_counts,
                               int64_t bytes_per_row) {
    std::size_t left = 0;
    std::size_t right = 0;
    int64_t bytes = 0;
    while (left < lhs_partitions.size() && right < rhs_partitions.size()) {
        if (lhs_partitions[left] == rhs_partitions[right]) {
            bytes += partition_transfer_bytes(lhs_partitions[left], partition_row_counts, bytes_per_row);
            left++;
            right++;
        } else if (lhs_partitions[left] < rhs_partitions[right]) {
            left++;
        } else {
            right++;
        }
    }
    return bytes;
}

int64_t handoff_bytes_host(const std::vector<int64_t> &needed_partitions,
                           const std::vector<int64_t> &already_on_lane,
                           const std::vector<int64_t> &partition_row_counts,
                           int64_t bytes_per_row) {
    std::unordered_set<int64_t> resident(already_on_lane.begin(), already_on_lane.end());
    int64_t bytes = 0;
    for (int64_t partition_id : needed_partitions) {
        if (resident.count(partition_id) == 0) {
            bytes += partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
        }
    }
    return bytes;
}

int64_t handoff_bytes_peer(const std::vector<int64_t> &needed_partitions,
                           const std::vector<int64_t> &already_on_lane,
                           const std::vector<int64_t> &other_lane_resident,
                           const std::vector<int64_t> &partition_row_counts,
                           int64_t bytes_per_row) {
    std::unordered_set<int64_t> resident(already_on_lane.begin(), already_on_lane.end());
    std::unordered_set<int64_t> peer(other_lane_resident.begin(), other_lane_resident.end());
    int64_t bytes = 0;
    for (int64_t partition_id : needed_partitions) {
        if (resident.count(partition_id) == 0 && peer.count(partition_id) > 0) {
            bytes += partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
        }
    }
    return bytes;
}

int64_t optimized_custom_schedule_restarts() {
    const char *raw = std::getenv("GEGE_CUSTOM_OPTIMIZER_RESTARTS");
    if (raw == nullptr) {
        return 8;
    }

    try {
        return std::max<int64_t>(std::stoll(std::string(raw)), 1);
    } catch (...) {
        return 8;
    }
}

int64_t optimized_custom_schedule_seed() {
    const char *raw = std::getenv("GEGE_CUSTOM_OPTIMIZER_SEED");
    if (raw == nullptr) {
        return 12345;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return 12345;
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getHybridCoverEdgeBucketOrdering(int num_partitions, int buffer_capacity);

std::vector<std::vector<int>> build_custom_template_buffer_states(int num_partitions, int buffer_capacity) {
    assert(buffer_capacity == 4);
    int32_t sub_chunk_per_perm = num_partitions / buffer_capacity;
    int32_t log2l = 0;

    while (pow(2, log2l) < num_partitions) {
        log2l += 1;
    }

    assert(pow(2, log2l) == num_partitions);

    std::vector<std::vector<std::vector<int>>> offset_supergroup = {
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}},
        {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}},
        {{0, 2, 3, 1}, {1, 3, 2, 0}, {2, 0, 1, 3}, {3, 1, 0, 2}},
        {{0, 3, 1, 2}, {1, 2, 0, 3}, {2, 1, 3, 0}, {3, 0, 2, 1}},
    };
    std::vector<std::vector<std::vector<int>>> p = {{{0, 1, 2, 3}}};

    for (int log4l_pre = 1; log4l_pre < log2l / 2; log4l_pre++) {
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        for (auto &s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int offset = 0; offset < pow(4, log4l_pre + 1); offset += pow(4, log4l_pre)) {
                for (auto &g : s) {
                    std::vector<int> g_cur;
                    for (auto &x : g) {
                        g_cur.emplace_back(x + offset);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }
        int32_t len = p_pre.size();
        for (int i = len - pow(4, log4l_pre - 1); i < len; i++) {
            auto s = p_pre[i];
            for (auto &offset_s : offset_supergroup) {
                std::vector<std::vector<int>> s_cur;
                for (auto &g : s) {
                    for (auto &offset_g : offset_s) {
                        std::vector<int> g_cur;
                        for (int j = 0; j < 4; j++) {
                            g_cur.emplace_back(g[j] * 4 + offset_g[j]);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> pairing_chunks = {
        {{0, 2}, {1, 3}},
        {{0, 3}, {1, 2}}
    };

    if (log2l % 2 == 1) {
        int32_t len_chunk = sub_chunk_per_perm;
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();

        for (auto &s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int i = 0; i < pow(2, log2l); i += pow(2, log2l - 1)) {
                for (auto &g : s) {
                    std::vector<int> g_cur;
                    for (auto &x : g) {
                        g_cur.emplace_back(x + i);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }

        int32_t len = p_pre.size();
        for (int i = len - pow(2, log2l - 3); i < len; i++) {
            std::vector<std::vector<int>> s = p_pre[i];
            for (auto &pairing_s : pairing_chunks) {
                std::vector<std::vector<int>> s_cur;
                for (auto &chunk_index : pairing_s) {
                    for (auto &g : s) {
                        std::vector<int> g_cur;
                        for (auto &x : g) {
                            g_cur.emplace_back(chunk_index[x / len_chunk] * len_chunk + x % len_chunk);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }

    std::vector<std::vector<int>> buffer_states;
    for (auto &supergroup : p) {
        for (auto &state : supergroup) {
            buffer_states.emplace_back(state);
        }
    }

    return buffer_states;
}

struct CustomStateMetrics {
    std::vector<int> partitions;
    int64_t weight = 0;
    int64_t batches = 0;
    int64_t bucket_count = 0;
};

struct CustomScheduleScore {
    int64_t worst_round_spread = 0;
    int64_t worst_batch_spread = 0;
    int64_t worst_state_weight = 0;
    int64_t total_round_spread = 0;
    int64_t continuity_hotness = 0;
    int64_t continuity_new_partitions = 0;
    int64_t total_abs_deviation = 0;

    auto as_tuple() const {
        return std::make_tuple(worst_round_spread, worst_batch_spread, worst_state_weight, total_round_spread, continuity_hotness,
                               continuity_new_partitions, total_abs_deviation);
    }
};

struct CustomEvaluatedSchedule {
    std::vector<int> slot_to_partition;
    std::vector<CustomStateMetrics> states;
    std::vector<std::vector<int>> rounds;
    std::vector<std::vector<int>> lane_rounds;
    CustomScheduleScore score;
};

int64_t compute_state_resident_weight(const std::vector<int> &partitions,
                                      const std::vector<int64_t> &edge_bucket_sizes,
                                      int num_partitions) {
    int64_t resident_weight = 0;
    for (auto src_part : partitions) {
        for (auto dst_part : partitions) {
            resident_weight += edge_bucket_sizes[src_part * num_partitions + dst_part];
        }
    }
    return resident_weight;
}

int select_startup_round(const std::vector<std::vector<int>> &lane_rounds,
                         const std::vector<int64_t> &resident_state_weights,
                         const std::vector<CustomStateMetrics> &state_metrics) {
    int best_round = 0;
    auto best_key = std::make_tuple(std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max());

    for (int round_idx = 0; round_idx < static_cast<int>(lane_rounds.size()); round_idx++) {
        int64_t max_resident_weight = 0;
        int64_t total_resident_weight = 0;
        int64_t max_assigned_weight = 0;
        int64_t total_assigned_weight = 0;

        for (auto state_idx : lane_rounds[round_idx]) {
            max_resident_weight = std::max<int64_t>(max_resident_weight, resident_state_weights[state_idx]);
            total_resident_weight += resident_state_weights[state_idx];
            max_assigned_weight = std::max<int64_t>(max_assigned_weight, state_metrics[state_idx].weight);
            total_assigned_weight += state_metrics[state_idx].weight;
        }

        auto key = std::make_tuple(max_resident_weight, total_resident_weight, max_assigned_weight, total_assigned_weight, round_idx);
        if (key < best_key) {
            best_key = key;
            best_round = round_idx;
        }
    }

    return best_round;
}

bool custom_score_better(const CustomScheduleScore &lhs, const CustomScheduleScore &rhs) {
    return lhs.as_tuple() < rhs.as_tuple();
}

int int_pow_local(int a, int x) {
    int ans = 1;
    int temp = a;
    while (x) {
        if (x & 1) {
            ans *= temp;
        }
        temp *= temp;
        x >>= 1;
    }
    return ans;
}

std::vector<std::vector<int>> build_custom_template_states(int num_partitions, int buffer_capacity) {
    assert(buffer_capacity == 4);
    int32_t sub_chunk_per_perm = num_partitions / buffer_capacity;
    int32_t log2l = 0;

    while (int_pow_local(2, log2l) < num_partitions) {
        log2l += 1;
    }

    assert(int_pow_local(2, log2l) == num_partitions);

    std::vector<std::vector<std::vector<int>>> offset_supergroup = {
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}},
        {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}},
        {{0, 2, 3, 1}, {1, 3, 2, 0}, {2, 0, 1, 3}, {3, 1, 0, 2}},
        {{0, 3, 1, 2}, {1, 2, 0, 3}, {2, 1, 3, 0}, {3, 0, 2, 1}},
    };
    std::vector<std::vector<std::vector<int>>> p = {{{0, 1, 2, 3}}};

    for (int log4l_pre = 1; log4l_pre < log2l / 2; log4l_pre++) {
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        for (auto &s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int offset = 0; offset < int_pow_local(4, log4l_pre + 1); offset += int_pow_local(4, log4l_pre)) {
                for (auto &g : s) {
                    std::vector<int> g_cur;
                    for (auto &x : g) {
                        g_cur.emplace_back(x + offset);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }
        int32_t len = p_pre.size();
        for (int i = len - int_pow_local(4, log4l_pre - 1); i < len; i++) {
            auto s = p_pre[i];
            for (auto &offset_s : offset_supergroup) {
                std::vector<std::vector<int>> s_cur;
                for (auto &g : s) {
                    for (auto &offset_g : offset_s) {
                        std::vector<int> g_cur;
                        for (int j = 0; j < 4; j++) {
                            g_cur.emplace_back(g[j] * 4 + offset_g[j]);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }

    std::vector<std::vector<std::vector<int>>> pairing_chunks = {
        {{0, 2}, {1, 3}},
        {{0, 3}, {1, 2}}
    };

    if (log2l % 2 == 1) {
        int32_t len_chunk = sub_chunk_per_perm;
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();

        for (auto &s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int i = 0; i < int_pow_local(2, log2l); i += int_pow_local(2, log2l - 1)) {
                for (auto &g : s) {
                    std::vector<int> g_cur;
                    for (auto &x : g) {
                        g_cur.emplace_back(x + i);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }

        int32_t len = p_pre.size();
        for (int i = len - int_pow_local(2, log2l - 3); i < len; i++) {
            std::vector<std::vector<int>> s = p_pre[i];
            for (auto &pairing_s : pairing_chunks) {
                std::vector<std::vector<int>> s_cur;
                for (auto &chunk_index : pairing_s) {
                    for (auto &g : s) {
                        std::vector<int> g_cur;
                        for (auto &x : g) {
                            g_cur.emplace_back(chunk_index[x / len_chunk] * len_chunk + x % len_chunk);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }

    std::vector<std::vector<int>> buffer_states;
    for (auto &supergroup : p) {
        for (auto &state : supergroup) {
            buffer_states.emplace_back(state);
        }
    }
    return buffer_states;
}

std::vector<std::vector<int>> build_slot_pair_owners(const std::vector<std::vector<int>> &template_states, int num_slots) {
    std::vector<std::vector<int>> owners(num_slots, std::vector<int>(num_slots, -1));
    for (int state_idx = 0; state_idx < template_states.size(); state_idx++) {
        for (auto src_slot : template_states[state_idx]) {
            for (auto dst_slot : template_states[state_idx]) {
                if (owners[src_slot][dst_slot] == -1) {
                    owners[src_slot][dst_slot] = state_idx;
                }
            }
        }
    }

    for (int src_slot = 0; src_slot < num_slots; src_slot++) {
        for (int dst_slot = 0; dst_slot < num_slots; dst_slot++) {
            if (owners[src_slot][dst_slot] == -1) {
                throw std::runtime_error("No owner state found for slot pair");
            }
        }
    }

    return owners;
}

std::vector<int64_t> build_partition_hotness(const std::vector<int64_t> &edge_bucket_sizes, int num_partitions) {
    std::vector<int64_t> hotness(num_partitions, 0);
    for (int partition = 0; partition < num_partitions; partition++) {
        int64_t outgoing = 0;
        int64_t incoming = 0;
        for (int other = 0; other < num_partitions; other++) {
            outgoing += edge_bucket_sizes[partition * num_partitions + other];
            incoming += edge_bucket_sizes[other * num_partitions + partition];
        }
        hotness[partition] = outgoing + incoming - edge_bucket_sizes[partition * num_partitions + partition];
    }
    return hotness;
}

std::vector<int> lite_initial_assignment(const std::vector<std::vector<int>> &template_states,
                                         const std::vector<int64_t> &edge_bucket_sizes,
                                         const std::vector<int64_t> &hotness,
                                         int num_partitions,
                                         int active_devices) {
    auto slot_pair_owners = build_slot_pair_owners(template_states, num_partitions);
    std::vector<int64_t> state_weights(template_states.size(), 0);
    std::vector<int> slot_to_partition(num_partitions, -1);
    std::vector<int> assigned_slots;
    const double total_weight = std::accumulate(edge_bucket_sizes.begin(), edge_bucket_sizes.end(), 0.0);
    const double target_state_weight = total_weight / static_cast<double>(template_states.size());

    std::vector<int> sorted_partitions(num_partitions);
    std::iota(sorted_partitions.begin(), sorted_partitions.end(), 0);
    std::sort(sorted_partitions.begin(), sorted_partitions.end(), [&](int lhs, int rhs) {
        if (hotness[lhs] != hotness[rhs]) {
            return hotness[lhs] > hotness[rhs];
        }
        return lhs < rhs;
    });

    for (auto partition : sorted_partitions) {
        int best_slot = -1;
        std::tuple<int64_t, int64_t, double, double, int> best_key{std::numeric_limits<int64_t>::max(),
                                                                    std::numeric_limits<int64_t>::max(),
                                                                    std::numeric_limits<double>::max(),
                                                                    std::numeric_limits<double>::max(),
                                                                    std::numeric_limits<int>::max()};

        for (int slot = 0; slot < num_partitions; slot++) {
            if (slot_to_partition[slot] != -1) {
                continue;
            }

            auto candidate_weights = state_weights;
            int diagonal_owner = slot_pair_owners[slot][slot];
            candidate_weights[diagonal_owner] += edge_bucket_sizes[partition * num_partitions + partition];

            for (auto other_slot : assigned_slots) {
                int other_partition = slot_to_partition[other_slot];
                int forward_owner = slot_pair_owners[slot][other_slot];
                int reverse_owner = slot_pair_owners[other_slot][slot];
                candidate_weights[forward_owner] += edge_bucket_sizes[partition * num_partitions + other_partition];
                candidate_weights[reverse_owner] += edge_bucket_sizes[other_partition * num_partitions + partition];
            }

            int64_t max_round_max = 0;
            int64_t max_round_spread = 0;
            double total_over_target = 0.0;
            double total_abs_deviation = 0.0;
            for (int round_start = 0; round_start < candidate_weights.size(); round_start += active_devices) {
                auto begin = candidate_weights.begin() + round_start;
                auto end = begin + std::min<int>(active_devices, candidate_weights.size() - round_start);
                auto [round_min_it, round_max_it] = std::minmax_element(begin, end);
                max_round_max = std::max<int64_t>(max_round_max, *round_max_it);
                max_round_spread = std::max<int64_t>(max_round_spread, *round_max_it - *round_min_it);
            }
            for (auto weight : candidate_weights) {
                total_over_target += std::max<double>(weight - target_state_weight, 0.0);
                total_abs_deviation += std::abs(weight - target_state_weight);
            }

            auto candidate_key = std::make_tuple(max_round_max, max_round_spread, total_over_target, total_abs_deviation, slot);
            if (candidate_key < best_key) {
                best_key = candidate_key;
                best_slot = slot;
            }
        }

        if (best_slot == -1) {
            throw std::runtime_error("Failed to construct optimized CUSTOM initial assignment");
        }

        slot_to_partition[best_slot] = partition;
        int diagonal_owner = slot_pair_owners[best_slot][best_slot];
        state_weights[diagonal_owner] += edge_bucket_sizes[partition * num_partitions + partition];
        for (auto other_slot : assigned_slots) {
            int other_partition = slot_to_partition[other_slot];
            int forward_owner = slot_pair_owners[best_slot][other_slot];
            int reverse_owner = slot_pair_owners[other_slot][best_slot];
            state_weights[forward_owner] += edge_bucket_sizes[partition * num_partitions + other_partition];
            state_weights[reverse_owner] += edge_bucket_sizes[other_partition * num_partitions + partition];
        }
        assigned_slots.emplace_back(best_slot);
    }

    return slot_to_partition;
}

std::pair<int64_t, int64_t> custom_state_transition_cost(const CustomStateMetrics &previous_state,
                                                         const CustomStateMetrics &next_state,
                                                         const std::vector<int64_t> &hotness) {
    int64_t transition_hotness = 0;
    int64_t transition_new_partitions = 0;

    for (auto partition : next_state.partitions) {
        if (std::find(previous_state.partitions.begin(), previous_state.partitions.end(), partition) == previous_state.partitions.end()) {
            transition_hotness += hotness[partition];
            transition_new_partitions++;
        }
    }
    return std::make_pair(transition_hotness, transition_new_partitions);
}

std::tuple<std::vector<std::vector<int>>, int64_t, int64_t> optimize_custom_lane_assignment(
    const std::vector<std::vector<int>> &rounds,
    const std::vector<CustomStateMetrics> &states,
    const std::vector<int64_t> &hotness) {
    if (rounds.empty()) {
        return std::make_tuple(std::vector<std::vector<int>>(), 0, 0);
    }

    std::vector<std::vector<std::vector<int>>> all_permutations;
    all_permutations.reserve(rounds.size());
    for (auto &round : rounds) {
        std::vector<int> permutation(round.size());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::vector<std::vector<int>> round_permutations;
        do {
            round_permutations.emplace_back(permutation);
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        all_permutations.emplace_back(std::move(round_permutations));
    }

    std::vector<std::vector<std::pair<int64_t, int64_t>>> dp(rounds.size());
    std::vector<std::vector<int>> backpointers(rounds.size());
    dp[0].assign(all_permutations[0].size(), std::make_pair(0, 0));
    backpointers[0].assign(all_permutations[0].size(), -1);

    for (int round_idx = 1; round_idx < rounds.size(); round_idx++) {
        dp[round_idx].assign(all_permutations[round_idx].size(),
                             std::make_pair(std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()));
        backpointers[round_idx].assign(all_permutations[round_idx].size(), -1);

        for (int permutation_idx = 0; permutation_idx < all_permutations[round_idx].size(); permutation_idx++) {
            const auto &permutation = all_permutations[round_idx][permutation_idx];

            for (int previous_permutation_idx = 0; previous_permutation_idx < all_permutations[round_idx - 1].size(); previous_permutation_idx++) {
                const auto &previous_permutation = all_permutations[round_idx - 1][previous_permutation_idx];
                int64_t transition_hotness = 0;
                int64_t transition_new_partitions = 0;

                for (int lane_idx = 0; lane_idx < permutation.size(); lane_idx++) {
                    const auto &previous_state = states[rounds[round_idx - 1][previous_permutation[lane_idx]]];
                    const auto &next_state = states[rounds[round_idx][permutation[lane_idx]]];
                    auto [lane_hotness, lane_new_partitions] = custom_state_transition_cost(previous_state, next_state, hotness);
                    transition_hotness += lane_hotness;
                    transition_new_partitions += lane_new_partitions;
                }

                auto candidate_cost =
                    std::make_pair(dp[round_idx - 1][previous_permutation_idx].first + transition_hotness,
                                   dp[round_idx - 1][previous_permutation_idx].second + transition_new_partitions);
                if (candidate_cost < dp[round_idx][permutation_idx]) {
                    dp[round_idx][permutation_idx] = candidate_cost;
                    backpointers[round_idx][permutation_idx] = previous_permutation_idx;
                }
            }
        }
    }

    int best_final_idx = 0;
    for (int permutation_idx = 1; permutation_idx < dp.back().size(); permutation_idx++) {
        if (dp.back()[permutation_idx] < dp.back()[best_final_idx]) {
            best_final_idx = permutation_idx;
        }
    }

    std::vector<int> chosen(rounds.size(), -1);
    chosen.back() = best_final_idx;
    for (int round_idx = rounds.size() - 1; round_idx > 0; round_idx--) {
        chosen[round_idx - 1] = backpointers[round_idx][chosen[round_idx]];
    }

    std::vector<std::vector<int>> lane_rounds;
    lane_rounds.reserve(rounds.size());
    for (int round_idx = 0; round_idx < rounds.size(); round_idx++) {
        std::vector<int> lane_round;
        lane_round.reserve(rounds[round_idx].size());
        for (auto local_idx : all_permutations[round_idx][chosen[round_idx]]) {
            lane_round.emplace_back(rounds[round_idx][local_idx]);
        }
        lane_rounds.emplace_back(std::move(lane_round));
    }

    return std::make_tuple(lane_rounds, dp.back()[best_final_idx].first, dp.back()[best_final_idx].second);
}

CustomEvaluatedSchedule summarize_custom_schedule(const std::vector<std::vector<int>> &template_states,
                                                 const std::vector<int> &slot_to_partition,
                                                 const std::vector<int64_t> &edge_bucket_sizes,
                                                 const std::vector<int64_t> &hotness,
                                                 int num_partitions,
                                                 int active_devices,
                                                 int batch_size) {
    std::vector<std::vector<int>> mapped_states = template_states;
    for (auto &state : mapped_states) {
        for (auto &slot : state) {
            slot = slot_to_partition[slot];
        }
    }

    auto edge_buckets = greedyAssignEdgeBucketsToBuffers(mapped_states, num_partitions);

    std::vector<CustomStateMetrics> state_metrics;
    state_metrics.reserve(mapped_states.size());
    for (int state_idx = 0; state_idx < mapped_states.size(); state_idx++) {
        int64_t weight = 0;
        for (auto &[src, dst] : edge_buckets[state_idx]) {
            weight += edge_bucket_sizes[src * num_partitions + dst];
        }
        int64_t batches = (weight + batch_size - 1) / batch_size;
        state_metrics.push_back({mapped_states[state_idx], weight, batches, static_cast<int64_t>(edge_buckets[state_idx].size())});
    }

    std::vector<std::vector<int>> rounds;
    for (int i = 0; i < state_metrics.size(); i += active_devices) {
        std::vector<int> round;
        for (int j = i; j < std::min<int>(i + active_devices, state_metrics.size()); j++) {
            round.emplace_back(j);
        }
        rounds.emplace_back(std::move(round));
    }

    auto [lane_rounds, continuity_hotness, continuity_new_partitions] = optimize_custom_lane_assignment(rounds, state_metrics, hotness);

    int64_t worst_round_spread = 0;
    int64_t worst_batch_spread = 0;
    int64_t worst_state_weight = 0;
    int64_t total_round_spread = 0;
    double total_abs_deviation = 0.0;
    for (auto &round : rounds) {
        int64_t round_min_weight = std::numeric_limits<int64_t>::max();
        int64_t round_max_weight = 0;
        int64_t round_min_batches = std::numeric_limits<int64_t>::max();
        int64_t round_max_batches = 0;
        double round_mean_weight = 0.0;
        for (auto state_idx : round) {
            round_min_weight = std::min<int64_t>(round_min_weight, state_metrics[state_idx].weight);
            round_max_weight = std::max<int64_t>(round_max_weight, state_metrics[state_idx].weight);
            round_min_batches = std::min<int64_t>(round_min_batches, state_metrics[state_idx].batches);
            round_max_batches = std::max<int64_t>(round_max_batches, state_metrics[state_idx].batches);
            worst_state_weight = std::max<int64_t>(worst_state_weight, state_metrics[state_idx].weight);
            round_mean_weight += static_cast<double>(state_metrics[state_idx].weight);
        }
        round_mean_weight /= static_cast<double>(round.size());
        for (auto state_idx : round) {
            total_abs_deviation += std::abs(static_cast<double>(state_metrics[state_idx].weight) - round_mean_weight);
        }
        int64_t round_spread = round_max_weight - round_min_weight;
        int64_t batch_spread = round_max_batches - round_min_batches;
        worst_round_spread = std::max<int64_t>(worst_round_spread, round_spread);
        worst_batch_spread = std::max<int64_t>(worst_batch_spread, batch_spread);
        total_round_spread += round_spread;
    }

    CustomScheduleScore score = {
        worst_round_spread,
        worst_batch_spread,
        worst_state_weight,
        total_round_spread,
        continuity_hotness,
        continuity_new_partitions,
        static_cast<int64_t>(total_abs_deviation),
    };

    return {slot_to_partition, state_metrics, rounds, lane_rounds, score};
}

CustomEvaluatedSchedule steepest_descent_custom_schedule(const std::vector<std::vector<int>> &template_states,
                                                         const std::vector<int> &initial_assignment,
                                                         const std::vector<int64_t> &edge_bucket_sizes,
                                                         const std::vector<int64_t> &hotness,
                                                         int num_partitions,
                                                         int active_devices,
                                                         int batch_size) {
    std::vector<int> assignment = initial_assignment;
    auto current = summarize_custom_schedule(template_states, assignment, edge_bucket_sizes, hotness, num_partitions, active_devices, batch_size);

    bool improved = true;
    while (improved) {
        improved = false;
        auto best_candidate = current;
        std::pair<int, int> best_swap{-1, -1};

        for (int i = 0; i < assignment.size(); i++) {
            for (int j = i + 1; j < assignment.size(); j++) {
                auto candidate_assignment = assignment;
                std::swap(candidate_assignment[i], candidate_assignment[j]);
                auto candidate =
                    summarize_custom_schedule(template_states, candidate_assignment, edge_bucket_sizes, hotness, num_partitions, active_devices, batch_size);
                if (custom_score_better(candidate.score, best_candidate.score)) {
                    best_candidate = std::move(candidate);
                    best_swap = std::make_pair(i, j);
                }
            }
        }

        if (best_swap.first != -1) {
            std::swap(assignment[best_swap.first], assignment[best_swap.second]);
            current = std::move(best_candidate);
            improved = true;
        }
    }

    return current;
}

CustomEvaluatedSchedule optimize_custom_schedule(const std::vector<std::vector<int>> &template_states,
                                                 const std::vector<int64_t> &edge_bucket_sizes,
                                                 int num_partitions,
                                                 int active_devices,
                                                 int batch_size) {
    auto hotness = build_partition_hotness(edge_bucket_sizes, num_partitions);
    std::mt19937 rng(static_cast<uint32_t>(optimized_custom_schedule_seed()));
    int64_t restarts = optimized_custom_schedule_restarts();

    std::vector<std::vector<int>> initial_assignments;
    initial_assignments.emplace_back(lite_initial_assignment(template_states, edge_bucket_sizes, hotness, num_partitions, active_devices));

    std::vector<int> identity(num_partitions);
    std::iota(identity.begin(), identity.end(), 0);
    initial_assignments.emplace_back(identity);

    for (int64_t restart = 1; restart < restarts; restart++) {
        auto candidate = identity;
        std::shuffle(candidate.begin(), candidate.end(), rng);
        initial_assignments.emplace_back(std::move(candidate));
    }

    bool has_best = false;
    CustomEvaluatedSchedule best;
    for (auto &initial_assignment : initial_assignments) {
        auto candidate = steepest_descent_custom_schedule(template_states, initial_assignment, edge_bucket_sizes, hotness, num_partitions,
                                                          active_devices, batch_size);
        if (!has_best || custom_score_better(candidate.score, best.score)) {
            best = std::move(candidate);
            has_best = true;
        }
    }

    if (!has_best) {
        throw std::runtime_error("No optimized CUSTOM schedule candidates were generated");
    }

    return best;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> build_optimized_custom_edge_bucket_ordering(int num_partitions,
                                                                                                      int buffer_capacity,
                                                                                                      int active_devices,
                                                                                                      int batch_size,
                                                                                                      const std::vector<int64_t> &edge_bucket_sizes) {
    auto template_states = build_custom_template_states(num_partitions, buffer_capacity);
    auto optimized = optimize_custom_schedule(template_states, edge_bucket_sizes, num_partitions, active_devices, batch_size);

    std::vector<std::vector<int>> mapped_states = template_states;
    for (auto &state : mapped_states) {
        for (auto &slot : state) {
            slot = optimized.slot_to_partition[slot];
        }
    }

    auto edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(mapped_states, num_partitions);
    std::vector<int64_t> resident_state_weights;
    resident_state_weights.reserve(mapped_states.size());
    for (const auto &state : mapped_states) {
        resident_state_weights.emplace_back(compute_state_resident_weight(state, edge_bucket_sizes, num_partitions));
    }
    std::vector<std::vector<int>> ordered_states;
    std::vector<std::vector<std::pair<int, int>>> ordered_buckets;
    ordered_states.reserve(mapped_states.size());
    ordered_buckets.reserve(edge_buckets_per_buffer.size());

    auto ordered_lane_rounds = optimized.lane_rounds;
    int startup_round = select_startup_round(ordered_lane_rounds, resident_state_weights, optimized.states);
    if (startup_round > 0) {
        std::rotate(ordered_lane_rounds.begin(), ordered_lane_rounds.begin() + startup_round, ordered_lane_rounds.end());
    }

    for (const auto &lane_round : ordered_lane_rounds) {
        for (auto state_idx : lane_round) {
            ordered_states.emplace_back(mapped_states[state_idx]);
            ordered_buckets.emplace_back(edge_buckets_per_buffer[state_idx]);
        }
    }

    std::ostringstream slot_mapping;
    for (int idx = 0; idx < optimized.slot_to_partition.size(); idx++) {
        if (idx > 0) {
            slot_mapping << ",";
        }
        slot_mapping << optimized.slot_to_partition[idx];
    }

    // SPDLOG_INFO(
    //     "Using optimized CUSTOM ordering: worst_round_spread={:.3f}M worst_batch_spread={} total_round_spread={:.3f}M continuity_hotness={:.3f}M continuity_new_partitions={}",
    //     optimized.score.worst_round_spread / 1000000.0,
    //     optimized.score.worst_batch_spread,
    //     optimized.score.total_round_spread / 1000000.0,
    //     optimized.score.continuity_hotness / 1000000.0,
    //     optimized.score.continuity_new_partitions);
    // SPDLOG_INFO("Optimized CUSTOM slot_to_partition=[{}]", slot_mapping.str());
    if (!ordered_lane_rounds.empty()) {
        int64_t startup_max_resident_weight = 0;
        int64_t startup_total_resident_weight = 0;
        for (auto state_idx : ordered_lane_rounds.front()) {
            startup_max_resident_weight = std::max<int64_t>(startup_max_resident_weight, resident_state_weights[state_idx]);
            startup_total_resident_weight += resident_state_weights[state_idx];
        }
        // SPDLOG_INFO("Optimized CUSTOM startup round={} startup_max_resident_edges={:.3f}M startup_total_resident_edges={:.3f}M",
        //             startup_round,
        //             startup_max_resident_weight / 1000000.0,
        //             startup_total_resident_weight / 1000000.0);
    }

    return convertEdgeBucketOrderToTensors(ordered_states, ordered_buckets);
}

bool states_disjoint(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs) {
    for (auto left_part : lhs) {
        for (auto right_part : rhs) {
            if (left_part == right_part) {
                return false;
            }
        }
    }
    return true;
}

bool search_disjoint_groups(const std::vector<std::vector<bool>> &compatible, const std::vector<int64_t> &remaining, int active_devices,
                            std::vector<std::vector<int64_t>> &groups);

bool search_group_members(const std::vector<std::vector<bool>> &compatible, const std::vector<int64_t> &remaining, const int active_devices,
                          const int target_group_size, const std::vector<int64_t> &candidates, std::vector<int64_t> &current_group,
                          std::vector<std::vector<int64_t>> &groups) {
    if (current_group.size() == static_cast<std::size_t>(target_group_size)) {
        std::vector<int64_t> next_remaining;
        next_remaining.reserve(remaining.size() - current_group.size());
        for (auto state_idx : remaining) {
            if (std::find(current_group.begin(), current_group.end(), state_idx) == current_group.end()) {
                next_remaining.emplace_back(state_idx);
            }
        }
        groups.emplace_back(current_group);
        if (search_disjoint_groups(compatible, next_remaining, active_devices, groups)) {
            return true;
        }
        groups.pop_back();
        return false;
    }

    if (current_group.size() + candidates.size() < static_cast<std::size_t>(target_group_size)) {
        return false;
    }

    for (std::size_t i = 0; i < candidates.size(); i++) {
        int64_t candidate = candidates[i];
        std::vector<int64_t> next_candidates;
        next_candidates.reserve(candidates.size() - i - 1);
        for (std::size_t j = i + 1; j < candidates.size(); j++) {
            if (compatible[candidate][candidates[j]]) {
                next_candidates.emplace_back(candidates[j]);
            }
        }
        current_group.emplace_back(candidate);
        if (search_group_members(compatible, remaining, active_devices, target_group_size, next_candidates, current_group, groups)) {
            return true;
        }
        current_group.pop_back();
    }

    return false;
}

bool search_disjoint_groups(const std::vector<std::vector<bool>> &compatible, const std::vector<int64_t> &remaining, int active_devices,
                            std::vector<std::vector<int64_t>> &groups) {
    if (remaining.empty()) {
        return true;
    }

    int target_group_size = std::min<int>(active_devices, remaining.size());
    if (target_group_size <= 1) {
        groups.emplace_back(remaining);
        return true;
    }

    auto anchor_it = std::min_element(remaining.begin(), remaining.end(), [&](int64_t lhs, int64_t rhs) {
        int lhs_degree = 0;
        int rhs_degree = 0;
        for (auto state_idx : remaining) {
            lhs_degree += compatible[lhs][state_idx] ? 1 : 0;
            rhs_degree += compatible[rhs][state_idx] ? 1 : 0;
        }
        return lhs_degree < rhs_degree;
    });
    int64_t anchor = *anchor_it;

    std::vector<int64_t> candidates;
    for (auto state_idx : remaining) {
        if (state_idx != anchor && compatible[anchor][state_idx]) {
            candidates.emplace_back(state_idx);
        }
    }

    std::vector<int64_t> current_group = {anchor};
    return search_group_members(compatible, remaining, active_devices, target_group_size, candidates, current_group, groups);
}

std::vector<std::vector<int64_t>> build_disjoint_groups(const vector<torch::Tensor> &buffer_states, int active_devices) {
    std::vector<std::vector<int64_t>> groups;
    if (active_devices <= 1 || buffer_states.size() <= 1) {
        groups.reserve(buffer_states.size());
        for (std::size_t i = 0; i < buffer_states.size(); i++) {
            groups.push_back({static_cast<int64_t>(i)});
        }
        return groups;
    }

    std::vector<std::vector<int64_t>> state_partitions;
    state_partitions.reserve(buffer_states.size());
    for (auto &state : buffer_states) {
        state_partitions.emplace_back(tensor_to_partitions(state));
    }

    std::vector<std::vector<bool>> compatible(buffer_states.size(), std::vector<bool>(buffer_states.size(), false));
    for (std::size_t i = 0; i < buffer_states.size(); i++) {
        compatible[i][i] = true;
        for (std::size_t j = i + 1; j < buffer_states.size(); j++) {
            bool disjoint = states_disjoint(state_partitions[i], state_partitions[j]);
            compatible[i][j] = disjoint;
            compatible[j][i] = disjoint;
        }
    }

    std::vector<int64_t> remaining(buffer_states.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    if (!search_disjoint_groups(compatible, remaining, active_devices, groups)) {
        groups.clear();
        groups.reserve(buffer_states.size());
        for (std::size_t i = 0; i < buffer_states.size(); i++) {
            groups.push_back({static_cast<int64_t>(i)});
        }
    }

    return groups;
}

std::vector<StateAccessSummary> build_state_access_summaries(const vector<torch::Tensor> &buffer_states,
                                                             const vector<torch::Tensor> &edge_buckets_per_buffer) {
    std::vector<StateAccessSummary> summaries(buffer_states.size());
    for (std::size_t i = 0; i < buffer_states.size(); i++) {
        auto partitions = tensor_to_partitions(buffer_states[i]);
        std::sort(partitions.begin(), partitions.end());
        summaries[i].partitions = std::move(partitions);

        if (i >= edge_buckets_per_buffer.size()) {
            continue;
        }

        if (!edge_buckets_per_buffer[i].defined()) {
            continue;
        }

        auto edge_buckets = edge_buckets_per_buffer[i].to(torch::kCPU).to(torch::kInt64).contiguous();
        if (!edge_buckets.defined() || edge_buckets.numel() == 0) {
            continue;
        }
        summaries[i].total_bucket_edges = edge_buckets.size(0);
        auto accessor = edge_buckets.accessor<int64_t, 2>();
        for (int64_t row = 0; row < edge_buckets.size(0); row++) {
            summaries[i].incident_bucket_counts[accessor[row][0]]++;
            summaries[i].incident_bucket_counts[accessor[row][1]]++;
        }
    }
    return summaries;
}

int64_t state_access_overlap_score(const StateAccessSummary &lhs, const StateAccessSummary &rhs) {
    std::size_t i = 0;
    std::size_t j = 0;
    int64_t score = 0;
    while (i < lhs.partitions.size() && j < rhs.partitions.size()) {
        if (lhs.partitions[i] == rhs.partitions[j]) {
            int64_t partition_id = lhs.partitions[i];
            auto lhs_it = lhs.incident_bucket_counts.find(partition_id);
            auto rhs_it = rhs.incident_bucket_counts.find(partition_id);
            int64_t lhs_incident = lhs_it == lhs.incident_bucket_counts.end() ? 0 : lhs_it->second;
            int64_t rhs_incident = rhs_it == rhs.incident_bucket_counts.end() ? 0 : rhs_it->second;
            score += 1000 + std::min(lhs_incident, rhs_incident);
            i++;
            j++;
        } else if (lhs.partitions[i] < rhs.partitions[j]) {
            i++;
        } else {
            j++;
        }
    }
    return score;
}

struct GroupAlignmentResult {
    int64_t score = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> ordered_group;
};

void search_best_group_alignment(const std::vector<int64_t> &prev_group,
                                 const std::vector<int64_t> &candidate_group,
                                 const std::vector<StateAccessSummary> &summaries,
                                 std::vector<bool> &used,
                                 std::vector<int64_t> &current,
                                 std::size_t slot,
                                 int64_t current_score,
                                 GroupAlignmentResult &best) {
    std::size_t target = std::min(prev_group.size(), candidate_group.size());
    if (slot == target) {
        std::vector<int64_t> ordered = current;
        for (auto candidate_state : candidate_group) {
            if (std::find(ordered.begin(), ordered.end(), candidate_state) == ordered.end()) {
                ordered.emplace_back(candidate_state);
            }
        }
        if (current_score > best.score || (current_score == best.score && ordered < best.ordered_group)) {
            best.score = current_score;
            best.ordered_group = std::move(ordered);
        }
        return;
    }

    for (std::size_t i = 0; i < candidate_group.size(); i++) {
        if (used[i]) {
            continue;
        }
        used[i] = true;
        int64_t candidate_state = candidate_group[i];
        current.emplace_back(candidate_state);
        int64_t step_score = state_access_overlap_score(summaries[prev_group[slot]], summaries[candidate_state]);
        search_best_group_alignment(prev_group, candidate_group, summaries, used, current, slot + 1, current_score + step_score, best);
        current.pop_back();
        used[i] = false;
    }
}

GroupAlignmentResult get_best_group_alignment(const std::vector<int64_t> &prev_group,
                                              const std::vector<int64_t> &candidate_group,
                                              const std::vector<StateAccessSummary> &summaries) {
    GroupAlignmentResult result;
    if (candidate_group.empty()) {
        result.score = 0;
        return result;
    }
    if (prev_group.empty()) {
        result.score = 0;
        result.ordered_group = candidate_group;
        return result;
    }

    std::vector<bool> used(candidate_group.size(), false);
    std::vector<int64_t> current;
    current.reserve(candidate_group.size());
    search_best_group_alignment(prev_group, candidate_group, summaries, used, current, 0, 0, result);
    if (result.score == std::numeric_limits<int64_t>::min()) {
        result.score = 0;
        result.ordered_group = candidate_group;
    }
    return result;
}

struct GroupSearchResult {
    int64_t score = std::numeric_limits<int64_t>::min();
    std::vector<int64_t> ordered_group;
    std::vector<int64_t> chosen_states;
};

void search_best_disjoint_group(const std::vector<std::vector<bool>> &compatible,
                                const std::vector<int64_t> &remaining,
                                const std::vector<int64_t> &prev_group,
                                const std::vector<StateAccessSummary> &summaries,
                                int target_group_size,
                                std::size_t start_idx,
                                std::vector<int64_t> &current_group,
                                GroupSearchResult &best) {
    if (current_group.size() == static_cast<std::size_t>(target_group_size)) {
        GroupSearchResult candidate;
        candidate.chosen_states = current_group;
        if (prev_group.empty()) {
            int64_t score = 0;
            for (auto remaining_state : remaining) {
                if (std::find(current_group.begin(), current_group.end(), remaining_state) != current_group.end()) {
                    continue;
                }
                int64_t best_overlap = 0;
                for (auto group_state : current_group) {
                    best_overlap = std::max(best_overlap, state_access_overlap_score(summaries[group_state], summaries[remaining_state]));
                }
                score += best_overlap;
            }
            candidate.score = score;
            candidate.ordered_group = current_group;
        } else {
            auto alignment = get_best_group_alignment(prev_group, current_group, summaries);
            candidate.score = alignment.score;
            candidate.ordered_group = std::move(alignment.ordered_group);
        }

        if (candidate.score > best.score || (candidate.score == best.score && candidate.ordered_group < best.ordered_group)) {
            best = std::move(candidate);
        }
        return;
    }

    for (std::size_t i = start_idx; i < remaining.size(); i++) {
        int64_t candidate_state = remaining[i];
        bool valid = true;
        for (auto chosen_state : current_group) {
            if (!compatible[candidate_state][chosen_state]) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }
        current_group.emplace_back(candidate_state);
        search_best_disjoint_group(compatible, remaining, prev_group, summaries, target_group_size, i + 1, current_group, best);
        current_group.pop_back();
    }
}

double lane_assignment_transition_cost(const StateAccessSummary &previous_state,
                                       const StateAccessSummary &next_state,
                                       const std::vector<int64_t> &partition_row_counts,
                                       const LaneMatchCostConfig &cfg,
                                       int64_t bytes_per_row) {
    const int64_t host_bytes =
        handoff_bytes_host(next_state.partitions, previous_state.partitions, partition_row_counts, bytes_per_row);
    const int64_t overlap_bytes =
        resident_overlap_bytes(previous_state.partitions, next_state.partitions, partition_row_counts, bytes_per_row);
    double cost = bytes_per_row > 0
                      ? static_cast<double>(host_bytes) / static_cast<double>(std::max<int64_t>(cfg.host_bandwidth_bps, 1))
                      : static_cast<double>(host_bytes);
    if (overlap_bytes == 0) {
        cost += cfg.boundary_weight;
    }
    return cost;
}

double lane_group_imbalance_penalty(const std::vector<int64_t> &group,
                                    const std::vector<StateAccessSummary> &summaries,
                                    const LaneMatchCostConfig &cfg) {
    if (group.empty()) {
        return 0.0;
    }

    double total_edges = 0.0;
    for (auto state_idx : group) {
        total_edges += static_cast<double>(summaries[state_idx].total_bucket_edges);
    }
    double mean_edges = total_edges / static_cast<double>(group.size());
    if (mean_edges <= 0.0) {
        return 0.0;
    }

    double variance = 0.0;
    for (auto state_idx : group) {
        double diff = static_cast<double>(summaries[state_idx].total_bucket_edges) - mean_edges;
        variance += diff * diff;
    }

    return cfg.imbalance_weight * (variance / mean_edges);
}

struct LaneAlignmentResult {
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int64_t> ordered_group;
};

LaneAlignmentResult search_best_lane_alignment_dp(const std::vector<int64_t> &prev_group,
                                                  const std::vector<int64_t> &candidate_group,
                                                  const std::vector<StateAccessSummary> &summaries,
                                                  const std::vector<int64_t> &partition_row_counts,
                                                  const LaneMatchCostConfig &cfg,
                                                  int64_t bytes_per_row,
                                                  uint64_t used_mask,
                                                  std::unordered_map<uint64_t, LaneAlignmentResult> &memo) {
    const std::size_t slot = static_cast<std::size_t>(__builtin_popcountll(used_mask));
    if (slot == prev_group.size()) {
        return LaneAlignmentResult{0.0, {}};
    }

    auto memo_it = memo.find(used_mask);
    if (memo_it != memo.end()) {
        return memo_it->second;
    }

    LaneAlignmentResult best;
    for (std::size_t candidate_idx = 0; candidate_idx < candidate_group.size(); candidate_idx++) {
        if (((used_mask >> candidate_idx) & 1ULL) != 0ULL) {
            continue;
        }

        LaneAlignmentResult suffix = search_best_lane_alignment_dp(prev_group, candidate_group, summaries, partition_row_counts, cfg,
                                                                   bytes_per_row, used_mask | (1ULL << candidate_idx), memo);
        if (suffix.cost == std::numeric_limits<double>::infinity()) {
            continue;
        }

        double step_cost = lane_assignment_transition_cost(summaries[prev_group[slot]], summaries[candidate_group[candidate_idx]],
                                                           partition_row_counts, cfg, bytes_per_row);
        std::vector<int64_t> ordered_group;
        ordered_group.reserve(1 + suffix.ordered_group.size());
        ordered_group.emplace_back(candidate_group[candidate_idx]);
        ordered_group.insert(ordered_group.end(), suffix.ordered_group.begin(), suffix.ordered_group.end());

        double candidate_cost = step_cost + suffix.cost;
        if (candidate_cost < best.cost || (candidate_cost == best.cost && ordered_group < best.ordered_group)) {
            best.cost = candidate_cost;
            best.ordered_group = std::move(ordered_group);
        }
    }

    memo.emplace(used_mask, best);
    return best;
}

LaneAlignmentResult get_best_lane_alignment(const std::vector<int64_t> &prev_group,
                                            const std::vector<int64_t> &candidate_group,
                                            const std::vector<StateAccessSummary> &summaries,
                                            const std::vector<int64_t> &partition_row_counts,
                                            const LaneMatchCostConfig &cfg,
                                            int64_t bytes_per_row) {
    if (candidate_group.empty()) {
        return LaneAlignmentResult{0.0, {}};
    }
    if (prev_group.empty()) {
        return LaneAlignmentResult{0.0, candidate_group};
    }
    if (prev_group.size() != candidate_group.size() || candidate_group.size() >= 64) {
        return LaneAlignmentResult{};
    }

    std::unordered_map<uint64_t, LaneAlignmentResult> memo;
    return search_best_lane_alignment_dp(prev_group, candidate_group, summaries, partition_row_counts, cfg, bytes_per_row, 0ULL, memo);
}

struct LaneGroupAssignmentResult {
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int64_t> ordered_group;
    std::vector<int64_t> chosen_states;
};

void search_best_cost_aware_group(const std::vector<std::vector<bool>> &compatible,
                                  const std::vector<int64_t> &remaining,
                                  const std::vector<int64_t> &prev_group,
                                  const std::vector<StateAccessSummary> &summaries,
                                  const std::vector<int64_t> &partition_row_counts,
                                  const LaneMatchCostConfig &cfg,
                                  int64_t bytes_per_row,
                                  int target_group_size,
                                  std::size_t start_idx,
                                  std::vector<int64_t> &current_group,
                                  LaneGroupAssignmentResult &best) {
    if (current_group.size() == static_cast<std::size_t>(target_group_size)) {
        LaneAlignmentResult alignment =
            get_best_lane_alignment(prev_group, current_group, summaries, partition_row_counts, cfg, bytes_per_row);
        if (alignment.cost == std::numeric_limits<double>::infinity()) {
            return;
        }

        LaneGroupAssignmentResult candidate;
        candidate.chosen_states = current_group;
        candidate.ordered_group = std::move(alignment.ordered_group);
        candidate.cost = alignment.cost + lane_group_imbalance_penalty(current_group, summaries, cfg);
        if (candidate.cost < best.cost || (candidate.cost == best.cost && candidate.ordered_group < best.ordered_group)) {
            best = std::move(candidate);
        }
        return;
    }

    if (current_group.size() + (remaining.size() - start_idx) < static_cast<std::size_t>(target_group_size)) {
        return;
    }

    for (std::size_t candidate_idx = start_idx; candidate_idx < remaining.size(); candidate_idx++) {
        int64_t candidate_state = remaining[candidate_idx];
        bool valid = true;
        for (auto chosen_state : current_group) {
            if (!compatible[candidate_state][chosen_state]) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        current_group.emplace_back(candidate_state);
        search_best_cost_aware_group(compatible, remaining, prev_group, summaries, partition_row_counts, cfg, bytes_per_row,
                                     target_group_size, candidate_idx + 1, current_group, best);
        current_group.pop_back();
    }
}

LaneGroupAssignmentResult get_best_cost_aware_lane_group(const std::vector<std::vector<bool>> &compatible,
                                                         const std::vector<int64_t> &remaining,
                                                         const std::vector<int64_t> &prev_group,
                                                         const std::vector<StateAccessSummary> &summaries,
                                                         const std::vector<int64_t> &partition_row_counts,
                                                         const LaneMatchCostConfig &cfg,
                                                         int64_t bytes_per_row,
                                                         int target_group_size) {
    LaneGroupAssignmentResult best;
    if (remaining.empty()) {
        return best;
    }
    if (target_group_size <= 1 || prev_group.size() != static_cast<std::size_t>(target_group_size)) {
        GroupSearchResult greedy;
        std::vector<int64_t> current_group;
        current_group.reserve(std::min<std::size_t>(remaining.size(), static_cast<std::size_t>(std::max(target_group_size, 1))));
        search_best_disjoint_group(compatible, remaining, prev_group, summaries, std::max(target_group_size, 1), 0, current_group, greedy);
        if (!greedy.chosen_states.empty()) {
            best.cost = 0.0;
            best.chosen_states = greedy.chosen_states;
            best.ordered_group = greedy.ordered_group;
        }
        return best;
    }

    std::vector<int64_t> current_group;
    current_group.reserve(target_group_size);
    search_best_cost_aware_group(compatible, remaining, prev_group, summaries, partition_row_counts, cfg, bytes_per_row, target_group_size, 0,
                                 current_group, best);

    if (best.cost == std::numeric_limits<double>::infinity()) {
        GroupSearchResult greedy;
        current_group.clear();
        search_best_disjoint_group(compatible, remaining, prev_group, summaries, target_group_size, 0, current_group, greedy);
        if (!greedy.chosen_states.empty()) {
            best.cost = 0.0;
            best.chosen_states = greedy.chosen_states;
            best.ordered_group = greedy.ordered_group;
        }
    }
    return best;
}

struct AccessAwareGeneratedState {
    std::vector<int> partitions;
    std::vector<std::pair<int, int>> newly_covered_buckets;
};

struct AccessAwareGroupSearch {
    int64_t score = std::numeric_limits<int64_t>::min();
    std::vector<AccessAwareGeneratedState> states;
};

int64_t partition_overlap_count(const std::vector<int> &lhs, const std::vector<int> &rhs) {
    int64_t overlap = 0;
    for (auto left_part : lhs) {
        for (auto right_part : rhs) {
            if (left_part == right_part) {
                overlap++;
            }
        }
    }
    return overlap;
}

std::vector<int> sorted_partitions_from_mask(uint64_t mask, int num_partitions) {
    std::vector<int> partitions;
    for (int part = 0; part < num_partitions; part++) {
        if (((mask >> part) & 1ULL) != 0ULL) {
            partitions.emplace_back(part);
        }
    }
    return partitions;
}

AccessAwareGeneratedState build_generated_state(uint64_t mask, const std::vector<bool> &uncovered, int num_partitions) {
    AccessAwareGeneratedState state;
    state.partitions = sorted_partitions_from_mask(mask, num_partitions);
    for (auto src_part : state.partitions) {
        for (auto dst_part : state.partitions) {
            int bucket_idx = src_part * num_partitions + dst_part;
            if (uncovered[bucket_idx]) {
                state.newly_covered_buckets.emplace_back(src_part, dst_part);
            }
        }
    }
    return state;
}

int64_t state_new_bucket_gain(uint64_t mask, const std::vector<bool> &uncovered, int num_partitions) {
    int64_t gain = 0;
    for (int src_part = 0; src_part < num_partitions; src_part++) {
        if (((mask >> src_part) & 1ULL) == 0ULL) {
            continue;
        }
        for (int dst_part = 0; dst_part < num_partitions; dst_part++) {
            if (((mask >> dst_part) & 1ULL) == 0ULL) {
                continue;
            }
            gain += uncovered[src_part * num_partitions + dst_part] ? 1 : 0;
        }
    }
    return gain;
}

int64_t group_new_bucket_gain(const std::vector<uint64_t> &group_masks, const std::vector<bool> &uncovered, int num_partitions) {
    int64_t gain = 0;
    for (auto mask : group_masks) {
        gain += state_new_bucket_gain(mask, uncovered, num_partitions);
    }
    return gain;
}

void search_access_aware_groups(uint64_t available_mask, int num_partitions, int buffer_capacity,
                                const std::vector<bool> &uncovered, const std::vector<std::vector<int>> &prev_group,
                                std::vector<uint64_t> &current_group, AccessAwareGroupSearch &best) {
    int remaining_partitions = static_cast<int>(__builtin_popcountll(available_mask));
    if (remaining_partitions == 0) {
        AccessAwareGroupSearch candidate;
        std::vector<std::vector<int>> current_partitions;
        current_partitions.reserve(current_group.size());
        for (auto mask : current_group) {
            current_partitions.emplace_back(sorted_partitions_from_mask(mask, num_partitions));
        }

        int64_t coverage_gain = group_new_bucket_gain(current_group, uncovered, num_partitions);
        int64_t overlap_gain = 0;
        if (!prev_group.empty() && prev_group.size() == current_partitions.size()) {
            for (std::size_t lane = 0; lane < current_partitions.size(); lane++) {
                overlap_gain += partition_overlap_count(prev_group[lane], current_partitions[lane]);
            }
        }

        // Coverage dominates; overlap breaks ties toward better locality.
        candidate.score = coverage_gain * 100 + overlap_gain * 7;
        for (auto mask : current_group) {
            candidate.states.emplace_back(build_generated_state(mask, uncovered, num_partitions));
        }
        if (candidate.score > best.score) {
            best = std::move(candidate);
        }
        return;
    }

    if (remaining_partitions < buffer_capacity) {
        return;
    }

    int anchor = 0;
    while (((available_mask >> anchor) & 1ULL) == 0ULL) {
        anchor++;
    }

    std::vector<int> rest;
    rest.reserve(remaining_partitions - 1);
    for (int part = anchor + 1; part < num_partitions; part++) {
        if (((available_mask >> part) & 1ULL) != 0ULL) {
            rest.emplace_back(part);
        }
    }

    for (std::size_t i = 0; i < rest.size(); i++) {
        for (std::size_t j = i + 1; j < rest.size(); j++) {
            for (std::size_t k = j + 1; k < rest.size(); k++) {
                uint64_t state_mask = (1ULL << anchor) | (1ULL << rest[i]) | (1ULL << rest[j]) | (1ULL << rest[k]);
                current_group.emplace_back(state_mask);
                search_access_aware_groups(available_mask & ~state_mask, num_partitions, buffer_capacity, uncovered, prev_group, current_group, best);
                current_group.pop_back();
            }
        }
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> generate_access_aware_states(int num_partitions, int buffer_capacity, int active_devices) {
    if (buffer_capacity != 4 || active_devices <= 1 || active_devices * buffer_capacity != num_partitions || num_partitions > 63) {
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    const int total_buckets = num_partitions * num_partitions;
    std::vector<bool> uncovered(total_buckets, true);
    int uncovered_count = total_buckets;

    std::vector<std::vector<int>> prev_group;
    std::vector<std::vector<int>> buffer_states;
    std::vector<std::vector<std::pair<int, int>>> edge_buckets_per_buffer;

    int superstep = 0;
    const int max_supersteps = total_buckets;
    while (uncovered_count > 0 && superstep < max_supersteps) {
        AccessAwareGroupSearch best_group;
        std::vector<uint64_t> current_group;
        current_group.reserve(active_devices);
        const uint64_t all_partitions_mask = (1ULL << num_partitions) - 1ULL;
        search_access_aware_groups(all_partitions_mask, num_partitions, buffer_capacity, uncovered, prev_group, current_group, best_group);

        if (best_group.states.empty()) {
            break;
        }

        int64_t covered_this_step = 0;
        std::vector<std::vector<int>> current_partitions;
        current_partitions.reserve(best_group.states.size());
        for (auto &state : best_group.states) {
            current_partitions.emplace_back(state.partitions.begin(), state.partitions.end());
            buffer_states.emplace_back(state.partitions.begin(), state.partitions.end());
            edge_buckets_per_buffer.emplace_back();
            auto &assigned = edge_buckets_per_buffer.back();
            assigned.reserve(state.newly_covered_buckets.size());
            for (auto &bucket : state.newly_covered_buckets) {
                int bucket_idx = bucket.first * num_partitions + bucket.second;
                if (!uncovered[bucket_idx]) {
                    continue;
                }
                uncovered[bucket_idx] = false;
                uncovered_count--;
                covered_this_step++;
                assigned.emplace_back(bucket);
            }
        }

        if (covered_this_step == 0) {
            break;
        }

        prev_group = std::move(current_partitions);
        superstep++;
    }

    if (uncovered_count > 0 || buffer_states.empty()) {
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

}  // namespace

std::vector<int64_t> getSingleGpuGpuAwareCustomPermutation(const vector<torch::Tensor> &buffer_states,
                                                           const vector<int64_t> &edge_bucket_sizes,
                                                           int num_partitions) {
    std::vector<int64_t> identity(buffer_states.size());
    std::iota(identity.begin(), identity.end(), 0);
    if (buffer_states.size() <= 1 || edge_bucket_sizes.size() != static_cast<size_t>(num_partitions * num_partitions)) {
        return identity;
    }

    auto hotness = build_partition_hotness(edge_bucket_sizes, num_partitions);
    std::vector<std::vector<int64_t>> state_partitions;
    state_partitions.reserve(buffer_states.size());
    std::vector<int64_t> resident_weights;
    resident_weights.reserve(buffer_states.size());
    for (auto &state : buffer_states) {
        auto partitions = tensor_to_partitions(state);
        std::sort(partitions.begin(), partitions.end());
        resident_weights.emplace_back(compute_state_resident_weight(std::vector<int>(partitions.begin(), partitions.end()), edge_bucket_sizes, num_partitions));
        state_partitions.emplace_back(std::move(partitions));
    }

    auto transition_score = [&](int64_t prev_idx, int64_t next_idx) {
        int64_t overlap_count = 0;
        int64_t shared_hotness = 0;
        std::size_t i = 0;
        std::size_t j = 0;
        const auto &lhs = state_partitions[prev_idx];
        const auto &rhs = state_partitions[next_idx];
        while (i < lhs.size() && j < rhs.size()) {
            if (lhs[i] == rhs[j]) {
                overlap_count++;
                shared_hotness += hotness[lhs[i]];
                i++;
                j++;
            } else if (lhs[i] < rhs[j]) {
                i++;
            } else {
                j++;
            }
        }
        return std::make_pair(overlap_count, shared_hotness);
    };

    std::vector<int64_t> best_order;
    std::tuple<int64_t, int64_t, int64_t, int64_t> best_key{-1, -1, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};

    for (int64_t start_idx = 0; start_idx < static_cast<int64_t>(buffer_states.size()); start_idx++) {
        std::vector<int64_t> order;
        order.reserve(buffer_states.size());
        std::vector<bool> used(buffer_states.size(), false);
        order.emplace_back(start_idx);
        used[start_idx] = true;

        int64_t overlap_transitions = 0;
        int64_t total_shared_hotness = 0;

        while (order.size() < buffer_states.size()) {
            int64_t prev_idx = order.back();
            int64_t best_next = -1;
            std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> best_next_key{
                std::numeric_limits<int64_t>::min(),
                std::numeric_limits<int64_t>::min(),
                std::numeric_limits<int64_t>::min(),
                std::numeric_limits<int64_t>::min(),
                std::numeric_limits<int64_t>::min()};

            for (int64_t candidate_idx = 0; candidate_idx < static_cast<int64_t>(buffer_states.size()); candidate_idx++) {
                if (used[candidate_idx]) {
                    continue;
                }

                auto [overlap_count, shared_hotness] = transition_score(prev_idx, candidate_idx);
                int64_t future_overlap_choices = 0;
                for (int64_t other_idx = 0; other_idx < static_cast<int64_t>(buffer_states.size()); other_idx++) {
                    if (used[other_idx] || other_idx == candidate_idx) {
                        continue;
                    }
                    if (transition_score(candidate_idx, other_idx).first > 0) {
                        future_overlap_choices++;
                    }
                }

                auto candidate_key = std::make_tuple(overlap_count,
                                                     shared_hotness,
                                                     future_overlap_choices,
                                                     resident_weights[candidate_idx],
                                                     -candidate_idx);
                if (candidate_key > best_next_key) {
                    best_next_key = candidate_key;
                    best_next = candidate_idx;
                }
            }

            if (best_next == -1) {
                break;
            }

            auto [chosen_overlap, chosen_hotness] = transition_score(prev_idx, best_next);
            overlap_transitions += chosen_overlap > 0 ? 1 : 0;
            total_shared_hotness += chosen_hotness;
            used[best_next] = true;
            order.emplace_back(best_next);
        }

        if (order.size() != buffer_states.size()) {
            continue;
        }

        auto candidate_key =
            std::make_tuple(overlap_transitions, total_shared_hotness, -resident_weights[start_idx], -start_idx);
        if (candidate_key > best_key) {
            best_key = candidate_key;
            best_order = std::move(order);
        }
    }

    if (best_order.empty()) {
        return identity;
    }

    return best_order;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getEdgeBucketOrdering(EdgeBucketOrdering edge_bucket_ordering, int num_partitions, int buffer_capacity,
                                                                               int fine_to_coarse_ratio, int num_cache_partitions,
                                                                               bool randomly_assign_edge_buckets) {
    switch (edge_bucket_ordering) {
        case EdgeBucketOrdering::OLD_BETA:
            SPDLOG_INFO("Generating Old Beta Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, 1, 0, false);
        case EdgeBucketOrdering::NEW_BETA:
            SPDLOG_INFO("Generating New Beta Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, 1, 0, true);
        case EdgeBucketOrdering::ALL_BETA:
            return getCustomEdgeBucketOrdering();
        case EdgeBucketOrdering::COMET:
            SPDLOG_INFO("Generating COMET Ordering");
            return getTwoLevelBetaOrdering(num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions, randomly_assign_edge_buckets);
        case EdgeBucketOrdering::CUSTOM:
            SPDLOG_INFO("Generating CUSTOM Ordering");
            if (hybrid_cover_ordering_enabled()) {
                return getHybridCoverEdgeBucketOrdering(num_partitions, buffer_capacity);
            }
            if (contrastive_greedy_cover_ordering_enabled()) {
                return getGreedyCoverEdgeBucketOrdering(num_partitions, buffer_capacity);
            }
            return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, randomly_assign_edge_buckets);
        default:
            SPDLOG_ERROR("Not implemented");
            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
            return ret;
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getNodePartitionOrdering(NodePartitionOrdering node_partition_ordering, Indices train_nodes,
                                                                                  int64_t total_num_nodes, int num_partitions, int buffer_capacity,
                                                                                  int fine_to_coarse_ratio, int num_cache_partitions) {
    switch (node_partition_ordering) {
        case NodePartitionOrdering::DISPERSED:
            SPDLOG_INFO("Generating Dispersed Ordering");
            return getDispersedNodePartitionOrdering(train_nodes, total_num_nodes, num_partitions, buffer_capacity, fine_to_coarse_ratio, num_cache_partitions);
        case NodePartitionOrdering::SEQUENTIAL:
            SPDLOG_INFO("Generating Sequential Ordering");
            return getSequentialNodePartitionOrdering(train_nodes, total_num_nodes, num_partitions, buffer_capacity);
        case NodePartitionOrdering::CUSTOM:
            return getCustomNodePartitionOrdering();
        default:
            SPDLOG_ERROR("Not implemented");
            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
            return ret;
    }
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> convertEdgeBucketOrderToTensors(vector<vector<int>> buffer_states,
                                                                                         vector<vector<std::pair<int, int>>> edge_buckets_per_buffer) {
    vector<torch::Tensor> ret_buffer_states;
    vector<torch::Tensor> ret_edge_buckets_per_buffer;

    for (auto b : buffer_states) {
        ret_buffer_states.emplace_back(torch::tensor(b, torch::kInt64));
    }

    for (auto edge_buckets : edge_buckets_per_buffer) {
        torch::Tensor tmp = torch::zeros({(int64_t)edge_buckets.size(), 2}, torch::kInt64);

        for (int i = 0; i < edge_buckets.size(); i++) {
            tmp[i][0] = std::get<0>(edge_buckets[i]);
            tmp[i][1] = std::get<1>(edge_buckets[i]);
        }

        ret_edge_buckets_per_buffer.emplace_back(tmp);
    }

    return std::forward_as_tuple(ret_buffer_states, ret_edge_buckets_per_buffer);
}

vector<vector<int>> getBetaOrderingHelper(int num_partitions, int buffer_capacity) {
    vector<vector<int>> buffer_states;
    Indices all_partitions = torch::randperm(num_partitions, torch::kInt32);

    // get all buffer states
    Indices in_buffer = all_partitions.index_select(0, torch::arange(buffer_capacity));

    Indices combined = torch::cat({all_partitions, in_buffer});
    auto uniques = unique_with_counts_sorted(combined);
    auto vals = std::get<0>(uniques);
    auto counts = std::get<1>(uniques);
    Indices on_disk = vals.masked_select(counts == 1);

    int *data_ptr_ = (int *)in_buffer.data_ptr();
    buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));

    while (on_disk.size(0) >= 1) {
        in_buffer = in_buffer.index_select(0, torch::randperm(in_buffer.size(0), torch::kInt64));
        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        for (int i = 0; i < on_disk.size(0); i++) {
            auto admit_id = on_disk[i].clone();

            on_disk[i] = in_buffer[-1];

            in_buffer[-1] = admit_id;

            data_ptr_ = (int *)in_buffer.data_ptr();
            buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));
        }

        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        int num_replaced = 0;
        for (int i = 0; i < buffer_capacity - 1; i++) {
            if (i >= on_disk.size(0)) {
                break;
            }
            num_replaced++;
            in_buffer[i] = on_disk[i];

            data_ptr_ = (int *)in_buffer.data_ptr();
            buffer_states.emplace_back(vector<int>(data_ptr_, data_ptr_ + in_buffer.size(0)));
        }
        on_disk = on_disk.narrow(0, num_replaced, on_disk.size(0) - num_replaced);
    }

    return buffer_states;
}

vector<vector<std::pair<int, int>>> greedyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions) {
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(buffer_states.size());
    torch::Tensor interacted = torch::zeros({num_partitions, num_partitions}, torch::kInt32);
    auto interacted_accessor = interacted.accessor<int32_t, 2>();

    for (int i = 0; i < buffer_states.size(); i++) {
        for (int j = 0; j < buffer_states[i].size(); j++) {
            for (int k = 0; k < buffer_states[i].size(); k++) {
                int32_t src_part = buffer_states[i][j];
                int32_t dst_part = buffer_states[i][k];
                if (interacted_accessor[src_part][dst_part] == 1) {
                    continue;
                }
                interacted_accessor[src_part][dst_part] = 1;
                edge_buckets_per_buffer[i].emplace_back(std::make_pair(src_part, dst_part));
            }
        }
    }

    return edge_buckets_per_buffer;
}

namespace {

void build_greedy_cover_candidates(int num_partitions,
                                   int buffer_capacity,
                                   int next_partition,
                                   std::vector<int> &current,
                                   std::vector<std::vector<int>> &candidates) {
    if (static_cast<int>(current.size()) == buffer_capacity) {
        candidates.emplace_back(current);
        return;
    }

    int remaining_slots = buffer_capacity - static_cast<int>(current.size());
    for (int partition = next_partition; partition <= num_partitions - remaining_slots; partition++) {
        current.emplace_back(partition);
        build_greedy_cover_candidates(num_partitions, buffer_capacity, partition + 1, current, candidates);
        current.pop_back();
    }
}

int64_t count_uncovered_buckets_for_state(const std::vector<int> &state,
                                          const std::vector<uint8_t> &covered,
                                          int num_partitions) {
    int64_t uncovered = 0;
    for (auto src_part : state) {
        for (auto dst_part : state) {
            if (covered[src_part * num_partitions + dst_part] == 0) {
                uncovered++;
            }
        }
    }
    return uncovered;
}

int state_overlap_count(const std::vector<int> &lhs, const std::vector<int> &rhs) {
    if (lhs.empty() || rhs.empty()) {
        return 0;
    }

    // Some planner paths preserve slot order from tensors rather than sorting resident partitions.
    // Overlap accounting is set-based, so normalize locally instead of assuming sorted inputs.
    std::vector<int> lhs_sorted(lhs.begin(), lhs.end());
    std::vector<int> rhs_sorted(rhs.begin(), rhs.end());
    std::sort(lhs_sorted.begin(), lhs_sorted.end());
    std::sort(rhs_sorted.begin(), rhs_sorted.end());

    int overlap = 0;
    std::size_t lhs_idx = 0;
    std::size_t rhs_idx = 0;
    while (lhs_idx < lhs_sorted.size() && rhs_idx < rhs_sorted.size()) {
        if (lhs_sorted[lhs_idx] == rhs_sorted[rhs_idx]) {
            overlap++;
            lhs_idx++;
            rhs_idx++;
        } else if (lhs_sorted[lhs_idx] < rhs_sorted[rhs_idx]) {
            lhs_idx++;
        } else {
            rhs_idx++;
        }
    }
    return overlap;
}

void append_directed_anchor_pairs(const std::vector<int> &anchors, std::vector<std::pair<int, int>> &buckets) {
    for (auto src_part : anchors) {
        for (auto dst_part : anchors) {
            buckets.emplace_back(src_part, dst_part);
        }
    }
}

void append_directed_anchor_stream_pairs(const std::vector<int> &anchors,
                                         int stream_partner,
                                         std::vector<std::pair<int, int>> &buckets) {
    for (auto anchor_part : anchors) {
        buckets.emplace_back(anchor_part, stream_partner);
        buckets.emplace_back(stream_partner, anchor_part);
    }
}

std::vector<int> hybrid_cover_partner_order(const std::vector<int> &partners, HybridCoverVariant variant) {
    if (partners.size() <= 1) {
        return partners;
    }

    std::vector<int> ordered;
    ordered.reserve(partners.size());
    switch (variant) {
        case HybridCoverVariant::LEGACY_ROTATED:
            for (std::size_t idx = 1; idx < partners.size(); idx++) {
                ordered.emplace_back(partners[idx]);
            }
            ordered.emplace_back(partners[0]);
            break;
        case HybridCoverVariant::NATURAL:
            ordered = partners;
            break;
        case HybridCoverVariant::REVERSED:
            ordered.assign(partners.rbegin(), partners.rend());
            break;
    }
    return ordered;
}

void append_hybrid_cover_microstate(LanePlan &lane,
                                    int64_t superstate_id,
                                    std::vector<int> state,
                                    std::vector<std::pair<int, int>> buckets) {
    MicrostatePlan microstate;
    microstate.microstate_id = static_cast<int64_t>(lane.microstates.size());
    microstate.lane_id = lane.lane_id;
    microstate.superstate_id = superstate_id;
    microstate.resident_partitions = std::move(state);
    microstate.edge_buckets = std::move(buckets);
    lane.microstates.emplace_back(std::move(microstate));
}

void build_hybrid_cover_plan_recursive(const std::vector<int> &remaining_partitions,
                                       LanePlan &lane,
                                       int64_t &superstate_count,
                                       HybridCoverVariant variant) {
    if (remaining_partitions.empty()) {
        return;
    }

    if (remaining_partitions.size() < 4) {
        throw std::runtime_error("Hybrid-Cover recursion expected at least four partitions");
    }

    if (remaining_partitions.size() == 4) {
        std::vector<int> state = remaining_partitions;
        std::vector<std::pair<int, int>> buckets;
        buckets.reserve(16);
        for (auto src_part : state) {
            for (auto dst_part : state) {
                buckets.emplace_back(src_part, dst_part);
            }
        }
        append_hybrid_cover_microstate(lane, superstate_count, std::move(state), std::move(buckets));
        superstate_count++;
        return;
    }

    std::vector<int> anchors(remaining_partitions.begin(), remaining_partitions.begin() + 3);
    std::vector<int> partners(remaining_partitions.begin() + 3, remaining_partitions.end());
    std::vector<int> partner_order = hybrid_cover_partner_order(partners, variant);

    for (std::size_t idx = 0; idx < partner_order.size(); idx++) {
        int stream_partner = partner_order[idx];
        std::vector<int> state = anchors;
        state.emplace_back(stream_partner);

        std::vector<std::pair<int, int>> buckets;
        buckets.reserve(idx == 0 ? 15 : 6);
        if (idx == 0) {
            append_directed_anchor_pairs(anchors, buckets);
        }
        append_directed_anchor_stream_pairs(anchors, stream_partner, buckets);

        append_hybrid_cover_microstate(lane, superstate_count, std::move(state), std::move(buckets));
    }

    superstate_count++;
    build_hybrid_cover_plan_recursive(partners, lane, superstate_count, variant);
}

PlanVariant hybrid_cover_variant_plan_variant(HybridCoverVariant variant) {
    switch (variant) {
        case HybridCoverVariant::LEGACY_ROTATED:
            return PlanVariant::HYBRID_COVER_LEGACY_ROTATED;
        case HybridCoverVariant::NATURAL:
            return PlanVariant::HYBRID_COVER_NATURAL;
        case HybridCoverVariant::REVERSED:
            return PlanVariant::HYBRID_COVER_REVERSED;
    }
    return PlanVariant::DEFAULT;
}

int64_t partition_rows_for(const std::vector<int64_t> &partition_row_counts, int partition_id) {
    if (partition_id < 0 || static_cast<std::size_t>(partition_id) >= partition_row_counts.size()) {
        return 0;
    }
    return partition_row_counts[partition_id];
}

void populate_fragment_exact_semantics(StateflowPlan &plan) {
    std::unordered_map<int64_t, int64_t> bucket_ownership_counts;
    for (const auto &lane : plan.lanes) {
        for (const auto &microstate : lane.microstates) {
            for (const auto &fragment : microstate.active_fragments) {
                for (const auto &[src_part, dst_part] : fragment.edge_buckets) {
                    int64_t key = static_cast<int64_t>(src_part) * plan.num_partitions + dst_part;
                    bucket_ownership_counts[key]++;
                }
            }
        }
    }

    for (auto &lane : plan.lanes) {
        for (auto &microstate : lane.microstates) {
            for (auto &fragment : microstate.active_fragments) {
                bool exact = !fragment.edge_buckets.empty();
                for (const auto &[src_part, dst_part] : fragment.edge_buckets) {
                    int64_t key = static_cast<int64_t>(src_part) * plan.num_partitions + dst_part;
                    auto count_it = bucket_ownership_counts.find(key);
                    if (count_it == bucket_ownership_counts.end() || count_it->second != 1) {
                        exact = false;
                        break;
                    }
                }
                fragment.exact_semantics_tag = exact;
            }
        }
    }
}

void populate_fragment_estimates(StateflowPlan &plan, const std::vector<int64_t> &edge_bucket_sizes) {
    const bool have_sizes =
        edge_bucket_sizes.size() == static_cast<std::size_t>(plan.num_partitions * plan.num_partitions);
    for (auto &lane : plan.lanes) {
        for (auto &microstate : lane.microstates) {
            for (auto &fragment : microstate.active_fragments) {
                int64_t fragment_edges = 0;
                for (const auto &[src_part, dst_part] : fragment.edge_buckets) {
                    if (have_sizes) {
                        fragment_edges += edge_bucket_sizes[src_part * plan.num_partitions + dst_part];
                    } else {
                        fragment_edges++;
                    }
                }
                fragment.estimated_edges = fragment_edges;
            }
        }
    }
}

void lift_stateflow_plan_ir(StateflowPlan &plan, const std::vector<int64_t> &partition_row_counts) {
    plan.total_handoffs = 0;
    plan.total_admitted_objects = 0;
    for (int i = 0; i < 4; i++) {
        plan.total_admissions_by_role[i] = 0;
    }
    int64_t next_object_id = 0;
    int64_t next_fragment_id = 0;
    int64_t next_handoff_id = 0;

    for (auto &lane : plan.lanes) {
        lane.handoffs.clear();
        std::unordered_map<int, int64_t> current_object_id;

        const std::size_t ms_count = lane.microstates.size();
        for (std::size_t idx = 0; idx < ms_count; idx++) {
            auto &microstate = lane.microstates[idx];
            microstate.lane_id = lane.lane_id;
            bool same_superstate_as_prev = idx > 0 && lane.microstates[idx - 1].superstate_id == microstate.superstate_id;

            std::unordered_set<int> prev_set;
            if (idx > 0) {
                for (auto p : lane.microstates[idx - 1].resident_partitions) {
                    prev_set.insert(p);
                }
            }
            std::unordered_set<int> next_set;
            if (idx + 1 < ms_count) {
                for (auto p : lane.microstates[idx + 1].resident_partitions) {
                    next_set.insert(p);
                }
            }

            microstate.resident_objects.clear();
            microstate.resident_objects.reserve(microstate.resident_partitions.size());
            microstate.admitted_object_ids.clear();
            microstate.evicted_object_ids.clear();
            std::unordered_map<int, int64_t> microstate_partition_to_object_id;

            for (int slot_id = 0; slot_id < static_cast<int>(microstate.resident_partitions.size()); slot_id++) {
                int part_id = microstate.resident_partitions[slot_id];
                bool carried_in = prev_set.count(part_id) > 0;
                bool carried_out = next_set.count(part_id) > 0;
                bool same_slot_carried =
                    idx > 0 && slot_id < static_cast<int>(lane.microstates[idx - 1].resident_partitions.size()) &&
                    lane.microstates[idx - 1].resident_partitions[slot_id] == part_id;

                int64_t obj_id;
                bool newly_admitted = false;
                auto it = current_object_id.find(part_id);
                if (carried_in && it != current_object_id.end()) {
                    obj_id = it->second;
                } else {
                    obj_id = next_object_id++;
                    current_object_id[part_id] = obj_id;
                    microstate.admitted_object_ids.emplace_back(obj_id);
                    plan.total_admitted_objects++;
                    newly_admitted = true;
                }
                microstate_partition_to_object_id[part_id] = obj_id;

                ResidentObjectRole role;
                if (carried_in && !same_slot_carried) {
                    role = ResidentObjectRole::SURVIVOR;
                } else if (!carried_in && !carried_out) {
                    role = ResidentObjectRole::STREAM;
                } else if (same_superstate_as_prev && same_slot_carried) {
                    role = ResidentObjectRole::ANCHOR;
                } else if (carried_in) {
                    role = ResidentObjectRole::SURVIVOR;
                } else {
                    role = ResidentObjectRole::INCOMING;
                }

                if (newly_admitted) {
                    plan.total_admissions_by_role[static_cast<int>(role)]++;
                }

                ResidentObjectPlan obj;
                obj.object_id = obj_id;
                obj.partition_id = part_id;
                obj.slot_id = slot_id;
                obj.role = role;
                obj.rows = partition_rows_for(partition_row_counts, part_id);
                microstate.resident_objects.emplace_back(obj);
            }

            microstate.active_fragments.clear();
            microstate.active_fragments.reserve(microstate.edge_buckets.size());
            std::unordered_set<int> non_stream_partitions;
            for (const auto &resident_object : microstate.resident_objects) {
                if (resident_object.role != ResidentObjectRole::STREAM) {
                    non_stream_partitions.insert(resident_object.partition_id);
                }
            }
            for (std::size_t fidx = 0; fidx < microstate.edge_buckets.size(); fidx++) {
                const auto &[src_part, dst_part] = microstate.edge_buckets[fidx];
                bool src_prev_resident = prev_set.count(src_part) > 0;
                bool dst_prev_resident = prev_set.count(dst_part) > 0;
                bool src_anchor_resident = non_stream_partitions.count(src_part) > 0;
                bool dst_anchor_resident = non_stream_partitions.count(dst_part) > 0;

                FragmentKind kind;
                if (src_prev_resident && dst_prev_resident) {
                    kind = FragmentKind::FULLY_RESIDENT;
                } else if (src_anchor_resident && dst_anchor_resident) {
                    kind = FragmentKind::ANCHOR_ANCHOR;
                } else {
                    kind = FragmentKind::ANCHOR_STREAM;
                }

                FragmentPlan frag;
                frag.fragment_id = next_fragment_id++;
                frag.edge_buckets.emplace_back(src_part, dst_part);
                frag.fragment_kind = kind;
                frag.estimated_edges = 0;
                auto src_obj_it = microstate_partition_to_object_id.find(src_part);
                auto dst_obj_it = microstate_partition_to_object_id.find(dst_part);
                if (src_obj_it != microstate_partition_to_object_id.end()) {
                    frag.required_object_ids.emplace_back(src_obj_it->second);
                }
                if (dst_obj_it != microstate_partition_to_object_id.end() && dst_part != src_part) {
                    frag.required_object_ids.emplace_back(dst_obj_it->second);
                }
                frag.exact_semantics_tag = false;
                microstate.active_fragments.emplace_back(std::move(frag));
            }

            std::unordered_set<int64_t> fragment_bucket_union;
            fragment_bucket_union.reserve(microstate.active_fragments.size());
            for (auto &frag : microstate.active_fragments) {
                for (const auto &[src_part, dst_part] : frag.edge_buckets) {
                    int64_t key = static_cast<int64_t>(src_part) * plan.num_partitions + dst_part;
                    fragment_bucket_union.insert(key);
                }
            }
            std::unordered_set<int64_t> legacy_bucket_set;
            legacy_bucket_set.reserve(microstate.edge_buckets.size());
            for (const auto &[src_part, dst_part] : microstate.edge_buckets) {
                int64_t key = static_cast<int64_t>(src_part) * plan.num_partitions + dst_part;
                legacy_bucket_set.insert(key);
            }
            if (fragment_bucket_union != legacy_bucket_set) {
                throw std::runtime_error("Stateflow fragment union does not match legacy bucket set");
            }

            std::unordered_set<int> curr_set(microstate.resident_partitions.begin(), microstate.resident_partitions.end());
            if (idx + 1 < ms_count) {
                for (int part_id : microstate.resident_partitions) {
                    if (next_set.count(part_id) == 0) {
                        auto obj_it = microstate_partition_to_object_id.find(part_id);
                        if (obj_it != microstate_partition_to_object_id.end()) {
                            microstate.evicted_object_ids.emplace_back(obj_it->second);
                        }
                    }
                }
            }
            for (auto it2 = current_object_id.begin(); it2 != current_object_id.end(); ) {
                if (curr_set.count(it2->first) == 0) {
                    it2 = current_object_id.erase(it2);
                } else {
                    ++it2;
                }
            }

            if (idx > 0) {
                const auto &prev_ms = lane.microstates[idx - 1];
                HandoffPlan handoff;
                handoff.handoff_id = next_handoff_id++;
                handoff.src_microstate_id = prev_ms.microstate_id;
                handoff.dst_microstate_id = microstate.microstate_id;
                handoff.src_lane_id = lane.lane_id;
                handoff.dst_lane_id = lane.lane_id;
                std::unordered_map<int, int64_t> prev_partition_to_object_id;
                for (const auto &obj : prev_ms.resident_objects) {
                    prev_partition_to_object_id[obj.partition_id] = obj.object_id;
                }

                for (int slot = 0; slot < static_cast<int>(prev_ms.resident_partitions.size()); slot++) {
                    int part = prev_ms.resident_partitions[slot];
                    auto prev_obj_it = prev_partition_to_object_id.find(part);
                    if (prev_obj_it == prev_partition_to_object_id.end()) {
                        continue;
                    }
                    if (curr_set.count(part) > 0) {
                        handoff.kept_object_ids.emplace_back(prev_obj_it->second);
                        int dst_slot = -1;
                        for (int s2 = 0; s2 < static_cast<int>(microstate.resident_partitions.size()); s2++) {
                            if (microstate.resident_partitions[s2] == part) {
                                dst_slot = s2;
                                break;
                            }
                        }
                        handoff.slot_mapping.emplace_back(slot, dst_slot);
                    } else {
                        handoff.evicted_object_ids.emplace_back(prev_obj_it->second);
                    }
                }
                for (int64_t object_id : microstate.admitted_object_ids) {
                    handoff.admitted_object_ids.emplace_back(object_id);
                }

                const int64_t admitted_count = static_cast<int64_t>(handoff.admitted_object_ids.size());
                const int64_t kept_count = static_cast<int64_t>(handoff.kept_object_ids.size());
                if (admitted_count == 0) {
                    handoff.mode = HandoffMode::DELAYED_KEEP_ALIVE;
                } else if (kept_count > 0) {
                    handoff.mode = HandoffMode::ROTATING_OVERWRITE;
                } else {
                    handoff.mode = HandoffMode::FULL_RELOAD;
                }
                handoff.estimated_cost = admitted_count;

                lane.handoffs.emplace_back(std::move(handoff));
                plan.total_handoffs++;
            }
        }
    }
}

int64_t estimate_plan_bucket_edges(StateflowPlan &plan, const std::vector<int64_t> &edge_bucket_sizes);

const ResidentObjectPlan *find_resident_object(const MicrostatePlan &microstate, int partition_id) {
    for (const auto &obj : microstate.resident_objects) {
        if (obj.partition_id == partition_id) {
            return &obj;
        }
    }
    return nullptr;
}

const MicrostatePlan *find_microstate_for_superstate(const LanePlan &lane, int64_t superstate_id) {
    for (const auto &microstate : lane.microstates) {
        if (microstate.superstate_id == superstate_id) {
            return &microstate;
        }
    }
    return nullptr;
}

void populate_cross_lane_handoffs(StateflowPlan &plan,
                                  const std::vector<int64_t> &partition_row_counts,
                                  const PlanEmbeddingLayout &layout) {
    plan.cross_lane_handoffs.clear();
    plan.total_cross_lane_handoffs = 0;
    plan.total_peer_handoff_bytes = 0;
    const int64_t bytes_per_row = plan_embedding_bytes_per_row(layout);
    std::unordered_set<int64_t> emitted_peer_targets;

    int64_t next_handoff_id = plan.total_handoffs;
    for (auto &lane : plan.lanes) {
        for (std::size_t idx = 1; idx < lane.microstates.size(); idx++) {
            const auto &prev_ms = lane.microstates[idx - 1];
            const auto &microstate = lane.microstates[idx];
            std::unordered_set<int> prev_resident(prev_ms.resident_partitions.begin(), prev_ms.resident_partitions.end());

            for (int partition_id : microstate.resident_partitions) {
                if (prev_resident.count(partition_id) > 0) {
                    continue;
                }

                const LanePlan *src_lane = nullptr;
                const MicrostatePlan *src_microstate = nullptr;
                for (const auto &other_lane : plan.lanes) {
                    if (other_lane.lane_id == lane.lane_id) {
                        continue;
                    }
                    const auto *candidate_src = find_microstate_for_superstate(other_lane, microstate.superstate_id - 1);
                    if (candidate_src == nullptr) {
                        continue;
                    }
                    if (std::find(candidate_src->resident_partitions.begin(), candidate_src->resident_partitions.end(), partition_id) ==
                        candidate_src->resident_partitions.end()) {
                        continue;
                    }
                    // All candidate sources relay the same partition payload, so
                    // source selection is a deterministic lane-id tiebreak.
                    bool better_source = src_lane == nullptr || other_lane.lane_id < src_lane->lane_id;
                    if (better_source) {
                        src_lane = &other_lane;
                        src_microstate = candidate_src;
                    }
                }

                if (src_lane == nullptr || src_microstate == nullptr) {
                    continue;
                }

                const ResidentObjectPlan *src_obj = find_resident_object(*src_microstate, partition_id);
                const ResidentObjectPlan *dst_obj = find_resident_object(microstate, partition_id);
                if (src_obj == nullptr || dst_obj == nullptr) {
                    continue;
                }

                int64_t target_key = (static_cast<int64_t>(lane.lane_id) << 40) |
                                     (static_cast<int64_t>(microstate.microstate_id) << 20) |
                                     static_cast<int64_t>(partition_id);
                if (!emitted_peer_targets.insert(target_key).second) {
                    continue;
                }

                HandoffPlan handoff;
                handoff.handoff_id = next_handoff_id++;
                handoff.src_microstate_id = src_microstate->microstate_id;
                handoff.dst_microstate_id = microstate.microstate_id;
                handoff.src_lane_id = src_lane->lane_id;
                handoff.dst_lane_id = lane.lane_id;
                handoff.admitted_object_ids = {dst_obj->object_id};
                handoff.slot_mapping = {{src_obj->slot_id, dst_obj->slot_id}};
                handoff.mode = HandoffMode::PEER_RELAY;
                handoff.peer_bytes = partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
                handoff.estimated_cost = handoff.peer_bytes;

                plan.cross_lane_handoffs.emplace_back(std::move(handoff));
                plan.total_cross_lane_handoffs++;
                plan.total_peer_handoff_bytes += partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
            }
        }
    }
    plan.total_handoffs += plan.total_cross_lane_handoffs;
}

void finalize_stateflow_plan(StateflowPlan &plan,
                             const std::vector<int64_t> &edge_bucket_sizes = {},
                             const std::vector<int64_t> &partition_row_counts = {},
                             const PlanEmbeddingLayout &layout = {}) {
    plan.total_microstates = 0;
    plan.total_bucket_assignments = 0;
    plan.total_partition_loads = 0;
    plan.max_overlap = 0;
    plan.boundary_count = 0;

    for (auto &lane : plan.lanes) {
        for (std::size_t idx = 0; idx < lane.microstates.size(); idx++) {
            auto &microstate = lane.microstates[idx];
            plan.total_microstates++;
            plan.total_bucket_assignments += static_cast<int64_t>(microstate.edge_buckets.size());
            if (idx == 0) {
                plan.total_partition_loads += static_cast<int64_t>(microstate.resident_partitions.size());
                microstate.overlap_with_prev = 0;
                microstate.admitted_partitions = static_cast<int64_t>(microstate.resident_partitions.size());
            } else {
                plan.boundary_count++;
                int64_t overlap = state_overlap_count(lane.microstates[idx - 1].resident_partitions, microstate.resident_partitions);
                microstate.overlap_with_prev = overlap;
                microstate.admitted_partitions = static_cast<int64_t>(plan.buffer_capacity) - overlap;
                plan.total_partition_loads += microstate.admitted_partitions;
                plan.max_overlap = std::max<int64_t>(plan.max_overlap, overlap);
            }
        }
    }

    lift_stateflow_plan_ir(plan, partition_row_counts);
    populate_cross_lane_handoffs(plan, partition_row_counts, layout);
    populate_fragment_exact_semantics(plan);
    populate_fragment_estimates(plan, edge_bucket_sizes);
    plan.estimated_bucket_edges = estimate_plan_bucket_edges(plan, edge_bucket_sizes);
}

bool stateflow_plan_valid(const StateflowPlan &plan) {
    if (plan.lanes.empty()) {
        return false;
    }
    for (const auto &lane : plan.lanes) {
        if (!lane.microstates.empty()) {
            return true;
        }
    }
    return false;
}

StateflowPlan tensor_ordering_to_stateflow_plan(const std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> &ordering,
                                                PlanFamily family,
                                                int num_partitions,
                                                int buffer_capacity,
                                                PlanVariant family_variant = PlanVariant::DEFAULT,
                                                const std::vector<int64_t> &edge_bucket_sizes = {},
                                                const std::vector<int64_t> &partition_row_counts = {}) {
    StateflowPlan plan;
    plan.family = family;
    plan.family_variant = family_variant;
    plan.gpu_count = 1;
    plan.buffer_capacity = buffer_capacity;
    plan.num_partitions = num_partitions;
    plan.lanes.resize(1);
    plan.lanes[0].lane_id = 0;

    const auto &buffer_states = std::get<0>(ordering);
    const auto &edge_buckets_per_buffer = std::get<1>(ordering);
    if (buffer_states.size() != edge_buckets_per_buffer.size()) {
        return plan;
    }

    for (std::size_t idx = 0; idx < buffer_states.size(); idx++) {
        auto state_tensor = buffer_states[idx].to(torch::kCPU).to(torch::kInt64).contiguous();
        auto bucket_tensor = edge_buckets_per_buffer[idx].to(torch::kCPU).to(torch::kInt64).contiguous();

        auto *state_ptr = state_tensor.data_ptr<int64_t>();
        std::vector<int> resident_partitions;
        resident_partitions.reserve(state_tensor.numel());
        for (int64_t offset = 0; offset < state_tensor.numel(); offset++) {
            resident_partitions.emplace_back(static_cast<int>(state_ptr[offset]));
        }

        std::vector<std::pair<int, int>> edge_buckets;
        if (bucket_tensor.numel() > 0) {
            auto accessor = bucket_tensor.accessor<int64_t, 2>();
            edge_buckets.reserve(bucket_tensor.size(0));
            for (int64_t row = 0; row < bucket_tensor.size(0); row++) {
                edge_buckets.emplace_back(static_cast<int>(accessor[row][0]), static_cast<int>(accessor[row][1]));
            }
        }

        append_hybrid_cover_microstate(plan.lanes[0], static_cast<int64_t>(idx), std::move(resident_partitions), std::move(edge_buckets));
    }

    plan.total_superstates = static_cast<int64_t>(plan.lanes[0].microstates.size());
    finalize_stateflow_plan(plan, edge_bucket_sizes, partition_row_counts);
    return plan;
}

int64_t estimate_plan_bucket_edges(StateflowPlan &plan, const std::vector<int64_t> &edge_bucket_sizes) {
    populate_fragment_estimates(plan, edge_bucket_sizes);
    int64_t estimated_edges = 0;
    for (auto &lane : plan.lanes) {
        for (auto &microstate : lane.microstates) {
            std::unordered_set<int64_t> admitted_ids(microstate.admitted_object_ids.begin(), microstate.admitted_object_ids.end());
            for (auto &fragment : microstate.active_fragments) {
                // Charge edge volume only when the fragment depends on newly admitted resident objects.
                // This better matches the current single-GPU runtime cost, where retained objects stay hot
                // and most remap/swap overhead is driven by work introduced by the incoming partitions.
                bool touches_admitted_object = admitted_ids.empty();
                for (int64_t object_id : fragment.required_object_ids) {
                    if (admitted_ids.count(object_id) > 0) {
                        touches_admitted_object = true;
                        break;
                    }
                }

                if (!touches_admitted_object) {
                    continue;
                }
                estimated_edges += fragment.estimated_edges;
            }
        }
    }
    return estimated_edges;
}

std::string ir_histogram_string(const StateflowPlan &plan) {
    std::unordered_map<int, int64_t> role_hist;
    std::unordered_map<int, int64_t> kind_hist;
    std::unordered_map<int, int64_t> mode_hist;

    for (const auto &lane : plan.lanes) {
        for (const auto &ms : lane.microstates) {
            for (const auto &obj : ms.resident_objects) {
                role_hist[static_cast<int>(obj.role)]++;
            }
            for (const auto &frag : ms.active_fragments) {
                kind_hist[static_cast<int>(frag.fragment_kind)]++;
            }
        }
        for (const auto &h : lane.handoffs) {
            mode_hist[static_cast<int>(h.mode)]++;
        }
    }
    for (const auto &handoff : plan.cross_lane_handoffs) {
        mode_hist[static_cast<int>(handoff.mode)]++;
    }

    std::ostringstream oss;
    oss << "roles={ANCHOR:" << role_hist[static_cast<int>(ResidentObjectRole::ANCHOR)]
        << ",STREAM:" << role_hist[static_cast<int>(ResidentObjectRole::STREAM)]
        << ",SURVIVOR:" << role_hist[static_cast<int>(ResidentObjectRole::SURVIVOR)]
        << ",INCOMING:" << role_hist[static_cast<int>(ResidentObjectRole::INCOMING)] << "}";
    oss << " fragments={FULLY_RESIDENT:" << kind_hist[static_cast<int>(FragmentKind::FULLY_RESIDENT)]
        << ",ANCHOR_ANCHOR:" << kind_hist[static_cast<int>(FragmentKind::ANCHOR_ANCHOR)]
        << ",ANCHOR_STREAM:" << kind_hist[static_cast<int>(FragmentKind::ANCHOR_STREAM)] << "}";
    oss << " handoffs={FULL_RELOAD:" << mode_hist[static_cast<int>(HandoffMode::FULL_RELOAD)]
        << ",ROTATING_OVERWRITE:" << mode_hist[static_cast<int>(HandoffMode::ROTATING_OVERWRITE)]
        << ",PEER_RELAY:" << mode_hist[static_cast<int>(HandoffMode::PEER_RELAY)]
        << ",DELAYED_KEEP_ALIVE:" << mode_hist[static_cast<int>(HandoffMode::DELAYED_KEEP_ALIVE)] << "}";
    return oss.str();
}

template <typename T>
bool sorted_vectors_equal(std::vector<T> lhs, std::vector<T> rhs) {
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    return lhs == rhs;
}

bool validate_stateflow_plan_exact_semantics_impl(const StateflowPlan &plan) {
    const bool debug_validate = std::getenv("GEGE_STATEFLOW_DEBUG_VALIDATE") != nullptr;
    auto fail = [&](const std::string &reason) {
        if (debug_validate) {
            SPDLOG_WARN("Stateflow validation failed for family={} variant={}: {}",
                        static_cast<int>(plan.family), static_cast<int>(plan.family_variant), reason);
        }
        return false;
    };
    if (plan.num_partitions <= 0) {
        return fail("num_partitions <= 0");
    }
    if (plan.total_admitted_objects != plan.total_partition_loads) {
        return fail("total_admitted_objects != total_partition_loads");
    }
    int64_t lane_local_handoffs = 0;
    for (const auto &lane : plan.lanes) {
        lane_local_handoffs += static_cast<int64_t>(lane.handoffs.size());
    }
    if (plan.total_handoffs != lane_local_handoffs + static_cast<int64_t>(plan.cross_lane_handoffs.size())) {
        return fail("total_handoffs mismatch");
    }
    if (plan.total_cross_lane_handoffs != static_cast<int64_t>(plan.cross_lane_handoffs.size())) {
        return fail("total_cross_lane_handoffs mismatch");
    }

    std::unordered_map<int64_t, int64_t> bucket_counts;
    for (const auto &lane : plan.lanes) {
        if (lane.handoffs.size() + (lane.microstates.empty() ? 0 : 1) != lane.microstates.size()) {
            return fail("lane handoff count does not match microstate count");
        }

        for (const auto &ms : lane.microstates) {
            if (ms.lane_id != lane.lane_id) {
                return fail("microstate lane_id mismatch");
            }
            if (ms.resident_objects.size() != ms.resident_partitions.size()) {
                return fail("resident_objects size != resident_partitions size");
            }

            std::unordered_set<int> resident_partition_ids(ms.resident_partitions.begin(), ms.resident_partitions.end());
            std::unordered_set<int> used_slots;
            std::unordered_set<int64_t> resident_object_ids;
            std::unordered_map<int, int64_t> partition_to_object_id;
            for (const auto &obj : ms.resident_objects) {
                if (obj.partition_id < 0 || obj.partition_id >= plan.num_partitions) {
                    return fail("resident object partition out of bounds");
                }
                if (obj.slot_id < 0 || obj.slot_id >= static_cast<int>(ms.resident_partitions.size())) {
                    return fail("resident object slot out of bounds");
                }
                if (resident_partition_ids.count(obj.partition_id) == 0) {
                    return fail("resident object partition not found in resident_partitions");
                }
                if (!used_slots.insert(obj.slot_id).second) {
                    return fail("duplicate resident object slot");
                }
                if (!resident_object_ids.insert(obj.object_id).second) {
                    return fail("duplicate resident object id");
                }
                partition_to_object_id[obj.partition_id] = obj.object_id;
                if (ms.resident_partitions[obj.slot_id] != obj.partition_id) {
                    return fail("resident object slot does not map to partition");
                }
            }

            std::unordered_set<int64_t> ms_bucket_keys;
            for (const auto &frag : ms.active_fragments) {
                if (!frag.exact_semantics_tag) {
                    return fail("fragment exact_semantics_tag false");
                }
                if (frag.edge_buckets.empty() || frag.required_object_ids.empty()) {
                    return fail("fragment missing edge buckets or required object ids");
                }
                for (int64_t object_id : frag.required_object_ids) {
                    if (resident_object_ids.count(object_id) == 0) {
                        return fail("fragment required object id missing from resident set");
                    }
                }
                for (const auto &[s, d] : frag.edge_buckets) {
                    if (s < 0 || s >= plan.num_partitions || d < 0 || d >= plan.num_partitions) {
                        return fail("fragment edge bucket partition out of bounds");
                    }
                    if (resident_partition_ids.count(s) == 0 || resident_partition_ids.count(d) == 0) {
                        return fail("fragment edge bucket references non-resident partition");
                    }
                    int64_t key = static_cast<int64_t>(s) * plan.num_partitions + static_cast<int64_t>(d);
                    ms_bucket_keys.insert(key);
                }
            }
            std::unordered_set<int64_t> expected_ms_bucket_keys;
            for (const auto &[s, d] : ms.edge_buckets) {
                int64_t key = static_cast<int64_t>(s) * plan.num_partitions + static_cast<int64_t>(d);
                expected_ms_bucket_keys.insert(key);
            }
            if (ms_bucket_keys != expected_ms_bucket_keys) {
                return fail("fragment union does not match microstate edge buckets");
            }
            for (int64_t key : ms_bucket_keys) {
                bucket_counts[key]++;
            }
        }

        for (std::size_t idx = 1; idx < lane.microstates.size(); idx++) {
            const auto &prev_ms = lane.microstates[idx - 1];
            const auto &ms = lane.microstates[idx];
            const auto &handoff = lane.handoffs[idx - 1];
            if (handoff.src_microstate_id != prev_ms.microstate_id || handoff.dst_microstate_id != ms.microstate_id) {
                return fail("lane-local handoff endpoints mismatch");
            }
            if (handoff.src_lane_id != lane.lane_id || handoff.dst_lane_id != lane.lane_id) {
                return fail("lane-local handoff lane ids mismatch");
            }
            if (handoff.mode == HandoffMode::PEER_RELAY) {
                return fail("lane-local PEER_RELAY requires cross-lane descriptor");
            }

            std::unordered_map<int, int64_t> prev_partition_to_object_id;
            for (const auto &obj : prev_ms.resident_objects) {
                prev_partition_to_object_id[obj.partition_id] = obj.object_id;
            }

            std::vector<int64_t> expected_kept_object_ids;
            std::vector<int64_t> expected_evicted_object_ids;
            std::vector<int64_t> expected_admitted_object_ids(ms.admitted_object_ids.begin(), ms.admitted_object_ids.end());
            std::vector<std::pair<int, int>> expected_slot_mapping;
            std::unordered_set<int> curr_set(ms.resident_partitions.begin(), ms.resident_partitions.end());

            for (int slot = 0; slot < static_cast<int>(prev_ms.resident_partitions.size()); slot++) {
                int part = prev_ms.resident_partitions[slot];
                auto prev_obj_it = prev_partition_to_object_id.find(part);
                if (prev_obj_it == prev_partition_to_object_id.end()) {
                    return fail("previous partition missing object id");
                }
                if (curr_set.count(part) > 0) {
                    expected_kept_object_ids.emplace_back(prev_obj_it->second);
                    int dst_slot = -1;
                    for (int s2 = 0; s2 < static_cast<int>(ms.resident_partitions.size()); s2++) {
                        if (ms.resident_partitions[s2] == part) {
                            dst_slot = s2;
                            break;
                        }
                    }
                    expected_slot_mapping.emplace_back(slot, dst_slot);
                } else {
                    expected_evicted_object_ids.emplace_back(prev_obj_it->second);
                }
            }

            if (!sorted_vectors_equal(expected_kept_object_ids, handoff.kept_object_ids) ||
                !sorted_vectors_equal(expected_admitted_object_ids, handoff.admitted_object_ids) ||
                !sorted_vectors_equal(expected_evicted_object_ids, handoff.evicted_object_ids) ||
                !sorted_vectors_equal(expected_slot_mapping, handoff.slot_mapping)) {
                return fail("lane-local handoff payload mismatch");
            }

            HandoffMode expected_mode;
            if (expected_admitted_object_ids.empty()) {
                expected_mode = HandoffMode::DELAYED_KEEP_ALIVE;
            } else if (!expected_kept_object_ids.empty()) {
                expected_mode = HandoffMode::ROTATING_OVERWRITE;
            } else {
                expected_mode = HandoffMode::FULL_RELOAD;
            }
            if (handoff.mode != expected_mode) {
                return fail("lane-local handoff mode mismatch");
            }

            std::unordered_set<int> prev_set(prev_ms.resident_partitions.begin(), prev_ms.resident_partitions.end());
            std::unordered_set<int> local_admitted_partitions;
            for (int64_t object_id : handoff.admitted_object_ids) {
                bool found_partition = false;
                for (const auto &obj : ms.resident_objects) {
                    if (obj.object_id == object_id) {
                        local_admitted_partitions.insert(obj.partition_id);
                        found_partition = true;
                        break;
                    }
                }
                if (!found_partition) {
                    return fail("admitted object id missing from resident objects");
                }
            }

            std::unordered_set<int> peer_admitted_partitions;
            for (const auto &peer_handoff : plan.cross_lane_handoffs) {
                if (peer_handoff.dst_lane_id != lane.lane_id || peer_handoff.dst_microstate_id != ms.microstate_id) {
                    continue;
                }
                if (peer_handoff.slot_mapping.size() != 1) {
                    return fail("peer handoff slot mapping size != 1");
                }
                int dst_slot = peer_handoff.slot_mapping.front().second;
                if (dst_slot < 0 || dst_slot >= static_cast<int>(ms.resident_partitions.size())) {
                    return fail("peer handoff dst slot out of bounds");
                }
                int partition_id = ms.resident_partitions[dst_slot];
                if (prev_set.count(partition_id) > 0) {
                    return fail("peer admitted partition already resident in previous microstate");
                }
                if (local_admitted_partitions.count(partition_id) == 0) {
                    return fail("peer admitted partition not in local admitted set");
                }
                if (!peer_admitted_partitions.insert(partition_id).second) {
                    return fail("duplicate peer-admitted partition for destination microstate");
                }
            }

            std::unordered_set<int> host_admitted_partitions = local_admitted_partitions;
            for (int partition_id : peer_admitted_partitions) {
                if (host_admitted_partitions.erase(partition_id) == 0) {
                    return fail("peer-admitted partition missing from local admitted set");
                }
            }

            std::unordered_set<int> expected_new_partitions;
            std::unordered_set<int> kept_partitions;
            for (int partition_id : ms.resident_partitions) {
                if (prev_set.count(partition_id) > 0) {
                    kept_partitions.insert(partition_id);
                } else {
                    expected_new_partitions.insert(partition_id);
                }
            }

            std::unordered_set<int> covered_new_partitions = host_admitted_partitions;
            covered_new_partitions.insert(peer_admitted_partitions.begin(), peer_admitted_partitions.end());
            if (covered_new_partitions != expected_new_partitions) {
                return fail("covered new partitions do not match expected new partitions");
            }

            std::unordered_set<int> covered_partitions = kept_partitions;
            covered_partitions.insert(covered_new_partitions.begin(), covered_new_partitions.end());
            if (covered_partitions != curr_set) {
                return fail("covered partitions do not match current resident set");
            }
        }
    }

    for (int64_t superstate_id = 0; superstate_id < plan.total_superstates; superstate_id++) {
        std::unordered_map<int, int> resident_partition_to_lane;
        for (const auto &lane : plan.lanes) {
            for (const auto &microstate : lane.microstates) {
                if (microstate.superstate_id != superstate_id) {
                    continue;
                }
                for (int partition_id : microstate.resident_partitions) {
                    auto [it, inserted] = resident_partition_to_lane.emplace(partition_id, lane.lane_id);
                    if (!inserted && it->second != lane.lane_id) {
                        return fail("duplicate resident partition across lanes within superstate");
                    }
                }
            }
        }
    }

    std::unordered_set<int64_t> peer_target_keys;
    for (const auto &handoff : plan.cross_lane_handoffs) {
        if (handoff.mode != HandoffMode::PEER_RELAY) {
            return fail("cross-lane handoff mode is not PEER_RELAY");
        }
        if (handoff.src_lane_id < 0 || handoff.dst_lane_id < 0 || handoff.src_lane_id == handoff.dst_lane_id) {
            return fail("cross-lane handoff lane ids invalid");
        }
        const LanePlan *src_lane = nullptr;
        const LanePlan *dst_lane = nullptr;
        const MicrostatePlan *src_microstate = nullptr;
        const MicrostatePlan *dst_microstate = nullptr;
        for (const auto &lane : plan.lanes) {
            if (lane.lane_id == handoff.src_lane_id) {
                src_lane = &lane;
                for (const auto &microstate : lane.microstates) {
                    if (microstate.microstate_id == handoff.src_microstate_id) {
                        src_microstate = &microstate;
                        break;
                    }
                }
            }
            if (lane.lane_id == handoff.dst_lane_id) {
                dst_lane = &lane;
                for (const auto &microstate : lane.microstates) {
                    if (microstate.microstate_id == handoff.dst_microstate_id) {
                        dst_microstate = &microstate;
                        break;
                    }
                }
            }
        }
        if (src_lane == nullptr || dst_lane == nullptr || src_microstate == nullptr || dst_microstate == nullptr) {
            return fail("cross-lane handoff references missing lane or microstate");
        }
        if (src_microstate->superstate_id + 1 != dst_microstate->superstate_id) {
            return fail("cross-lane handoff superstate transition invalid");
        }
        if (handoff.admitted_object_ids.size() != 1 || handoff.slot_mapping.size() != 1) {
            return fail("cross-lane handoff payload size invalid");
        }
        int partition_id = -1;
        int dst_slot = handoff.slot_mapping.front().second;
        int src_slot = handoff.slot_mapping.front().first;
        if (src_slot < 0 || dst_slot < 0 ||
            src_slot >= static_cast<int>(src_microstate->resident_partitions.size()) ||
            dst_slot >= static_cast<int>(dst_microstate->resident_partitions.size())) {
            return fail("cross-lane handoff slot out of bounds");
        }
        partition_id = dst_microstate->resident_partitions[dst_slot];
        if (src_microstate->resident_partitions[src_slot] != partition_id) {
            return fail("cross-lane handoff source slot does not hold destination partition");
        }
        int64_t peer_target_key = (static_cast<int64_t>(handoff.dst_lane_id) << 40) |
                                  (static_cast<int64_t>(handoff.dst_microstate_id) << 20) |
                                  static_cast<int64_t>(partition_id);
        if (!peer_target_keys.insert(peer_target_key).second) {
            return fail("duplicate cross-lane peer target");
        }
        bool admitted_object_matches = false;
        for (const auto &obj : dst_microstate->resident_objects) {
            if (obj.partition_id == partition_id && obj.object_id == handoff.admitted_object_ids.front()) {
                admitted_object_matches = true;
                break;
            }
        }
        if (!admitted_object_matches) {
            return fail("cross-lane handoff admitted object id does not match destination partition");
        }
        const auto &dst_lane_microstates = dst_lane->microstates;
        auto dst_it = std::find_if(dst_lane_microstates.begin(), dst_lane_microstates.end(),
                                   [&](const auto &microstate) { return microstate.microstate_id == dst_microstate->microstate_id; });
        if (dst_it == dst_lane_microstates.begin() || dst_it == dst_lane_microstates.end()) {
            return fail("cross-lane destination microstate missing previous microstate");
        }
        const auto &prev_dst_microstate = *(dst_it - 1);
        if (std::find(prev_dst_microstate.resident_partitions.begin(), prev_dst_microstate.resident_partitions.end(), partition_id) !=
            prev_dst_microstate.resident_partitions.end()) {
            return fail("cross-lane admitted partition already present in previous destination microstate");
        }
        if (handoff.peer_bytes <= 0) {
            return fail("cross-lane peer_bytes <= 0");
        }
    }

    const int64_t expected_total_buckets = static_cast<int64_t>(plan.num_partitions) * static_cast<int64_t>(plan.num_partitions);
    if (static_cast<int64_t>(bucket_counts.size()) != expected_total_buckets) {
        return fail("bucket coverage cardinality mismatch");
    }
    for (int src_part = 0; src_part < plan.num_partitions; src_part++) {
        for (int dst_part = 0; dst_part < plan.num_partitions; dst_part++) {
            int64_t key = static_cast<int64_t>(src_part) * plan.num_partitions + dst_part;
            auto count_it = bucket_counts.find(key);
            if (count_it == bucket_counts.end() || count_it->second != 1) {
                return fail("bucket coverage is not exact-once");
            }
        }
    }

    return true;
}

std::string overlap_histogram_string(const StateflowPlan &plan) {
    std::unordered_map<int64_t, int64_t> histogram;
    for (const auto &lane : plan.lanes) {
        for (std::size_t idx = 1; idx < lane.microstates.size(); idx++) {
            histogram[lane.microstates[idx].overlap_with_prev]++;
        }
    }

    if (histogram.empty()) {
        return "[]";
    }

    std::vector<std::pair<int64_t, int64_t>> sorted_histogram(histogram.begin(), histogram.end());
    std::sort(sorted_histogram.begin(), sorted_histogram.end());
    std::ostringstream oss;
    oss << "[";
    for (std::size_t idx = 0; idx < sorted_histogram.size(); idx++) {
        if (idx > 0) {
            oss << ", ";
        }
        oss << sorted_histogram[idx].first << ":" << sorted_histogram[idx].second;
    }
    oss << "]";
    return oss.str();
}

std::string stateflow_plan_name(const StateflowPlan &plan) {
    if (plan.family_variant == PlanVariant::DEFAULT) {
        return planFamilyName(plan.family);
    }

    std::ostringstream oss;
    oss << planFamilyName(plan.family) << ":" << planVariantName(plan.family_variant);
    return oss.str();
}

std::string stateflow_cost_breakdown_string(const StateflowPlan &plan) {
    std::ostringstream oss;
    oss << "cost_terms={admit:" << plan.cost_breakdown.admitted_partition_cost
        << ",bucket_edges:" << plan.cost_breakdown.bucket_edge_cost
        << ",boundary:" << plan.cost_breakdown.boundary_cost
        << ",lane_imbalance:" << plan.cost_breakdown.lane_imbalance_cost
        << ",selection:" << plan.cost_breakdown.selection_cost
        << "} raw={admitted_partitions:" << plan.total_partition_loads
        << ",weighted_admission_load:" << plan.cost_breakdown.weighted_admission_load
        << ",admits_by_role={anchor:" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::ANCHOR)]
        << ",stream:" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::STREAM)]
        << ",survivor:" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::SURVIVOR)]
        << ",incoming:" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::INCOMING)]
        << "},estimated_bucket_edges:" << plan.estimated_bucket_edges
        << ",boundary_count:" << plan.boundary_count
        << ",lane_imbalance:" << plan.cost_breakdown.lane_imbalance
        << ",cross_lane_handoffs:" << plan.total_cross_lane_handoffs
        << ",peer_handoff_bytes:" << plan.total_peer_handoff_bytes
        << ",planned_peer_bytes:" << plan.cost_breakdown.planned_peer_bytes
        << "}";
    return oss.str();
}

std::string json_escape_string(const std::string &input) {
    std::ostringstream oss;
    for (char ch : input) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << ch;
                break;
        }
    }
    return oss.str();
}

void log_stateflow_plan_summary(const StateflowPlan &plan, const std::string &label) {
    SPDLOG_INFO(
        "{} family={} gpu_count={} buffer_capacity={} lanes={} microstates={} handoffs={} cross_lane_handoffs={} total_admitted_objects={} overlap_hist={} {}",
        label, stateflow_plan_name(plan), plan.gpu_count, plan.buffer_capacity, plan.lanes.size(), plan.total_microstates,
        plan.total_handoffs, plan.total_cross_lane_handoffs, plan.total_admitted_objects, overlap_histogram_string(plan),
        stateflow_cost_breakdown_string(plan));
    SPDLOG_INFO("{} IR {}", label, ir_histogram_string(plan));
}

double handoff_mode_multiplier(HandoffMode mode) {
    switch (mode) {
        case HandoffMode::FULL_RELOAD:
            return stateflow_cost_env("GEGE_STATEFLOW_MODE_MULT_FULL_RELOAD", 1.0);
        case HandoffMode::ROTATING_OVERWRITE:
            return stateflow_cost_env("GEGE_STATEFLOW_MODE_MULT_ROTATING", 1.0);
        case HandoffMode::PEER_RELAY:
            return stateflow_cost_env("GEGE_STATEFLOW_MODE_MULT_PEER_RELAY", 1.0);
        case HandoffMode::DELAYED_KEEP_ALIVE:
            return stateflow_cost_env("GEGE_STATEFLOW_MODE_MULT_DELAYED", 0.0);
    }
    return 1.0;
}

double resident_role_multiplier(ResidentObjectRole role) {
    switch (role) {
        case ResidentObjectRole::ANCHOR:
            return stateflow_cost_env("GEGE_STATEFLOW_ROLE_MULT_ANCHOR", 1.0);
        case ResidentObjectRole::STREAM:
            return stateflow_cost_env("GEGE_STATEFLOW_ROLE_MULT_STREAM", 1.0);
        case ResidentObjectRole::SURVIVOR:
            return stateflow_cost_env("GEGE_STATEFLOW_ROLE_MULT_SURVIVOR", 1.0);
        case ResidentObjectRole::INCOMING:
            return stateflow_cost_env("GEGE_STATEFLOW_ROLE_MULT_INCOMING", 1.0);
    }
    return 1.0;
}

double score_stateflow_plan(StateflowPlan &plan,
                            const std::vector<int64_t> &edge_bucket_sizes,
                            const PlanEmbeddingLayout &layout = {}) {
    double alpha = stateflow_cost_env("GEGE_STATEFLOW_COST_ALPHA", 1.0);
    double beta = stateflow_cost_env("GEGE_STATEFLOW_COST_BETA", 1e-7);
    // Boundary count is a secondary penalty on single GPU: the dominant cost already comes from
    // how many objects are admitted across those boundaries. Keep gamma below alpha to avoid
    // double-counting swap pressure and incorrectly preferring low-boundary / high-admission plans.
    double gamma = stateflow_cost_env("GEGE_STATEFLOW_COST_GAMMA", 0.95);
    double delta = stateflow_cost_env("GEGE_STATEFLOW_COST_DELTA", 0.25);

    plan.estimated_bucket_edges = estimate_plan_bucket_edges(plan, edge_bucket_sizes);

    double weighted_admission_load = 0.0;
    double selection_admission_load = 0.0;
    const int64_t bytes_per_row = plan_embedding_bytes_per_row(layout);
    for (auto &lane : plan.lanes) {
        for (std::size_t idx = 0; idx < lane.microstates.size(); idx++) {
            auto &microstate = lane.microstates[idx];
            if (microstate.admitted_object_ids.empty()) {
                continue;
            }
            std::unordered_map<int64_t, ResidentObjectRole> id_to_role;
            std::unordered_map<int64_t, int64_t> id_to_rows;
            id_to_role.reserve(microstate.resident_objects.size());
            id_to_rows.reserve(microstate.resident_objects.size());
            for (const auto &obj : microstate.resident_objects) {
                id_to_role[obj.object_id] = obj.role;
                id_to_rows[obj.object_id] = obj.rows;
            }
            double mode_mult = 1.0;
            int64_t handoff_cost = 0;
            if (idx > 0 && idx - 1 < lane.handoffs.size()) {
                mode_mult = handoff_mode_multiplier(lane.handoffs[idx - 1].mode);
            }
            for (int64_t object_id : microstate.admitted_object_ids) {
                auto role_it = id_to_role.find(object_id);
                double role_mult = role_it == id_to_role.end() ? 1.0 : resident_role_multiplier(role_it->second);
                weighted_admission_load += role_mult * mode_mult;
                int64_t transfer_units = 1;
                if (bytes_per_row > 0) {
                    auto rows_it = id_to_rows.find(object_id);
                    const int64_t rows = rows_it == id_to_rows.end() ? 0 : rows_it->second;
                    transfer_units = std::max<int64_t>(rows, 1) * bytes_per_row;
                }
                selection_admission_load += static_cast<double>(transfer_units);
                handoff_cost++;
            }
            if (idx > 0 && idx - 1 < lane.handoffs.size()) {
                lane.handoffs[idx - 1].estimated_cost = static_cast<int64_t>(std::llround(handoff_cost * mode_mult));
            }
        }
    }

    int64_t lane_min = std::numeric_limits<int64_t>::max();
    int64_t lane_max = 0;
    std::vector<int64_t> lane_edges_by_lane;
    lane_edges_by_lane.reserve(plan.lanes.size());
    for (const auto &lane : plan.lanes) {
        int64_t lane_edges = 0;
        if (edge_bucket_sizes.size() == static_cast<std::size_t>(plan.num_partitions * plan.num_partitions)) {
            for (const auto &microstate : lane.microstates) {
                for (const auto &[src_part, dst_part] : microstate.edge_buckets) {
                    lane_edges += edge_bucket_sizes[src_part * plan.num_partitions + dst_part];
                }
            }
        } else {
            for (const auto &microstate : lane.microstates) {
                lane_edges += static_cast<int64_t>(microstate.edge_buckets.size());
            }
        }
        lane_min = std::min(lane_min, lane_edges);
        lane_max = std::max(lane_max, lane_edges);
        lane_edges_by_lane.emplace_back(lane_edges);
    }
    int64_t lane_imbalance = lane_min == std::numeric_limits<int64_t>::max() ? 0 : lane_max - lane_min;
    if (plan.lanes.size() <= 1) {
        lane_imbalance = 0;
    }
    double lane_imbalance_cost = 0.0;
    if (plan.lanes.size() > 1 && !lane_edges_by_lane.empty()) {
        const double mean_lane_edges =
            static_cast<double>(std::accumulate(lane_edges_by_lane.begin(), lane_edges_by_lane.end(), int64_t{0})) /
            static_cast<double>(lane_edges_by_lane.size());
        if (mean_lane_edges > 0.0) {
            for (auto lane_edges : lane_edges_by_lane) {
                lane_imbalance_cost += std::pow(static_cast<double>(lane_edges) - mean_lane_edges, 2.0) / mean_lane_edges;
            }
            lane_imbalance_cost *= delta;
        }
    }

    const double selection_admission_basis = plan.lanes.size() > 1 ? selection_admission_load : weighted_admission_load;
    plan.cost_breakdown.weighted_admission_load = weighted_admission_load;
    plan.cost_breakdown.admitted_partition_cost = alpha * selection_admission_basis;
    plan.cost_breakdown.bucket_edge_cost = beta * static_cast<double>(plan.estimated_bucket_edges);
    plan.cost_breakdown.boundary_cost = gamma * static_cast<double>(plan.boundary_count);
    plan.cost_breakdown.lane_imbalance_cost = lane_imbalance_cost;
    plan.cost_breakdown.lane_imbalance = lane_imbalance;
    plan.cost_breakdown.selection_cost = plan.cost_breakdown.admitted_partition_cost +
                                         plan.cost_breakdown.bucket_edge_cost +
                                         plan.cost_breakdown.boundary_cost +
                                         plan.cost_breakdown.lane_imbalance_cost;
    plan.cost_breakdown.planned_peer_bytes = plan.total_peer_handoff_bytes;

    plan.estimated_cost = plan.cost_breakdown.selection_cost;
    return plan.estimated_cost;
}

MicrostatePlan tensor_state_to_microstate(torch::Tensor state_tensor,
                                          torch::Tensor bucket_tensor,
                                          int64_t microstate_id,
                                          int64_t superstate_id) {
    MicrostatePlan microstate;
    microstate.microstate_id = microstate_id;
    microstate.superstate_id = superstate_id;

    state_tensor = state_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *state_ptr = state_tensor.data_ptr<int64_t>();
    microstate.resident_partitions.reserve(state_tensor.numel());
    for (int64_t offset = 0; offset < state_tensor.numel(); offset++) {
        microstate.resident_partitions.emplace_back(static_cast<int>(state_ptr[offset]));
    }

    bucket_tensor = bucket_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    if (bucket_tensor.numel() > 0) {
        auto accessor = bucket_tensor.accessor<int64_t, 2>();
        microstate.edge_buckets.reserve(bucket_tensor.size(0));
        for (int64_t row = 0; row < bucket_tensor.size(0); row++) {
            microstate.edge_buckets.emplace_back(static_cast<int>(accessor[row][0]), static_cast<int>(accessor[row][1]));
        }
    }

    return microstate;
}

StateflowPlan build_multi_gpu_stateflow_plan_from_permutation(const vector<torch::Tensor> &buffer_states,
                                                              const vector<torch::Tensor> &edge_buckets_per_buffer,
                                                              const std::vector<int64_t> &permutation,
                                                              int active_devices,
                                                              PlanVariant family_variant = PlanVariant::DEFAULT,
                                                              const std::vector<int64_t> &edge_bucket_sizes = {},
                                                              const std::vector<int64_t> &partition_row_counts = {},
                                                              const PlanEmbeddingLayout &layout = {}) {
    StateflowPlan plan;
    if (buffer_states.empty() || edge_buckets_per_buffer.size() != buffer_states.size() || active_devices <= 1 ||
        permutation.size() != buffer_states.size()) {
        return plan;
    }

    plan.family = PlanFamily::CUSTOM;
    plan.family_variant = family_variant;
    plan.gpu_count = active_devices;
    plan.buffer_capacity = buffer_states.front().numel();
    plan.num_partitions = 0;
    for (const auto &state : buffer_states) {
        auto state_cpu = state.to(torch::kCPU).to(torch::kInt64).contiguous();
        auto *data = state_cpu.data_ptr<int64_t>();
        for (int64_t idx = 0; idx < state_cpu.numel(); idx++) {
            plan.num_partitions = std::max<int64_t>(plan.num_partitions, data[idx] + 1);
        }
    }

    plan.lanes.resize(active_devices);
    for (int lane_idx = 0; lane_idx < active_devices; lane_idx++) {
        plan.lanes[lane_idx].lane_id = lane_idx;
    }

    for (std::size_t ordered_idx = 0; ordered_idx < permutation.size(); ordered_idx++) {
        int64_t state_idx = permutation[ordered_idx];
        if (state_idx < 0 || state_idx >= static_cast<int64_t>(buffer_states.size())) {
            return StateflowPlan{};
        }
        int lane_idx = static_cast<int>(ordered_idx % static_cast<std::size_t>(active_devices));
        int64_t round_idx = static_cast<int64_t>(ordered_idx / static_cast<std::size_t>(active_devices));
        plan.lanes[lane_idx].microstates.emplace_back(
            tensor_state_to_microstate(buffer_states[state_idx], edge_buckets_per_buffer[state_idx],
                                       static_cast<int64_t>(plan.lanes[lane_idx].microstates.size()), round_idx));
    }

    plan.total_superstates = (static_cast<int64_t>(permutation.size()) + active_devices - 1) / active_devices;
    finalize_stateflow_plan(plan, edge_bucket_sizes, partition_row_counts, layout);
    return plan;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> permute_tensor_ordering(
    const std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> &ordering,
    const std::vector<int64_t> &permutation) {
    const auto &buffer_states = std::get<0>(ordering);
    const auto &edge_buckets = std::get<1>(ordering);
    if (permutation.size() != buffer_states.size() || edge_buckets.size() != buffer_states.size()) {
        return ordering;
    }

    std::vector<torch::Tensor> reordered_states;
    std::vector<torch::Tensor> reordered_buckets;
    reordered_states.reserve(buffer_states.size());
    reordered_buckets.reserve(edge_buckets.size());
    for (auto idx : permutation) {
        if (idx < 0 || idx >= static_cast<int64_t>(buffer_states.size())) {
            return ordering;
        }
        reordered_states.emplace_back(buffer_states[idx]);
        reordered_buckets.emplace_back(edge_buckets[idx]);
    }

    return std::make_tuple(std::move(reordered_states), std::move(reordered_buckets));
}

StateflowPlan build_single_gpu_gpu_aware_custom_candidate(const StateflowPlan &custom_plan,
                                                          int num_partitions,
                                                          const std::vector<int64_t> &edge_bucket_sizes,
                                                          const std::vector<int64_t> &partition_row_counts) {
    // Round-trips through the legacy (buffer_states, edge_buckets) tensor pair because the
    // gpu-aware permutation helper operates on that representation. Safe for cost ablation
    // since tensor_ordering_to_stateflow_plan + lift_stateflow_plan_ir are deterministic.
    auto projected = projectStateflowPlanToLegacySchedule(custom_plan);
    const auto &buffer_states = std::get<0>(projected);
    const auto &edge_buckets = std::get<1>(projected);
    if (buffer_states.size() <= 1 || edge_buckets.size() != buffer_states.size()) {
        return StateflowPlan{};
    }

    auto permutation = getSingleGpuGpuAwareCustomPermutation(buffer_states, edge_bucket_sizes, num_partitions);
    bool changed = false;
    for (std::size_t idx = 0; idx < permutation.size(); idx++) {
        if (permutation[idx] != static_cast<int64_t>(idx)) {
            changed = true;
            break;
        }
    }
    if (!changed) {
        return StateflowPlan{};
    }

    auto reordered = permute_tensor_ordering(projected, permutation);
    return tensor_ordering_to_stateflow_plan(reordered, PlanFamily::CUSTOM, num_partitions, custom_plan.buffer_capacity,
                                             PlanVariant::CUSTOM_GPU_AWARE, edge_bucket_sizes, partition_row_counts);
}

StateflowPlan compileHybridCoverStateflowPlanVariant(int num_partitions,
                                                     int buffer_capacity,
                                                     HybridCoverVariant variant,
                                                     const std::vector<int64_t> &edge_bucket_sizes = {},
                                                     const std::vector<int64_t> &partition_row_counts = {}) {
    StateflowPlan plan;
    plan.family = PlanFamily::HYBRID_COVER;
    plan.family_variant = hybrid_cover_variant_plan_variant(variant);
    plan.gpu_count = 1;
    plan.buffer_capacity = buffer_capacity;
    plan.num_partitions = num_partitions;

    if (buffer_capacity != 4 || num_partitions < 4 || num_partitions % 3 != 1) {
        return plan;
    }

    std::vector<int> ordered_partitions(num_partitions);
    std::iota(ordered_partitions.begin(), ordered_partitions.end(), 0);

    plan.lanes.resize(1);
    plan.lanes[0].lane_id = 0;
    build_hybrid_cover_plan_recursive(ordered_partitions, plan.lanes[0], plan.total_superstates, variant);
    finalize_stateflow_plan(plan, edge_bucket_sizes, partition_row_counts);
    return plan;
}

}  // namespace

bool validateStateflowPlanExactSemantics(const StateflowPlan &plan) {
    return validate_stateflow_plan_exact_semantics_impl(plan);
}

std::string planFamilyName(PlanFamily family) {
    switch (family) {
        case PlanFamily::CUSTOM:
            return "CUSTOM";
        case PlanFamily::HYBRID_COVER:
            return "HYBRID_COVER";
        default:
            return "UNKNOWN";
    }
}

std::string planVariantName(PlanVariant variant) {
    switch (variant) {
        case PlanVariant::DEFAULT:
            return "default";
        case PlanVariant::CUSTOM_CANONICAL:
            return "canonical";
        case PlanVariant::CUSTOM_GPU_AWARE:
            return "gpu_aware";
        case PlanVariant::CUSTOM_REVERSED:
            return "reversed";
        case PlanVariant::CUSTOM_LEGACY_RANDOM:
            return "legacy_random";
        case PlanVariant::HYBRID_COVER_LEGACY_ROTATED:
            return "legacy_rotated";
        case PlanVariant::HYBRID_COVER_NATURAL:
            return "natural";
        case PlanVariant::HYBRID_COVER_REVERSED:
            return "reversed";
        case PlanVariant::MULTI_GPU_DISJOINT_ROUNDS:
            return "disjoint_rounds";
        case PlanVariant::MULTI_GPU_LANE_MATCHED:
            return "lane_matched";
    }
    return "unknown";
}

std::string residentObjectRoleName(ResidentObjectRole role) {
    switch (role) {
        case ResidentObjectRole::ANCHOR:
            return "ANCHOR";
        case ResidentObjectRole::STREAM:
            return "STREAM";
        case ResidentObjectRole::SURVIVOR:
            return "SURVIVOR";
        case ResidentObjectRole::INCOMING:
            return "INCOMING";
    }
    return "UNKNOWN";
}

std::string fragmentKindName(FragmentKind kind) {
    switch (kind) {
        case FragmentKind::FULLY_RESIDENT:
            return "FULLY_RESIDENT";
        case FragmentKind::ANCHOR_ANCHOR:
            return "ANCHOR_ANCHOR";
        case FragmentKind::ANCHOR_STREAM:
            return "ANCHOR_STREAM";
    }
    return "UNKNOWN";
}

std::string handoffModeName(HandoffMode mode) {
    switch (mode) {
        case HandoffMode::FULL_RELOAD:
            return "FULL_RELOAD";
        case HandoffMode::ROTATING_OVERWRITE:
            return "ROTATING_OVERWRITE";
        case HandoffMode::PEER_RELAY:
            return "PEER_RELAY";
        case HandoffMode::DELAYED_KEEP_ALIVE:
            return "DELAYED_KEEP_ALIVE";
    }
    return "UNKNOWN";
}

std::string stateflowPlanToText(const StateflowPlan &plan, bool include_microstates) {
    std::ostringstream oss;
    oss << "StateflowPlan family=" << stateflow_plan_name(plan)
        << " gpu_count=" << plan.gpu_count
        << " buffer_capacity=" << plan.buffer_capacity
        << " lanes=" << plan.lanes.size()
        << " microstates=" << plan.total_microstates
        << " handoffs=" << plan.total_handoffs
        << " cross_lane_handoffs=" << plan.total_cross_lane_handoffs
        << " total_admitted_objects=" << plan.total_admitted_objects
        << " overlap_hist=" << overlap_histogram_string(plan)
        << " " << stateflow_cost_breakdown_string(plan) << "\n";
    oss << "IR " << ir_histogram_string(plan) << "\n";

    if (!include_microstates) {
        return oss.str();
    }

    for (const auto &lane : plan.lanes) {
        oss << "lane " << lane.lane_id << "\n";
        for (const auto &microstate : lane.microstates) {
            oss << "  microstate " << microstate.microstate_id
                << " superstate=" << microstate.superstate_id
                << " overlap_with_prev=" << microstate.overlap_with_prev
                << " admitted_partitions=" << microstate.admitted_partitions
                << " resident=[";
            for (std::size_t idx = 0; idx < microstate.resident_partitions.size(); idx++) {
                if (idx > 0) {
                    oss << ",";
                }
                oss << microstate.resident_partitions[idx];
            }
            oss << "] buckets=" << microstate.edge_buckets.size() << "\n";
        }
        for (const auto &handoff : lane.handoffs) {
            oss << "  handoff " << handoff.handoff_id
                << " " << handoffModeName(handoff.mode)
                << " lane=" << handoff.src_lane_id << "->" << handoff.dst_lane_id
                << " " << handoff.src_microstate_id << "->" << handoff.dst_microstate_id
                << " kept=" << handoff.kept_object_ids.size()
                << " admitted=" << handoff.admitted_object_ids.size()
                << " evicted=" << handoff.evicted_object_ids.size() << "\n";
        }
    }
    for (const auto &handoff : plan.cross_lane_handoffs) {
        oss << "cross_lane_handoff " << handoff.handoff_id
            << " " << handoffModeName(handoff.mode)
            << " lane=" << handoff.src_lane_id << "->" << handoff.dst_lane_id
            << " " << handoff.src_microstate_id << "->" << handoff.dst_microstate_id
            << " admitted=" << handoff.admitted_object_ids.size()
            << " peer_bytes=" << handoff.peer_bytes << "\n";
    }

    return oss.str();
}

std::string stateflowPlanToJson(const StateflowPlan &plan, bool include_microstates) {
    std::ostringstream oss;
    auto append_int_vector = [&oss](const auto &values) {
        oss << "[";
        for (std::size_t idx = 0; idx < values.size(); idx++) {
            if (idx > 0) {
                oss << ",";
            }
            oss << values[idx];
        }
        oss << "]";
    };
    auto append_bucket_pairs = [&oss](const std::vector<std::pair<int, int>> &pairs) {
        oss << "[";
        for (std::size_t idx = 0; idx < pairs.size(); idx++) {
            if (idx > 0) {
                oss << ",";
            }
            oss << "[" << pairs[idx].first << "," << pairs[idx].second << "]";
        }
        oss << "]";
    };

    oss << "{";
    oss << "\"name\":\"" << json_escape_string(stateflow_plan_name(plan)) << "\"";
    oss << ",\"family\":\"" << json_escape_string(planFamilyName(plan.family)) << "\"";
    oss << ",\"variant\":\"" << json_escape_string(planVariantName(plan.family_variant)) << "\"";
    oss << ",\"gpu_count\":" << plan.gpu_count;
    oss << ",\"buffer_capacity\":" << plan.buffer_capacity;
    oss << ",\"num_partitions\":" << plan.num_partitions;
    oss << ",\"total_microstates\":" << plan.total_microstates;
    oss << ",\"total_superstates\":" << plan.total_superstates;
    oss << ",\"total_handoffs\":" << plan.total_handoffs;
    oss << ",\"total_cross_lane_handoffs\":" << plan.total_cross_lane_handoffs;
    oss << ",\"total_admitted_objects\":" << plan.total_admitted_objects;
    oss << ",\"total_peer_handoff_bytes\":" << plan.total_peer_handoff_bytes;
    oss << ",\"total_partition_loads\":" << plan.total_partition_loads;
    oss << ",\"total_bucket_assignments\":" << plan.total_bucket_assignments;
    oss << ",\"boundary_count\":" << plan.boundary_count;
    oss << ",\"max_overlap\":" << plan.max_overlap;
    oss << ",\"estimated_bucket_edges\":" << plan.estimated_bucket_edges;
    oss << ",\"estimated_cost\":" << plan.estimated_cost;
    oss << ",\"admissions_by_role\":{";
    oss << "\"anchor\":" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::ANCHOR)];
    oss << ",\"stream\":" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::STREAM)];
    oss << ",\"survivor\":" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::SURVIVOR)];
    oss << ",\"incoming\":" << plan.total_admissions_by_role[static_cast<int>(ResidentObjectRole::INCOMING)];
    oss << "}";
    oss << ",\"cost_breakdown\":{";
    oss << "\"admitted_partition_cost\":" << plan.cost_breakdown.admitted_partition_cost;
    oss << ",\"bucket_edge_cost\":" << plan.cost_breakdown.bucket_edge_cost;
    oss << ",\"boundary_cost\":" << plan.cost_breakdown.boundary_cost;
    oss << ",\"lane_imbalance_cost\":" << plan.cost_breakdown.lane_imbalance_cost;
    oss << ",\"selection_cost\":" << plan.cost_breakdown.selection_cost;
    oss << ",\"lane_imbalance\":" << plan.cost_breakdown.lane_imbalance;
    oss << ",\"planned_peer_bytes\":" << plan.cost_breakdown.planned_peer_bytes;
    oss << ",\"weighted_admission_load\":" << plan.cost_breakdown.weighted_admission_load;
    oss << "}";

    if (include_microstates) {
        oss << ",\"lanes\":[";
        for (std::size_t lane_idx = 0; lane_idx < plan.lanes.size(); lane_idx++) {
            if (lane_idx > 0) {
                oss << ",";
            }
            const auto &lane = plan.lanes[lane_idx];
            oss << "{\"lane_id\":" << lane.lane_id;
            oss << ",\"microstates\":[";
            for (std::size_t ms_idx = 0; ms_idx < lane.microstates.size(); ms_idx++) {
                if (ms_idx > 0) {
                    oss << ",";
                }
                const auto &microstate = lane.microstates[ms_idx];
                oss << "{\"microstate_id\":" << microstate.microstate_id;
                oss << ",\"lane_id\":" << microstate.lane_id;
                oss << ",\"superstate_id\":" << microstate.superstate_id;
                oss << ",\"overlap_with_prev\":" << microstate.overlap_with_prev;
                oss << ",\"admitted_partitions\":" << microstate.admitted_partitions;
                oss << ",\"resident_partitions\":";
                append_int_vector(microstate.resident_partitions);
                oss << ",\"edge_buckets\":";
                append_bucket_pairs(microstate.edge_buckets);
                oss << ",\"admitted_object_ids\":";
                append_int_vector(microstate.admitted_object_ids);
                oss << ",\"evicted_object_ids\":";
                append_int_vector(microstate.evicted_object_ids);
                oss << ",\"resident_objects\":[";
                for (std::size_t obj_idx = 0; obj_idx < microstate.resident_objects.size(); obj_idx++) {
                    if (obj_idx > 0) {
                        oss << ",";
                    }
                    const auto &obj = microstate.resident_objects[obj_idx];
                    oss << "{\"object_id\":" << obj.object_id
                        << ",\"partition_id\":" << obj.partition_id
                        << ",\"slot_id\":" << obj.slot_id
                        << ",\"role\":\"" << json_escape_string(residentObjectRoleName(obj.role)) << "\""
                        << ",\"rows\":" << obj.rows << "}";
                }
                oss << "]";
                oss << ",\"active_fragments\":[";
                for (std::size_t frag_idx = 0; frag_idx < microstate.active_fragments.size(); frag_idx++) {
                    if (frag_idx > 0) {
                        oss << ",";
                    }
                    const auto &frag = microstate.active_fragments[frag_idx];
                    oss << "{\"fragment_id\":" << frag.fragment_id;
                    oss << ",\"required_object_ids\":";
                    append_int_vector(frag.required_object_ids);
                    oss << ",\"edge_buckets\":";
                    append_bucket_pairs(frag.edge_buckets);
                    oss << ",\"estimated_edges\":" << frag.estimated_edges;
                    oss << ",\"fragment_kind\":\"" << json_escape_string(fragmentKindName(frag.fragment_kind)) << "\"";
                    oss << ",\"exact_semantics_tag\":" << (frag.exact_semantics_tag ? "true" : "false") << "}";
                }
                oss << "]";
                oss << "}";
            }
            oss << "],\"handoffs\":[";
            for (std::size_t handoff_idx = 0; handoff_idx < lane.handoffs.size(); handoff_idx++) {
                if (handoff_idx > 0) {
                    oss << ",";
                }
                const auto &handoff = lane.handoffs[handoff_idx];
                oss << "{\"handoff_id\":" << handoff.handoff_id;
                oss << ",\"src_microstate_id\":" << handoff.src_microstate_id;
                oss << ",\"dst_microstate_id\":" << handoff.dst_microstate_id;
                oss << ",\"kept_object_ids\":";
                append_int_vector(handoff.kept_object_ids);
                oss << ",\"admitted_object_ids\":";
                append_int_vector(handoff.admitted_object_ids);
                oss << ",\"evicted_object_ids\":";
                append_int_vector(handoff.evicted_object_ids);
                oss << ",\"slot_mapping\":";
                append_bucket_pairs(handoff.slot_mapping);
                oss << ",\"mode\":\"" << json_escape_string(handoffModeName(handoff.mode)) << "\"";
                oss << ",\"estimated_cost\":" << handoff.estimated_cost << "}";
            }
            oss << "]}";
        }
        oss << "],\"cross_lane_handoffs\":[";
        for (std::size_t handoff_idx = 0; handoff_idx < plan.cross_lane_handoffs.size(); handoff_idx++) {
            if (handoff_idx > 0) {
                oss << ",";
            }
            const auto &handoff = plan.cross_lane_handoffs[handoff_idx];
            oss << "{\"handoff_id\":" << handoff.handoff_id;
            oss << ",\"src_lane_id\":" << handoff.src_lane_id;
            oss << ",\"dst_lane_id\":" << handoff.dst_lane_id;
            oss << ",\"src_microstate_id\":" << handoff.src_microstate_id;
            oss << ",\"dst_microstate_id\":" << handoff.dst_microstate_id;
            oss << ",\"admitted_object_ids\":";
            append_int_vector(handoff.admitted_object_ids);
            oss << ",\"slot_mapping\":";
            append_bucket_pairs(handoff.slot_mapping);
            oss << ",\"mode\":\"" << json_escape_string(handoffModeName(handoff.mode)) << "\"";
            oss << ",\"estimated_cost\":" << handoff.estimated_cost;
            oss << ",\"peer_bytes\":" << handoff.peer_bytes << "}";
        }
        oss << "]";
    }

    oss << "}";
    return oss.str();
}

std::vector<int64_t> computePartitionRowCounts(int64_t total_rows, int num_partitions) {
    std::vector<int64_t> partition_row_counts;
    if (total_rows < 0 || num_partitions <= 0) {
        return partition_row_counts;
    }

    partition_row_counts.resize(num_partitions, 0);
    if (num_partitions == 0) {
        return partition_row_counts;
    }

    const int64_t partition_size = (total_rows + num_partitions - 1) / num_partitions;
    for (int partition = 0; partition < num_partitions; partition++) {
        const int64_t offset = static_cast<int64_t>(partition) * partition_size;
        if (offset >= total_rows) {
            partition_row_counts[partition] = 0;
            continue;
        }
        partition_row_counts[partition] = std::min<int64_t>(partition_size, total_rows - offset);
    }
    return partition_row_counts;
}

StateflowPlan compileHybridCoverStateflowPlan(int num_partitions,
                                              int buffer_capacity,
                                              const std::vector<int64_t> &edge_bucket_sizes,
                                              const std::vector<int64_t> &partition_row_counts) {
    StateflowPlan plan = compileHybridCoverStateflowPlanVariant(num_partitions, buffer_capacity, HybridCoverVariant::LEGACY_ROTATED,
                                                                edge_bucket_sizes, partition_row_counts);
    if (!stateflow_plan_valid(plan)) {
        SPDLOG_WARN("Hybrid-Cover ordering requires buffer_capacity=4 and num_partitions=3k+1; falling back to CUSTOM");
    }
    return plan;
}

StateflowPlan compileCustomStateflowPlan(int num_partitions,
                                         int buffer_capacity,
                                         bool randomly_assign_edge_buckets,
                                         const std::vector<int64_t> &edge_bucket_sizes,
                                         const std::vector<int64_t> &partition_row_counts) {
    if (!randomly_assign_edge_buckets) {
        auto buffer_states = build_custom_template_buffer_states(num_partitions, buffer_capacity);
        auto edge_buckets = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
        auto ordering = convertEdgeBucketOrderToTensors(std::move(buffer_states), std::move(edge_buckets));
        return tensor_ordering_to_stateflow_plan(ordering, PlanFamily::CUSTOM, num_partitions, buffer_capacity,
                                                 PlanVariant::CUSTOM_CANONICAL, edge_bucket_sizes, partition_row_counts);
    }

    auto ordering = getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, randomly_assign_edge_buckets);
    return tensor_ordering_to_stateflow_plan(ordering, PlanFamily::CUSTOM, num_partitions, buffer_capacity,
                                             PlanVariant::CUSTOM_LEGACY_RANDOM, edge_bucket_sizes, partition_row_counts);
}

StateflowPlan build_reversed_microstate_candidate(const StateflowPlan &source,
                                                  const std::vector<int64_t> &edge_bucket_sizes,
                                                  const std::vector<int64_t> &partition_row_counts) {
    if (source.family != PlanFamily::CUSTOM || source.lanes.size() != 1) {
        return StateflowPlan{};
    }
    const auto &source_ms = source.lanes[0].microstates;
    if (source_ms.size() < 2) {
        return StateflowPlan{};
    }

    auto projected = projectStateflowPlanToLegacySchedule(source);
    const auto &buffer_states = std::get<0>(projected);
    const auto &edge_buckets = std::get<1>(projected);
    if (buffer_states.size() != source_ms.size() || edge_buckets.size() != source_ms.size()) {
        return StateflowPlan{};
    }

    std::vector<int64_t> permutation(buffer_states.size());
    for (std::size_t idx = 0; idx < permutation.size(); idx++) {
        permutation[idx] = static_cast<int64_t>(buffer_states.size() - 1 - idx);
    }
    auto reordered = permute_tensor_ordering(projected, permutation);
    return tensor_ordering_to_stateflow_plan(reordered, PlanFamily::CUSTOM, static_cast<int>(source.num_partitions),
                                             static_cast<int>(source.buffer_capacity), PlanVariant::CUSTOM_REVERSED,
                                             edge_bucket_sizes, partition_row_counts);
}

std::string normalize_plan_token(const std::string &raw) {
    std::string normalized;
    normalized.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch)) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return normalized;
}

bool stateflow_plan_less(const StateflowPlan &lhs, const StateflowPlan &rhs) {
    if (lhs.estimated_cost != rhs.estimated_cost) {
        return lhs.estimated_cost < rhs.estimated_cost;
    }
    if (lhs.total_partition_loads != rhs.total_partition_loads) {
        return lhs.total_partition_loads < rhs.total_partition_loads;
    }
    if (lhs.max_overlap != rhs.max_overlap) {
        return lhs.max_overlap > rhs.max_overlap;
    }
    if (lhs.family != rhs.family) {
        return static_cast<int>(lhs.family) < static_cast<int>(rhs.family);
    }
    return static_cast<int>(lhs.family_variant) < static_cast<int>(rhs.family_variant);
}

bool force_matches_family(const StateflowPlan &plan, const std::string &forced_family) {
    if (forced_family.empty()) {
        return true;
    }

    const std::string normalized = normalize_plan_token(forced_family);
    if (normalized.empty()) {
        return true;
    }

    const std::string family_name = normalize_plan_token(planFamilyName(plan.family));
    if (normalized == family_name) {
        return true;
    }

    if (normalized == "hybridcover" && plan.family == PlanFamily::HYBRID_COVER) {
        return true;
    }

    return false;
}

bool force_matches_variant(const StateflowPlan &plan, const std::string &forced_variant) {
    if (forced_variant.empty()) {
        return true;
    }

    const std::string normalized = normalize_plan_token(forced_variant);
    if (normalized.empty()) {
        return true;
    }

    return normalized == normalize_plan_token(planVariantName(plan.family_variant)) ||
           normalized == normalize_plan_token(stateflow_plan_name(plan));
}

std::vector<StateflowPlan> enumerateSingleGpuStateflowPlans(int num_partitions,
                                                            int buffer_capacity,
                                                            bool randomly_assign_edge_buckets,
                                                            const std::vector<int64_t> &edge_bucket_sizes,
                                                            bool allow_hybrid_cover,
                                                            const std::vector<int64_t> &partition_row_counts) {
    std::vector<StateflowPlan> candidates;
    auto append_candidate = [&](StateflowPlan candidate) {
        if (!stateflow_plan_valid(candidate)) {
            return;
        }
        if (!validateStateflowPlanExactSemantics(candidate)) {
            SPDLOG_WARN("Skipping invalid Stateflow planner candidate {}", stateflow_plan_name(candidate));
            return;
        }
        score_stateflow_plan(candidate, edge_bucket_sizes);
        SPDLOG_DEBUG(
            "Stateflow planner candidate family={} cost={:.3f} loads={} boundaries={} directed_buckets={} "
            "estimated_bucket_edges={} overlap_hist={} {}",
            stateflow_plan_name(candidate), candidate.estimated_cost, candidate.total_partition_loads,
            candidate.boundary_count, candidate.total_bucket_assignments, candidate.estimated_bucket_edges,
            overlap_histogram_string(candidate), stateflow_cost_breakdown_string(candidate));
        candidates.emplace_back(std::move(candidate));
    };

    StateflowPlan custom_plan = compileCustomStateflowPlan(num_partitions, buffer_capacity, randomly_assign_edge_buckets,
                                                           edge_bucket_sizes, partition_row_counts);
    const bool custom_valid = stateflow_plan_valid(custom_plan);

    std::vector<std::function<StateflowPlan()>> factories;
    if (custom_valid) {
        factories.emplace_back([&custom_plan]() { return custom_plan; });
        factories.emplace_back([&custom_plan, num_partitions, &edge_bucket_sizes, &partition_row_counts]() {
            return build_single_gpu_gpu_aware_custom_candidate(custom_plan, num_partitions, edge_bucket_sizes, partition_row_counts);
        });
        factories.emplace_back([&custom_plan, &edge_bucket_sizes, &partition_row_counts]() {
            return build_reversed_microstate_candidate(custom_plan, edge_bucket_sizes, partition_row_counts);
        });
        factories.emplace_back([num_partitions, buffer_capacity, randomly_assign_edge_buckets, &edge_bucket_sizes, &partition_row_counts]() {
            return compileCustomStateflowPlan(num_partitions, buffer_capacity, !randomly_assign_edge_buckets,
                                              edge_bucket_sizes, partition_row_counts);
        });
    }
    if (allow_hybrid_cover) {
        factories.emplace_back([num_partitions, buffer_capacity, &edge_bucket_sizes, &partition_row_counts]() {
            return compileHybridCoverStateflowPlanVariant(num_partitions, buffer_capacity, HybridCoverVariant::LEGACY_ROTATED,
                                                          edge_bucket_sizes, partition_row_counts);
        });
        factories.emplace_back([num_partitions, buffer_capacity, &edge_bucket_sizes, &partition_row_counts]() {
            return compileHybridCoverStateflowPlanVariant(num_partitions, buffer_capacity, HybridCoverVariant::NATURAL,
                                                          edge_bucket_sizes, partition_row_counts);
        });
        factories.emplace_back([num_partitions, buffer_capacity, &edge_bucket_sizes, &partition_row_counts]() {
            return compileHybridCoverStateflowPlanVariant(num_partitions, buffer_capacity, HybridCoverVariant::REVERSED,
                                                          edge_bucket_sizes, partition_row_counts);
        });
    }

    for (auto &factory : factories) {
        append_candidate(factory());
    }

    return candidates;
}

StateflowPlan compileSingleGpuStateflowPlan(int num_partitions,
                                            int buffer_capacity,
                                            bool randomly_assign_edge_buckets,
                                            const std::vector<int64_t> &edge_bucket_sizes,
                                            bool allow_hybrid_cover,
                                            const std::vector<int64_t> &partition_row_counts) {
    std::vector<StateflowPlan> candidates = enumerateSingleGpuStateflowPlans(num_partitions, buffer_capacity,
                                                                             randomly_assign_edge_buckets, edge_bucket_sizes,
                                                                             allow_hybrid_cover, partition_row_counts);

    if (candidates.empty()) {
        return StateflowPlan{};
    }

    const char *forced_family_env = std::getenv("GEGE_STATEFLOW_FORCE_FAMILY");
    const char *forced_variant_env = std::getenv("GEGE_STATEFLOW_FORCE_VARIANT");
    const std::string forced_family = forced_family_env != nullptr ? std::string(forced_family_env) : std::string();
    const std::string forced_variant = forced_variant_env != nullptr ? std::string(forced_variant_env) : std::string();
    std::vector<StateflowPlan> filtered_candidates;
    if (!forced_family.empty() || !forced_variant.empty()) {
        for (const auto &candidate : candidates) {
            if (!force_matches_family(candidate, forced_family) || !force_matches_variant(candidate, forced_variant)) {
                continue;
            }
            filtered_candidates.emplace_back(candidate);
        }
        if (filtered_candidates.empty()) {
            SPDLOG_WARN("Stateflow force override family='{}' variant='{}' matched no single-GPU candidates; ignoring override",
                        forced_family, forced_variant);
        } else {
            SPDLOG_INFO("Stateflow force override family='{}' variant='{}' matched {} candidate(s)",
                        forced_family, forced_variant, filtered_candidates.size());
            candidates = std::move(filtered_candidates);
        }
    }

    auto best_it = std::min_element(candidates.begin(), candidates.end(), stateflow_plan_less);

    SPDLOG_INFO("Stateflow planner selected family={} cost={:.3f} loads={} boundaries={} max_overlap={} {}",
                stateflow_plan_name(*best_it), best_it->estimated_cost, best_it->total_partition_loads,
                best_it->boundary_count, best_it->max_overlap, stateflow_cost_breakdown_string(*best_it));
    log_stateflow_plan_summary(*best_it, "Stateflow single-GPU selected");
    return *best_it;
}

std::vector<StateflowPlan> enumerateMultiGpuStateflowPlans(const vector<torch::Tensor> &buffer_states,
                                                           const vector<torch::Tensor> &edge_buckets_per_buffer,
                                                           int active_devices,
                                                           const std::vector<int64_t> &edge_bucket_sizes,
                                                           const std::vector<int64_t> &partition_row_counts,
                                                           const PlanEmbeddingLayout &layout) {
    if (active_devices <= 1 || buffer_states.empty() || edge_buckets_per_buffer.size() != buffer_states.size()) {
        return {};
    }

    std::vector<StateflowPlan> candidates;

    auto grouped_permutation = getDisjointBufferStatePermutation(buffer_states, active_devices);
    StateflowPlan grouped_plan =
        build_multi_gpu_stateflow_plan_from_permutation(buffer_states, edge_buckets_per_buffer, grouped_permutation, active_devices,
                                                        PlanVariant::MULTI_GPU_DISJOINT_ROUNDS, edge_bucket_sizes,
                                                        partition_row_counts, layout);
    if (stateflow_plan_valid(grouped_plan) && validateStateflowPlanExactSemantics(grouped_plan)) {
        score_stateflow_plan(grouped_plan, edge_bucket_sizes, layout);
        SPDLOG_DEBUG(
            "Stateflow multi-GPU candidate policy=disjoint_rounds cost={:.3f} rounds={} loads={} boundaries={} "
            "directed_buckets={} estimated_bucket_edges={} overlap_hist={}",
            grouped_plan.estimated_cost, grouped_plan.total_superstates, grouped_plan.total_partition_loads,
            grouped_plan.boundary_count, grouped_plan.total_bucket_assignments, grouped_plan.estimated_bucket_edges,
            overlap_histogram_string(grouped_plan));
        candidates.emplace_back(std::move(grouped_plan));
    }

    auto lane_matched_permutation =
        getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, active_devices, partition_row_counts, layout);
    StateflowPlan lane_matched_plan = build_multi_gpu_stateflow_plan_from_permutation(buffer_states, edge_buckets_per_buffer,
                                                                                      lane_matched_permutation, active_devices,
                                                                                      PlanVariant::MULTI_GPU_LANE_MATCHED,
                                                                                      edge_bucket_sizes, partition_row_counts, layout);
    if (stateflow_plan_valid(lane_matched_plan) && validateStateflowPlanExactSemantics(lane_matched_plan)) {
        score_stateflow_plan(lane_matched_plan, edge_bucket_sizes, layout);
        SPDLOG_DEBUG(
            "Stateflow multi-GPU candidate policy=lane_matched cost={:.3f} rounds={} loads={} boundaries={} "
            "directed_buckets={} estimated_bucket_edges={} overlap_hist={}",
            lane_matched_plan.estimated_cost, lane_matched_plan.total_superstates, lane_matched_plan.total_partition_loads,
            lane_matched_plan.boundary_count, lane_matched_plan.total_bucket_assignments, lane_matched_plan.estimated_bucket_edges,
            overlap_histogram_string(lane_matched_plan));
        candidates.emplace_back(std::move(lane_matched_plan));
    }

    return candidates;
}

StateflowPlan compileMultiGpuStateflowPlan(const vector<torch::Tensor> &buffer_states,
                                           const vector<torch::Tensor> &edge_buckets_per_buffer,
                                           int active_devices,
                                           const std::vector<int64_t> &edge_bucket_sizes,
                                           const std::vector<int64_t> &partition_row_counts,
                                           const PlanEmbeddingLayout &layout) {
    std::vector<StateflowPlan> candidates =
        enumerateMultiGpuStateflowPlans(buffer_states, edge_buckets_per_buffer, active_devices, edge_bucket_sizes, partition_row_counts, layout);

    if (candidates.empty()) {
        return StateflowPlan{};
    }

    const char *forced_family_env = std::getenv("GEGE_STATEFLOW_FORCE_FAMILY");
    const char *forced_variant_env = std::getenv("GEGE_STATEFLOW_FORCE_VARIANT");
    const std::string forced_family = forced_family_env != nullptr ? std::string(forced_family_env) : std::string();
    const std::string forced_variant = forced_variant_env != nullptr ? std::string(forced_variant_env) : std::string();
    std::vector<StateflowPlan> filtered_candidates;
    if (!forced_family.empty() || !forced_variant.empty()) {
        for (const auto &candidate : candidates) {
            if (!force_matches_family(candidate, forced_family) || !force_matches_variant(candidate, forced_variant)) {
                continue;
            }
            filtered_candidates.emplace_back(candidate);
        }
        if (filtered_candidates.empty()) {
            SPDLOG_WARN("Stateflow force override family='{}' variant='{}' matched no multi-GPU candidates; ignoring override",
                        forced_family, forced_variant);
        } else {
            SPDLOG_INFO("Stateflow force override family='{}' variant='{}' matched {} multi-GPU candidate(s)",
                        forced_family, forced_variant, filtered_candidates.size());
            candidates = std::move(filtered_candidates);
        }
    }

    auto best_it = std::min_element(candidates.begin(), candidates.end(), [](const StateflowPlan &lhs, const StateflowPlan &rhs) {
        if (lhs.estimated_cost != rhs.estimated_cost) {
            return lhs.estimated_cost < rhs.estimated_cost;
        }
        if (lhs.total_partition_loads != rhs.total_partition_loads) {
            return lhs.total_partition_loads < rhs.total_partition_loads;
        }
        if (lhs.max_overlap != rhs.max_overlap) {
            return lhs.max_overlap > rhs.max_overlap;
        }
        return lhs.total_superstates < rhs.total_superstates;
    });

    SPDLOG_INFO("Stateflow multi-GPU selected rounds={} cost={:.3f} loads={} boundaries={} max_overlap={} handoffs={} active_devices={}",
                best_it->total_superstates, best_it->estimated_cost, best_it->total_partition_loads,
                best_it->boundary_count, best_it->max_overlap, best_it->total_handoffs, active_devices);
    log_stateflow_plan_summary(*best_it, "Stateflow multi-GPU selected");
    if (!validateStateflowPlanExactSemantics(*best_it)) {
        SPDLOG_WARN("Stateflow multi-GPU plan failed exact-semantics validation");
    }
    return *best_it;
}

MultiGpuSchedule projectStateflowPlanToMultiGpuSchedule(const StateflowPlan &plan) {
    MultiGpuSchedule schedule;
    schedule.buffer_states_per_device.resize(plan.lanes.size());
    schedule.edge_buckets_per_device.resize(plan.lanes.size());

    for (std::size_t lane_idx = 0; lane_idx < plan.lanes.size(); lane_idx++) {
        const auto &lane = plan.lanes[lane_idx];
        std::vector<std::vector<int>> buffer_states;
        std::vector<std::vector<std::pair<int, int>>> edge_buckets;
        buffer_states.reserve(lane.microstates.size());
        edge_buckets.reserve(lane.microstates.size());
        for (const auto &microstate : lane.microstates) {
            buffer_states.emplace_back(microstate.resident_partitions);
            edge_buckets.emplace_back(microstate.edge_buckets);
        }
        auto tensors = convertEdgeBucketOrderToTensors(std::move(buffer_states), std::move(edge_buckets));
        schedule.buffer_states_per_device[lane_idx] = std::move(std::get<0>(tensors));
        schedule.edge_buckets_per_device[lane_idx] = std::move(std::get<1>(tensors));
    }

    for (const auto &handoff : plan.cross_lane_handoffs) {
        PeerHandoffDescriptor descriptor;
        descriptor.src_lane_id = handoff.src_lane_id;
        descriptor.dst_lane_id = handoff.dst_lane_id;
        descriptor.src_microstate_id = handoff.src_microstate_id;
        descriptor.dst_microstate_id = handoff.dst_microstate_id;
        descriptor.bytes = handoff.peer_bytes;
        if (!handoff.slot_mapping.empty()) {
            descriptor.src_slot_id = handoff.slot_mapping.front().first;
            descriptor.dst_slot_id = handoff.slot_mapping.front().second;
        }
        for (const auto &lane : plan.lanes) {
            if (lane.lane_id != handoff.dst_lane_id) {
                continue;
            }
            for (const auto &microstate : lane.microstates) {
                if (microstate.microstate_id != handoff.dst_microstate_id) {
                    continue;
                }
                descriptor.round_idx = microstate.superstate_id;
                if (descriptor.dst_slot_id >= 0 && descriptor.dst_slot_id < static_cast<int>(microstate.resident_partitions.size())) {
                    descriptor.partition_id = microstate.resident_partitions[descriptor.dst_slot_id];
                }
                break;
            }
            break;
        }
        schedule.peer_handoffs.emplace_back(std::move(descriptor));
    }

    return schedule;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> projectStateflowPlanToLegacySchedule(const StateflowPlan &plan) {
    std::vector<std::vector<int>> buffer_states;
    std::vector<std::vector<std::pair<int, int>>> edge_buckets_per_buffer;

    std::size_t max_lane_microstates = 0;
    for (const auto &lane : plan.lanes) {
        max_lane_microstates = std::max(max_lane_microstates, lane.microstates.size());
    }

    for (std::size_t round_idx = 0; round_idx < max_lane_microstates; round_idx++) {
        for (const auto &lane : plan.lanes) {
            if (round_idx >= lane.microstates.size()) {
                continue;
            }
            const auto &microstate = lane.microstates[round_idx];
            buffer_states.emplace_back(microstate.resident_partitions);
            edge_buckets_per_buffer.emplace_back(microstate.edge_buckets);
        }
    }

    return convertEdgeBucketOrderToTensors(std::move(buffer_states), std::move(edge_buckets_per_buffer));
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> stateflowPlanToTensorOrdering(const StateflowPlan &plan) {
    return projectStateflowPlanToLegacySchedule(plan);
}

namespace {

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getHybridCoverEdgeBucketOrdering(int num_partitions, int buffer_capacity) {
    StateflowPlan plan = compileHybridCoverStateflowPlan(num_partitions, buffer_capacity);
    if (!stateflow_plan_valid(plan)) {
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    int64_t single_partition_rotations = 0;
    int64_t boundary_rotations = 0;
    for (const auto &microstate : plan.lanes[0].microstates) {
        if (microstate.microstate_id == 0) {
            continue;
        }
        if (microstate.overlap_with_prev == plan.buffer_capacity - 1) {
            single_partition_rotations++;
        } else {
            boundary_rotations++;
        }
    }

    SPDLOG_INFO(
        "Generating {} Ordering: superstates={} microstates={} directed_buckets={} total_partition_loads={} "
        "single_partition_rotations={} boundary_rotations={} max_overlap={} total_handoffs={}",
        stateflow_plan_name(plan), plan.total_superstates, plan.total_microstates, plan.total_bucket_assignments,
        plan.total_partition_loads, single_partition_rotations, boundary_rotations, plan.max_overlap,
        plan.total_handoffs);
    log_stateflow_plan_summary(plan, "Stateflow plan");
    if (!validateStateflowPlanExactSemantics(plan)) {
        SPDLOG_WARN("Stateflow plan {} failed exact-semantics validation", planFamilyName(plan.family));
    }

    return projectStateflowPlanToLegacySchedule(plan);
}

int64_t mark_state_buckets_covered(const std::vector<int> &state,
                                   std::vector<uint8_t> &covered,
                                   int num_partitions) {
    int64_t newly_covered = 0;
    for (auto src_part : state) {
        for (auto dst_part : state) {
            auto bucket_idx = src_part * num_partitions + dst_part;
            if (covered[bucket_idx] == 0) {
                covered[bucket_idx] = 1;
                newly_covered++;
            }
        }
    }
    return newly_covered;
}

std::vector<std::vector<int>> reorder_states_for_max_overlap(const std::vector<std::vector<int>> &states) {
    const int num_states = static_cast<int>(states.size());
    if (num_states <= 2) {
        return states;
    }

    std::vector<std::vector<int>> overlap(num_states, std::vector<int>(num_states, 0));
    for (int lhs = 0; lhs < num_states; lhs++) {
        for (int rhs = 0; rhs < num_states; rhs++) {
            if (lhs == rhs) {
                continue;
            }
            overlap[lhs][rhs] = state_overlap_count(states[lhs], states[rhs]);
        }
    }

    std::vector<int> order;
    order.reserve(num_states);

    if (num_states <= 18) {
        const int64_t num_masks = 1LL << num_states;
        constexpr int kMinScore = std::numeric_limits<int>::min() / 4;
        std::vector<int> dp(num_masks * num_states, kMinScore);
        std::vector<int16_t> parent(num_masks * num_states, -1);

        for (int state_idx = 0; state_idx < num_states; state_idx++) {
            dp[(1LL << state_idx) * num_states + state_idx] = 0;
        }

        for (int64_t mask = 1; mask < num_masks; mask++) {
            for (int last = 0; last < num_states; last++) {
                int current_score = dp[mask * num_states + last];
                if (current_score == kMinScore) {
                    continue;
                }
                for (int next = 0; next < num_states; next++) {
                    if ((mask & (1LL << next)) != 0) {
                        continue;
                    }
                    int64_t next_mask = mask | (1LL << next);
                    int next_score = current_score + overlap[last][next];
                    int64_t next_offset = next_mask * num_states + next;
                    if (next_score > dp[next_offset]) {
                        dp[next_offset] = next_score;
                        parent[next_offset] = static_cast<int16_t>(last);
                    }
                }
            }
        }

        const int64_t full_mask = num_masks - 1;
        int best_last = 0;
        int best_score = kMinScore;
        for (int last = 0; last < num_states; last++) {
            int score = dp[full_mask * num_states + last];
            if (score > best_score) {
                best_score = score;
                best_last = last;
            }
        }

        order.assign(num_states, -1);
        int64_t mask = full_mask;
        int current = best_last;
        for (int pos = num_states - 1; pos >= 0; pos--) {
            order[pos] = current;
            int64_t offset = mask * num_states + current;
            int previous = parent[offset];
            mask ^= (1LL << current);
            current = previous;
        }
    } else {
        std::vector<uint8_t> used(num_states, 0);
        order.emplace_back(0);
        used[0] = 1;
        while (static_cast<int>(order.size()) < num_states) {
            int previous = order.back();
            int best_next = -1;
            int best_overlap = -1;
            for (int candidate = 0; candidate < num_states; candidate++) {
                if (used[candidate]) {
                    continue;
                }
                if (overlap[previous][candidate] > best_overlap) {
                    best_overlap = overlap[previous][candidate];
                    best_next = candidate;
                }
            }
            if (best_next < 0) {
                break;
            }
            used[best_next] = 1;
            order.emplace_back(best_next);
        }
    }

    if (static_cast<int>(order.size()) != num_states) {
        return states;
    }

    std::vector<std::vector<int>> reordered;
    reordered.reserve(states.size());
    for (auto state_idx : order) {
        reordered.emplace_back(states[state_idx]);
    }
    return reordered;
}

}  // namespace

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getGreedyCoverEdgeBucketOrdering(int num_partitions,
                                                                                          int buffer_capacity) {
    if (buffer_capacity <= 0 || buffer_capacity > num_partitions) {
        SPDLOG_WARN("Invalid greedy cover ordering parameters: num_partitions={} buffer_capacity={}",
                    num_partitions, buffer_capacity);
        return convertEdgeBucketOrderToTensors({}, {});
    }

    std::vector<std::vector<int>> candidates;
    std::vector<int> current;
    build_greedy_cover_candidates(num_partitions, buffer_capacity, 0, current, candidates);
    if (candidates.empty()) {
        return convertEdgeBucketOrderToTensors({}, {});
    }

    std::vector<uint8_t> covered(num_partitions * num_partitions, 0);
    int64_t covered_count = 0;
    const int64_t total_buckets = static_cast<int64_t>(num_partitions) * static_cast<int64_t>(num_partitions);
    std::vector<std::vector<int>> buffer_states;
    buffer_states.reserve(candidates.size());

    while (covered_count < total_buckets) {
        int64_t best_gain = -1;
        int best_overlap = -1;
        int64_t best_balance_score = std::numeric_limits<int64_t>::max();
        int best_idx = -1;

        for (int candidate_idx = 0; candidate_idx < static_cast<int>(candidates.size()); candidate_idx++) {
            const auto &candidate = candidates[candidate_idx];
            int64_t gain = count_uncovered_buckets_for_state(candidate, covered, num_partitions);
            if (gain == 0) {
                continue;
            }

            int overlap = buffer_states.empty() ? 0 : state_overlap_count(buffer_states.back(), candidate);
            int64_t balance_score = 0;
            for (auto partition : candidate) {
                balance_score += partition;
            }

            if (gain > best_gain ||
                (gain == best_gain && overlap > best_overlap) ||
                (gain == best_gain && overlap == best_overlap && balance_score < best_balance_score)) {
                best_gain = gain;
                best_overlap = overlap;
                best_balance_score = balance_score;
                best_idx = candidate_idx;
            }
        }

        if (best_idx < 0) {
            SPDLOG_WARN("Greedy cover ordering stopped early: covered={} total={}", covered_count, total_buckets);
            break;
        }

        auto state = candidates[best_idx];
        covered_count += mark_state_buckets_covered(state, covered, num_partitions);
        buffer_states.emplace_back(std::move(state));
    }

    double pre_reorder_retained_avg = 0.0;
    if (buffer_states.size() > 1) {
        int64_t pre_reorder_retained_total = 0;
        for (std::size_t idx = 1; idx < buffer_states.size(); idx++) {
            pre_reorder_retained_total += state_overlap_count(buffer_states[idx - 1], buffer_states[idx]);
        }
        pre_reorder_retained_avg =
            static_cast<double>(pre_reorder_retained_total) / static_cast<double>(buffer_states.size() - 1);
    }

    buffer_states = reorder_states_for_max_overlap(buffer_states);

    int64_t retained_total = 0;
    for (std::size_t idx = 1; idx < buffer_states.size(); idx++) {
        retained_total += state_overlap_count(buffer_states[idx - 1], buffer_states[idx]);
    }
    double retained_avg = buffer_states.size() > 1
                              ? static_cast<double>(retained_total) / static_cast<double>(buffer_states.size() - 1)
                              : 0.0;

    auto edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    int64_t assigned_buckets = 0;
    for (auto &buckets : edge_buckets_per_buffer) {
        assigned_buckets += static_cast<int64_t>(buckets.size());
    }
    SPDLOG_INFO("Generating GREEDY_COVER Ordering: states={} assigned_buckets={} retained_avg={:.3f} pre_reorder_retained_avg={:.3f}",
                buffer_states.size(), assigned_buckets, retained_avg, pre_reorder_retained_avg);
    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

vector<vector<std::pair<int, int>>> randomlyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions) {
    // get edge buckets from buffer states
    Indices all_partitions = torch::arange(num_partitions, torch::kInt32);
    torch::Tensor left_col = all_partitions.repeat_interleave(num_partitions);
    torch::Tensor right_col = all_partitions.repeat({num_partitions});
    torch::Tensor all_buckets = torch::stack({left_col, right_col}, 1);
    auto all_buckets_accessor = all_buckets.accessor<int32_t, 2>();

    int num_buffers = buffer_states.size();
    int buffer_size = buffer_states[0].size();
    int num_buckets = all_buckets.size(0);

    torch::Tensor choices = torch::zeros({num_buckets, num_buffers}, torch::kInt32);
    int32_t *choices_mem = choices.data_ptr<int32_t>();

#pragma omp parallel for
    for (int i = 0; i < num_buffers; i++) {
        for (int j = 0; j < buffer_size; j++) {
            for (int k = j; k < buffer_size; k++) {
                int src_part = buffer_states[i][j];
                int dst_part = buffer_states[i][k];
                *(choices_mem + (src_part * num_partitions + dst_part) * num_buffers + i) = 1;
                *(choices_mem + (dst_part * num_partitions + src_part) * num_buffers + i) = 1;
            }
        }
    }

    torch::Tensor pick = torch::zeros({num_buckets}, torch::kInt32);
    torch::Tensor pick_one_hot = torch::zeros({num_buckets, num_buffers}, torch::kInt32);
    int32_t *pick_mem = pick.data_ptr<int32_t>();
    int32_t *pick_one_hot_mem = pick_one_hot.data_ptr<int32_t>();
    auto pick_accessor = pick.accessor<int32_t, 1>();

    // setup seeds
    unsigned int num_threads = 1;
#ifdef GEGE_OMP
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }
#endif
    std::vector<unsigned int> tid_seeds(num_threads);

    for (int i = 0; i < num_threads; i++) {
        tid_seeds[i] = rand();
    }

#pragma omp parallel
    {
#ifdef GEGE_OMP
        unsigned int seed = tid_seeds[omp_get_thread_num()];
#else
        unsigned int seed = tid_seeds[0];
#endif

#pragma omp for
        for (int i = 0; i < num_buckets; i++) {
            torch::Tensor buffer_choices = torch::nonzero(choices[i]);
            buffer_choices = torch::reshape(buffer_choices, {buffer_choices.size(0)});
            int32_t buffer_choice = buffer_choices[rand_r(&seed) % buffer_choices.size(0)].item<int32_t>();

            int32_t src_part = all_buckets_accessor[i][0];
            int32_t dst_part = all_buckets_accessor[i][1];
            *(pick_mem + (src_part * num_partitions + dst_part)) = buffer_choice;
            *(pick_one_hot_mem + (src_part * num_partitions + dst_part) * num_buffers + buffer_choice) = 1;
        }
    }

    torch::Tensor num_edge_buckets_per_buffer = torch::sum(pick_one_hot, 0);

    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
        edge_buckets_per_buffer[i] = vector<std::pair<int, int>>(num_edge_buckets_per_buffer[i].item<int>());
    }

    vector<int> indices(num_buffers, 0);
    for (int i = 0; i < num_buckets; i++) {
        int32_t src_part = all_buckets_accessor[i][0];
        int32_t dst_part = all_buckets_accessor[i][1];
        std::pair<int, int> pair = std::make_pair(src_part, dst_part);

        int32_t buffer_choice = pick_accessor[i];

        edge_buckets_per_buffer[buffer_choice][indices[buffer_choice]] = pair;
        indices[buffer_choice] += 1;
    }

    return edge_buckets_per_buffer;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getTwoLevelBetaOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio,
                                                                                 int num_cache_partitions, bool randomly_assign_edge_buckets) {
    int coarse_num_partitions = num_partitions / fine_to_coarse_ratio;
    int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;

    coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
    coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;

    vector<vector<int>> coarse_buffer_states = getBetaOrderingHelper(coarse_num_partitions, coarse_buffer_capacity);

    int cached_fine_partitions = num_cache_partitions * fine_to_coarse_ratio;
    torch::Tensor fine_to_coarse_map = torch::arange(cached_fine_partitions, torch::kInt32);
    fine_to_coarse_map = torch::cat({fine_to_coarse_map, torch::randperm(num_partitions - cached_fine_partitions, torch::kInt32) + cached_fine_partitions});
    int *data_ptr_ = (int *)fine_to_coarse_map.data_ptr();

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        for (int j = 0; j < coarse_buffer_states[i].size(); j++) {
            coarse_buffer_states[i][j] += num_cache_partitions;
        }
        for (int j = 0; j < num_cache_partitions; j++) {
            coarse_buffer_states[i].emplace_back(j);
        }
    }

    // convert to fine buffer states
    vector<vector<int>> buffer_states;

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        vector<int> fine_buffer_state(buffer_capacity, 0);
        for (int j = 0; j < coarse_buffer_states[i].size(); j++) {
            int *start = (int *)data_ptr_ + coarse_buffer_states[i][j] * fine_to_coarse_ratio;
            int *end = (int *)data_ptr_ + (coarse_buffer_states[i][j] + 1) * fine_to_coarse_ratio;
            vector<int> fine_partitions = vector<int>(start, end);

            for (int k = j * fine_to_coarse_ratio; k < (j + 1) * fine_to_coarse_ratio; k++) {
                fine_buffer_state[k] = fine_partitions[k - j * fine_to_coarse_ratio];
            }
        }

        buffer_states.emplace_back(fine_buffer_state);
    }

    // assign edge buckets
    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer;
    if (randomly_assign_edge_buckets) {
        edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    } else {
        edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    }

    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDispersedNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                           int buffer_capacity, int fine_to_coarse_ratio,
                                                                                           int num_cache_partitions) {
    int coarse_num_partitions = num_partitions / fine_to_coarse_ratio;
    int coarse_buffer_capacity = buffer_capacity / fine_to_coarse_ratio;

    coarse_num_partitions = coarse_num_partitions - num_cache_partitions;
    coarse_buffer_capacity = coarse_buffer_capacity - num_cache_partitions;

    // create coarse buffer states
    vector<torch::Tensor> coarse_buffer_states;
    Indices all_coarse_partitions = torch::randperm(coarse_num_partitions, torch::kInt32);
    Indices in_buffer = all_coarse_partitions.narrow(0, 0, coarse_buffer_capacity);
    Indices on_disk = all_coarse_partitions.narrow(0, coarse_buffer_capacity, coarse_num_partitions - coarse_buffer_capacity);
    coarse_buffer_states.emplace_back(in_buffer);

    while (on_disk.size(0) > 0) {
        in_buffer = in_buffer.index_select(0, torch::randperm(in_buffer.size(0), torch::kInt64));
        on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));

        in_buffer[-1] = on_disk[0];
        coarse_buffer_states.emplace_back(in_buffer);
        on_disk = on_disk.narrow(0, 1, on_disk.size(0) - 1);
    }

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        coarse_buffer_states[i] =
            torch::cat({coarse_buffer_states[i] + num_cache_partitions, torch::arange(num_cache_partitions, coarse_buffer_states[i].options())});
    }

    // convert to fine buffer states
    torch::Tensor fine_to_coarse_map = torch::randperm(num_partitions, torch::kInt32);
    int *data_ptr_ = (int *)fine_to_coarse_map.data_ptr();

    vector<torch::Tensor> buffer_states;

    for (int i = 0; i < coarse_buffer_states.size(); i++) {
        vector<int> fine_buffer_state(buffer_capacity, 0);
        torch::Tensor coarse_buffer_state = coarse_buffer_states[i];
        auto coarse_buffer_state_accessor = coarse_buffer_state.accessor<int32_t, 1>();

        for (int j = 0; j < coarse_buffer_state.size(0); j++) {
            int *start = (int *)data_ptr_ + coarse_buffer_state_accessor[j] * fine_to_coarse_ratio;
            int *end = (int *)data_ptr_ + (coarse_buffer_state_accessor[j] + 1) * fine_to_coarse_ratio;
            vector<int> fine_partitions = vector<int>(start, end);

            for (int k = j * fine_to_coarse_ratio; k < (j + 1) * fine_to_coarse_ratio; k++) {
                fine_buffer_state[k] = fine_partitions[k - j * fine_to_coarse_ratio];
            }
        }

        buffer_states.emplace_back(torch::from_blob(fine_buffer_state.data(), {(int)fine_buffer_state.size()}, torch::kInt32).clone());
    }

    // randomly assign train nodes to buffers

    int64_t partition_size = ceil((double)total_num_nodes / num_partitions);
    torch::Tensor train_nodes_partition =
        torch::floor(train_nodes.to(torch::kFloat64).div(static_cast<double>(partition_size))).to(torch::kInt32);

    std::vector<std::vector<int>> partition_buffer_states(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        for (int j = 0; j < buffer_states.size(); j++) {
            bool partition_in_buffer = false;
            auto buffer_state_accessor = buffer_states[j].accessor<int32_t, 1>();

            for (int k = 0; k < buffer_capacity; k++) {
                if (buffer_state_accessor[k] == i) {
                    partition_in_buffer = true;
                    break;
                }
            }
            if (partition_in_buffer) {
                partition_buffer_states[i].emplace_back(j);
            }
        }
    }

    torch::Tensor train_nodes_buffer_choice = torch::zeros_like(train_nodes);
    std::vector<torch::Tensor> train_nodes_per_buffer(buffer_states.size());
    auto train_nodes_partition_accessor = train_nodes_partition.accessor<int32_t, 1>();  // todo

    for (int i = 0; i < train_nodes.size(0); i++) {
        int partition_id = train_nodes_partition_accessor[i];
        int rand_id = rand() % partition_buffer_states[partition_id].size();
        train_nodes_buffer_choice[i] = partition_buffer_states[partition_id][rand_id];
    }

    for (int i = 0; i < buffer_states.size(); i++) {
        train_nodes_per_buffer[i] = train_nodes.masked_select(train_nodes_buffer_choice == i);
    }

    return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getSequentialNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                            int buffer_capacity) {
    int64_t partition_size = ceil((double)total_num_nodes / num_partitions);
    torch::Tensor train_nodes_partition =
        torch::floor(train_nodes.to(torch::kFloat64).div(static_cast<double>(partition_size))).to(torch::kInt32);

    int32_t max_train_partition = torch::max(train_nodes_partition).item<int32_t>();
    int32_t num_train_partitions = max_train_partition + 1;
    SPDLOG_INFO("Num Train Partitions: {}", num_train_partitions);

    vector<torch::Tensor> buffer_states;
    Indices in_buffer = torch::arange(num_train_partitions, torch::kInt32);
    Indices on_disk = torch::arange(num_train_partitions, num_partitions, torch::kInt32);
    on_disk = on_disk.index_select(0, torch::randperm(on_disk.size(0), torch::kInt64));
    on_disk = on_disk.narrow(0, 0, buffer_capacity - num_train_partitions);

    buffer_states.emplace_back(torch::cat({in_buffer, on_disk}));

    std::vector<torch::Tensor> train_nodes_per_buffer;
    train_nodes_per_buffer.emplace_back(train_nodes.clone());

    return std::forward_as_tuple(buffer_states, train_nodes_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomNodePartitionOrdering() {
    SPDLOG_ERROR("Not implemented");
    std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> ret;
    return ret;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getAccessAwareCustomEdgeBucketOrdering(int num_partitions, int buffer_capacity, int active_devices) {
    SPDLOG_INFO("Generating access-aware CUSTOM Ordering");
    return generate_access_aware_states(num_partitions, buffer_capacity, active_devices);
}

std::vector<int64_t> getDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states, int active_devices) {
    auto groups = build_disjoint_groups(buffer_states, active_devices);
    if (groups.empty()) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());
    for (auto &group : groups) {
        for (auto state_idx : group) {
            permutation.emplace_back(state_idx);
        }
    }

    if (permutation.size() != buffer_states.size()) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    return permutation;
}

std::vector<int64_t> getAccessAwareDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states,
                                                                  const vector<torch::Tensor>& edge_buckets_per_buffer,
                                                                  int active_devices,
                                                                  const std::vector<int64_t> &partition_row_counts,
                                                                  const PlanEmbeddingLayout &layout) {
    if (active_devices <= 1 || buffer_states.size() <= 1 || edge_buckets_per_buffer.size() != buffer_states.size()) {
        return getDisjointBufferStatePermutation(buffer_states, active_devices);
    }

    std::vector<std::vector<int64_t>> state_partitions;
    state_partitions.reserve(buffer_states.size());
    for (auto &state : buffer_states) {
        state_partitions.emplace_back(tensor_to_partitions(state));
    }

    std::vector<std::vector<bool>> compatible(buffer_states.size(), std::vector<bool>(buffer_states.size(), false));
    for (std::size_t i = 0; i < buffer_states.size(); i++) {
        compatible[i][i] = true;
        for (std::size_t j = i + 1; j < buffer_states.size(); j++) {
            bool disjoint = states_disjoint(state_partitions[i], state_partitions[j]);
            compatible[i][j] = disjoint;
            compatible[j][i] = disjoint;
        }
    }

    auto summaries = build_state_access_summaries(buffer_states, edge_buckets_per_buffer);

    std::vector<int64_t> remaining(buffer_states.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());

    std::vector<int64_t> previous_group;
    LaneMatchCostConfig lane_match_cfg = lane_match_cost_config_from_env();
    LaneMatchSolver solver = stateflow_lane_match_solver();
    const int64_t bytes_per_row = plan_embedding_bytes_per_row(layout);
    while (!remaining.empty()) {
        int target_group_size = std::min<int>(active_devices, remaining.size());
        std::vector<int64_t> ordered_group;
        std::vector<int64_t> chosen_states;
        if (solver == LaneMatchSolver::OPTIMAL2 && target_group_size >= 2 &&
            previous_group.size() == static_cast<std::size_t>(target_group_size)) {
            auto best_group =
                get_best_cost_aware_lane_group(compatible, remaining, previous_group, summaries, partition_row_counts, lane_match_cfg,
                                               bytes_per_row, target_group_size);
            ordered_group = std::move(best_group.ordered_group);
            chosen_states = std::move(best_group.chosen_states);
        } else {
            GroupSearchResult best_group;
            std::vector<int64_t> current_group;
            current_group.reserve(target_group_size);
            search_best_disjoint_group(compatible, remaining, previous_group, summaries, target_group_size, 0, current_group, best_group);
            ordered_group = std::move(best_group.ordered_group);
            chosen_states = std::move(best_group.chosen_states);
        }

        if (chosen_states.empty()) {
            return getDisjointBufferStatePermutation(buffer_states, active_devices);
        }

        for (auto state_idx : ordered_group) {
            permutation.emplace_back(state_idx);
        }

        previous_group = ordered_group;
        std::vector<int64_t> next_remaining;
        next_remaining.reserve(remaining.size() - chosen_states.size());
        for (auto state_idx : remaining) {
            if (std::find(chosen_states.begin(), chosen_states.end(), state_idx) == chosen_states.end()) {
                next_remaining.emplace_back(state_idx);
            }
        }
        remaining = std::move(next_remaining);
    }

    if (permutation.size() != buffer_states.size()) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
    }

    return permutation;
}

int32_t pow(int32_t a, int32_t x)
{
    int32_t ans = 1, temp = a;
    while(x) {
        if (x & 1) {
            ans = ans * temp;
        }
        temp *= temp;
        x >>= 1;
    }
    return ans;
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomEdgeBucketOrdering(int num_partitions, int buffer_capacity, bool randomly_assign_edge_buckets)
{
    assert(buffer_capacity == 4);
    int32_t sub_chunk_per_perm = num_partitions / buffer_capacity;
    int32_t log2l = 0;

    while(pow(2, log2l) < num_partitions) {
        log2l += 1;
    }

    assert(pow(2, log2l) == num_partitions);

    std::vector<std::vector<std::vector<int>>> offset_supergroup = {
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}},
        {{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}},
        {{0, 2, 3, 1}, {1, 3, 2, 0}, {2, 0, 1, 3}, {3, 1, 0, 2}},
        {{0, 3, 1, 2}, {1, 2, 0, 3}, {2, 1, 3, 0}, {3, 0, 2, 1}},
    };
    std::vector<std::vector<std::vector<int>>> p = {{{0, 1, 2, 3}}};

    for (int log4l_pre = 1; log4l_pre < log2l / 2; log4l_pre ++) {
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        for (auto& s : p_pre) {
            std::vector<std::vector<int>> s_cur;
            for (int offset = 0; offset < pow(4, log4l_pre + 1); offset += pow(4, log4l_pre)) {
                for (auto& g : s) {
                    std::vector<int> g_cur;
                    for(auto& x : g) {
                        g_cur.emplace_back(x + offset);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }
        int32_t len = p_pre.size();
        for (int i = len - pow(4, log4l_pre - 1); i < len; i ++) {
            auto s = p_pre[i];
            for (auto& offset_s : offset_supergroup) {
                std::vector<std::vector<int>> s_cur;
                for (auto& g : s) {
                    for(auto& offset_g : offset_s){
                        std::vector<int> g_cur;
                        for (int j = 0; j < 4; j ++) {
                            g_cur.emplace_back(g[j] * 4 + offset_g[j]);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }
        }
    }
    std::vector<std::vector<std::vector<int>>> pairing_chunks = {
        {{0, 2}, {1, 3}},
        {{0, 3}, {1, 2}}
    };

    if (log2l % 2 == 1) {
        int32_t len_chunk = sub_chunk_per_perm;
        auto p_pre = p;
        p = std::vector<std::vector<std::vector<int>>>();
        
        for (auto& s: p_pre) {
            std::vector<std::vector<int>> s_cur;
            for(int i = 0; i < pow(2, log2l); i += pow(2, log2l - 1)) {
                for (auto& g : s) {
                    std::vector<int> g_cur;
                    for (auto& x : g) {
                        g_cur.emplace_back(x + i);
                    }
                    s_cur.emplace_back(g_cur);
                }
            }
            p.emplace_back(s_cur);
        }

        int32_t len = p_pre.size();
        for (int i = len - pow(2, log2l - 3); i < len; i ++) {
            std::vector<std::vector<int>> s = p_pre[i];
            for (auto& pairing_s : pairing_chunks) {
                std::vector<std::vector<int>> s_cur;
                for (auto& chunk_index : pairing_s) {
                    for (auto& g : s) {
                        std::vector<int> g_cur;
                        for (auto& x : g) {
                            g_cur.emplace_back(chunk_index[x / len_chunk] * len_chunk + x % len_chunk);
                        }
                        s_cur.emplace_back(g_cur);
                    }
                }
                p.emplace_back(s_cur);
            }

        }
    }
    std::vector<std::vector<int>> buffer_states;
    Indices all_partitions_map = torch::randperm(num_partitions, torch::kInt32);
    for (auto& p1 : p) {
        for(auto& p2 : p1) {
            buffer_states.emplace_back(p2);
        } 
    }
    for(int i = 0; i < buffer_states.size(); i ++){
        for(int j = 0; j < buffer_states[i].size(); j ++) {
            // std::cout << buffer_states[i][j] << " ";
            buffer_states[i][j] = all_partitions_map[buffer_states[i][j]].item<int>();
        }
    }

    Indices all_buffer_map = torch::randperm(buffer_states.size(), torch::kInt32);
    std::vector<std::vector<int>> shuffle_buffer_states;
    for (int i = 0; i < buffer_states.size(); i ++) {
        shuffle_buffer_states.push_back(buffer_states[all_buffer_map[i].item<int>()]);
    }
    buffer_states = shuffle_buffer_states;

    std::vector<std::vector<std::pair<int, int>>> edge_buckets_per_buffer;
    if (randomly_assign_edge_buckets) {
        edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    } else {
        edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    }
    // for(auto const& edge_buckets : edge_buckets_per_buffer) {
        // std::cout << edge_buckets.size() << ": ";
        // for(auto const& bucket : edge_buckets) {
        //     std::cout << "(" << bucket.first << "," << bucket.second << ") "<< " ";
        // }
        // std::cout << std::endl;
    // }
    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);

}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getOptimizedCustomEdgeBucketOrdering(int num_partitions,
                                                                                               int buffer_capacity,
                                                                                               int active_devices,
                                                                                               int batch_size,
                                                                                               const vector<int64_t> &edge_bucket_sizes) {
    if (!optimized_custom_schedule_enabled()) {
        SPDLOG_INFO("GEGE_OPTIMIZED_CUSTOM_SCHEDULE disabled; falling back to standard CUSTOM ordering");
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    if (buffer_capacity != 4 || active_devices <= 0 || active_devices > buffer_capacity) {
        SPDLOG_INFO("Optimized CUSTOM ordering currently supports buffer_capacity=4 and active_devices in [1, {}]; falling back to standard CUSTOM ordering",
                    buffer_capacity);
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    if (edge_bucket_sizes.size() != static_cast<size_t>(num_partitions * num_partitions)) {
        SPDLOG_WARN("Optimized CUSTOM ordering expected {} edge bucket sizes, found {}; falling back to standard CUSTOM ordering",
                    num_partitions * num_partitions, edge_bucket_sizes.size());
        return getCustomEdgeBucketOrdering(num_partitions, buffer_capacity, false);
    }

    return build_optimized_custom_edge_bucket_ordering(num_partitions, buffer_capacity, active_devices, batch_size, edge_bucket_sizes);
}
