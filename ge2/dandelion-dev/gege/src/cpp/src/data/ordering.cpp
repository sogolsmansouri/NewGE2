#include "common/datatypes.h"
#include "data/ordering.h"
#include "reporting/logger.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

bool bounded_greedy_cover_q4_enabled() {
    const char *raw = std::getenv("GEGE_BOUNDED_GREEDY_COVER_Q4");
    if (raw == nullptr) {
        return false;
    }

    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

bool bounded_greedy_cover_reverse_enabled() {
    const char *raw = std::getenv("GEGE_BOUNDED_GREEDY_COVER_REVERSE");
    if (raw == nullptr) {
        return false;
    }

    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE");
}

bool bounded_q4_optimal88_enabled() {
    const char *raw = std::getenv("GEGE_BOUNDED_Q4_OPTIMAL88");
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

bool stateflow_allow_peer_relay_default() {
    return stateflow_env_bool("GEGE_PARTITION_BUFFER_PEER_RELAY", "PARTITION_BUFFER_PEER_RELAY", false);
}

LaneMatchCostConfig lane_match_cost_config_from_env() {
    LaneMatchCostConfig cfg;
    cfg.peer_bandwidth_bps =
        std::max<int64_t>(1, stateflow_env_int64("GEGE_STATEFLOW_PEER_BANDWIDTH_BPS", "STATEFLOW_PEER_BANDWIDTH_BPS", 32000000000LL));
    cfg.host_bandwidth_bps =
        std::max<int64_t>(1, stateflow_env_int64("GEGE_STATEFLOW_HOST_BANDWIDTH_BPS", "STATEFLOW_HOST_BANDWIDTH_BPS", 16000000000LL));
    cfg.max_admits_per_transition =
        stateflow_env_int64("GEGE_STATEFLOW_MAX_ADMITS", "STATEFLOW_MAX_ADMITS", bounded_greedy_cover_q4_enabled() ? 3 : -1);
    if (bounded_q4_optimal88_enabled() && cfg.max_admits_per_transition >= 0 && cfg.max_admits_per_transition < 3) {
        static std::atomic<bool> warned_opt88_max_admits_clamp{false};
        bool expected = false;
        if (warned_opt88_max_admits_clamp.compare_exchange_strong(expected, true)) {
            SPDLOG_WARN("GEGE_BOUNDED_Q4_OPTIMAL88 requires max_admits>=3; clamping GEGE_STATEFLOW_MAX_ADMITS from {} to 3",
                        cfg.max_admits_per_transition);
        }
        cfg.max_admits_per_transition = 3;
    }
    cfg.imbalance_weight = stateflow_cost_env("GEGE_STATEFLOW_LANE_IMBALANCE_WEIGHT", 1.0);
    cfg.boundary_weight = stateflow_cost_env("GEGE_STATEFLOW_LANE_BOUNDARY_WEIGHT", 1.0);
    cfg.allow_peer_relay =
        stateflow_env_bool("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "STATEFLOW_ALLOW_PEER_RELAY", stateflow_allow_peer_relay_default());
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

int64_t optimal88_peer_handoff_budget(int active_devices) {
    const int64_t default_budget = active_devices == 2 ? 51 : -1;
    if (active_devices == 2) {
        return stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_2GPU_MAX_PEER_HANDOFFS", nullptr,
                                   stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MAX_PEER_HANDOFFS", nullptr,
                                                       default_budget));
    }
    return stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MULTIGPU_MAX_PEER_HANDOFFS", nullptr,
                               stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MAX_PEER_HANDOFFS", nullptr,
                                                   default_budget));
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

int64_t handoff_bytes_peer_from_lane_group(const std::vector<int64_t> &needed_partitions,
                                           const std::vector<int64_t> &already_on_lane,
                                           const std::vector<StateAccessSummary> &summaries,
                                           const std::vector<int64_t> &prev_group,
                                           std::size_t lane_slot,
                                           const std::vector<int64_t> &partition_row_counts,
                                           int64_t bytes_per_row) {
    if (prev_group.empty() || lane_slot >= prev_group.size()) {
        return 0;
    }

    std::unordered_set<int64_t> resident(already_on_lane.begin(), already_on_lane.end());
    std::unordered_set<int64_t> peer_resident;
    for (std::size_t idx = 0; idx < prev_group.size(); idx++) {
        if (idx == lane_slot) {
            continue;
        }
        int64_t state_idx = prev_group[idx];
        if (state_idx < 0 || state_idx >= static_cast<int64_t>(summaries.size())) {
            continue;
        }
        for (int64_t partition_id : summaries[static_cast<std::size_t>(state_idx)].partitions) {
            peer_resident.insert(partition_id);
        }
    }

    int64_t bytes = 0;
    for (int64_t partition_id : needed_partitions) {
        if (resident.count(partition_id) == 0 && peer_resident.count(partition_id) > 0) {
            bytes += partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
        }
    }
    return bytes;
}

double peer_bytes_as_host_equivalent(int64_t peer_bytes, const LaneMatchCostConfig &cfg) {
    if (peer_bytes <= 0) {
        return 0.0;
    }
    return static_cast<double>(peer_bytes) *
           static_cast<double>(std::max<int64_t>(cfg.host_bandwidth_bps, 1)) /
           static_cast<double>(std::max<int64_t>(cfg.peer_bandwidth_bps, 1));
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
std::vector<std::vector<int>> reorder_states_for_max_overlap(const std::vector<std::vector<int>> &states);
std::vector<std::vector<int>> build_custom_template_buffer_states(int num_partitions, int buffer_capacity);

bool is_positive_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool can_lift_q4_custom_template(int num_partitions, int buffer_capacity) {
    if (buffer_capacity <= 4 || buffer_capacity % 4 != 0) {
        return false;
    }

    int fine_per_coarse = buffer_capacity / 4;
    if (fine_per_coarse <= 1 || num_partitions % fine_per_coarse != 0) {
        return false;
    }

    int coarse_num_partitions = num_partitions / fine_per_coarse;
    return coarse_num_partitions >= 4 && is_positive_power_of_two(coarse_num_partitions);
}

std::vector<std::vector<int>> expand_coarse_custom_states(const std::vector<std::vector<int>> &coarse_states,
                                                          int fine_per_coarse) {
    std::vector<std::vector<int>> expanded_states;
    expanded_states.reserve(coarse_states.size());
    for (const auto &coarse_state : coarse_states) {
        std::vector<int> fine_state;
        fine_state.reserve(coarse_state.size() * fine_per_coarse);
        for (int coarse_part : coarse_state) {
            int fine_start = coarse_part * fine_per_coarse;
            for (int offset = 0; offset < fine_per_coarse; offset++) {
                fine_state.emplace_back(fine_start + offset);
            }
        }
        expanded_states.emplace_back(std::move(fine_state));
    }
    return expanded_states;
}

std::vector<std::vector<int>> build_lifted_q4_custom_template_buffer_states(int num_partitions, int buffer_capacity) {
    int fine_per_coarse = buffer_capacity / 4;
    int coarse_num_partitions = num_partitions / fine_per_coarse;
    auto coarse_states = build_custom_template_buffer_states(coarse_num_partitions, 4);
    return expand_coarse_custom_states(coarse_states, fine_per_coarse);
}

std::vector<std::vector<int>> build_randomized_lifted_q4_custom_template_buffer_states(int num_partitions, int buffer_capacity) {
    int fine_per_coarse = buffer_capacity / 4;
    int coarse_num_partitions = num_partitions / fine_per_coarse;
    auto coarse_states = build_custom_template_buffer_states(coarse_num_partitions, 4);

    Indices coarse_partitions_map = torch::randperm(coarse_num_partitions, torch::kInt32);
    for (auto &state : coarse_states) {
        for (auto &partition : state) {
            partition = coarse_partitions_map[partition].item<int>();
        }
    }

    return expand_coarse_custom_states(coarse_states, fine_per_coarse);
}

std::vector<int> block_pair_group_slice(int group_id, int buffer_capacity, int begin, int end) {
    std::vector<int> partitions;
    partitions.reserve(std::max(0, end - begin));
    int group_start = group_id * buffer_capacity;
    for (int offset = begin; offset < end; offset++) {
        partitions.emplace_back(group_start + offset);
    }
    return partitions;
}

std::vector<int> concat_partition_slices(const std::vector<int> &lhs, const std::vector<int> &rhs) {
    std::vector<int> state;
    state.reserve(lhs.size() + rhs.size());
    state.insert(state.end(), lhs.begin(), lhs.end());
    state.insert(state.end(), rhs.begin(), rhs.end());
    return state;
}

std::vector<std::vector<int>> build_block_pair_cover_buffer_states(int num_partitions, int buffer_capacity) {
    std::vector<std::vector<int>> buffer_states;
    if (num_partitions <= 0 || buffer_capacity <= 0 || buffer_capacity > num_partitions) {
        SPDLOG_WARN("Invalid CUSTOM block-pair cover parameters: num_partitions={} buffer_capacity={}",
                    num_partitions, buffer_capacity);
        return buffer_states;
    }

    if (num_partitions % buffer_capacity != 0 || buffer_capacity % 2 != 0) {
        SPDLOG_WARN(
            "CUSTOM block-pair cover requires num_partitions divisible by an even buffer_capacity; got num_partitions={} buffer_capacity={}",
            num_partitions, buffer_capacity);
        return buffer_states;
    }

    int num_groups = num_partitions / buffer_capacity;
    int half = buffer_capacity / 2;
    buffer_states.reserve(num_groups + 4 * (num_groups * (num_groups - 1) / 2));

    for (int group = 0; group < num_groups; group++) {
        buffer_states.emplace_back(block_pair_group_slice(group, buffer_capacity, 0, buffer_capacity));
    }

    for (int lhs_group = 0; lhs_group < num_groups; lhs_group++) {
        auto lhs_a = block_pair_group_slice(lhs_group, buffer_capacity, 0, half);
        auto lhs_b = block_pair_group_slice(lhs_group, buffer_capacity, half, buffer_capacity);
        for (int rhs_group = lhs_group + 1; rhs_group < num_groups; rhs_group++) {
            auto rhs_a = block_pair_group_slice(rhs_group, buffer_capacity, 0, half);
            auto rhs_b = block_pair_group_slice(rhs_group, buffer_capacity, half, buffer_capacity);

            buffer_states.emplace_back(concat_partition_slices(lhs_a, rhs_a));
            buffer_states.emplace_back(concat_partition_slices(lhs_a, rhs_b));
            buffer_states.emplace_back(concat_partition_slices(lhs_b, rhs_b));
            buffer_states.emplace_back(concat_partition_slices(lhs_b, rhs_a));
        }
    }

    buffer_states = reorder_states_for_max_overlap(buffer_states);
    return buffer_states;
}

std::vector<std::vector<int>> build_custom_template_buffer_states(int num_partitions, int buffer_capacity) {
    if (can_lift_q4_custom_template(num_partitions, buffer_capacity)) {
        return build_lifted_q4_custom_template_buffer_states(num_partitions, buffer_capacity);
    }

    if (buffer_capacity != 4) {
        return build_block_pair_cover_buffer_states(num_partitions, buffer_capacity);
    }

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
                                                             const vector<torch::Tensor> &edge_buckets_per_buffer,
                                                             const vector<int64_t> &edge_bucket_sizes = {}) {
    std::vector<StateAccessSummary> summaries(buffer_states.size());
    int64_t edge_size_num_partitions = 0;
    if (!edge_bucket_sizes.empty()) {
        const auto root = static_cast<int64_t>(std::llround(std::sqrt(static_cast<long double>(edge_bucket_sizes.size()))));
        if (root > 0 && root * root == static_cast<int64_t>(edge_bucket_sizes.size())) {
            edge_size_num_partitions = root;
        }
    }
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
        auto accessor = edge_buckets.accessor<int64_t, 2>();
        for (int64_t row = 0; row < edge_buckets.size(0); row++) {
            const int64_t src_part = accessor[row][0];
            const int64_t dst_part = accessor[row][1];
            int64_t bucket_edges = 1;
            if (edge_size_num_partitions > 0 &&
                src_part >= 0 && src_part < edge_size_num_partitions &&
                dst_part >= 0 && dst_part < edge_size_num_partitions) {
                bucket_edges = edge_bucket_sizes[static_cast<std::size_t>(src_part * edge_size_num_partitions + dst_part)];
            }
            summaries[i].total_bucket_edges += bucket_edges;
            summaries[i].incident_bucket_counts[src_part] += bucket_edges;
            summaries[i].incident_bucket_counts[dst_part] += bucket_edges;
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

int64_t state_access_overlap_count(const StateAccessSummary &lhs, const StateAccessSummary &rhs) {
    std::size_t i = 0;
    std::size_t j = 0;
    int64_t overlap = 0;
    while (i < lhs.partitions.size() && j < rhs.partitions.size()) {
        if (lhs.partitions[i] == rhs.partitions[j]) {
            overlap++;
            i++;
            j++;
        } else if (lhs.partitions[i] < rhs.partitions[j]) {
            i++;
        } else {
            j++;
        }
    }
    return overlap;
}

int64_t state_access_admit_count(const StateAccessSummary &previous_state, const StateAccessSummary &next_state) {
    return static_cast<int64_t>(next_state.partitions.size()) -
           state_access_overlap_count(previous_state, next_state);
}

bool lane_transition_respects_max_admits(const StateAccessSummary &previous_state,
                                         const StateAccessSummary &next_state,
                                         const LaneMatchCostConfig &cfg) {
    if (cfg.max_admits_per_transition < 0) {
        return true;
    }
    return state_access_admit_count(previous_state, next_state) <= cfg.max_admits_per_transition;
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
                                       const std::vector<StateAccessSummary> &summaries,
                                       const std::vector<int64_t> &prev_group,
                                       std::size_t lane_slot,
                                       const std::vector<int64_t> &partition_row_counts,
                                       const LaneMatchCostConfig &cfg,
                                       int64_t bytes_per_row) {
    if (!lane_transition_respects_max_admits(previous_state, next_state, cfg)) {
        return std::numeric_limits<double>::infinity();
    }

    const int64_t host_bytes =
        handoff_bytes_host(next_state.partitions, previous_state.partitions, partition_row_counts, bytes_per_row);
    int64_t peer_bytes = 0;
    if (cfg.allow_peer_relay) {
        peer_bytes = handoff_bytes_peer_from_lane_group(next_state.partitions, previous_state.partitions, summaries, prev_group,
                                                        lane_slot, partition_row_counts, bytes_per_row);
        peer_bytes = std::min(peer_bytes, host_bytes);
    }
    const int64_t overlap_bytes =
        resident_overlap_bytes(previous_state.partitions, next_state.partitions, partition_row_counts, bytes_per_row);
    const double effective_host_equivalent_bytes =
        static_cast<double>(host_bytes - peer_bytes) + peer_bytes_as_host_equivalent(peer_bytes, cfg);
    double cost = bytes_per_row > 0
                      ? effective_host_equivalent_bytes / static_cast<double>(std::max<int64_t>(cfg.host_bandwidth_bps, 1))
                      : effective_host_equivalent_bytes;
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

        double step_cost = lane_assignment_transition_cost(summaries[prev_group[slot]], summaries[candidate_group[candidate_idx]],
                                                           summaries, prev_group, slot, partition_row_counts, cfg, bytes_per_row);
        if (step_cost == std::numeric_limits<double>::infinity()) {
            continue;
        }

        LaneAlignmentResult suffix = search_best_lane_alignment_dp(prev_group, candidate_group, summaries, partition_row_counts, cfg,
                                                                   bytes_per_row, used_mask | (1ULL << candidate_idx), memo);
        if (suffix.cost == std::numeric_limits<double>::infinity()) {
            continue;
        }

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
    if (candidate_group.size() >= 64) {
        return LaneAlignmentResult{};
    }

    std::vector<int64_t> prev_lane_prefix;
    prev_lane_prefix.reserve(std::min(prev_group.size(), candidate_group.size()));
    for (std::size_t idx = 0; idx < prev_group.size() && idx < candidate_group.size(); idx++) {
        prev_lane_prefix.emplace_back(prev_group[idx]);
    }
    if (prev_lane_prefix.empty()) {
        return LaneAlignmentResult{0.0, candidate_group};
    }
    if (prev_lane_prefix.size() != candidate_group.size()) {
        return LaneAlignmentResult{};
    }

    std::unordered_map<uint64_t, LaneAlignmentResult> memo;
    return search_best_lane_alignment_dp(prev_lane_prefix, candidate_group, summaries, partition_row_counts, cfg, bytes_per_row, 0ULL, memo);
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
    if (prev_group.empty()) {
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
        if (cfg.max_admits_per_transition >= 0) {
            return best;
        }
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

struct TwoGpuRoundPair {
    int64_t lhs = -1;
    int64_t rhs = -1;
};

struct TwoGpuTransitionCost {
    int64_t host_bytes = 0;
    int64_t peer_bytes = 0;
    int64_t host_partitions = 0;
    int64_t peer_partitions = 0;
    int64_t lane_admits = 0;
    int64_t same_overlap = 0;
    int64_t max_lane_admits = 0;
    int64_t bad_lane_transitions = 0;
    long double score = 0.0L;
};

struct TwoGpuPathCost {
    int64_t host_bytes = 0;
    int64_t peer_bytes = 0;
    int64_t host_partitions = 0;
    int64_t peer_partitions = 0;
    int64_t lane_admits = 0;
    int64_t same_overlap = 0;
    int64_t max_lane_admits = 0;
    int64_t bad_lane_transitions = 0;
    int64_t first_transition_peer_partitions = 0;
    int64_t peer_partitions_over_budget = 0;
    int64_t lane_edge_imbalance = 0;
    long double score = 0.0L;
};

struct TwoGpuOrientedPath {
    std::vector<TwoGpuRoundPair> ordered_pairs;
    std::vector<TwoGpuRoundPair> oriented_rounds;
    TwoGpuPathCost cost;
};

bool two_gpu_pair_valid(const TwoGpuRoundPair &pair, const std::vector<StateAccessSummary> &summaries) {
    if (pair.lhs < 0 || pair.rhs < 0 || pair.lhs == pair.rhs ||
        pair.lhs >= static_cast<int64_t>(summaries.size()) ||
        pair.rhs >= static_cast<int64_t>(summaries.size())) {
        return false;
    }
    return states_disjoint(summaries[static_cast<std::size_t>(pair.lhs)].partitions,
                           summaries[static_cast<std::size_t>(pair.rhs)].partitions);
}

bool partition_in_sorted_list(const std::vector<int64_t> &partitions, int64_t partition_id) {
    return std::binary_search(partitions.begin(), partitions.end(), partition_id);
}

TwoGpuTransitionCost two_gpu_transition_cost(const TwoGpuRoundPair &prev_round,
                                             const TwoGpuRoundPair &next_round,
                                             const std::vector<StateAccessSummary> &summaries,
                                             const std::vector<int64_t> &partition_row_counts,
                                             const LaneMatchCostConfig &cfg,
                                             int64_t bytes_per_row) {
    TwoGpuTransitionCost cost;
    const int64_t prev_states[2] = {prev_round.lhs, prev_round.rhs};
    const int64_t next_states[2] = {next_round.lhs, next_round.rhs};

    for (int lane = 0; lane < 2; lane++) {
        const auto &same_prev = summaries[static_cast<std::size_t>(prev_states[lane])].partitions;
        const auto &other_prev = summaries[static_cast<std::size_t>(prev_states[1 - lane])].partitions;
        const auto &needed = summaries[static_cast<std::size_t>(next_states[lane])].partitions;
        int64_t lane_admits = 0;
        for (int64_t partition_id : needed) {
            if (partition_in_sorted_list(same_prev, partition_id)) {
                cost.same_overlap++;
                continue;
            }
            lane_admits++;
            const int64_t bytes = partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
            if (partition_in_sorted_list(other_prev, partition_id) && cfg.allow_peer_relay) {
                cost.peer_bytes += bytes;
                cost.peer_partitions++;
            } else {
                cost.host_bytes += bytes;
                cost.host_partitions++;
            }
        }
        cost.lane_admits += lane_admits;
        cost.max_lane_admits = std::max<int64_t>(cost.max_lane_admits, lane_admits);
        if (cfg.max_admits_per_transition >= 0 && lane_admits > cfg.max_admits_per_transition) {
            cost.bad_lane_transitions++;
        }
    }

    // For Opt88 lane matching, host admits are the expensive critical-path
    // movement. Keep the per-lane admit cap first, then minimize CPU->GPU
    // admits before minimizing total lane admits/peer relay.
    cost.score = static_cast<long double>(cost.bad_lane_transitions) * 1000000000000.0L +
                 static_cast<long double>(cost.host_partitions) * 1000000.0L +
                 static_cast<long double>(cost.lane_admits) * 1000.0L +
                 static_cast<long double>(cost.max_lane_admits);
    return cost;
}

bool two_gpu_path_cost_less(const TwoGpuPathCost &lhs, const TwoGpuPathCost &rhs) {
    if (lhs.bad_lane_transitions != rhs.bad_lane_transitions) {
        return lhs.bad_lane_transitions < rhs.bad_lane_transitions;
    }
    if (lhs.peer_partitions_over_budget != rhs.peer_partitions_over_budget) {
        return lhs.peer_partitions_over_budget < rhs.peer_partitions_over_budget;
    }
    if (lhs.host_partitions != rhs.host_partitions) {
        return lhs.host_partitions < rhs.host_partitions;
    }
    if (lhs.lane_admits != rhs.lane_admits) {
        return lhs.lane_admits < rhs.lane_admits;
    }
    if (lhs.max_lane_admits != rhs.max_lane_admits) {
        return lhs.max_lane_admits < rhs.max_lane_admits;
    }
    if (lhs.same_overlap != rhs.same_overlap) {
        return lhs.same_overlap > rhs.same_overlap;
    }
    if (lhs.first_transition_peer_partitions != rhs.first_transition_peer_partitions) {
        return lhs.first_transition_peer_partitions < rhs.first_transition_peer_partitions;
    }
    if (lhs.host_bytes != rhs.host_bytes) {
        return lhs.host_bytes < rhs.host_bytes;
    }
    if (lhs.peer_bytes != rhs.peer_bytes) {
        return lhs.peer_bytes < rhs.peer_bytes;
    }
    return lhs.score < rhs.score;
}

TwoGpuPathCost evaluate_two_gpu_oriented_path(const std::vector<TwoGpuRoundPair> &oriented_rounds,
                                              const std::vector<StateAccessSummary> &summaries,
                                              const std::vector<int64_t> &partition_row_counts,
                                              const LaneMatchCostConfig &cfg,
                                              int64_t bytes_per_row) {
    TwoGpuPathCost path_cost;
    if (oriented_rounds.empty()) {
        return path_cost;
    }

    int64_t lane_edges[2] = {0, 0};
    for (const auto &round : oriented_rounds) {
        lane_edges[0] += summaries[static_cast<std::size_t>(round.lhs)].total_bucket_edges;
        lane_edges[1] += summaries[static_cast<std::size_t>(round.rhs)].total_bucket_edges;
    }
    path_cost.lane_edge_imbalance = std::llabs(lane_edges[0] - lane_edges[1]);

    for (std::size_t idx = 1; idx < oriented_rounds.size(); idx++) {
        auto step = two_gpu_transition_cost(oriented_rounds[idx - 1], oriented_rounds[idx], summaries,
                                            partition_row_counts, cfg, bytes_per_row);
        path_cost.host_bytes += step.host_bytes;
        path_cost.peer_bytes += step.peer_bytes;
        path_cost.host_partitions += step.host_partitions;
        path_cost.peer_partitions += step.peer_partitions;
        if (idx == 1) {
            path_cost.first_transition_peer_partitions += step.peer_partitions;
        }
        path_cost.lane_admits += step.lane_admits;
        path_cost.same_overlap += step.same_overlap;
        path_cost.max_lane_admits = std::max(path_cost.max_lane_admits, step.max_lane_admits);
        path_cost.bad_lane_transitions += step.bad_lane_transitions;
        path_cost.score += step.score;
    }

    const int64_t peer_budget = optimal88_peer_handoff_budget(2);
    if (peer_budget >= 0) {
        path_cost.peer_partitions_over_budget = std::max<int64_t>(path_cost.peer_partitions - peer_budget, 0);
    }

    return path_cost;
}

TwoGpuTransitionCost best_two_gpu_pair_transition_cost(const TwoGpuRoundPair &prev_pair,
                                                       const TwoGpuRoundPair &next_pair,
                                                       const std::vector<StateAccessSummary> &summaries,
                                                       const std::vector<int64_t> &partition_row_counts,
                                                       const LaneMatchCostConfig &cfg,
                                                       int64_t bytes_per_row) {
    auto c0 = two_gpu_transition_cost(prev_pair, next_pair, summaries, partition_row_counts, cfg, bytes_per_row);
    TwoGpuRoundPair swapped{next_pair.rhs, next_pair.lhs};
    auto c1 = two_gpu_transition_cost(prev_pair, swapped, summaries, partition_row_counts, cfg, bytes_per_row);
    TwoGpuPathCost p0;
    p0.score = c0.score;
    p0.bad_lane_transitions = c0.bad_lane_transitions;
    p0.host_bytes = c0.host_bytes;
    p0.peer_bytes = c0.peer_bytes;
    p0.host_partitions = c0.host_partitions;
    p0.peer_partitions = c0.peer_partitions;
    p0.lane_admits = c0.lane_admits;
    p0.same_overlap = c0.same_overlap;
    p0.max_lane_admits = c0.max_lane_admits;
    TwoGpuPathCost p1;
    p1.score = c1.score;
    p1.bad_lane_transitions = c1.bad_lane_transitions;
    p1.host_bytes = c1.host_bytes;
    p1.peer_bytes = c1.peer_bytes;
    p1.host_partitions = c1.host_partitions;
    p1.peer_partitions = c1.peer_partitions;
    p1.lane_admits = c1.lane_admits;
    p1.same_overlap = c1.same_overlap;
    p1.max_lane_admits = c1.max_lane_admits;
    return two_gpu_path_cost_less(p1, p0) ? c1 : c0;
}

TwoGpuOrientedPath orient_two_gpu_pair_order_dp(const std::vector<TwoGpuRoundPair> &ordered_pairs,
                                                const std::vector<StateAccessSummary> &summaries,
                                                const std::vector<int64_t> &partition_row_counts,
                                                const LaneMatchCostConfig &cfg,
                                                int64_t bytes_per_row) {
    TwoGpuOrientedPath result;
    result.ordered_pairs = ordered_pairs;
    if (ordered_pairs.empty()) {
        return result;
    }

    struct Cell {
        long double score = std::numeric_limits<long double>::infinity();
        int prev_orientation = -1;
    };

    const std::size_t n = ordered_pairs.size();
    std::vector<std::array<Cell, 2>> dp(n);
    dp[0][0].score = 0.0L;
    dp[0][1].score = 0.0L;

    auto orient = [](const TwoGpuRoundPair &pair, int orientation) {
        return orientation == 0 ? pair : TwoGpuRoundPair{pair.rhs, pair.lhs};
    };

    for (std::size_t idx = 1; idx < n; idx++) {
        for (int curr_o = 0; curr_o < 2; curr_o++) {
            auto curr = orient(ordered_pairs[idx], curr_o);
            for (int prev_o = 0; prev_o < 2; prev_o++) {
                auto prev = orient(ordered_pairs[idx - 1], prev_o);
                auto step = two_gpu_transition_cost(prev, curr, summaries, partition_row_counts, cfg, bytes_per_row);
                long double candidate_score = dp[idx - 1][prev_o].score + step.score;
                if (candidate_score < dp[idx][curr_o].score) {
                    dp[idx][curr_o].score = candidate_score;
                    dp[idx][curr_o].prev_orientation = prev_o;
                }
            }
        }
    }

    int orientation = dp[n - 1][1].score < dp[n - 1][0].score ? 1 : 0;
    result.oriented_rounds.assign(n, {});
    for (int idx = static_cast<int>(n) - 1; idx >= 0; idx--) {
        result.oriented_rounds[static_cast<std::size_t>(idx)] = orient(ordered_pairs[static_cast<std::size_t>(idx)], orientation);
        orientation = idx > 0 ? dp[static_cast<std::size_t>(idx)][orientation].prev_orientation : 0;
    }
    result.cost = evaluate_two_gpu_oriented_path(result.oriented_rounds, summaries, partition_row_counts, cfg, bytes_per_row);
    return result;
}

std::vector<TwoGpuRoundPair> greedy_order_two_gpu_pairs(const std::vector<TwoGpuRoundPair> &pairs,
                                                        const std::vector<StateAccessSummary> &summaries,
                                                        const std::vector<int64_t> &partition_row_counts,
                                                        const LaneMatchCostConfig &cfg,
                                                        int64_t bytes_per_row,
                                                        std::mt19937_64 &rng) {
    if (pairs.empty()) {
        return {};
    }

    std::vector<uint8_t> used(pairs.size(), 0);
    std::vector<TwoGpuRoundPair> ordered;
    ordered.reserve(pairs.size());
    std::uniform_int_distribution<int> start_dist(0, static_cast<int>(pairs.size()) - 1);
    int current = start_dist(rng);
    used[static_cast<std::size_t>(current)] = 1;
    ordered.emplace_back(pairs[static_cast<std::size_t>(current)]);

    while (ordered.size() < pairs.size()) {
        int best_idx = -1;
        TwoGpuTransitionCost best_cost;
        bool have_best = false;
        for (int candidate = 0; candidate < static_cast<int>(pairs.size()); candidate++) {
            if (used[static_cast<std::size_t>(candidate)] != 0) {
                continue;
            }
            auto cost = best_two_gpu_pair_transition_cost(ordered.back(), pairs[static_cast<std::size_t>(candidate)],
                                                          summaries, partition_row_counts, cfg, bytes_per_row);
            TwoGpuPathCost current_key;
            current_key.score = cost.score;
            current_key.bad_lane_transitions = cost.bad_lane_transitions;
            current_key.host_bytes = cost.host_bytes;
            current_key.peer_bytes = cost.peer_bytes;
            current_key.host_partitions = cost.host_partitions;
            current_key.peer_partitions = cost.peer_partitions;
            current_key.lane_admits = cost.lane_admits;
            current_key.same_overlap = cost.same_overlap;
            current_key.max_lane_admits = cost.max_lane_admits;
            TwoGpuPathCost best_key;
            best_key.score = best_cost.score;
            best_key.bad_lane_transitions = best_cost.bad_lane_transitions;
            best_key.host_bytes = best_cost.host_bytes;
            best_key.peer_bytes = best_cost.peer_bytes;
            best_key.host_partitions = best_cost.host_partitions;
            best_key.peer_partitions = best_cost.peer_partitions;
            best_key.lane_admits = best_cost.lane_admits;
            best_key.same_overlap = best_cost.same_overlap;
            best_key.max_lane_admits = best_cost.max_lane_admits;
            if (!have_best || two_gpu_path_cost_less(current_key, best_key)) {
                best_idx = candidate;
                best_cost = cost;
                have_best = true;
            }
        }
        if (best_idx < 0) {
            break;
        }
        used[static_cast<std::size_t>(best_idx)] = 1;
        ordered.emplace_back(pairs[static_cast<std::size_t>(best_idx)]);
    }

    if (ordered.size() != pairs.size()) {
        return pairs;
    }
    return ordered;
}

TwoGpuOrientedPath improve_two_gpu_pair_order(std::vector<TwoGpuRoundPair> ordered_pairs,
                                              const std::vector<StateAccessSummary> &summaries,
                                              const std::vector<int64_t> &partition_row_counts,
                                              const LaneMatchCostConfig &cfg,
                                              int64_t bytes_per_row,
                                              int max_passes) {
    TwoGpuOrientedPath best =
        orient_two_gpu_pair_order_dp(ordered_pairs, summaries, partition_row_counts, cfg, bytes_per_row);
    const int n = static_cast<int>(ordered_pairs.size());
    for (int pass = 0; pass < std::max(0, max_passes); pass++) {
        bool improved = false;
        for (int begin = 0; begin < n - 2 && !improved; begin++) {
            for (int end = begin + 1; end < n && !improved; end++) {
                std::vector<TwoGpuRoundPair> candidate = ordered_pairs;
                std::reverse(candidate.begin() + begin, candidate.begin() + end + 1);
                auto candidate_path =
                    orient_two_gpu_pair_order_dp(candidate, summaries, partition_row_counts, cfg, bytes_per_row);
                if (two_gpu_path_cost_less(candidate_path.cost, best.cost)) {
                    ordered_pairs = std::move(candidate);
                    best = std::move(candidate_path);
                    improved = true;
                }
            }
        }
        if (!improved) {
            break;
        }
    }
    return best;
}

std::vector<TwoGpuRoundPair> build_random_two_gpu_matching(const std::vector<StateAccessSummary> &summaries,
                                                           std::mt19937_64 &rng) {
    const int n = static_cast<int>(summaries.size());
    std::vector<int64_t> remaining(n);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::shuffle(remaining.begin(), remaining.end(), rng);
    std::vector<TwoGpuRoundPair> pairs;
    pairs.reserve(static_cast<std::size_t>(n / 2));

    while (!remaining.empty()) {
        auto anchor_it = std::min_element(remaining.begin(), remaining.end(), [&](int64_t lhs, int64_t rhs) {
            int lhs_degree = 0;
            int rhs_degree = 0;
            for (auto candidate : remaining) {
                if (candidate == lhs) {
                    continue;
                }
                lhs_degree += states_disjoint(summaries[static_cast<std::size_t>(lhs)].partitions,
                                              summaries[static_cast<std::size_t>(candidate)].partitions) ? 1 : 0;
                rhs_degree += states_disjoint(summaries[static_cast<std::size_t>(rhs)].partitions,
                                              summaries[static_cast<std::size_t>(candidate)].partitions) ? 1 : 0;
            }
            if (lhs_degree != rhs_degree) {
                return lhs_degree < rhs_degree;
            }
            return lhs < rhs;
        });
        int64_t anchor = *anchor_it;
        std::vector<int64_t> candidates;
        for (auto state_idx : remaining) {
            if (state_idx == anchor) {
                continue;
            }
            if (states_disjoint(summaries[static_cast<std::size_t>(anchor)].partitions,
                                summaries[static_cast<std::size_t>(state_idx)].partitions)) {
                candidates.emplace_back(state_idx);
            }
        }
        if (candidates.empty()) {
            return {};
        }
        std::shuffle(candidates.begin(), candidates.end(), rng);
        int64_t partner = candidates.front();
        pairs.push_back(anchor < partner ? TwoGpuRoundPair{anchor, partner} : TwoGpuRoundPair{partner, anchor});

        remaining.erase(std::remove(remaining.begin(), remaining.end(), anchor), remaining.end());
        remaining.erase(std::remove(remaining.begin(), remaining.end(), partner), remaining.end());
    }

    return pairs;
}

std::vector<TwoGpuRoundPair> build_farthest_two_gpu_matching(const std::vector<StateAccessSummary> &summaries) {
    const int n = static_cast<int>(summaries.size());
    std::vector<uint8_t> used(static_cast<std::size_t>(n), 0);
    std::vector<TwoGpuRoundPair> pairs;
    pairs.reserve(static_cast<std::size_t>(n / 2));
    for (int state_idx = 0; state_idx < n; state_idx++) {
        if (used[static_cast<std::size_t>(state_idx)] != 0) {
            continue;
        }
        int best_partner = -1;
        auto best_key = std::make_tuple(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
        for (int candidate = 0; candidate < n; candidate++) {
            if (candidate == state_idx || used[static_cast<std::size_t>(candidate)] != 0) {
                continue;
            }
            if (!states_disjoint(summaries[static_cast<std::size_t>(state_idx)].partitions,
                                 summaries[static_cast<std::size_t>(candidate)].partitions)) {
                continue;
            }
            auto key = std::make_tuple(std::abs(candidate - state_idx), candidate);
            if (key > best_key) {
                best_key = key;
                best_partner = candidate;
            }
        }
        if (best_partner < 0) {
            return {};
        }
        used[static_cast<std::size_t>(state_idx)] = 1;
        used[static_cast<std::size_t>(best_partner)] = 1;
        pairs.push_back(state_idx < best_partner ? TwoGpuRoundPair{state_idx, best_partner}
                                                 : TwoGpuRoundPair{best_partner, state_idx});
    }
    return pairs;
}

std::vector<int64_t> getTw16TwentyStateTwoGpuLaneMatchedPermutation(const vector<torch::Tensor>& buffer_states,
                                                                    const vector<torch::Tensor>& edge_buckets_per_buffer,
                                                                    const vector<int64_t> &edge_bucket_sizes,
                                                                    const std::vector<int64_t> &partition_row_counts,
                                                                    const PlanEmbeddingLayout &layout) {
    if (buffer_states.size() != 20 || edge_buckets_per_buffer.size() != buffer_states.size()) {
        return {};
    }

    auto summaries = build_state_access_summaries(buffer_states, edge_buckets_per_buffer, edge_bucket_sizes);
    for (const auto &summary : summaries) {
        if (summary.partitions.size() != 4) {
            return {};
        }
        for (auto partition : summary.partitions) {
            if (partition < 0 || partition >= 16) {
                return {};
            }
        }
    }

    LaneMatchCostConfig cfg = lane_match_cost_config_from_env();
    cfg.allow_peer_relay = stateflow_env_bool("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "STATEFLOW_ALLOW_PEER_RELAY",
                                              stateflow_allow_peer_relay_default());
    if (cfg.max_admits_per_transition < 0) {
        cfg.max_admits_per_transition = 3;
    }
    const int64_t bytes_per_row = plan_embedding_bytes_per_row(layout);

    auto find_state_with_partitions = [&](std::vector<int64_t> partitions,
                                          const std::vector<uint8_t> &seen) -> int64_t {
        std::sort(partitions.begin(), partitions.end());
        for (std::size_t idx = 0; idx < summaries.size(); idx++) {
            if (seen[idx] != 0) {
                continue;
            }
            if (summaries[idx].partitions == partitions) {
                return static_cast<int64_t>(idx);
            }
        }
        return -1;
    };

    using PartitionRound = std::pair<std::vector<int64_t>, std::vector<int64_t>>;
    const std::vector<PartitionRound> partition_rounds = {
        {{0, 4, 8, 12}, {1, 5, 9, 13}},
        {{0, 1, 2, 3}, {12, 13, 14, 15}},
        {{0, 5, 10, 15}, {3, 6, 9, 12}},
        {{4, 5, 6, 7}, {8, 9, 10, 11}},
        {{1, 4, 11, 14}, {2, 7, 8, 13}},
        {{0, 7, 9, 14}, {2, 5, 11, 12}},
        {{0, 6, 11, 13}, {2, 4, 9, 15}},
        {{1, 6, 8, 15}, {3, 4, 10, 13}},
        {{1, 7, 10, 12}, {3, 5, 8, 14}},
        {{2, 6, 10, 14}, {3, 7, 11, 15}},
    };

    std::vector<uint8_t> seen(buffer_states.size(), 0);
    std::vector<TwoGpuRoundPair> oriented_rounds;
    oriented_rounds.reserve(partition_rounds.size());
    for (const auto &round : partition_rounds) {
        TwoGpuRoundPair pair;
        pair.lhs = find_state_with_partitions(round.first, seen);
        if (pair.lhs < 0) {
            return {};
        }
        seen[static_cast<std::size_t>(pair.lhs)] = 1;
        pair.rhs = find_state_with_partitions(round.second, seen);
        if (pair.rhs < 0) {
            return {};
        }
        seen[static_cast<std::size_t>(pair.rhs)] = 1;
        if (!two_gpu_pair_valid(pair, summaries)) {
            return {};
        }
        oriented_rounds.emplace_back(pair);
    }

    if (oriented_rounds.size() * 2 != buffer_states.size()) {
        return {};
    }
    for (auto value : seen) {
        if (value == 0) {
            return {};
        }
    }

    auto path_cost = evaluate_two_gpu_oriented_path(oriented_rounds, summaries, partition_row_counts, cfg, bytes_per_row);
    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());
    for (const auto &round : oriented_rounds) {
        permutation.emplace_back(round.lhs);
        permutation.emplace_back(round.rhs);
    }

    SPDLOG_INFO(
        "TW16 p16/q4 2-GPU lane schedule selected rounds={} microstates={} transition_host_admits={} peer_handoffs={} peer_handoffs_over_budget={} lane_transition_admits={} same_lane_overlap={} max_lane_admits={} bad_lane_transitions={} first_transition_peer_handoffs={} host_bytes={} peer_bytes={} lane_edge_imbalance={}",
        oriented_rounds.size(), buffer_states.size(), path_cost.host_partitions,
        path_cost.peer_partitions, path_cost.peer_partitions_over_budget,
        path_cost.lane_admits, path_cost.same_overlap,
        path_cost.max_lane_admits, path_cost.bad_lane_transitions,
        path_cost.first_transition_peer_partitions,
        path_cost.host_bytes, path_cost.peer_bytes, path_cost.lane_edge_imbalance);

    return permutation;
}

std::vector<int64_t> getOptimal88TwoGpuLaneMatchedPermutation(const vector<torch::Tensor>& buffer_states,
                                                              const vector<torch::Tensor>& edge_buckets_per_buffer,
                                                              const vector<int64_t> &edge_bucket_sizes,
                                                              const std::vector<int64_t> &partition_row_counts,
                                                              const PlanEmbeddingLayout &layout) {
    if (buffer_states.size() != 88 || edge_buckets_per_buffer.size() != buffer_states.size()) {
        return {};
    }

    auto summaries = build_state_access_summaries(buffer_states, edge_buckets_per_buffer, edge_bucket_sizes);
    LaneMatchCostConfig cfg = lane_match_cost_config_from_env();
    cfg.allow_peer_relay = stateflow_env_bool("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "STATEFLOW_ALLOW_PEER_RELAY",
                                              stateflow_allow_peer_relay_default());
    if (cfg.max_admits_per_transition < 0) {
        cfg.max_admits_per_transition = 3;
    }
    const int64_t bytes_per_row = plan_embedding_bytes_per_row(layout);
    const int restarts = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_2GPU_RESTARTS", nullptr, 96)));
    const int two_opt_passes = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_2GPU_2OPT_PASSES", nullptr, 12)));
    const int64_t seed = stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_2GPU_SEED", nullptr, 29);
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    TwoGpuOrientedPath best_path;
    bool have_best = false;
    auto try_oriented_rounds = [&](std::vector<TwoGpuRoundPair> oriented_rounds) {
        if (oriented_rounds.size() * 2 != buffer_states.size()) {
            return;
        }
        std::vector<uint8_t> seen(buffer_states.size(), 0);
        for (const auto &round : oriented_rounds) {
            if (round.lhs < 0 || round.rhs < 0 ||
                round.lhs >= static_cast<int64_t>(buffer_states.size()) ||
                round.rhs >= static_cast<int64_t>(buffer_states.size()) ||
                seen[static_cast<std::size_t>(round.lhs)] != 0 ||
                seen[static_cast<std::size_t>(round.rhs)] != 0 ||
                !two_gpu_pair_valid(round, summaries)) {
                return;
            }
            seen[static_cast<std::size_t>(round.lhs)] = 1;
            seen[static_cast<std::size_t>(round.rhs)] = 1;
        }
        TwoGpuOrientedPath candidate;
        candidate.ordered_pairs = oriented_rounds;
        candidate.oriented_rounds = std::move(oriented_rounds);
        candidate.cost =
            evaluate_two_gpu_oriented_path(candidate.oriented_rounds, summaries, partition_row_counts, cfg, bytes_per_row);
        if (!have_best || two_gpu_path_cost_less(candidate.cost, best_path.cost)) {
            best_path = std::move(candidate);
            have_best = true;
        }
    };
    auto find_state_with_partitions = [&](std::vector<int64_t> partitions,
                                          const std::vector<uint8_t> &seen) -> int64_t {
        std::sort(partitions.begin(), partitions.end());
        for (std::size_t idx = 0; idx < summaries.size(); idx++) {
            if (seen[idx] != 0) {
                continue;
            }
            if (summaries[idx].partitions == partitions) {
                return static_cast<int64_t>(idx);
            }
        }
        return -1;
    };
    using PartitionRound = std::pair<std::vector<int64_t>, std::vector<int64_t>>;
    auto try_partition_rounds = [&](const std::vector<PartitionRound> &partition_rounds) {
        if (partition_rounds.size() * 2 != buffer_states.size()) {
            return;
        }
        std::vector<uint8_t> seen(buffer_states.size(), 0);
        std::vector<TwoGpuRoundPair> oriented_rounds;
        oriented_rounds.reserve(partition_rounds.size());
        for (const auto &round : partition_rounds) {
            TwoGpuRoundPair pair;
            pair.lhs = find_state_with_partitions(round.first, seen);
            if (pair.lhs < 0) {
                return;
            }
            seen[static_cast<std::size_t>(pair.lhs)] = 1;
            pair.rhs = find_state_with_partitions(round.second, seen);
            if (pair.rhs < 0) {
                return;
            }
            seen[static_cast<std::size_t>(pair.rhs)] = 1;
            if (!two_gpu_pair_valid(pair, summaries)) {
                return;
            }
            oriented_rounds.emplace_back(pair);
        }
        try_oriented_rounds(std::move(oriented_rounds));
    };
    auto try_matching = [&](std::vector<TwoGpuRoundPair> pairs) {
        if (pairs.size() * 2 != buffer_states.size()) {
            return;
        }
        for (const auto &pair : pairs) {
            if (!two_gpu_pair_valid(pair, summaries)) {
                return;
            }
        }
        auto ordered = greedy_order_two_gpu_pairs(pairs, summaries, partition_row_counts, cfg, bytes_per_row, rng);
        auto candidate = improve_two_gpu_pair_order(std::move(ordered), summaries, partition_row_counts, cfg,
                                                    bytes_per_row, two_opt_passes);
        if (!have_best || two_gpu_path_cost_less(candidate.cost, best_path.cost)) {
            best_path = std::move(candidate);
            have_best = true;
        }
    };

    // Seed the search with the best known Opt88 2-GPU schedule over this
    // cover: 186 host admits, 51 peer admits, and 237 total lane admits. Store
    // it by partition set rather than state index so the seed remains valid if
    // the 88-state cover construction order changes.
    try_partition_rounds(std::vector<PartitionRound>{
        {{24, 25, 26, 27}, {2, 6, 10, 14}},
        {{4, 15, 16, 27}, {7, 10, 20, 25}},
        {{0, 1, 27, 29}, {2, 4, 18, 20}},
        {{0, 1, 26, 28}, {18, 23, 24, 29}},
        {{9, 14, 27, 28}, {8, 15, 26, 29}},
        {{4, 13, 21, 28}, {6, 7, 26, 29}},
        {{4, 6, 7, 28}, {5, 14, 17, 26}},
        {{5, 6, 7, 27}, {4, 9, 23, 26}},
        {{1, 5, 9, 13}, {6, 15, 23, 30}},
        {{3, 6, 9, 12}, {2, 5, 16, 23}},
        {{3, 14, 16, 29}, {2, 5, 25, 31}},
        {{14, 16, 18, 19}, {9, 15, 25, 31}},
        {{6, 13, 18, 25}, {16, 21, 26, 31}},
        {{6, 11, 21, 24}, {1, 8, 16, 25}},
        {{8, 14, 24, 30}, {0, 6, 16, 22}},
        {{4, 5, 24, 30}, {2, 9, 22, 29}},
        {{1, 15, 22, 24}, {7, 9, 16, 30}},
        {{18, 22, 26, 30}, {9, 10, 11, 16}},
        {{1, 12, 18, 31}, {10, 11, 17, 23}},
        {{1, 4, 11, 14}, {10, 12, 26, 28}},
        {{12, 13, 14, 15}, {1, 10, 21, 30}},
        {{12, 13, 16, 17}, {0, 5, 10, 15}},
        {{0, 13, 19, 30}, {5, 8, 10, 22}},
        {{0, 9, 17, 19}, {8, 11, 22, 27}},
        {{9, 17, 18, 24}, {11, 13, 27, 29}},
        {{17, 20, 27, 30}, {10, 13, 24, 31}},
        {{6, 8, 17, 31}, {16, 20, 24, 28}},
        {{1, 6, 19, 20}, {28, 29, 30, 31}},
        {{8, 9, 20, 21}, {1, 2, 3, 30}},
        {{14, 15, 20, 21}, {0, 2, 3, 24}},
        {{3, 5, 19, 21}, {0, 14, 23, 25}},
        {{5, 12, 20, 29}, {7, 14, 22, 31}},
        {{4, 10, 19, 29}, {0, 11, 20, 31}},
        {{0, 4, 8, 12}, {5, 11, 18, 28}},
        {{7, 12, 19, 24}, {3, 8, 23, 28}},
        {{19, 23, 27, 31}, {3, 7, 11, 15}},
        {{1, 7, 17, 23}, {2, 11, 19, 26}},
        {{2, 7, 8, 13}, {11, 12, 25, 30}},
        {{2, 12, 21, 27}, {19, 22, 25, 28}},
        {{3, 10, 18, 27}, {12, 21, 22, 23}},
        {{0, 7, 18, 21}, {13, 20, 22, 23}},
        {{17, 21, 25, 29}, {3, 13, 20, 26}},
        {{2, 15, 17, 28}, {3, 4, 25, 31}},
        {{8, 15, 18, 19}, {3, 4, 17, 22}},
    });

    try_matching(build_farthest_two_gpu_matching(summaries));
    for (int restart = 0; restart < restarts; restart++) {
        try_matching(build_random_two_gpu_matching(summaries, rng));
    }

    if (!have_best || best_path.oriented_rounds.size() * 2 != buffer_states.size()) {
        return {};
    }

    std::vector<uint8_t> seen(buffer_states.size(), 0);
    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());
    for (const auto &round : best_path.oriented_rounds) {
        if (round.lhs < 0 || round.rhs < 0 ||
            round.lhs >= static_cast<int64_t>(buffer_states.size()) ||
            round.rhs >= static_cast<int64_t>(buffer_states.size()) ||
            seen[static_cast<std::size_t>(round.lhs)] != 0 ||
            seen[static_cast<std::size_t>(round.rhs)] != 0 ||
            !two_gpu_pair_valid(round, summaries)) {
            return {};
        }
        seen[static_cast<std::size_t>(round.lhs)] = 1;
        seen[static_cast<std::size_t>(round.rhs)] = 1;
        permutation.emplace_back(round.lhs);
        permutation.emplace_back(round.rhs);
    }

    SPDLOG_INFO(
        "Optimal88 2-GPU lane schedule selected rounds={} microstates={} transition_host_admits={} peer_handoffs={} peer_handoffs_over_budget={} lane_transition_admits={} same_lane_overlap={} max_lane_admits={} bad_lane_transitions={} first_transition_peer_handoffs={} host_bytes={} peer_bytes={} lane_edge_imbalance={}",
        best_path.oriented_rounds.size(), buffer_states.size(), best_path.cost.host_partitions,
        best_path.cost.peer_partitions, best_path.cost.peer_partitions_over_budget,
        best_path.cost.lane_admits, best_path.cost.same_overlap,
        best_path.cost.max_lane_admits, best_path.cost.bad_lane_transitions,
        best_path.cost.first_transition_peer_partitions,
        best_path.cost.host_bytes, best_path.cost.peer_bytes, best_path.cost.lane_edge_imbalance);

    return permutation;
}

vector<torch::Tensor> assignRoundBalancedMultiGpuEdgeBuckets(const vector<torch::Tensor> &buffer_states,
                                                             const std::vector<int64_t> &permutation,
                                                             int active_devices,
                                                             int num_partitions,
                                                             const vector<int64_t> &edge_bucket_sizes) {
    if (buffer_states.empty() || permutation.size() != buffer_states.size() ||
        active_devices <= 1 ||
        permutation.size() % static_cast<std::size_t>(active_devices) != 0 ||
        num_partitions <= 0) {
        return {};
    }

    const bool have_sizes = edge_bucket_sizes.size() == static_cast<std::size_t>(num_partitions * num_partitions);
    std::vector<std::vector<uint8_t>> contains(buffer_states.size(), std::vector<uint8_t>(num_partitions, 0));
    for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
        auto partitions = tensor_to_partitions(buffer_states[state_idx]);
        for (auto partition : partitions) {
            if (partition >= 0 && partition < num_partitions) {
                contains[state_idx][partition] = 1;
            }
        }
    }

    std::vector<int64_t> position_of_state(buffer_states.size(), -1);
    for (std::size_t pos = 0; pos < permutation.size(); pos++) {
        const int64_t state_idx = permutation[pos];
        if (state_idx < 0 || state_idx >= static_cast<int64_t>(buffer_states.size()) ||
            position_of_state[static_cast<std::size_t>(state_idx)] >= 0) {
            return {};
        }
        position_of_state[static_cast<std::size_t>(state_idx)] = static_cast<int64_t>(pos);
    }

    const std::size_t num_rounds = permutation.size() / static_cast<std::size_t>(active_devices);
    const int num_buckets = num_partitions * num_partitions;
    std::vector<int64_t> bucket_weight(static_cast<std::size_t>(num_buckets), 1);
    std::vector<std::vector<int>> compatible_states(static_cast<std::size_t>(num_buckets));
    for (int src = 0; src < num_partitions; src++) {
        for (int dst = 0; dst < num_partitions; dst++) {
            const int bucket_idx = src * num_partitions + dst;
            bucket_weight[static_cast<std::size_t>(bucket_idx)] =
                have_sizes ? edge_bucket_sizes[static_cast<std::size_t>(bucket_idx)] : 1;
            for (int state_idx = 0; state_idx < static_cast<int>(buffer_states.size()); state_idx++) {
                if (contains[state_idx][src] != 0 && contains[state_idx][dst] != 0) {
                    compatible_states[static_cast<std::size_t>(bucket_idx)].emplace_back(state_idx);
                }
            }
            if (compatible_states[static_cast<std::size_t>(bucket_idx)].empty()) {
                SPDLOG_WARN("Round-balanced {}-GPU bucket assignment found no compatible state for bucket ({},{})",
                            active_devices, src, dst);
                return {};
            }
        }
    }

    std::vector<int> owner(static_cast<std::size_t>(num_buckets), -1);
    vector<vector<std::pair<int, int>>> assigned_buckets(buffer_states.size());
    std::vector<int64_t> assigned_weight(buffer_states.size(), 0);
    std::vector<int64_t> assigned_count(buffer_states.size(), 0);
    std::vector<std::vector<int64_t>> round_lane_weight(num_rounds,
                                                        std::vector<int64_t>(static_cast<std::size_t>(active_devices), 0));

    auto add_bucket_to_state = [&](int bucket_idx, int state_idx, int sign) {
        const int64_t pos = position_of_state[static_cast<std::size_t>(state_idx)];
        const std::size_t round_idx = static_cast<std::size_t>(pos / active_devices);
        const std::size_t lane_idx = static_cast<std::size_t>(pos % active_devices);
        const int64_t weight = bucket_weight[static_cast<std::size_t>(bucket_idx)] * sign;
        assigned_weight[static_cast<std::size_t>(state_idx)] += weight;
        assigned_count[static_cast<std::size_t>(state_idx)] += sign;
        round_lane_weight[round_idx][lane_idx] += weight;
    };

    for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
        int best_state = compatible_states[static_cast<std::size_t>(bucket_idx)].front();
        for (auto state_idx : compatible_states[static_cast<std::size_t>(bucket_idx)]) {
            if (state_idx < best_state) {
                best_state = state_idx;
            }
        }
        owner[static_cast<std::size_t>(bucket_idx)] = best_state;
        add_bucket_to_state(bucket_idx, best_state, 1);
    }

    auto round_delta = [&round_lane_weight](std::size_t round_idx) {
        auto minmax = std::minmax_element(round_lane_weight[round_idx].begin(),
                                          round_lane_weight[round_idx].end());
        if (minmax.first == round_lane_weight[round_idx].end() ||
            minmax.second == round_lane_weight[round_idx].end()) {
            return int64_t{0};
        }
        return *minmax.second - *minmax.first;
    };
    auto imbalance_key = [&round_lane_weight, &round_delta]() {
        int64_t max_delta = 0;
        int64_t total_delta = 0;
        for (std::size_t round_idx = 0; round_idx < round_lane_weight.size(); round_idx++) {
            const int64_t delta = round_delta(round_idx);
            max_delta = std::max(max_delta, delta);
            total_delta += delta;
        }
        return std::make_pair(max_delta, total_delta);
    };

    std::vector<int> bucket_order(num_buckets);
    std::iota(bucket_order.begin(), bucket_order.end(), 0);
    std::sort(bucket_order.begin(), bucket_order.end(), [&](int lhs, int rhs) {
        return std::make_tuple(bucket_weight[static_cast<std::size_t>(lhs)], lhs) >
               std::make_tuple(bucket_weight[static_cast<std::size_t>(rhs)], rhs);
    });

    const int max_iterations = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_Q4_MULTIGPU_ROUND_BALANCE_ITERS",
                                                "GEGE_BOUNDED_Q4_2GPU_ROUND_BALANCE_ITERS", 10000)));
    auto initial_key = imbalance_key();
    int64_t moves = 0;
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        auto current_key = imbalance_key();
        struct Move {
            bool valid = false;
            int bucket_idx = -1;
            int src_state = -1;
            int dst_state = -1;
            std::pair<int64_t, int64_t> key{std::numeric_limits<int64_t>::max(),
                                            std::numeric_limits<int64_t>::max()};
        };
        Move best_move;
        for (auto bucket_idx : bucket_order) {
            const int src_state = owner[static_cast<std::size_t>(bucket_idx)];
            for (auto dst_state : compatible_states[static_cast<std::size_t>(bucket_idx)]) {
                if (dst_state == src_state) {
                    continue;
                }
                add_bucket_to_state(bucket_idx, src_state, -1);
                add_bucket_to_state(bucket_idx, dst_state, 1);
                auto candidate_key = imbalance_key();
                add_bucket_to_state(bucket_idx, dst_state, -1);
                add_bucket_to_state(bucket_idx, src_state, 1);
                if (candidate_key < current_key &&
                    (!best_move.valid ||
                     std::make_tuple(candidate_key.first, candidate_key.second,
                                     -bucket_weight[static_cast<std::size_t>(bucket_idx)], src_state, dst_state, bucket_idx) <
                         std::make_tuple(best_move.key.first, best_move.key.second,
                                         -bucket_weight[static_cast<std::size_t>(best_move.bucket_idx)],
                                         best_move.src_state, best_move.dst_state, best_move.bucket_idx))) {
                    best_move.valid = true;
                    best_move.bucket_idx = bucket_idx;
                    best_move.src_state = src_state;
                    best_move.dst_state = dst_state;
                    best_move.key = candidate_key;
                }
            }
        }
        if (!best_move.valid) {
            break;
        }
        add_bucket_to_state(best_move.bucket_idx, best_move.src_state, -1);
        add_bucket_to_state(best_move.bucket_idx, best_move.dst_state, 1);
        owner[static_cast<std::size_t>(best_move.bucket_idx)] = best_move.dst_state;
        moves++;
    }

    for (int bucket_idx = 0; bucket_idx < num_buckets; bucket_idx++) {
        const int state_idx = owner[static_cast<std::size_t>(bucket_idx)];
        assigned_buckets[static_cast<std::size_t>(state_idx)].emplace_back(bucket_idx / num_partitions,
                                                                           bucket_idx % num_partitions);
    }
    auto final_key = imbalance_key();
    SPDLOG_INFO("Round-balanced {}-GPU bucket assignment moved={} max_round_delta {} -> {} total_round_delta {} -> {}",
                active_devices, moves, initial_key.first, final_key.first, initial_key.second, final_key.second);

    vector<torch::Tensor> ret_edge_buckets_per_buffer;
    ret_edge_buckets_per_buffer.reserve(assigned_buckets.size());
    for (const auto &edge_buckets : assigned_buckets) {
        torch::Tensor tmp = torch::zeros({static_cast<int64_t>(edge_buckets.size()), 2}, torch::kInt64);
        for (int64_t idx = 0; idx < static_cast<int64_t>(edge_buckets.size()); idx++) {
            tmp[idx][0] = edge_buckets[static_cast<std::size_t>(idx)].first;
            tmp[idx][1] = edge_buckets[static_cast<std::size_t>(idx)].second;
        }
        ret_edge_buckets_per_buffer.emplace_back(tmp);
    }
    return ret_edge_buckets_per_buffer;
}

struct MultiGpuRoundGroup {
    std::vector<int64_t> states;
};

struct MultiGpuTransitionCost {
    int64_t host_bytes = 0;
    int64_t peer_bytes = 0;
    int64_t host_partitions = 0;
    int64_t peer_partitions = 0;
    int64_t lane_admits = 0;
    int64_t same_overlap = 0;
    int64_t max_lane_admits = 0;
    int64_t bad_lane_transitions = 0;
    long double score = 0.0L;
};

struct MultiGpuPathCost {
    int64_t host_bytes = 0;
    int64_t peer_bytes = 0;
    int64_t host_partitions = 0;
    int64_t peer_partitions = 0;
    int64_t lane_admits = 0;
    int64_t same_overlap = 0;
    int64_t max_lane_admits = 0;
    int64_t bad_lane_transitions = 0;
    int64_t first_transition_peer_partitions = 0;
    int64_t peer_partitions_over_budget = 0;
    int64_t lane_edge_imbalance = 0;
    long double score = 0.0L;
};

struct MultiGpuOrientedPath {
    std::vector<MultiGpuRoundGroup> ordered_groups;
    std::vector<MultiGpuRoundGroup> oriented_rounds;
    MultiGpuPathCost cost;
};

bool multi_gpu_path_cost_less(const MultiGpuPathCost &lhs, const MultiGpuPathCost &rhs) {
    if (lhs.bad_lane_transitions != rhs.bad_lane_transitions) {
        return lhs.bad_lane_transitions < rhs.bad_lane_transitions;
    }
    if (lhs.first_transition_peer_partitions != rhs.first_transition_peer_partitions) {
        return lhs.first_transition_peer_partitions < rhs.first_transition_peer_partitions;
    }
    if (lhs.peer_partitions_over_budget != rhs.peer_partitions_over_budget) {
        return lhs.peer_partitions_over_budget < rhs.peer_partitions_over_budget;
    }
    if (lhs.host_partitions != rhs.host_partitions) {
        return lhs.host_partitions < rhs.host_partitions;
    }
    if (lhs.lane_admits != rhs.lane_admits) {
        return lhs.lane_admits < rhs.lane_admits;
    }
    if (lhs.max_lane_admits != rhs.max_lane_admits) {
        return lhs.max_lane_admits < rhs.max_lane_admits;
    }
    if (lhs.same_overlap != rhs.same_overlap) {
        return lhs.same_overlap > rhs.same_overlap;
    }
    if (lhs.host_bytes != rhs.host_bytes) {
        return lhs.host_bytes < rhs.host_bytes;
    }
    if (lhs.peer_bytes != rhs.peer_bytes) {
        return lhs.peer_bytes < rhs.peer_bytes;
    }
    return lhs.score < rhs.score;
}

MultiGpuPathCost path_cost_from_transition(const MultiGpuTransitionCost &transition) {
    MultiGpuPathCost cost;
    cost.host_bytes = transition.host_bytes;
    cost.peer_bytes = transition.peer_bytes;
    cost.host_partitions = transition.host_partitions;
    cost.peer_partitions = transition.peer_partitions;
    cost.lane_admits = transition.lane_admits;
    cost.same_overlap = transition.same_overlap;
    cost.max_lane_admits = transition.max_lane_admits;
    cost.bad_lane_transitions = transition.bad_lane_transitions;
    cost.score = transition.score;
    return cost;
}

bool multi_gpu_round_group_valid(const MultiGpuRoundGroup &group,
                                 const std::vector<StateAccessSummary> &summaries,
                                 int active_devices) {
    if (active_devices <= 0 || group.states.size() != static_cast<std::size_t>(active_devices)) {
        return false;
    }
    std::vector<int64_t> sorted_states = group.states;
    std::sort(sorted_states.begin(), sorted_states.end());
    if (std::adjacent_find(sorted_states.begin(), sorted_states.end()) != sorted_states.end()) {
        return false;
    }
    for (auto state_idx : sorted_states) {
        if (state_idx < 0 || state_idx >= static_cast<int64_t>(summaries.size())) {
            return false;
        }
    }
    for (std::size_t i = 0; i < group.states.size(); i++) {
        for (std::size_t j = i + 1; j < group.states.size(); j++) {
            if (!states_disjoint(summaries[static_cast<std::size_t>(group.states[i])].partitions,
                                 summaries[static_cast<std::size_t>(group.states[j])].partitions)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<MultiGpuRoundGroup> multi_gpu_group_orientations(const MultiGpuRoundGroup &group) {
    std::vector<int64_t> states = group.states;
    std::sort(states.begin(), states.end());
    std::vector<MultiGpuRoundGroup> orientations;
    do {
        orientations.push_back(MultiGpuRoundGroup{states});
    } while (std::next_permutation(states.begin(), states.end()));
    return orientations;
}

MultiGpuTransitionCost multi_gpu_transition_cost(const MultiGpuRoundGroup &prev_round,
                                                 const MultiGpuRoundGroup &next_round,
                                                 const std::vector<StateAccessSummary> &summaries,
                                                 const std::vector<int64_t> &partition_row_counts,
                                                 const LaneMatchCostConfig &cfg,
                                                 int64_t bytes_per_row) {
    MultiGpuTransitionCost cost;
    if (prev_round.states.size() != next_round.states.size()) {
        cost.bad_lane_transitions = std::numeric_limits<int64_t>::max() / 4;
        cost.score = std::numeric_limits<long double>::infinity();
        return cost;
    }

    const std::size_t lanes = next_round.states.size();
    for (std::size_t lane = 0; lane < lanes; lane++) {
        const auto &same_prev = summaries[static_cast<std::size_t>(prev_round.states[lane])].partitions;
        const auto &needed = summaries[static_cast<std::size_t>(next_round.states[lane])].partitions;
        int64_t lane_admits = 0;
        for (int64_t partition_id : needed) {
            if (partition_in_sorted_list(same_prev, partition_id)) {
                cost.same_overlap++;
                continue;
            }

            lane_admits++;
            const int64_t bytes = partition_transfer_bytes(partition_id, partition_row_counts, bytes_per_row);
            bool peer_available = false;
            if (cfg.allow_peer_relay) {
                for (std::size_t peer_lane = 0; peer_lane < lanes; peer_lane++) {
                    if (peer_lane == lane) {
                        continue;
                    }
                    const auto &peer_prev = summaries[static_cast<std::size_t>(prev_round.states[peer_lane])].partitions;
                    if (partition_in_sorted_list(peer_prev, partition_id)) {
                        peer_available = true;
                        break;
                    }
                }
            }

            if (peer_available) {
                cost.peer_bytes += bytes;
                cost.peer_partitions++;
            } else {
                cost.host_bytes += bytes;
                cost.host_partitions++;
            }
        }
        cost.lane_admits += lane_admits;
        cost.max_lane_admits = std::max<int64_t>(cost.max_lane_admits, lane_admits);
        if (cfg.max_admits_per_transition >= 0 && lane_admits > cfg.max_admits_per_transition) {
            cost.bad_lane_transitions++;
        }
    }

    // Runtime-oriented Opt88 objective: keep transitions feasible for the
    // per-lane admit cap, then minimize CPU->GPU admits. Peer relay still
    // counts as lane movement, but it should not dominate host movement.
    cost.score = static_cast<long double>(cost.bad_lane_transitions) * 1000000000000.0L +
                 static_cast<long double>(cost.host_partitions) * 1000000.0L +
                 static_cast<long double>(cost.lane_admits) * 1000.0L +
                 static_cast<long double>(cost.max_lane_admits);
    return cost;
}

MultiGpuPathCost evaluate_multi_gpu_oriented_path(const std::vector<MultiGpuRoundGroup> &oriented_rounds,
                                                  const std::vector<StateAccessSummary> &summaries,
                                                  const std::vector<int64_t> &partition_row_counts,
                                                  const LaneMatchCostConfig &cfg,
                                                  int64_t bytes_per_row) {
    MultiGpuPathCost path_cost;
    if (oriented_rounds.empty()) {
        return path_cost;
    }

    const std::size_t lanes = oriented_rounds.front().states.size();
    std::vector<int64_t> lane_edges(lanes, 0);
    for (const auto &round : oriented_rounds) {
        for (std::size_t lane = 0; lane < round.states.size(); lane++) {
            lane_edges[lane] += summaries[static_cast<std::size_t>(round.states[lane])].total_bucket_edges;
        }
    }
    auto minmax_edges = std::minmax_element(lane_edges.begin(), lane_edges.end());
    if (minmax_edges.first != lane_edges.end() && minmax_edges.second != lane_edges.end()) {
        path_cost.lane_edge_imbalance = *minmax_edges.second - *minmax_edges.first;
    }

    for (std::size_t idx = 1; idx < oriented_rounds.size(); idx++) {
        auto step = multi_gpu_transition_cost(oriented_rounds[idx - 1], oriented_rounds[idx], summaries,
                                              partition_row_counts, cfg, bytes_per_row);
        path_cost.host_bytes += step.host_bytes;
        path_cost.peer_bytes += step.peer_bytes;
        path_cost.host_partitions += step.host_partitions;
        path_cost.peer_partitions += step.peer_partitions;
        if (idx == 1) {
            path_cost.first_transition_peer_partitions += step.peer_partitions;
        }
        path_cost.lane_admits += step.lane_admits;
        path_cost.same_overlap += step.same_overlap;
        path_cost.max_lane_admits = std::max(path_cost.max_lane_admits, step.max_lane_admits);
        path_cost.bad_lane_transitions += step.bad_lane_transitions;
        path_cost.score += step.score;
    }

    const int64_t peer_budget = optimal88_peer_handoff_budget(static_cast<int>(lanes));
    if (peer_budget >= 0) {
        path_cost.peer_partitions_over_budget = std::max<int64_t>(path_cost.peer_partitions - peer_budget, 0);
    }

    return path_cost;
}

MultiGpuTransitionCost best_multi_gpu_group_transition_cost(const MultiGpuRoundGroup &prev_group,
                                                            const MultiGpuRoundGroup &next_group,
                                                            const std::vector<StateAccessSummary> &summaries,
                                                            const std::vector<int64_t> &partition_row_counts,
                                                            const LaneMatchCostConfig &cfg,
                                                            int64_t bytes_per_row) {
    MultiGpuTransitionCost best;
    bool have_best = false;
    auto prev_orientations = multi_gpu_group_orientations(prev_group);
    auto next_orientations = multi_gpu_group_orientations(next_group);
    for (const auto &prev_orientation : prev_orientations) {
        for (const auto &next_orientation : next_orientations) {
            auto candidate = multi_gpu_transition_cost(prev_orientation, next_orientation, summaries,
                                                       partition_row_counts, cfg, bytes_per_row);
            auto candidate_key = path_cost_from_transition(candidate);
            auto best_key = path_cost_from_transition(best);
            if (!have_best || multi_gpu_path_cost_less(candidate_key, best_key)) {
                best = candidate;
                have_best = true;
            }
        }
    }
    return best;
}

MultiGpuOrientedPath orient_multi_gpu_group_order_dp(const std::vector<MultiGpuRoundGroup> &ordered_groups,
                                                     const std::vector<StateAccessSummary> &summaries,
                                                     const std::vector<int64_t> &partition_row_counts,
                                                     const LaneMatchCostConfig &cfg,
                                                     int64_t bytes_per_row) {
    MultiGpuOrientedPath result;
    result.ordered_groups = ordered_groups;
    if (ordered_groups.empty()) {
        return result;
    }

    std::vector<std::vector<MultiGpuRoundGroup>> orientations_by_round;
    orientations_by_round.reserve(ordered_groups.size());
    for (const auto &group : ordered_groups) {
        orientations_by_round.emplace_back(multi_gpu_group_orientations(group));
    }

    struct Cell {
        long double score = std::numeric_limits<long double>::infinity();
        int prev_orientation = -1;
    };

    std::vector<std::vector<Cell>> dp(ordered_groups.size());
    for (std::size_t idx = 0; idx < ordered_groups.size(); idx++) {
        dp[idx].resize(orientations_by_round[idx].size());
    }
    for (auto &cell : dp[0]) {
        cell.score = 0.0L;
    }

    for (std::size_t round_idx = 1; round_idx < ordered_groups.size(); round_idx++) {
        for (std::size_t curr_o = 0; curr_o < orientations_by_round[round_idx].size(); curr_o++) {
            const auto &curr = orientations_by_round[round_idx][curr_o];
            for (std::size_t prev_o = 0; prev_o < orientations_by_round[round_idx - 1].size(); prev_o++) {
                if (dp[round_idx - 1][prev_o].score == std::numeric_limits<long double>::infinity()) {
                    continue;
                }
                const auto &prev = orientations_by_round[round_idx - 1][prev_o];
                auto step = multi_gpu_transition_cost(prev, curr, summaries, partition_row_counts, cfg, bytes_per_row);
                long double candidate_score = dp[round_idx - 1][prev_o].score + step.score;
                if (candidate_score < dp[round_idx][curr_o].score) {
                    dp[round_idx][curr_o].score = candidate_score;
                    dp[round_idx][curr_o].prev_orientation = static_cast<int>(prev_o);
                }
            }
        }
    }

    int orientation = 0;
    long double best_score = std::numeric_limits<long double>::infinity();
    for (std::size_t orient_idx = 0; orient_idx < dp.back().size(); orient_idx++) {
        if (dp.back()[orient_idx].score < best_score) {
            best_score = dp.back()[orient_idx].score;
            orientation = static_cast<int>(orient_idx);
        }
    }

    result.oriented_rounds.assign(ordered_groups.size(), {});
    for (int round_idx = static_cast<int>(ordered_groups.size()) - 1; round_idx >= 0; round_idx--) {
        result.oriented_rounds[static_cast<std::size_t>(round_idx)] =
            orientations_by_round[static_cast<std::size_t>(round_idx)][static_cast<std::size_t>(orientation)];
        orientation = round_idx > 0
                          ? dp[static_cast<std::size_t>(round_idx)][static_cast<std::size_t>(orientation)].prev_orientation
                          : 0;
        if (orientation < 0) {
            orientation = 0;
        }
    }
    result.cost = evaluate_multi_gpu_oriented_path(result.oriented_rounds, summaries, partition_row_counts, cfg, bytes_per_row);
    return result;
}

std::vector<MultiGpuRoundGroup> greedy_order_multi_gpu_groups(const std::vector<MultiGpuRoundGroup> &groups,
                                                              const std::vector<StateAccessSummary> &summaries,
                                                              const std::vector<int64_t> &partition_row_counts,
                                                              const LaneMatchCostConfig &cfg,
                                                              int64_t bytes_per_row,
                                                              std::mt19937_64 &rng) {
    if (groups.empty()) {
        return {};
    }

    std::vector<uint8_t> used(groups.size(), 0);
    std::vector<MultiGpuRoundGroup> ordered;
    ordered.reserve(groups.size());
    std::uniform_int_distribution<int> start_dist(0, static_cast<int>(groups.size()) - 1);
    int current = start_dist(rng);
    used[static_cast<std::size_t>(current)] = 1;
    ordered.emplace_back(groups[static_cast<std::size_t>(current)]);

    while (ordered.size() < groups.size()) {
        int best_idx = -1;
        MultiGpuTransitionCost best_cost;
        bool have_best = false;
        for (int candidate = 0; candidate < static_cast<int>(groups.size()); candidate++) {
            if (used[static_cast<std::size_t>(candidate)] != 0) {
                continue;
            }
            auto cost = best_multi_gpu_group_transition_cost(ordered.back(), groups[static_cast<std::size_t>(candidate)],
                                                             summaries, partition_row_counts, cfg, bytes_per_row);
            auto current_key = path_cost_from_transition(cost);
            auto best_key = path_cost_from_transition(best_cost);
            if (!have_best || multi_gpu_path_cost_less(current_key, best_key)) {
                best_idx = candidate;
                best_cost = cost;
                have_best = true;
            }
        }
        if (best_idx < 0) {
            break;
        }
        used[static_cast<std::size_t>(best_idx)] = 1;
        ordered.emplace_back(groups[static_cast<std::size_t>(best_idx)]);
    }

    if (ordered.size() != groups.size()) {
        return groups;
    }
    return ordered;
}

MultiGpuOrientedPath improve_multi_gpu_group_order(std::vector<MultiGpuRoundGroup> ordered_groups,
                                                   const std::vector<StateAccessSummary> &summaries,
                                                   const std::vector<int64_t> &partition_row_counts,
                                                   const LaneMatchCostConfig &cfg,
                                                   int64_t bytes_per_row,
                                                   int max_passes) {
    MultiGpuOrientedPath best =
        orient_multi_gpu_group_order_dp(ordered_groups, summaries, partition_row_counts, cfg, bytes_per_row);
    const int n = static_cast<int>(ordered_groups.size());
    for (int pass = 0; pass < std::max(0, max_passes); pass++) {
        bool improved = false;
        for (int begin = 0; begin < n - 2 && !improved; begin++) {
            for (int end = begin + 1; end < n && !improved; end++) {
                std::vector<MultiGpuRoundGroup> candidate = ordered_groups;
                std::reverse(candidate.begin() + begin, candidate.begin() + end + 1);
                auto candidate_path =
                    orient_multi_gpu_group_order_dp(candidate, summaries, partition_row_counts, cfg, bytes_per_row);
                if (multi_gpu_path_cost_less(candidate_path.cost, best.cost)) {
                    ordered_groups = std::move(candidate);
                    best = std::move(candidate_path);
                    improved = true;
                }
            }
        }
        if (!improved) {
            break;
        }
    }
    return best;
}

std::vector<MultiGpuRoundGroup> groups_from_permutation(const std::vector<int64_t> &permutation,
                                                        int active_devices,
                                                        const std::vector<StateAccessSummary> &summaries) {
    std::vector<MultiGpuRoundGroup> groups;
    if (active_devices <= 1 || permutation.empty() || permutation.size() % static_cast<std::size_t>(active_devices) != 0) {
        return groups;
    }
    groups.reserve(permutation.size() / static_cast<std::size_t>(active_devices));
    std::vector<uint8_t> seen(summaries.size(), 0);
    for (std::size_t offset = 0; offset < permutation.size(); offset += static_cast<std::size_t>(active_devices)) {
        MultiGpuRoundGroup group;
        group.states.reserve(active_devices);
        for (int lane = 0; lane < active_devices; lane++) {
            int64_t state_idx = permutation[offset + static_cast<std::size_t>(lane)];
            if (state_idx < 0 || state_idx >= static_cast<int64_t>(summaries.size()) ||
                seen[static_cast<std::size_t>(state_idx)] != 0) {
                return {};
            }
            seen[static_cast<std::size_t>(state_idx)] = 1;
            group.states.emplace_back(state_idx);
        }
        if (!multi_gpu_round_group_valid(group, summaries, active_devices)) {
            return {};
        }
        groups.emplace_back(std::move(group));
    }
    return groups;
}

std::vector<MultiGpuRoundGroup> build_farthest_multi_gpu_grouping(const std::vector<StateAccessSummary> &summaries,
                                                                  int active_devices) {
    std::vector<int64_t> remaining(summaries.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<MultiGpuRoundGroup> groups;
    groups.reserve(summaries.size() / static_cast<std::size_t>(active_devices));

    while (!remaining.empty()) {
        auto anchor_it = std::min_element(remaining.begin(), remaining.end(), [&](int64_t lhs, int64_t rhs) {
            int lhs_degree = 0;
            int rhs_degree = 0;
            for (auto candidate : remaining) {
                if (candidate == lhs || candidate == rhs) {
                    continue;
                }
                lhs_degree += states_disjoint(summaries[static_cast<std::size_t>(lhs)].partitions,
                                              summaries[static_cast<std::size_t>(candidate)].partitions) ? 1 : 0;
                rhs_degree += states_disjoint(summaries[static_cast<std::size_t>(rhs)].partitions,
                                              summaries[static_cast<std::size_t>(candidate)].partitions) ? 1 : 0;
            }
            if (lhs_degree != rhs_degree) {
                return lhs_degree < rhs_degree;
            }
            return lhs < rhs;
        });

        MultiGpuRoundGroup group;
        group.states.emplace_back(*anchor_it);
        while (group.states.size() < static_cast<std::size_t>(active_devices)) {
            int64_t best_candidate = -1;
            auto best_key = std::make_tuple(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::min(),
                                            std::numeric_limits<int64_t>::min());
            for (auto candidate : remaining) {
                if (std::find(group.states.begin(), group.states.end(), candidate) != group.states.end()) {
                    continue;
                }
                bool compatible = true;
                int64_t min_distance = std::numeric_limits<int64_t>::max();
                for (auto chosen : group.states) {
                    if (!states_disjoint(summaries[static_cast<std::size_t>(candidate)].partitions,
                                         summaries[static_cast<std::size_t>(chosen)].partitions)) {
                        compatible = false;
                        break;
                    }
                    min_distance = std::min<int64_t>(min_distance, std::llabs(candidate - chosen));
                }
                if (!compatible) {
                    continue;
                }
                auto key = std::make_tuple(min_distance, summaries[static_cast<std::size_t>(candidate)].total_bucket_edges,
                                           candidate);
                if (key > best_key) {
                    best_key = key;
                    best_candidate = candidate;
                }
            }
            if (best_candidate < 0) {
                return {};
            }
            group.states.emplace_back(best_candidate);
        }

        if (!multi_gpu_round_group_valid(group, summaries, active_devices)) {
            return {};
        }
        for (auto state_idx : group.states) {
            remaining.erase(std::remove(remaining.begin(), remaining.end(), state_idx), remaining.end());
        }
        groups.emplace_back(std::move(group));
    }
    return groups;
}

std::vector<MultiGpuRoundGroup> build_random_multi_gpu_grouping(const std::vector<StateAccessSummary> &summaries,
                                                                int active_devices,
                                                                std::mt19937_64 &rng) {
    std::vector<int64_t> remaining(summaries.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    std::shuffle(remaining.begin(), remaining.end(), rng);
    std::vector<MultiGpuRoundGroup> groups;
    groups.reserve(summaries.size() / static_cast<std::size_t>(active_devices));

    while (!remaining.empty()) {
        auto anchor_it = std::min_element(remaining.begin(), remaining.end(), [&](int64_t lhs, int64_t rhs) {
            int lhs_degree = 0;
            int rhs_degree = 0;
            for (auto candidate : remaining) {
                if (candidate == lhs || candidate == rhs) {
                    continue;
                }
                lhs_degree += states_disjoint(summaries[static_cast<std::size_t>(lhs)].partitions,
                                              summaries[static_cast<std::size_t>(candidate)].partitions) ? 1 : 0;
                rhs_degree += states_disjoint(summaries[static_cast<std::size_t>(rhs)].partitions,
                                              summaries[static_cast<std::size_t>(candidate)].partitions) ? 1 : 0;
            }
            if (lhs_degree != rhs_degree) {
                return lhs_degree < rhs_degree;
            }
            return lhs < rhs;
        });

        MultiGpuRoundGroup group;
        group.states.emplace_back(*anchor_it);
        while (group.states.size() < static_cast<std::size_t>(active_devices)) {
            std::vector<int64_t> candidates;
            for (auto candidate : remaining) {
                if (std::find(group.states.begin(), group.states.end(), candidate) != group.states.end()) {
                    continue;
                }
                bool compatible = true;
                for (auto chosen : group.states) {
                    if (!states_disjoint(summaries[static_cast<std::size_t>(candidate)].partitions,
                                         summaries[static_cast<std::size_t>(chosen)].partitions)) {
                        compatible = false;
                        break;
                    }
                }
                if (compatible) {
                    candidates.emplace_back(candidate);
                }
            }
            if (candidates.empty()) {
                return {};
            }
            std::shuffle(candidates.begin(), candidates.end(), rng);
            group.states.emplace_back(candidates.front());
        }

        if (!multi_gpu_round_group_valid(group, summaries, active_devices)) {
            return {};
        }
        for (auto state_idx : group.states) {
            remaining.erase(std::remove(remaining.begin(), remaining.end(), state_idx), remaining.end());
        }
        groups.emplace_back(std::move(group));
    }
    return groups;
}

std::vector<int64_t> getOptimal88MultiGpuLaneMatchedPermutation(const vector<torch::Tensor>& buffer_states,
                                                                const vector<torch::Tensor>& edge_buckets_per_buffer,
                                                                const vector<int64_t> &edge_bucket_sizes,
                                                                const std::vector<int64_t> &partition_row_counts,
                                                                const PlanEmbeddingLayout &layout,
                                                                int active_devices) {
    if (buffer_states.size() != 88 || edge_buckets_per_buffer.size() != buffer_states.size() ||
        active_devices <= 2 || active_devices > 4 ||
        buffer_states.size() % static_cast<std::size_t>(active_devices) != 0) {
        return {};
    }

    auto summaries = build_state_access_summaries(buffer_states, edge_buckets_per_buffer, edge_bucket_sizes);
    LaneMatchCostConfig cfg = lane_match_cost_config_from_env();
    cfg.allow_peer_relay = stateflow_env_bool("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "STATEFLOW_ALLOW_PEER_RELAY",
                                              stateflow_allow_peer_relay_default());
    if (cfg.max_admits_per_transition < 0) {
        cfg.max_admits_per_transition = 3;
    }
    const int64_t bytes_per_row = plan_embedding_bytes_per_row(layout);
    const int restarts = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MULTIGPU_RESTARTS", nullptr,
                                                 stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_2GPU_RESTARTS", nullptr, 64))));
    const int two_opt_passes = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MULTIGPU_2OPT_PASSES", nullptr,
                                                 stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_2GPU_2OPT_PASSES", nullptr, 8))));
    const int64_t seed = stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MULTIGPU_SEED", nullptr, 31);
    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    MultiGpuOrientedPath best_path;
    bool have_best = false;
    auto try_grouping = [&](std::vector<MultiGpuRoundGroup> groups) {
        if (groups.size() * static_cast<std::size_t>(active_devices) != buffer_states.size()) {
            return;
        }
        for (const auto &group : groups) {
            if (!multi_gpu_round_group_valid(group, summaries, active_devices)) {
                return;
            }
        }
        auto ordered = greedy_order_multi_gpu_groups(groups, summaries, partition_row_counts, cfg, bytes_per_row, rng);
        auto candidate = improve_multi_gpu_group_order(std::move(ordered), summaries, partition_row_counts, cfg,
                                                       bytes_per_row, two_opt_passes);
        if (!have_best || multi_gpu_path_cost_less(candidate.cost, best_path.cost)) {
            best_path = std::move(candidate);
            have_best = true;
        }
    };

    auto disjoint_perm = getDisjointBufferStatePermutation(buffer_states, active_devices);
    try_grouping(groups_from_permutation(disjoint_perm, active_devices, summaries));

    auto access_perm =
        getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, active_devices, partition_row_counts, layout);
    try_grouping(groups_from_permutation(access_perm, active_devices, summaries));

    try_grouping(build_farthest_multi_gpu_grouping(summaries, active_devices));
    for (int restart = 0; restart < restarts; restart++) {
        try_grouping(build_random_multi_gpu_grouping(summaries, active_devices, rng));
    }

    if (!have_best || best_path.oriented_rounds.size() * static_cast<std::size_t>(active_devices) != buffer_states.size()) {
        return {};
    }

    std::vector<uint8_t> seen(buffer_states.size(), 0);
    std::vector<int64_t> permutation;
    permutation.reserve(buffer_states.size());
    for (const auto &round : best_path.oriented_rounds) {
        if (!multi_gpu_round_group_valid(round, summaries, active_devices)) {
            return {};
        }
        for (auto state_idx : round.states) {
            if (state_idx < 0 || state_idx >= static_cast<int64_t>(buffer_states.size()) ||
                seen[static_cast<std::size_t>(state_idx)] != 0) {
                return {};
            }
            seen[static_cast<std::size_t>(state_idx)] = 1;
            permutation.emplace_back(state_idx);
        }
    }

    SPDLOG_INFO(
        "Optimal88 {}-GPU lane schedule selected rounds={} microstates={} transition_host_admits={} peer_handoffs={} peer_handoffs_over_budget={} lane_transition_admits={} same_lane_overlap={} max_lane_admits={} bad_lane_transitions={} first_transition_peer_handoffs={} host_bytes={} peer_bytes={} lane_edge_imbalance={}",
        active_devices, best_path.oriented_rounds.size(), buffer_states.size(), best_path.cost.host_partitions,
        best_path.cost.peer_partitions, best_path.cost.peer_partitions_over_budget,
        best_path.cost.lane_admits, best_path.cost.same_overlap,
        best_path.cost.max_lane_admits, best_path.cost.bad_lane_transitions,
        best_path.cost.first_transition_peer_partitions,
        best_path.cost.host_bytes, best_path.cost.peer_bytes, best_path.cost.lane_edge_imbalance);

    return permutation;
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
            if (bounded_greedy_cover_q4_enabled()) {
                return getBoundedGreedyCoverEdgeBucketOrdering(num_partitions, buffer_capacity, {});
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

    return std::make_tuple(std::move(ret_buffer_states), std::move(ret_edge_buckets_per_buffer));
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

int64_t mark_state_buckets_covered(const std::vector<int> &state,
                                   std::vector<uint8_t> &covered,
                                   int num_partitions);

std::vector<std::vector<int>> load_bounded_state_order_file(const char *path,
                                                            int num_partitions,
                                                            int buffer_capacity) {
    if (path == nullptr || std::string(path).empty()) {
        return {};
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        SPDLOG_WARN("Could not open GEGE_BOUNDED_STATE_ORDER_FILE='{}'; using generated bounded schedule", path);
        return {};
    }

    std::vector<std::vector<int>> states;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        line_number++;
        std::size_t begin = line.find("state=[");
        if (begin == std::string::npos) {
            begin = line.find('[');
        } else {
            begin += std::string("state=").size();
        }
        if (begin == std::string::npos) {
            continue;
        }
        std::size_t end = line.find(']', begin);
        if (end == std::string::npos) {
            continue;
        }

        std::vector<int> state;
        std::string token;
        for (std::size_t idx = begin + 1; idx < end; idx++) {
            char ch = line[idx];
            if ((ch >= '0' && ch <= '9') || ch == '-') {
                token.push_back(ch);
            } else if (!token.empty()) {
                state.emplace_back(std::stoi(token));
                token.clear();
            }
        }
        if (!token.empty()) {
            state.emplace_back(std::stoi(token));
        }

        if (state.empty()) {
            continue;
        }
        if (static_cast<int>(state.size()) != buffer_capacity) {
            SPDLOG_WARN("Ignoring GEGE_BOUNDED_STATE_ORDER_FILE='{}': line {} has {} partitions, expected {}",
                        path, line_number, state.size(), buffer_capacity);
            return {};
        }
        std::vector<int> sorted_state = state;
        std::sort(sorted_state.begin(), sorted_state.end());
        if (std::adjacent_find(sorted_state.begin(), sorted_state.end()) != sorted_state.end()) {
            SPDLOG_WARN("Ignoring GEGE_BOUNDED_STATE_ORDER_FILE='{}': line {} has duplicate partitions",
                        path, line_number);
            return {};
        }
        for (int partition : state) {
            if (partition < 0 || partition >= num_partitions) {
                SPDLOG_WARN("Ignoring GEGE_BOUNDED_STATE_ORDER_FILE='{}': line {} has partition {} outside [0,{})",
                            path, line_number, partition, num_partitions);
                return {};
            }
        }
        states.emplace_back(std::move(state));
    }

    if (states.empty()) {
        SPDLOG_WARN("GEGE_BOUNDED_STATE_ORDER_FILE='{}' did not contain any states; using generated bounded schedule", path);
        return {};
    }

    std::vector<uint8_t> covered(static_cast<std::size_t>(num_partitions) * static_cast<std::size_t>(num_partitions), 0);
    int64_t covered_count = 0;
    for (const auto &state : states) {
        covered_count += mark_state_buckets_covered(state, covered, num_partitions);
    }
    const int64_t total_buckets = static_cast<int64_t>(num_partitions) * static_cast<int64_t>(num_partitions);
    if (covered_count != total_buckets) {
        SPDLOG_WARN("Ignoring GEGE_BOUNDED_STATE_ORDER_FILE='{}': covers {} directed buckets, expected {}",
                    path, covered_count, total_buckets);
        return {};
    }

    SPDLOG_INFO("Loaded bounded state order from '{}' states={} total_buckets={}",
                path, states.size(), covered_count);
    return states;
}

std::vector<std::vector<int>> build_greedy_cover_state_set(int num_partitions, int buffer_capacity) {
    std::vector<std::vector<int>> candidates;
    std::vector<int> current;
    build_greedy_cover_candidates(num_partitions, buffer_capacity, 0, current, candidates);
    if (candidates.empty()) {
        return {};
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
            SPDLOG_WARN("Greedy cover state-set generation stopped early: covered={} total={}", covered_count, total_buckets);
            break;
        }

        auto state = candidates[best_idx];
        covered_count += mark_state_buckets_covered(state, covered, num_partitions);
        buffer_states.emplace_back(std::move(state));
    }

    return buffer_states;
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> bounded_path_score(
    const std::vector<int> &order,
    const std::vector<std::vector<int>> &overlap) {
    int64_t zero_overlap = 0;
    int64_t total_overlap = 0;
    int64_t overlap_three = 0;
    int64_t overlap_two = 0;
    int64_t overlap_one = 0;

    for (std::size_t idx = 1; idx < order.size(); idx++) {
        int value = overlap[order[idx - 1]][order[idx]];
        total_overlap += value;
        if (value == 0) {
            zero_overlap++;
        } else if (value == 1) {
            overlap_one++;
        } else if (value == 2) {
            overlap_two++;
        } else if (value == 3) {
            overlap_three++;
        }
    }

    return std::make_tuple(-zero_overlap, total_overlap, overlap_three, overlap_two, -overlap_one);
}

double bounded_path_score_scalar(const std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> &score) {
    return static_cast<double>(std::get<0>(score)) * 1000000.0 +
           static_cast<double>(std::get<1>(score)) * 1000.0 +
           static_cast<double>(std::get<2>(score)) * 30.0 +
           static_cast<double>(std::get<3>(score));
}

std::vector<int> build_bounded_greedy_path_once(const std::vector<std::vector<int>> &overlap,
                                                std::mt19937_64 &rng,
                                                int start_idx,
                                                int candidate_pool,
                                                int min_overlap) {
    const int num_states = static_cast<int>(overlap.size());
    if (num_states == 0) {
        return {};
    }

    std::vector<uint8_t> used(num_states, 0);
    std::vector<int> order;
    order.reserve(num_states);
    order.emplace_back(start_idx);
    used[start_idx] = 1;

    while (static_cast<int>(order.size()) < num_states) {
        struct Option {
            int overlap = 0;
            int future_degree = 0;
            uint64_t jitter = 0;
            bool prepend = false;
            int state_idx = -1;
        };

        std::vector<Option> options;
        options.reserve(num_states * 2);
        for (int side = 0; side < 2; side++) {
            int endpoint = side == 0 ? order.front() : order.back();
            for (int candidate = 0; candidate < num_states; candidate++) {
                if (used[candidate] != 0) {
                    continue;
                }
                int shared = overlap[endpoint][candidate];
                if (shared < min_overlap) {
                    continue;
                }

                int future_degree = 0;
                for (int other = 0; other < num_states; other++) {
                    if (other == candidate || used[other] != 0) {
                        continue;
                    }
                    if (overlap[candidate][other] >= min_overlap) {
                        future_degree++;
                    }
                }
                options.push_back(Option{shared, future_degree, rng(), side == 0, candidate});
            }
        }

        if (options.empty()) {
            return {};
        }

        std::sort(options.begin(), options.end(), [](const Option &lhs, const Option &rhs) {
            return std::make_tuple(lhs.overlap, lhs.future_degree, lhs.jitter) >
                   std::make_tuple(rhs.overlap, rhs.future_degree, rhs.jitter);
        });

        int limit = std::min<int>(std::max(candidate_pool, 1), options.size());
        std::uniform_int_distribution<int> pick_dist(0, limit - 1);
        const auto chosen = options[pick_dist(rng)];
        used[chosen.state_idx] = 1;
        if (chosen.prepend) {
            order.insert(order.begin(), chosen.state_idx);
        } else {
            order.emplace_back(chosen.state_idx);
        }
    }

    return order;
}

bool bounded_path_respects_min_overlap(const std::vector<int> &order,
                                       const std::vector<std::vector<int>> &overlap,
                                       int min_overlap) {
    if (order.size() <= 1) {
        return true;
    }
    for (std::size_t idx = 1; idx < order.size(); idx++) {
        if (overlap[order[idx - 1]][order[idx]] < min_overlap) {
            return false;
        }
    }
    return true;
}

int path_internal_overlap(const std::vector<int> &path,
                          const std::vector<std::vector<int>> &overlap) {
    int total = 0;
    for (std::size_t idx = 1; idx < path.size(); idx++) {
        total += overlap[path[idx - 1]][path[idx]];
    }
    return total;
}

struct BoundedPathRuntimeScore {
    int64_t below_min_overlap = 0;
    int64_t total_overlap = 0;
    int64_t strong_transition_count = 0;
    int64_t max_min_overlap_run = 0;
    int64_t min_overlap_run_sum_sq = 0;
    int64_t max_window_admits = 0;
    int64_t short_evict_reuse_count3 = 0;
    int64_t short_evict_reuse_count5 = 0;
    int64_t short_evict_reuse_transitions3 = 0;
    int64_t capped_evict_reuse_distance_sum = 0;
};

BoundedPathRuntimeScore bounded_path_runtime_score(const std::vector<int> &order,
                                                   const std::vector<std::vector<int>> &overlap,
                                                   const std::vector<std::vector<int>> &states,
                                                   int min_overlap,
                                                   int buffer_capacity) {
    BoundedPathRuntimeScore score;
    int64_t current_min_run = 0;
    std::vector<int64_t> admits;
    admits.reserve(order.size() > 0 ? order.size() - 1 : 0);

    for (std::size_t idx = 1; idx < order.size(); idx++) {
        const int shared = overlap[order[idx - 1]][order[idx]];
        score.total_overlap += shared;
        if (shared < min_overlap) {
            score.below_min_overlap += 1;
        }
        if (shared > min_overlap) {
            score.strong_transition_count += 1;
        }
        if (shared == min_overlap) {
            current_min_run += 1;
        } else {
            if (current_min_run > 0) {
                score.max_min_overlap_run = std::max(score.max_min_overlap_run, current_min_run);
                score.min_overlap_run_sum_sq += current_min_run * current_min_run;
                current_min_run = 0;
            }
        }
        admits.emplace_back(std::max<int64_t>(0, static_cast<int64_t>(buffer_capacity - shared)));
    }
    if (current_min_run > 0) {
        score.max_min_overlap_run = std::max(score.max_min_overlap_run, current_min_run);
        score.min_overlap_run_sum_sq += current_min_run * current_min_run;
    }

    const int window = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_RUNTIME_BALANCE_WINDOW", nullptr, 10)));
    int64_t window_sum = 0;
    for (std::size_t idx = 0; idx < admits.size(); idx++) {
        window_sum += admits[idx];
        if (idx >= static_cast<std::size_t>(window)) {
            window_sum -= admits[idx - static_cast<std::size_t>(window)];
        }
        score.max_window_admits = std::max(score.max_window_admits, window_sum);
    }

    const int64_t short_reuse_window3 =
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_RUNTIME_SHORT_REUSE_WINDOW3", nullptr, 3));
    const int64_t short_reuse_window5 =
        std::max<int64_t>(short_reuse_window3,
                          stateflow_env_int64("GEGE_BOUNDED_RUNTIME_SHORT_REUSE_WINDOW5", nullptr, 5));
    const int64_t reuse_distance_cap =
        std::max<int64_t>(short_reuse_window5, stateflow_env_int64("GEGE_BOUNDED_RUNTIME_REUSE_DISTANCE_CAP", nullptr, 12));
    for (std::size_t idx = 0; idx + 1 < order.size(); idx++) {
        bool transition_has_short_reuse = false;
        for (auto partition : states[order[idx]]) {
            if (std::find(states[order[idx + 1]].begin(), states[order[idx + 1]].end(), partition) !=
                states[order[idx + 1]].end()) {
                continue;
            }

            int64_t next_use_distance = std::numeric_limits<int64_t>::max() / 4;
            for (std::size_t future = idx + 2; future < order.size(); future++) {
                if (std::find(states[order[future]].begin(), states[order[future]].end(), partition) !=
                    states[order[future]].end()) {
                    next_use_distance = static_cast<int64_t>(future - (idx + 1));
                    break;
                }
            }
            if (next_use_distance <= short_reuse_window3) {
                score.short_evict_reuse_count3 += 1;
                transition_has_short_reuse = true;
            }
            if (next_use_distance <= short_reuse_window5) {
                score.short_evict_reuse_count5 += 1;
            }
            score.capped_evict_reuse_distance_sum += std::min<int64_t>(next_use_distance, reuse_distance_cap);
        }
        if (transition_has_short_reuse) {
            score.short_evict_reuse_transitions3 += 1;
        }
    }
    return score;
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
           int64_t, int64_t, int64_t, int64_t, int64_t>
bounded_path_runtime_score_key(const BoundedPathRuntimeScore &score, int64_t target_min_overlap_run) {
    const int64_t run_over_target = std::max<int64_t>(score.max_min_overlap_run - target_min_overlap_run, 0);
    return std::make_tuple(-score.below_min_overlap,
                           score.total_overlap,
                           score.strong_transition_count,
                           -run_over_target,
                           -score.short_evict_reuse_count3,
                           -score.short_evict_reuse_transitions3,
                           -score.short_evict_reuse_count5,
                           score.capped_evict_reuse_distance_sum,
                           -score.max_min_overlap_run,
                           -score.min_overlap_run_sum_sq,
                           -score.max_window_admits);
}

double bounded_path_runtime_score_scalar(const BoundedPathRuntimeScore &score,
                                         int64_t target_min_overlap_run) {
    const int64_t run_over_target = std::max<int64_t>(score.max_min_overlap_run - target_min_overlap_run, 0);
    return static_cast<double>(-score.below_min_overlap) * 1.0e12 +
           static_cast<double>(score.total_overlap) * 1.0e9 +
           static_cast<double>(score.strong_transition_count) * 1.0e6 -
           static_cast<double>(run_over_target) * 1.0e5 -
           static_cast<double>(score.short_evict_reuse_count3) * 1.0e4 -
           static_cast<double>(score.short_evict_reuse_transitions3) * 1.0e3 -
           static_cast<double>(score.short_evict_reuse_count5) * 100.0 +
           static_cast<double>(score.capped_evict_reuse_distance_sum) -
           static_cast<double>(score.max_min_overlap_run) * 10.0 -
           static_cast<double>(score.min_overlap_run_sum_sq) -
           static_cast<double>(score.max_window_admits);
}

std::vector<int> improve_bounded_path_runtime_balance(std::vector<int> order,
                                                      const std::vector<std::vector<int>> &overlap,
                                                      const std::vector<std::vector<int>> &states,
                                                      int min_overlap,
                                                      int buffer_capacity) {
    if (order.size() <= 2) {
        return order;
    }

    const int64_t iterations =
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_RUNTIME_BALANCE_ITERS", nullptr, 200000));
    if (iterations == 0) {
        return order;
    }

    const int64_t target_min_overlap_run =
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_RUNTIME_BALANCE_TARGET_RUN", nullptr, 6));
    const int64_t seed = stateflow_env_int64("GEGE_BOUNDED_RUNTIME_BALANCE_SEED", nullptr, 12345);
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<int> current = order;
    auto current_score = bounded_path_runtime_score(current, overlap, states, min_overlap, buffer_capacity);
    auto current_key = bounded_path_runtime_score_key(current_score, target_min_overlap_run);
    std::vector<int> best = current;
    auto best_score = current_score;
    auto best_key = current_key;

    auto validate_candidate = [&](const std::vector<int> &candidate) {
        return static_cast<int>(candidate.size()) == static_cast<int>(order.size()) &&
               bounded_path_respects_min_overlap(candidate, overlap, min_overlap);
    };

    const std::vector<int> segment_lengths{1, 1, 1, 2, 2, 3, 4, 5, 8, 12};
    for (int64_t iter = 0; iter < iterations; iter++) {
        std::vector<int> candidate = current;
        const int op = static_cast<int>(rng() % 4);
        const int n = static_cast<int>(candidate.size());

        if (op == 0) {
            const int segment_length = segment_lengths[static_cast<std::size_t>(rng() % segment_lengths.size())];
            if (segment_length >= n) {
                continue;
            }
            std::uniform_int_distribution<int> begin_dist(0, n - segment_length);
            const int begin = begin_dist(rng);
            std::vector<int> segment(candidate.begin() + begin, candidate.begin() + begin + segment_length);
            std::vector<int> rest;
            rest.reserve(candidate.size() - segment.size());
            rest.insert(rest.end(), candidate.begin(), candidate.begin() + begin);
            rest.insert(rest.end(), candidate.begin() + begin + segment_length, candidate.end());
            if ((rng() % 5) == 0) {
                std::reverse(segment.begin(), segment.end());
            }
            if (!bounded_path_respects_min_overlap(segment, overlap, min_overlap)) {
                continue;
            }
            std::uniform_int_distribution<int> pos_dist(0, static_cast<int>(rest.size()));
            const int position = pos_dist(rng);
            if (position > 0 && overlap[rest[position - 1]][segment.front()] < min_overlap) {
                continue;
            }
            if (position < static_cast<int>(rest.size()) && overlap[segment.back()][rest[position]] < min_overlap) {
                continue;
            }
            candidate.clear();
            candidate.reserve(order.size());
            candidate.insert(candidate.end(), rest.begin(), rest.begin() + position);
            candidate.insert(candidate.end(), segment.begin(), segment.end());
            candidate.insert(candidate.end(), rest.begin() + position, rest.end());
        } else if (op == 1) {
            std::uniform_int_distribution<int> index_dist(0, n - 1);
            int left = index_dist(rng);
            int right = index_dist(rng);
            if (left == right) {
                continue;
            }
            if (left > right) {
                std::swap(left, right);
            }
            if (right - left > 20) {
                continue;
            }
            std::reverse(candidate.begin() + left, candidate.begin() + right + 1);
        } else if (op == 2) {
            std::uniform_int_distribution<int> index_dist(0, n - 1);
            int left = index_dist(rng);
            int right = index_dist(rng);
            if (left == right) {
                continue;
            }
            std::swap(candidate[left], candidate[right]);
        } else {
            std::uniform_int_distribution<int> index_dist(0, n - 1);
            int removed_idx = index_dist(rng);
            int removed = candidate[removed_idx];
            candidate.erase(candidate.begin() + removed_idx);
            std::vector<int> positions;
            positions.reserve(candidate.size() + 1);
            for (int position = 0; position <= static_cast<int>(candidate.size()); position++) {
                if (position > 0 && overlap[candidate[position - 1]][removed] < min_overlap) {
                    continue;
                }
                if (position < static_cast<int>(candidate.size()) && overlap[removed][candidate[position]] < min_overlap) {
                    continue;
                }
                positions.emplace_back(position);
            }
            if (positions.empty()) {
                continue;
            }
            const int position = positions[static_cast<std::size_t>(rng() % positions.size())];
            candidate.insert(candidate.begin() + position, removed);
        }

        if (!validate_candidate(candidate)) {
            continue;
        }

        auto candidate_score = bounded_path_runtime_score(candidate, overlap, states, min_overlap, buffer_capacity);
        auto candidate_key = bounded_path_runtime_score_key(candidate_score, target_min_overlap_run);
        if (candidate_key >= current_key) {
            current = std::move(candidate);
            current_score = candidate_score;
            current_key = candidate_key;
            if (current_key > best_key) {
                best = current;
                best_score = current_score;
                best_key = current_key;
            }
            continue;
        }

        const double delta =
            bounded_path_runtime_score_scalar(candidate_score, target_min_overlap_run) -
            bounded_path_runtime_score_scalar(current_score, target_min_overlap_run);
        const double temperature = std::max(1.0, 1000.0 * std::pow(0.999995, static_cast<double>(iter)));
        if (unit(rng) < std::exp(std::min(0.0, delta) / temperature)) {
            current = std::move(candidate);
            current_score = candidate_score;
            current_key = candidate_key;
        }
    }

    return best;
}

std::vector<int> longest_strong_path_in_subset(const std::vector<std::vector<int>> &strong_adj,
                                               const std::vector<std::vector<int>> &overlap,
                                               const std::vector<uint8_t> &allowed,
                                               int64_t dfs_budget) {
    const int num_states = static_cast<int>(strong_adj.size());
    int allowed_count = 0;
    for (auto value : allowed) {
        allowed_count += value != 0 ? 1 : 0;
    }
    if (allowed_count <= 0) {
        return {};
    }

    std::vector<int> starts;
    starts.reserve(static_cast<std::size_t>(allowed_count));
    for (int state_idx = 0; state_idx < num_states; state_idx++) {
        if (allowed[state_idx] != 0) {
            starts.emplace_back(state_idx);
        }
    }
    auto remaining_degree = [&](int state_idx, const std::vector<uint8_t> &used) {
        int degree = 0;
        for (auto neighbor : strong_adj[state_idx]) {
            if (allowed[neighbor] != 0 && used[neighbor] == 0) {
                degree++;
            }
        }
        return degree;
    };
    std::vector<uint8_t> empty_used(static_cast<std::size_t>(num_states), 0);
    std::sort(starts.begin(), starts.end(), [&](int lhs, int rhs) {
        return std::make_tuple(remaining_degree(lhs, empty_used), lhs) <
               std::make_tuple(remaining_degree(rhs, empty_used), rhs);
    });

    std::vector<int> best_path;
    std::vector<int> path;
    std::vector<uint8_t> used(static_cast<std::size_t>(num_states), 0);
    int64_t expansions = 0;
    dfs_budget = std::max<int64_t>(1, dfs_budget);

    std::function<bool(int, int)> dfs = [&](int current, int used_count) {
        expansions++;
        if (path.size() > best_path.size() ||
            (path.size() == best_path.size() &&
             path_internal_overlap(path, overlap) > path_internal_overlap(best_path, overlap))) {
            best_path = path;
        }
        if (static_cast<int>(best_path.size()) == allowed_count) {
            return true;
        }
        if (expansions >= dfs_budget) {
            return false;
        }
        if (static_cast<int>(path.size()) + (allowed_count - used_count) <= static_cast<int>(best_path.size())) {
            return false;
        }

        std::vector<int> options;
        for (auto neighbor : strong_adj[current]) {
            if (allowed[neighbor] != 0 && used[neighbor] == 0) {
                options.emplace_back(neighbor);
            }
        }
        std::sort(options.begin(), options.end(), [&](int lhs, int rhs) {
            return std::make_tuple(remaining_degree(lhs, used), -overlap[current][lhs], lhs) <
                   std::make_tuple(remaining_degree(rhs, used), -overlap[current][rhs], rhs);
        });

        for (auto next : options) {
            used[next] = 1;
            path.emplace_back(next);
            if (dfs(next, used_count + 1)) {
                return true;
            }
            path.pop_back();
            used[next] = 0;
            if (expansions >= dfs_budget) {
                break;
            }
        }
        return false;
    };

    for (auto start : starts) {
        std::fill(used.begin(), used.end(), 0);
        path.clear();
        used[start] = 1;
        path.emplace_back(start);
        if (dfs(start, 1)) {
            break;
        }
        if (expansions >= dfs_budget && !best_path.empty()) {
            break;
        }
    }

    return best_path;
}

std::vector<int> build_strong_path_cover_order(const std::vector<std::vector<int>> &overlap,
                                               int min_overlap,
                                               int buffer_capacity) {
    const int num_states = static_cast<int>(overlap.size());
    if (num_states <= 2) {
        std::vector<int> order(num_states);
        std::iota(order.begin(), order.end(), 0);
        return order;
    }

    const int strong_overlap = std::min(buffer_capacity, min_overlap + 1);
    if (strong_overlap <= min_overlap) {
        return {};
    }

    std::vector<std::vector<int>> strong_adj(static_cast<std::size_t>(num_states));
    for (int lhs = 0; lhs < num_states; lhs++) {
        for (int rhs = lhs + 1; rhs < num_states; rhs++) {
            if (overlap[lhs][rhs] >= strong_overlap) {
                strong_adj[lhs].emplace_back(rhs);
                strong_adj[rhs].emplace_back(lhs);
            }
        }
    }
    for (auto &neighbors : strong_adj) {
        std::sort(neighbors.begin(), neighbors.end());
    }

    std::vector<std::vector<int>> components;
    std::vector<uint8_t> visited(static_cast<std::size_t>(num_states), 0);
    for (int state_idx = 0; state_idx < num_states; state_idx++) {
        if (visited[state_idx] != 0) {
            continue;
        }
        std::vector<int> stack{state_idx};
        std::vector<int> component;
        visited[state_idx] = 1;
        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            component.emplace_back(current);
            for (auto neighbor : strong_adj[current]) {
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    stack.emplace_back(neighbor);
                }
            }
        }
        components.emplace_back(std::move(component));
    }
    std::sort(components.begin(), components.end(), [](const auto &lhs, const auto &rhs) {
        return std::make_tuple(lhs.size(), lhs.empty() ? 0 : -lhs.front()) >
               std::make_tuple(rhs.size(), rhs.empty() ? 0 : -rhs.front());
    });

    const int64_t dfs_budget = std::max<int64_t>(
        1, stateflow_env_int64("GEGE_BOUNDED_STRONG_PATH_DFS_BUDGET", nullptr, 250000));
    std::vector<std::vector<int>> fragments;
    for (const auto &component : components) {
        std::vector<uint8_t> remaining(static_cast<std::size_t>(num_states), 0);
        int remaining_count = 0;
        for (auto state_idx : component) {
            remaining[state_idx] = 1;
            remaining_count++;
        }

        while (remaining_count > 0) {
            std::vector<int> fragment = longest_strong_path_in_subset(strong_adj, overlap, remaining, dfs_budget);
            if (fragment.empty()) {
                for (auto state_idx : component) {
                    if (remaining[state_idx] != 0) {
                        fragment = {state_idx};
                        break;
                    }
                }
            }
            if (fragment.empty()) {
                return {};
            }
            for (auto state_idx : fragment) {
                if (remaining[state_idx] != 0) {
                    remaining[state_idx] = 0;
                    remaining_count--;
                }
            }
            fragments.emplace_back(std::move(fragment));
        }
    }

    std::sort(fragments.begin(), fragments.end(), [&](const auto &lhs, const auto &rhs) {
        return std::make_tuple(lhs.size(), path_internal_overlap(lhs, overlap), lhs.empty() ? 0 : -lhs.front()) >
               std::make_tuple(rhs.size(), path_internal_overlap(rhs, overlap), rhs.empty() ? 0 : -rhs.front());
    });

    std::vector<int> order;
    for (const auto &fragment : fragments) {
        struct InsertCandidate {
            bool valid = false;
            int delta = std::numeric_limits<int>::min();
            int internal_overlap = 0;
            int position = 0;
            bool reversed = false;
            std::vector<int> fragment;
        };

        InsertCandidate best;
        for (int reverse_flag = 0; reverse_flag < 2; reverse_flag++) {
            std::vector<int> oriented = fragment;
            if (reverse_flag != 0) {
                std::reverse(oriented.begin(), oriented.end());
            }
            int internal = path_internal_overlap(oriented, overlap);
            for (int position = 0; position <= static_cast<int>(order.size()); position++) {
                bool valid = true;
                int delta = internal;
                if (!order.empty()) {
                    if (position > 0 && overlap[order[position - 1]][oriented.front()] < min_overlap) {
                        valid = false;
                    }
                    if (position < static_cast<int>(order.size()) &&
                        overlap[oriented.back()][order[position]] < min_overlap) {
                        valid = false;
                    }
                    if (!valid) {
                        continue;
                    }
                    if (position > 0) {
                        delta += overlap[order[position - 1]][oriented.front()];
                    }
                    if (position < static_cast<int>(order.size())) {
                        delta += overlap[oriented.back()][order[position]];
                    }
                    if (position > 0 && position < static_cast<int>(order.size())) {
                        delta -= overlap[order[position - 1]][order[position]];
                    }
                }

                auto key = std::make_tuple(delta, internal, static_cast<int>(oriented.size()), -position, -reverse_flag);
                auto best_key = std::make_tuple(best.delta, best.internal_overlap, static_cast<int>(best.fragment.size()),
                                                -best.position, best.reversed ? -1 : 0);
                if (!best.valid || key > best_key) {
                    best.valid = true;
                    best.delta = delta;
                    best.internal_overlap = internal;
                    best.position = position;
                    best.reversed = reverse_flag != 0;
                    best.fragment = oriented;
                }
            }
        }

        if (!best.valid) {
            return {};
        }
        order.insert(order.begin() + best.position, best.fragment.begin(), best.fragment.end());
    }

    if (static_cast<int>(order.size()) != num_states || !bounded_path_respects_min_overlap(order, overlap, min_overlap)) {
        return {};
    }

    const int max_passes = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_PATH_INSERT_LOCAL_PASSES", nullptr, 4)));
    const std::vector<int> segment_lengths{1, 2, 3, 4, 5, 8, 12, 20};
    for (int pass = 0; pass < max_passes; pass++) {
        bool improved = false;
        auto current_score = bounded_path_score(order, overlap);
        for (auto requested_length : segment_lengths) {
            int segment_length = std::min<int>(requested_length, order.size());
            if (segment_length <= 0) {
                continue;
            }
            for (int begin = 0; begin + segment_length <= static_cast<int>(order.size()); begin++) {
                std::vector<int> segment(order.begin() + begin, order.begin() + begin + segment_length);
                std::vector<int> rest;
                rest.reserve(order.size() - segment.size());
                rest.insert(rest.end(), order.begin(), order.begin() + begin);
                rest.insert(rest.end(), order.begin() + begin + segment_length, order.end());

                for (int reverse_flag = 0; reverse_flag < 2; reverse_flag++) {
                    std::vector<int> oriented = segment;
                    if (reverse_flag != 0) {
                        std::reverse(oriented.begin(), oriented.end());
                    }
                    for (int position = 0; position <= static_cast<int>(rest.size()); position++) {
                        if (reverse_flag == 0 && position == begin) {
                            continue;
                        }
                        std::vector<int> candidate;
                        candidate.reserve(order.size());
                        candidate.insert(candidate.end(), rest.begin(), rest.begin() + position);
                        candidate.insert(candidate.end(), oriented.begin(), oriented.end());
                        candidate.insert(candidate.end(), rest.begin() + position, rest.end());
                        if (!bounded_path_respects_min_overlap(candidate, overlap, min_overlap)) {
                            continue;
                        }
                        auto candidate_score = bounded_path_score(candidate, overlap);
                        if (candidate_score > current_score) {
                            order = std::move(candidate);
                            improved = true;
                            break;
                        }
                    }
                    if (improved) {
                        break;
                    }
                }
                if (improved) {
                    break;
                }
            }
            if (improved) {
                break;
            }
        }
        if (!improved) {
            break;
        }
    }

    return bounded_path_respects_min_overlap(order, overlap, min_overlap) ? order : std::vector<int>{};
}

std::vector<int> improve_bounded_greedy_path(std::vector<int> order,
                                             const std::vector<std::vector<int>> &overlap,
                                             std::mt19937_64 &rng,
                                             int iterations,
                                             int min_overlap) {
    if (order.size() <= 2 || iterations <= 0) {
        return order;
    }

    std::vector<int> best = order;
    auto current_score = bounded_path_score(order, overlap);
    auto best_score = current_score;
    double temperature = 10.0;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (int iter = 0; iter < iterations; iter++) {
        std::vector<int> candidate = order;
        std::uniform_int_distribution<int> index_dist(0, static_cast<int>(order.size()) - 1);
        int left = index_dist(rng);
        int right = index_dist(rng);
        if (left == right) {
            continue;
        }
        if (left > right) {
            std::swap(left, right);
        }

        if (iter % 3 == 0) {
            std::reverse(candidate.begin() + left, candidate.begin() + right + 1);
        } else if (iter % 3 == 1) {
            int value = candidate[left];
            candidate.erase(candidate.begin() + left);
            int insert_pos = right;
            if (insert_pos > static_cast<int>(candidate.size())) {
                insert_pos = static_cast<int>(candidate.size());
            }
            candidate.insert(candidate.begin() + insert_pos, value);
        } else {
            std::swap(candidate[left], candidate[right]);
        }
        if (!bounded_path_respects_min_overlap(candidate, overlap, min_overlap)) {
            continue;
        }

        auto candidate_score = bounded_path_score(candidate, overlap);
        const double delta = bounded_path_score_scalar(candidate_score) - bounded_path_score_scalar(current_score);
        if (candidate_score > current_score || unit(rng) < std::exp(std::min(0.0, delta) / std::max(temperature, 1e-9))) {
            order = std::move(candidate);
            current_score = candidate_score;
            if (current_score > best_score) {
                best = order;
                best_score = current_score;
            }
        }
        temperature *= 0.9995;
    }

    return best;
}

std::vector<std::vector<int>> reorder_states_for_bounded_admits(const std::vector<std::vector<int>> &states,
                                                                int max_admits) {
    const int num_states = static_cast<int>(states.size());
    if (num_states <= 2 || states.front().empty()) {
        return states;
    }

    const int buffer_capacity = static_cast<int>(states.front().size());
    const int min_overlap = std::max(0, buffer_capacity - max_admits);
    std::vector<std::vector<int>> overlap(num_states, std::vector<int>(num_states, 0));
    for (int lhs = 0; lhs < num_states; lhs++) {
        for (int rhs = lhs + 1; rhs < num_states; rhs++) {
            int shared = state_overlap_count(states[lhs], states[rhs]);
            overlap[lhs][rhs] = shared;
            overlap[rhs][lhs] = shared;
        }
    }

    auto convert_order_to_states = [&](const std::vector<int> &order) {
        std::vector<std::vector<int>> reordered;
        reordered.reserve(states.size());
        for (auto state_idx : order) {
            reordered.emplace_back(states[state_idx]);
        }
        return reordered;
    };

    std::vector<int> strong_order;
    if (stateflow_env_bool("GEGE_BOUNDED_STRONG_PATH_REORDER", nullptr, true)) {
        strong_order = build_strong_path_cover_order(overlap, min_overlap, buffer_capacity);
    }
    if (static_cast<int>(strong_order.size()) == num_states &&
        bounded_path_respects_min_overlap(strong_order, overlap, min_overlap)) {
        strong_order = improve_bounded_path_runtime_balance(std::move(strong_order), overlap, states, min_overlap, buffer_capacity);
        auto score = bounded_path_score(strong_order, overlap);
        auto runtime_score = bounded_path_runtime_score(strong_order, overlap, states, min_overlap, buffer_capacity);
        const int64_t transition_overlap = std::get<1>(score);
        const int64_t transition_admits =
            static_cast<int64_t>(std::max(0, num_states - 1)) * static_cast<int64_t>(buffer_capacity) -
            transition_overlap;
        SPDLOG_INFO(
            "Bounded greedy cover dynamic strong-path reorder selected states={} max_admits={} min_overlap={} transition_admits={} transition_overlap={} overlap_hist=[1:{}, 2:{}, 3:{}] max_min_overlap_run={} min_overlap_run_sum_sq={} max_window_admits={} short_reuse3={} short_reuse3_transitions={} short_reuse5={} capped_reuse_distance={}",
            num_states, max_admits, min_overlap, transition_admits, transition_overlap,
            -std::get<4>(score), std::get<3>(score), std::get<2>(score),
            runtime_score.max_min_overlap_run, runtime_score.min_overlap_run_sum_sq,
            runtime_score.max_window_admits, runtime_score.short_evict_reuse_count3,
            runtime_score.short_evict_reuse_transitions3, runtime_score.short_evict_reuse_count5,
            runtime_score.capped_evict_reuse_distance_sum);
        return convert_order_to_states(strong_order);
    }

    const int restarts = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_GREEDY_RESTARTS", nullptr, 512)));
    const int local_iterations = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_GREEDY_LOCAL_ITERS", nullptr, 768)));
    const int candidate_pool = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_GREEDY_CANDIDATE_POOL", nullptr, 30)));
    const int64_t seed = stateflow_env_int64("GEGE_BOUNDED_GREEDY_SEED", nullptr, 12345);

    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    std::vector<int> best_order;
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> best_score{
        std::numeric_limits<int64_t>::min(), 0, 0, 0, 0};

    for (int restart = 0; restart < restarts; restart++) {
        int start_idx = restart % num_states;
        if (restart >= num_states) {
            std::uniform_int_distribution<int> start_dist(0, num_states - 1);
            start_idx = start_dist(rng);
        }

        auto order = build_bounded_greedy_path_once(overlap, rng, start_idx, candidate_pool, min_overlap);
        if (static_cast<int>(order.size()) != num_states) {
            continue;
        }
        order = improve_bounded_greedy_path(std::move(order), overlap, rng, local_iterations, min_overlap);
        if (!bounded_path_respects_min_overlap(order, overlap, min_overlap)) {
            continue;
        }
        auto score = bounded_path_score(order, overlap);
        if (score > best_score) {
            best_score = score;
            best_order = std::move(order);
        }
    }

    if (best_order.empty() || -std::get<0>(best_score) != 0) {
        SPDLOG_WARN("Bounded greedy cover could not find a full overlap>={} path; falling back to max-overlap reorder",
                    min_overlap);
        return reorder_states_for_max_overlap(states);
    }

    best_order = improve_bounded_path_runtime_balance(std::move(best_order), overlap, states, min_overlap, buffer_capacity);
    return convert_order_to_states(best_order);
}

int64_t count_uncovered_diagonal_gain(const std::vector<int> &state,
                                      const std::vector<uint8_t> &covered,
                                      int num_partitions) {
    int64_t gain = 0;
    for (auto partition : state) {
        if (partition >= 0 && partition < num_partitions &&
            covered[partition * num_partitions + partition] == 0) {
            gain++;
        }
    }
    return gain;
}

bool state_conflicts_with_round_partitions(const std::vector<int> &state,
                                           const std::vector<uint8_t> &round_partitions) {
    for (auto partition : state) {
        if (partition >= 0 && partition < static_cast<int>(round_partitions.size()) &&
            round_partitions[partition] != 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<int>> build_bounded_single_gpu_cover_states(int num_partitions,
                                                                    int buffer_capacity,
                                                                    int max_admits) {
    std::vector<std::vector<int>> candidates;
    std::vector<int> current;
    build_greedy_cover_candidates(num_partitions, buffer_capacity, 0, current, candidates);
    if (candidates.empty()) {
        return {};
    }

    const int min_overlap = std::max(0, buffer_capacity - std::max(0, max_admits));
    const int64_t total_buckets = static_cast<int64_t>(num_partitions) * static_cast<int64_t>(num_partitions);
    std::vector<uint8_t> covered(num_partitions * num_partitions, 0);
    int64_t covered_count = 0;
    std::vector<std::vector<int>> states;
    states.reserve(static_cast<std::size_t>(std::max(num_partitions * 4, 32)));

    const int max_states = static_cast<int>(
        std::max<int64_t>(num_partitions * num_partitions,
                          stateflow_env_int64("GEGE_BOUNDED_SINGLE_GPU_MAX_STATES", nullptr,
                                              std::max<int64_t>(num_partitions * 8, 128))));

    while (covered_count < total_buckets && static_cast<int>(states.size()) < max_states) {
        int best_idx = -1;
        int64_t best_gain = -1;
        int64_t best_diagonal_gain = -1;
        int best_overlap = -1;
        int64_t best_balance_score = std::numeric_limits<int64_t>::max();

        for (int candidate_idx = 0; candidate_idx < static_cast<int>(candidates.size()); candidate_idx++) {
            const auto &candidate = candidates[candidate_idx];
            int overlap = states.empty() ? buffer_capacity : state_overlap_count(states.back(), candidate);
            if (!states.empty() && overlap < min_overlap) {
                continue;
            }

            int64_t gain = count_uncovered_buckets_for_state(candidate, covered, num_partitions);
            if (gain <= 0) {
                continue;
            }
            int64_t diagonal_gain = count_uncovered_diagonal_gain(candidate, covered, num_partitions);
            int64_t balance_score = 0;
            for (auto partition : candidate) {
                balance_score += partition;
            }

            if (gain > best_gain ||
                (gain == best_gain && diagonal_gain > best_diagonal_gain) ||
                (gain == best_gain && diagonal_gain == best_diagonal_gain && overlap > best_overlap) ||
                (gain == best_gain && diagonal_gain == best_diagonal_gain && overlap == best_overlap &&
                 balance_score < best_balance_score)) {
                best_idx = candidate_idx;
                best_gain = gain;
                best_diagonal_gain = diagonal_gain;
                best_overlap = overlap;
                best_balance_score = balance_score;
            }
        }

        if (best_idx < 0) {
            SPDLOG_WARN("Bounded single-GPU q{} cover stalled under max_admits={} after states={} covered={} total={}",
                        buffer_capacity, max_admits, states.size(), covered_count, total_buckets);
            return {};
        }

        states.emplace_back(candidates[best_idx]);
        covered_count += mark_state_buckets_covered(states.back(), covered, num_partitions);
    }

    if (covered_count < total_buckets) {
        SPDLOG_WARN("Bounded single-GPU q{} cover exceeded max_states={} under max_admits={} covered={} total={}",
                    buffer_capacity, max_states, max_admits, covered_count, total_buckets);
        return {};
    }

    int64_t transition_admits = 0;
    int64_t transition_overlap = 0;
    for (std::size_t idx = 1; idx < states.size(); idx++) {
        int overlap = state_overlap_count(states[idx - 1], states[idx]);
        transition_overlap += overlap;
        transition_admits += buffer_capacity - overlap;
    }
    SPDLOG_INFO("Bounded single-GPU q{} cover selected states={} max_admits={} transition_admits={} transition_overlap={}",
                buffer_capacity, states.size(), max_admits, transition_admits, transition_overlap);
    return states;
}

void mark_round_partitions(const std::vector<int> &state,
                           std::vector<uint8_t> &round_partitions) {
    for (auto partition : state) {
        if (partition >= 0 && partition < static_cast<int>(round_partitions.size())) {
            round_partitions[partition] = 1;
        }
    }
}

struct BoundedMultiGpuRoundOption {
    double score = 0.0;
    int candidate_idx = -1;
    int64_t uncovered_gain = 0;
    int overlap = 0;
};

std::vector<std::vector<int>> build_bounded_multi_gpu_round_states(int num_partitions,
                                                                    int buffer_capacity,
                                                                    int active_devices,
                                                                    int max_admits) {
    std::vector<std::vector<int>> candidates;
    std::vector<int> current;
    build_greedy_cover_candidates(num_partitions, buffer_capacity, 0, current, candidates);
    if (candidates.empty() || active_devices <= 1) {
        return {};
    }

    if (active_devices * buffer_capacity > num_partitions) {
        SPDLOG_WARN(
            "Bounded multi-GPU q{} cover requires active_devices * buffer_capacity <= num_partitions for disjoint visible rounds; got active_devices={} num_partitions={}",
            buffer_capacity, active_devices, num_partitions);
        return {};
    }

    const int min_overlap = std::max(0, buffer_capacity - std::max(0, max_admits));
    const int64_t total_buckets = static_cast<int64_t>(num_partitions) * static_cast<int64_t>(num_partitions);
    const int64_t optimistic_round_capacity =
        std::max<int64_t>(1, static_cast<int64_t>(active_devices) * buffer_capacity * buffer_capacity);
    const int lower_bound_rounds =
        static_cast<int>((total_buckets + optimistic_round_capacity - 1) / optimistic_round_capacity);
    const int max_rounds = static_cast<int>(
        std::max<int64_t>(lower_bound_rounds + 16,
                          stateflow_env_int64("GEGE_BOUNDED_MULTI_GPU_MAX_ROUNDS", nullptr,
                                              std::max<int64_t>(num_partitions * 4, lower_bound_rounds * 4))));
    const int restarts = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_MULTI_GPU_RESTARTS", nullptr, 12)));
    const int topk = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_MULTI_GPU_TOPK", nullptr, 64)));
    const int64_t seed = stateflow_env_int64("GEGE_BOUNDED_MULTI_GPU_SEED", nullptr, 12345);

    std::vector<std::vector<int>> best_states;
    int64_t best_total_admits = std::numeric_limits<int64_t>::max();
    int64_t best_total_overlap = std::numeric_limits<int64_t>::min();

    for (int restart = 0; restart < restarts; restart++) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed + restart * 9973));
        std::uniform_real_distribution<double> unit(0.0, 1.0);

        std::vector<uint8_t> covered(num_partitions * num_partitions, 0);
        int64_t covered_count = 0;
        std::vector<std::vector<int>> lane_previous(active_devices);
        std::vector<std::vector<int>> states;
        states.reserve(static_cast<std::size_t>(max_rounds * active_devices));
        int64_t total_admits = 0;
        int64_t total_overlap = 0;
        bool failed = false;

        for (int round = 0; round < max_rounds && covered_count < total_buckets; round++) {
            std::vector<int> lane_order(active_devices);
            std::iota(lane_order.begin(), lane_order.end(), 0);
            if (restart > 0) {
                std::shuffle(lane_order.begin(), lane_order.end(), rng);
            }

            std::vector<int> chosen_by_lane(active_devices, -1);
            std::vector<uint8_t> round_partitions(num_partitions, 0);

            for (auto lane : lane_order) {
                std::vector<BoundedMultiGpuRoundOption> options;
                options.reserve(std::min<int>(topk * 4, candidates.size()));
                for (int candidate_idx = 0; candidate_idx < static_cast<int>(candidates.size()); candidate_idx++) {
                    const auto &candidate = candidates[candidate_idx];
                    if (state_conflicts_with_round_partitions(candidate, round_partitions)) {
                        continue;
                    }

                    int overlap = lane_previous[lane].empty()
                                      ? 0
                                      : state_overlap_count(lane_previous[lane], candidate);
                    if (!lane_previous[lane].empty() && overlap < min_overlap) {
                        continue;
                    }

                    int64_t uncovered_gain = count_uncovered_buckets_for_state(candidate, covered, num_partitions);
                    int64_t diagonal_gain = count_uncovered_diagonal_gain(candidate, covered, num_partitions);
                    double score = static_cast<double>(uncovered_gain) * 100000.0 +
                                   static_cast<double>(diagonal_gain) * 1000.0 +
                                   static_cast<double>(overlap) * 10.0;
                    if (uncovered_gain == 0 && covered_count < total_buckets) {
                        score -= 1000000.0;
                    }
                    if (restart > 0) {
                        score += unit(rng);
                    }

                    options.push_back(BoundedMultiGpuRoundOption{score, candidate_idx, uncovered_gain, overlap});
                }

                if (options.empty()) {
                    failed = true;
                    break;
                }

                const int keep = std::min<int>(topk, options.size());
                std::partial_sort(options.begin(), options.begin() + keep, options.end(),
                                  [](const auto &lhs, const auto &rhs) {
                                      if (lhs.score != rhs.score) {
                                          return lhs.score > rhs.score;
                                      }
                                      return lhs.candidate_idx < rhs.candidate_idx;
                                  });
                options.resize(keep);

                int option_idx = 0;
                if (restart > 0 && options.size() > 1 && unit(rng) >= 0.7) {
                    const int random_limit = std::min<int>(10, options.size());
                    option_idx = static_cast<int>(unit(rng) * unit(rng) * random_limit);
                    option_idx = std::min(option_idx, random_limit - 1);
                }

                const int candidate_idx = options[option_idx].candidate_idx;
                chosen_by_lane[lane] = candidate_idx;
                mark_round_partitions(candidates[candidate_idx], round_partitions);
            }

            if (failed) {
                break;
            }

            for (int lane = 0; lane < active_devices; lane++) {
                int candidate_idx = chosen_by_lane[lane];
                if (candidate_idx < 0) {
                    failed = true;
                    break;
                }

                const auto &candidate = candidates[candidate_idx];
                if (!lane_previous[lane].empty()) {
                    int overlap = state_overlap_count(lane_previous[lane], candidate);
                    total_overlap += overlap;
                    total_admits += static_cast<int64_t>(buffer_capacity - overlap);
                }
                covered_count += mark_state_buckets_covered(candidate, covered, num_partitions);
                states.emplace_back(candidate);
                lane_previous[lane] = candidate;
            }

            if (failed) {
                break;
            }
        }

        if (failed || covered_count < total_buckets) {
            continue;
        }

        if (best_states.empty() ||
            states.size() < best_states.size() ||
            (states.size() == best_states.size() && total_admits < best_total_admits) ||
            (states.size() == best_states.size() && total_admits == best_total_admits &&
             total_overlap > best_total_overlap)) {
            best_states = std::move(states);
            best_total_admits = total_admits;
            best_total_overlap = total_overlap;
        }
    }

    if (best_states.empty()) {
        SPDLOG_WARN(
            "Bounded multi-GPU q{} cover failed to find exact coverage with active_devices={} max_admits={} restarts={} max_rounds={}",
            buffer_capacity, active_devices, max_admits, restarts, max_rounds);
        return {};
    }

    SPDLOG_INFO(
        "Bounded multi-GPU q{} cover selected states={} rounds={} active_devices={} max_admits={} lane_transition_admits={} lane_transition_overlap={}",
        buffer_capacity, best_states.size(),
        (static_cast<int64_t>(best_states.size()) + active_devices - 1) / active_devices,
        active_devices, max_admits, best_total_admits, best_total_overlap);
    return best_states;
}

vector<vector<std::pair<int, int>>> balancedAssignEdgeBucketsToBuffers(const vector<vector<int>> &buffer_states,
                                                                       int num_partitions,
                                                                       const vector<int64_t> &edge_bucket_sizes,
                                                                       int initial_state_count = 1) {
    if (buffer_states.empty()) {
        return {};
    }

    const bool have_sizes = edge_bucket_sizes.size() == static_cast<std::size_t>(num_partitions * num_partitions);
    std::vector<std::vector<uint8_t>> contains(buffer_states.size(), std::vector<uint8_t>(num_partitions, 0));
    for (std::size_t state_idx = 0; state_idx < buffer_states.size(); state_idx++) {
        for (auto partition : buffer_states[state_idx]) {
            if (partition >= 0 && partition < num_partitions) {
                contains[state_idx][partition] = 1;
            }
        }
    }

    struct BucketCandidate {
        int src = 0;
        int dst = 0;
        int64_t weight = 1;
    };
    std::vector<BucketCandidate> buckets;
    buckets.reserve(static_cast<std::size_t>(num_partitions) * static_cast<std::size_t>(num_partitions));
    for (int src = 0; src < num_partitions; src++) {
        for (int dst = 0; dst < num_partitions; dst++) {
            int64_t weight = have_sizes ? edge_bucket_sizes[src * num_partitions + dst] : 1;
            buckets.push_back(BucketCandidate{src, dst, weight});
        }
    }
    std::sort(buckets.begin(), buckets.end(), [](const BucketCandidate &lhs, const BucketCandidate &rhs) {
        return std::make_tuple(lhs.weight, lhs.src, lhs.dst) > std::make_tuple(rhs.weight, rhs.src, rhs.dst);
    });

    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer(buffer_states.size());
    std::vector<int64_t> assigned_weight(buffer_states.size(), 0);
    std::vector<int64_t> assigned_count(buffer_states.size(), 0);
    std::vector<std::vector<int>> compatible_states(static_cast<std::size_t>(num_partitions) *
                                                    static_cast<std::size_t>(num_partitions));
    std::vector<int> fixed_bucket_state(static_cast<std::size_t>(num_partitions) *
                                        static_cast<std::size_t>(num_partitions), -1);

    const char *pin_initial_raw = std::getenv("GEGE_PIN_INITIAL_BUCKET_ASSIGNMENT");
    const bool pin_initial_buckets =
        pin_initial_raw != nullptr &&
        !(std::string(pin_initial_raw) == "0" || std::string(pin_initial_raw) == "false" ||
          std::string(pin_initial_raw) == "False" || std::string(pin_initial_raw) == "FALSE");
    if (pin_initial_buckets) {
        // Opt-in debug mode: force buckets first visible in an initial lane
        // state to be trained there. The default min-max assignment is still
        // exact and avoids coupling coverage correctness to first visibility.
        int clamped_initial_states =
            std::min<int>(std::max(1, initial_state_count), static_cast<int>(buffer_states.size()));
        for (int state_idx = 0; state_idx < clamped_initial_states; state_idx++) {
            for (int src = 0; src < num_partitions; src++) {
                if (contains[state_idx][src] == 0) {
                    continue;
                }
                for (int dst = 0; dst < num_partitions; dst++) {
                    if (contains[state_idx][dst] == 0) {
                        continue;
                    }
                    int bucket_idx = src * num_partitions + dst;
                    if (fixed_bucket_state[bucket_idx] < 0) {
                        fixed_bucket_state[bucket_idx] = state_idx;
                    }
                }
            }
        }
    }

    auto assign_bucket_to_state = [&](int state_idx, const BucketCandidate &bucket) {
        if (state_idx < 0 || state_idx >= static_cast<int>(edge_buckets_per_buffer.size())) {
            return;
        }
        edge_buckets_per_buffer[state_idx].emplace_back(bucket.src, bucket.dst);
        assigned_weight[state_idx] += bucket.weight;
        assigned_count[state_idx]++;
    };

    for (const auto &bucket : buckets) {
        int bucket_idx = bucket.src * num_partitions + bucket.dst;
        int fixed_state = fixed_bucket_state[bucket_idx];
        if (fixed_state >= 0) {
            assign_bucket_to_state(fixed_state, bucket);
        }
    }

    for (const auto &bucket : buckets) {
        int bucket_idx = bucket.src * num_partitions + bucket.dst;
        int best_state = -1;
        auto best_key = std::make_tuple(std::numeric_limits<int64_t>::max(),
                                        std::numeric_limits<int64_t>::max(),
                                        std::numeric_limits<int>::max());
        for (int state_idx = 0; state_idx < static_cast<int>(buffer_states.size()); state_idx++) {
            if (contains[state_idx][bucket.src] == 0 || contains[state_idx][bucket.dst] == 0) {
                continue;
            }
            compatible_states[bucket_idx].emplace_back(state_idx);
            auto key = std::make_tuple(assigned_weight[state_idx], assigned_count[state_idx], state_idx);
            if (key < best_key) {
                best_key = key;
                best_state = state_idx;
            }
        }

        if (fixed_bucket_state[bucket_idx] >= 0) {
            continue;
        }

        if (best_state < 0) {
            SPDLOG_WARN("Balanced bucket assignment found no compatible state for bucket ({},{}); falling back to first-cover assignment",
                        bucket.src, bucket.dst);
            return greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
        }

        assign_bucket_to_state(best_state, bucket);
    }

    const char *minmax_raw = std::getenv("GEGE_MINMAX_BUCKET_ASSIGNMENT");
    bool minmax_enabled = have_sizes &&
                          !(minmax_raw != nullptr &&
                            (std::string(minmax_raw) == "0" || std::string(minmax_raw) == "false" ||
                             std::string(minmax_raw) == "False" || std::string(minmax_raw) == "FALSE"));
    if (minmax_enabled) {
        auto sum_sq_load = [&assigned_weight]() {
            long double value = 0.0L;
            for (auto load : assigned_weight) {
                value += static_cast<long double>(load) * static_cast<long double>(load);
            }
            return value;
        };
        auto current_max_load = [&assigned_weight]() {
            return *std::max_element(assigned_weight.begin(), assigned_weight.end());
        };
        auto erase_bucket_from_state = [](vector<std::pair<int, int>> &state_buckets, const std::pair<int, int> &bucket) {
            auto itr = std::find(state_buckets.begin(), state_buckets.end(), bucket);
            if (itr != state_buckets.end()) {
                state_buckets.erase(itr);
            }
        };
        auto max_after_move = [&assigned_weight](int src_state, int dst_state, int64_t bucket_weight) {
            int64_t max_load = 0;
            for (int state_idx = 0; state_idx < static_cast<int>(assigned_weight.size()); state_idx++) {
                int64_t load = assigned_weight[state_idx];
                if (state_idx == src_state) {
                    load -= bucket_weight;
                } else if (state_idx == dst_state) {
                    load += bucket_weight;
                }
                max_load = std::max(max_load, load);
            }
            return max_load;
        };
        auto sum_sq_after_move = [&assigned_weight](long double current_sum_sq,
                                                    int src_state,
                                                    int dst_state,
                                                    int64_t bucket_weight) {
            long double src_before = static_cast<long double>(assigned_weight[src_state]);
            long double dst_before = static_cast<long double>(assigned_weight[dst_state]);
            long double src_after = src_before - static_cast<long double>(bucket_weight);
            long double dst_after = dst_before + static_cast<long double>(bucket_weight);
            return current_sum_sq - src_before * src_before - dst_before * dst_before +
                   src_after * src_after + dst_after * dst_after;
        };

        const int max_iterations = static_cast<int>(
            std::max<int64_t>(1, stateflow_env_int64("GEGE_MINMAX_BUCKET_ASSIGNMENT_ITERS", nullptr, 10000)));
        const int64_t initial_max_load = current_max_load();
        const long double initial_sum_sq = sum_sq_load();
        int64_t moves = 0;
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            const int64_t current_max = current_max_load();
            const long double current_sum_sq = sum_sq_load();

            struct Move {
                bool valid = false;
                int64_t new_max = std::numeric_limits<int64_t>::max();
                long double new_sum_sq = std::numeric_limits<long double>::infinity();
                int64_t moved_weight = 0;
                int src_state = -1;
                int dst_state = -1;
                std::pair<int, int> bucket{-1, -1};
            };

            Move best_move;
            for (int src_state = 0; src_state < static_cast<int>(edge_buckets_per_buffer.size()); src_state++) {
                if (assigned_weight[src_state] != current_max) {
                    continue;
                }
                for (const auto &bucket : edge_buckets_per_buffer[src_state]) {
                    int bucket_idx = bucket.first * num_partitions + bucket.second;
                    if (fixed_bucket_state[bucket_idx] == src_state) {
                        continue;
                    }
                    int64_t bucket_weight = edge_bucket_sizes[bucket_idx];
                    for (auto dst_state : compatible_states[bucket_idx]) {
                        if (dst_state == src_state || fixed_bucket_state[bucket_idx] >= 0) {
                            continue;
                        }
                        int64_t new_max = max_after_move(src_state, dst_state, bucket_weight);
                        if (new_max > current_max) {
                            continue;
                        }
                        long double new_sum_sq = sum_sq_after_move(current_sum_sq, src_state, dst_state, bucket_weight);
                        if (new_max < current_max ||
                            (new_max == current_max && new_sum_sq + 0.5L < current_sum_sq)) {
                            if (!best_move.valid ||
                                std::make_tuple(new_max, new_sum_sq, -bucket_weight, src_state, dst_state, bucket) <
                                    std::make_tuple(best_move.new_max, best_move.new_sum_sq, -best_move.moved_weight,
                                                    best_move.src_state, best_move.dst_state, best_move.bucket)) {
                                best_move.valid = true;
                                best_move.new_max = new_max;
                                best_move.new_sum_sq = new_sum_sq;
                                best_move.moved_weight = bucket_weight;
                                best_move.src_state = src_state;
                                best_move.dst_state = dst_state;
                                best_move.bucket = bucket;
                            }
                        }
                    }
                }
            }

            if (!best_move.valid) {
                for (int src_state = 0; src_state < static_cast<int>(edge_buckets_per_buffer.size()); src_state++) {
                    if (assigned_weight[src_state] < current_max - 1) {
                        continue;
                    }
                    for (const auto &bucket : edge_buckets_per_buffer[src_state]) {
                        int bucket_idx = bucket.first * num_partitions + bucket.second;
                        if (fixed_bucket_state[bucket_idx] == src_state) {
                            continue;
                        }
                        int64_t bucket_weight = edge_bucket_sizes[bucket_idx];
                        for (auto dst_state : compatible_states[bucket_idx]) {
                            if (dst_state == src_state || fixed_bucket_state[bucket_idx] >= 0) {
                                continue;
                            }
                            int64_t new_max = max_after_move(src_state, dst_state, bucket_weight);
                            if (new_max > current_max) {
                                continue;
                            }
                            long double new_sum_sq = sum_sq_after_move(current_sum_sq, src_state, dst_state, bucket_weight);
                            if (new_sum_sq + 0.5L < current_sum_sq) {
                                if (!best_move.valid ||
                                    std::make_tuple(new_sum_sq, new_max, -bucket_weight, src_state, dst_state, bucket) <
                                        std::make_tuple(best_move.new_sum_sq, best_move.new_max, -best_move.moved_weight,
                                                        best_move.src_state, best_move.dst_state, best_move.bucket)) {
                                    best_move.valid = true;
                                    best_move.new_max = new_max;
                                    best_move.new_sum_sq = new_sum_sq;
                                    best_move.moved_weight = bucket_weight;
                                    best_move.src_state = src_state;
                                    best_move.dst_state = dst_state;
                                    best_move.bucket = bucket;
                                }
                            }
                        }
                    }
                }
            }

            if (!best_move.valid) {
                break;
            }

            erase_bucket_from_state(edge_buckets_per_buffer[best_move.src_state], best_move.bucket);
            edge_buckets_per_buffer[best_move.dst_state].emplace_back(best_move.bucket);
            assigned_weight[best_move.src_state] -= best_move.moved_weight;
            assigned_weight[best_move.dst_state] += best_move.moved_weight;
            assigned_count[best_move.src_state]--;
            assigned_count[best_move.dst_state]++;
            moves++;
        }
        if (moves > 0) {
            SPDLOG_INFO("Min-max bucket assignment moved {} buckets max_load {} -> {} sum_sq {:.3e} -> {:.3e}",
                        moves, initial_max_load, current_max_load(), static_cast<double>(initial_sum_sq),
                        static_cast<double>(sum_sq_load()));
        }
    }

    return edge_buckets_per_buffer;
}

void log_cover_ordering_summary(const char *label,
                                const std::vector<std::vector<int>> &buffer_states,
                                const std::vector<std::vector<std::pair<int, int>>> &edge_buckets_per_buffer,
                                int num_partitions,
                                const std::vector<int64_t> &edge_bucket_sizes) {
    const int buffer_capacity = buffer_states.empty() ? 0 : static_cast<int>(buffer_states.front().size());
    std::vector<int64_t> overlap_hist(buffer_capacity + 1, 0);
    std::vector<int64_t> admit_hist(buffer_capacity + 1, 0);
    int64_t transition_admits = 0;
    int64_t max_admits = 0;
    for (std::size_t idx = 1; idx < buffer_states.size(); idx++) {
        int overlap = state_overlap_count(buffer_states[idx - 1], buffer_states[idx]);
        if (overlap >= 0 && overlap < static_cast<int>(overlap_hist.size())) {
            overlap_hist[overlap]++;
        }
        int admits = buffer_capacity - overlap;
        if (admits >= 0 && admits < static_cast<int>(admit_hist.size())) {
            admit_hist[admits]++;
        }
        transition_admits += admits;
        max_admits = std::max<int64_t>(max_admits, admits);
    }

    std::vector<int64_t> bucket_counts;
    bucket_counts.reserve(edge_buckets_per_buffer.size());
    int64_t total_buckets = 0;
    int64_t min_edges = std::numeric_limits<int64_t>::max();
    int64_t max_edges = 0;
    int64_t total_edges = 0;
    const bool have_sizes = edge_bucket_sizes.size() == static_cast<std::size_t>(num_partitions * num_partitions);
    for (const auto &buckets : edge_buckets_per_buffer) {
        bucket_counts.emplace_back(static_cast<int64_t>(buckets.size()));
        total_buckets += static_cast<int64_t>(buckets.size());
        int64_t state_edges = 0;
        if (have_sizes) {
            for (const auto &[src, dst] : buckets) {
                state_edges += edge_bucket_sizes[src * num_partitions + dst];
            }
            min_edges = std::min(min_edges, state_edges);
            max_edges = std::max(max_edges, state_edges);
            total_edges += state_edges;
        }
    }
    if (!have_sizes) {
        min_edges = 0;
    }

    auto hist_to_string = [](const std::vector<int64_t> &hist) {
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (std::size_t idx = 0; idx < hist.size(); idx++) {
            if (hist[idx] == 0) {
                continue;
            }
            if (!first) {
                oss << ", ";
            }
            first = false;
            oss << idx << ":" << hist[idx];
        }
        oss << "]";
        return oss.str();
    };

    SPDLOG_INFO(
        "{} states={} transitions={} total_buckets={} max_admits={} transition_admits={} overlap_hist={} admit_hist={} edge_total={} edge_min={} edge_max={}",
        label, buffer_states.size(), buffer_states.size() > 0 ? buffer_states.size() - 1 : 0, total_buckets,
        max_admits, transition_admits, hist_to_string(overlap_hist), hist_to_string(admit_hist),
        total_edges, min_edges == std::numeric_limits<int64_t>::max() ? 0 : min_edges, max_edges);
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

bool stateflow_plan_respects_max_admits(const StateflowPlan &plan, int64_t max_admits_per_transition) {
    if (max_admits_per_transition < 0) {
        return true;
    }
    for (const auto &lane : plan.lanes) {
        for (const auto &microstate : lane.microstates) {
            if (microstate.microstate_id == 0) {
                continue;
            }
            if (microstate.admitted_partitions > max_admits_per_transition) {
                return false;
            }
        }
    }
    return true;
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
    const LaneMatchCostConfig lane_match_cfg = lane_match_cost_config_from_env();
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

    double selection_admission_basis = plan.lanes.size() > 1 ? selection_admission_load : weighted_admission_load;
    if (plan.lanes.size() > 1 && lane_match_cfg.allow_peer_relay && plan.total_peer_handoff_bytes > 0) {
        const double peer_host_equivalent_bytes = peer_bytes_as_host_equivalent(plan.total_peer_handoff_bytes, lane_match_cfg);
        selection_admission_basis =
            std::max(0.0, selection_admission_load - static_cast<double>(plan.total_peer_handoff_bytes)) + peer_host_equivalent_bytes;
    }
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
        case PlanVariant::MULTI_GPU_OPTIMAL88_LANE_MATCHED:
            return "optimal88_lane_matched";
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
    const LaneMatchCostConfig lane_match_cfg = lane_match_cost_config_from_env();

    std::vector<int64_t> identity_permutation(buffer_states.size());
    std::iota(identity_permutation.begin(), identity_permutation.end(), 0);
    StateflowPlan input_round_plan =
        build_multi_gpu_stateflow_plan_from_permutation(buffer_states, edge_buckets_per_buffer, identity_permutation, active_devices,
                                                        PlanVariant::MULTI_GPU_DISJOINT_ROUNDS, edge_bucket_sizes,
                                                        partition_row_counts, layout);
    if (stateflow_plan_valid(input_round_plan) &&
        stateflow_plan_respects_max_admits(input_round_plan, lane_match_cfg.max_admits_per_transition) &&
        validateStateflowPlanExactSemantics(input_round_plan)) {
        score_stateflow_plan(input_round_plan, edge_bucket_sizes, layout);
        SPDLOG_DEBUG(
            "Stateflow multi-GPU candidate policy=input_rounds cost={:.3f} rounds={} loads={} boundaries={} "
            "directed_buckets={} estimated_bucket_edges={} overlap_hist={}",
            input_round_plan.estimated_cost, input_round_plan.total_superstates, input_round_plan.total_partition_loads,
            input_round_plan.boundary_count, input_round_plan.total_bucket_assignments, input_round_plan.estimated_bucket_edges,
            overlap_histogram_string(input_round_plan));
        candidates.emplace_back(std::move(input_round_plan));
    }

    auto grouped_permutation = getDisjointBufferStatePermutation(buffer_states, active_devices);
    StateflowPlan grouped_plan =
        build_multi_gpu_stateflow_plan_from_permutation(buffer_states, edge_buckets_per_buffer, grouped_permutation, active_devices,
                                                        PlanVariant::MULTI_GPU_DISJOINT_ROUNDS, edge_bucket_sizes,
                                                        partition_row_counts, layout);
    if (stateflow_plan_valid(grouped_plan) &&
        stateflow_plan_respects_max_admits(grouped_plan, lane_match_cfg.max_admits_per_transition) &&
        validateStateflowPlanExactSemantics(grouped_plan)) {
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
    if (stateflow_plan_valid(lane_matched_plan) &&
        stateflow_plan_respects_max_admits(lane_matched_plan, lane_match_cfg.max_admits_per_transition) &&
        validateStateflowPlanExactSemantics(lane_matched_plan)) {
        score_stateflow_plan(lane_matched_plan, edge_bucket_sizes, layout);
        SPDLOG_DEBUG(
            "Stateflow multi-GPU candidate policy=lane_matched cost={:.3f} rounds={} loads={} boundaries={} "
            "directed_buckets={} estimated_bucket_edges={} overlap_hist={}",
            lane_matched_plan.estimated_cost, lane_matched_plan.total_superstates, lane_matched_plan.total_partition_loads,
            lane_matched_plan.boundary_count, lane_matched_plan.total_bucket_assignments, lane_matched_plan.estimated_bucket_edges,
            overlap_histogram_string(lane_matched_plan));
        candidates.emplace_back(std::move(lane_matched_plan));
    }

    if (active_devices == 2 && buffer_states.size() == 20) {
        auto tw16_permutation =
            getTw16TwentyStateTwoGpuLaneMatchedPermutation(buffer_states, edge_buckets_per_buffer,
                                                           edge_bucket_sizes, partition_row_counts, layout);
        if (!tw16_permutation.empty()) {
            StateflowPlan tw16_plan =
                build_multi_gpu_stateflow_plan_from_permutation(buffer_states, edge_buckets_per_buffer,
                                                                tw16_permutation, active_devices,
                                                                PlanVariant::MULTI_GPU_LANE_MATCHED,
                                                                edge_bucket_sizes, partition_row_counts, layout);
            if (stateflow_plan_valid(tw16_plan) &&
                stateflow_plan_respects_max_admits(tw16_plan, lane_match_cfg.max_admits_per_transition) &&
                validateStateflowPlanExactSemantics(tw16_plan)) {
                score_stateflow_plan(tw16_plan, edge_bucket_sizes, layout);
                SPDLOG_INFO("Using TW16 p16/q4 2-GPU lane-matched plan as the selected bounded q4 schedule");
                std::vector<StateflowPlan> tw16_only;
                tw16_only.emplace_back(std::move(tw16_plan));
                return tw16_only;
            }
        }
    }

    if ((active_devices == 2 || active_devices == 4) && bounded_q4_optimal88_enabled() && buffer_states.size() == 88) {
        auto optimal88_permutation = active_devices == 2
                                         ? getOptimal88TwoGpuLaneMatchedPermutation(buffer_states, edge_buckets_per_buffer,
                                                                                   edge_bucket_sizes, partition_row_counts, layout)
                                         : getOptimal88MultiGpuLaneMatchedPermutation(buffer_states, edge_buckets_per_buffer,
                                                                                     edge_bucket_sizes, partition_row_counts, layout,
                                                                                     active_devices);
        if (!optimal88_permutation.empty()) {
            const vector<torch::Tensor> *optimal88_edge_buckets = &edge_buckets_per_buffer;
            vector<torch::Tensor> round_balanced_edge_buckets;
            if (active_devices == 2 || active_devices == 4) {
                round_balanced_edge_buckets =
                    assignRoundBalancedMultiGpuEdgeBuckets(buffer_states, optimal88_permutation, active_devices,
                                                           static_cast<int>(edge_bucket_sizes.empty()
                                                                                ? 0
                                                                                : std::llround(std::sqrt(static_cast<long double>(edge_bucket_sizes.size())))),
                                                           edge_bucket_sizes);
                if (!round_balanced_edge_buckets.empty()) {
                    optimal88_edge_buckets = &round_balanced_edge_buckets;
                }
            }
            StateflowPlan optimal88_plan =
                build_multi_gpu_stateflow_plan_from_permutation(buffer_states, *optimal88_edge_buckets,
                                                                optimal88_permutation, active_devices,
                                                                PlanVariant::MULTI_GPU_OPTIMAL88_LANE_MATCHED,
                                                                edge_bucket_sizes, partition_row_counts, layout);
            if (stateflow_plan_valid(optimal88_plan) &&
                stateflow_plan_respects_max_admits(optimal88_plan, lane_match_cfg.max_admits_per_transition) &&
                validateStateflowPlanExactSemantics(optimal88_plan)) {
                score_stateflow_plan(optimal88_plan, edge_bucket_sizes, layout);
                SPDLOG_DEBUG(
                    "Stateflow multi-GPU candidate policy=optimal88_lane_matched cost={:.3f} rounds={} loads={} boundaries={} "
                    "directed_buckets={} estimated_bucket_edges={} overlap_hist={}",
                    optimal88_plan.estimated_cost, optimal88_plan.total_superstates, optimal88_plan.total_partition_loads,
                    optimal88_plan.boundary_count, optimal88_plan.total_bucket_assignments, optimal88_plan.estimated_bucket_edges,
                    overlap_histogram_string(optimal88_plan));
                if (active_devices == 2 || active_devices == 4) {
                    SPDLOG_INFO("Using Optimal88 {}-GPU lane-matched plan as the selected bounded q4 schedule",
                                active_devices);
                    std::vector<StateflowPlan> optimal88_only;
                    optimal88_only.emplace_back(std::move(optimal88_plan));
                    return optimal88_only;
                }
                candidates.emplace_back(std::move(optimal88_plan));
            }
        }
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

struct Bounded88PathResult {
    std::vector<int> order;
    int64_t transition_overlap = -1;
    std::vector<int64_t> overlap_hist;
};

uint64_t bounded_state_mask(const std::vector<int> &state) {
    uint64_t mask = 0;
    for (auto partition : state) {
        if (partition >= 0 && partition < 64) {
            mask |= (uint64_t{1} << static_cast<unsigned>(partition));
        }
    }
    return mask;
}

int bounded_pair_index(int lhs, int rhs, int num_partitions) {
    if (lhs > rhs) {
        std::swap(lhs, rhs);
    }
    return lhs * num_partitions + rhs;
}

std::vector<int> bounded_state_pair_indices(const std::vector<int> &state, int num_partitions) {
    std::vector<int> pairs;
    for (std::size_t i = 0; i < state.size(); i++) {
        for (std::size_t j = i + 1; j < state.size(); j++) {
            pairs.emplace_back(bounded_pair_index(state[i], state[j], num_partitions));
        }
    }
    return pairs;
}

std::vector<int> bounded_pair_counts_for_states(const std::vector<std::vector<int>> &states, int num_partitions) {
    std::vector<int> counts(static_cast<std::size_t>(num_partitions) * static_cast<std::size_t>(num_partitions), 0);
    for (const auto &state : states) {
        for (auto pair_idx : bounded_state_pair_indices(state, num_partitions)) {
            counts[pair_idx]++;
        }
    }
    return counts;
}

bool bounded_q4_cover_is_valid(const std::vector<std::vector<int>> &states,
                               int num_partitions,
                               int buffer_capacity) {
    if (states.empty()) {
        return false;
    }
    std::vector<int64_t> vertex_counts(static_cast<std::size_t>(num_partitions), 0);
    std::vector<int> pair_counts(static_cast<std::size_t>(num_partitions) * static_cast<std::size_t>(num_partitions), 0);
    for (const auto &state : states) {
        if (static_cast<int>(state.size()) != buffer_capacity) {
            return false;
        }
        std::vector<int> sorted_state = state;
        std::sort(sorted_state.begin(), sorted_state.end());
        if (std::adjacent_find(sorted_state.begin(), sorted_state.end()) != sorted_state.end()) {
            return false;
        }
        for (auto partition : sorted_state) {
            if (partition < 0 || partition >= num_partitions) {
                return false;
            }
            vertex_counts[partition]++;
        }
        for (auto pair_idx : bounded_state_pair_indices(sorted_state, num_partitions)) {
            pair_counts[pair_idx]++;
        }
    }

    for (int partition = 0; partition < num_partitions; partition++) {
        if (vertex_counts[partition] <= 0) {
            return false;
        }
    }

    for (int lhs = 0; lhs < num_partitions; lhs++) {
        for (int rhs = lhs + 1; rhs < num_partitions; rhs++) {
            if (pair_counts[bounded_pair_index(lhs, rhs, num_partitions)] <= 0) {
                return false;
            }
        }
    }
    return true;
}

Bounded88PathResult best_bounded_q4_path_for_states(const std::vector<std::vector<int>> &states,
                                                    int buffer_capacity,
                                                    int max_admits,
                                                    std::mt19937_64 &rng,
                                                    int restarts,
                                                    int max_2opt_passes) {
    Bounded88PathResult best;
    const int num_states = static_cast<int>(states.size());
    if (num_states == 0) {
        return best;
    }
    const int min_overlap = std::max(0, buffer_capacity - std::max(0, max_admits));
    std::vector<uint64_t> masks;
    masks.reserve(states.size());
    for (const auto &state : states) {
        masks.emplace_back(bounded_state_mask(state));
    }

    std::vector<std::vector<int>> overlap(num_states, std::vector<int>(num_states, 0));
    for (int lhs = 0; lhs < num_states; lhs++) {
        for (int rhs = lhs + 1; rhs < num_states; rhs++) {
            const int shared = static_cast<int>(__builtin_popcountll(masks[lhs] & masks[rhs]));
            overlap[lhs][rhs] = shared;
            overlap[rhs][lhs] = shared;
        }
    }

    auto score_path = [&](const std::vector<int> &path, std::vector<int64_t> *hist_out) {
        int64_t score = 0;
        bool valid = true;
        std::vector<int64_t> hist(static_cast<std::size_t>(buffer_capacity + 1), 0);
        for (std::size_t idx = 1; idx < path.size(); idx++) {
            int shared = overlap[path[idx - 1]][path[idx]];
            if (shared < min_overlap) {
                valid = false;
            }
            if (shared >= 0 && shared < static_cast<int>(hist.size())) {
                hist[shared]++;
            }
            score += shared;
        }
        if (!valid) {
            return int64_t{-1};
        }
        if (hist_out != nullptr) {
            *hist_out = std::move(hist);
        }
        return score;
    };

    auto improve_2opt = [&](std::vector<int> path) {
        std::vector<std::pair<int, int>> ranges;
        ranges.reserve(static_cast<std::size_t>(num_states) * static_cast<std::size_t>(num_states));
        for (int begin = 0; begin < num_states - 2; begin++) {
            for (int end = begin + 1; end < num_states - 1; end++) {
                ranges.emplace_back(begin, end);
            }
        }
        for (int pass = 0; pass < max_2opt_passes; pass++) {
            bool improved = false;
            std::shuffle(ranges.begin(), ranges.end(), rng);
            for (const auto &[begin, end] : ranges) {
                int old_score = 0;
                int new_score = 0;
                if (begin > 0) {
                    old_score += overlap[path[begin - 1]][path[begin]];
                    new_score += overlap[path[begin - 1]][path[end]];
                }
                if (end + 1 < num_states) {
                    old_score += overlap[path[end]][path[end + 1]];
                    new_score += overlap[path[begin]][path[end + 1]];
                }
                if (new_score > old_score) {
                    std::reverse(path.begin() + begin, path.begin() + end + 1);
                    improved = true;
                    break;
                }
            }
            if (!improved) {
                break;
            }
        }
        return path;
    };

    restarts = std::max(1, restarts);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int restart = 0; restart < restarts; restart++) {
        std::vector<int> path;
        path.reserve(states.size());
        if (unit(rng) < 0.7) {
            std::uniform_int_distribution<int> start_dist(0, num_states - 1);
            int start = start_dist(rng);
            std::vector<uint8_t> used(static_cast<std::size_t>(num_states), 0);
            path.emplace_back(start);
            used[start] = 1;
            while (static_cast<int>(path.size()) < num_states) {
                int current = path.back();
                int best_shared = -1;
                std::vector<int> choices;
                for (int candidate = 0; candidate < num_states; candidate++) {
                    if (used[candidate] != 0) {
                        continue;
                    }
                    int shared = overlap[current][candidate];
                    if (shared > best_shared) {
                        best_shared = shared;
                        choices.clear();
                        choices.emplace_back(candidate);
                    } else if (shared == best_shared) {
                        choices.emplace_back(candidate);
                    }
                }
                if (choices.empty()) {
                    break;
                }
                std::uniform_int_distribution<int> pick(0, static_cast<int>(choices.size()) - 1);
                int next = choices[pick(rng)];
                used[next] = 1;
                path.emplace_back(next);
            }
        } else {
            path.resize(static_cast<std::size_t>(num_states));
            std::iota(path.begin(), path.end(), 0);
            std::shuffle(path.begin(), path.end(), rng);
        }
        if (static_cast<int>(path.size()) != num_states) {
            continue;
        }

        path = improve_2opt(std::move(path));
        std::vector<int64_t> hist;
        int64_t score = score_path(path, &hist);
        if (score > best.transition_overlap) {
            best.transition_overlap = score;
            best.order = std::move(path);
            best.overlap_hist = std::move(hist);
        }
    }
    return best;
}

struct Bounded88LoadScore {
    int64_t max_load = std::numeric_limits<int64_t>::max();
    long double sum_sq = std::numeric_limits<long double>::infinity();
};

Bounded88LoadScore score_bounded_q4_balanced_load(const std::vector<std::vector<int>> &states,
                                                  int num_partitions,
                                                  const std::vector<int64_t> &edge_bucket_sizes,
                                                  int max_iterations) {
    Bounded88LoadScore score;
    if (states.empty()) {
        return score;
    }
    const bool have_sizes = edge_bucket_sizes.size() == static_cast<std::size_t>(num_partitions * num_partitions);
    std::vector<std::vector<uint8_t>> contains(states.size(), std::vector<uint8_t>(num_partitions, 0));
    for (std::size_t state_idx = 0; state_idx < states.size(); state_idx++) {
        for (auto partition : states[state_idx]) {
            contains[state_idx][partition] = 1;
        }
    }

    struct Bucket {
        int src = 0;
        int dst = 0;
        int64_t weight = 1;
    };
    std::vector<Bucket> buckets;
    buckets.reserve(static_cast<std::size_t>(num_partitions) * static_cast<std::size_t>(num_partitions));
    for (int src = 0; src < num_partitions; src++) {
        for (int dst = 0; dst < num_partitions; dst++) {
            buckets.push_back(Bucket{src, dst, have_sizes ? edge_bucket_sizes[src * num_partitions + dst] : int64_t{1}});
        }
    }
    std::sort(buckets.begin(), buckets.end(), [](const Bucket &lhs, const Bucket &rhs) {
        return std::make_tuple(lhs.weight, lhs.src, lhs.dst) > std::make_tuple(rhs.weight, rhs.src, rhs.dst);
    });

    std::vector<std::vector<int>> compatible(static_cast<std::size_t>(num_partitions) *
                                             static_cast<std::size_t>(num_partitions));
    std::vector<std::vector<std::pair<int, int>>> assigned(states.size());
    std::vector<int64_t> loads(states.size(), 0);
    std::vector<int64_t> counts(states.size(), 0);
    for (const auto &bucket : buckets) {
        int bucket_idx = bucket.src * num_partitions + bucket.dst;
        int best_state = -1;
        auto best_key = std::make_tuple(std::numeric_limits<int64_t>::max(),
                                        std::numeric_limits<int64_t>::max(),
                                        std::numeric_limits<int>::max());
        for (int state_idx = 0; state_idx < static_cast<int>(states.size()); state_idx++) {
            if (contains[state_idx][bucket.src] == 0 || contains[state_idx][bucket.dst] == 0) {
                continue;
            }
            compatible[bucket_idx].emplace_back(state_idx);
            auto key = std::make_tuple(loads[state_idx], counts[state_idx], state_idx);
            if (key < best_key) {
                best_key = key;
                best_state = state_idx;
            }
        }
        if (best_state < 0) {
            return score;
        }
        assigned[best_state].emplace_back(bucket.src, bucket.dst);
        loads[best_state] += bucket.weight;
        counts[best_state]++;
    }

    auto current_max = [&]() {
        return *std::max_element(loads.begin(), loads.end());
    };
    auto current_sum_sq = [&]() {
        long double value = 0.0L;
        for (auto load : loads) {
            value += static_cast<long double>(load) * static_cast<long double>(load);
        }
        return value;
    };
    auto max_after_move = [&](int src_state, int dst_state, int64_t weight) {
        int64_t max_value = 0;
        for (int state_idx = 0; state_idx < static_cast<int>(loads.size()); state_idx++) {
            int64_t load = loads[state_idx];
            if (state_idx == src_state) {
                load -= weight;
            } else if (state_idx == dst_state) {
                load += weight;
            }
            max_value = std::max(max_value, load);
        }
        return max_value;
    };
    auto sum_sq_after_move = [&](long double before, int src_state, int dst_state, int64_t weight) {
        long double src_before = static_cast<long double>(loads[src_state]);
        long double dst_before = static_cast<long double>(loads[dst_state]);
        long double src_after = src_before - static_cast<long double>(weight);
        long double dst_after = dst_before + static_cast<long double>(weight);
        return before - src_before * src_before - dst_before * dst_before +
               src_after * src_after + dst_after * dst_after;
    };

    for (int iteration = 0; iteration < std::max(0, max_iterations); iteration++) {
        int64_t max_load = current_max();
        long double sum_sq = current_sum_sq();
        struct Move {
            bool valid = false;
            int64_t new_max = std::numeric_limits<int64_t>::max();
            long double new_sum_sq = std::numeric_limits<long double>::infinity();
            int64_t weight = 0;
            int src_state = -1;
            int dst_state = -1;
            std::pair<int, int> bucket{-1, -1};
        };
        Move best_move;
        for (int src_state = 0; src_state < static_cast<int>(assigned.size()); src_state++) {
            if (loads[src_state] != max_load) {
                continue;
            }
            for (const auto &bucket_pair : assigned[src_state]) {
                int bucket_idx = bucket_pair.first * num_partitions + bucket_pair.second;
                int64_t weight = have_sizes ? edge_bucket_sizes[bucket_idx] : int64_t{1};
                for (auto dst_state : compatible[bucket_idx]) {
                    if (dst_state == src_state) {
                        continue;
                    }
                    int64_t new_max = max_after_move(src_state, dst_state, weight);
                    if (new_max > max_load) {
                        continue;
                    }
                    long double new_sum_sq = sum_sq_after_move(sum_sq, src_state, dst_state, weight);
                    if (new_max < max_load || new_sum_sq + 0.5L < sum_sq) {
                        if (!best_move.valid ||
                            std::make_tuple(new_max, new_sum_sq, -weight, src_state, dst_state, bucket_pair) <
                                std::make_tuple(best_move.new_max, best_move.new_sum_sq, -best_move.weight,
                                                best_move.src_state, best_move.dst_state, best_move.bucket)) {
                            best_move = Move{true, new_max, new_sum_sq, weight, src_state, dst_state, bucket_pair};
                        }
                    }
                }
            }
        }
        if (!best_move.valid) {
            break;
        }
        auto &src_buckets = assigned[best_move.src_state];
        auto itr = std::find(src_buckets.begin(), src_buckets.end(), best_move.bucket);
        if (itr != src_buckets.end()) {
            src_buckets.erase(itr);
        }
        assigned[best_move.dst_state].emplace_back(best_move.bucket);
        loads[best_move.src_state] -= best_move.weight;
        loads[best_move.dst_state] += best_move.weight;
        counts[best_move.src_state]--;
        counts[best_move.dst_state]++;
    }

    score.max_load = current_max();
    score.sum_sq = current_sum_sq();
    return score;
}

std::vector<std::array<int, 3>> all_gf2_direction_triples(int num_partitions) {
    std::vector<std::array<int, 3>> triples;
    for (int lhs = 1; lhs < num_partitions; lhs++) {
        for (int rhs = lhs + 1; rhs < num_partitions; rhs++) {
            int third = lhs ^ rhs;
            if (third <= 0 || third >= num_partitions) {
                continue;
            }
            std::array<int, 3> triple{lhs, rhs, third};
            std::sort(triple.begin(), triple.end());
            triples.emplace_back(triple);
        }
    }
    std::sort(triples.begin(), triples.end());
    triples.erase(std::unique(triples.begin(), triples.end()), triples.end());
    return triples;
}

std::vector<std::array<int, 3>> build_gf2_partial_spread_direction_cover(int num_partitions) {
    auto triples = all_gf2_direction_triples(num_partitions);
    std::vector<std::array<int, 3>> selected;
    std::vector<uint8_t> used(static_cast<std::size_t>(num_partitions), 0);

    auto finish_partial_spread = [&]() -> std::vector<std::array<int, 3>> {
        std::vector<int> remaining;
        for (int direction = 1; direction < num_partitions; direction++) {
            if (used[direction] == 0) {
                remaining.emplace_back(direction);
            }
        }
        if (remaining.size() != 4) {
            return {};
        }
        for (int a_idx = 0; a_idx < 4; a_idx++) {
            for (int b_idx = a_idx + 1; b_idx < 4; b_idx++) {
                std::vector<int> other;
                for (int idx = 0; idx < 4; idx++) {
                    if (idx != a_idx && idx != b_idx) {
                        other.emplace_back(remaining[idx]);
                    }
                }
                int duplicate = remaining[a_idx] ^ remaining[b_idx];
                if (duplicate <= 0 || duplicate >= num_partitions || used[duplicate] == 0) {
                    continue;
                }
                if ((other[0] ^ other[1]) != duplicate) {
                    continue;
                }
                std::array<int, 3> first{duplicate, remaining[a_idx], remaining[b_idx]};
                std::array<int, 3> second{duplicate, other[0], other[1]};
                std::sort(first.begin(), first.end());
                std::sort(second.begin(), second.end());
                if (second < first) {
                    std::swap(first, second);
                }
                return {first, second};
            }
        }
        return {};
    };

    std::function<std::vector<std::array<int, 3>>()> dfs = [&]() -> std::vector<std::array<int, 3>> {
        if (selected.size() == 9) {
            auto tail = finish_partial_spread();
            if (!tail.empty()) {
                std::vector<std::array<int, 3>> result = selected;
                result.insert(result.end(), tail.begin(), tail.end());
                return result;
            }
            return {};
        }
        for (const auto &triple : triples) {
            if (used[triple[0]] != 0 || used[triple[1]] != 0 || used[triple[2]] != 0) {
                continue;
            }
            selected.emplace_back(triple);
            used[triple[0]] = used[triple[1]] = used[triple[2]] = 1;
            auto result = dfs();
            if (!result.empty()) {
                return result;
            }
            used[triple[0]] = used[triple[1]] = used[triple[2]] = 0;
            selected.pop_back();
        }
        return {};
    };

    return dfs();
}

std::vector<std::vector<int>> build_gf2_affine_q4_cover_states(int num_partitions, int buffer_capacity) {
    if (buffer_capacity != 4 || num_partitions != 32) {
        return {};
    }
    auto direction_cover = build_gf2_partial_spread_direction_cover(num_partitions);
    if (direction_cover.size() != 11) {
        SPDLOG_WARN("Failed to generate GF(2)^5 affine q4 direction cover");
        return {};
    }

    std::vector<std::vector<int>> states;
    states.reserve(direction_cover.size() * 8);
    for (const auto &triple : direction_cover) {
        int x = -1;
        int y = -1;
        for (int i = 0; i < 3 && x < 0; i++) {
            for (int j = i + 1; j < 3; j++) {
                int candidate = triple[i] ^ triple[j];
                if (candidate == triple[0] || candidate == triple[1] || candidate == triple[2]) {
                    x = triple[i];
                    y = triple[j];
                    break;
                }
            }
        }
        if (x < 0 || y < 0) {
            return {};
        }
        std::array<int, 4> subspace{0, x, y, x ^ y};
        std::vector<uint8_t> seen(static_cast<std::size_t>(num_partitions), 0);
        for (int base = 0; base < num_partitions; base++) {
            if (seen[base] != 0) {
                continue;
            }
            std::vector<int> state;
            state.reserve(4);
            for (auto offset : subspace) {
                int partition = base ^ offset;
                state.emplace_back(partition);
                seen[partition] = 1;
            }
            std::sort(state.begin(), state.end());
            states.emplace_back(std::move(state));
        }
    }

    if (!bounded_q4_cover_is_valid(states, num_partitions, buffer_capacity)) {
        SPDLOG_WARN("Generated GF(2)^5 affine q4 cover failed validation");
        return {};
    }
    return states;
}

bool bounded_state_exists_except(const std::vector<std::vector<int>> &states,
                                 const std::vector<int> &needle,
                                 int skip_lhs,
                                 int skip_rhs) {
    for (int idx = 0; idx < static_cast<int>(states.size()); idx++) {
        if (idx == skip_lhs || idx == skip_rhs) {
            continue;
        }
        if (states[idx] == needle) {
            return true;
        }
    }
    return false;
}

bool try_bounded_q4_two_block_trade(const std::vector<std::vector<int>> &states,
                                    const std::vector<int> &pair_counts,
                                    int num_partitions,
                                    std::mt19937_64 &rng,
                                    int max_alternatives,
                                    std::vector<std::vector<int>> *out_states,
                                    std::vector<int> *out_pair_counts) {
    if (states.size() < 2) {
        return false;
    }
    std::uniform_int_distribution<int> state_dist(0, static_cast<int>(states.size()) - 1);
    int lhs_idx = state_dist(rng);
    int rhs_idx = state_dist(rng);
    if (lhs_idx == rhs_idx) {
        return false;
    }
    if (lhs_idx > rhs_idx) {
        std::swap(lhs_idx, rhs_idx);
    }

    const auto &old_lhs = states[lhs_idx];
    const auto &old_rhs = states[rhs_idx];
    std::vector<int> common;
    std::vector<int> only;
    for (auto value : old_lhs) {
        if (std::find(old_rhs.begin(), old_rhs.end(), value) != old_rhs.end()) {
            common.emplace_back(value);
        } else {
            only.emplace_back(value);
        }
    }
    for (auto value : old_rhs) {
        if (std::find(old_lhs.begin(), old_lhs.end(), value) == old_lhs.end()) {
            only.emplace_back(value);
        }
    }
    std::sort(common.begin(), common.end());
    std::sort(only.begin(), only.end());
    const int need = static_cast<int>(old_lhs.size()) - static_cast<int>(common.size());
    if (need < 0 || need > static_cast<int>(only.size())) {
        return false;
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> alternatives;
    const int only_count = static_cast<int>(only.size());
    const int64_t subset_count = int64_t{1} << only_count;
    for (int64_t mask = 0; mask < subset_count; mask++) {
        if (__builtin_popcountll(static_cast<unsigned long long>(mask)) != need) {
            continue;
        }
        std::vector<int> lhs = common;
        std::vector<int> rhs = common;
        for (int idx = 0; idx < only_count; idx++) {
            if ((mask & (int64_t{1} << idx)) != 0) {
                lhs.emplace_back(only[idx]);
            } else {
                rhs.emplace_back(only[idx]);
            }
        }
        std::sort(lhs.begin(), lhs.end());
        std::sort(rhs.begin(), rhs.end());
        if (lhs == rhs) {
            continue;
        }
        auto old_pair = std::minmax(old_lhs, old_rhs);
        auto new_pair = std::minmax(lhs, rhs);
        if (old_pair.first == new_pair.first && old_pair.second == new_pair.second) {
            continue;
        }
        alternatives.emplace_back(std::move(lhs), std::move(rhs));
    }
    if (alternatives.empty()) {
        return false;
    }
    std::shuffle(alternatives.begin(), alternatives.end(), rng);
    if (max_alternatives > 0 && static_cast<int>(alternatives.size()) > max_alternatives) {
        alternatives.resize(static_cast<std::size_t>(max_alternatives));
    }

    const auto old_lhs_pairs = bounded_state_pair_indices(old_lhs, num_partitions);
    const auto old_rhs_pairs = bounded_state_pair_indices(old_rhs, num_partitions);
    for (const auto &[new_lhs, new_rhs] : alternatives) {
        if (bounded_state_exists_except(states, new_lhs, lhs_idx, rhs_idx) ||
            bounded_state_exists_except(states, new_rhs, lhs_idx, rhs_idx)) {
            continue;
        }

        std::vector<int> next_pair_counts = pair_counts;
        for (auto pair_idx : old_lhs_pairs) {
            next_pair_counts[pair_idx]--;
        }
        for (auto pair_idx : old_rhs_pairs) {
            next_pair_counts[pair_idx]--;
        }
        for (auto pair_idx : bounded_state_pair_indices(new_lhs, num_partitions)) {
            next_pair_counts[pair_idx]++;
        }
        for (auto pair_idx : bounded_state_pair_indices(new_rhs, num_partitions)) {
            next_pair_counts[pair_idx]++;
        }

        bool valid = true;
        for (int lhs = 0; lhs < num_partitions && valid; lhs++) {
            for (int rhs = lhs + 1; rhs < num_partitions; rhs++) {
                if (next_pair_counts[bounded_pair_index(lhs, rhs, num_partitions)] <= 0) {
                    valid = false;
                    break;
                }
            }
        }
        if (!valid) {
            continue;
        }

        std::vector<std::vector<int>> next_states = states;
        next_states[lhs_idx] = new_lhs;
        next_states[rhs_idx] = new_rhs;
        *out_states = std::move(next_states);
        *out_pair_counts = std::move(next_pair_counts);
        return true;
    }
    return false;
}

std::vector<std::vector<int>> apply_bounded_order(const std::vector<std::vector<int>> &states,
                                                  const std::vector<int> &order) {
    std::vector<std::vector<int>> ordered;
    ordered.reserve(states.size());
    for (auto state_idx : order) {
        if (state_idx < 0 || state_idx >= static_cast<int>(states.size())) {
            return {};
        }
        ordered.emplace_back(states[state_idx]);
    }
    return ordered;
}

std::vector<std::vector<int>> build_weighted_optimal88_q4_states(int num_partitions,
                                                                 int buffer_capacity,
                                                                 int max_admits,
                                                                 const std::vector<int64_t> &edge_bucket_sizes,
                                                                 bool use_load_tiebreaker) {
    if (num_partitions != 32 || buffer_capacity != 4 || max_admits < 3) {
        SPDLOG_WARN(
            "Ignoring GEGE_BOUNDED_Q4_OPTIMAL88=1: requires num_partitions=32 buffer_capacity=4 max_admits>=3; got num_partitions={} buffer_capacity={} max_admits={}",
            num_partitions, buffer_capacity, max_admits);
        return {};
    }

    auto current_states = build_gf2_affine_q4_cover_states(num_partitions, buffer_capacity);
    if (current_states.empty()) {
        return {};
    }

    const int iterations = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_ITERS", nullptr, 5000)));
    const int path_restarts = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_PATH_RESTARTS", nullptr, 40)));
    const int path_2opt_passes = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_PATH_2OPT_PASSES", nullptr, 80)));
    const int trade_alternatives = static_cast<int>(
        std::max<int64_t>(1, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_TRADE_ALTS", nullptr, 30)));
    const int minmax_iters = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MINMAX_ITERS", nullptr, 10000)));
    const int target_admits = static_cast<int>(
        std::max<int64_t>(0, stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_TARGET_ADMITS", nullptr, 229)));
    const int min_path_overlap = static_cast<int>(
        stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_MIN_PATH_OVERLAP", nullptr, 109));
    const bool stop_at_target =
        stateflow_env_bool("GEGE_BOUNDED_Q4_OPTIMAL88_STOP_AT_TARGET", nullptr, true);
    const int64_t seed = stateflow_env_int64("GEGE_BOUNDED_Q4_OPTIMAL88_SEED", nullptr, 17);

    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    auto current_path =
        best_bounded_q4_path_for_states(current_states, buffer_capacity, max_admits, rng, path_restarts, path_2opt_passes);
    if (current_path.order.empty()) {
        SPDLOG_WARN("Dynamic p32/q4 88-state scheduler could not order initial affine cover");
        return {};
    }
    auto load_score_for = [&](const std::vector<std::vector<int>> &states) {
        Bounded88LoadScore load;
        if (use_load_tiebreaker) {
            return score_bounded_q4_balanced_load(states, num_partitions, edge_bucket_sizes, minmax_iters);
        }
        load.max_load = 0;
        load.sum_sq = 0.0L;
        return load;
    };

    auto current_load = load_score_for(current_states);
    auto current_pairs = bounded_pair_counts_for_states(current_states, num_partitions);
    auto best_states = current_states;
    auto best_path = current_path;
    auto best_load = current_load;

    auto transition_admits_for_path = [buffer_capacity](const std::vector<std::vector<int>> &states,
                                                        const Bounded88PathResult &path) {
        return static_cast<int64_t>(std::max(0, static_cast<int>(states.size()) - 1)) *
                   static_cast<int64_t>(buffer_capacity) -
               path.transition_overlap;
    };

    auto score_key = [&](const std::vector<std::vector<int>> &states,
                         const Bounded88LoadScore &load,
                         const Bounded88PathResult &path) {
        const int64_t admits = transition_admits_for_path(states, path);
        const bool above_target = target_admits > 0 && admits > target_admits;
        return std::make_tuple(admits,
                               -path.transition_overlap,
                               use_load_tiebreaker && !above_target ? load.max_load : int64_t{0},
                               use_load_tiebreaker && !above_target ? load.sum_sq : 0.0L);
    };
    auto current_key = score_key(current_states, current_load, current_path);
    auto best_key = current_key;

    int tested = 0;
    int accepted = 0;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int iter = 0; iter < iterations; iter++) {
        std::vector<std::vector<int>> candidate_states;
        std::vector<int> candidate_pairs;
        if (!try_bounded_q4_two_block_trade(current_states, current_pairs, num_partitions, rng,
                                            trade_alternatives, &candidate_states, &candidate_pairs)) {
            continue;
        }
        tested++;
        auto candidate_path =
            best_bounded_q4_path_for_states(candidate_states, buffer_capacity, max_admits, rng, path_restarts, path_2opt_passes);
        if (candidate_path.order.empty() || candidate_path.transition_overlap < min_path_overlap) {
            continue;
        }
        auto candidate_load = load_score_for(candidate_states);
        auto candidate_key = score_key(candidate_states, candidate_load, candidate_path);
        if (candidate_key <= current_key || unit(rng) < 0.01) {
            current_states = candidate_states;
            current_pairs = candidate_pairs;
            current_path = candidate_path;
            current_load = candidate_load;
            current_key = candidate_key;
            accepted++;
        }
        if (candidate_key < best_key) {
            best_states = std::move(candidate_states);
            best_path = std::move(candidate_path);
            best_load = candidate_load;
            best_key = candidate_key;
            const int64_t best_admits = transition_admits_for_path(best_states, best_path);
            if (stop_at_target && best_admits <= target_admits) {
                break;
            }
        }
    }

    const int64_t final_admits = transition_admits_for_path(best_states, best_path);
    SPDLOG_INFO(
        "Dynamic p32/q4 88-state bounded scheduler selected states={} tested={} accepted={} transition_overlap={} transition_admits={} load_tiebreaker={} load_max={} load_sq={:.3e}",
        best_states.size(), tested, accepted, best_path.transition_overlap, final_admits,
        use_load_tiebreaker,
        best_load.max_load == std::numeric_limits<int64_t>::max() ? int64_t{0} : best_load.max_load,
        static_cast<double>(best_load.sum_sq));

    auto ordered = apply_bounded_order(best_states, best_path.order);
    if (!bounded_q4_cover_is_valid(ordered, num_partitions, buffer_capacity)) {
        SPDLOG_WARN("Dynamic p32/q4 88-state bounded scheduler produced invalid cover");
        return {};
    }
    return ordered;
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

    std::vector<std::vector<int>> buffer_states = build_greedy_cover_state_set(num_partitions, buffer_capacity);
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

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getBoundedGreedyCoverEdgeBucketOrdering(
    int num_partitions,
    int buffer_capacity,
    const vector<int64_t> &edge_bucket_sizes) {
    if (buffer_capacity != 4 || num_partitions <= 0 || buffer_capacity > num_partitions) {
        SPDLOG_WARN("Bounded GREEDY_COVER q4 requires buffer_capacity=4 and valid num_partitions; got num_partitions={} buffer_capacity={}",
                    num_partitions, buffer_capacity);
        return getGreedyCoverEdgeBucketOrdering(num_partitions, buffer_capacity);
    }

    const int requested_max_admits = static_cast<int>(
        std::min<int64_t>(buffer_capacity,
                          std::max<int64_t>(0, stateflow_env_int64("GEGE_STATEFLOW_MAX_ADMITS",
                                                                    "STATEFLOW_MAX_ADMITS", 3))));
    std::vector<std::vector<int>> buffer_states;
    bool loaded_state_order = false;
    if (bounded_q4_optimal88_enabled()) {
        buffer_states = build_weighted_optimal88_q4_states(num_partitions, buffer_capacity,
                                                           requested_max_admits, edge_bucket_sizes,
                                                           false);
        loaded_state_order = !buffer_states.empty();
    }
    if (buffer_states.empty()) {
        buffer_states =
            load_bounded_state_order_file(std::getenv("GEGE_BOUNDED_STATE_ORDER_FILE"), num_partitions, buffer_capacity);
        loaded_state_order = !buffer_states.empty();
    }
    if (!loaded_state_order) {
        buffer_states = build_greedy_cover_state_set(num_partitions, buffer_capacity);
    }
    if (requested_max_admits < 3) {
        auto constrained_states =
            build_bounded_single_gpu_cover_states(num_partitions, buffer_capacity, requested_max_admits);
        if (!constrained_states.empty()) {
            buffer_states = std::move(constrained_states);
        }
    }
    if (buffer_states.empty()) {
        return convertEdgeBucketOrderToTensors({}, {});
    }

    double pre_reorder_retained_avg = 0.0;
    if (buffer_states.size() > 1) {
        int64_t retained_total = 0;
        for (std::size_t idx = 1; idx < buffer_states.size(); idx++) {
            retained_total += state_overlap_count(buffer_states[idx - 1], buffer_states[idx]);
        }
        pre_reorder_retained_avg = static_cast<double>(retained_total) / static_cast<double>(buffer_states.size() - 1);
    }

    if (!loaded_state_order && requested_max_admits >= 3 && requested_max_admits < buffer_capacity) {
        buffer_states = reorder_states_for_bounded_admits(buffer_states, requested_max_admits);
    }
    if (bounded_greedy_cover_reverse_enabled()) {
        std::reverse(buffer_states.begin(), buffer_states.end());
        SPDLOG_INFO("Reversed BOUNDED_GREEDY_COVER_Q4 state order");
    }

    int64_t retained_total = 0;
    for (std::size_t idx = 1; idx < buffer_states.size(); idx++) {
        retained_total += state_overlap_count(buffer_states[idx - 1], buffer_states[idx]);
    }
    double retained_avg = buffer_states.size() > 1
                              ? static_cast<double>(retained_total) / static_cast<double>(buffer_states.size() - 1)
                              : 0.0;

    auto edge_buckets_per_buffer = balancedAssignEdgeBucketsToBuffers(buffer_states, num_partitions, edge_bucket_sizes);
    log_cover_ordering_summary("Generating BOUNDED_GREEDY_COVER_Q4 Ordering", buffer_states, edge_buckets_per_buffer,
                               num_partitions, edge_bucket_sizes);
    SPDLOG_INFO("BOUNDED_GREEDY_COVER_Q4 retained_avg={:.3f} pre_reorder_retained_avg={:.3f}",
                retained_avg, pre_reorder_retained_avg);
    return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
}

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getBoundedGreedyCoverMultiGpuEdgeBucketOrdering(
    int num_partitions,
    int buffer_capacity,
    int active_devices,
    const vector<int64_t> &edge_bucket_sizes) {
    if (buffer_capacity != 4 || num_partitions <= 0 || buffer_capacity > num_partitions || active_devices <= 1) {
        SPDLOG_WARN(
            "Bounded multi-GPU GREEDY_COVER q4 requires active_devices>1, buffer_capacity=4, and valid num_partitions; got active_devices={} num_partitions={} buffer_capacity={}",
            active_devices, num_partitions, buffer_capacity);
        return getBoundedGreedyCoverEdgeBucketOrdering(num_partitions, buffer_capacity, edge_bucket_sizes);
    }

    const int max_admits = static_cast<int>(
        std::min<int64_t>(buffer_capacity, std::max<int64_t>(0, stateflow_env_int64("GEGE_STATEFLOW_MAX_ADMITS",
                                                                                   "STATEFLOW_MAX_ADMITS", 3))));
    std::vector<std::vector<int>> buffer_states;
    bool using_optimal88_states = false;
    if ((active_devices == 2 || active_devices == 4) && bounded_q4_optimal88_enabled()) {
        buffer_states = build_weighted_optimal88_q4_states(num_partitions, buffer_capacity,
                                                           std::max(max_admits, 3), edge_bucket_sizes,
                                                           false);
        if (!buffer_states.empty()) {
            using_optimal88_states = true;
            SPDLOG_INFO("Using Optimal88 p32/q4 state cover as bounded multi-GPU input states={}", buffer_states.size());
        }
    }
    if (buffer_states.empty()) {
        buffer_states =
            load_bounded_state_order_file(std::getenv("GEGE_BOUNDED_STATE_ORDER_FILE"), num_partitions, buffer_capacity);
        if (!buffer_states.empty()) {
            SPDLOG_INFO("Using GEGE_BOUNDED_STATE_ORDER_FILE as bounded multi-GPU input states={}", buffer_states.size());
        }
    }
    if (buffer_states.empty()) {
        buffer_states = build_bounded_multi_gpu_round_states(num_partitions, buffer_capacity, active_devices, max_admits);
    }
    if (buffer_states.empty()) {
        return getBoundedGreedyCoverEdgeBucketOrdering(num_partitions, buffer_capacity, edge_bucket_sizes);
    }

    vector<vector<std::pair<int, int>>> edge_buckets_per_buffer;
    if ((active_devices == 2 || active_devices == 4) && using_optimal88_states) {
        SPDLOG_INFO("Using first-cover edge-bucket assignment for Optimal88 {}-GPU bounded q4 schedule",
                    active_devices);
        edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
    } else {
        edge_buckets_per_buffer =
            balancedAssignEdgeBucketsToBuffers(buffer_states, num_partitions, edge_bucket_sizes, active_devices);
    }
    log_cover_ordering_summary("Generating BOUNDED_MULTI_GPU_GREEDY_COVER_Q4 Ordering (legacy interleaved view)",
                               buffer_states, edge_buckets_per_buffer, num_partitions, edge_bucket_sizes);
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
        if (solver == LaneMatchSolver::OPTIMAL2 && !previous_group.empty()) {
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
    if (buffer_capacity != 4) {
        bool lifted_q4 = can_lift_q4_custom_template(num_partitions, buffer_capacity);
        auto buffer_states = lifted_q4
                                 ? build_randomized_lifted_q4_custom_template_buffer_states(num_partitions, buffer_capacity)
                                 : build_custom_template_buffer_states(num_partitions, buffer_capacity);
        if (buffer_states.empty()) {
            return convertEdgeBucketOrderToTensors({}, {});
        }

        if (!lifted_q4) {
            Indices all_partitions_map = torch::randperm(num_partitions, torch::kInt32);
            for (auto &state : buffer_states) {
                for (auto &partition : state) {
                    partition = all_partitions_map[partition].item<int>();
                }
            }
        }

        Indices all_buffer_map = torch::randperm(buffer_states.size(), torch::kInt32);
        std::vector<std::vector<int>> shuffle_buffer_states;
        shuffle_buffer_states.reserve(buffer_states.size());
        for (int i = 0; i < static_cast<int>(buffer_states.size()); i++) {
            shuffle_buffer_states.push_back(buffer_states[all_buffer_map[i].item<int>()]);
        }
        buffer_states = shuffle_buffer_states;

        std::vector<std::vector<std::pair<int, int>>> edge_buckets_per_buffer;
        if (randomly_assign_edge_buckets) {
            edge_buckets_per_buffer = randomlyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
        } else {
            edge_buckets_per_buffer = greedyAssignEdgeBucketsToBuffers(buffer_states, num_partitions);
        }

        int64_t assigned_buckets = 0;
        int64_t retained_total = 0;
        for (const auto &buckets : edge_buckets_per_buffer) {
            assigned_buckets += static_cast<int64_t>(buckets.size());
        }
        for (std::size_t idx = 1; idx < buffer_states.size(); idx++) {
            retained_total += state_overlap_count(buffer_states[idx - 1], buffer_states[idx]);
        }
        double retained_avg = buffer_states.size() > 1
                                  ? static_cast<double>(retained_total) / static_cast<double>(buffer_states.size() - 1)
                                  : 0.0;
        SPDLOG_INFO("Generating CUSTOM {} Ordering: states={} assigned_buckets={} retained_avg={:.3f}",
                    lifted_q4 ? "lifted-q4" : "block-pair cover", buffer_states.size(), assigned_buckets, retained_avg);
        return convertEdgeBucketOrderToTensors(buffer_states, edge_buckets_per_buffer);
    }

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
