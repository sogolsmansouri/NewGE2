#include "data/ordering.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class ScopedEnvVar {
  public:
    ScopedEnvVar(const char *name, const char *value) : name_(name) {
        const char *existing = std::getenv(name_);
        if (existing != nullptr) {
            had_previous_ = true;
            previous_ = existing;
        }
        if (value != nullptr) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            setenv(name_, previous_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

  private:
    const char *name_;
    bool had_previous_ = false;
    std::string previous_;
};

StateflowPlan build_valid_multi_gpu_plan(PlanVariant variant, int active_devices = 2) {
    constexpr int kNumPartitions = 16;
    constexpr int kBufferCapacity = 4;

    auto [buffer_states, edge_buckets] = getCustomEdgeBucketOrdering(kNumPartitions, kBufferCapacity, false);
    std::vector<int64_t> edge_bucket_sizes(static_cast<std::size_t>(kNumPartitions * kNumPartitions), 1);
    std::vector<int64_t> partition_row_counts(static_cast<std::size_t>(kNumPartitions), 1024);
    auto candidates =
        enumerateMultiGpuStateflowPlans(buffer_states, edge_buckets, active_devices, edge_bucket_sizes, partition_row_counts, {});

    for (const auto &candidate : candidates) {
        if (candidate.family_variant == variant) {
            return candidate;
        }
    }

    throw std::runtime_error("failed to build requested multi-GPU candidate");
}

void expect_true(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_false(bool condition, const std::string &message) {
    if (condition) {
        throw std::runtime_error(message);
    }
}

std::vector<int64_t> tensor_to_vector(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
}

void test_reject_cross_lane_handoff_in_lane_plan() {
    auto plan = build_valid_multi_gpu_plan(PlanVariant::MULTI_GPU_DISJOINT_ROUNDS);
    expect_true(validateStateflowPlanExactSemantics(plan), "baseline disjoint plan should validate");
    expect_true(plan.lanes.size() >= 2 && !plan.lanes[0].handoffs.empty(), "expected two lanes and one lane-local handoff");

    auto &handoff = plan.lanes[0].handoffs[0];
    handoff.dst_lane_id = plan.lanes[1].lane_id;

    expect_false(validateStateflowPlanExactSemantics(plan), "lane-local handoff with cross-lane dst should fail validation");
}

void test_reject_peer_relay_without_cross_lane_descriptor() {
    auto plan = build_valid_multi_gpu_plan(PlanVariant::MULTI_GPU_DISJOINT_ROUNDS);
    expect_true(validateStateflowPlanExactSemantics(plan), "baseline disjoint plan should validate");
    expect_true(!plan.lanes[0].handoffs.empty(), "expected at least one lane-local handoff");

    plan.lanes[0].handoffs[0].mode = HandoffMode::PEER_RELAY;

    expect_false(validateStateflowPlanExactSemantics(plan), "lane-local PEER_RELAY without cross-lane descriptor should fail");
}

void test_reject_bucket_coverage_gap() {
    auto plan = build_valid_multi_gpu_plan(PlanVariant::MULTI_GPU_DISJOINT_ROUNDS);
    expect_true(validateStateflowPlanExactSemantics(plan), "baseline disjoint plan should validate");
    expect_true(!plan.lanes.empty() && !plan.lanes[0].microstates.empty(), "expected at least one microstate");

    auto &microstate = plan.lanes[0].microstates[0];
    expect_true(!microstate.edge_buckets.empty() && !microstate.active_fragments.empty(), "expected edge buckets and fragments");

    const auto removed_bucket = microstate.edge_buckets.back();
    microstate.edge_buckets.pop_back();

    auto frag_it = std::find_if(microstate.active_fragments.begin(), microstate.active_fragments.end(), [&](const FragmentPlan &frag) {
        return frag.edge_buckets.size() == 1 && frag.edge_buckets.front() == removed_bucket;
    });
    expect_true(frag_it != microstate.active_fragments.end(), "expected a fragment matching the removed bucket");
    microstate.active_fragments.erase(frag_it);

    expect_false(validateStateflowPlanExactSemantics(plan), "bucket coverage gap should fail validation");
}

void test_four_gpu_lane_matched_candidate_validates() {
    auto plan = build_valid_multi_gpu_plan(PlanVariant::MULTI_GPU_LANE_MATCHED, 4);
    expect_true(plan.gpu_count == 4, "expected 4-GPU lane-matched candidate");
    expect_true(validateStateflowPlanExactSemantics(plan), "4-GPU lane-matched plan should validate");
}

void test_peer_aware_lane_matching_reduces_lane_matched_cost() {
    ScopedEnvVar host_only_env("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "0");
    auto host_only = build_valid_multi_gpu_plan(PlanVariant::MULTI_GPU_LANE_MATCHED);
    expect_true(validateStateflowPlanExactSemantics(host_only), "host-only lane-matched plan should validate");
    expect_true(host_only.total_peer_handoff_bytes > 0, "expected lane-matched plan to expose peer opportunities");

    ScopedEnvVar peer_aware_env("GEGE_STATEFLOW_ALLOW_PEER_RELAY", "1");
    auto peer_aware = build_valid_multi_gpu_plan(PlanVariant::MULTI_GPU_LANE_MATCHED);
    expect_true(validateStateflowPlanExactSemantics(peer_aware), "peer-aware lane-matched plan should validate");
    expect_true(peer_aware.total_peer_handoff_bytes > 0, "expected peer-aware lane-matched plan to expose peer opportunities");
    expect_true(peer_aware.estimated_cost <= host_only.estimated_cost,
                "peer-aware lane-matched cost should not exceed the host-only cost");
}

void expect_lane_max_admits(const StateflowPlan &plan, int64_t max_admits) {
    for (const auto &lane : plan.lanes) {
        for (const auto &microstate : lane.microstates) {
            if (microstate.microstate_id == 0) {
                continue;
            }
            expect_true(microstate.admitted_partitions <= max_admits,
                        "lane transition exceeded max admits");
        }
    }
}

void expect_tensor_schedule_max_admits(const std::vector<torch::Tensor> &buffer_states,
                                       int64_t buffer_capacity,
                                       int64_t max_admits) {
    for (std::size_t state_idx = 1; state_idx < buffer_states.size(); state_idx++) {
        auto previous = tensor_to_vector(buffer_states[state_idx - 1]);
        auto current = tensor_to_vector(buffer_states[state_idx]);
        int64_t shared = 0;
        for (auto lhs : previous) {
            for (auto rhs : current) {
                if (lhs == rhs) {
                    shared++;
                    break;
                }
            }
        }
        expect_true(buffer_capacity - shared <= max_admits,
                    "single-GPU transition exceeded max admits");
    }
}

void expect_tensor_schedule_max_admit_run(const std::vector<torch::Tensor> &buffer_states,
                                          int64_t buffer_capacity,
                                          int64_t max_admits,
                                          int64_t max_run) {
    int64_t current_run = 0;
    int64_t observed_max_run = 0;
    for (std::size_t state_idx = 1; state_idx < buffer_states.size(); state_idx++) {
        auto previous = tensor_to_vector(buffer_states[state_idx - 1]);
        auto current = tensor_to_vector(buffer_states[state_idx]);
        int64_t shared = 0;
        for (auto lhs : previous) {
            for (auto rhs : current) {
                if (lhs == rhs) {
                    shared++;
                    break;
                }
            }
        }
        if (buffer_capacity - shared == max_admits) {
            current_run++;
            observed_max_run = std::max(observed_max_run, current_run);
        } else {
            current_run = 0;
        }
    }
    expect_true(observed_max_run <= max_run,
                "single-GPU bounded schedule clustered too many max-admit swaps");
}

void expect_exact_diagonal_bucket_coverage(const std::vector<torch::Tensor> &edge_buckets,
                                           int num_partitions) {
    std::vector<int64_t> bucket_counts(static_cast<std::size_t>(num_partitions * num_partitions), 0);
    for (const auto &bucket_tensor : edge_buckets) {
        if (!bucket_tensor.defined() || bucket_tensor.numel() == 0) {
            continue;
        }
        auto values = tensor_to_vector(bucket_tensor);
        expect_true(values.size() % 2 == 0, "edge bucket tensor should contain src,dst pairs");
        for (std::size_t idx = 0; idx + 1 < values.size(); idx += 2) {
            int64_t src = values[idx];
            int64_t dst = values[idx + 1];
            expect_true(src >= 0 && src < num_partitions && dst >= 0 && dst < num_partitions,
                        "edge bucket id out of range");
            bucket_counts[static_cast<std::size_t>(src * num_partitions + dst)]++;
        }
    }

    for (int src = 0; src < num_partitions; src++) {
        for (int dst = 0; dst < num_partitions; dst++) {
            expect_true(bucket_counts[static_cast<std::size_t>(src * num_partitions + dst)] == 1,
                        "directed bucket should be assigned exactly once");
        }
        expect_true(bucket_counts[static_cast<std::size_t>(src * num_partitions + src)] == 1,
                    "diagonal bucket should be assigned exactly once");
    }
}

void expect_static_p30_q4_schedule_stats(const std::vector<torch::Tensor> &buffer_states) {
    constexpr int kNumPartitions = 30;
    constexpr int kBufferCapacity = 4;

    expect_true(buffer_states.size() == 75, "p30/q4 static schedule should have 75 states");

    std::vector<int64_t> rep_counts(kNumPartitions, 0);
    std::vector<int64_t> overlap_hist(kBufferCapacity + 1, 0);
    int64_t transition_overlap = 0;
    std::vector<int64_t> previous;

    for (const auto &state_tensor : buffer_states) {
        auto partitions = tensor_to_vector(state_tensor);
        expect_true(partitions.size() == kBufferCapacity, "p30/q4 state should have four partitions");
        std::sort(partitions.begin(), partitions.end());
        expect_true(std::adjacent_find(partitions.begin(), partitions.end()) == partitions.end(),
                    "p30/q4 state should not repeat partitions");
        for (auto partition : partitions) {
            expect_true(partition >= 0 && partition < kNumPartitions, "p30/q4 partition id out of range");
            rep_counts[static_cast<std::size_t>(partition)]++;
        }

        if (!previous.empty()) {
            int64_t overlap = 0;
            for (auto lhs : previous) {
                for (auto rhs : partitions) {
                    if (lhs == rhs) {
                        overlap++;
                        break;
                    }
                }
            }
            expect_true(overlap >= 0 && overlap <= kBufferCapacity, "p30/q4 transition overlap out of range");
            overlap_hist[static_cast<std::size_t>(overlap)]++;
            transition_overlap += overlap;
        }
        previous = std::move(partitions);
    }

    int64_t rep10 = 0;
    for (auto count : rep_counts) {
        if (count == 10) {
            rep10++;
        }
    }
    const int64_t transition_admits =
        static_cast<int64_t>(buffer_states.size() - 1) * kBufferCapacity - transition_overlap;

    expect_true(rep10 == 30, "p30/q4 static schedule should have rep_hist={10: 30}");
    expect_true(overlap_hist[1] == 60, "p30/q4 static schedule should have 60 one-overlap transitions");
    expect_true(overlap_hist[2] == 14, "p30/q4 static schedule should have 14 two-overlap transitions");
    expect_true(transition_admits == 208, "p30/q4 static schedule should have transition_admits=208");
}

void test_bounded_q4_static_p30_schedule_stats() {
    constexpr int kNumPartitions = 30;
    constexpr int kBufferCapacity = 4;

    ScopedEnvVar order_file_env("GEGE_BOUNDED_STATE_ORDER_FILE", nullptr);
    ScopedEnvVar optimal88_env("GEGE_BOUNDED_Q4_OPTIMAL88", "0");
    ScopedEnvVar max_admits_env("GEGE_STATEFLOW_MAX_ADMITS", "3");
    ScopedEnvVar minmax_env("GEGE_MINMAX_BUCKET_ASSIGNMENT", "1");

    std::vector<int64_t> edge_bucket_sizes(static_cast<std::size_t>(kNumPartitions * kNumPartitions), 1);
    auto [buffer_states, edge_buckets] =
        getBoundedGreedyCoverEdgeBucketOrdering(kNumPartitions, kBufferCapacity, edge_bucket_sizes);

    expect_true(buffer_states.size() == edge_buckets.size(), "p30/q4 state and edge-bucket schedule sizes should match");
    expect_static_p30_q4_schedule_stats(buffer_states);
    expect_exact_diagonal_bucket_coverage(edge_buckets, kNumPartitions);
}

void test_bounded_q4_scheduler_trains_diagonal_buckets_once() {
    constexpr int kNumPartitions = 32;
    constexpr int kBufferCapacity = 4;

    ScopedEnvVar max_admits_env("GEGE_STATEFLOW_MAX_ADMITS", "3");
    ScopedEnvVar minmax_env("GEGE_MINMAX_BUCKET_ASSIGNMENT", "1");
    std::vector<int64_t> edge_bucket_sizes(static_cast<std::size_t>(kNumPartitions * kNumPartitions), 1);
    auto [buffer_states, edge_buckets] =
        getBoundedGreedyCoverEdgeBucketOrdering(kNumPartitions, kBufferCapacity, edge_bucket_sizes);

    expect_true(buffer_states.size() == edge_buckets.size(), "state and edge-bucket schedule sizes should match");
    expect_exact_diagonal_bucket_coverage(edge_buckets, kNumPartitions);
    expect_tensor_schedule_max_admits(buffer_states, kBufferCapacity, 3);
    expect_tensor_schedule_max_admit_run(buffer_states, kBufferCapacity, 3, 6);
}

void test_bounded_q4_scheduler_is_parameterized() {
    constexpr int kNumPartitions = 20;
    constexpr int kBufferCapacity = 4;

    ScopedEnvVar max_admits_env("GEGE_STATEFLOW_MAX_ADMITS", "3");
    ScopedEnvVar minmax_env("GEGE_MINMAX_BUCKET_ASSIGNMENT", "1");
    std::vector<int64_t> edge_bucket_sizes(static_cast<std::size_t>(kNumPartitions * kNumPartitions), 1);
    auto [buffer_states, edge_buckets] =
        getBoundedGreedyCoverEdgeBucketOrdering(kNumPartitions, kBufferCapacity, edge_bucket_sizes);

    expect_true(buffer_states.size() == edge_buckets.size(), "state and edge-bucket schedule sizes should match");
    expect_true(!buffer_states.empty(), "parameterized bounded scheduler should produce states");
    expect_exact_diagonal_bucket_coverage(edge_buckets, kNumPartitions);
    expect_tensor_schedule_max_admits(buffer_states, kBufferCapacity, 3);
}

StateflowPlan build_bounded_q4_multi_gpu_plan(int active_devices) {
    constexpr int kNumPartitions = 32;
    constexpr int kBufferCapacity = 4;
    std::vector<int64_t> edge_bucket_sizes(static_cast<std::size_t>(kNumPartitions * kNumPartitions), 1);
    std::vector<int64_t> partition_row_counts(static_cast<std::size_t>(kNumPartitions), 1024);
    auto [buffer_states, edge_buckets] =
        getBoundedGreedyCoverMultiGpuEdgeBucketOrdering(kNumPartitions, kBufferCapacity, active_devices, edge_bucket_sizes);

    ScopedEnvVar max_admits_env("GEGE_STATEFLOW_MAX_ADMITS", "3");
    auto candidates =
        enumerateMultiGpuStateflowPlans(buffer_states, edge_buckets, active_devices, edge_bucket_sizes, partition_row_counts, {});
    for (const auto &candidate : candidates) {
        if (candidate.family_variant == PlanVariant::MULTI_GPU_DISJOINT_ROUNDS ||
            candidate.family_variant == PlanVariant::MULTI_GPU_LANE_MATCHED) {
            return candidate;
        }
    }

    std::string variants;
    for (const auto &candidate : candidates) {
        if (!variants.empty()) {
            variants += ",";
        }
        variants += planVariantName(candidate.family_variant);
    }
    throw std::runtime_error("failed to build bounded q4 multi-GPU lane candidate; candidate_count=" +
                             std::to_string(candidates.size()) + " variants=[" + variants + "]");
}

void test_bounded_q4_lane_matched_respects_three_admit_cap() {
    auto two_gpu = build_bounded_q4_multi_gpu_plan(2);
    expect_true(two_gpu.total_microstates >= 88 && two_gpu.total_microstates <= 120,
                "expected p32/q4 bounded 2-GPU state count near the covering lower bound");
    expect_true(two_gpu.total_bucket_assignments == 1024, "expected exact directed bucket count for 2-GPU plan");
    expect_true(validateStateflowPlanExactSemantics(two_gpu), "2-GPU bounded q4 lane-matched plan should validate");
    expect_lane_max_admits(two_gpu, 3);

    auto three_gpu = build_bounded_q4_multi_gpu_plan(3);
    expect_true(three_gpu.total_microstates >= 88 && three_gpu.total_microstates <= 120,
                "expected p32/q4 bounded 3-GPU state count near the covering lower bound");
    expect_true(three_gpu.total_bucket_assignments == 1024, "expected exact directed bucket count for 3-GPU plan");
    expect_true(validateStateflowPlanExactSemantics(three_gpu), "3-GPU bounded q4 lane-matched plan should validate");
    expect_lane_max_admits(three_gpu, 3);

    auto four_gpu = build_bounded_q4_multi_gpu_plan(4);
    expect_true(four_gpu.total_microstates >= 88 && four_gpu.total_microstates <= 120,
                "expected p32/q4 bounded 4-GPU state count near the covering lower bound");
    expect_true(four_gpu.total_bucket_assignments == 1024, "expected exact directed bucket count for 4-GPU plan");
    expect_true(validateStateflowPlanExactSemantics(four_gpu), "4-GPU bounded q4 lane-matched plan should validate");
    expect_lane_max_admits(four_gpu, 3);
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"reject_cross_lane_handoff_in_lane_plan", test_reject_cross_lane_handoff_in_lane_plan},
        {"reject_peer_relay_without_cross_lane_descriptor", test_reject_peer_relay_without_cross_lane_descriptor},
        {"reject_bucket_coverage_gap", test_reject_bucket_coverage_gap},
        {"four_gpu_lane_matched_candidate_validates", test_four_gpu_lane_matched_candidate_validates},
        {"peer_aware_lane_matching_reduces_lane_matched_cost", test_peer_aware_lane_matching_reduces_lane_matched_cost},
        {"bounded_q4_static_p30_schedule_stats", test_bounded_q4_static_p30_schedule_stats},
        {"bounded_q4_scheduler_trains_diagonal_buckets_once", test_bounded_q4_scheduler_trains_diagonal_buckets_once},
        {"bounded_q4_scheduler_is_parameterized", test_bounded_q4_scheduler_is_parameterized},
        {"bounded_q4_lane_matched_respects_three_admit_cap", test_bounded_q4_lane_matched_respects_three_admit_cap},
    };

    for (const auto &[name, test] : tests) {
        test();
        std::cout << "[PASS] " << name << "\n";
    }

    return 0;
}
