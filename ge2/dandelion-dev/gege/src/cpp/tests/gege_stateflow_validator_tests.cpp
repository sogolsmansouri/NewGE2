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

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"reject_cross_lane_handoff_in_lane_plan", test_reject_cross_lane_handoff_in_lane_plan},
        {"reject_peer_relay_without_cross_lane_descriptor", test_reject_peer_relay_without_cross_lane_descriptor},
        {"reject_bucket_coverage_gap", test_reject_bucket_coverage_gap},
        {"four_gpu_lane_matched_candidate_validates", test_four_gpu_lane_matched_candidate_validates},
        {"peer_aware_lane_matching_reduces_lane_matched_cost", test_peer_aware_lane_matching_reduces_lane_matched_cost},
    };

    for (const auto &[name, test] : tests) {
        test();
        std::cout << "[PASS] " << name << "\n";
    }

    return 0;
}
