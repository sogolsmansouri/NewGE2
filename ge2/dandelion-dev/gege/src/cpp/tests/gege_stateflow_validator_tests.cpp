#include "data/ordering.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

StateflowPlan build_valid_multi_gpu_plan(PlanVariant variant) {
    constexpr int kNumPartitions = 16;
    constexpr int kBufferCapacity = 4;
    constexpr int kActiveDevices = 2;

    auto [buffer_states, edge_buckets] = getCustomEdgeBucketOrdering(kNumPartitions, kBufferCapacity, false);
    std::vector<int64_t> edge_bucket_sizes(static_cast<std::size_t>(kNumPartitions * kNumPartitions), 1);
    std::vector<int64_t> partition_row_counts(static_cast<std::size_t>(kNumPartitions), 1024);
    auto candidates =
        enumerateMultiGpuStateflowPlans(buffer_states, edge_buckets, kActiveDevices, edge_bucket_sizes, partition_row_counts, {});

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

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"reject_cross_lane_handoff_in_lane_plan", test_reject_cross_lane_handoff_in_lane_plan},
        {"reject_peer_relay_without_cross_lane_descriptor", test_reject_peer_relay_without_cross_lane_descriptor},
        {"reject_bucket_coverage_gap", test_reject_bucket_coverage_gap},
    };

    for (const auto &[name, test] : tests) {
        test();
        std::cout << "[PASS] " << name << "\n";
    }

    return 0;
}
