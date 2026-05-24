#pragma once

#include "batch.h"
#include <string>
#include <tuple>
#include <vector>

using std::pair;

enum class PlanFamily {
    UNKNOWN = 0,
    CUSTOM = 1,
    HYBRID_COVER = 2,
};

enum class PlanVariant {
    DEFAULT = 0,
    CUSTOM_CANONICAL = 1,
    CUSTOM_GPU_AWARE = 2,
    CUSTOM_REVERSED = 3,
    CUSTOM_LEGACY_RANDOM = 4,
    HYBRID_COVER_LEGACY_ROTATED = 5,
    HYBRID_COVER_NATURAL = 6,
    HYBRID_COVER_REVERSED = 7,
    MULTI_GPU_DISJOINT_ROUNDS = 8,
    MULTI_GPU_LANE_MATCHED = 9,
    MULTI_GPU_OPTIMAL88_LANE_MATCHED = 10,
};

enum class ResidentObjectRole {
    ANCHOR = 0,
    STREAM = 1,
    SURVIVOR = 2,
    INCOMING = 3,
};

enum class FragmentKind {
    FULLY_RESIDENT = 0,
    ANCHOR_ANCHOR = 1,
    ANCHOR_STREAM = 2,
};

enum class HandoffMode {
    FULL_RELOAD = 0,
    ROTATING_OVERWRITE = 1,
    PEER_RELAY = 2,
    DELAYED_KEEP_ALIVE = 3,
};

struct ResidentObjectPlan {
    int64_t object_id = -1;
    int partition_id = -1;
    int slot_id = -1;
    ResidentObjectRole role = ResidentObjectRole::INCOMING;
    int64_t rows = 0;
};

struct FragmentPlan {
    int64_t fragment_id = -1;
    std::vector<int64_t> required_object_ids;
    std::vector<std::pair<int, int>> edge_buckets;
    int64_t estimated_edges = 0;
    FragmentKind fragment_kind = FragmentKind::FULLY_RESIDENT;
    bool exact_semantics_tag = true;
};

struct HandoffPlan {
    int64_t handoff_id = -1;
    int64_t src_microstate_id = -1;
    int64_t dst_microstate_id = -1;
    int64_t src_lane_id = -1;
    int64_t dst_lane_id = -1;
    std::vector<int64_t> kept_object_ids;
    std::vector<int64_t> admitted_object_ids;
    std::vector<int64_t> evicted_object_ids;
    std::vector<std::pair<int, int>> slot_mapping;
    HandoffMode mode = HandoffMode::FULL_RELOAD;
    int64_t estimated_cost = 0;
    int64_t peer_bytes = 0;
};

struct PlanCostBreakdown {
    double admitted_partition_cost = 0.0;
    double bucket_edge_cost = 0.0;
    double boundary_cost = 0.0;
    double lane_imbalance_cost = 0.0;
    double selection_cost = 0.0;
    int64_t lane_imbalance = 0;
    int64_t planned_peer_bytes = 0;
    double weighted_admission_load = 0.0;
};

struct LaneMatchCostConfig {
    int64_t peer_bandwidth_bps = 32000000000LL;
    int64_t host_bandwidth_bps = 16000000000LL;
    int64_t max_admits_per_transition = -1;
    double imbalance_weight = 1.0;
    double boundary_weight = 1.0;
    bool allow_peer_relay = false;
};

struct PlanEmbeddingLayout {
    int64_t embedding_dim = 0;
    int64_t dtype_size = 0;
    int64_t optimizer_state_multiplier = 1;
};

struct MicrostatePlan {
    int64_t microstate_id = -1;
    int64_t lane_id = 0;
    int64_t superstate_id = -1;
    std::vector<int> resident_partitions;
    std::vector<std::pair<int, int>> edge_buckets;
    std::vector<ResidentObjectPlan> resident_objects;
    std::vector<FragmentPlan> active_fragments;
    std::vector<int64_t> admitted_object_ids;
    std::vector<int64_t> evicted_object_ids;
    int64_t overlap_with_prev = 0;
    int64_t admitted_partitions = 0;
};

struct LanePlan {
    int64_t lane_id = 0;
    std::vector<MicrostatePlan> microstates;
    std::vector<HandoffPlan> handoffs;
};

struct StateflowPlan {
    PlanFamily family = PlanFamily::UNKNOWN;
    PlanVariant family_variant = PlanVariant::DEFAULT;
    int64_t gpu_count = 1;
    int64_t buffer_capacity = 0;
    int64_t num_partitions = 0;
    int64_t total_microstates = 0;
    int64_t total_superstates = 0;
    int64_t total_handoffs = 0;
    int64_t total_cross_lane_handoffs = 0;
    int64_t total_admitted_objects = 0;
    int64_t total_peer_handoff_bytes = 0;
    int64_t total_admissions_by_role[4] = {0, 0, 0, 0};
    int64_t total_bucket_assignments = 0;
    int64_t total_partition_loads = 0;
    int64_t max_overlap = 0;
    int64_t boundary_count = 0;
    int64_t estimated_bucket_edges = 0;
    double estimated_cost = 0.0;
    PlanCostBreakdown cost_breakdown;
    std::vector<LanePlan> lanes;
    std::vector<HandoffPlan> cross_lane_handoffs;
};

struct PeerHandoffDescriptor {
    int64_t src_lane_id = -1;
    int64_t dst_lane_id = -1;
    int64_t src_microstate_id = -1;
    int64_t dst_microstate_id = -1;
    int64_t round_idx = -1;
    int partition_id = -1;
    int src_slot_id = -1;
    int dst_slot_id = -1;
    int64_t bytes = 0;
};

struct MultiGpuSchedule {
    std::vector<std::vector<torch::Tensor>> buffer_states_per_device;
    std::vector<std::vector<torch::Tensor>> edge_buckets_per_device;
    std::vector<PeerHandoffDescriptor> peer_handoffs;
};

std::string planFamilyName(PlanFamily family);
std::string planVariantName(PlanVariant variant);
std::string residentObjectRoleName(ResidentObjectRole role);
std::string fragmentKindName(FragmentKind kind);
std::string handoffModeName(HandoffMode mode);

std::vector<int64_t> computePartitionRowCounts(int64_t total_rows, int num_partitions);

StateflowPlan compileCustomStateflowPlan(int num_partitions,
                                         int buffer_capacity,
                                         bool randomly_assign_edge_buckets,
                                         const std::vector<int64_t> &edge_bucket_sizes = {},
                                         const std::vector<int64_t> &partition_row_counts = {});

StateflowPlan compileHybridCoverStateflowPlan(int num_partitions,
                                              int buffer_capacity,
                                              const std::vector<int64_t> &edge_bucket_sizes = {},
                                              const std::vector<int64_t> &partition_row_counts = {});

StateflowPlan compileSingleGpuStateflowPlan(int num_partitions,
                                            int buffer_capacity,
                                            bool randomly_assign_edge_buckets,
                                            const std::vector<int64_t> &edge_bucket_sizes,
                                            bool allow_hybrid_cover,
                                            const std::vector<int64_t> &partition_row_counts = {});

std::vector<StateflowPlan> enumerateSingleGpuStateflowPlans(int num_partitions,
                                                            int buffer_capacity,
                                                            bool randomly_assign_edge_buckets,
                                                            const std::vector<int64_t> &edge_bucket_sizes,
                                                            bool allow_hybrid_cover,
                                                            const std::vector<int64_t> &partition_row_counts = {});

StateflowPlan compileMultiGpuStateflowPlan(const vector<torch::Tensor> &buffer_states,
                                           const vector<torch::Tensor> &edge_buckets_per_buffer,
                                           int active_devices,
                                           const std::vector<int64_t> &edge_bucket_sizes,
                                           const std::vector<int64_t> &partition_row_counts = {},
                                           const PlanEmbeddingLayout &layout = {});

std::vector<StateflowPlan> enumerateMultiGpuStateflowPlans(const vector<torch::Tensor> &buffer_states,
                                                           const vector<torch::Tensor> &edge_buckets_per_buffer,
                                                           int active_devices,
                                                           const std::vector<int64_t> &edge_bucket_sizes,
                                                           const std::vector<int64_t> &partition_row_counts = {},
                                                           const PlanEmbeddingLayout &layout = {});

MultiGpuSchedule projectStateflowPlanToMultiGpuSchedule(const StateflowPlan &plan);
std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> projectStateflowPlanToLegacySchedule(const StateflowPlan &plan);
std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> stateflowPlanToTensorOrdering(const StateflowPlan &plan);
std::string stateflowPlanToText(const StateflowPlan &plan, bool include_microstates = false);
std::string stateflowPlanToJson(const StateflowPlan &plan, bool include_microstates = false);
bool validateStateflowPlanExactSemantics(const StateflowPlan &plan);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getEdgeBucketOrdering(EdgeBucketOrdering edge_bucket_ordering, int num_partitions, int buffer_capacity,
                                                                               int fine_to_coarse_ratio, int num_cache_partitions,
                                                                               bool randomly_assign_edge_buckets);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> convertEdgeBucketOrderToTensors(vector<vector<int>> buffer_states,
                                                                                         vector<vector<std::pair<int, int>>> edge_buckets_per_buffer);

vector<vector<int>> getBetaOrderingHelper(int num_partitions, int buffer_capacity);

vector<vector<std::pair<int, int>>> greedyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions);

vector<vector<std::pair<int, int>>> randomlyAssignEdgeBucketsToBuffers(vector<vector<int>> buffer_states, int num_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getTwoLevelBetaOrdering(int num_partitions, int buffer_capacity, int fine_to_coarse_ratio,
                                                                                 int num_cache_partitions, bool randomly_assign_edge_buckets);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomEdgeBucketOrdering(int num_partitions = 4, int buffer_capacity = 1, bool randomly_assign_edge_buckets = false);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getGreedyCoverEdgeBucketOrdering(int num_partitions,
                                                                                          int buffer_capacity);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getBoundedGreedyCoverEdgeBucketOrdering(
    int num_partitions,
    int buffer_capacity,
    const vector<int64_t>& edge_bucket_sizes);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getBoundedGreedyCoverMultiGpuEdgeBucketOrdering(
    int num_partitions,
    int buffer_capacity,
    int active_devices,
    const vector<int64_t>& edge_bucket_sizes);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getOptimizedCustomEdgeBucketOrdering(
    int num_partitions,
    int buffer_capacity,
    int active_devices,
    int batch_size,
    const vector<int64_t>& edge_bucket_sizes);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getAccessAwareCustomEdgeBucketOrdering(int num_partitions, int buffer_capacity, int active_devices);

std::vector<int64_t> getSingleGpuGpuAwareCustomPermutation(const vector<torch::Tensor>& buffer_states,
                                                           const vector<int64_t>& edge_bucket_sizes,
                                                           int num_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getNodePartitionOrdering(NodePartitionOrdering node_partition_ordering, Indices train_nodes,
                                                                                  int64_t total_num_nodes, int num_partitions, int buffer_capacity,
                                                                                  int fine_to_coarse_ratio, int num_cache_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getDispersedNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                           int buffer_capacity, int fine_to_coarse_ratio,
                                                                                           int num_cache_partitions);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getSequentialNodePartitionOrdering(Indices train_nodes, int64_t total_num_nodes, int num_partitions,
                                                                                            int buffer_capacity);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> getCustomNodePartitionOrdering();

std::vector<int64_t> getDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states, int active_devices);

std::vector<int64_t> getAccessAwareDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states,
                                                                  const vector<torch::Tensor>& edge_buckets_per_buffer,
                                                                  int active_devices,
                                                                  const std::vector<int64_t> &partition_row_counts = {},
                                                                  const PlanEmbeddingLayout &layout = {});
