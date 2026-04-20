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

struct MicrostatePlan {
    int64_t microstate_id = -1;
    int64_t superstate_id = -1;
    std::vector<int> resident_partitions;
    std::vector<std::pair<int, int>> edge_buckets;
    int64_t overlap_with_prev = 0;
    int64_t admitted_partitions = 0;
};

struct LanePlan {
    int64_t lane_id = 0;
    std::vector<MicrostatePlan> microstates;
};

struct StateflowPlan {
    PlanFamily family = PlanFamily::UNKNOWN;
    int64_t gpu_count = 1;
    int64_t buffer_capacity = 0;
    int64_t num_partitions = 0;
    int64_t total_microstates = 0;
    int64_t total_superstates = 0;
    int64_t total_bucket_assignments = 0;
    int64_t total_partition_loads = 0;
    int64_t max_overlap = 0;
    int64_t boundary_count = 0;
    int64_t estimated_bucket_edges = 0;
    double estimated_cost = 0.0;
    std::vector<LanePlan> lanes;
};

std::string planFamilyName(PlanFamily family);

StateflowPlan compileCustomStateflowPlan(int num_partitions, int buffer_capacity, bool randomly_assign_edge_buckets);

StateflowPlan compileHybridCoverStateflowPlan(int num_partitions, int buffer_capacity);

StateflowPlan compileSingleGpuStateflowPlan(int num_partitions,
                                            int buffer_capacity,
                                            bool randomly_assign_edge_buckets,
                                            const std::vector<int64_t> &edge_bucket_sizes,
                                            bool allow_hybrid_cover);

StateflowPlan compileMultiGpuStateflowPlan(const vector<torch::Tensor> &buffer_states,
                                           const vector<torch::Tensor> &edge_buckets_per_buffer,
                                           int active_devices,
                                           const std::vector<int64_t> &edge_bucket_sizes);

std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> stateflowPlanToTensorOrdering(const StateflowPlan &plan);

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
                                                                  int active_devices);
