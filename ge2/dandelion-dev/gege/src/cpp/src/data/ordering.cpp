#include "common/datatypes.h"
#include "data/ordering.h"
#include "reporting/logger.h"

#include <algorithm>
#include <numeric>

#ifdef GEGE_OMP
#include "omp.h"
#endif

namespace {

std::vector<int64_t> tensor_to_partitions(torch::Tensor tensor) {
    tensor = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    auto *data = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + tensor.numel());
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

}  // namespace

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
    auto uniques = torch::_unique2(combined, true, false, true);
    auto vals = std::get<0>(uniques);
    auto counts = std::get<2>(uniques);
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
    torch::Tensor train_nodes_partition = train_nodes.divide(partition_size, "trunc");

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
    torch::Tensor train_nodes_partition = train_nodes.divide(partition_size, "trunc");

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

std::vector<int64_t> getDisjointBufferStatePermutation(const vector<torch::Tensor>& buffer_states, int active_devices) {
    if (active_devices <= 1 || buffer_states.size() <= 1) {
        std::vector<int64_t> identity(buffer_states.size());
        std::iota(identity.begin(), identity.end(), 0);
        return identity;
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
    std::vector<std::vector<int64_t>> groups;
    if (!search_disjoint_groups(compatible, remaining, active_devices, groups)) {
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
