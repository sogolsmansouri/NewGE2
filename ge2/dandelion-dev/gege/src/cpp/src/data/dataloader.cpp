#include "data/dataloader.h"

#include "common/util.h"
#include "data/ordering.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#ifdef GEGE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace {

bool parse_env_flag(const char *name, bool default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    std::string value(raw);
    if (value == "0" || value == "false" || value == "False" || value == "FALSE") {
        return false;
    }
    if (value == "1" || value == "true" || value == "True" || value == "TRUE") {
        return true;
    }
    return default_value;
}

int64_t parse_env_int(const char *name, int64_t default_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }

    try {
        return std::stoll(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

bool stage_debug_enabled() {
    static bool enabled = parse_env_flag("GEGE_STAGE_DEBUG", false);
    return enabled;
}

bool fast_map_tensors_enabled() {
    static bool enabled = parse_env_flag("GEGE_FAST_MAP_TENSORS", true);
    return enabled;
}

bool verify_node_mapping_enabled() {
    static bool enabled = parse_env_flag("GEGE_VERIFY_NODE_MAPPING", false);
    return enabled;
}

bool partition_buffer_lp_fast_path_env_enabled(bool default_value) {
    return parse_env_flag("GEGE_PARTITION_BUFFER_LP_FAST_PATH", default_value);
}

bool partition_buffer_pipeline_timing_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_PIPELINE_TIMING", false);
    return enabled;
}

bool partition_buffer_block_shuffle_enabled() {
    static bool enabled = parse_env_flag("GEGE_PARTITION_BUFFER_BLOCK_SHUFFLE", false);
    return enabled;
}

bool partition_buffer_bucket_batch_env_enabled(bool default_value) {
    return parse_env_flag("GEGE_PARTITION_BUFFER_BUCKET_BATCHES", default_value);
}

bool partition_buffer_overlap_env_enabled(bool default_value) {
    return parse_env_flag("GEGE_PARTITION_BUFFER_OVERLAP", default_value);
}

int64_t partition_buffer_block_shuffle_block_size() {
    static int64_t block_size = parse_env_int("GEGE_PARTITION_BUFFER_BLOCK_SIZE", 0);
    return block_size;
}

int64_t partition_buffer_pipeline_timing_max() {
    static int64_t max_timings = std::max<int64_t>(parse_env_int("GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX", 8), 0);
    return max_timings;
}

std::atomic<int64_t> &partition_buffer_pipeline_timing_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_log_partition_buffer_pipeline_timing(int64_t &timing_id) {
    if (!partition_buffer_pipeline_timing_enabled()) {
        return false;
    }

    timing_id = partition_buffer_pipeline_timing_counter().fetch_add(1);
    return timing_id < partition_buffer_pipeline_timing_max();
}

bool negative_sampler_filtered(const shared_ptr<NegativeSampler> &negative_sampler) {
    if (negative_sampler == nullptr) {
        return false;
    }

    if (instance_of<NegativeSampler, NegativeSamplingBase>(negative_sampler)) {
        return std::dynamic_pointer_cast<NegativeSamplingBase>(negative_sampler)->filtered_;
    }

    if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler)) {
        return std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler)->filtered_;
    }

    return true;
}

int64_t stage_debug_max_batches() {
    static int64_t max_batches = parse_env_int("GEGE_STAGE_DEBUG_MAX_BATCHES", 20);
    return std::max<int64_t>(max_batches, 0);
}

std::atomic<int64_t> &stage_debug_counter() {
    static std::atomic<int64_t> counter{0};
    return counter;
}

bool should_run_stage_debug(int64_t &debug_batch_id) {
    if (!stage_debug_enabled()) {
        return false;
    }
    debug_batch_id = stage_debug_counter().fetch_add(1);
    return debug_batch_id < stage_debug_max_batches();
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int64_t elapsed_ns(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

struct BlockDescriptor {
    int64_t src_start;
    int64_t size;
    int64_t dst_start;
};

struct ActiveEdgePlan {
    EdgeList active_edges;
    torch::Tensor block_starts;
    torch::Tensor block_sizes;
    int64_t block_total_size = 0;
    int64_t num_active_buckets = 0;
    double bucket_lookup_ms = 0.0;
    double gather_ms = 0.0;
    double shuffle_ms = 0.0;
    double total_ms = 0.0;
    std::string shuffle_mode = "global_randperm";
    bool use_bucket_batches = false;
};

ActiveEdgePlan build_active_edge_plan(GraphModelStorage *graph_storage, const shared_ptr<InMemorySubgraphState> &subgraph_state,
                                      torch::Tensor edge_bucket_ids, int64_t batch_size, bool train, LearningTask learning_task,
                                      int32_t device_idx, int epochs_processed, int loaded_subgraphs, bool log_timing) {
    ActiveEdgePlan plan;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto phase_start = total_start;

    if (graph_storage->useInMemorySubGraph()) {
        int num_partitions = graph_storage->getNumPartitions();
        plan.num_active_buckets = edge_bucket_ids.size(0);
        edge_bucket_ids = edge_bucket_ids.select(1, 0) * num_partitions + edge_bucket_ids.select(1, 1);
        torch::Tensor in_memory_edge_bucket_idx = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());
        torch::Tensor edge_bucket_sizes = torch::empty({edge_bucket_ids.size(0)}, edge_bucket_ids.options());

        auto edge_bucket_ids_accessor = edge_bucket_ids.accessor<int64_t, 1>();
        auto in_memory_edge_bucket_idx_accessor = in_memory_edge_bucket_idx.accessor<int64_t, 1>();
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto all_edge_bucket_sizes_accessor = subgraph_state->in_memory_edge_bucket_sizes_.accessor<int64_t, 1>();
        auto all_edge_bucket_starts_accessor = subgraph_state->in_memory_edge_bucket_starts_.accessor<int64_t, 1>();

        auto tup = torch::sort(subgraph_state->in_memory_edge_bucket_ids_);
        torch::Tensor sorted_in_memory_ids = std::get<0>(tup);
        torch::Tensor in_memory_id_indices = std::get<1>(tup);
        auto in_memory_id_indices_accessor = in_memory_id_indices.accessor<int64_t, 1>();

#pragma omp parallel for
        for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
            int64_t edge_bucket_id = edge_bucket_ids_accessor[i];
            int64_t idx = torch::searchsorted(sorted_in_memory_ids, edge_bucket_id).item<int64_t>();
            idx = in_memory_id_indices_accessor[idx];
            in_memory_edge_bucket_idx_accessor[i] = idx;
            edge_bucket_sizes_accessor[i] = all_edge_bucket_sizes_accessor[idx];
        }

        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            plan.bucket_lookup_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }

        torch::Tensor local_offsets = edge_bucket_sizes.cumsum(0);
        int64_t total_size = local_offsets[-1].item<int64_t>();
        local_offsets = local_offsets - edge_bucket_sizes;
        auto local_offsets_accessor = local_offsets.accessor<int64_t, 1>();

        bool use_block_shuffle = partition_buffer_block_shuffle_enabled() && total_size > 0;
        bool default_bucket_batches = train && learning_task == LearningTask::LINK_PREDICTION && graph_storage->partitionBufferLPFastPathEnabled();
        plan.use_bucket_batches = partition_buffer_bucket_batch_env_enabled(default_bucket_batches) && total_size > 0;
        int64_t configured_block_size = partition_buffer_block_shuffle_block_size();
        int64_t block_size = configured_block_size > 0 ? configured_block_size : std::max<int64_t>(batch_size, 1024);

        std::vector<BlockDescriptor> blocks;
        if (use_block_shuffle || plan.use_bucket_batches) {
            std::vector<int64_t> bucket_order(static_cast<std::size_t>(in_memory_edge_bucket_idx.size(0)));
            std::iota(bucket_order.begin(), bucket_order.end(), 0);

            const uint64_t seed = static_cast<uint64_t>(epochs_processed + 1) * 1315423911ULL ^
                                  static_cast<uint64_t>(device_idx + 1) * 2654435761ULL ^
                                  static_cast<uint64_t>(loaded_subgraphs + 1) * 2246822519ULL ^
                                  static_cast<uint64_t>(plan.num_active_buckets);
            std::mt19937_64 gen(seed);
            std::shuffle(bucket_order.begin(), bucket_order.end(), gen);

            blocks.reserve(static_cast<std::size_t>(std::max<int64_t>(plan.num_active_buckets, 1)));
            for (int64_t order_idx : bucket_order) {
                int64_t idx = in_memory_edge_bucket_idx_accessor[order_idx];
                int64_t edge_bucket_size = edge_bucket_sizes_accessor[order_idx];
                int64_t edge_bucket_start = all_edge_bucket_starts_accessor[idx];
                for (int64_t offset = 0; offset < edge_bucket_size; offset += block_size) {
                    blocks.push_back(BlockDescriptor{edge_bucket_start + offset, std::min<int64_t>(block_size, edge_bucket_size - offset), 0});
                }
            }

            if (use_block_shuffle) {
                std::shuffle(blocks.begin(), blocks.end(), gen);
                plan.shuffle_mode = fmt::format("bucket_block(block_size={})", block_size);
            } else {
                plan.shuffle_mode = fmt::format("bucket_batch(block_size={})", block_size);
            }

            int64_t running_dst = 0;
            for (auto &block : blocks) {
                block.dst_start = running_dst;
                running_dst += block.size;
            }
        }

        if (plan.use_bucket_batches) {
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            plan.block_starts = torch::empty({static_cast<int64_t>(blocks.size())}, opts);
            plan.block_sizes = torch::empty({static_cast<int64_t>(blocks.size())}, opts);
            auto starts_accessor = plan.block_starts.accessor<int64_t, 1>();
            auto sizes_accessor = plan.block_sizes.accessor<int64_t, 1>();
            for (int64_t i = 0; i < static_cast<int64_t>(blocks.size()); i++) {
                starts_accessor[i] = blocks[static_cast<std::size_t>(i)].src_start;
                sizes_accessor[i] = blocks[static_cast<std::size_t>(i)].size;
            }
            plan.block_total_size = total_size;
            plan.active_edges = torch::Tensor();
        } else {
            plan.active_edges = torch::empty({total_size, graph_storage->storage_ptrs_.edges->dim1_size_},
                                             subgraph_state->all_in_memory_mapped_edges_.options());

            if (use_block_shuffle) {
#pragma omp parallel for
                for (int64_t i = 0; i < static_cast<int64_t>(blocks.size()); i++) {
                    auto &block = blocks[static_cast<std::size_t>(i)];
                    plan.active_edges.narrow(0, block.dst_start, block.size) =
                        subgraph_state->all_in_memory_mapped_edges_.narrow(0, block.src_start, block.size);
                }
            } else {
#pragma omp parallel for
                for (int i = 0; i < in_memory_edge_bucket_idx.size(0); i++) {
                    int64_t idx = in_memory_edge_bucket_idx_accessor[i];
                    int64_t edge_bucket_size = edge_bucket_sizes_accessor[i];
                    int64_t edge_bucket_start = all_edge_bucket_starts_accessor[idx];
                    int64_t local_offset = local_offsets_accessor[i];

                    plan.active_edges.narrow(0, local_offset, edge_bucket_size) =
                        subgraph_state->all_in_memory_mapped_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
                }
            }
        }

        if (log_timing) {
            auto now = std::chrono::high_resolution_clock::now();
            plan.gather_ms = elapsed_ms(phase_start, now);
            phase_start = now;
        }
    } else {
        plan.active_edges = graph_storage->storage_ptrs_.edges->range(0, graph_storage->storage_ptrs_.edges->getDim0());
    }

    if (!plan.use_bucket_batches && (!graph_storage->useInMemorySubGraph() || !partition_buffer_block_shuffle_enabled())) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto perm = torch::randperm(plan.active_edges.size(0), opts);
        perm = perm.to(plan.active_edges.device());
        plan.active_edges = plan.active_edges.index_select(0, perm);
    }

    if (log_timing) {
        auto now = std::chrono::high_resolution_clock::now();
        plan.shuffle_ms = elapsed_ms(phase_start, now);
        plan.total_ms = elapsed_ms(total_start, now);
    }

    return plan;
}

std::vector<shared_ptr<Batch>> build_batches_from_blocks(torch::Tensor block_starts, torch::Tensor block_sizes, int64_t batch_size, bool train,
                                                         LearningTask task, int batch_id_offset) {
    std::vector<shared_ptr<Batch>> batches;
    auto sizes_accessor = block_sizes.accessor<int64_t, 1>();
    int64_t block_begin = 0;
    int64_t batch_id = 0;
    int64_t start_idx = 0;

    while (block_begin < block_starts.size(0)) {
        int64_t block_end = block_begin;
        int64_t accumulated_size = 0;
        while (block_end < block_starts.size(0)) {
            int64_t next_size = sizes_accessor[block_end];
            if (accumulated_size > 0 && accumulated_size + next_size > batch_size) {
                break;
            }
            accumulated_size += next_size;
            block_end++;
            if (accumulated_size >= batch_size) {
                break;
            }
        }

        shared_ptr<Batch> curr_batch = std::make_shared<Batch>(train);
        curr_batch->batch_id_ = batch_id + batch_id_offset;
        curr_batch->start_idx_ = start_idx;
        curr_batch->batch_size_ = accumulated_size;
        curr_batch->task_ = task;
        curr_batch->edge_block_starts_ = block_starts.narrow(0, block_begin, block_end - block_begin);
        curr_batch->edge_block_sizes_ = block_sizes.narrow(0, block_begin, block_end - block_begin);

        batches.emplace_back(curr_batch);
        batch_id++;
        start_idx += accumulated_size;
        block_begin = block_end;
    }

    return batches;
}

}  // namespace

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, shared_ptr<TrainingConfig> training_config,
                       shared_ptr<EvaluationConfig> evaluation_config, shared_ptr<EncoderConfig> encoder_config, vector<torch::Device> devices,
                       NegativeSamplingMethod nsm) {
    current_edge_ = 0;
    train_ = true;
    epochs_processed_ = 0;
    batches_processed_ = 0;
    false_negative_edges = 0;
    swap_tasks_completed = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    waiting_for_batches_ = false;

    single_dataset_ = false;
    async_barrier = 0;
    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    training_config_ = training_config;
    evaluation_config_ = evaluation_config;
    only_root_features_ = false;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);

    devices_ = devices;
    activate_devices_ = 0;
    active_edge_block_starts_.resize(devices_.size());
    active_edge_block_sizes_.resize(devices_.size());
    active_edge_block_total_sizes_.assign(devices_.size(), 0);
    device_swap_barrier_wait_ns_.assign(devices_.size(), 0);
    device_swap_update_ns_.assign(devices_.size(), 0);
    device_swap_rebuild_ns_.assign(devices_.size(), 0);
    device_swap_sync_wait_ns_.assign(devices_.size(), 0);
    device_swap_count_.assign(devices_.size(), 0);
    prepared_supersteps_.resize(devices_.size());

    negative_sampling_method_ = nsm;

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (negative_sampling_method_ == NegativeSamplingMethod::RNS) {
            training_negative_sampler_ = std::make_shared<RNS>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->superbatch_negative_plan_batches, training_config_->negative_sampling->local_filter_mode,
                training_config_->negative_sampling->tournament_selection,
                training_config_->negative_sampling->tiled_tournament_scores,
                training_config_->negative_sampling->tiled_tournament_groups_per_tile);
        } else if (negative_sampling_method_ == NegativeSamplingMethod::DNS) {
            training_negative_sampler_ = std::make_shared<DNS>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->superbatch_negative_plan_batches, training_config_->negative_sampling->local_filter_mode,
                training_config_->negative_sampling->tournament_selection,
                training_config_->negative_sampling->tiled_tournament_scores,
                training_config_->negative_sampling->tiled_tournament_groups_per_tile);
        } else if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
            training_negative_sampler_ = std::make_shared<KBGAN>(
                training_config_->negative_sampling->num_chunks, training_config_->negative_sampling->negatives_per_positive,
                training_config_->negative_sampling->degree_fraction, training_config_->negative_sampling->filtered,
                training_config_->negative_sampling->superbatch_negative_plan_batches, training_config_->negative_sampling->local_filter_mode,
                training_config_->negative_sampling->tournament_selection,
                training_config_->negative_sampling->tiled_tournament_scores,
                training_config_->negative_sampling->tiled_tournament_groups_per_tile);
        } else {
            SPDLOG_INFO("NegativeSampling: not supported");
        }
        evaluation_negative_sampler_ = std::make_shared<CorruptNodeNegativeSampler>(
            evaluation_config_->negative_sampling->num_chunks, evaluation_config_->negative_sampling->negatives_per_positive,
            evaluation_config_->negative_sampling->degree_fraction, evaluation_config_->negative_sampling->filtered,
            evaluation_config_->negative_sampling->local_filter_mode);
    } else {
        training_negative_sampler_ = nullptr;
        evaluation_negative_sampler_ = nullptr;
    }

    if (encoder_config != nullptr) {
        if (!encoder_config->train_neighbor_sampling.empty()) {
            training_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->train_neighbor_sampling);

            if (!encoder_config->eval_neighbor_sampling.empty()) {
                evaluation_neighbor_sampler_ = std::make_shared<LayeredNeighborSampler>(graph_storage_, encoder_config->eval_neighbor_sampling);
            } else {
                evaluation_neighbor_sampler_ = training_neighbor_sampler_;
            }

        } else {
            training_neighbor_sampler_ = nullptr;
            evaluation_neighbor_sampler_ = nullptr;
        }
    } else {
        training_neighbor_sampler_ = nullptr;
        evaluation_neighbor_sampler_ = nullptr;
    }

    negative_sampler_ = training_negative_sampler_;
    neighbor_sampler_ = training_neighbor_sampler_;
    refreshGraphStorageMode();
}

DataLoader::DataLoader(shared_ptr<GraphModelStorage> graph_storage, LearningTask learning_task, int batch_size, shared_ptr<NegativeSampler> negative_sampler,
                       shared_ptr<NeighborSampler> neighbor_sampler, bool train) {
    current_edge_ = 0;
    train_ = train;
    epochs_processed_ = 0;
    false_negative_edges = 0;
    batches_processed_ = 0;
    sampler_lock_ = new std::mutex();
    batch_lock_ = new std::mutex;
    batch_cv_ = new std::condition_variable;
    async_barrier = 0;
    waiting_for_batches_ = false;

    batch_size_ = batch_size;
    single_dataset_ = true;

    graph_storage_ = graph_storage;
    learning_task_ = learning_task;
    only_root_features_ = false;

    edge_sampler_ = std::make_shared<RandomEdgeSampler>(graph_storage_);
    devices_ = {graph_storage_->storage_ptrs_.edges->device_};
    activate_devices_ = 0;
    active_edge_block_starts_.resize(devices_.size());
    active_edge_block_sizes_.resize(devices_.size());
    active_edge_block_total_sizes_.assign(devices_.size(), 0);
    device_swap_barrier_wait_ns_.assign(devices_.size(), 0);
    device_swap_update_ns_.assign(devices_.size(), 0);
    device_swap_rebuild_ns_.assign(devices_.size(), 0);
    device_swap_sync_wait_ns_.assign(devices_.size(), 0);
    device_swap_count_.assign(devices_.size(), 0);
    prepared_supersteps_.resize(devices_.size());
    negative_sampler_ = negative_sampler;
    neighbor_sampler_ = neighbor_sampler;

    training_negative_sampler_ = nullptr;
    evaluation_negative_sampler_ = nullptr;

    training_neighbor_sampler_ = nullptr;
    evaluation_neighbor_sampler_ = nullptr;

    refreshGraphStorageMode();

    loadStorage();
}

DataLoader::~DataLoader() {
    delete sampler_lock_;
    delete batch_lock_;
    delete batch_cv_;
}

void DataLoader::refreshGraphStorageMode() {
    bool enable_fast_path = false;
    if (graph_storage_ != nullptr && learning_task_ == LearningTask::LINK_PREDICTION && train_ && graph_storage_->useInMemorySubGraph() &&
        neighbor_sampler_ == nullptr && !negative_sampler_filtered(negative_sampler_)) {
        enable_fast_path = partition_buffer_lp_fast_path_env_enabled(true);
    }
    graph_storage_->setPartitionBufferLPFastPathEnabled(enable_fast_path);

    if (enable_fast_path) {
        SPDLOG_INFO("Using partition-buffer LP fast path (arithmetic remap, no in-memory graph build)");
        if (partition_buffer_bucket_batch_env_enabled(true)) {
            SPDLOG_INFO("Using partition-buffer bucket-batch execution (descriptor batching, no full active edge tensor)");
        }
        if (partition_buffer_overlap_env_enabled(devices_.size() > 1)) {
            SPDLOG_INFO("Using partition-buffer next-superstep overlap (prepare next state and batches before swap)");
        }
    }
}

void DataLoader::advanceEdgeBucketIterator_(int32_t device_idx) {
    if (device_idx < 0 || device_idx >= edge_buckets_per_buffer_iterators_.size()) {
        return;
    }

    if (edge_buckets_per_buffer_iterators_[device_idx] != edge_buckets_per_buffer_.end()) {
        edge_buckets_per_buffer_iterators_[device_idx]++;
    }
}

void DataLoader::maybePrepareNextSuperstep(int32_t device_idx) {
    bool default_overlap = train_ && learning_task_ == LearningTask::LINK_PREDICTION && devices_.size() > 1 &&
                           graph_storage_ != nullptr && graph_storage_->partitionBufferLPFastPathEnabled();
    if (!partition_buffer_overlap_env_enabled(default_overlap)) {
        return;
    }

    if (!partition_buffer_bucket_batch_env_enabled(true)) {
        return;
    }

    if (!graph_storage_->useInMemorySubGraph() || !graph_storage_->hasSwap(device_idx)) {
        return;
    }

    if (device_idx < 0 || device_idx >= prepared_supersteps_.size() ||
        device_idx >= edge_buckets_per_buffer_iterators_.size() ||
        edge_buckets_per_buffer_iterators_[device_idx] == edge_buckets_per_buffer_.end()) {
        return;
    }

    if (prepared_supersteps_[device_idx].valid()) {
        return;
    }

    torch::Tensor next_edge_bucket_ids = (*edge_buckets_per_buffer_iterators_[device_idx]).clone();
    auto swap_ids = graph_storage_->getNextSwapIds(device_idx);
    auto graph_storage = graph_storage_;
    int64_t batch_size = batch_size_;
    bool train = train_;
    LearningTask task = learning_task_;
    int batch_id_offset = batch_id_offset_;
    int epochs_processed = epochs_processed_;
    int loaded_subgraphs_snapshot = loaded_subgraphs;

    prepared_supersteps_[device_idx] = std::async(std::launch::async,
                                                  [graph_storage, next_edge_bucket_ids, swap_ids, batch_size, train, task, batch_id_offset,
                                                   epochs_processed, loaded_subgraphs_snapshot, device_idx]() mutable {
        PreparedSuperstep prepared;
        prepared.subgraph_state = graph_storage->prepareNextInMemorySubGraph(swap_ids, device_idx);
        ActiveEdgePlan plan = build_active_edge_plan(graph_storage.get(), prepared.subgraph_state, next_edge_bucket_ids, batch_size, train, task,
                                                     device_idx, epochs_processed, loaded_subgraphs_snapshot, false);
        if (plan.use_bucket_batches && plan.block_starts.defined() && plan.block_starts.numel() > 0) {
            prepared.batches = build_batches_from_blocks(plan.block_starts, plan.block_sizes, batch_size, train, task, batch_id_offset);
        }
        return prepared;
    });
}

void DataLoader::resetPerfStats() {
    swap_barrier_wait_ns_.store(0);
    swap_update_ns_.store(0);
    swap_rebuild_ns_.store(0);
    swap_sync_wait_ns_.store(0);
    swap_count_.store(0);
    std::fill(device_swap_barrier_wait_ns_.begin(), device_swap_barrier_wait_ns_.end(), 0);
    std::fill(device_swap_update_ns_.begin(), device_swap_update_ns_.end(), 0);
    std::fill(device_swap_rebuild_ns_.begin(), device_swap_rebuild_ns_.end(), 0);
    std::fill(device_swap_sync_wait_ns_.begin(), device_swap_sync_wait_ns_.end(), 0);
    std::fill(device_swap_count_.begin(), device_swap_count_.end(), 0);
}

DataLoaderPerfStats DataLoader::getPerfStats() const {
    DataLoaderPerfStats stats;
    stats.swap_barrier_wait_ns = swap_barrier_wait_ns_.load();
    stats.swap_update_ns = swap_update_ns_.load();
    stats.swap_rebuild_ns = swap_rebuild_ns_.load();
    stats.swap_sync_wait_ns = swap_sync_wait_ns_.load();
    stats.swap_count = swap_count_.load();
    stats.device_swap_barrier_wait_ns = device_swap_barrier_wait_ns_;
    stats.device_swap_update_ns = device_swap_update_ns_;
    stats.device_swap_rebuild_ns = device_swap_rebuild_ns_;
    stats.device_swap_sync_wait_ns = device_swap_sync_wait_ns_;
    stats.device_swap_count = device_swap_count_;
    return stats;
}

void DataLoader::nextEpoch() {
    batch_id_offset_ = 0;
    total_batches_processed_ = 0;
    epochs_processed_++;
    if (negative_sampler_ != nullptr) {
        negative_sampler_->resetPlanCache();
    }
    for (auto &prepared_superstep : prepared_supersteps_) {
        if (prepared_superstep.valid()) {
            prepared_superstep.wait();
        }
    }
    for (int device_idx = 0; device_idx < active_edge_block_starts_.size(); device_idx++) {
        active_edge_block_starts_[device_idx] = torch::Tensor();
        active_edge_block_sizes_[device_idx] = torch::Tensor();
        active_edge_block_total_sizes_[device_idx] = 0;
    }
    buffer_states_.clear();
    if (graph_storage_->useInMemorySubGraph()) {
        unloadStorage();
    }
}

void DataLoader::setActiveEdges(int32_t device_idx) {
    int64_t timing_id = -1;
    bool log_timing = should_log_partition_buffer_pipeline_timing(timing_id);
    active_edge_block_starts_[device_idx] = torch::Tensor();
    active_edge_block_sizes_[device_idx] = torch::Tensor();
    active_edge_block_total_sizes_[device_idx] = 0;

    ActiveEdgePlan plan;
    if (graph_storage_->useInMemorySubGraph()) {
        if (device_idx < 0 || device_idx >= edge_buckets_per_buffer_iterators_.size()) {
            throw std::runtime_error("setActiveEdges received invalid device index for edge bucket iterator");
        }
        if (edge_buckets_per_buffer_iterators_[device_idx] == edge_buckets_per_buffer_.end()) {
            throw std::runtime_error("setActiveEdges reached end of edge bucket iterator before active state consumption completed");
        }
        torch::Tensor edge_bucket_ids = *edge_buckets_per_buffer_iterators_[device_idx];
        advanceEdgeBucketIterator_(device_idx);
        plan = build_active_edge_plan(graph_storage_.get(), graph_storage_->current_subgraph_states_[device_idx], edge_bucket_ids, batch_size_, train_,
                                      learning_task_, device_idx, epochs_processed_, loaded_subgraphs, log_timing);
    } else {
        plan = build_active_edge_plan(graph_storage_.get(), nullptr, torch::Tensor(), batch_size_, train_, learning_task_, device_idx, epochs_processed_,
                                      loaded_subgraphs, log_timing);
    }

    active_edge_block_starts_[device_idx] = plan.block_starts;
    active_edge_block_sizes_[device_idx] = plan.block_sizes;
    active_edge_block_total_sizes_[device_idx] = plan.block_total_size;
    if (log_timing) {
        SPDLOG_INFO(
            "[partition-buffer-pipeline][setActiveEdges {}] device={} buckets={} edges={} shuffle_mode={} bucket_lookup_ms={:.3f} gather_ms={:.3f} shuffle_ms={:.3f} total_ms={:.3f}",
            timing_id, device_idx, plan.num_active_buckets, plan.use_bucket_batches ? active_edge_block_total_sizes_[device_idx] : plan.active_edges.size(0),
            plan.shuffle_mode, plan.bucket_lookup_ms, plan.gather_ms, plan.shuffle_ms, plan.total_ms);
    }
    graph_storage_->setActiveEdges(plan.active_edges, device_idx);
}

void DataLoader::setActiveNodes() {
    torch::Tensor node_ids;

    if (graph_storage_->useInMemorySubGraph()) {
        node_ids = *node_ids_per_buffer_iterator_++;
    } else {
        node_ids = graph_storage_->storage_ptrs_.nodes->range(0, graph_storage_->storage_ptrs_.nodes->getDim0());
        if (node_ids.sizes().size() == 2) {
            node_ids = node_ids.flatten(0, 1);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(node_ids.device());
    node_ids = (node_ids.index_select(0, torch::randperm(node_ids.size(0), opts)));
    graph_storage_->setActiveNodes(node_ids);
}

void DataLoader::initializeBatches(bool prepare_encode, int32_t device_idx) {
    int64_t batch_id = 0;
    int64_t start_idx = 0;
    bool use_bucket_batches = false;

    int64_t num_items;
    if (prepare_encode) {
        num_items = graph_storage_->getNumNodes();
    } else {
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            setActiveEdges(device_idx);
            use_bucket_batches = active_edge_block_starts_[device_idx].defined() && active_edge_block_starts_[device_idx].numel() > 0;
            num_items = use_bucket_batches ? active_edge_block_total_sizes_[device_idx] : graph_storage_->getNumActiveEdges(device_idx);
        } else {
            setActiveNodes();
            num_items = graph_storage_->getNumActiveNodes();
        }
    }

    int64_t batch_size = batch_size_;
    vector<shared_ptr<Batch>> batches;
    if (use_bucket_batches) {
        torch::Tensor block_starts = active_edge_block_starts_[device_idx];
        torch::Tensor block_sizes = active_edge_block_sizes_[device_idx];
        auto starts_accessor = block_starts.accessor<int64_t, 1>();
        auto sizes_accessor = block_sizes.accessor<int64_t, 1>();

        int64_t block_begin = 0;
        while (block_begin < block_starts.size(0)) {
            int64_t block_end = block_begin;
            int64_t accumulated_size = 0;
            while (block_end < block_starts.size(0)) {
                int64_t next_size = sizes_accessor[block_end];
                if (accumulated_size > 0 && accumulated_size + next_size > batch_size_) {
                    break;
                }
                accumulated_size += next_size;
                block_end++;
                if (accumulated_size >= batch_size_) {
                    break;
                }
            }

            shared_ptr<Batch> curr_batch = std::make_shared<Batch>(train_);
            curr_batch->batch_id_ = batch_id + batch_id_offset_;
            curr_batch->start_idx_ = start_idx;
            curr_batch->batch_size_ = accumulated_size;
            curr_batch->task_ = prepare_encode ? LearningTask::ENCODE : learning_task_;
            curr_batch->edge_block_starts_ = block_starts.narrow(0, block_begin, block_end - block_begin);
            curr_batch->edge_block_sizes_ = block_sizes.narrow(0, block_begin, block_end - block_begin);

            batches.emplace_back(curr_batch);
            batch_id++;
            start_idx += accumulated_size;
            block_begin = block_end;
        }
    } else {
        while (start_idx < num_items) {
            if (num_items - (start_idx + batch_size) < 0) {
                batch_size = num_items - start_idx;
            }
            shared_ptr<Batch> curr_batch = std::make_shared<Batch>(train_);
            curr_batch->batch_id_ = batch_id + batch_id_offset_;
            curr_batch->start_idx_ = start_idx;
            curr_batch->batch_size_ = batch_size;

            if (prepare_encode) {
                curr_batch->task_ = LearningTask::ENCODE;
            } else {
                curr_batch->task_ = learning_task_;
            }

            batches.emplace_back(curr_batch);
            batch_id++;
            start_idx += batch_size;
        }
    }
    all_batches_[device_idx] = batches;
    batches_left_[device_idx] = batches.size();
    batch_iterators_[device_idx] = all_batches_[device_idx].begin();
}

void DataLoader::setBufferOrdering() {
    int physical_devices = std::max<int>(devices_.size(), 1);
    int requested_active_devices = physical_devices;
    if (training_config_ != nullptr && training_config_->logical_active_devices > 0) {
        requested_active_devices = training_config_->logical_active_devices;
    } else {
        const char *logical_devices_env = std::getenv("GEGE_LOGICAL_ACTIVE_DEVICES");
        if (logical_devices_env != nullptr && logical_devices_env[0] != '\0') {
            try {
                requested_active_devices = std::max(0, std::stoi(logical_devices_env));
            } catch (const std::exception &) {
                SPDLOG_WARN("Ignoring invalid GEGE_LOGICAL_ACTIVE_DEVICES={}", logical_devices_env);
            }
        }
    }

    auto reorder_buffer_ordering = [&](std::vector<torch::Tensor> &buffer_states, std::vector<torch::Tensor> &edge_buckets_per_buffer) {
        if (requested_active_devices <= 1 || buffer_states.size() <= 1 || edge_buckets_per_buffer.size() != buffer_states.size()) {
            return;
        }

        if (requested_active_devices != physical_devices) {
            SPDLOG_INFO("Using logical_active_devices={} for buffer ordering on {} physical device(s)", requested_active_devices, physical_devices);
        }

        bool access_aware_scheduler = false;
        const char *access_aware_env = std::getenv("GEGE_ACCESS_AWARE_SCHEDULER");
        if (access_aware_env != nullptr && access_aware_env[0] != '\0' && std::string(access_aware_env) != "0") {
            access_aware_scheduler = true;
        }

        auto permutation = access_aware_scheduler
            ? getAccessAwareDisjointBufferStatePermutation(buffer_states, edge_buckets_per_buffer, requested_active_devices)
            : getDisjointBufferStatePermutation(buffer_states, requested_active_devices);
        bool changed = false;
        for (std::size_t i = 0; i < permutation.size(); i++) {
            if (permutation[i] != static_cast<int64_t>(i)) {
                changed = true;
                break;
            }
        }

        if (!changed) {
            return;
        }

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

        if (access_aware_scheduler) {
            SPDLOG_INFO("Reordered {} buffer states into access-aware disjoint multi-GPU supersteps for {} logical device(s) on {} physical device(s)",
                        buffer_states.size(), requested_active_devices, physical_devices);
        } else {
            SPDLOG_INFO("Reordered {} buffer states into disjoint multi-GPU supersteps for {} logical device(s) on {} physical device(s)",
                        buffer_states.size(), requested_active_devices, physical_devices);
        }
    };

    shared_ptr<PartitionBufferOptions> options;
    if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, MemPartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)) {
        options = std::dynamic_pointer_cast<MemPartitionBufferStorage>(graph_storage_->storage_ptrs_.node_embeddings)->options_;
    } else if (instance_of<Storage, PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)) {
        options = std::dynamic_pointer_cast<PartitionBufferStorage>(graph_storage_->storage_ptrs_.node_features)->options_;
    }

    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        if (graph_storage_->useInMemorySubGraph()) {
            bool access_aware_state_generation = false;
            const char *access_aware_state_generation_env = std::getenv("GEGE_ACCESS_AWARE_STATE_GENERATION");
            if (access_aware_state_generation_env != nullptr && access_aware_state_generation_env[0] != '\0' &&
                std::string(access_aware_state_generation_env) != "0") {
                access_aware_state_generation = true;
            }

            std::tuple<vector<torch::Tensor>, vector<torch::Tensor>> tup;
            if (access_aware_state_generation && options->edge_bucket_ordering == EdgeBucketOrdering::CUSTOM) {
                tup = getAccessAwareCustomEdgeBucketOrdering(options->num_partitions, options->buffer_capacity, requested_active_devices);
                SPDLOG_INFO("Using access-aware state generation for CUSTOM ordering with {} logical device(s)", requested_active_devices);
            } else {
                tup = getEdgeBucketOrdering(options->edge_bucket_ordering, options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio,
                                            options->num_cache_partitions, options->randomly_assign_edge_buckets);
            }
            buffer_states_ = std::get<0>(tup);
            edge_buckets_per_buffer_ = std::get<1>(tup);
            if (!access_aware_state_generation) {
                reorder_buffer_ordering(buffer_states_, edge_buckets_per_buffer_);
            }
            SPDLOG_INFO("buffer_states_ sizes() {}", buffer_states_.size());
            auto edge_buckets_per_buffer_iterator = edge_buckets_per_buffer_.begin();
            edge_buckets_per_buffer_iterators_.resize(devices_.size());
            for (int i = 0; i < devices_.size(); i ++) {
                edge_buckets_per_buffer_iterators_[i] = edge_buckets_per_buffer_iterator;
                edge_buckets_per_buffer_iterator++;
            }
            graph_storage_->setBufferOrdering(buffer_states_);
        }
    } else {
        if (graph_storage_->useInMemorySubGraph()) {
            graph_storage_->storage_ptrs_.train_nodes->load();
            int64_t num_train_nodes = graph_storage_->storage_ptrs_.nodes->getDim0();
            auto tup = getNodePartitionOrdering(
                options->node_partition_ordering, graph_storage_->storage_ptrs_.train_nodes->range(0, num_train_nodes).flatten(0, 1),
                graph_storage_->getNumNodes(), options->num_partitions, options->buffer_capacity, options->fine_to_coarse_ratio, options->num_cache_partitions);
            buffer_states_ = std::get<0>(tup);
            node_ids_per_buffer_ = std::get<1>(tup);

            node_ids_per_buffer_iterator_ = node_ids_per_buffer_.begin();

            graph_storage_->setBufferOrdering(buffer_states_);
        }
    }
}

void DataLoader::clearBatches() { all_batches_ = std::vector<std::vector<shared_ptr<Batch>>>(); }

shared_ptr<Batch> DataLoader::getNextBatch(int32_t device_idx) {
    // std::unique_lock batch_lock(*batch_lock_);
    // // batch_cv_->wait(batch_lock, [this] { return !waiting_for_batches_; });

    shared_ptr<Batch> batch;
    if (batch_iterators_[device_idx] != all_batches_[device_idx].end()) {
        batch = *batch_iterators_[device_idx];
        batch_iterators_[device_idx]++;

        // all_reads_[device_idx] = true;

        // check if all batches have been read
        if (batch_iterators_[device_idx] == all_batches_[device_idx].end()) {
            ++ async_barrier;
            maybePrepareNextSuperstep(device_idx);
            // all_reads_[device_idx] = true;
            if (graph_storage_->useInMemorySubGraph()) {
                if (!graph_storage_->hasSwap(device_idx)) {
                    all_reads_[device_idx] = true;
                }
            } else {
                all_reads_[device_idx] = true;
            }
        }
    } else {
        batch = nullptr;
        if (graph_storage_->useInMemorySubGraph()) {
            if (graph_storage_->hasSwap(device_idx)) {
                // wait for all batches to finish before swapping
                if (swap_tasks_completed == all_batches_.size()) {
                    swap_tasks_completed = 0;
                }
                auto swap_barrier_start = std::chrono::high_resolution_clock::now();
                // batch_cv_->wait(batch_lock, [this, device_idx] { 
                //     return async_barrier.load() % all_batches_.size() == 0; });
                // batch_lock.unlock();
                while(async_barrier.load() % all_batches_.size() != 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                auto swap_barrier_elapsed = elapsed_ns(swap_barrier_start, std::chrono::high_resolution_clock::now());
                swap_barrier_wait_ns_.fetch_add(swap_barrier_elapsed);
                device_swap_barrier_wait_ns_[device_idx] += swap_barrier_elapsed;

#ifdef GEGE_CUDA
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif
                PreparedSuperstep prepared_superstep;
                bool use_prepared_superstep = false;
                auto update_start = std::chrono::high_resolution_clock::now();
                if (prepared_supersteps_[device_idx].valid()) {
                    prepared_superstep = prepared_supersteps_[device_idx].get();
                    use_prepared_superstep = prepared_superstep.subgraph_state != nullptr && !prepared_superstep.batches.empty();
                }

                if (use_prepared_superstep) {
                    graph_storage_->performSwap(device_idx);
                    graph_storage_->current_subgraph_states_[device_idx] = prepared_superstep.subgraph_state;
                    if (device_idx == 0) {
                        graph_storage_->current_subgraph_state_ = prepared_superstep.subgraph_state;
                    }
                    graph_storage_->setActiveEdges(torch::Tensor(), device_idx);
                } else {
                    graph_storage_->updateInMemorySubGraph(device_idx);
                }

                auto swap_update_elapsed = elapsed_ns(update_start, std::chrono::high_resolution_clock::now());
                swap_update_ns_.fetch_add(swap_update_elapsed);
                device_swap_update_ns_[device_idx] += swap_update_elapsed;
#ifdef GEGE_CUDA
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif
                auto rebuild_start = std::chrono::high_resolution_clock::now();
                if (use_prepared_superstep) {
                    all_batches_[device_idx] = std::move(prepared_superstep.batches);
                    batches_left_[device_idx] = all_batches_[device_idx].size();
                    batch_iterators_[device_idx] = all_batches_[device_idx].begin();
                } else {
                    initializeBatches(false, device_idx);
                }
                auto swap_rebuild_elapsed = elapsed_ns(rebuild_start, std::chrono::high_resolution_clock::now());
                swap_rebuild_ns_.fetch_add(swap_rebuild_elapsed);
                device_swap_rebuild_ns_[device_idx] += swap_rebuild_elapsed;
#ifdef GEGE_CUDA
                c10::cuda::CUDACachingAllocator::emptyCache();
#endif

                swap_tasks_completed ++;
                activate_devices_ ++;
                swap_count_.fetch_add(1);
                device_swap_count_[device_idx] += 1;

                auto swap_sync_start = std::chrono::high_resolution_clock::now();
                while(swap_tasks_completed.load() != all_batches_.size()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                auto swap_sync_elapsed = elapsed_ns(swap_sync_start, std::chrono::high_resolution_clock::now());
                swap_sync_wait_ns_.fetch_add(swap_sync_elapsed);
                device_swap_sync_wait_ns_[device_idx] += swap_sync_elapsed;
                // auto t2 = std::chrono::high_resolution_clock::now();
                // SPDLOG_INFO("Finished swapping subgraph for device {}", device_idx);
                // SPDLOG_INFO("Time to swap subgraph for device {}: {} ms", device_idx, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
                // if (device_idx == 0) {
                //     graph_storage_->updateInMemorySubGraph();
                //     for (int i = 0; i < devices_.size(); i ++) {
                //         initializeBatches(false, i);
                //     }
                //     swap_tasks_completed = true;
                // } else {
                //     batch_cv_->wait(batch_lock, [this, device_idx] {
                //         return swap_tasks_completed.load() == true;
                //     });
                // }
                batch = *batch_iterators_[device_idx];
                batch_iterators_[device_idx]++;

                // check if all batches have been read
                if (batch_iterators_[device_idx] == all_batches_[device_idx].end()) {
                    if (graph_storage_->useInMemorySubGraph()) {
                        if (!graph_storage_->hasSwap(device_idx)) {
                            all_reads_[device_idx] = true;
                        }
                    } else {
                        all_reads_[device_idx] = true;
                    }
                }
            } else {
                all_reads_[device_idx] = true;
            }
        } else {
            all_reads_[device_idx] = true;
        }
    }
    return batch;
}

bool DataLoader::hasNextBatch(int32_t device_idx) {
    batch_lock_->lock();
    bool ret = !all_reads_[device_idx];
    batch_lock_->unlock();
    return ret;
}

void DataLoader::finishedBatch(int32_t device_idx) {
    batch_lock_->lock();
    batches_left_[device_idx]--;
    if (batches_left_[device_idx] == 0) {
        activate_devices_ --;
    }
    total_batches_processed_++;
    batch_lock_->unlock();
    batch_cv_->notify_all();
}

shared_ptr<Batch> DataLoader::getBatch(at::optional<torch::Device> device, bool perform_map, int32_t device_idx) {
    shared_ptr<Batch> batch = getNextBatch(device_idx);
    
    if (batch == nullptr) {
        return batch;
    }
 
    if (batch->task_ == LearningTask::LINK_PREDICTION) {
        edgeSample(batch, device_idx);
    } else if (batch->task_ == LearningTask::NODE_CLASSIFICATION || batch->task_ == LearningTask::ENCODE) {
        nodeSample(batch, device_idx);
    }

    loadCPUParameters(batch);

    if (device.has_value()) {
        if (device.value().is_cuda()) {
            batch->to(device.value());
            loadGPUParameters(batch);
            batch->dense_graph_.performMap();
        }
    }

    if (perform_map) {
        batch->dense_graph_.performMap();
    }

    return batch;
}

void DataLoader::edgeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    int64_t debug_batch_id = -1;
    bool run_stage_debug = should_run_stage_debug(debug_batch_id);
    auto edge_sample_start = std::chrono::high_resolution_clock::now();
    auto step_start = edge_sample_start;

    if (!batch->edges_.defined()) {
        batch->edges_ = edge_sampler_->getEdges(batch, device_idx);
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][edgeSample][batch {}][step 1] getEdges ms={:.3f} edges={}x{}",
                    debug_batch_id, elapsed_ms(step_start, now), batch->edges_.size(0), batch->edges_.size(1));
        step_start = now;
    }

    if (negative_sampler_ != nullptr) {
        negativeSample(batch);
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t src_neg_numel = batch->src_neg_indices_.defined() ? batch->src_neg_indices_.numel() : 0;
        int64_t dst_neg_numel = batch->dst_neg_indices_.defined() ? batch->dst_neg_indices_.numel() : 0;
        SPDLOG_INFO("[stage-debug][edgeSample][batch {}][step 2] negativeSample ms={:.3f} src_neg_numel={} dst_neg_numel={}",
                    debug_batch_id, elapsed_ms(step_start, now), src_neg_numel, dst_neg_numel);
        step_start = now;
    }
    // std::cout << batch->src_neg_indices_ << std::endl;
    // std::cout << batch->edges_.sizes() << " " <<  batch->src_neg_indices_.sizes() << " " << batch->dst_neg_indices_.sizes() << std::endl;

    // int32_t false_negative_edges_src = 0;
    // int32_t false_negative_edges_dst = 0;
    // for (int32_t i = 0; i < 20; i++) {
    //     for (int32_t j = 0; j < batch->src_neg_indices_.size(1); j++) {
    //         if (batch->edges_[i][0].item<int64_t>() == batch->src_neg_indices_[i / 500][j].item<int64_t>()) {
    //             false_negative_edges_src ++;
    //             // std::cout << "[ " << i << " " << j << " " 
    //             //           << batch->edges_[i][0].item<int64_t>() << " " << batch->edges_[i][-1].item<int64_t>() << " "
    //             //           << batch->src_neg_indices_[i / 500][j].item<int64_t>() << " " << batch->dst_neg_indices_[i / 500][j].item<int64_t>() << " ]" << std::endl;

    //         }
    //         if (batch->edges_[i][-1].item<int32_t>() == batch->dst_neg_indices_[i / 500][j].item<int32_t>()) {
    //             false_negative_edges_dst++;
    //         }
    //     }
    // }
    // SPDLOG_INFO("false_negative_edges_src: {}", false_negative_edges_src);
    // SPDLOG_INFO("false_negative_edges_dst: {}", false_negative_edges_dst);
    auto map_collect_start = std::chrono::high_resolution_clock::now();
    std::vector<torch::Tensor> all_ids = {batch->edges_.select(1, 0), batch->edges_.select(1, -1)};

    if (batch->src_neg_indices_.defined()) {
        all_ids.emplace_back(batch->src_neg_indices_.flatten(0, 1));
    }

    if (batch->dst_neg_indices_.defined()) {
        all_ids.emplace_back(batch->dst_neg_indices_.flatten(0, 1));
    }
    auto map_collect_end = std::chrono::high_resolution_clock::now();

    torch::Tensor src_mapping;
    torch::Tensor dst_mapping;
    torch::Tensor src_neg_mapping;
    torch::Tensor dst_neg_mapping;

    std::vector<torch::Tensor> mapped_tensors;
    double map_lookup_ms = 0.0;
    double map_verify_ms = 0.0;
    double remap_assign_ms = 0.0;
    MapTensorTiming map_tensor_timing;
    bool has_map_tensor_timing = false;

    if (neighbor_sampler_ != nullptr) {
        auto map_lookup_start = std::chrono::high_resolution_clock::now();
        // get unique nodes in edges and negatives
        batch->root_node_indices_ = std::get<0>(torch::_unique(torch::cat(all_ids)));

        // sample neighbors and get unique nodes
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();

        // map edges and negatives to their corresponding index in unique_node_indices_
        auto tup = torch::sort(batch->unique_node_indices_);
        torch::Tensor sorted_map = std::get<0>(tup);
        torch::Tensor map_to_unsorted = std::get<1>(tup);

        mapped_tensors = apply_tensor_map(sorted_map, all_ids);
        auto map_lookup_end = std::chrono::high_resolution_clock::now();
        map_lookup_ms = elapsed_ms(map_lookup_start, map_lookup_end);

        int64_t num_nbrs_sampled = batch->dense_graph_.hop_offsets_[-2].item<int64_t>();

        auto remap_assign_start = std::chrono::high_resolution_clock::now();
        src_mapping = map_to_unsorted.index_select(0, mapped_tensors[0]) - num_nbrs_sampled;
        dst_mapping = map_to_unsorted.index_select(0, mapped_tensors[1]) - num_nbrs_sampled;

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = map_to_unsorted.index_select(0, mapped_tensors[2]).reshape(batch->src_neg_indices_.sizes()) - num_nbrs_sampled;
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = map_to_unsorted.index_select(0, mapped_tensors[3]).reshape(batch->dst_neg_indices_.sizes()) - num_nbrs_sampled;
        }
        auto remap_assign_end = std::chrono::high_resolution_clock::now();
        remap_assign_ms = elapsed_ms(remap_assign_start, remap_assign_end);
    } else {
        // map edges and negatives to their corresponding index in unique_node_indices_
        bool map_sorted = !fast_map_tensors_enabled();
        auto map_lookup_start = std::chrono::high_resolution_clock::now();
        auto tup = map_tensors(all_ids, map_sorted, run_stage_debug ? &map_tensor_timing : nullptr, graph_storage_->getNumNodesInMemory(device_idx));
        auto map_lookup_end = std::chrono::high_resolution_clock::now();
        map_lookup_ms = elapsed_ms(map_lookup_start, map_lookup_end);
        has_map_tensor_timing = run_stage_debug;
    

        batch->unique_node_indices_ = std::get<0>(tup);

        auto map_verify_start = std::chrono::high_resolution_clock::now();
        if (verify_node_mapping_enabled()) {
            // Optional expensive guardrail: avoid synchronizing .item() on every batch unless requested.
            if (torch::any(batch->unique_node_indices_ < 0).item<bool>()) {
                SPDLOG_ERROR("Node mapping is broken. Try repartition again.");
                throw std::runtime_error("");
            }
        }
        auto map_verify_end = std::chrono::high_resolution_clock::now();
        map_verify_ms = elapsed_ms(map_verify_start, map_verify_end);


        auto remap_assign_start = std::chrono::high_resolution_clock::now();
        mapped_tensors = std::get<1>(tup);

        src_mapping = mapped_tensors[0];
        dst_mapping = mapped_tensors[1];

        if (batch->src_neg_indices_.defined()) {
            src_neg_mapping = mapped_tensors[2].reshape(batch->src_neg_indices_.sizes());
        }

        if (batch->dst_neg_indices_.defined()) {
            dst_neg_mapping = mapped_tensors[3].reshape(batch->dst_neg_indices_.sizes());
        }
        auto remap_assign_end = std::chrono::high_resolution_clock::now();
        remap_assign_ms = elapsed_ms(remap_assign_start, remap_assign_end);
    }
    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t input_ids_numel = 0;
        for (const auto &ids : all_ids) {
            input_ids_numel += ids.numel();
        }
        int64_t unique_numel = batch->unique_node_indices_.defined() ? batch->unique_node_indices_.numel() : 0;
        int64_t duplicate_count = std::max<int64_t>(input_ids_numel - unique_numel, 0);
        double duplicate_ratio = input_ids_numel > 0 ? (double)duplicate_count / (double)input_ids_numel : 0.0;
        SPDLOG_INFO(
            "[stage-debug][edgeSample][batch {}][step 3] map/remap ms={:.3f} unique_nodes={} input_ids={} duplicates={} duplicate_ratio={:.6f} neighbor_sampler={} fast_map={} verify_map={}",
            debug_batch_id, elapsed_ms(step_start, now), unique_numel, input_ids_numel, duplicate_count, duplicate_ratio,
            neighbor_sampler_ != nullptr, fast_map_tensors_enabled(), verify_node_mapping_enabled());
        SPDLOG_INFO(
            "[stage-debug][edgeSample][batch {}][step 3 breakdown] collect_ids_ms={:.3f} map_lookup_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f}",
            debug_batch_id, elapsed_ms(map_collect_start, map_collect_end), map_lookup_ms, map_verify_ms, remap_assign_ms);
        if (has_map_tensor_timing) {
            SPDLOG_INFO(
                "[stage-debug][edgeSample][batch {}][step 3 map_tensors] validate_ms={:.3f} cat_ms={:.3f} unique_ms={:.3f} unique_wall_ms={:.3f} split_ms={:.3f} total_ms={:.3f}",
                debug_batch_id, map_tensor_timing.validate_ms, map_tensor_timing.cat_ms, map_tensor_timing.unique_ms, map_tensor_timing.unique_wall_ms,
                map_tensor_timing.split_ms, map_tensor_timing.total_ms);
            SPDLOG_INFO(
                "[stage-debug][edgeSample][batch {}][step 3 map_tensors backend] requested_backend={} executed_backend={} fallback={} fallback_backend={} fallback_reason={} total_fallbacks={} cuco_compiled={} capture_file={}",
                debug_batch_id, map_tensor_timing.unique_requested_backend, map_tensor_timing.unique_executed_backend,
                map_tensor_timing.unique_used_fallback, map_tensor_timing.unique_fallback_backend, map_tensor_timing.unique_fallback_reason,
                map_tensor_timing.unique_total_fallbacks, map_tensor_timing.unique_cuco_compiled, map_tensor_timing.capture_path);
        }
        step_start = now;
    }

    if (batch->edges_.size(1) == 2) {
        batch->edges_ = torch::stack({src_mapping, dst_mapping}).transpose(0, 1);
    } else if (batch->edges_.size(1) == 3) {
        batch->edges_ = torch::stack({src_mapping, batch->edges_.select(1, 1), dst_mapping}).transpose(0, 1);
    } else {
        throw TensorSizeMismatchException(batch->edges_, "Edge list must be a 3 or 2 column tensor");
    }

    batch->src_neg_indices_mapping_ = src_neg_mapping;
    batch->dst_neg_indices_mapping_ = dst_neg_mapping;

    if (run_stage_debug) {
        auto now = std::chrono::high_resolution_clock::now();
        SPDLOG_INFO("[stage-debug][edgeSample][batch {}][step 4] finalize ms={:.3f} total_ms={:.3f}",
                    debug_batch_id, elapsed_ms(step_start, now), elapsed_ms(edge_sample_start, now));
    }
}

void DataLoader::nodeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    if (batch->task_ == LearningTask::ENCODE) {
        torch::TensorOptions node_opts = torch::TensorOptions().dtype(torch::kInt64).device(graph_storage_->storage_ptrs_.edges->device_);
        batch->root_node_indices_ = torch::arange(batch->start_idx_, batch->start_idx_ + batch->batch_size_, node_opts);
    } else {
        batch->root_node_indices_ = graph_storage_->getNodeIdsRange(batch->start_idx_, batch->batch_size_).to(torch::kInt64);
    }

    if (graph_storage_->storage_ptrs_.node_labels != nullptr) {
        batch->node_labels_ = graph_storage_->getNodeLabels(batch->root_node_indices_).flatten(0, 1);
    }

    if (graph_storage_->current_subgraph_state_->global_to_local_index_map_.defined()) {
        batch->root_node_indices_ = graph_storage_->current_subgraph_state_->global_to_local_index_map_.index_select(0, batch->root_node_indices_);
    }

    if (neighbor_sampler_ != nullptr) {
        batch->dense_graph_ = neighbor_sampler_->getNeighbors(batch->root_node_indices_, graph_storage_->current_subgraph_state_->in_memory_subgraph_);
        batch->unique_node_indices_ = batch->dense_graph_.getNodeIDs();
    } else {
        batch->unique_node_indices_ = batch->root_node_indices_;
    }
}

void DataLoader::negativeSample(shared_ptr<Batch> batch, int32_t device_idx) {
    std::tie(batch->src_neg_indices_, batch->src_neg_filter_) =
        negative_sampler_->getNegatives(graph_storage_->current_subgraph_states_[device_idx]->in_memory_subgraph_, batch->edges_, true);

    std::tie(batch->dst_neg_indices_, batch->dst_neg_filter_) =
        negative_sampler_->getNegatives(graph_storage_->current_subgraph_states_[device_idx]->in_memory_subgraph_, batch->edges_, false);
}

void DataLoader::loadCPUParameters(shared_ptr<Batch> batch) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_);
            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_);
                if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
                    batch->node_embeddings_g_ = graph_storage_->getNodeEmbeddingsG(batch->unique_node_indices_);
                    batch->node_embeddings_state_g_ = graph_storage_->getNodeEmbeddingStateG(batch->unique_node_indices_);
                }
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ != torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }

    batch->load_timestamp_ = timestamp_;
}

void DataLoader::loadGPUParameters(shared_ptr<Batch> batch, int32_t device_idx) {
    if (graph_storage_->storage_ptrs_.node_embeddings != nullptr) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {

            batch->node_embeddings_ = graph_storage_->getNodeEmbeddings(batch->unique_node_indices_, device_idx);

            if (train_) {
                batch->node_embeddings_state_ = graph_storage_->getNodeEmbeddingState(batch->unique_node_indices_, device_idx);
                if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
                    batch->node_embeddings_g_ = graph_storage_->getNodeEmbeddingsG(batch->unique_node_indices_, device_idx);
                    batch->node_embeddings_state_g_ = graph_storage_->getNodeEmbeddingStateG(batch->unique_node_indices_, device_idx);
                }
            }
        }
    }

    if (graph_storage_->storage_ptrs_.node_features != nullptr) {
        if (graph_storage_->storage_ptrs_.node_features->device_ == torch::kCUDA) {
            if (only_root_features_) {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->root_node_indices_);
            } else {
                batch->node_features_ = graph_storage_->getNodeFeatures(batch->unique_node_indices_);
            }
        }
    }
}

void DataLoader::updateEmbeddings(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_, device_idx);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_, device_idx);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddings(batch->unique_node_indices_, batch->node_gradients_);
            graph_storage_->updateAddNodeEmbeddingState(batch->unique_node_indices_, batch->node_state_update_);
        }
        batch->clear();
    }
}

void DataLoader::updateEmbeddingsG(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    if (gpu) {
        if (graph_storage_->storage_ptrs_.node_embeddings_g->device_ == torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddingsG(batch->unique_node_indices_, batch->node_gradients_g_, device_idx);
            graph_storage_->updateAddNodeEmbeddingStateG(batch->unique_node_indices_, batch->node_state_update_g_, device_idx);
        }
    } else {
        batch->host_transfer_.synchronize();
        if (graph_storage_->storage_ptrs_.node_embeddings_g->device_ != torch::kCUDA) {
            graph_storage_->updateAddNodeEmbeddingsG(batch->unique_node_indices_, batch->node_gradients_g_);
            graph_storage_->updateAddNodeEmbeddingStateG(batch->unique_node_indices_, batch->node_state_update_g_);
        }
        batch->clear();
    }
}

void DataLoader::loadStorage() {
    if (negative_sampler_ != nullptr) {
        negative_sampler_->resetPlanCache();
    }
    for (auto &prepared_superstep : prepared_supersteps_) {
        if (prepared_superstep.valid()) {
            prepared_superstep.wait();
        }
    }
    setBufferOrdering();
    graph_storage_->load();
    if (negative_sampling_method_ == NegativeSamplingMethod::GAN) {
        graph_storage_->load_g();
    }

    batch_id_offset_ = 0;
    total_batches_processed_ = 0;

    all_batches_ = std::vector<std::vector<shared_ptr<Batch>>>(devices_.size());
    batches_left_ = std::vector<int32_t>(devices_.size());
    batch_iterators_ = std::vector<std::vector<shared_ptr<Batch>>::iterator>(devices_.size());
    all_reads_ = std::vector<bool>(devices_.size());
    active_edge_block_starts_.resize(devices_.size());
    active_edge_block_sizes_.resize(devices_.size());
    active_edge_block_total_sizes_.assign(devices_.size(), 0);
    prepared_supersteps_.clear();
    prepared_supersteps_.resize(devices_.size());

    for (int device_idx = 0; device_idx < devices_.size(); device_idx ++) {
        all_reads_[device_idx] = false;
        active_edge_block_starts_[device_idx] = torch::Tensor();
        active_edge_block_sizes_[device_idx] = torch::Tensor();
    }

    if (!buffer_states_.empty()) {
        for(int device_idx = 0; device_idx < devices_.size(); device_idx ++) {
            graph_storage_->initializeInMemorySubGraph(buffer_states_[loaded_subgraphs ++], devices_[device_idx], device_idx);
        }
    } else {
        graph_storage_->initializeInMemorySubGraph(torch::empty({}));
    }

    if (negative_sampler_ != nullptr) {
        if (instance_of<NegativeSampler, CorruptNodeNegativeSampler>(negative_sampler_)) {
            if (std::dynamic_pointer_cast<CorruptNodeNegativeSampler>(negative_sampler_)->filtered_) {
                graph_storage_->sortAllEdges();
            }
        }
    }
}
