#include "engine/trainer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "configuration/options.h"
#include "reporting/logger.h"
#ifdef GEGE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#endif

using std::get;
using std::tie;

namespace {

bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return normalized != "0" && normalized != "false" && normalized != "off" && normalized != "no";
}

int64_t elapsed_ns(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

double ns_to_ms(int64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
}

double ns_to_ms(double ns) {
    return ns / 1.0e6;
}

std::string format_vector(const std::vector<int64_t> &values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    return oss.str();
}

std::string format_ms_vector(const std::vector<int64_t> &values) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << ns_to_ms(values[i]);
    }
    return oss.str();
}

double spread_ms(const std::vector<int64_t> &values) {
    if (values.empty()) {
        return 0.0;
    }
    auto minmax = std::minmax_element(values.begin(), values.end());
    return ns_to_ms(*minmax.second - *minmax.first);
}

#ifdef GEGE_CUDA
void record_tensor_on_current_stream(const torch::Tensor &tensor) {
    if (!tensor.defined() || !tensor.is_cuda() || tensor.numel() == 0) {
        return;
    }

    auto stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}

void record_tensor_vector_on_current_stream(const std::vector<torch::Tensor> &tensors) {
    for (const auto &tensor : tensors) {
        record_tensor_on_current_stream(tensor);
    }
}

void record_graph_tensors_on_current_stream(const GegeGraph &graph) {
    record_tensor_on_current_stream(graph.src_sorted_edges_);
    record_tensor_on_current_stream(graph.dst_sorted_edges_);
    record_tensor_on_current_stream(graph.active_in_memory_subgraph_);
    record_tensor_on_current_stream(graph.node_ids_);
    record_tensor_on_current_stream(graph.out_sorted_uniques_);
    record_tensor_on_current_stream(graph.out_offsets_);
    record_tensor_on_current_stream(graph.out_num_neighbors_);
    record_tensor_on_current_stream(graph.in_sorted_uniques_);
    record_tensor_on_current_stream(graph.in_offsets_);
    record_tensor_on_current_stream(graph.in_num_neighbors_);
    record_tensor_on_current_stream(graph.all_src_sorted_edges_);
    record_tensor_on_current_stream(graph.all_dst_sorted_edges_);
}

void record_dense_graph_tensors_on_current_stream(const DENSEGraph &graph) {
    record_graph_tensors_on_current_stream(graph);
    record_tensor_on_current_stream(graph.hop_offsets_);
    record_tensor_on_current_stream(graph.in_neighbors_mapping_);
    record_tensor_on_current_stream(graph.out_neighbors_mapping_);
    record_tensor_vector_on_current_stream(graph.in_neighbors_vec_);
    record_tensor_vector_on_current_stream(graph.out_neighbors_vec_);
    record_tensor_on_current_stream(graph.node_properties_);
}

void record_batch_tensors_on_current_stream(const shared_ptr<Batch> &batch) {
    if (batch == nullptr) {
        return;
    }

    record_tensor_on_current_stream(batch->root_node_indices_);
    record_tensor_on_current_stream(batch->unique_node_indices_);
    record_tensor_on_current_stream(batch->unique_node_active_mask_);
    record_tensor_on_current_stream(batch->node_embeddings_);
    record_tensor_on_current_stream(batch->node_gradients_);
    record_tensor_on_current_stream(batch->node_embeddings_state_);
    record_tensor_on_current_stream(batch->node_state_update_);
    record_tensor_on_current_stream(batch->node_embeddings_g_);
    record_tensor_on_current_stream(batch->node_gradients_g_);
    record_tensor_on_current_stream(batch->node_embeddings_state_g_);
    record_tensor_on_current_stream(batch->node_state_update_g_);
    record_tensor_on_current_stream(batch->qual_embeddings_);
    record_tensor_on_current_stream(batch->qual_gradients_);
    record_tensor_on_current_stream(batch->qual_embeddings_state_);
    record_tensor_on_current_stream(batch->qual_state_update_);
    record_tensor_on_current_stream(batch->qual_indices_);
    record_tensor_on_current_stream(batch->node_features_);
    record_tensor_on_current_stream(batch->node_labels_);
    record_tensor_on_current_stream(batch->src_neg_indices_mapping_);
    record_tensor_on_current_stream(batch->dst_neg_indices_mapping_);
    record_tensor_on_current_stream(batch->edges_);
    record_tensor_on_current_stream(batch->resident_src_embeddings_);
    record_tensor_on_current_stream(batch->resident_dst_embeddings_);
    record_tensor_on_current_stream(batch->resident_src_neg_embeddings_);
    record_tensor_on_current_stream(batch->resident_dst_neg_embeddings_);
    record_tensor_on_current_stream(batch->resident_src_embeddings_state_);
    record_tensor_on_current_stream(batch->resident_dst_embeddings_state_);
    record_tensor_on_current_stream(batch->resident_src_neg_embeddings_state_);
    record_tensor_on_current_stream(batch->resident_dst_neg_embeddings_state_);
    record_dense_graph_tensors_on_current_stream(batch->dense_graph_);
    record_tensor_on_current_stream(batch->encoded_uniques_);
    record_tensor_on_current_stream(batch->neg_edges_);
    record_tensor_on_current_stream(batch->rel_neg_indices_);
    record_tensor_on_current_stream(batch->src_neg_indices_);
    record_tensor_on_current_stream(batch->dst_neg_indices_);
    record_tensor_on_current_stream(batch->src_neg_filter_);
    record_tensor_on_current_stream(batch->dst_neg_filter_);
}
#endif

// These timers bracket host-side regions in the multi-GPU training loop.
// CUDA kernels may still execute asynchronously after the call returns.
struct DeviceEpochTiming {
    int64_t batch_count = 0;
    int64_t sync_count = 0;
    int64_t batch_fetch_region_ns = 0;
    int64_t gpu_load_region_ns = 0;
    int64_t map_region_ns = 0;
    int64_t compute_region_ns = 0;
    int64_t embedding_update_region_ns = 0;
    int64_t embedding_update_g_region_ns = 0;
    int64_t dense_sync_wait_ns = 0;
    int64_t dense_sync_wait_excl_all_reduce_ns = 0;
    int64_t dense_sync_all_reduce_ns = 0;
    int64_t finalize_region_ns = 0;
    int64_t prepared_batch_launches = 0;
    int64_t prepared_batch_hits = 0;
    int64_t prepared_batch_direct_fetches = 0;
    int64_t prepared_batch_boundary_blocks = 0;
    int64_t prepared_batch_wait_ns = 0;
};

struct SingleDeviceStateTiming {
    int64_t state_idx = -1;
    int64_t batch_count = 0;
    int64_t batch_fetch_region_ns = 0;
    int64_t gpu_load_region_ns = 0;
    int64_t map_region_ns = 0;
    int64_t compute_region_ns = 0;
    int64_t embedding_update_region_ns = 0;
    int64_t embedding_update_g_region_ns = 0;
    int64_t finalize_region_ns = 0;
};

struct SampleSummary {
    std::size_t count = 0;
    int64_t min = 0;
    int64_t max = 0;
    double median = 0.0;
    double average = 0.0;
};

int64_t sum_member(const std::vector<DeviceEpochTiming> &timings, int64_t DeviceEpochTiming::*member) {
    int64_t total = 0;
    for (const auto &timing : timings) {
        total += timing.*member;
    }
    return total;
}

std::vector<int64_t> collect_ns(const std::vector<DeviceEpochTiming> &timings, int64_t DeviceEpochTiming::*member) {
    std::vector<int64_t> values;
    values.reserve(timings.size());
    for (const auto &timing : timings) {
        values.emplace_back(timing.*member);
    }
    return values;
}

SampleSummary summarize_samples(const std::vector<int64_t> &values) {
    SampleSummary summary;
    summary.count = values.size();
    if (values.empty()) {
        return summary;
    }

    auto minmax = std::minmax_element(values.begin(), values.end());
    summary.min = *minmax.first;
    summary.max = *minmax.second;

    int64_t total = 0;
    for (auto value : values) {
        total += value;
    }
    summary.average = static_cast<double>(total) / static_cast<double>(values.size());

    std::vector<int64_t> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    std::size_t mid = sorted.size() / 2;
    if (sorted.size() % 2 == 0) {
        summary.median = static_cast<double>(sorted[mid - 1] + sorted[mid]) / 2.0;
    } else {
        summary.median = static_cast<double>(sorted[mid]);
    }

    return summary;
}

std::vector<int64_t> collect_frame_cache_stats(const std::vector<FrameCachePerfStats> &stats, int64_t FrameCachePerfStats::*member) {
    std::vector<int64_t> values;
    values.reserve(stats.size());
    for (const auto &stat : stats) {
        values.emplace_back(stat.*member);
    }
    return values;
}

double average_frame_cache_stat(const FrameCachePerfStats &stats, int64_t FrameCachePerfStats::*member) {
    if (stats.swap_samples == 0) {
        return 0.0;
    }
    return static_cast<double>(stats.*member) / static_cast<double>(stats.swap_samples);
}

void log_frame_cache_perf_stats(int64_t epoch, const DataLoaderPerfStats &perf_stats) {
    if (perf_stats.frame_cache.swap_samples <= 0) {
        return;
    }

    SPDLOG_INFO(
        "[perf][epoch {}][frame_cache] swap_samples={} visible_install_parts={} visible_install_rows={} hidden_publish_parts={} hidden_publish_rows={} "
        "fallback_visible_admit_parts={} fallback_visible_admit_rows={} preload_miss_swaps={} partial_preload_swaps={} delayed_stale_writeback_swaps={} "
        "async_admit_valid_before_swap_swaps={} async_evict_in_flight_before_swap_swaps={} reserved_preload_frames_avg={:.2f} "
        "free_frames_before_swap_avg={:.2f} free_frames_after_publish_avg={:.2f} stale_backlog_before_swap_max={} stale_backlog_after_publish_max={}",
        epoch, perf_stats.frame_cache.swap_samples, perf_stats.frame_cache.visible_install_parts, perf_stats.frame_cache.visible_install_rows,
        perf_stats.frame_cache.hidden_publish_parts, perf_stats.frame_cache.hidden_publish_rows,
        perf_stats.frame_cache.fallback_visible_admit_parts, perf_stats.frame_cache.fallback_visible_admit_rows,
        perf_stats.frame_cache.preload_miss_swap_count, perf_stats.frame_cache.partial_preload_swap_count,
        perf_stats.frame_cache.delayed_stale_writeback_swap_count, perf_stats.frame_cache.async_admit_valid_before_swap_count,
        perf_stats.frame_cache.async_evict_in_flight_before_swap_count,
        average_frame_cache_stat(perf_stats.frame_cache, &FrameCachePerfStats::reserved_preload_frames_sum),
        average_frame_cache_stat(perf_stats.frame_cache, &FrameCachePerfStats::free_frames_before_swap_sum),
        average_frame_cache_stat(perf_stats.frame_cache, &FrameCachePerfStats::free_frames_after_publish_sum),
        perf_stats.frame_cache.stale_backlog_before_swap_max, perf_stats.frame_cache.stale_backlog_after_publish_max);

    if (!perf_stats.device_frame_cache.empty()) {
        SPDLOG_INFO(
            "[perf][epoch {}][frame_cache][device] swap_samples={} hidden_publish_parts={} fallback_visible_admit_parts={} preload_miss_swaps={} "
            "partial_preload_swaps={} stale_backlog_after_publish_max={}",
            epoch,
            format_vector(collect_frame_cache_stats(perf_stats.device_frame_cache, &FrameCachePerfStats::swap_samples)),
            format_vector(collect_frame_cache_stats(perf_stats.device_frame_cache, &FrameCachePerfStats::hidden_publish_parts)),
            format_vector(collect_frame_cache_stats(perf_stats.device_frame_cache, &FrameCachePerfStats::fallback_visible_admit_parts)),
            format_vector(collect_frame_cache_stats(perf_stats.device_frame_cache, &FrameCachePerfStats::preload_miss_swap_count)),
            format_vector(collect_frame_cache_stats(perf_stats.device_frame_cache, &FrameCachePerfStats::partial_preload_swap_count)),
            format_vector(collect_frame_cache_stats(perf_stats.device_frame_cache, &FrameCachePerfStats::stale_backlog_after_publish_max)));
    }
}

}  // namespace

SynchronousTrainer::SynchronousTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->getPlannedEpochItemCount();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->getPlannedEpochItemCount();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {
    auto parse_optional_env_int = [](const char *name) -> std::optional<int32_t> {
        const char *env_value = std::getenv(name);
        if (env_value == nullptr || env_value[0] == '\0') {
            return std::nullopt;
        }
        try {
            return std::max(0, std::stoi(env_value));
        } catch (const std::exception &) {
            SPDLOG_WARN("Ignoring invalid {}={}", name, env_value);
            return std::nullopt;
        }
    };
    std::optional<int32_t> profiled_logical_lane = parse_optional_env_int("GEGE_PROFILE_LOGICAL_LANE");
    const bool prepared_batch_pipeline_requested = env_flag_enabled("GEGE_PREPARED_BATCH_PIPELINE");
    bool prepared_batch_pipeline_enabled = prepared_batch_pipeline_requested;
    std::string prepared_batch_pipeline_disabled_reason;
    if (prepared_batch_pipeline_enabled && learning_task_ != LearningTask::LINK_PREDICTION) {
        prepared_batch_pipeline_enabled = false;
        prepared_batch_pipeline_disabled_reason = "only link prediction is currently supported";
    }
    if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_ == nullptr) {
        prepared_batch_pipeline_enabled = false;
        prepared_batch_pipeline_disabled_reason = "graph storage is unavailable";
    }
    if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_->embeddingsOffDevice()) {
        prepared_batch_pipeline_enabled = false;
        prepared_batch_pipeline_disabled_reason = "off-device node embeddings would be read before prior updates finish";
    }
    if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_->embeddingsOffDeviceG()) {
        prepared_batch_pipeline_enabled = false;
        prepared_batch_pipeline_disabled_reason = "off-device generator embeddings would be read before prior updates finish";
    }
    if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_->storage_ptrs_.qual_embeddings != nullptr &&
        dataloader_->graph_storage_->storage_ptrs_.qual_embeddings->device_ != torch::kCUDA) {
        prepared_batch_pipeline_enabled = false;
        prepared_batch_pipeline_disabled_reason = "off-device qualifier embeddings would be read before prior updates finish";
    }
    std::string prepared_batch_launch_stage = "after_fetch";
    if (const char *launch_stage_env = std::getenv("GEGE_PREPARED_BATCH_PIPELINE_LAUNCH_STAGE")) {
        prepared_batch_launch_stage = launch_stage_env;
    }
    if (prepared_batch_launch_stage != "after_fetch" &&
        prepared_batch_launch_stage != "after_gpu_load" &&
        prepared_batch_launch_stage != "after_decoder_gather" &&
        prepared_batch_launch_stage != "after_forward" &&
        prepared_batch_launch_stage != "after_compute") {
        SPDLOG_WARN("Ignoring invalid GEGE_PREPARED_BATCH_PIPELINE_LAUNCH_STAGE={}; using after_fetch", prepared_batch_launch_stage);
        prepared_batch_launch_stage = "after_fetch";
    }
    if (prepared_batch_pipeline_requested) {
        if (prepared_batch_pipeline_enabled) {
            SPDLOG_INFO("[prepared_batch_pipeline] enabled=1 mode=intra_state_async_prepare max_prefetch=1 launch_stage={}",
                        prepared_batch_launch_stage);
        } else {
            SPDLOG_WARN("[prepared_batch_pipeline] enabled=0 reason={}", prepared_batch_pipeline_disabled_reason);
        }
    }

    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }
    dataloader_->initializeBatches(false);
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto epoch_start_wall = std::chrono::high_resolution_clock::now();
        if (has_last_epoch_end_) {
            SPDLOG_INFO("Inter-Epoch Gap: {}ms", elapsed_ns(last_epoch_end_, epoch_start_wall) / 1000000);
        }
        dataloader_->resetPerfStats();
        timer.start();
        std::vector<SingleDeviceStateTiming> state_timings;
        std::future<shared_ptr<Batch>> prepared_batch_future;
        int64_t prepared_batch_launches = 0;
        int64_t prepared_batch_hits = 0;
        int64_t prepared_batch_direct_fetches = 0;
        int64_t prepared_batch_boundary_blocks = 0;
        int64_t prepared_batch_wait_ns = 0;
        auto launch_prepared_batch = [&](bool wait_for_current_cuda_stream = false) {
            if (!prepared_batch_pipeline_enabled || prepared_batch_future.valid()) {
                return;
            }
            if (!dataloader_->canPrepareNextBatchInCurrentState()) {
                prepared_batch_boundary_blocks++;
                return;
            }
            prepared_batch_launches++;
            auto loader = dataloader_;
            std::uintptr_t wait_event_value = 0;
#ifdef GEGE_CUDA
            if (wait_for_current_cuda_stream && !loader->devices_.empty() && loader->devices_[0].is_cuda()) {
                c10::cuda::CUDAGuard device_guard(loader->devices_[0]);
                auto current_stream = c10::cuda::getCurrentCUDAStream(loader->devices_[0].index());
                cudaEvent_t wait_event = nullptr;
                AT_CUDA_CHECK(cudaEventCreateWithFlags(&wait_event, cudaEventDisableTiming));
                AT_CUDA_CHECK(cudaEventRecord(wait_event, current_stream.stream()));
                wait_event_value = reinterpret_cast<std::uintptr_t>(wait_event);
            }
#else
            (void)wait_for_current_cuda_stream;
#endif
            prepared_batch_future = std::async(std::launch::async, [loader, wait_event_value]() {
#ifdef GEGE_CUDA
                auto destroy_wait_event = [&]() {
                    if (wait_event_value != 0) {
                        auto wait_event = reinterpret_cast<cudaEvent_t>(wait_event_value);
                        cudaEventDestroy(wait_event);
                    }
                };
                if (!loader->devices_.empty() && loader->devices_[0].is_cuda()) {
                    try {
                        c10::cuda::CUDAGuard device_guard(loader->devices_[0]);
                        auto prepare_stream = c10::cuda::getStreamFromPool(false, loader->devices_[0].index());
                        if (wait_event_value != 0) {
                            auto wait_event = reinterpret_cast<cudaEvent_t>(wait_event_value);
                            AT_CUDA_CHECK(cudaStreamWaitEvent(prepare_stream.stream(), wait_event, 0));
                        }
                        c10::cuda::CUDAStreamGuard stream_guard(prepare_stream);
                        shared_ptr<Batch> batch = loader->getBatch();
                        AT_CUDA_CHECK(cudaStreamSynchronize(prepare_stream.stream()));
                        destroy_wait_event();
                        return batch;
                    } catch (...) {
                        destroy_wait_event();
                        throw;
                    }
                }
                destroy_wait_event();
#endif
                return loader->getBatch();
            });
        };
        auto fetch_batch = [&]() -> shared_ptr<Batch> {
            if (prepared_batch_future.valid()) {
                auto wait_start = std::chrono::high_resolution_clock::now();
                shared_ptr<Batch> batch = prepared_batch_future.get();
                prepared_batch_wait_ns += elapsed_ns(wait_start, std::chrono::high_resolution_clock::now());
                prepared_batch_hits++;
                return batch;
            }
            prepared_batch_direct_fetches++;
            return dataloader_->getBatch();
        };
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (prepared_batch_future.valid() || dataloader_->hasNextBatch()) {
            auto batch_fetch_start = std::chrono::high_resolution_clock::now();
            shared_ptr<Batch> batch = fetch_batch();
            auto batch_fetch_end = std::chrono::high_resolution_clock::now();
            if (batch == nullptr) {
                break;
            }

            int64_t state_idx = -1;
            if (!dataloader_->device_current_state_index_.empty()) {
                state_idx = dataloader_->device_current_state_index_[0];
            }
            if (state_timings.empty() || state_timings.back().state_idx != state_idx) {
                SingleDeviceStateTiming timing;
                timing.state_idx = state_idx;
                state_timings.emplace_back(timing);
            }
            auto &state_timing = state_timings.back();
            state_timing.batch_count++;
            state_timing.batch_fetch_region_ns += elapsed_ns(batch_fetch_start, batch_fetch_end);
            if (prepared_batch_launch_stage == "after_fetch") {
                launch_prepared_batch();
            }

            auto gpu_load_start = std::chrono::high_resolution_clock::now();
            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }
            auto gpu_load_end = std::chrono::high_resolution_clock::now();
            state_timing.gpu_load_region_ns += elapsed_ns(gpu_load_start, gpu_load_end);
#ifdef GEGE_CUDA
            if (prepared_batch_pipeline_enabled) {
                record_batch_tensors_on_current_stream(batch);
            }
#endif
            if (prepared_batch_launch_stage == "after_gpu_load") {
                launch_prepared_batch();
            }

            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }

            auto map_start = std::chrono::high_resolution_clock::now();
            batch->dense_graph_.performMap();
            auto map_end = std::chrono::high_resolution_clock::now();
            state_timing.map_region_ns += elapsed_ns(map_start, map_end);

            auto compute_start = std::chrono::high_resolution_clock::now();
            std::function<void()> post_forward_callback;
            std::function<void()> post_decoder_gather_callback;
            if (prepared_batch_launch_stage == "after_decoder_gather") {
                post_decoder_gather_callback = [&]() { launch_prepared_batch(true); };
            }
            if (prepared_batch_launch_stage == "after_forward") {
                post_forward_callback = [&]() { launch_prepared_batch(true); };
            }
            if (post_forward_callback || post_decoder_gather_callback) {
                model_->train_batch_with_callback(batch, true, post_forward_callback, post_decoder_gather_callback);
            } else {
                model_->train_batch(batch);
            }
            auto compute_end = std::chrono::high_resolution_clock::now();
            state_timing.compute_region_ns += elapsed_ns(compute_start, compute_end);
            if (prepared_batch_launch_stage == "after_compute") {
                launch_prepared_batch();
            }

            if (batch->node_gradients_.defined()) {
                auto embedding_update_start = std::chrono::high_resolution_clock::now();
                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddings(batch, true);
                }
                dataloader_->updateEmbeddings(batch, false);
                auto embedding_update_end = std::chrono::high_resolution_clock::now();
                state_timing.embedding_update_region_ns += elapsed_ns(embedding_update_start, embedding_update_end);
            }

            if (batch->node_embeddings_g_.defined()) {
                auto embedding_update_g_start = std::chrono::high_resolution_clock::now();
                if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                    batch->embeddingsToHostG();
                } else {
                    dataloader_->updateEmbeddingsG(batch, true);
                }
                dataloader_->updateEmbeddingsG(batch, false);
                auto embedding_update_g_end = std::chrono::high_resolution_clock::now();
                state_timing.embedding_update_g_region_ns += elapsed_ns(embedding_update_g_start, embedding_update_g_end);
            }

            auto finalize_start = std::chrono::high_resolution_clock::now();
#ifdef GEGE_CUDA
            if (prepared_batch_pipeline_enabled) {
                record_batch_tensors_on_current_stream(batch);
            }
#endif
            batch->clear();
            dataloader_->finishedBatch();
            auto finalize_end = std::chrono::high_resolution_clock::now();
            state_timing.finalize_region_ns += elapsed_ns(finalize_start, finalize_end);
            progress_reporter_->addResult(batch->batch_size_);
        }

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        last_epoch_end_ = std::chrono::high_resolution_clock::now();
        has_last_epoch_end_ = true;
        timer.stop();

        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->getPlannedEpochItemCount();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->getPlannedEpochItemCount();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);

        auto perf_stats = dataloader_->getPerfStats();
        if (prepared_batch_pipeline_requested) {
            SPDLOG_INFO(
                "[perf][epoch {}][prepared_batch_pipeline] enabled={} launches={} hits={} direct_fetches={} boundary_blocks={} wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), prepared_batch_pipeline_enabled ? 1 : 0, prepared_batch_launches, prepared_batch_hits,
                prepared_batch_direct_fetches, prepared_batch_boundary_blocks, ns_to_ms(prepared_batch_wait_ns));
        }
        if (profiled_logical_lane.has_value() && !state_timings.empty()) {
            const std::vector<int64_t> *active_bucket_samples = nullptr;
            const std::vector<int64_t> *active_edge_samples = nullptr;
            const std::vector<int64_t> *swap_batch_samples = nullptr;
            const std::vector<int64_t> *rebuild_samples = nullptr;
            if (!perf_stats.device_swap_active_bucket_samples.empty()) {
                active_bucket_samples = &perf_stats.device_swap_active_bucket_samples[0];
            }
            if (!perf_stats.device_swap_active_edge_samples.empty()) {
                active_edge_samples = &perf_stats.device_swap_active_edge_samples[0];
            }
            if (!perf_stats.device_swap_batch_count_samples.empty()) {
                swap_batch_samples = &perf_stats.device_swap_batch_count_samples[0];
            }
            if (!perf_stats.device_swap_rebuild_samples_ns.empty()) {
                rebuild_samples = &perf_stats.device_swap_rebuild_samples_ns[0];
            }

            double lane_process_ms_total = 0.0;
            double lane_rebuild_ms_total = 0.0;
            for (std::size_t state_pos = 0; state_pos < state_timings.size(); state_pos++) {
                const auto &state_timing = state_timings[state_pos];
                int64_t active_buckets = (active_bucket_samples != nullptr && state_pos < active_bucket_samples->size())
                    ? (*active_bucket_samples)[state_pos]
                    : -1;
                int64_t active_edges = (active_edge_samples != nullptr && state_pos < active_edge_samples->size())
                    ? (*active_edge_samples)[state_pos]
                    : -1;
                int64_t profiled_batches = (swap_batch_samples != nullptr && state_pos < swap_batch_samples->size())
                    ? (*swap_batch_samples)[state_pos]
                    : state_timing.batch_count;
                int64_t rebuild_ns = (rebuild_samples != nullptr && state_pos > 0 && (state_pos - 1) < rebuild_samples->size())
                    ? (*rebuild_samples)[state_pos - 1]
                    : 0;
                int64_t process_ns = state_timing.batch_fetch_region_ns + state_timing.gpu_load_region_ns + state_timing.map_region_ns +
                                     state_timing.compute_region_ns + state_timing.embedding_update_region_ns +
                                     state_timing.embedding_update_g_region_ns + state_timing.finalize_region_ns;
                lane_process_ms_total += ns_to_ms(process_ns);
                lane_rebuild_ms_total += ns_to_ms(rebuild_ns);
                SPDLOG_INFO(
                    "[perf][epoch {}][logical_lane {}][state_pos {}] state_idx={} active_buckets={} active_edges={} batches={} process_ms={:.3f} batch_fetch_ms={:.3f} gpu_load_ms={:.3f} map_ms={:.3f} compute_ms={:.3f} embedding_update_ms={:.3f} embedding_update_g_ms={:.3f} finalize_ms={:.3f} rebuild_ms={:.3f}",
                    dataloader_->getEpochsProcessed(), profiled_logical_lane.value(), state_pos, state_timing.state_idx, active_buckets, active_edges,
                    profiled_batches, ns_to_ms(process_ns), ns_to_ms(state_timing.batch_fetch_region_ns), ns_to_ms(state_timing.gpu_load_region_ns),
                    ns_to_ms(state_timing.map_region_ns), ns_to_ms(state_timing.compute_region_ns),
                    ns_to_ms(state_timing.embedding_update_region_ns), ns_to_ms(state_timing.embedding_update_g_region_ns),
                    ns_to_ms(state_timing.finalize_region_ns), ns_to_ms(rebuild_ns));
            }
            SPDLOG_INFO("[perf][epoch {}][logical_lane {}] process_ms_total={:.3f} rebuild_ms_total={:.3f} cycle_ms_total={:.3f} states={}",
                        dataloader_->getEpochsProcessed(), profiled_logical_lane.value(), lane_process_ms_total, lane_rebuild_ms_total,
                        lane_process_ms_total + lane_rebuild_ms_total, state_timings.size());
        }
        if (!state_timings.empty()) {
            int64_t batch_fetch_region_ns = 0;
            for (const auto &state_timing : state_timings) {
                batch_fetch_region_ns += state_timing.batch_fetch_region_ns;
            }
            SPDLOG_INFO(
                "[perf][epoch {}][batch_fetch] total_ms={:.3f} get_next_batch_ms={:.3f} get_next_direct_ms={:.3f} get_next_swap_path_ms={:.3f} get_next_swap_overhead_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
                dataloader_->getEpochsProcessed(), ns_to_ms(batch_fetch_region_ns), ns_to_ms(perf_stats.get_next_batch_ns),
                ns_to_ms(perf_stats.get_next_batch_direct_ns), ns_to_ms(perf_stats.get_next_batch_swap_path_ns),
                ns_to_ms(perf_stats.get_next_batch_swap_overhead_ns), ns_to_ms(perf_stats.edge_sample_ns),
                ns_to_ms(perf_stats.node_sample_ns), ns_to_ms(perf_stats.load_cpu_parameters_ns),
                ns_to_ms(perf_stats.get_batch_device_prepare_ns), ns_to_ms(perf_stats.get_batch_perform_map_ns), ns_to_ms(perf_stats.get_batch_overhead_ns));
            SPDLOG_INFO(
                "[perf][epoch {}][edge_sample] total_ms={:.3f} get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} compact_active_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f} unaccounted_ms={:.3f}",
                dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.edge_sample_ns), ns_to_ms(perf_stats.edge_get_edges_ns),
                ns_to_ms(perf_stats.edge_negative_sample_ns), ns_to_ms(perf_stats.edge_map_collect_ids_ns),
                ns_to_ms(perf_stats.edge_map_lookup_ns), ns_to_ms(perf_stats.edge_compact_active_ns),
                ns_to_ms(perf_stats.edge_map_verify_ns), ns_to_ms(perf_stats.edge_remap_assign_ns),
                ns_to_ms(perf_stats.edge_finalize_ns), ns_to_ms(perf_stats.edge_unaccounted_ns));
            SPDLOG_INFO(
                "[perf][epoch {}][map_tensors] calls={} validate_ms={:.3f} cat_ms={:.3f} unique_wall_ms={:.3f} split_ms={:.3f} total_ms={:.3f}",
                dataloader_->getEpochsProcessed(), perf_stats.map_tensor_calls, ns_to_ms(perf_stats.map_tensor_validate_ns),
                ns_to_ms(perf_stats.map_tensor_cat_ns), ns_to_ms(perf_stats.map_tensor_unique_ns), ns_to_ms(perf_stats.map_tensor_split_ns),
                ns_to_ms(perf_stats.map_tensor_total_ns));
            if (perf_stats.map_tensor_bitmap_zero_ns || perf_stats.map_tensor_bitmap_output_init_ns || perf_stats.map_tensor_bitmap_mark_ns ||
                perf_stats.map_tensor_bitmap_count_ns || perf_stats.map_tensor_bitmap_scan_ns || perf_stats.map_tensor_bitmap_extract_ns ||
                perf_stats.map_tensor_bitmap_mask_ns || perf_stats.map_tensor_bitmap_inverse_ns) {
                SPDLOG_INFO(
                    "[perf][epoch {}][map_tensors_device] bitmap_zero_ms={:.3f} output_init_ms={:.3f} mark_ms={:.3f} count_ms={:.3f} scan_ms={:.3f} extract_ms={:.3f} mask_ms={:.3f} inverse_ms={:.3f}",
                    dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.map_tensor_bitmap_zero_ns),
                    ns_to_ms(perf_stats.map_tensor_bitmap_output_init_ns), ns_to_ms(perf_stats.map_tensor_bitmap_mark_ns),
                    ns_to_ms(perf_stats.map_tensor_bitmap_count_ns), ns_to_ms(perf_stats.map_tensor_bitmap_scan_ns),
                    ns_to_ms(perf_stats.map_tensor_bitmap_extract_ns), ns_to_ms(perf_stats.map_tensor_bitmap_mask_ns),
                    ns_to_ms(perf_stats.map_tensor_bitmap_inverse_ns));
            }
            SPDLOG_INFO(
                "[perf][epoch {}][negative_sampler] calls={} call_ms_total={:.3f} plan_lock_calls={} plan_lock_wait_ms_total={:.3f}",
                dataloader_->getEpochsProcessed(), perf_stats.negative_sampler.get_negatives_call_count,
                ns_to_ms(perf_stats.negative_sampler.get_negatives_total_ns), perf_stats.negative_sampler.plan_lock_wait_count,
                ns_to_ms(perf_stats.negative_sampler.plan_lock_wait_ns));
            SPDLOG_INFO(
                "[perf][epoch {}][negative_sampler_breakdown] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f} state_pool_hits={} planned_uniform_fetches={} cuda_calls={} cpu_calls={}",
                dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.negative_sampler.uniform_randint_ns),
                ns_to_ms(perf_stats.negative_sampler.sample_edge_randint_ns), ns_to_ms(perf_stats.negative_sampler.materialize_ns),
                ns_to_ms(perf_stats.negative_sampler.filter_ns), perf_stats.negative_sampler.state_pool_hit_count,
                perf_stats.negative_sampler.planned_uniform_fetch_count, perf_stats.negative_sampler.cuda_call_count,
                perf_stats.negative_sampler.cpu_call_count);
        }
        if (perf_stats.swap_count > 0) {
            SPDLOG_INFO(
                "[perf][epoch {}] swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), perf_stats.swap_count, ns_to_ms(perf_stats.swap_barrier_wait_ns),
                ns_to_ms(perf_stats.swap_update_ns), ns_to_ms(perf_stats.swap_rebuild_ns), ns_to_ms(perf_stats.swap_sync_wait_ns));
            if (!perf_stats.device_swap_count.empty()) {
                SPDLOG_INFO("[perf][epoch {}][device] swap_count={}", dataloader_->getEpochsProcessed(),
                            format_vector(perf_stats.device_swap_count));
                SPDLOG_INFO("[perf][epoch {}][device] swap_barrier_wait_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_barrier_wait_ns));
                SPDLOG_INFO("[perf][epoch {}][device] swap_update_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_update_ns));
                SPDLOG_INFO("[perf][epoch {}][device] swap_rebuild_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_rebuild_ns));
                SPDLOG_INFO("[perf][epoch {}][device] swap_sync_wait_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.device_swap_sync_wait_ns));
                SPDLOG_INFO(
                    "[perf][epoch {}][spread] swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                    dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_swap_barrier_wait_ns),
                    spread_ms(perf_stats.device_swap_update_ns), spread_ms(perf_stats.device_swap_rebuild_ns),
                    spread_ms(perf_stats.device_swap_sync_wait_ns));
            }
        }
        log_frame_cache_perf_stats(dataloader_->getEpochsProcessed(), perf_stats);
    }
}

SynchronousMultiGPUTrainer::SynchronousMultiGPUTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->getPlannedEpochItemCount();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->getPlannedEpochItemCount();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousMultiGPUTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }

    dataloader_->activate_devices_ = model_->device_models_.size();

    for (int i = 0; i < model_->device_models_.size(); i++) {
        dataloader_->initializeBatches(false, i);
    }
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    Timer timer = Timer(false);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto epoch_start_wall = std::chrono::high_resolution_clock::now();
        if (has_last_epoch_end_) {
            SPDLOG_INFO("Inter-Epoch Gap: {}ms", elapsed_ns(last_epoch_end_, epoch_start_wall) / 1000000);
        }
        dataloader_->resetPerfStats();
        timer.start();
        std::atomic<int64_t> need_sync = 0;
        std::atomic<int64_t> sync_round = 0;
        std::atomic<int64_t> all_reduce_ns = 0;
        std::atomic<int64_t> all_reduce_calls = 0;
        std::vector<DeviceEpochTiming> device_timings(model_->device_models_.size());
        std::vector<std::atomic<int64_t>> sync_batch_counts(model_->device_models_.size());
        std::vector<std::atomic<int64_t>> sync_round_all_reduce_ns(model_->device_models_.size());
        for (auto &count : sync_batch_counts) {
            count.store(0);
        }
        for (auto &round_all_reduce_ns : sync_round_all_reduce_ns) {
            round_all_reduce_ns.store(0);
        }
        int dense_sync_batches = 1;
        if (dataloader_->training_config_ != nullptr) {
            dense_sync_batches = std::max(dataloader_->training_config_->dense_sync_batches, 1);
        }
        const bool prepared_batch_pipeline_requested =
            env_flag_enabled("GEGE_PREPARED_BATCH_PIPELINE") || env_flag_enabled("GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE");
        bool prepared_batch_pipeline_enabled = prepared_batch_pipeline_requested;
        std::string prepared_batch_pipeline_disabled_reason;
        if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_ == nullptr) {
            prepared_batch_pipeline_enabled = false;
            prepared_batch_pipeline_disabled_reason = "graph storage is unavailable";
        }
        if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_->embeddingsOffDevice()) {
            prepared_batch_pipeline_enabled = false;
            prepared_batch_pipeline_disabled_reason = "off-device node embeddings would be read before prior updates finish";
        }
        if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_->embeddingsOffDeviceG()) {
            prepared_batch_pipeline_enabled = false;
            prepared_batch_pipeline_disabled_reason = "off-device generator embeddings would be read before prior updates finish";
        }
        if (prepared_batch_pipeline_enabled && dataloader_->graph_storage_->storage_ptrs_.qual_embeddings != nullptr &&
            dataloader_->graph_storage_->storage_ptrs_.qual_embeddings->device_ != torch::kCUDA) {
            prepared_batch_pipeline_enabled = false;
            prepared_batch_pipeline_disabled_reason = "off-device qualifier embeddings would be read before prior updates finish";
        }
        std::string prepared_batch_launch_stage = "after_fetch";
        if (const char *launch_stage_env = std::getenv("GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE_LAUNCH_STAGE")) {
            prepared_batch_launch_stage = launch_stage_env;
        } else if (const char *launch_stage_env = std::getenv("GEGE_PREPARED_BATCH_PIPELINE_LAUNCH_STAGE")) {
            prepared_batch_launch_stage = launch_stage_env;
        }
        if (prepared_batch_launch_stage != "after_fetch" && prepared_batch_launch_stage != "after_gpu_load" &&
            prepared_batch_launch_stage != "after_compute") {
            SPDLOG_WARN("Ignoring invalid multi-GPU prepared batch launch stage {}; using after_fetch", prepared_batch_launch_stage);
            prepared_batch_launch_stage = "after_fetch";
        }
        const bool prepared_batch_low_priority = env_flag_enabled("GEGE_MULTI_GPU_PREPARED_BATCH_LOW_PRIORITY");
        if (prepared_batch_pipeline_requested) {
            if (prepared_batch_pipeline_enabled) {
                SPDLOG_INFO("[multi_gpu_prepared_batch_pipeline] enabled=1 mode=per_device_async_prepare max_prefetch=1 launch_stage={} low_priority={}",
                            prepared_batch_launch_stage, prepared_batch_low_priority ? 1 : 0);
            } else {
                SPDLOG_WARN("[multi_gpu_prepared_batch_pipeline] enabled=0 reason={}", prepared_batch_pipeline_disabled_reason);
            }
        }
        std::vector<std::thread> threads;

        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        SPDLOG_INFO("[perf][epoch {}] dense_sync_batches={} active_devices={}", dataloader_->getEpochsProcessed() + 1, dense_sync_batches,
                    model_->device_models_.size());
        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_round, &all_reduce_ns, &all_reduce_calls, &device_timings, &sync_batch_counts,
                                              &sync_round_all_reduce_ns, dense_sync_batches, prepared_batch_pipeline_enabled,
                                              prepared_batch_launch_stage, prepared_batch_low_priority, device_idx] {
                int64_t local_batches_since_sync = 0;
                std::future<shared_ptr<Batch>> prepared_batch_future;
                int prepare_stream_priority = 0;
#ifdef GEGE_CUDA
                if (prepared_batch_pipeline_enabled && prepared_batch_low_priority &&
                    device_idx >= 0 && static_cast<std::size_t>(device_idx) < dataloader_->devices_.size() &&
                    dataloader_->devices_[device_idx].is_cuda()) {
                    c10::cuda::CUDAGuard device_guard(dataloader_->devices_[device_idx]);
                    int least_priority = 0;
                    int greatest_priority = 0;
                    AT_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
                    prepare_stream_priority = least_priority;
                }
#endif
                auto launch_prepared_batch = [&]() {
                    if (!prepared_batch_pipeline_enabled || prepared_batch_future.valid()) {
                        return;
                    }
                    if (!dataloader_->canPrepareNextBatchInCurrentState(device_idx)) {
                        device_timings[device_idx].prepared_batch_boundary_blocks++;
                        return;
                    }
                    device_timings[device_idx].prepared_batch_launches++;
                    auto loader = dataloader_;
                    auto low_priority = prepared_batch_low_priority;
                    auto priority = prepare_stream_priority;
                    prepared_batch_future = std::async(std::launch::async, [loader, low_priority, priority, device_idx]() {
#ifdef GEGE_CUDA
                        if (device_idx >= 0 && static_cast<std::size_t>(device_idx) < loader->devices_.size() &&
                            loader->devices_[device_idx].is_cuda()) {
                            c10::cuda::CUDAGuard device_guard(loader->devices_[device_idx]);
                            auto prepare_stream = low_priority ? c10::cuda::getStreamFromPool(priority, loader->devices_[device_idx].index())
                                                               : c10::cuda::getStreamFromPool(false, loader->devices_[device_idx].index());
                            c10::cuda::CUDAStreamGuard stream_guard(prepare_stream);
                            shared_ptr<Batch> batch = loader->getBatch(c10::nullopt, false, device_idx);
                            AT_CUDA_CHECK(cudaStreamSynchronize(prepare_stream.stream()));
                            return batch;
                        }
#endif
                        return loader->getBatch(c10::nullopt, false, device_idx);
                    });
                };
                auto fetch_batch = [&]() -> shared_ptr<Batch> {
                    if (prepared_batch_future.valid()) {
                        auto wait_start = std::chrono::high_resolution_clock::now();
                        shared_ptr<Batch> batch = prepared_batch_future.get();
                        device_timings[device_idx].prepared_batch_wait_ns += elapsed_ns(wait_start, std::chrono::high_resolution_clock::now());
                        device_timings[device_idx].prepared_batch_hits++;
                        return batch;
                    }
                    device_timings[device_idx].prepared_batch_direct_fetches++;
                    return dataloader_->getBatch(c10::nullopt, false, device_idx);
                };
                while (prepared_batch_future.valid() || dataloader_->hasNextBatch(device_idx)) {
                    auto batch_fetch_start = std::chrono::high_resolution_clock::now();
                    shared_ptr<Batch> batch = fetch_batch();
                    auto batch_fetch_end = std::chrono::high_resolution_clock::now();
                    if (batch == nullptr) {
                        break;
                    }
                    device_timings[device_idx].batch_fetch_region_ns += elapsed_ns(batch_fetch_start, batch_fetch_end);
                    if (prepared_batch_launch_stage == "after_fetch") {
                        launch_prepared_batch();
                    }

                    bool has_relation = (batch->edges_.size(1) == 3);

                    auto gpu_load_start = std::chrono::high_resolution_clock::now();
                    dataloader_->loadGPUParameters(batch, device_idx);
                    auto gpu_load_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].gpu_load_region_ns += elapsed_ns(gpu_load_start, gpu_load_end);
#ifdef GEGE_CUDA
                    if (prepared_batch_pipeline_enabled) {
                        record_batch_tensors_on_current_stream(batch);
                    }
#endif
                    if (prepared_batch_launch_stage == "after_gpu_load") {
                        launch_prepared_batch();
                    }

                    if (batch->node_embeddings_.defined()) {
                        batch->node_embeddings_.requires_grad_();
                    }

                    auto map_start = std::chrono::high_resolution_clock::now();
                    batch->dense_graph_.performMap();
                    auto map_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].map_region_ns += elapsed_ns(map_start, map_end);

                    auto compute_start = std::chrono::high_resolution_clock::now();
                    model_->device_models_[device_idx]->train_batch(batch, false);
                    auto compute_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].compute_region_ns += elapsed_ns(compute_start, compute_end);
                    if (prepared_batch_launch_stage == "after_compute") {
                        launch_prepared_batch();
                    }

                    if (batch->node_gradients_.defined()) {
                        auto embedding_update_start = std::chrono::high_resolution_clock::now();
                        if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                            batch->embeddingsToHost();
                        } else {
                            dataloader_->updateEmbeddings(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddings(batch, false, device_idx);
                        auto embedding_update_end = std::chrono::high_resolution_clock::now();
                        device_timings[device_idx].embedding_update_region_ns += elapsed_ns(embedding_update_start, embedding_update_end);
                    }

                    if (batch->node_embeddings_g_.defined()) {
                        auto embedding_update_g_start = std::chrono::high_resolution_clock::now();
                        if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                            batch->embeddingsToHostG();
                        } else {
                            dataloader_->updateEmbeddingsG(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddingsG(batch, false, device_idx);
                        auto embedding_update_g_end = std::chrono::high_resolution_clock::now();
                        device_timings[device_idx].embedding_update_g_region_ns += elapsed_ns(embedding_update_g_start, embedding_update_g_end);
                    }

                    if (has_relation) {
                        local_batches_since_sync++;
                        bool should_sync = (local_batches_since_sync >= dense_sync_batches) || (dataloader_->batches_left_[device_idx] == 1);
                        if (should_sync) {
                            auto sync_start = std::chrono::high_resolution_clock::now();
                            int64_t round = sync_round.load();
                            sync_round_all_reduce_ns[device_idx].store(0);
                            sync_batch_counts[device_idx].store(local_batches_since_sync);
                            int64_t arrivals = need_sync.fetch_add(1) + 1;
                            device_timings[device_idx].sync_count++;

                            if (arrivals == dataloader_->activate_devices_.load()) {
                                auto all_reduce_start = std::chrono::high_resolution_clock::now();
                                std::vector<int64_t> grad_scales(sync_batch_counts.size(), 1);
                                std::vector<int32_t> round_participants;
                                for (int i = 0; i < sync_batch_counts.size(); i++) {
                                    int64_t sync_batches = sync_batch_counts[i].load();
                                    if (sync_batches > 0) {
                                        grad_scales[i] = std::max<int64_t>(sync_batches, 1);
                                        round_participants.emplace_back(i);
                                    }
                                }
                                model_->all_reduce(grad_scales);
                                int64_t round_all_reduce_elapsed = elapsed_ns(all_reduce_start, std::chrono::high_resolution_clock::now());
                                all_reduce_ns.fetch_add(round_all_reduce_elapsed);
                                all_reduce_calls.fetch_add(1);
                                for (auto &round_all_reduce_ns : sync_round_all_reduce_ns) {
                                    round_all_reduce_ns.store(round_all_reduce_elapsed);
                                }
                                for (auto participant_idx : round_participants) {
                                    sync_batch_counts[participant_idx].store(0);
                                }
                                need_sync.store(0);
                                sync_round.fetch_add(1);
                            }
                            while (sync_round.load() == round) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                            int64_t sync_elapsed = elapsed_ns(sync_start, std::chrono::high_resolution_clock::now());
                            int64_t round_all_reduce_elapsed = sync_round_all_reduce_ns[device_idx].load();
                            device_timings[device_idx].dense_sync_wait_ns += sync_elapsed;
                            device_timings[device_idx].dense_sync_all_reduce_ns += round_all_reduce_elapsed;
                            device_timings[device_idx].dense_sync_wait_excl_all_reduce_ns +=
                                std::max<int64_t>(sync_elapsed - round_all_reduce_elapsed, 0LL);
                            local_batches_since_sync = 0;
                        }
                    }

                    auto finalize_start = std::chrono::high_resolution_clock::now();
                    batch->clear();
                    dataloader_->finishedBatch(device_idx);
                    auto finalize_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].finalize_region_ns += elapsed_ns(finalize_start, finalize_end);
                    device_timings[device_idx].batch_count++;
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        last_epoch_end_ = std::chrono::high_resolution_clock::now();
        has_last_epoch_end_ = true;
        timer.stop();
        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->getPlannedEpochItemCount();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->getPlannedEpochItemCount();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);

        auto perf_stats = dataloader_->getPerfStats();
        bool have_device_swap_stats =
            perf_stats.device_swap_count.size() == device_timings.size() &&
            perf_stats.device_swap_barrier_wait_ns.size() == device_timings.size() &&
            perf_stats.device_swap_update_ns.size() == device_timings.size() &&
            perf_stats.device_swap_rebuild_ns.size() == device_timings.size() &&
            perf_stats.device_swap_sync_wait_ns.size() == device_timings.size();
        bool have_device_batch_fetch_stats =
            perf_stats.device_get_next_batch_ns.size() == device_timings.size() &&
            perf_stats.device_get_next_batch_direct_ns.size() == device_timings.size() &&
            perf_stats.device_get_next_batch_swap_path_ns.size() == device_timings.size() &&
            perf_stats.device_get_next_batch_swap_overhead_ns.size() == device_timings.size() &&
            perf_stats.device_edge_sample_ns.size() == device_timings.size() &&
            perf_stats.device_edge_get_edges_ns.size() == device_timings.size() &&
            perf_stats.device_edge_negative_sample_ns.size() == device_timings.size() &&
            perf_stats.device_edge_map_collect_ids_ns.size() == device_timings.size() &&
            perf_stats.device_edge_map_lookup_ns.size() == device_timings.size() &&
            perf_stats.device_edge_map_verify_ns.size() == device_timings.size() &&
            perf_stats.device_edge_remap_assign_ns.size() == device_timings.size() &&
            perf_stats.device_edge_compact_active_ns.size() == device_timings.size() &&
            perf_stats.device_edge_finalize_ns.size() == device_timings.size() &&
            perf_stats.device_edge_unaccounted_ns.size() == device_timings.size() &&
            perf_stats.device_node_sample_ns.size() == device_timings.size() &&
            perf_stats.device_load_cpu_parameters_ns.size() == device_timings.size() &&
            perf_stats.device_get_batch_device_prepare_ns.size() == device_timings.size() &&
            perf_stats.device_get_batch_perform_map_ns.size() == device_timings.size() &&
            perf_stats.device_get_batch_overhead_ns.size() == device_timings.size();
        bool have_device_swap_state_samples =
            perf_stats.device_swap_active_bucket_samples.size() == device_timings.size() &&
            perf_stats.device_swap_active_edge_samples.size() == device_timings.size() &&
            perf_stats.device_swap_batch_count_samples.size() == device_timings.size() &&
            perf_stats.device_swap_rebuild_samples_ns.size() == device_timings.size();
        bool have_device_negative_sampler_stats =
            perf_stats.negative_sampler.device_get_negatives_total_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_get_negatives_call_count.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_plan_lock_wait_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_plan_lock_wait_count.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_state_pool_hit_count.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_planned_uniform_fetch_count.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_cuda_call_count.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_cpu_call_count.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_uniform_randint_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_sample_edge_randint_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_materialize_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_filter_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_get_negatives_samples_ns.size() == device_timings.size() &&
            perf_stats.negative_sampler.device_plan_lock_wait_samples_ns.size() == device_timings.size();
        if (!have_device_swap_stats &&
            (!perf_stats.device_swap_count.empty() || !perf_stats.device_swap_barrier_wait_ns.empty() ||
             !perf_stats.device_swap_update_ns.empty() || !perf_stats.device_swap_rebuild_ns.empty() ||
             !perf_stats.device_swap_sync_wait_ns.empty())) {
            SPDLOG_WARN("[perf][epoch {}] device swap stats are unavailable or size-mismatched for {} GPU timing entries",
                        dataloader_->getEpochsProcessed(), device_timings.size());
        }
        if (!have_device_batch_fetch_stats &&
            (!perf_stats.device_get_next_batch_ns.empty() || !perf_stats.device_edge_sample_ns.empty() ||
             !perf_stats.device_edge_get_edges_ns.empty() || !perf_stats.device_edge_negative_sample_ns.empty() ||
             !perf_stats.device_edge_map_collect_ids_ns.empty() || !perf_stats.device_edge_map_lookup_ns.empty() ||
             !perf_stats.device_edge_map_verify_ns.empty() || !perf_stats.device_edge_remap_assign_ns.empty() ||
             !perf_stats.device_edge_compact_active_ns.empty() || !perf_stats.device_edge_finalize_ns.empty() ||
             !perf_stats.device_edge_unaccounted_ns.empty() || !perf_stats.device_node_sample_ns.empty() ||
             !perf_stats.device_load_cpu_parameters_ns.empty() || !perf_stats.device_get_batch_device_prepare_ns.empty() ||
             !perf_stats.device_get_batch_perform_map_ns.empty() || !perf_stats.device_get_batch_overhead_ns.empty())) {
            SPDLOG_WARN("[perf][epoch {}] device batch-fetch stats are unavailable or size-mismatched for {} GPU timing entries",
                        dataloader_->getEpochsProcessed(), device_timings.size());
        }
        if (!have_device_swap_state_samples &&
            (!perf_stats.device_swap_active_bucket_samples.empty() || !perf_stats.device_swap_active_edge_samples.empty() ||
             !perf_stats.device_swap_batch_count_samples.empty() || !perf_stats.device_swap_rebuild_samples_ns.empty())) {
            SPDLOG_WARN("[perf][epoch {}] device swap-state samples are unavailable or size-mismatched for {} GPU timing entries",
                        dataloader_->getEpochsProcessed(), device_timings.size());
        }
        if (!have_device_negative_sampler_stats &&
            (!perf_stats.negative_sampler.device_get_negatives_total_ns.empty() ||
             !perf_stats.negative_sampler.device_get_negatives_call_count.empty() ||
             !perf_stats.negative_sampler.device_plan_lock_wait_ns.empty() ||
             !perf_stats.negative_sampler.device_plan_lock_wait_count.empty() ||
             !perf_stats.negative_sampler.device_state_pool_hit_count.empty() ||
             !perf_stats.negative_sampler.device_planned_uniform_fetch_count.empty() ||
             !perf_stats.negative_sampler.device_cuda_call_count.empty() ||
             !perf_stats.negative_sampler.device_cpu_call_count.empty() ||
             !perf_stats.negative_sampler.device_uniform_randint_ns.empty() ||
             !perf_stats.negative_sampler.device_sample_edge_randint_ns.empty() ||
             !perf_stats.negative_sampler.device_materialize_ns.empty() ||
             !perf_stats.negative_sampler.device_filter_ns.empty() ||
             !perf_stats.negative_sampler.device_get_negatives_samples_ns.empty() ||
             !perf_stats.negative_sampler.device_plan_lock_wait_samples_ns.empty())) {
            SPDLOG_WARN("[perf][epoch {}] negative-sampler device stats are unavailable or size-mismatched for {} GPU timing entries",
                        dataloader_->getEpochsProcessed(), device_timings.size());
        }
        SPDLOG_INFO(
            "[perf][epoch {}] dense_sync_batches={} batches={} sync_points={} all_reduce_calls={} batch_fetch_region_sum_ms={:.3f} gpu_load_region_sum_ms={:.3f} map_region_sum_ms={:.3f} compute_region_sum_ms={:.3f} embedding_update_region_sum_ms={:.3f} embedding_update_g_region_sum_ms={:.3f} dense_sync_wait_sum_ms={:.3f} dense_sync_wait_excl_all_reduce_sum_ms={:.3f} all_reduce_total_ms={:.3f} finalize_region_sum_ms={:.3f} swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
            dataloader_->getEpochsProcessed(), dense_sync_batches, sum_member(device_timings, &DeviceEpochTiming::batch_count),
            sum_member(device_timings, &DeviceEpochTiming::sync_count), all_reduce_calls.load(),
            ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::batch_fetch_region_ns)),
            ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::gpu_load_region_ns)), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::map_region_ns)),
            ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::compute_region_ns)), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::embedding_update_region_ns)),
            ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::embedding_update_g_region_ns)), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::dense_sync_wait_ns)),
            ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::dense_sync_wait_excl_all_reduce_ns)), ns_to_ms(all_reduce_ns.load()),
            ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::finalize_region_ns)),
            perf_stats.swap_count, ns_to_ms(perf_stats.swap_barrier_wait_ns), ns_to_ms(perf_stats.swap_update_ns), ns_to_ms(perf_stats.swap_rebuild_ns),
            ns_to_ms(perf_stats.swap_sync_wait_ns));
        if (perf_stats.peer_relay.peer_bytes_executed > 0 || perf_stats.peer_relay.host_fallback_bytes > 0) {
            SPDLOG_INFO(
                "[perf][epoch {}][peer_relay] peer_bytes_executed={} host_bytes_saved={} host_fallback_bytes={} peer_copy_count={} host_fallback_count={} descriptor_mismatch_count={} peer_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), perf_stats.peer_relay.peer_bytes_executed, perf_stats.peer_relay.peer_bytes_executed,
                perf_stats.peer_relay.host_fallback_bytes, perf_stats.peer_relay.peer_copy_count, perf_stats.peer_relay.host_fallback_count,
                perf_stats.peer_relay.descriptor_mismatch_count,
                ns_to_ms(perf_stats.peer_relay.peer_sync_wait_ns));
        }
        if (prepared_batch_pipeline_requested) {
            SPDLOG_INFO(
                "[perf][epoch {}][prepared_batch] enabled={} launches={} hits={} direct_fetches={} boundary_blocks={} wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), prepared_batch_pipeline_enabled ? 1 : 0,
                sum_member(device_timings, &DeviceEpochTiming::prepared_batch_launches),
                sum_member(device_timings, &DeviceEpochTiming::prepared_batch_hits),
                sum_member(device_timings, &DeviceEpochTiming::prepared_batch_direct_fetches),
                sum_member(device_timings, &DeviceEpochTiming::prepared_batch_boundary_blocks),
                ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::prepared_batch_wait_ns)));
        }
        SPDLOG_INFO(
            "[perf][epoch {}][batch_fetch] total_ms={:.3f} get_next_batch_ms={:.3f} get_next_direct_ms={:.3f} get_next_swap_path_ms={:.3f} get_next_swap_overhead_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
            dataloader_->getEpochsProcessed(), ns_to_ms(sum_member(device_timings, &DeviceEpochTiming::batch_fetch_region_ns)),
            ns_to_ms(perf_stats.get_next_batch_ns), ns_to_ms(perf_stats.get_next_batch_direct_ns),
            ns_to_ms(perf_stats.get_next_batch_swap_path_ns), ns_to_ms(perf_stats.get_next_batch_swap_overhead_ns),
            ns_to_ms(perf_stats.edge_sample_ns), ns_to_ms(perf_stats.node_sample_ns), ns_to_ms(perf_stats.load_cpu_parameters_ns),
            ns_to_ms(perf_stats.get_batch_device_prepare_ns), ns_to_ms(perf_stats.get_batch_perform_map_ns), ns_to_ms(perf_stats.get_batch_overhead_ns));
        SPDLOG_INFO(
            "[perf][epoch {}][edge_sample] total_ms={:.3f} get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} compact_active_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f} unaccounted_ms={:.3f}",
            dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.edge_sample_ns), ns_to_ms(perf_stats.edge_get_edges_ns),
            ns_to_ms(perf_stats.edge_negative_sample_ns), ns_to_ms(perf_stats.edge_map_collect_ids_ns), ns_to_ms(perf_stats.edge_map_lookup_ns),
            ns_to_ms(perf_stats.edge_compact_active_ns), ns_to_ms(perf_stats.edge_map_verify_ns),
            ns_to_ms(perf_stats.edge_remap_assign_ns), ns_to_ms(perf_stats.edge_finalize_ns), ns_to_ms(perf_stats.edge_unaccounted_ns));
        SPDLOG_INFO(
            "[perf][epoch {}][map_tensors] calls={} validate_ms={:.3f} cat_ms={:.3f} unique_wall_ms={:.3f} split_ms={:.3f} total_ms={:.3f}",
            dataloader_->getEpochsProcessed(), perf_stats.map_tensor_calls, ns_to_ms(perf_stats.map_tensor_validate_ns),
            ns_to_ms(perf_stats.map_tensor_cat_ns), ns_to_ms(perf_stats.map_tensor_unique_ns), ns_to_ms(perf_stats.map_tensor_split_ns),
            ns_to_ms(perf_stats.map_tensor_total_ns));
        SPDLOG_INFO(
            "[perf][epoch {}][negative_sampler] calls={} call_ms_total={:.3f} plan_lock_calls={} plan_lock_wait_ms_total={:.3f}",
            dataloader_->getEpochsProcessed(), perf_stats.negative_sampler.get_negatives_call_count,
            ns_to_ms(perf_stats.negative_sampler.get_negatives_total_ns), perf_stats.negative_sampler.plan_lock_wait_count,
            ns_to_ms(perf_stats.negative_sampler.plan_lock_wait_ns));
        SPDLOG_INFO(
            "[perf][epoch {}][negative_sampler_breakdown] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f} state_pool_hits={} planned_uniform_fetches={} cuda_calls={} cpu_calls={}",
            dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.negative_sampler.uniform_randint_ns),
            ns_to_ms(perf_stats.negative_sampler.sample_edge_randint_ns), ns_to_ms(perf_stats.negative_sampler.materialize_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_ns), perf_stats.negative_sampler.state_pool_hit_count,
            perf_stats.negative_sampler.planned_uniform_fetch_count, perf_stats.negative_sampler.cuda_call_count,
            perf_stats.negative_sampler.cpu_call_count);
        SPDLOG_INFO(
            "[perf][epoch {}][negative_filter_breakdown] deg_chunk_ids_ms={:.3f} deg_mask_ms={:.3f} deg_nonzero_ms={:.3f} deg_gather_ms={:.3f} deg_finalize_ms={:.3f} gpu_prepare_ms={:.3f} gpu_searchsorted_ms={:.3f} gpu_offsets_ms={:.3f} gpu_repeat_interleave_ms={:.3f} gpu_neighbor_gather_ms={:.3f} gpu_relation_filter_ms={:.3f} gpu_finalize_ms={:.3f}",
            dataloader_->getEpochsProcessed(), ns_to_ms(perf_stats.negative_sampler.filter_deg_chunk_ids_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_deg_mask_ns), ns_to_ms(perf_stats.negative_sampler.filter_deg_nonzero_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_deg_gather_ns), ns_to_ms(perf_stats.negative_sampler.filter_deg_finalize_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_gpu_prepare_ns), ns_to_ms(perf_stats.negative_sampler.filter_gpu_searchsorted_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_gpu_offsets_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_gpu_repeat_interleave_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_gpu_neighbor_gather_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_gpu_relation_filter_ns),
            ns_to_ms(perf_stats.negative_sampler.filter_gpu_finalize_ns));
        log_frame_cache_perf_stats(dataloader_->getEpochsProcessed(), perf_stats);
        for (int32_t device_idx = 0; device_idx < static_cast<int32_t>(device_timings.size()); device_idx++) {
            const auto &timing = device_timings[device_idx];
            int64_t swap_count = have_device_swap_stats ? perf_stats.device_swap_count[device_idx] : 0;
            int64_t swap_barrier = have_device_swap_stats ? perf_stats.device_swap_barrier_wait_ns[device_idx] : 0;
            int64_t swap_update = have_device_swap_stats ? perf_stats.device_swap_update_ns[device_idx] : 0;
            int64_t swap_rebuild = have_device_swap_stats ? perf_stats.device_swap_rebuild_ns[device_idx] : 0;
            int64_t swap_sync = have_device_swap_stats ? perf_stats.device_swap_sync_wait_ns[device_idx] : 0;
            int64_t get_next_batch = have_device_batch_fetch_stats ? perf_stats.device_get_next_batch_ns[device_idx] : 0;
            int64_t get_next_direct = have_device_batch_fetch_stats ? perf_stats.device_get_next_batch_direct_ns[device_idx] : 0;
            int64_t get_next_swap_path = have_device_batch_fetch_stats ? perf_stats.device_get_next_batch_swap_path_ns[device_idx] : 0;
            int64_t get_next_swap_overhead = have_device_batch_fetch_stats ? perf_stats.device_get_next_batch_swap_overhead_ns[device_idx] : 0;
            int64_t edge_sample = have_device_batch_fetch_stats ? perf_stats.device_edge_sample_ns[device_idx] : 0;
            int64_t edge_get_edges = have_device_batch_fetch_stats ? perf_stats.device_edge_get_edges_ns[device_idx] : 0;
            int64_t edge_negative_sample = have_device_batch_fetch_stats ? perf_stats.device_edge_negative_sample_ns[device_idx] : 0;
            int64_t edge_map_collect_ids = have_device_batch_fetch_stats ? perf_stats.device_edge_map_collect_ids_ns[device_idx] : 0;
            int64_t edge_map_lookup = have_device_batch_fetch_stats ? perf_stats.device_edge_map_lookup_ns[device_idx] : 0;
            int64_t edge_map_verify = have_device_batch_fetch_stats ? perf_stats.device_edge_map_verify_ns[device_idx] : 0;
            int64_t edge_remap_assign = have_device_batch_fetch_stats ? perf_stats.device_edge_remap_assign_ns[device_idx] : 0;
            int64_t edge_compact_active = have_device_batch_fetch_stats ? perf_stats.device_edge_compact_active_ns[device_idx] : 0;
            int64_t edge_finalize = have_device_batch_fetch_stats ? perf_stats.device_edge_finalize_ns[device_idx] : 0;
            int64_t edge_unaccounted = have_device_batch_fetch_stats ? perf_stats.device_edge_unaccounted_ns[device_idx] : 0;
            int64_t node_sample = have_device_batch_fetch_stats ? perf_stats.device_node_sample_ns[device_idx] : 0;
            int64_t load_cpu_parameters = have_device_batch_fetch_stats ? perf_stats.device_load_cpu_parameters_ns[device_idx] : 0;
            int64_t device_prepare = have_device_batch_fetch_stats ? perf_stats.device_get_batch_device_prepare_ns[device_idx] : 0;
            int64_t perform_map = have_device_batch_fetch_stats ? perf_stats.device_get_batch_perform_map_ns[device_idx] : 0;
            int64_t get_batch_overhead = have_device_batch_fetch_stats ? perf_stats.device_get_batch_overhead_ns[device_idx] : 0;
            SPDLOG_INFO(
                "[perf][epoch {}][gpu {}] batches={} sync_points={} batch_fetch_region_ms={:.3f} gpu_load_region_ms={:.3f} map_region_ms={:.3f} compute_region_ms={:.3f} embedding_update_region_ms={:.3f} embedding_update_g_region_ms={:.3f} dense_sync_wait_ms={:.3f} dense_sync_wait_excl_all_reduce_ms={:.3f} dense_sync_all_reduce_ms={:.3f} finalize_region_ms={:.3f} swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), device_idx, timing.batch_count, timing.sync_count, ns_to_ms(timing.batch_fetch_region_ns),
                ns_to_ms(timing.gpu_load_region_ns), ns_to_ms(timing.map_region_ns), ns_to_ms(timing.compute_region_ns), ns_to_ms(timing.embedding_update_region_ns),
                ns_to_ms(timing.embedding_update_g_region_ns), ns_to_ms(timing.dense_sync_wait_ns), ns_to_ms(timing.dense_sync_wait_excl_all_reduce_ns),
                ns_to_ms(timing.dense_sync_all_reduce_ns), ns_to_ms(timing.finalize_region_ns), swap_count,
                ns_to_ms(swap_barrier), ns_to_ms(swap_update), ns_to_ms(swap_rebuild), ns_to_ms(swap_sync));
            SPDLOG_INFO(
                "[perf][epoch {}][gpu {}][batch_fetch] total_ms={:.3f} get_next_batch_ms={:.3f} get_next_direct_ms={:.3f} get_next_swap_path_ms={:.3f} get_next_swap_overhead_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
                dataloader_->getEpochsProcessed(), device_idx, ns_to_ms(timing.batch_fetch_region_ns), ns_to_ms(get_next_batch), ns_to_ms(get_next_direct),
                ns_to_ms(get_next_swap_path), ns_to_ms(get_next_swap_overhead), ns_to_ms(edge_sample), ns_to_ms(node_sample),
                ns_to_ms(load_cpu_parameters), ns_to_ms(device_prepare), ns_to_ms(perform_map), ns_to_ms(get_batch_overhead));
            if (prepared_batch_pipeline_requested) {
                SPDLOG_INFO(
                    "[perf][epoch {}][gpu {}][prepared_batch] launches={} hits={} direct_fetches={} boundary_blocks={} wait_ms={:.3f}",
                    dataloader_->getEpochsProcessed(), device_idx, timing.prepared_batch_launches, timing.prepared_batch_hits,
                    timing.prepared_batch_direct_fetches, timing.prepared_batch_boundary_blocks, ns_to_ms(timing.prepared_batch_wait_ns));
            }
            SPDLOG_INFO(
                "[perf][epoch {}][gpu {}][edge_sample] total_ms={:.3f} get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} compact_active_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f} unaccounted_ms={:.3f}",
                dataloader_->getEpochsProcessed(), device_idx, ns_to_ms(edge_sample), ns_to_ms(edge_get_edges), ns_to_ms(edge_negative_sample),
                ns_to_ms(edge_map_collect_ids), ns_to_ms(edge_map_lookup), ns_to_ms(edge_compact_active), ns_to_ms(edge_map_verify),
                ns_to_ms(edge_remap_assign), ns_to_ms(edge_finalize), ns_to_ms(edge_unaccounted));
            if (have_device_swap_state_samples) {
                auto active_bucket_summary = summarize_samples(perf_stats.device_swap_active_bucket_samples[device_idx]);
                auto active_edge_summary = summarize_samples(perf_stats.device_swap_active_edge_samples[device_idx]);
                auto batch_count_summary = summarize_samples(perf_stats.device_swap_batch_count_samples[device_idx]);
                auto rebuild_summary = summarize_samples(perf_stats.device_swap_rebuild_samples_ns[device_idx]);
                SPDLOG_INFO(
                    "[perf][epoch {}][gpu {}][swap_state] states={} active_buckets_min={} active_buckets_med={:.1f} active_buckets_avg={:.1f} active_buckets_max={} active_edges_min={} active_edges_med={:.1f} active_edges_avg={:.1f} active_edges_max={} batches_min={} batches_med={:.1f} batches_avg={:.1f} batches_max={} rebuild_ms_min={:.3f} rebuild_ms_med={:.3f} rebuild_ms_avg={:.3f} rebuild_ms_max={:.3f}",
                    dataloader_->getEpochsProcessed(), device_idx, active_edge_summary.count, active_bucket_summary.min, active_bucket_summary.median,
                    active_bucket_summary.average, active_bucket_summary.max, active_edge_summary.min, active_edge_summary.median,
                    active_edge_summary.average, active_edge_summary.max, batch_count_summary.min, batch_count_summary.median,
                    batch_count_summary.average, batch_count_summary.max, ns_to_ms(rebuild_summary.min), ns_to_ms(rebuild_summary.median),
                    ns_to_ms(rebuild_summary.average), ns_to_ms(rebuild_summary.max));
            }
            if (have_device_negative_sampler_stats) {
                auto negative_call_summary = summarize_samples(perf_stats.negative_sampler.device_get_negatives_samples_ns[device_idx]);
                auto negative_lock_summary = summarize_samples(perf_stats.negative_sampler.device_plan_lock_wait_samples_ns[device_idx]);
                SPDLOG_INFO(
                    "[perf][epoch {}][gpu {}][negative_sampler] calls={} call_ms_total={:.3f} call_ms_min={:.3f} call_ms_med={:.3f} call_ms_avg={:.3f} call_ms_max={:.3f} plan_lock_calls={} plan_lock_wait_ms_total={:.3f} plan_lock_wait_ms_min={:.3f} plan_lock_wait_ms_med={:.3f} plan_lock_wait_ms_avg={:.3f} plan_lock_wait_ms_max={:.3f}",
                    dataloader_->getEpochsProcessed(), device_idx, perf_stats.negative_sampler.device_get_negatives_call_count[device_idx],
                    ns_to_ms(perf_stats.negative_sampler.device_get_negatives_total_ns[device_idx]), ns_to_ms(negative_call_summary.min),
                    ns_to_ms(negative_call_summary.median), ns_to_ms(negative_call_summary.average), ns_to_ms(negative_call_summary.max),
                    perf_stats.negative_sampler.device_plan_lock_wait_count[device_idx],
                    ns_to_ms(perf_stats.negative_sampler.device_plan_lock_wait_ns[device_idx]), ns_to_ms(negative_lock_summary.min),
                    ns_to_ms(negative_lock_summary.median), ns_to_ms(negative_lock_summary.average), ns_to_ms(negative_lock_summary.max));
                SPDLOG_INFO(
                    "[perf][epoch {}][gpu {}][negative_sampler_breakdown] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f} state_pool_hits={} planned_uniform_fetches={} cuda_calls={} cpu_calls={}",
                    dataloader_->getEpochsProcessed(), device_idx, ns_to_ms(perf_stats.negative_sampler.device_uniform_randint_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_sample_edge_randint_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_materialize_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_ns[device_idx]),
                    perf_stats.negative_sampler.device_state_pool_hit_count[device_idx],
                    perf_stats.negative_sampler.device_planned_uniform_fetch_count[device_idx],
                    perf_stats.negative_sampler.device_cuda_call_count[device_idx],
                    perf_stats.negative_sampler.device_cpu_call_count[device_idx]);
                SPDLOG_INFO(
                    "[perf][epoch {}][gpu {}][negative_filter_breakdown] deg_chunk_ids_ms={:.3f} deg_mask_ms={:.3f} deg_nonzero_ms={:.3f} deg_gather_ms={:.3f} deg_finalize_ms={:.3f} gpu_prepare_ms={:.3f} gpu_searchsorted_ms={:.3f} gpu_offsets_ms={:.3f} gpu_repeat_interleave_ms={:.3f} gpu_neighbor_gather_ms={:.3f} gpu_relation_filter_ms={:.3f} gpu_finalize_ms={:.3f}",
                    dataloader_->getEpochsProcessed(), device_idx,
                    ns_to_ms(perf_stats.negative_sampler.device_filter_deg_chunk_ids_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_deg_mask_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_deg_nonzero_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_deg_gather_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_deg_finalize_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_prepare_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_searchsorted_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_offsets_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_repeat_interleave_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_neighbor_gather_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_relation_filter_ns[device_idx]),
                    ns_to_ms(perf_stats.negative_sampler.device_filter_gpu_finalize_ns[device_idx]));
            }
        }
        SPDLOG_INFO(
            "[perf][epoch {}][spread] batch_fetch_region_ms={:.3f} gpu_load_region_ms={:.3f} map_region_ms={:.3f} compute_region_ms={:.3f} embedding_update_region_ms={:.3f} embedding_update_g_region_ms={:.3f} dense_sync_wait_ms={:.3f} dense_sync_wait_excl_all_reduce_ms={:.3f} dense_sync_all_reduce_ms={:.3f} finalize_region_ms={:.3f}",
            dataloader_->getEpochsProcessed(), spread_ms(collect_ns(device_timings, &DeviceEpochTiming::batch_fetch_region_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::gpu_load_region_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::map_region_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::compute_region_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::embedding_update_region_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::embedding_update_g_region_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::dense_sync_wait_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::dense_sync_wait_excl_all_reduce_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::dense_sync_all_reduce_ns)),
            spread_ms(collect_ns(device_timings, &DeviceEpochTiming::finalize_region_ns)));
        if (have_device_batch_fetch_stats) {
            SPDLOG_INFO(
                "[perf][epoch {}][spread][batch_fetch] get_next_batch_ms={:.3f} get_next_direct_ms={:.3f} get_next_swap_path_ms={:.3f} get_next_swap_overhead_ms={:.3f} edge_sample_ms={:.3f} node_sample_ms={:.3f} load_cpu_parameters_ms={:.3f} device_prepare_ms={:.3f} perform_map_ms={:.3f} overhead_ms={:.3f}",
                dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_get_next_batch_ns),
                spread_ms(perf_stats.device_get_next_batch_direct_ns), spread_ms(perf_stats.device_get_next_batch_swap_path_ns),
                spread_ms(perf_stats.device_get_next_batch_swap_overhead_ns), spread_ms(perf_stats.device_edge_sample_ns),
                spread_ms(perf_stats.device_node_sample_ns), spread_ms(perf_stats.device_load_cpu_parameters_ns),
                spread_ms(perf_stats.device_get_batch_device_prepare_ns), spread_ms(perf_stats.device_get_batch_perform_map_ns),
                spread_ms(perf_stats.device_get_batch_overhead_ns));
            SPDLOG_INFO(
                "[perf][epoch {}][spread][edge_sample] get_edges_ms={:.3f} negative_sample_ms={:.3f} collect_ids_ms={:.3f} map_lookup_ms={:.3f} compact_active_ms={:.3f} verify_ms={:.3f} remap_assign_ms={:.3f} finalize_ms={:.3f} unaccounted_ms={:.3f}",
                dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_edge_get_edges_ns),
                spread_ms(perf_stats.device_edge_negative_sample_ns), spread_ms(perf_stats.device_edge_map_collect_ids_ns),
                spread_ms(perf_stats.device_edge_map_lookup_ns), spread_ms(perf_stats.device_edge_compact_active_ns),
                spread_ms(perf_stats.device_edge_map_verify_ns), spread_ms(perf_stats.device_edge_remap_assign_ns),
                spread_ms(perf_stats.device_edge_finalize_ns), spread_ms(perf_stats.device_edge_unaccounted_ns));
        }
        if (have_device_negative_sampler_stats) {
            SPDLOG_INFO(
                "[perf][epoch {}][spread][negative_sampler] uniform_randint_ms={:.3f} sample_edge_randint_ms={:.3f} materialize_ms={:.3f} filter_ms={:.3f}",
                dataloader_->getEpochsProcessed(), spread_ms(perf_stats.negative_sampler.device_uniform_randint_ns),
                spread_ms(perf_stats.negative_sampler.device_sample_edge_randint_ns),
                spread_ms(perf_stats.negative_sampler.device_materialize_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_ns));
            SPDLOG_INFO(
                "[perf][epoch {}][spread][negative_filter] deg_chunk_ids_ms={:.3f} deg_mask_ms={:.3f} deg_nonzero_ms={:.3f} deg_gather_ms={:.3f} deg_finalize_ms={:.3f} gpu_prepare_ms={:.3f} gpu_searchsorted_ms={:.3f} gpu_offsets_ms={:.3f} gpu_repeat_interleave_ms={:.3f} gpu_neighbor_gather_ms={:.3f} gpu_relation_filter_ms={:.3f} gpu_finalize_ms={:.3f}",
                dataloader_->getEpochsProcessed(), spread_ms(perf_stats.negative_sampler.device_filter_deg_chunk_ids_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_deg_mask_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_deg_nonzero_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_deg_gather_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_deg_finalize_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_prepare_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_searchsorted_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_offsets_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_repeat_interleave_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_neighbor_gather_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_relation_filter_ns),
                spread_ms(perf_stats.negative_sampler.device_filter_gpu_finalize_ns));
        }
        if (have_device_swap_stats) {
            SPDLOG_INFO("[perf][epoch {}][device] swap_count={}", dataloader_->getEpochsProcessed(),
                        format_vector(perf_stats.device_swap_count));
            SPDLOG_INFO("[perf][epoch {}][device] swap_barrier_wait_ms={}", dataloader_->getEpochsProcessed(),
                        format_ms_vector(perf_stats.device_swap_barrier_wait_ns));
            SPDLOG_INFO("[perf][epoch {}][device] swap_update_ms={}", dataloader_->getEpochsProcessed(),
                        format_ms_vector(perf_stats.device_swap_update_ns));
            SPDLOG_INFO("[perf][epoch {}][device] swap_rebuild_ms={}", dataloader_->getEpochsProcessed(),
                        format_ms_vector(perf_stats.device_swap_rebuild_ns));
            SPDLOG_INFO("[perf][epoch {}][device] swap_sync_wait_ms={}", dataloader_->getEpochsProcessed(),
                        format_ms_vector(perf_stats.device_swap_sync_wait_ns));
            if (!perf_stats.peer_relay.device_peer_bytes_executed.empty()) {
                SPDLOG_INFO("[perf][epoch {}][device] peer_bytes_executed={}", dataloader_->getEpochsProcessed(),
                            format_vector(perf_stats.peer_relay.device_peer_bytes_executed));
                SPDLOG_INFO("[perf][epoch {}][device] host_fallback_bytes={}", dataloader_->getEpochsProcessed(),
                            format_vector(perf_stats.peer_relay.device_host_fallback_bytes));
                if (!perf_stats.peer_relay.device_descriptor_mismatch_count.empty()) {
                    SPDLOG_INFO("[perf][epoch {}][device] descriptor_mismatch_count={}", dataloader_->getEpochsProcessed(),
                                format_vector(perf_stats.peer_relay.device_descriptor_mismatch_count));
                }
                SPDLOG_INFO("[perf][epoch {}][device] peer_sync_wait_ms={}", dataloader_->getEpochsProcessed(),
                            format_ms_vector(perf_stats.peer_relay.device_peer_sync_wait_ns));
            }
            SPDLOG_INFO(
                "[perf][epoch {}][spread] swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_swap_barrier_wait_ns),
                spread_ms(perf_stats.device_swap_update_ns), spread_ms(perf_stats.device_swap_rebuild_ns),
                spread_ms(perf_stats.device_swap_sync_wait_ns));
        }
    }
}
