#include "engine/trainer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

#include "configuration/options.h"
#include "reporting/logger.h"
#ifdef GEGE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using std::get;
using std::tie;

namespace {

int64_t elapsed_ns(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

double ns_to_ms(int64_t ns) {
    return static_cast<double>(ns) / 1.0e6;
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

}

SynchronousTrainer::SynchronousTrainer(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model, int logs_per_epoch) {
    dataloader_ = dataloader;
    model_ = model;
    learning_task_ = dataloader_->learning_task_;

    std::string item_name;
    int64_t num_items = 0;
    if (learning_task_ == LearningTask::LINK_PREDICTION) {
        item_name = "Edges";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}

void SynchronousTrainer::train(int num_epochs) {

    if (!dataloader_->single_dataset_) {
        dataloader_->setTrainSet();
    }
    dataloader_->initializeBatches(false);
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    
    Timer timer = Timer(false);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        dataloader_->resetPerfStats();
        timer.start();
        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        while (dataloader_->hasNextBatch()) {
            // gets data and parameters for the next batch
            Timer timer0 = Timer(false);
            timer0.start();

            shared_ptr<Batch> batch = dataloader_->getBatch();

            if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                batch->to(model_->device_);
            } else {
                dataloader_->loadGPUParameters(batch);
            }

            if (batch->node_embeddings_.defined()) {
                batch->node_embeddings_.requires_grad_();
            }

            batch->dense_graph_.performMap();

            model_->train_batch(batch);


            
            if (batch->node_embeddings_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                    batch->embeddingsToHost();
                } else {
                    dataloader_->updateEmbeddings(batch, true);
                }
                dataloader_->updateEmbeddings(batch, false);        
            }

            if (batch->node_embeddings_g_.defined()) {
                if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                    batch->embeddingsToHostG();
                } else {
                    dataloader_->updateEmbeddingsG(batch, true);
                }
                dataloader_->updateEmbeddingsG(batch, false);        
            }

            batch->clear();
            // notify that the batch has been completed
            dataloader_->finishedBatch();

            // log progress
            progress_reporter_->addResult(batch->batch_size_);

        }
        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        timer.stop();
        
        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
        }

        int64_t epoch_time = timer.getDuration();
        float items_per_second = (float)num_items / ((float)epoch_time / 1000);
        SPDLOG_INFO("Epoch Runtime: {}ms", epoch_time);
        SPDLOG_INFO("{} per Second: {}", item_name, items_per_second);

        auto perf_stats = dataloader_->getPerfStats();
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
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
    } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
        item_name = "Nodes";
        num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
    }

    progress_reporter_ = std::make_shared<ProgressReporter>(item_name, num_items, logs_per_epoch);
}



void SynchronousMultiGPUTrainer::train(int num_epochs) {
    if (!dataloader_->single_dataset_) {
	    dataloader_->setTrainSet();
    }

    dataloader_->activate_devices_ = model_->device_models_.size();

    for (int i = 0; i < model_->device_models_.size(); i ++) {
        dataloader_->initializeBatches(false, i);
    }
#ifdef GEGE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    Timer timer = Timer(false); 
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
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
        std::vector<std::thread> threads;

        SPDLOG_INFO("################ Starting training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        SPDLOG_INFO("[perf][epoch {}] dense_sync_batches={} active_devices={}", dataloader_->getEpochsProcessed() + 1, dense_sync_batches,
                    model_->device_models_.size());
        for (int32_t device_idx = 0; device_idx < model_->device_models_.size(); device_idx ++) {
            threads.emplace_back(std::thread([this, &need_sync, &sync_round, &all_reduce_ns, &all_reduce_calls, &device_timings, &sync_batch_counts,
                                              &sync_round_all_reduce_ns, dense_sync_batches, device_idx] {
                int64_t local_batches_since_sync = 0;
                while (dataloader_->hasNextBatch(device_idx)) {
                    // gets data and parameters for the next batch

                    auto batch_fetch_start = std::chrono::high_resolution_clock::now();
                    shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                    auto batch_fetch_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].batch_fetch_region_ns += elapsed_ns(batch_fetch_start, batch_fetch_end);

                    bool has_relation = (batch->edges_.size(1) == 3);

                    auto gpu_load_start = std::chrono::high_resolution_clock::now();
                    dataloader_->loadGPUParameters(batch, device_idx);
                    auto gpu_load_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].gpu_load_region_ns += elapsed_ns(gpu_load_start, gpu_load_end);

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

                    if (batch->node_embeddings_.defined()) {
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

                    if(has_relation) {
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
                    // notify that the batch has been completed
                    dataloader_->finishedBatch(device_idx);
                    auto finalize_end = std::chrono::high_resolution_clock::now();
                    device_timings[device_idx].finalize_region_ns += elapsed_ns(finalize_start, finalize_end);
                    device_timings[device_idx].batch_count++;
                 }
            }));
        }
        for(auto &thread : threads){ thread.join(); }
        // if (model_->device_models_.size() > 1)
        //     model_->all_reduce();

        SPDLOG_INFO("################ Finished training epoch {} ################", dataloader_->getEpochsProcessed() + 1);
        timer.stop();
        // notify that the epoch has been completed
        dataloader_->nextEpoch();
        progress_reporter_->clear();

        std::string item_name;
        int64_t num_items = 0;
        if (learning_task_ == LearningTask::LINK_PREDICTION) {
            item_name = "Edges";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_edges->getDim0();
        } else if (learning_task_ == LearningTask::NODE_CLASSIFICATION) {
            item_name = "Nodes";
            num_items = dataloader_->graph_storage_->storage_ptrs_.train_nodes->getDim0();
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
        if (!have_device_swap_stats &&
            (!perf_stats.device_swap_count.empty() || !perf_stats.device_swap_barrier_wait_ns.empty() ||
             !perf_stats.device_swap_update_ns.empty() || !perf_stats.device_swap_rebuild_ns.empty() ||
             !perf_stats.device_swap_sync_wait_ns.empty())) {
            SPDLOG_WARN("[perf][epoch {}] device swap stats are unavailable or size-mismatched for {} GPU timing entries",
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
        for (int32_t device_idx = 0; device_idx < static_cast<int32_t>(device_timings.size()); device_idx++) {
            const auto &timing = device_timings[device_idx];
            int64_t swap_count = have_device_swap_stats ? perf_stats.device_swap_count[device_idx] : 0;
            int64_t swap_barrier = have_device_swap_stats ? perf_stats.device_swap_barrier_wait_ns[device_idx] : 0;
            int64_t swap_update = have_device_swap_stats ? perf_stats.device_swap_update_ns[device_idx] : 0;
            int64_t swap_rebuild = have_device_swap_stats ? perf_stats.device_swap_rebuild_ns[device_idx] : 0;
            int64_t swap_sync = have_device_swap_stats ? perf_stats.device_swap_sync_wait_ns[device_idx] : 0;
            SPDLOG_INFO(
                "[perf][epoch {}][gpu {}] batches={} sync_points={} batch_fetch_region_ms={:.3f} gpu_load_region_ms={:.3f} map_region_ms={:.3f} compute_region_ms={:.3f} embedding_update_region_ms={:.3f} embedding_update_g_region_ms={:.3f} dense_sync_wait_ms={:.3f} dense_sync_wait_excl_all_reduce_ms={:.3f} dense_sync_all_reduce_ms={:.3f} finalize_region_ms={:.3f} swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), device_idx, timing.batch_count, timing.sync_count, ns_to_ms(timing.batch_fetch_region_ns),
                ns_to_ms(timing.gpu_load_region_ns), ns_to_ms(timing.map_region_ns), ns_to_ms(timing.compute_region_ns), ns_to_ms(timing.embedding_update_region_ns),
                ns_to_ms(timing.embedding_update_g_region_ns), ns_to_ms(timing.dense_sync_wait_ns), ns_to_ms(timing.dense_sync_wait_excl_all_reduce_ns),
                ns_to_ms(timing.dense_sync_all_reduce_ns), ns_to_ms(timing.finalize_region_ns), swap_count,
                ns_to_ms(swap_barrier), ns_to_ms(swap_update), ns_to_ms(swap_rebuild), ns_to_ms(swap_sync));
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
            SPDLOG_INFO(
                "[perf][epoch {}][spread] swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
                dataloader_->getEpochsProcessed(), spread_ms(perf_stats.device_swap_barrier_wait_ns),
                spread_ms(perf_stats.device_swap_update_ns), spread_ms(perf_stats.device_swap_rebuild_ns),
                spread_ms(perf_stats.device_swap_sync_wait_ns));
        }
    }
}
