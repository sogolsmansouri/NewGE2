#include "engine/trainer.h"

#include <chrono>

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
        std::atomic<bool> sync_finished = false;
        std::atomic<int64_t> sync_wait_ns = 0;
        std::atomic<int64_t> all_reduce_ns = 0;
        std::atomic<int64_t> all_reduce_calls = 0;
        std::vector<std::atomic<int64_t>> sync_batch_counts(model_->device_models_.size());
        for (auto &count : sync_batch_counts) {
            count.store(0);
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
            threads.emplace_back(std::thread([this, &need_sync, &sync_finished, &sync_wait_ns, &all_reduce_ns, &all_reduce_calls, &sync_batch_counts, dense_sync_batches, device_idx] {
                int64_t local_batches_since_sync = 0;
                while (dataloader_->hasNextBatch(device_idx)) {
                    // gets data and parameters for the next batch

                    shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                    bool has_relation = (batch->edges_.size(1) == 3);
                    dataloader_->loadGPUParameters(batch, device_idx);

                    if (batch->node_embeddings_.defined()) {
                        batch->node_embeddings_.requires_grad_();
                    }

                    batch->dense_graph_.performMap();

                    model_->device_models_[device_idx]->train_batch(batch, false);

                    if (batch->node_embeddings_.defined()) {
                        if (dataloader_->graph_storage_->embeddingsOffDevice()) {
                            batch->embeddingsToHost();
                        } else {
                            dataloader_->updateEmbeddings(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddings(batch, false, device_idx);
                    }

                    if (batch->node_embeddings_g_.defined()) {
                        if (dataloader_->graph_storage_->embeddingsOffDeviceG()) {
                            batch->embeddingsToHostG();
                        } else {
                            dataloader_->updateEmbeddingsG(batch, true, device_idx);
                        }
                        dataloader_->updateEmbeddingsG(batch, false, device_idx);
                    }


                    // if(has_relation) {
                    //     if (dataloader_->batches_left_[device_idx] == 1) {
                    //         sync_finished = false;
                    //         need_sync ++;

                    //         if (need_sync == dataloader_->activate_devices_) {
                    //             model_->all_reduce_rel();
                    //             sync_finished = true;
                    //             need_sync = 0;
                    //         }
                    //         while (!sync_finished) {
                    //             std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    //         }
                    //     }
                    // }

                    if(has_relation) {
                        local_batches_since_sync++;
                        bool should_sync = (local_batches_since_sync >= dense_sync_batches) || (dataloader_->batches_left_[device_idx] == 1);
                        if (should_sync) {
                            auto sync_start = std::chrono::high_resolution_clock::now();
                            sync_finished.store(false);
                            sync_batch_counts[device_idx].store(local_batches_since_sync);
                            int64_t arrivals = need_sync.fetch_add(1) + 1;

                            if (arrivals == dataloader_->activate_devices_.load()) {
                                auto all_reduce_start = std::chrono::high_resolution_clock::now();
                                std::vector<int64_t> grad_scales(sync_batch_counts.size(), 1);
                                for (int i = 0; i < sync_batch_counts.size(); i++) {
                                    grad_scales[i] = std::max<int64_t>(sync_batch_counts[i].load(), 1);
                                }
                                model_->all_reduce(grad_scales);
                                all_reduce_ns.fetch_add(elapsed_ns(all_reduce_start, std::chrono::high_resolution_clock::now()));
                                all_reduce_calls.fetch_add(1);
                                sync_finished.store(true);
                                need_sync.store(0);
                            }
                            while (!sync_finished.load()) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                            sync_wait_ns.fetch_add(elapsed_ns(sync_start, std::chrono::high_resolution_clock::now()));
                            local_batches_since_sync = 0;
                        }
                    }
                    
                    batch->clear();
                    // notify that the batch has been completed
                    dataloader_->finishedBatch(device_idx);
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
        SPDLOG_INFO(
            "[perf][epoch {}] dense_sync_batches={} all_reduce_calls={} dense_sync_wait_ms={:.3f} all_reduce_ms={:.3f} swap_count={} swap_barrier_wait_ms={:.3f} swap_update_ms={:.3f} swap_rebuild_ms={:.3f} swap_sync_wait_ms={:.3f}",
            dataloader_->getEpochsProcessed(), dense_sync_batches, all_reduce_calls.load(), ns_to_ms(sync_wait_ns.load()), ns_to_ms(all_reduce_ns.load()),
            perf_stats.swap_count, ns_to_ms(perf_stats.swap_barrier_wait_ns), ns_to_ms(perf_stats.swap_update_ns), ns_to_ms(perf_stats.swap_rebuild_ns),
            ns_to_ms(perf_stats.swap_sync_wait_ns));
    }
}
