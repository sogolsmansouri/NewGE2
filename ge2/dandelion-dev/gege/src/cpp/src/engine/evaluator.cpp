#include "engine/evaluator.h"

#include "configuration/constants.h"
#include "reporting/logger.h"

#include <thread>


SynchronousEvaluator::SynchronousEvaluator(shared_ptr<DataLoader> dataloader, shared_ptr<Model> model) {
    dataloader_ = dataloader;
    model_ = model;
}

void SynchronousEvaluator::evaluate(bool validation) {
    if (!dataloader_->single_dataset_) {
        if (validation) {
            dataloader_->setValidationSet();
        } else {
            dataloader_->setTestSet();
        }
    }

    int32_t num_devices = 1;
    if (dataloader_->graph_storage_->useInMemorySubGraph()) {
        num_devices = std::max<int32_t>(static_cast<int32_t>(dataloader_->devices_.size()), 1);
    }

    for (int32_t device_idx = 0; device_idx < num_devices; device_idx++) {
        dataloader_->initializeBatches(false, device_idx);
    }

    if (dataloader_->evaluation_negative_sampler_ != nullptr) {
        if (dataloader_->evaluation_config_->negative_sampling->filtered) {
            for (int32_t device_idx = 0; device_idx < num_devices; device_idx++) {
                dataloader_->graph_storage_->sortAllEdges(device_idx);
            }
        }
    }
    model_->reporter_->clear();
    Timer timer = Timer(false);
    timer.start();
    std::vector<std::thread> threads;
    threads.reserve(num_devices);
    for (int32_t device_idx = 0; device_idx < num_devices; device_idx++) {
        threads.emplace_back([this, device_idx]() {
            shared_ptr<Model> eval_model = model_;
            if (device_idx < model_->device_models_.size() && model_->device_models_[device_idx] != nullptr) {
                eval_model = model_->device_models_[device_idx];
            }

            while (dataloader_->hasNextBatch(device_idx)) {
                shared_ptr<Batch> batch = dataloader_->getBatch(c10::nullopt, false, device_idx);
                if (batch == nullptr) {
                    break;
                }
                batch->to(eval_model->device_);
                dataloader_->loadGPUParameters(batch, device_idx);
                batch->dense_graph_.performMap();
                eval_model->evaluate_batch(batch);
                dataloader_->finishedBatch(device_idx);
                batch->clear();
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    timer.stop();

    model_->reporter_->report();
}
