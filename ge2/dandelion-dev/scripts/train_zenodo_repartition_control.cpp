// Diagnostic: original GE2 training with its dormant epoch-repartition method.
#include "gege.h"
#include "common/util.h"
#include "configuration/util.h"
#include "engine/trainer.h"
#include "storage/checkpointer.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    try {
        TORCH_CHECK(argc == 3, "Usage: train CONFIG {fixed|repartition}");
        std::string mode(argv[2]);
        TORCH_CHECK(mode == "fixed" || mode == "repartition", "Unknown control mode");
        auto cfg = loadConfig(argv[1], true);
        TORCH_CHECK(devices_from_config(cfg->storage).size() == 1, "Single GPU only");
        TORCH_CHECK(!cfg->training->resume_training && cfg->training->resume_from_checkpoint.empty(), "Fresh training only");
        TORCH_CHECK(cfg->evaluation->epochs_per_eval > cfg->training->num_epochs,
                    "Use the separate fixed-query evaluator after training");
        auto initialized = gege_init(cfg, true);
        auto model = std::get<0>(initialized);
        auto storage = std::get<1>(initialized);
        auto loader = std::get<2>(initialized);
        auto mem = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage->storage_ptrs_.node_embeddings);
        TORCH_CHECK(mem && mem->options_->buffer_capacity == 4 && mem->options_->num_partitions == 16,
                    "This experiment requires the released p16/q4 recipe");
        TORCH_CHECK(std::dynamic_pointer_cast<InMemory>(storage->storage_ptrs_.train_edges), "CPU in-memory edges required");
        auto trainer = std::make_shared<SynchronousTrainer>(loader, model, cfg->training->logs_per_epoch);
        double repartition_seconds = 0;
        for (int epoch = 0; epoch < cfg->training->num_epochs; ++epoch) {
            if (epoch && mode == "repartition") {
                auto start = std::chrono::steady_clock::now();
                storage->rePartition();
                double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                repartition_seconds += seconds;
                std::cout << "REPARTITION before_epoch=" << epoch + 1 << " seconds=" << seconds << std::endl;
            }
            trainer->train(1);
        }
        CheckpointMeta metadata;
        metadata.num_epochs = loader->epochs_processed_;
        metadata.has_state = true;
        metadata.has_encoded = false;
        metadata.has_model = true;
        metadata.link_prediction = true;
        auto saver = std::make_shared<Checkpointer>(model, storage, cfg->training->checkpoint);
        saver->save(cfg->storage->model_dir, metadata);
        std::cout << "CONTROL_COMPLETE mode=" << mode << " epochs=" << metadata.num_epochs
                  << " repartition_seconds=" << repartition_seconds << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
