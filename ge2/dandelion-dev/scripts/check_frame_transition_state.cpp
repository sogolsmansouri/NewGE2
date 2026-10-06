// Diagnostic-only interposition: audit storage against a global update mirror.
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <set>
#include "data/dataloader.h"

namespace {
struct Mirror {
    torch::Tensor value;
    int64_t checks = 0;
};
std::map<MemPartitionBuffer *, Mirror> mirrors;
std::map<void *, torch::Tensor> global_mirrors;
std::map<void *, std::set<MemPartitionBuffer *>> registered_buffers;
std::map<void *, std::set<MemPartitionBuffer *>> synchronized_buffers;
std::map<int32_t, std::pair<int64_t, int64_t>> last_states;
std::mutex audit_lock;
bool injected = false;

void *original(const char *symbol) {
    auto result = dlsym(RTLD_NEXT, symbol);
    TORCH_CHECK(result, "Frame audit cannot resolve ", symbol);
    return result;
}

torch::Tensor host_ids(MemPartitionBuffer *buffer, torch::Tensor local_ids) {
    auto map = buffer->getPartitionToBufferSlotMap().to(torch::kCPU);
    auto slot_to_partition = -torch::ones({4}, map.options());
    for (int64_t partition = 0; partition < map.numel(); ++partition) {
        auto slot = map[partition].item<int64_t>();
        if (slot >= 0) {
            TORCH_CHECK(slot < 4, "Frame audit requires q=4");
            slot_to_partition[slot] = partition;
        }
    }
    TORCH_CHECK((slot_to_partition >= 0).all().item<bool>(), "Incomplete visible slot map");
    auto ids = local_ids.to(torch::kCPU).to(torch::kInt64);
    auto size = buffer->getPartitionSize();
    auto slots = torch::floor_divide(ids, size);
    TORCH_CHECK((slots >= 0).all().item<bool>() && (slots < 4).all().item<bool>(),
                "Local indices outside visible frame capacity");
    auto partitioned_ids = slot_to_partition.index_select(0, slots) * size + torch::remainder(ids, size);
    TORCH_CHECK((partitioned_ids < buffer->data_storage_.size(0)).all().item<bool>(),
                "Small audit fixture must not contain padded entity rows");
    return buffer->pos_.defined() ? buffer->pos_.index_select(0, partitioned_ids) : partitioned_ids;
}

Mirror &mirror(MemPartitionBuffer *buffer) {
    auto &result = mirrors[buffer];
    if (!result.value.defined()) {
        TORCH_CHECK(buffer->data_storage_.size(0) <= 65536,
                    "Frame audit is limited to small correctness fixtures");
        auto key = buffer->data_storage_.data_ptr();
        auto &global = global_mirrors[key];
        if (!global.defined()) global = buffer->data_storage_.clone();
        result.value = global;
    }
    return result;
}

void compare(torch::Tensor expected, torch::Tensor actual, const char *where) {
    actual = actual.to(expected.device());
    TORCH_CHECK(expected.sizes() == actual.sizes(), "Frame audit shape mismatch: ", where);
    auto error = (expected - actual).abs();
    auto mismatches = (expected != actual).sum().item<int64_t>();
    TORCH_CHECK(torch::isfinite(actual).all().item<bool>() && mismatches == 0,
                "FRAME_AUDIT_MISMATCH phase=", where, " values=", mismatches,
                " max_abs=", error.max().item<float>());
}

void compare_visible(MemPartitionBuffer *buffer, const char *where) {
    auto &state = mirror(buffer);
    auto ids = torch::arange(buffer->getNumInMemory(),
                            torch::TensorOptions().dtype(torch::kInt64).device(buffer->device_));
    auto global_ids = host_ids(buffer, ids);
    compare(state.value.index_select(0, global_ids), buffer->indexRead(ids), where);
    ++state.checks;
}

std::pair<MemPartitionBuffer *, MemPartitionBuffer *> buffers(DataLoader *loader, int32_t lane) {
    auto embedding = std::dynamic_pointer_cast<MemPartitionBufferStorage>(loader->graph_storage_->storage_ptrs_.node_embeddings);
    auto optimizer = std::dynamic_pointer_cast<MemPartitionBufferStorage>(loader->graph_storage_->storage_ptrs_.node_optimizer_state);
    TORCH_CHECK(embedding && optimizer && embedding->buffers_.size() == optimizer->buffers_.size(),
                "Frame audit requires matching embedding and optimizer buffers");
    for (auto *storage : {embedding.get(), optimizer.get()})
        for (auto *buffer : storage->buffers_)
            registered_buffers[buffer->data_storage_.data_ptr()].insert(buffer);
    return {embedding->buffers_.at(lane), optimizer->buffers_.at(lane)};
}
}

void DataLoader::loadGPUParameters(shared_ptr<Batch> batch, int32_t device_idx) {
    std::lock_guard<std::mutex> guard(audit_lock);
    TORCH_CHECK(!batch->resident_local_lp_direct_, "Unsupported frame-audit execution mode");
    torch::NoGradGuard no_grad;
    auto [embedding, optimizer] = buffers(this, device_idx);
    mirror(embedding);
    mirror(optimizer);
    auto state = device_current_state_index_.at(device_idx);
    auto inserted = last_states.emplace(device_idx, std::make_pair(-1, -1));
    auto &last = inserted.first->second;
    if (epochs_processed_ != last.first || state != last.second) {
        const char *inject = std::getenv("GEGE_FRAME_AUDIT_INJECT_CORRUPTION");
        if (last.second >= 0 && inject && std::string(inject) == "1" && !injected) {
            auto row = host_ids(embedding, torch::zeros({1}, torch::kInt64))[0].item<int64_t>();
            mirrors.at(embedding).value[row][0].add_(1);
            injected = true;
        }
        compare_visible(embedding, "state-entry-embedding");
        compare_visible(optimizer, "state-entry-adagrad");
        SPDLOG_INFO("[frame-audit-state] epoch={} state={} lane={} embedding=exact adagrad=exact",
                    epochs_processed_, state, device_idx);
        last = {epochs_processed_, state};
    }
    using Function = void (*)(DataLoader *, shared_ptr<Batch>, int32_t);
    reinterpret_cast<Function>(original("_ZN10DataLoader17loadGPUParametersESt10shared_ptrI5BatchEi"))(this, batch, device_idx);
    auto ids = host_ids(embedding, batch->unique_node_indices_);
    compare(mirrors.at(embedding).value.index_select(0, ids), batch->node_embeddings_, "batch-gather-embedding");
    compare(mirrors.at(optimizer).value.index_select(0, ids), batch->node_embeddings_state_, "batch-gather-adagrad");
}

void DataLoader::updateEmbeddings(shared_ptr<Batch> batch, bool gpu, int32_t device_idx) {
    std::lock_guard<std::mutex> guard(audit_lock);
    torch::NoGradGuard no_grad;
    auto [embedding, optimizer] = buffers(this, device_idx);
    if (gpu) {
        auto ids = host_ids(embedding, batch->unique_node_indices_);
        auto update = [&](MemPartitionBuffer *buffer, torch::Tensor delta) {
            if (!delta.defined()) return;
            auto values = delta.detach();
            if (batch->unique_node_active_mask_.defined())
                values = torch::where(batch->unique_node_active_mask_.reshape({-1, 1}), values, torch::zeros_like(values));
            mirrors.at(buffer).value.index_add_(0, ids, values.to(torch::kCPU));
        };
        update(embedding, batch->node_gradients_);
        update(optimizer, batch->node_state_update_);
    }
    using Function = void (*)(DataLoader *, shared_ptr<Batch>, bool, int32_t);
    reinterpret_cast<Function>(original("_ZN10DataLoader16updateEmbeddingsESt10shared_ptrI5BatchEbi"))(this, batch, gpu, device_idx);
    if (gpu) {
        compare_visible(embedding, "batch-store-embedding");
        compare_visible(optimizer, "batch-store-adagrad");
    }
}

void MemPartitionBuffer::sync(bool host_staging_current) {
    using Function = void (*)(MemPartitionBuffer *, bool);
    reinterpret_cast<Function>(original("_ZN18MemPartitionBuffer4syncEb"))(this, host_staging_current);
    std::lock_guard<std::mutex> guard(audit_lock);
    auto found = mirrors.find(this);
    if (found != mirrors.end()) {
        auto key = data_storage_.data_ptr();
        auto &completed = synchronized_buffers[key];
        completed.insert(this);
        if (completed == registered_buffers[key]) {
            compare(found->second.value, data_storage_, "final-host-synchronization");
            SPDLOG_INFO("[frame-audit-host] checks={} host=exact lanes={}", found->second.checks, completed.size());
            completed.clear();
        }
    }
}
