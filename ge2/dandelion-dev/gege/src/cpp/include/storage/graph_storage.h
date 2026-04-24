#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#ifdef GEGE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "configuration/constants.h"
#include "storage/storage.h"

struct GraphModelStoragePtrs {
    shared_ptr<Storage> edges = nullptr;
    shared_ptr<Storage> train_edges = nullptr;
    shared_ptr<Storage> validation_edges = nullptr;
    shared_ptr<Storage> test_edges = nullptr;
    shared_ptr<Storage> nodes = nullptr;
    shared_ptr<Storage> train_nodes = nullptr;
    shared_ptr<Storage> valid_nodes = nullptr;
    shared_ptr<Storage> test_nodes = nullptr;
    shared_ptr<Storage> node_features = nullptr;
    shared_ptr<Storage> node_labels = nullptr;
    shared_ptr<Storage> relation_features = nullptr;
    shared_ptr<Storage> relation_labels = nullptr;
    shared_ptr<Storage> node_embeddings = nullptr;
    shared_ptr<Storage> node_embeddings_g = nullptr;
    shared_ptr<Storage> encoded_nodes = nullptr;
    shared_ptr<Storage> node_optimizer_state = nullptr;
    shared_ptr<Storage> node_optimizer_state_g = nullptr;
    // N-ary arity-4: qualifier value embeddings always resident in GPU memory (InMemory storage)
    shared_ptr<Storage> qual_embeddings = nullptr;
    shared_ptr<Storage> qual_optimizer_state = nullptr;
    std::vector<shared_ptr<Storage>> filter_edges;
};

struct InMemorySubgraphState {
    EdgeList all_in_memory_edges_;
    EdgeList all_in_memory_mapped_edges_;
    torch::Tensor in_memory_partition_ids_;
    torch::Tensor in_memory_edge_bucket_ids_;
    torch::Tensor in_memory_edge_bucket_sizes_;
    torch::Tensor in_memory_edge_bucket_starts_;
    torch::Tensor global_to_local_index_map_;
    shared_ptr<GegeGraph> in_memory_subgraph_;
};

class GraphModelStorage {
   private:
    void _load(shared_ptr<Storage> storage);

    void _unload(shared_ptr<Storage> storage, bool write);

    bool shouldUsePartitionBufferLPFastPath_();

    torch::Tensor getPartitionToBufferSlotMap_(int32_t device_idx = 0);

    int64_t getPartitionSize_(int32_t device_idx = 0);

    torch::Tensor getGlobalToLocalMapForValidation_(bool get_current, int32_t device_idx = 0);

    torch::Tensor mapEdgesWithDenseMap_(torch::Tensor edges, torch::Tensor global_to_local_index_map, torch::Device device);

    torch::Tensor mapEdgesWithPartitionSlots_(torch::Tensor edges, torch::Tensor partition_to_buffer_slot, int64_t partition_size,
                                              torch::Device device);

    void startAsyncAdmitPreload_(int32_t device_idx = 0);

    int64_t num_nodes_;
    int64_t num_edges_;
    bool partition_buffer_lp_fast_path_enabled_;

   protected:
    bool train_;
    bool configured_full_graph_evaluation_;

    shared_ptr<InMemory> in_memory_embeddings_;
    shared_ptr<InMemory> in_memory_features_;

   public:
    // In memory subgraph for partition buffer

    std::vector<EdgeList> active_edges_;
    std::vector<torch::Device> devices_;
    Indices active_nodes_;
    torch::Tensor perm_;

    std::mutex *subgraph_lock_;
    std::condition_variable *subgraph_cv_;
    shared_ptr<InMemorySubgraphState> current_subgraph_state_;
    std::vector<shared_ptr<InMemorySubgraphState>> current_subgraph_states_;
    shared_ptr<InMemorySubgraphState> next_subgraph_state_;
    bool prefetch_;
    bool prefetch_complete_;

    GraphModelStoragePtrs storage_ptrs_;
    bool full_graph_evaluation_;

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, shared_ptr<StorageConfig> storage_config);

    GraphModelStorage(GraphModelStoragePtrs storage_ptrs, bool prefetch = false);

    ~GraphModelStorage();

    void load();

    void load_g();

    void unload(bool write);

    void initializeInMemorySubGraph(torch::Tensor buffer_state, torch::Device device = torch::kCPU, int32_t device_idx = 0);

    void updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids, int32_t device_idx = 0);

    void updateInMemorySubGraph(int32_t device_idx = 0);

    void getNextSubGraph(int32_t device_idx = 0);

    /**
     * Wait until the background prefetch builder has finished populating `next_subgraph_state_`.
     * Only meaningful when `prefetch_` is enabled and a prefetch is in flight.
     */
    bool waitForSubgraphPrefetchComplete(const std::atomic<bool> *stop_flag = nullptr);

    void notifySubgraphPrefetchWaiters();

    /**
     * Snapshot of the prefetched next in-memory subgraph state (may be nullptr if not ready).
     * Callers must not mutate the returned object.
     */
    shared_ptr<InMemorySubgraphState> getPrefetchedNextSubgraphStateSnapshot() const;

    EdgeList merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src);

    void setEdgesStorage(shared_ptr<Storage> edge_storage);

    void setNodesStorage(shared_ptr<Storage> node_storage);

    EdgeList getEdges(Indices indices, int32_t device_idx = 0);

    EdgeList getEdgesRange(int64_t start, int64_t size, int32_t device_idx = 0);

    EdgeList getCurrentStateMappedEdgesRange(int64_t start, int64_t size, int32_t device_idx = 0) {
        if (device_idx < 0 || static_cast<std::size_t>(device_idx) >= current_subgraph_states_.size() ||
            current_subgraph_states_[device_idx] == nullptr || !current_subgraph_states_[device_idx]->all_in_memory_mapped_edges_.defined()) {
            return torch::Tensor();
        }

        return current_subgraph_states_[device_idx]->all_in_memory_mapped_edges_.narrow(0, start, size);
    }

    EdgeList getMappedEdgesRangeForState(const shared_ptr<InMemorySubgraphState> &state, int64_t start, int64_t size) {
        if (state == nullptr || !state->all_in_memory_mapped_edges_.defined()) {
            return torch::Tensor();
        }
        return state->all_in_memory_mapped_edges_.narrow(0, start, size);
    }

    Indices getRandomNodeIds(int64_t size);

    Indices getNodeIdsRange(int64_t start, int64_t size);

    void shuffleEdges();

    torch::Tensor getNodeEmbeddings(Indices indices, int32_t device_idx = 0);

    torch::Tensor getNodeEmbeddingsG(Indices indices, int32_t device_idx = 0);

    torch::Tensor getNodeEmbeddingsRange(int64_t start, int64_t size);

    torch::Tensor getNodeFeatures(Indices indices);

    torch::Tensor getNodeFeaturesRange(int64_t start, int64_t size);

    torch::Tensor getEncodedNodes(Indices indices);

    torch::Tensor getEncodedNodesRange(int64_t start, int64_t size);

    torch::Tensor getNodeLabels(Indices indices);

    torch::Tensor getNodeLabelsRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddings(Indices indices, torch::Tensor values);

    void updateAddNodeEmbeddings(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    void updateAddNodeEmbeddingsMasked(Indices indices, torch::Tensor values, torch::Tensor active_mask, int32_t device_idx = 0);

    void updateAddNodeEmbeddingsG(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    void updatePutEncodedNodes(Indices indices, torch::Tensor values);

    void updatePutEncodedNodesRange(int64_t start, int64_t size, torch::Tensor values);

    OptimizerState getNodeEmbeddingState(Indices indices, int32_t device_idx = 0);

    OptimizerState getNodeEmbeddingStateG(Indices indices, int32_t device_idx = 0);

    OptimizerState getNodeEmbeddingStateRange(int64_t start, int64_t size);

    void updatePutNodeEmbeddingState(Indices indices, OptimizerState state);

    void updateAddNodeEmbeddingState(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    void updateAddNodeEmbeddingStateMasked(Indices indices, torch::Tensor values, torch::Tensor active_mask, int32_t device_idx = 0);

    void updateAddNodeEmbeddingStateG(Indices indices, torch::Tensor values, int32_t device_idx = 0);

    bool embeddingsOffDevice();

    bool embeddingsOffDeviceG();

    void sortAllEdges(int32_t device_idx = 0);

    int getNumPartitions() {
        int num_partitions = 1;

        if (useInMemorySubGraph()) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
                num_partitions = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->options_->num_partitions;
            }

            // assumes both the node features and node embeddings have the same number of partitions
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                num_partitions = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->options_->num_partitions;
            }
            if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                num_partitions = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->options_->num_partitions;
            }
        }

        return num_partitions;
    }

    void rePartition() {
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) { 
            auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
            torch::Tensor perm = torch::randperm(getNumNodes(), opts);
            auto tup = torch::sort(perm); 
            torch::Tensor pos = std::get<1>(tup);
            std::dynamic_pointer_cast<InMemory>(storage_ptrs_.edges)->rePartition(perm, getNumNodes(), getNumPartitions());
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->rePartition(perm, pos);
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->rePartition(perm, pos);
        }
    }

    bool useInMemorySubGraph() {
        bool embeddings_buffered = instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings);
        embeddings_buffered = embeddings_buffered || instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings);
        bool features_buffered = instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features);

        return (embeddings_buffered || features_buffered) && (train_ || (!full_graph_evaluation_));
    }

    void setPartitionBufferLPFastPathEnabled(bool enabled) { partition_buffer_lp_fast_path_enabled_ = enabled; }

    bool partitionBufferLPFastPathEnabled() { return shouldUsePartitionBufferLPFastPath_(); }

    void resetFrameCachePerfStats() {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->resetFrameCachePerfStats();
        }
    }

    void setStateflowPeerHandoffs(const std::vector<PeerHandoffDescriptor> &peer_handoffs) {
        auto configure_storage = [&](const shared_ptr<Storage> &storage) {
            if (storage != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage)) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage)->setStateflowPeerHandoffs(peer_handoffs);
            }
        };
        configure_storage(storage_ptrs_.node_embeddings);
        if (train_) {
            configure_storage(storage_ptrs_.node_optimizer_state);
        }
        configure_storage(storage_ptrs_.node_embeddings_g);
        if (train_) {
            configure_storage(storage_ptrs_.node_optimizer_state_g);
        }
    }

    void clearStateflowPeerHandoffs() { setStateflowPeerHandoffs({}); }

    void resetPeerRelayPerfStats() {
        auto reset_storage = [&](const shared_ptr<Storage> &storage) {
            if (storage != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage)) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage)->resetPeerRelayPerfStats();
            }
        };
        reset_storage(storage_ptrs_.node_embeddings);
        if (train_) {
            reset_storage(storage_ptrs_.node_optimizer_state);
        }
        reset_storage(storage_ptrs_.node_embeddings_g);
        if (train_) {
            reset_storage(storage_ptrs_.node_optimizer_state_g);
        }
    }

    void resetStateflowPeerRuntimeProgress() {
        auto reset_storage = [&](const shared_ptr<Storage> &storage) {
            if (storage != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage)) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage)->resetStateflowTransitionCounts();
            }
        };
        reset_storage(storage_ptrs_.node_embeddings);
        if (train_) {
            reset_storage(storage_ptrs_.node_optimizer_state);
        }
        reset_storage(storage_ptrs_.node_embeddings_g);
        if (train_) {
            reset_storage(storage_ptrs_.node_optimizer_state_g);
        }
    }

    PeerRelayPerfStats getPeerRelayPerfStats() const {
        PeerRelayPerfStats stats;
        auto accumulate_storage = [&](const shared_ptr<Storage> &storage) {
            if (storage == nullptr || !instance_of<Storage, MemPartitionBufferStorage>(storage)) {
                return;
            }
            auto current = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage)->getPeerRelayPerfStats();
            auto accumulate_vector = [](std::vector<int64_t> &dst, const std::vector<int64_t> &src) {
                if (src.empty()) {
                    return;
                }
                if (dst.size() < src.size()) {
                    dst.resize(src.size(), 0);
                }
                for (std::size_t idx = 0; idx < src.size(); idx++) {
                    dst[idx] += src[idx];
                }
            };
            stats.peer_bytes_executed += current.peer_bytes_executed;
            stats.host_fallback_bytes += current.host_fallback_bytes;
            stats.peer_copy_count += current.peer_copy_count;
            stats.host_fallback_count += current.host_fallback_count;
            stats.descriptor_mismatch_count += current.descriptor_mismatch_count;
            stats.peer_sync_wait_ns += current.peer_sync_wait_ns;
            accumulate_vector(stats.device_peer_bytes_executed, current.device_peer_bytes_executed);
            accumulate_vector(stats.device_host_fallback_bytes, current.device_host_fallback_bytes);
            accumulate_vector(stats.device_peer_copy_count, current.device_peer_copy_count);
            accumulate_vector(stats.device_host_fallback_count, current.device_host_fallback_count);
            accumulate_vector(stats.device_descriptor_mismatch_count, current.device_descriptor_mismatch_count);
            accumulate_vector(stats.device_peer_sync_wait_ns, current.device_peer_sync_wait_ns);
        };
        accumulate_storage(storage_ptrs_.node_embeddings);
        if (train_) {
            accumulate_storage(storage_ptrs_.node_optimizer_state);
        }
        accumulate_storage(storage_ptrs_.node_embeddings_g);
        if (train_) {
            accumulate_storage(storage_ptrs_.node_optimizer_state_g);
        }
        return stats;
    }

    FrameCachePerfStats getFrameCachePerfStats(int32_t device_idx = 0) const {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getFrameCachePerfStats(device_idx);
        }
        return FrameCachePerfStats();
    }

    std::vector<FrameCachePerfStats> getFrameCachePerfStatsAll() const {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getFrameCachePerfStatsAll();
        }
        return {};
    }

    bool hasSwap(int32_t device_idx = 0) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->hasSwap();
            }
            if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->hasSwap(device_idx);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->hasSwap();
        }

        return false;
    }

    std::pair<std::vector<int>, std::vector<int>> getNextSwapIds(int32_t device_idx = 0) {
        std::vector<int> evict_ids;
        std::vector<int> admit_ids;

        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            evict_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextEvict();
            admit_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextAdmit();
        } else if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            evict_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNextEvict();
            admit_ids = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNextAdmit();
        } else if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            evict_ids = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextEvict(device_idx);
            admit_ids = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNextAdmit(device_idx);
        }

        return std::make_pair(evict_ids, admit_ids);
    }

    void performSwap(int32_t device_idx = 0) {
        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->performNextSwap();
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->performNextSwap();
            }
        }

        if (storage_ptrs_.node_embeddings != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            std::uintptr_t swap_ready_event_handle = 0;
            auto read_env_flag = [](const char *name, bool default_value) {
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
            };
#ifdef GEGE_CUDA
            cudaEvent_t swap_ready_event = nullptr;
            bool swap_event_sync = read_env_flag("GEGE_MEM_SWAP_EVENT_SYNC", true);
            bool global_swap_sync = read_env_flag("GEGE_SYNC_BEFORE_SWAP", true);
            if (swap_event_sync && !global_swap_sync && device_idx >= 0 &&
                static_cast<std::size_t>(device_idx) < devices_.size() && devices_[device_idx].is_cuda()) {
                c10::cuda::CUDAGuard guard(devices_[device_idx]);
                AT_CUDA_CHECK(cudaEventCreateWithFlags(&swap_ready_event, cudaEventDisableTiming));
                AT_CUDA_CHECK(cudaEventRecord(swap_ready_event, c10::cuda::getCurrentCUDAStream(devices_[device_idx].index()).stream()));
                swap_ready_event_handle = reinterpret_cast<std::uintptr_t>(swap_ready_event);
            }
#endif
            auto embedding_storage = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings);
            std::shared_ptr<MemPartitionBufferStorage> optimizer_storage = nullptr;
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                optimizer_storage = std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state);
            }
            bool serialize_mem_swaps = read_env_flag("GEGE_STATEFLOW_SERIALIZE_MEM_SWAPS", false);

            std::vector<std::thread> threads;
            std::exception_ptr thread_exception = nullptr;
            std::mutex thread_exception_lock;
            auto run_mem_swap = [&](std::shared_ptr<MemPartitionBufferStorage> storage) {
                try {
                    storage->performNextSwap(device_idx, swap_ready_event_handle);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(thread_exception_lock);
                    if (thread_exception == nullptr) {
                        thread_exception = std::current_exception();
                    }
                }
            };

            if (optimizer_storage == nullptr) {
                run_mem_swap(embedding_storage);
            } else if (serialize_mem_swaps) {
                run_mem_swap(embedding_storage);
                if (optimizer_storage != nullptr && thread_exception == nullptr) {
                    run_mem_swap(optimizer_storage);
                }
            } else {
                threads.push_back(std::thread(run_mem_swap, embedding_storage));
                if (optimizer_storage != nullptr) {
                    threads.push_back(std::thread(run_mem_swap, optimizer_storage));
                }
                for(auto& thread : threads) {
                    thread.join();
                }
            }
#ifdef GEGE_CUDA
            if (swap_ready_event != nullptr) {
                AT_CUDA_CHECK(cudaEventDestroy(swap_ready_event));
            }
#endif
            if (thread_exception != nullptr) {
                std::rethrow_exception(thread_exception);
            }

            // std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->performNextSwap(device_idx);
            // if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
            //     std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->performNextSwap(device_idx);
            // }
        }

        if (storage_ptrs_.node_embeddings_g != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)->performNextSwap(device_idx);
            if (storage_ptrs_.node_optimizer_state_g != nullptr && train_) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state_g)->performNextSwap(device_idx);
            }
        }

        if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->performNextSwap();
        }
    }

    void setBufferOrdering(vector<torch::Tensor> buffer_states) {
        if (storage_ptrs_.node_embeddings != nullptr && (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings))) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_embeddings != nullptr && (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings))) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state != nullptr && train_) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_embeddings_g != nullptr && (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g))) {
            std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)->setBufferOrdering(buffer_states);
            if (storage_ptrs_.node_optimizer_state_g != nullptr && train_) {
                std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_optimizer_state_g)->setBufferOrdering(buffer_states);
            }
        }
        if (storage_ptrs_.node_features != nullptr && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->setBufferOrdering(buffer_states);
        }
    }

    void setActiveEdges(torch::Tensor active_edges, int32_t device_idx) { 
        active_edges_[device_idx] = active_edges;
    }

    void setActiveNodes(torch::Tensor node_ids) { active_nodes_ = node_ids; }

    int64_t getNumActiveEdges(int device_idx = 0) {
        if (active_edges_[device_idx].defined()) {
            return active_edges_[device_idx].size(0);
        } else {
            return storage_ptrs_.edges->getDim0();
        }
    }

    int64_t getNumActiveNodes() {
        if (active_nodes_.defined()) {
            return active_nodes_.size(0);
        } else {
            return storage_ptrs_.nodes->getDim0();
        }
    }

    int64_t getNumEdges() { return storage_ptrs_.edges->getDim0(); }

    int64_t getNumNodes() {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return storage_ptrs_.node_embeddings->getDim0();
        }

        if (storage_ptrs_.node_features != nullptr) {
            return storage_ptrs_.node_features->getDim0();
        }

        return num_nodes_;
    }

    int64_t getNumNodesInMemory(int32_t device_idx = 0) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (useInMemorySubGraph()) {
                if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings))
                    return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNumInMemory();
                if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings))
                    return std::dynamic_pointer_cast<MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)->getNumInMemory(device_idx);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (useInMemorySubGraph()) {
                return std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getNumInMemory();
            }
        }

        return getNumNodes();
    }

    void setTrainSet() {
        train_ = true;
        full_graph_evaluation_ = configured_full_graph_evaluation_;
        
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            storage_ptrs_.node_embeddings->device_ = torch::kCUDA;
        }

        if (storage_ptrs_.node_embeddings_g != nullptr && instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings_g)) {
            storage_ptrs_.node_embeddings_g->device_ = torch::kCUDA;
        }

        if (storage_ptrs_.train_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.train_edges);
        }

        if (storage_ptrs_.train_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.train_nodes);
        }
    }

    void setValidationSet() {
        train_ = false;
        full_graph_evaluation_ = configured_full_graph_evaluation_;

        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            full_graph_evaluation_ = true;
        }

        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            storage_ptrs_.node_embeddings->device_ = torch::kCPU;
        }

        if (storage_ptrs_.validation_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.validation_edges);
        }

        if (storage_ptrs_.valid_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.valid_nodes);
        }
    }

    void setTestSet() {
        train_ = false;
        full_graph_evaluation_ = configured_full_graph_evaluation_;

        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            full_graph_evaluation_ = true;
        }
        
        if (instance_of<Storage, MemPartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
            storage_ptrs_.node_embeddings->device_ = torch::kCPU;
        }

        if (storage_ptrs_.test_edges != nullptr) {
            setEdgesStorage(storage_ptrs_.test_edges);
        }

        if (storage_ptrs_.test_nodes != nullptr) {
            setNodesStorage(storage_ptrs_.test_nodes);
        }
    }

    void setFilterEdges(std::vector<shared_ptr<Storage>> filter_edges) { storage_ptrs_.filter_edges = filter_edges; }

    void addFilterEdges(shared_ptr<Storage> filter_edges) { storage_ptrs_.filter_edges.emplace_back(filter_edges); }
};
