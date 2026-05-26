#pragma once

#include "common/datatypes.h"
#include "data/batch.h"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

class Partition {
   public:
    std::mutex *lock_;            /**< Mutex lock to prevent race conditions */
    std::condition_variable *cv_; /**< Condition variable for signaling */
    void *data_ptr_;              /**< Pointer to partition in memory */
    int partition_id_;            /**< ID of the partition */

    bool present_; /**< If true this partition is present in the buffer */

    int64_t partition_size_; /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;     /**< Number of elements in each embedding */
    torch::Dtype dtype_;     /**< Datatype of the embeddings */
    int dtype_size_;         /**< Size in bytes of the datatype */
    int64_t total_size_;     /**< Total size in bytes of the partition */

    int64_t idx_offset_;         /**< Embedding ID offset of the partition */
    int64_t file_offset_;        /**< Offset in bytes of the partition in the embedding file */
    int buffer_idx_;             /**< Buffer entry ID of the partition in the buffer */
    int physical_frame_idx_;     /**< Physical frame backing the logical slot when frame-cache is enabled */

    torch::Tensor tensor_; /**< Tensor view of the partition */

    bool evicting_;

    Partition(int partition_id, int64_t partition_size, int embedding_size, torch::Dtype dtype, int64_t idx_offset, int64_t file_offset);

    ~Partition();

    torch::Tensor indexRead(Indices indices);
};

class PartitionedFile {
   public:
    int num_partitions_;       /**< Number of partitions in the file */
    int64_t partition_size_;   /**< Number of embeddings in each partition, the last partition may have fewer embeddings than this */
    int embedding_size_;       /**< Number of elements in each embedding */
    int64_t total_embeddings_; /**< Total number of embeddings */
    torch::Dtype dtype_;       /**< Datatype of the embeddings */
    int dtype_size_;           /**< Size in bytes of embedding element dtype */
    string filename_;          /**< Name of the backing file */
    int fd_;                   /**< File descriptor for the backing file */

    PartitionedFile(string filename, int num_partitions, int64_t partition_size, int embedding_size, int64_t total_embeddings, torch::Dtype dtype);

    void readPartition(void *addr, Partition *partition);

    void writePartition(Partition *partition, bool clear_mem = true);
};

class LookaheadBlock {
   private:
    std::thread *thread_;
    PartitionedFile *partitioned_file_;
    std::vector<void *> mems_;

    void run();

   public:
    int64_t total_size_;
    std::atomic<bool> present_;
    std::mutex *lock_;
    std::condition_variable cv_;
    std::vector<Partition *> partitions_;
    std::atomic<bool> done_;

    LookaheadBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_lookahead);

    ~LookaheadBlock();

    void start(std::vector<Partition *> first_partitions);

    void stop();

    void move_to_buffer(std::vector<void *> buff_addrs, std::vector<int64_t> buffer_idxs, std::vector<Partition *> next_partitions);
};

class AsyncWriteBlock {
   private:
    std::thread *thread_;
    PartitionedFile *partitioned_file_;
    std::vector<void *> mems_;

    void run();

   public:
    int64_t total_size_;
    std::atomic<bool> present_;
    std::mutex *lock_;
    std::condition_variable cv_;
    std::vector<Partition *> partitions_;
    std::atomic<bool> done_;

    AsyncWriteBlock(int64_t total_size, PartitionedFile *partitioned_file, int num_per_evict);

    ~AsyncWriteBlock();

    void start();

    void stop();

    void async_write(std::vector<Partition *> partitions);
};

class PartitionBuffer {
   protected:
    std::atomic<int64_t> size_;
    int capacity_;
    int num_partitions_;
    int64_t partition_size_;
    int embedding_size_;
    int fine_to_coarse_ratio_;
    int64_t total_embeddings_;
    torch::Dtype dtype_;
    int dtype_size_;

    void *buff_mem_, *buff_mem_unload_;
    void *tensor_mem_;
    bool loaded_;

    Indices in_buffer_ids_;
    torch::Tensor buffer_tensor_view_;
    std::vector<Partition *> partition_table_;

    bool prefetching_;
    LookaheadBlock *lookahead_block_;
    AsyncWriteBlock *async_write_block_;

    string filename_;
    PartitionedFile *partitioned_file_;

    torch::Tensor buffer_state_;
    std::vector<torch::Tensor> buffer_states_;
    std::vector<torch::Tensor>::iterator buffer_state_iterator_;

    torch::Tensor getBufferState();

    void admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs);

    void evict(std::vector<Partition *> evict_partitions);

    void startThreads();

    void stopThreads();

   public:
    PartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                    torch::Dtype dtype, string filename, bool prefetching);

    ~PartitionBuffer();

    void load();

    void write();

    void unload(bool write);

    std::vector<int> getNextAdmit();

    std::vector<int> getNextEvict();

    Indices getRandomIds(int64_t size);

    torch::Tensor indexRead(torch::Tensor indices);

    torch::Tensor getGlobalToLocalMap(bool get_current);

    torch::Tensor getPartitionToBufferSlotMap();

    void indexAdd(torch::Tensor indices, torch::Tensor values);

    void setBufferOrdering(std::vector<torch::Tensor> buffer_states);

    bool hasSwap();

    void performNextSwap();

    void sync();

    int64_t getNumInMemory() { return capacity_ * partition_size_; }

    int64_t getPartitionSize() const { return partition_size_; }
};

struct FrameCachePerfStats {
    int64_t swap_samples = 0;
    int64_t visible_install_parts = 0;
    int64_t visible_install_rows = 0;
    int64_t hidden_publish_parts = 0;
    int64_t hidden_publish_rows = 0;
    int64_t fallback_visible_admit_parts = 0;
    int64_t fallback_visible_admit_rows = 0;
    int64_t preload_miss_swap_count = 0;
    int64_t partial_preload_swap_count = 0;
    int64_t delayed_stale_writeback_swap_count = 0;
    int64_t async_admit_valid_before_swap_count = 0;
    int64_t async_evict_in_flight_before_swap_count = 0;
    int64_t reserved_preload_frames_sum = 0;
    int64_t free_frames_before_swap_sum = 0;
    int64_t free_frames_after_publish_sum = 0;
    int64_t stale_backlog_before_swap_max = 0;
    int64_t stale_backlog_after_publish_max = 0;
};

class MemPartitionBuffer : public PartitionBuffer {
    friend class MemPartitionBufferStorage;

   public:
    MemPartitionBuffer(int capacity, int num_partitions, int fine_to_coarse_ratio, int64_t partition_size, int embedding_size, int64_t total_embeddings,
                       torch::Dtype dtype, string filename, bool prefetching, torch::Device device, int buffer_sizes = 1);

    ~MemPartitionBuffer();

    void load(torch::Tensor data_storage);

    void indexAdd(torch::Tensor indices, torch::Tensor values);

    void indexAddMasked(torch::Tensor indices, torch::Tensor values, torch::Tensor active_mask);

    void performNextSwap(std::uintptr_t swap_ready_event = 0);

    void startAsyncAdmitPreload();

    void evict(std::vector<Partition *> evict_partitions);

    void admit(std::vector<Partition *> admit_partitions, std::vector<int64_t> buffer_idxs);

    torch::Tensor getGlobalToLocalMap(bool get_current);

    void unload(bool write);

    torch::Tensor indexRead(torch::Tensor indices);

    void setBufferOrdering(std::vector<torch::Tensor> buffer_states);

    void setPermutation(torch::Tensor perm, torch::Tensor pos);

    void sync(bool host_staging_current = false);

    bool hasDeviceResidentFrames() const;

    void resetFrameCachePerfStats() { frame_cache_perf_stats_ = FrameCachePerfStats(); }

    FrameCachePerfStats getFrameCachePerfStats() const { return frame_cache_perf_stats_; }

    torch::Tensor data_storage_;

    int buffer_sizes_;

    torch::Device device_ = torch::kCPU;
    torch::Tensor buffer_tensor_gpu_view_;
    torch::Tensor perm_;
    torch::Tensor pos_;

   private:
    void performNextSwapLegacy_(std::uintptr_t swap_ready_event);

    struct HiddenFramePublish {
        int partition_id = -1;
        int64_t logical_slot = -1;
        int64_t frame = -1;
    };

    void ensureBackingTensorsAllocated_();
    void resetFrameCacheState_();
    void refreshFrameCacheTensors_();
    void autoConfigureFrameCacheFromOrdering_();
    bool asyncAdmitPreloadEnabled_() const;
    bool frameCacheEnabled_() const;
    int64_t frameCacheReservedPreloadFrames_() const;
    int64_t frameCacheMaxStaleBacklog_() const;
    int64_t frameCacheStaleBacklogFromFreeFrames_(int64_t free_frames, int64_t preload_reserved_frames) const;
    int64_t logicalSlotToPhysicalFrame_(int64_t logical_slot) const;
    int64_t logicalSlotRowOffset_(int64_t logical_slot) const;
    int64_t partitionRowOffset_(const Partition *partition) const;
    torch::Tensor translateLogicalIndicesToPhysical_(torch::Tensor indices, const char *context = "buffer") const;
    void refreshHostPartitionRanges_();
    bool hasContiguousHostPartitionRange_(int partition_id) const;
    int64_t contiguousHostPartitionStart_(int partition_id) const;
    torch::Tensor hostPartitionRows_(Partition *partition);
    void copyPartitionFromHostToPinned_(Partition *partition, torch::Tensor pinned_view);
    void copyPartitionFromPinnedToHost_(Partition *partition, torch::Tensor pinned_view);
    double copyGpuBufferToHostStaging_();
    void startAsyncAdmitPreloadForPlan_(const std::vector<int> &admit_ids, const std::vector<int64_t> &evict_slots,
                                        int64_t reserved_hidden_frames = 0);
    void releaseReservedHiddenFrames_(const std::vector<HiddenFramePublish> &publishes);
    void joinAsyncAdmitPreload_();
    bool consumeAsyncAdmitPreload_(const std::vector<int> &admit_ids, const std::vector<int64_t> &evict_slots, double *wait_ms,
                                   int64_t *visible_install_rows = nullptr, int64_t *hidden_publish_rows = nullptr,
                                   int64_t *visible_install_parts = nullptr, int64_t *hidden_publish_parts = nullptr);
    struct AsyncEvictWritebackTask {
        std::thread thread;
        bool done = false;
    };
    void reapCompletedAsyncEvictWritebacks_();
    void waitForAsyncEvictWritebackSlot_();
    void joinAsyncEvictWriteback_();
    void joinAsyncEvictWritebackForPartitions_(const std::vector<int> &partition_ids);
    void flushPendingDelayedStaleWriteback_();
    std::vector<std::size_t> prioritizedWritebackOrder_(const std::vector<int> &partition_ids) const;
    void startAsyncEvictWriteback_(const std::vector<int> &evict_ids, const std::vector<int64_t> &row_offsets, torch::Tensor gpu_stage,
                                   torch::Tensor expected_host_stage = torch::Tensor(),
                                   std::vector<int64_t> release_frames = {},
                                   std::vector<int64_t> source_frame_offsets = {},
                                   std::uintptr_t source_ready_event = 0,
                                   bool destroy_source_ready_event = false);
    void ensureDirtyRowMaskAllocated_();
    void markDirtyRows_(torch::Tensor indices, torch::Tensor active_mask = torch::Tensor());
    void clearDirtyRowsForSlots_(const std::vector<int64_t> &slots, const std::vector<int> &partition_ids);
    int64_t countDirtyRowsForSlots_(const std::vector<int64_t> &slots, const std::vector<int> &partition_ids);

    bool use_pinned_host_buffer_ = true;
    int physical_frame_capacity_ = 0;
    int hidden_frame_capacity_ = 0;
    int64_t frame_cache_auto_max_transition_admits_ = 0;
    int64_t frame_cache_auto_max_stale_backlog_ = -1;
    std::vector<int64_t> logical_to_physical_frames_;
    std::vector<int64_t> free_physical_frames_;
    std::mutex free_physical_frames_lock_;
    torch::Tensor logical_to_physical_frame_cpu_;
    torch::Tensor logical_to_physical_frame_device_;
    std::vector<int64_t> partition_host_start_offsets_;
    std::vector<bool> partition_host_contiguous_;
    torch::Tensor dirty_row_mask_;

    std::mutex async_admit_preload_lock_;
    std::thread async_admit_preload_thread_;
    bool async_admit_preload_in_flight_ = false;
    bool async_admit_preload_valid_ = false;
    std::exception_ptr async_admit_preload_exception_ = nullptr;
    std::vector<int> async_admit_preload_admit_ids_;
    std::vector<int64_t> async_admit_preload_evict_slots_;
    std::vector<int64_t> async_admit_preload_row_offsets_;
    torch::Tensor async_admit_preload_gpu_tensor_;
    std::vector<HiddenFramePublish> async_admit_preload_hidden_publishes_;
    std::vector<HiddenFramePublish> pending_hidden_publishes_;
    std::vector<int> pending_delayed_stale_partition_ids_;
    std::vector<int64_t> pending_delayed_stale_row_offsets_;
    std::vector<int64_t> pending_delayed_stale_release_frames_;
    std::vector<int64_t> pending_delayed_stale_source_offsets_;
    std::uintptr_t pending_delayed_stale_ready_event_ = 0;
    bool pending_delayed_stale_destroy_ready_event_ = false;
    double async_admit_preload_host_load_ms_ = 0.0;
    double async_admit_preload_cpu_to_gpu_ms_ = 0.0;
    double async_admit_preload_total_ms_ = 0.0;
    std::mutex host_storage_lock_;
    std::mutex async_evict_writeback_lock_;
    std::condition_variable async_evict_writeback_cv_;
    std::vector<std::unique_ptr<AsyncEvictWritebackTask>> async_evict_writeback_tasks_;
    int async_evict_writeback_active_count_ = 0;
    bool async_evict_writeback_in_flight_ = false;
    std::exception_ptr async_evict_writeback_exception_ = nullptr;
    std::unordered_map<int, int> async_evict_writeback_pending_partition_counts_;
    FrameCachePerfStats frame_cache_perf_stats_;
};
