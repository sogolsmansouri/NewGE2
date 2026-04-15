# LiveJournal Single-GPU Reproduction

This branch carries the single-GPU LiveJournal optimization stack on top of base commit `1ce7580a9f3e36284ba2a7127675635d60c6582d`, plus the graph-storage fast-path fix required to preserve LP accuracy.

## Build

```bash
cd /path/to/dandelion-dev
mkdir -p build_ge2env_ge2py39
cd build_ge2env_ge2py39
cmake ../gege -DCMAKE_BUILD_TYPE=Release
make -j gege_train gege_eval
```

## Reproduce

```bash
cd /path/to/dandelion-dev
./scripts/run_lj_single_gpu_accuracy_gate.sh
```

Useful overrides:

```bash
RUN_NAME=lj_accuracy_gate_30e \
RUN_ROOT=/tmp/smansou2_ge2/lj_accuracy_gate_30e \
LOG_DIR=/tmp/smansou2_ge2/exp_logs \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_lj_single_gpu_accuracy_gate.sh
```

The runner applies the required single-GPU LJ flags:

```text
GEGE_UNIQUE_BACKEND=bitmap
GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
GEGE_EMULATE_DOT_SINGLE_RELATION=1
GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
GEGE_FAST_MAP_TENSORS=1
GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
GEGE_DEG_CHUNK_EXCLUSION=1
GEGE_BUCKET_STREAMING_LP=1
GEGE_BUCKET_BLOCK_EXECUTOR=1
GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
GEGE_CSR_GATHER=0
GEGE_CSR_UPDATE=0
GEGE_EMPTY_CACHE_AROUND_SWAP=0
GEGE_SYNC_BEFORE_SWAP=0
GEGE_MEM_SWAP_EVENT_SYNC=1
GEGE_PROFILE_LOGICAL_LANE=0
GEGE_FIXED_BUFFER_BITMAP_MAP=1
GEGE_FIXED_BUFFER_MASKED_UPDATE=1
GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=1
GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1
GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK=1
GEGE_EVAL_CHUNKED_RANKS=1
GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768
```

Optional defaults left off by the script unless explicitly enabled:

```text
GEGE_NEGATIVE_CHUNK_SAMPLE_CUDA=0
GEGE_NEGATIVE_FUSED_NODE_CHUNK_CUDA=0
GEGE_DEG_LOCAL_FILTER_PADDED=0
GEGE_SCORE_FILTER_CUDA=0
```

## Expected outcome

On a correct build, epoch 1 should land near `9.1s`, later epochs near `8.4s`, and the exact LJ evaluation after 30 epochs should stay close to:

```text
MRR ~= 0.1295
Hits@10 ~= 0.3108
Hits@100 ~= 0.614
```
