# Epoch-Time Reproduction Freeze, 2026-04-24

This note freezes the single-GPU epoch-time configurations used for the
April 24 Twitter16P and Freebase86M measurements.

## Build

Use the CUDA build in this checkout:

```bash
cmake --build /home/smansou2/newCode/ge2/dandelion-dev/gege/build-cuda-check -j --target gege_train gege_eval
```

The run script defaults to:

```text
/home/smansou2/newCode/ge2/dandelion-dev/gege/build-cuda-check/gege_train
/home/smansou2/newCode/ge2/dandelion-dev/gege/build-cuda-check/gege_eval
```

## Frozen Files

- Twitter config:
  `ge2/dandelion-dev/gege/configs/repro/twitter_16p_epoch168_20260424.yaml`
- Freebase config:
  `ge2/dandelion-dev/gege/configs/repro/freebase86m_16p_epoch158_20260424.yaml`
- Runner:
  `ge2/dandelion-dev/scripts/run_epoch_time_repro_20260424.sh`

## Runtime Flags

Common flags:

```bash
export CUDA_VISIBLE_DEVICES=0
export GEGE_HYBRID_COVER=1
export GEGE_STATEFLOW_PLANNER=1
export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
export GEGE_FAST_MAP_TENSORS=1
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
export GEGE_DEG_CHUNK_EXCLUSION=1
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_CSR_UPDATE_REDUCE=0
export GEGE_EMPTY_CACHE_AROUND_SWAP=0
export GEGE_SYNC_BEFORE_SWAP=0
export GEGE_MEM_SWAP_EVENT_SYNC=1
export GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1
export GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK=0
export GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1
export GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1
export GEGE_PREPARED_BATCH_PIPELINE=0
export GEGE_PREFETCH_PREPARE_NEXT_PARTITION=0
```

Dataset-specific flags:

```bash
# Twitter
export GEGE_BUCKET_STREAMING_LP=1
export GEGE_FRAME_CACHE_HIDDEN_FRAMES=2
export GEGE_UNIQUE_BITMAP_NUM_NODES=41652230
export GEGE_EMULATE_DOT_SINGLE_RELATION=1
export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=1

# Freebase
export GEGE_BUCKET_STREAMING_LP=0
export GEGE_FRAME_CACHE_HIDDEN_FRAMES=1
export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
export GEGE_EMULATE_DOT_SINGLE_RELATION=0
export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=0
```

The Freebase frozen config intentionally mirrors the GE2 technical-report
Table 4 Freebase DistMult quality target: `DISTMULT`, total embedding width
`100`, `GLOROT_UNIFORM`, encoder `bias: false`, ADAGRAD for dense/relation
parameters, and the RNS+DegreeNS hybrid negative sampler (`degree_fraction:
0.5`). The paper-10k eval helper also defaults to non-bucket/CSR-off Freebase
evaluation. The reference target for the Freebase DistMult row is
approximately `MRR=0.404`, `Hits@10=0.604`. The report also has a separate
Freebase ComplEx row at approximately `MRR=0.438`, `Hits@10=0.612`; do not use
ComplEx when reproducing the `0.404/0.604` DistMult row.

The runner also preloads Torch's packaged `libcudart.so.12` before tcmalloc to
avoid the CUDA runtime symbol mismatch seen with the Conda top-level runtime.

## Commands

Twitter 2-epoch repro:

```bash
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_epoch_time_repro_20260424.sh twitter-2e
```

Freebase 2-epoch repro:

```bash
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_epoch_time_repro_20260424.sh fb-2e
```

Freebase 10-epoch train plus paper-10k exact filtered eval:

```bash
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_epoch_time_repro_20260424.sh fb-10e-eval
```

Useful overrides:

```bash
export GEGE_RUN_NAME=fb86m_distmult_ge2_10e_eval_20260424
export GEGE_RUN_ROOT=/dev/shm/smansou2_ge2/fb86m_distmult_ge2_10e_eval_20260424
export GEGE_LOG_DIR=/home/smansou2/codex_runs/exp_logs
```

## Train log pipeline fingerprint

Each `*_train.log` produced by `run_epoch_time_repro_20260424.sh` starts with a
`gege_epoch_repro_pipeline_fingerprint` banner (the same lines are printed to the
terminal before `gege_train` output). It records the effective values for
`GEGE_PREPARED_BATCH_PIPELINE`, `GEGE_PREFETCH_PREPARE_NEXT_PARTITION`,
`GEGE_PARTITION_BUFFER_PIPELINE_TIMING`, `GEGE_FULL_PIPELINE_PREFETCH`,
`GEGE_BUCKET_STREAMING_LP`, `GEGE_FRAME_CACHE_HIDDEN_FRAMES`, plus the resolved
temp config’s first `prefetch:` and `prefetching:` lines. Use that header when
attributing epoch wall time to subgraph prefetch, prepared-batch pipelining, or
bucket streaming.

## Observed Epoch-Time Results

Twitter current k=2, three 2-epoch repeats:

| log | epoch 2 |
| --- | ---: |
| `twitter_k2_repeat1_2e_20260424_train.log` | `168.249s` |
| `twitter_k2_repeat2_2e_20260424_train.log` | `168.110s` |
| `twitter_k2_repeat3_2e_20260424_train.log` | `167.888s` |

Median epoch 2: `168.110s`.

The remap-validation attribution run:

| log | epoch 2 |
| --- | ---: |
| `twitter_k2_remap_validate_2e_20260424_train.log` | `167.749s` |

This shows `GEGE_REMAP_VALIDATE_SLOTS=1` does not explain the Twitter win.
The structural Twitter change is the hidden preloader using the second hidden
frame after delayed stale writeback, reducing fallback visible admits from
8 parts / 20.8M rows to 4 parts / 10.4M rows.

Freebase p=16/k=1, event-sync run:

| log | epoch 1 | epoch 2 |
| --- | ---: | ---: |
| `fb_k1_repeat1_2e_20260424_train.log` | `163.752s` | `157.796s` |

The Freebase win is primarily from event-based swap synchronization:
`GEGE_SYNC_BEFORE_SWAP=0` and `GEGE_MEM_SWAP_EVENT_SYNC=1`. In the epoch-2
run, `swap_update_ms + swap_rebuild_ms` was `14.232s`, compared with
`54.199s` in the older baseline.
