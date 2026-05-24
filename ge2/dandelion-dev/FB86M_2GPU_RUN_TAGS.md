# Freebase86M 2-GPU Run Tags

This file records the c32/ARC Freebase86M 2-GPU runs that should be treated as
known baselines. Use it before rerunning experiments.

CSV copy:

```bash
/home/smansou2/fb86m_run_tags.csv
```

## Tags

| tag | run | result |
| --- | --- | --- |
| `CORRECT_2GPU_BASELINE` | `fb86m_p32_q4_2gpu_arc_gpu02_3e_eval_correctness_20260520_154953` | Correct eval: MRR `0.477843`, Hits@10 `0.646700`; epoch times `87.056s`, `83.305s`, `83.686s`. |
| `FAST_INVALID_2GPU_EB2103` | `fb86m_p32_q4_2gpu_distmult_ser_nopeerrt_c32_gpu02_10e_20260520_124052` | Fast `78-82s` epochs but eval is invalid: MRR `0.000000`, Hits@10 `0.000000`. |
| `FAST_INVALID_2GPU_FASTSTACK` | `fb86m_p32_q4_2gpu_faststack_c32_gpu01_10e_eval_save_safe_20260519_140931` | Fast `56-61s` epochs but eval was killed and retry reported MRR `0.000000`. |
| `CURRENT_2GPU_PEERAUTO_KEEPHOT_INVALID` | `fb86m_p32_q4_2gpu_bestvalid_peerauto_keep_hot_3e_eval_retry_20260522_141131` | Current rebuilt binary with correctness flags and `KEEP_STORAGE_HOT=1`; fast `97.181s`, `97.071s`, `97.024s`, but eval is invalid: MRR `0.000000`, Hits@10 `0.000000`. |
| `CURRENT_2GPU_REPRO_PEER_AUTO_OOM` | `fb86m_p32_q4_2gpu_gpu02_repro_validsettings_2e_20260522_132751` | Current rebuilt binary, epoch 1 `97.291s`, aborts on epoch reload. |
| `CURRENT_2GPU_REPRO_PEER_OFF_OOM` | `fb86m_p32_q4_2gpu_gpu02_repro_peer_runtime_off_2e_20260522_133501` | Current rebuilt binary, epoch 1 `119.560s`, aborts on epoch reload. |
| `CURRENT_2GPU_REPRO_STABLE_SLOW` | `fb86m_p32_q4_2gpu_gpu02_repro_minimal_hot_2e_20260522_134205` | Current rebuilt binary, stable for 2 epochs with `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1`, but slow: `119.486s`, `119.697s`. |

## What Is Different

Hidden frame count is not the explanation. All of the relevant 2-GPU runs use:

```text
GEGE_FRAME_CACHE_HIDDEN_FRAMES=5
```

The known-correct baseline used:

```text
dense_sync_batches=1
GEGE_DEG_CHUNK_EXCLUSION=0
peer runtime: not enabled in the log
dataset_dir: /mnt/beegfs/smansou2/repos/ge2New/NewGE2/ge2/dandelion-dev/datasets/freebase86m_32p/
edge_min=1206793 edge_max=3651254
```

The fast invalid runs changed the execution path:

```text
dense_sync_batches=8
GEGE_DEG_CHUNK_EXCLUSION=1 for the EB2-103/faststack runs
dataset_dir: /mnt/beegfs/smansou2/repos/ge2New/NewGE2/ge2/datasets/freebase86m_32p_eb2_103/ for EB2-103 runs
```

Those fast runs cannot be used as correctness baselines until a successful eval
matches the expected MRR/Hits.

The current May 22 repros are not the same binary as the May 20 correct run. The
c32 binary at the reused path was rebuilt on May 21:

```text
gege_train sha256: c47066f0a3c61b79c0410bd3e0ea35d99da7c8a703ef89321c10d165fdd8f0cd
libge2.so sha256: 67087e7b247aa159bf15ee8fbb7e627d1925e8eef810f43f708b80a6d848834a
gege_train mtime: 2026-05-21 13:33:47 -0400
libge2.so mtime: 2026-05-21 13:33:45 -0400
```

The correct run executed on May 20, so the executable path alone is not a stable
identifier.

The source YAML for `CORRECT_2GPU_BASELINE` and
`CURRENT_2GPU_REPRO_STABLE_SLOW` is byte-identical. The relevant runtime
differences in the current stable repro are the safety overrides
`GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` and `GEGE_STATEFLOW_PEER_RUNTIME=off`,
plus the rebuilt binary. `save_model` and epoch count differ, but those do not
explain the epoch-body timing delta.

## Current Diagnosis

The current rebuilt binary does not reproduce the May 20 correct timing. The
main timing regression is in the swap path.

Known-correct May 20 baseline, epoch 1:

```text
Epoch Runtime: 87056ms
batch_fetch_region_sum_ms=49465.400
swap_update_ms=27359.586
compute_region_sum_ms=98016.220
embedding_update_region_sum_ms=12490.924
all_reduce_total_ms=4380.421
```

Current May 22 peer-runtime-auto repro, epoch 1:

```text
Epoch Runtime: 97291ms
batch_fetch_region_sum_ms=81498.846
swap_update_ms=59746.121
compute_region_sum_ms=106271.624
embedding_update_region_sum_ms=134.249
all_reduce_total_ms=852.246
```

Current May 22 peer-runtime-auto plus keep-hot repro, full 3-epoch eval:

```text
Epoch Runtime: 97181ms
Epoch Runtime: 97071ms
Epoch Runtime: 97024ms
train_status=0
eval_status=0
MRR: 0.000000
Hits@10: 0.000000
```

This shows that the fastest current peer-runtime path is not a valid
correctness repro, even with the baseline scheduler/config flags and
`GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1`.

The key runtime difference is visible in the peer-copy counters and checkpoint
state:

```text
CORRECT_2GPU_BASELINE:
peer_bytes_executed=0,0
[checkpoint-save] node_embeddings backend=mem_partition_buffer loaded=false device=cuda syncing_host_before_write=1

CURRENT_2GPU_PEERAUTO_KEEPHOT_INVALID:
peer_bytes_executed=21513544000,34421630400
[checkpoint-save] node_embeddings backend=mem_partition_buffer loaded=true device=cuda syncing_host_before_write=1
```

So the current fastest path is not just "baseline plus faster copy"; it changes
the partition-buffer residency/synchronization state at checkpoint time. The
next correctness check should be the slower peer-runtime-off plus keep-hot path,
which avoids the peer relay but should still survive the c32 epoch-boundary
allocation issue.

Current May 22 peer-runtime-off + keep-hot repro, epoch 1:

```text
Epoch Runtime: 119486ms
batch_fetch_region_sum_ms=125962.825
swap_update_ms=105546.043
compute_region_sum_ms=106210.442
embedding_update_region_sum_ms=137.880
all_reduce_total_ms=846.742
```

So the slow repro is not caused by the scheduler producing different state
counts: both the baseline and current repro use `microstates=114`, `rounds=57`,
`total_admitted_objects=228`, and `cross_lane_handoffs=26`. The regression is
that the current binary spends much more time moving/updating partition state.

There is also a separate c32 memory issue at the epoch boundary. Without
`GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1`, the current run tries to allocate
`34421660400` bytes while the `/dev/shm` run root is still consuming memory and
the host commit limit is too low, causing:

```text
DefaultCPUAllocator: can't allocate memory: you tried to allocate 34421660400 bytes
```

Use `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` for multi-epoch repros on c32, but
do not treat that as a performance fix.

## Baseline Rule

For paper numbers and correctness discussions, use `CORRECT_2GPU_BASELINE`
unless a newer run has:

```text
train_status=0
eval_status=0
MRR close to 0.477843
Hits@10 close to 0.646700
binary hash recorded
```

Fast training without a matching eval should be tagged as invalid or unverified.
