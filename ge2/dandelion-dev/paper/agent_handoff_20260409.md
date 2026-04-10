# GE² Optimization Handoff — Agent-Optimized Reference
**Date:** 2026-04-09  
**Purpose:** Compact reference for agents picking up this work. Read this before reading the full design documents.

---

## Context

**System:** GE² (Graph Embedding squared) — temporal knowledge-graph embedding trainer.  
**Training loop:** each epoch cycles over `q=4` visible partition states. Each state boundary triggers a `swap_update` (evict old partitions, admit new). Within each state, batches are fetched via `edgeSample()`, which runs the negative sampler and the compact-index mapper (`map_tensors`).  
**Datasets in play:**
- **LJ** — small, fits comfortably; used as a correctness/regression gate
- **Twitter q4** — large; current single-GPU baseline is `223.230 s`
- **FB86M q4** — large; current single-GPU baseline `164.098 s`

**Key bottlenecks on Twitter (nested accounting):**
```
223.085 s total
  115.190 s  batch_fetch
    61.603 s  swap_update
    53.530 s  edge_sample
      44.000 s  map_lookup
       7.930 s  negative_sample
       1.600 s  other
  107.895 s  compute/other
```

**Key bottlenecks on FB86M:**
```
164.098 s total
  153.515 s  batch_fetch
    65.485 s  swap_update
    86.620 s  map_lookup
     0.544 s  negative_sample
```

**Constraint:** must preserve exact `q=4` visible-state semantics and pass LJ accuracy gate (MRR `~0.1295`, Hits@10 `~0.311`).  
**Hardware:** single 24 GB GPU (A5000 or similar). Multi-GPU (2/4) is a downstream target.

**Primary design documents:**
- [epoch_time_optimization_record_20260407.md](/home/smansou2/newCode/ge2/dandelion-dev/paper/epoch_time_optimization_record_20260407.md) — full experimental log
- [versioned_frame_cache_design_20260408.md](/home/smansou2/newCode/ge2/dandelion-dev/paper/versioned_frame_cache_design_20260408.md) — storage layer design
- [state_compiled_runtime_design_20260409.md](/home/smansou2/newCode/ge2/dandelion-dev/paper/state_compiled_runtime_design_20260409.md) — runtime layer design

---

## Validated Baselines

| Dataset | Config | Epoch (s) | swap_update (s) | map_lookup (s) | negative_sample (s) |
|---------|--------|-----------|-----------------|----------------|---------------------|
| LJ | q4, 1 GPU | 8454.5 ms | — | — | — |
| Twitter | q4, 1 GPU | 223.230 | 59.414 | 43.508 | 9.107 |
| FB86M | q4, 1 GPU | 164.098 | 65.485 | 86.620 | 0.544 |

**LJ accuracy reference (30 epochs, exact eval):**
- MRR: `0.129537`
- Hits@10: `0.3108`

These numbers must not regress when new code paths are active. LJ is the gating test.

---

## Tried and Failed Hypotheses

### Dead — do not revisit without a fundamentally new mechanism

| Hypothesis | Result | Why it failed |
|------------|--------|---------------|
| `q=5/6` greedy cover scheduling | Twitter `199s` train-only, but LJ MRR drops to `0.075` | Changes negative visible domain; accuracy collapses |
| Active-tile / liveness-aware negatives (DEG+local) | LJ MRR `0.050724` | Restricts negative support without correction; breaks training |
| Dirty-row / dirty-tile writeback | Twitter dirties `99.1%` of visible state per state; state-0 unique delta `7.87 GiB` vs full `7.94 GiB` | Near-zero savings under current negative sampler |
| Full async staging (LJ-style) | OOM on FB/Twitter; FB needed `~12.31 GiB` extra HBM | Does not fit on 24 GB for large datasets |
| CPU-stage async evict / lower async memory margin / prefetch CPU remap | Slower or unstable | Did not move the real wall |
| `StateLocalMapEngine` v1 (smaller domain in existing bitmap mapper) | Twitter `43.508 → 196.376 s map_lookup`; total `223 → 327 s` | Same algorithm, just smaller domain — wrong cost model |
| `StatePositiveBatchTape` (compile positive unique+mappings, merge negatives at runtime) | Twitter `223 → 333 s`; `swap_rebuild 53 ms → 6.9 s`; per-state tape `1.35–1.87 GiB` | Re-enters old mapper; tape too large |
| `StatePositiveUniqueTape` (sorted positive uniques + negative miss merge) | Twitter state tapes `353–363 MiB`; run invalid when `/dev/shm` filled | Smaller metadata but still not a win |
| Resident-local direct LP (bypass compact mapper entirely) | LJ `17403 ms`, `map_lookup = 0`, but Twitter at `10%` already projecting many minutes | Compute/update cost explodes because embedding-row reuse is lost |
| Existing `sort` unique backend | Twitter projects to `~310 s` | Backend-only change is not enough |
| Existing negative-pool refresh (W=2) | Runtime and sampled-eval unchanged on LJ | Implementation too weak; no effect |

### Real wins (incorporated or available)

| Win | Before | After | Notes |
|-----|--------|-------|-------|
| FB86M padded CUDA DEG filter | `negative_sample ~36.871 s` | `0.584 s` | Already in code |
| Versioned frame-cache: hidden preload + delayed stale writeback | Twitter `223.230 → 215.207 s`; FB `164.098 → 154.289 s` | swap_update: Twitter `59.4 → 52.6 s`, FB `65.5 → 53.3 s` | Real but not enough alone |
| LJ partial async swap | `~9.5 s → ~8.5 s steady-state` | — | Valid; does not transfer to FB/Twitter |

---

## Implemented Code Paths

### Storage / frame-cache prototype
- [buffer.h](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/include/storage/buffer.h)
- [buffer.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/storage/buffer.cpp)

Status: hidden-only preload + delayed stale writeback is implemented and measured. The prototype proves the direction is real. The next version must not execute hidden-frame logic on every ordinary batch read/gather — the hot path must stay in pure logical q4 coordinates. **`swap_update` must actually drop; if it stays flat, the boundary-removal has not been implemented.**

Key constraint from boundary kill-test: on Twitter q4, a full hidden partition only fit for the first two swaps; remaining swaps fell back because free HBM stayed near `~1470 MiB` while the preload needed `~3010–4003 MiB`. Full-partition `k=1` is not viable as a sustained mechanism.

### State-compiled runtime experiments
- [batch.h](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/include/data/batch.h)
- [batch.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/batch.cpp)
- [dataloader.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/dataloader.cpp)
- [util.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/common/util.cpp)
- [unique_map_cuda.cu](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cuda/src/common/unique_map_cuda.cu)

Status: multiple tape/mapper shapes tried. All rejected so far (see failed hypotheses). The key existing structure that **does** work is `GraphModelStorage::updateInMemorySubGraph_()` in `graph_storage.cpp`, which already compiles state-local edge structures once per visible state — the hot path in `DataLoader::edgeSample()` is still fully dynamic per batch on top of that.

### Analyzers and simulators
- [gege_state_compiled_runtime_analyzer.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/gege_state_compiled_runtime_analyzer.cpp) — offline workspace/tape sizing
- [state_compiled_runtime_sim.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/state_compiled_runtime_sim.py)
- [versioned_frame_cache_sim.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/versioned_frame_cache_sim.py)
- [contrastive_tiled_cover_sim.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/contrastive_tiled_cover_sim.py)
- [liveness_negative_cache_sim.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/liveness_negative_cache_sim.py)

---

## Open Problems

### Problem 1: `map_lookup` wall on Twitter and FB
`map_lookup` is `43.508 s` on Twitter and `86.620 s` on FB. Every per-batch mapper experiment so far either preserved the cost or made it worse. The state-compiled workspace analysis says the positive-only workspace is small (`0.078 GiB` Twitter, `0.161 GiB` FB), but no correct implementation has beaten the baseline yet.

**What is not yet tried:** a generation-stamped state-local mapper with a new algorithmic cost model (proportional to touched IDs in batch, not to domain size). All prior attempts either used the same bitmap algorithm in a smaller domain, or tried to precompile full tapes.

### Problem 2: `swap_update` on Twitter and FB is not sufficiently hidden
The frame-cache prototype proved the direction is real but the current prototype does not sustain hidden preloads through a full epoch on Twitter. The next storage step requires either tiled hidden frames or a stronger delayed-commit / frame-reclamation policy.

### Problem 3: negative tape size vs negative quality
Fully materialized negative IDs are too large: Twitter `0.998–1.362 GiB` per state, FB `0.187–0.287 GiB`. But changing the negative domain breaks accuracy. The valid path is descriptor/seed replay — stores the RNG seed and sampler parameters, not the sampled IDs. This has not been prototyped yet.

### Problem 4: no single fix is enough for Twitter
Simulation residual-cost sweep says:
- Twitter aggressive case (map to 10%, neg to 0%, swap to 25%): `~130.6 s`
- Twitter moderate case (map to 25%, neg to 25%, swap to 35%): `~145.3 s`
- All three layers are required to move Twitter materially; none alone is sufficient.

---

## Most Promising Next Experiment

**Prototype: `StateLocalMapEngine` v2 — generation-stamped, touched-ID proportional**

This is the highest-upside next prototype because `map_lookup` is the single largest unexplained wall on FB (`86.620 s`) and the second largest on Twitter (`43.508 s`), and no correct implementation has attacked it yet.

### What to build

Per device, per visible state, allocate:
```
StateLocalMapWorkspace
  uint32 seen_generation[num_nodes_in_memory]   // zeroed once at state publish
  int32  compact_index[num_nodes_in_memory]      // filled incrementally per batch
  int32  unique_ids_buf[max_batch_touched_ids]   // scratch for this batch
  int32  inverse_buf[max_batch_touched_ids]      // scratch for this batch
  uint32 current_generation                      // incremented each batch
```

Per batch, the mapper:
1. flatten positive + negative local IDs for this batch
2. for each ID: if `seen_generation[id] != current_generation`, assign a new compact row, write `compact_index[id]`, bump counter
3. fill inverse tensors from `compact_index[id]`
4. increment `current_generation`

Key difference from all prior failed attempts: this does not use a bitmap algorithm. The cost is proportional to touched IDs per batch, not to domain size. It avoids clearing any large structure per batch.

### Sizing (from analyzer)
- Twitter: `num_nodes_in_memory ~10.7M` → `seen_generation` = `43 MiB`, `compact_index` = `43 MiB`; total workspace = `~0.078 GiB`
- FB86M: `num_nodes_in_memory ~21.6M` → total workspace = `~0.161 GiB`
- Both fit on GPU without memory pressure.

### Acceptance criteria
1. LJ safety gate passes (MRR within noise of `0.129537`)
2. Twitter `map_lookup` drops materially from `43.508 s`
3. FB86M `map_lookup` drops materially from `86.620 s`
4. No change to visible-state semantics, negative domain, or batch order

### After this prototype passes
Build `NegativeDescriptorTape` next: store per-batch `(rng_seed_hi, rng_seed_lo, num_chunks, num_uniform, num_degree)` bound to the state version, replay on GPU. This is seed-only replay, not materialized negative IDs. After both map and negative replay are validated, integrate with the tiled frame-cache storage layer.

### What to avoid
- Do not reuse the existing padded bitmap mapper as the inner kernel — that is what `StateLocalMapEngine` v1 did and it regressed
- Do not materialize full negative ID tapes — too large for Twitter
- Do not change q or the visible partition set — accuracy gate will fail
- Do not try to compile the mapper across states — the generation table resets at each state publish
