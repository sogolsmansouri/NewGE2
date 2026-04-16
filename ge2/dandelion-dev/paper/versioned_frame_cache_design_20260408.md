# Versioned Frame Cache for GE²

## Goal

Reduce epoch time on large datasets, especially Twitter and FB86M, by replacing GE²'s exposed state-boundary swap with a memory-bounded, asynchronous publish/commit pipeline that preserves the current logical `q=4` training semantics.

This design is aimed at the bottleneck that still remains on 1, 2, and 4 GPUs:

- large exposed `swap_update`
- state-boundary synchronization and barriers
- repeated host-mediated movement of embeddings and optimizer state

It is intentionally not another visible-state scheduler such as `q=5/6`, because those paths already showed accuracy regressions when the negative domain changed.

## Why The Existing Fixes Helped LJ But Not FB/Twitter

LJ improved because:

- its state footprint is small enough that extra staging fits in HBM
- its old batch-construction path had removable synchronization overhead
- partial async swap is enough to expose a large gain

FB/Twitter did not improve the same way because:

- full-stage async swap needs multi-GiB extra HBM and does not fit safely
- Twitter still has a very large exposed `map_lookup` wall
- both Twitter and FB still spend a large fraction of epoch time in state movement even after local fixes

Measured examples:

- Twitter 1 GPU: `223.085s = 107.895s compute/other + 115.190s batch_fetch`
- Twitter 1 GPU batch fetch: `61.603s swap_update + 53.530s edge_sample + 0.057s overhead`
- Twitter 4 GPU best still has `19.984s swap_update` and `1.442s swap_barrier_wait`
- FB86M 4 GPU best still has `11.816s swap_update` and `6.503s swap_barrier_wait`

So the real problem is not "the copy kernel is slow." The problem is that GE² still treats partition movement as a visible state-boundary action.

## Design Summary

The proposal is:

**Logical q4 state + hidden physical frames + publish-by-remap + delayed commit**

The key separation is:

- logical state: what training and the sampler are allowed to see
- physical frames: where partition data actually resides in HBM

The sampler-visible domain stays exactly as it is today. Only the storage/movement policy changes.

## Paper Framing And Novelty Boundary

The cleanest contribution is **not** "a better COVER" and **not** "general buffer management for KGE."

The precise claim is:

- `COVER` solves positive-bucket scheduling under a model where partition movement is still a visible state-boundary action.
- The remaining large-system bottleneck is **residency management**, not the positive-bucket order itself.
- `Versioned Frame Cache (VFC)` attacks that layer by separating:
  - logical visible q-state
  - physical frame residency
  - publish
  - commit

So the right VLDB framing is:

**GE² / COVER solves scheduling. VFC solves residency management under exact q-state semantics.**

This is stronger and more defensible than saying GE² "does not have a buffer manager" in the abstract. The concrete point is that GE²'s current model still exposes partition movement at state boundaries, while VFC turns that into:

- asynchronous prefetch into hidden capacity
- publish-by-remap
- deferred versioned commit

without changing the sampler-visible domain.

## Current-Main Reality Check

The older `223s` Twitter q4 baseline is still useful for the first frame-cache design and simulator calibration, but the current `main` tree is already faster than that baseline.

Controlled Twitter q4 ablation on current `main` (`947234e`, single GPU, same config, epoch-2 steady state) is:

- `k=0`: `222.085 s`, `swap_update_ms=52.991 s`, `map_lookup_ms=43.113 s`
- `k=1`: `213.398 s`, `swap_update_ms=43.828 s`, `map_lookup_ms=43.144 s`
- `k=2`: `206.909 s`, `swap_update_ms=37.449 s`, `map_lookup_ms=42.902 s`

This gives the clean current interpretation:

- VFC is already a real net-positive runtime optimization on `main`.
- `k=1` improves epoch time by about `8.687 s` versus `k=0`.
- `k=2` improves epoch time by about `15.176 s` versus `k=0`, and by about `6.489 s` versus `k=1`.
- The gain comes almost entirely from reduced `swap_update`.
- `map_lookup` stays essentially flat at about `43 s`, so VFC alone is not the whole Twitter answer.

So the honest end-to-end story on current code is:

1. VFC already attacks exposed movement and boundary synchronization successfully.
2. A separate state-local or descriptor-based mapping path is still needed to attack `map_lookup`.
3. Only after those two layers move materially should further sampler/liveness changes be revisited.

## Core Idea

Today:

1. train state `S`
2. evict visible partitions
3. admit next partitions
4. publish next state
5. train state `S+1`

Proposed:

1. train logical state `S`
2. while `S` trains, prefetch future partitions into hidden physical frames
3. at boundary, publish `S+1` by remapping logical slots to already-loaded frames
4. old frames become stale-but-owned frames and are committed later
5. commits happen on-demand and asynchronously, not synchronously at the boundary

This removes "install by copy" from the critical path and replaces it with "publish by pointer/frame-table update."

## Correctness Invariants

The design must preserve these invariants:

1. **Logical visibility invariant**
   Training for state `S` only reads from the current logical visible slots. Hidden prefetched frames are not visible to positive edges or negative sampling until the state boundary publish.

2. **Version invariant**
   Every physical frame has a `(partition_id, version_id, dirty)` tuple. Host storage and peer GPUs also have a version for each partition or tile. A newer version may not be overwritten or published as older data.

3. **Publish invariant**
   A logical slot can point to a prefetched hidden frame only after the admit data is complete and all required metadata for the next state is ready.

4. **Commit invariant**
   A stale visible frame can be reclaimed only after its dirty updates are safely committed or transferred to the next owner.

5. **Sampler invariant**
   Negative sampling support must remain the same as the current validated q4 configuration unless an explicitly corrected algorithm is introduced and revalidated.

## Data Structures

### 1. Logical Slot Table

Per device:

- `logical_slot -> physical_frame`
- size `q`
- used by batch construction, embedding gather, update, and the negative sampler

### 2. Physical Frame Table

Per device:

- `frame_id`
- `partition_id`
- `version_id`
- `state = {visible, hidden_prefetched, stale_dirty, stale_clean, free}`
- `last_writer_epoch`
- `bytes_dirty` or tile bitmap

Number of frames:

- baseline visible frames: `q`
- extra hidden/stale frames: `k`
- total physical frames: `q + k`

For Twitter, `k=1..3` full-partition frames is plausible.  
For FB86M, full-partition frames are too large; use tile frames instead of full-partition frames.

### 3. Commit Log

Per frame or tile:

- list or bitmap of modified row ranges
- optimizer-state dirty metadata
- destination owner: host or peer GPU

This allows delayed commit instead of immediate full writeback.

### 4. State Publish Record

For each future state:

- target logical slots
- target partition IDs
- required prefetched frames or tiles
- batch metadata readiness
- ready event handle

## Execution Model

### Current GE² Figure-3 Style

```text
Load -> Train -> Dump
```

### Proposed Task Graph

```text
PrefetchFrame(partition/tile)
BuildBatchMetadata(state)
TrainBatch(state, batch)
SealFrame(frame)
CommitFrameDelta(frame)
PublishState(state+1)
```

Dependencies:

- `PrefetchFrame` must finish before `PublishState`
- `TrainBatch` for state `S` must finish before the old visible frame is sealed
- `CommitFrameDelta` can run after a frame is sealed; it does not need to block publish if a hidden prefetched frame is already available
- `BuildBatchMetadata` for `S+1` should overlap with training of `S`

## What This Bypasses

### Swap Update

Current `swap_update` is dominated by:

- evict D2H
- admit H2D
- install/copy into current visible slots
- training-thread-visible synchronization

The design removes or reduces:

- visible install copies at the boundary
- immediate full writeback requirement
- CPU-thread blocking on transfer completion

It does **not** assume the copy disappears. It assumes the copy becomes hidden and decoupled.

### Why This Is Better Than Full Async Staging

Full async staging tried to create a complete shadow visible state. That failed on FB/Twitter because the extra memory footprint was too large.

This design is better because:

- it makes extra memory explicit and bounded
- it supports full-partition or tile frames
- it does not require a complete second visible state
- it can reclaim stale frames lazily

## Single-GPU Prototype Order

### Prototype 1: Twitter, full-partition hidden frame, no delayed commit policy

Goal:

- prove that publish-by-remap removes visible admit/install cost

Rules:

- keep logical `q=4`
- add `k=1` hidden frame
- prefetch one future partition into the hidden frame
- at boundary, remap one logical slot to the hidden frame
- old visible frame becomes stale and is written back immediately after publish, but off the critical path

This is the cleanest first version because Twitter has already shown some headroom for hidden admit.

### Prototype 2: Twitter, `k=2/3`, delayed commit queue

Goal:

- overlap multiple future admits and allow more stale frames to drain in the background

### Prototype 3: FB86M tiled frames

Goal:

- replace full-partition frames with tile frames because FB full extra partition frames do not fit

Rules:

- tile size fixed and memory-bounded
- logical visibility still at q4 partition level
- publish happens when all required tiles for a logical slot are ready

## Multi-GPU Extension

This design is intentionally multi-GPU extensible.

### 2/4 GPU ownership model

Each partition or tile has:

- owner GPU or host
- version
- clean/dirty state

On publish:

- if local hidden frame is ready, publish locally
- else if a peer GPU owns a newer clean version, relay from peer
- else fetch from host

This integrates naturally with the already-successful peer-relay direction.

### Why it extends better than single-GPU-only staging

- the logical visible domain does not change with GPU count
- the frame/version model naturally expresses peer ownership
- barrier time can be attacked by making publish local and delaying commit
- host traffic can be replaced with peer traffic where available

### Concrete 2/4 GPU Extension Plan

The 2/4 GPU version should not be "single-GPU hidden frames copied N times." The correct extension is a **distributed frame/version protocol**.

Per partition or tile, maintain:

- current logical owner set
- physical owner (`host` or one GPU)
- version id
- clean / dirty state
- commit destination

Per GPU, maintain:

- local visible logical slot table
- local hidden frame pool
- local stale-frame queue
- relay eligibility for peer-owned clean versions

#### 2 GPU path

Primary goals:

- reduce host-visible swap time
- convert some host admits into peer relay
- reduce barrier exposure by making publish cheap

Execution shape:

1. while state `S` trains, each GPU prefetches future local admits into hidden local frames when possible
2. if the next needed version already exists clean on the peer, use peer relay rather than host fetch
3. at publish, remap local logical slots to ready hidden frames or relayed frames
4. old visible frames become stale and enter delayed commit
5. commit drains to host or peer owner after publish, not before publish

The first 2-GPU prototype should stay conservative:

- only one hidden local frame per GPU
- relay only clean peer frames
- no mixed ownership of one logical slot at publish
- fall back to the current synchronous path when readiness is incomplete

#### 4 GPU path

The 4-GPU extension should use the same metadata model, but the system objective shifts slightly:

- barrier exposure matters more
- peer relay matters more
- host fallback should become less frequent on hot transitions

Execution shape:

1. prefetch locally where hidden capacity exists
2. if a needed partition/tile exists clean on another GPU, relay from peer instead of host
3. publish is still local remap, not barrier-sized install copy
4. stale commit is decoupled and drains asynchronously after publish

The key paper point at 4 GPUs is not only lower host traffic. It is that **publish becomes cheaper than the current state-boundary synchronization model**, so swap barriers shrink together with transfer exposure.

#### Multi-GPU invariants

The extra invariants beyond single GPU are:

1. a published local frame must point to the newest ready version visible to that GPU
2. peer relay may only consume a newer clean version, never an in-flight dirty one
3. delayed commit must preserve optimizer-state consistency together with embedding values
4. fallback must preserve the current logical q-state exactly when relay/prefetch readiness is incomplete

#### Multi-GPU measurements to expose

For 2/4 GPU evaluation, add:

- local publish count
- peer-relay publish count
- host-fallback publish count
- barrier wait before and after publish
- stale backlog depth per GPU
- hidden-frame occupancy per GPU
- fallback rate when hidden capacity is exhausted

## Relationship To Prior Systems

This is closest in spirit to:

- Marius: overlap movement with compute for graph embeddings
- BagPipe / ScratchPipe: future-aware embedding cache and lookahead decisions
- Herald: on-demand synchronization and "train in cache" philosophy for sparse embeddings
- vDNN / ZeRO-Infinity: dependency-aware offload/prefetch and memory-centric scheduling

What is different here:

- logical q4 semantics are preserved exactly
- the scheduled object is not only "future partition load" but a **versioned visible/invisible frame**
- state publish happens by remap, not by install copy
- commit is delayed and version-controlled, not forced at every state boundary

## What We Already Have In Code

This is not a greenfield design anymore. The current tree already contains a partial single-GPU frame-cache prototype:

- hidden-frame flags and plumbing in `buffer.cpp`
- logical-to-physical slot remap state
- hidden publish bookkeeping
- delayed stale-frame writeback
- boundary instrumentation for `visible_install_rows` and `hidden_publish_rows`

So the engineering task is no longer "invent frame cache." The task is:

1. keep the hot path purely logical
2. formalize frame/version/dirty metadata
3. make publish a true remap-only boundary step
4. make delayed commit a first-class bounded subsystem
5. extend the same abstraction to multi-GPU relay/ownership

## Memory Feasibility By Dataset

The central memory fact is that each extra hidden full-partition frame is paid twice:

- one extra partition for embeddings
- one extra partition for optimizer state

For `p=16`, `dim=100`, `float32`, the extra resident memory per hidden partition pair is approximately:

- Twitter: `~1.94 GiB`
- FB86M: `~4.01 GiB`

The visible `q=4` resident pair memory is approximately:

- Twitter: `~7.76 GiB`
- FB86M: `~16.03 GiB`

Implications:

- Twitter full-partition hidden frames are plausible:
  - `k=1`: about `9.70 GiB`
  - `k=2`: about `11.64 GiB`
  - `k=3`: about `13.58 GiB`
  before activations, mapper workspace, allocator fragmentation, and other runtime buffers
- FB86M full-partition hidden frames do **not** scale on 24 GB:
  - `k=1`: about `20.04 GiB`
  - `k=2`: about `24.04 GiB`
  before other runtime memory

This matches the runtime evidence:

- FB86M full-partition `k=1` can be made useful in the lean path
- FB86M full-partition `k=2` is a hard OOM on 24 GB

Therefore the correct physical strategy is:

- Twitter: full hidden partitions first
- FB86M: tiled hidden capacity, not full hidden partitions, as the real path

This is not a weakness of the architecture. It is the correct dataset-specific realization of the same VFC abstraction.

## Risks

1. **Commit backlog**
   If stale frames accumulate faster than they drain, hidden frame capacity is exhausted. The scheduler then falls back to the current synchronous boundary path.

2. **Optimizer-state consistency**
   Optimizer state must travel with embeddings. Partial commit policies must maintain exact optimizer semantics.

3. **Map/batch-construction wall remains**
   This design primarily attacks swap. Twitter still needs a state-local or tape-based mapper for the `map_lookup` wall.

4. **Memory fragmentation**
   Tile frames and variable-sized metadata must use a pool allocator or pre-sized arenas.

## What This Does Not Solve Alone

This is not the whole `223s -> 100s` answer for Twitter.

Even if swap becomes mostly hidden, Twitter still has a large `map_lookup` wall. So the full path is:

1. versioned frame cache to attack exposed movement
2. state-local or taped batch mapping to attack exposed remap/build
3. only then revisit sampler/liveness improvements if needed

## Updated Implementation Plan

The practical implementation order on the current tree should be:

### Phase 1: Lock down Twitter `k=1`

Goal:

- make the current hidden-only preload + delayed stale commit path reproducible, stable, and fully instrumented on current `main`

Required properties:

- no hidden-frame work in steady-state ordinary reads/updates
- publish reduces exposed boundary work rather than shifting cost downstream
- delayed stale commit has bounded backlog and explicit fallback

Outputs to log per swap:

- `visible_install_rows`
- `hidden_publish_rows`
- delayed stale rows / frames
- backlog depth
- fallback count / fallback rows

Current status:

- done for single-GPU Twitter q4 on `main`
- measured epoch-2 result: `222.085 s (k=0) -> 213.398 s (k=1)`
- measured frame-cache behavior:
  - `hidden_publish_parts=19`
  - `fallback_visible_admit_parts=38`
  - `preload_miss_swaps=0`

Interpretation:

- `k=1` is mechanically correct and already materially useful
- the path is capacity-limited rather than preload-limited
- one hidden frame hides exactly 1 of 3 admits per swap on Twitter q4

### Phase 2: Twitter `k=2/3`

Goal:

- turn the `k=1` correctness/storage prototype into a meaningful Twitter paper result

Rules:

- only continue past `k=1` if the hot path stays clean
- add stronger frame reclamation / backlog control before scaling hidden depth
- treat `k=2/3` as a storage-layer experiment, not a scheduler change

Current status:

- `k=2` is now validated on single-GPU Twitter q4 on `main`
- measured epoch-2 result: `206.909 s`
- measured swap result: `swap_update_ms=37.449 s`
- measured frame-cache behavior:
  - `hidden_publish_parts=38`
  - `fallback_visible_admit_parts=19`
  - `preload_miss_swaps=0`

Interpretation:

- VFC scales in the expected direction on Twitter
- the main remaining independent wall is still `map_lookup ~= 43 s`
- the next Twitter storage question is whether `k=3` keeps the same monotonic improvement without introducing new backlog or memory-pressure pathologies

### Phase 3: FB86M bridge experiment with lean `k=1`

Goal:

- retain the evidence that even one hidden full-partition frame plus delayed stale commit moves FB materially

This is a bridge experiment, not the final FB design.

### Phase 4: FB86M tiled hidden frames

Goal:

- realize the same VFC abstraction under FB memory limits

Requirements:

- fixed tile size
- pool allocator / pre-sized arena
- tile-level dirty metadata
- publish only when the required tile set for a logical slot is ready
- no row-level steady-state indirection in ordinary batch execution

### Phase 5: Pair VFC with the next map-layer system

Goal:

- attack the remaining large Twitter `map_lookup` wall

The current recommendation remains:

1. VFC for exposed movement
2. then a state-local or descriptor-based mapping layer

Twitter needs both layers. FB can still benefit materially even if only part of the current map bucket is removable.

## Immediate Engineering Checklist

Near-term work against the current tree:

1. add explicit memory-budget logging and impossible-config guards at startup
2. harden backlog / fallback accounting in the frame-cache swap path
3. make Twitter `k=1` a stable reproducible path on current `main`
4. add 2/4 GPU metrics for local publish, peer relay, host fallback, and barrier exposure
5. reintroduce FB hidden-frame experiments only through the lean `k=1` path
6. implement tiled hidden frames for FB before attempting larger FB hidden capacity

## Expected Impact

For Twitter single GPU:

- current controlled `main` ablation: `222.085s (k=0) -> 213.398s (k=1) -> 206.909s (k=2)`
- current remaining large walls at `k=2`: `swap_update ~= 37.4s`, `map_lookup ~= 42.9s`
- immediate next Twitter target is to keep reducing exposed swap without polluting the hot path, then attack the still-flat `map_lookup` wall separately

For FB86M single GPU:

- current best: `166.4s`
- main expected gain is from tiled hidden frames and delayed commit, not from full extra visible-state staging

For 2/4 GPU:

- the same frame/version model should reduce host-visible swap time and make peer relay first-class
- it should also reduce barrier exposure by making publish cheaper

Calibrated simulator status:

- Script: `scripts/versioned_frame_cache_sim.py`
- Outputs:
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_1gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_2gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_4gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_q4_1gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_q4_4gpu_calibrated.tsv`

Current calibrated takeaways:

- Twitter q4 is the strongest first target if implementation overhead stays off the hot path:
  - `1 GPU`: `223.230s -> 204.416s (k=1) -> 185.601s (k=2) -> 166.787s (k=3)`
  - `2 GPU`: `170.088s -> 161.516s -> 152.943s -> 144.371s`
  - `4 GPU`: `74.057s -> 67.272s -> 60.487s -> 53.702s`
- FB86M q4 still benefits, but the gains are smaller under full hidden partitions:
  - `1 GPU`: `166.393s -> 161.213s (k=0.25) -> 156.033s (k=0.5) -> 145.673s (k=1.0)`
  - `4 GPU`: `49.822s -> 48.372s -> 46.922s -> 44.021s`
- This supports the architecture, but it also says FB86M should move to tiled frames earlier than Twitter.
- The simulator result is much better than both runtime prototypes:
  - first hidden-frame prototype: `226.455s` on Twitter q4
  - second prototype after removing storage-hot-path translation: `225.546s`
- The second prototype proves the first regression source was real: `edge_sample` dropped from `140.917s` to `15.210s`, `map_lookup` from `66.754s` to `4.119s`, and `negative_sample` back to `8.333s`.
- But `swap_update` stayed flat at about `59.3s`, and total epoch time still did not improve. That means the architecture is plausible, but the prototype is still paying hidden-frame work somewhere outside the intended boundary paths.
- Current runtime on latest `main` now validates the monotonic direction of the design, though with smaller gains than the optimistic simulator:
  - `1 GPU measured`: `222.085s (k=0) -> 213.398s (k=1) -> 206.909s (k=2)`
  - `swap_update measured`: `52.991s -> 43.828s -> 37.449s`
  - `map_lookup measured`: stays roughly flat at `~43s`

## Concrete Implementation Plan

The section below is the original simulator-first phase sketch and is kept for historical continuity. The **current** engineering priority order is the updated implementation plan above.

### Phase 0: Simulator

Implemented:

- `q`
- hidden frame budget `k`
- partition/tile size
- publish latency
- delayed commit queue depth
- peer vs host source

Outputs:

- boundary bytes avoided
- bytes committed late
- average hidden-frame occupancy
- fallback rate when hidden frames are exhausted

Next refinement:

- add a tiled hidden-frame mode for FB86M instead of full partition-equivalent capacity only
- add sensitivity runs that optionally include `swap_sync_wait` in the exposed boundary term for multi-GPU calibration

### Phase 1: Twitter q4 hidden frame prototype

Files likely touched:

- `buffer.h/cpp`
- `storage.h/cpp`
- `graph_storage.h/cpp`

Features:

- one hidden frame
- publish-by-remap
- immediate delayed commit after publish
- exact q4 semantics

Constraint learned from the first prototype:

- hidden-frame logic must not execute on every ordinary q4 read, gather, or sampler access
- the hot path should remain in pure logical q4 coordinates
- frame indirection should be paid primarily at prefetch/publish/commit time, not in steady-state batch execution

Constraint sharpened by the second prototype:

- Removing hot-path storage translation is necessary but not sufficient.
- If `batch_fetch` drops while `runtime - batch_fetch` rises by the same amount,
  the implementation is still shifting work into downstream GPU load/update/compute
  instead of eliminating boundary work.
- The publish path must reduce `swap_update` itself. If `swap_update` stays flat,
  the prototype has not yet implemented the intended architecture.

Constraint sharpened by the boundary kill-test:

- On Twitter q4, one full hidden partition can be published by remap when it fits.
- But under the current memory guard, it fit for only the first two embedding
  swaps of the epoch; the remaining swaps fell back because free HBM stayed near
  `~1470 MiB` while the next hidden preload needed `~3010–4003 MiB`.
- So full hidden-partition `k=1` is not enough as a sustained epoch-level
  mechanism on this hardware/configuration.
- The next viable storage design must therefore be:
  - tiled hidden frames, or
  - a stronger delayed-commit / frame-reclamation policy that frees enough HBM
    for repeated preloads throughout the epoch.

### Phase 2: Event-driven delayed commit

Features:

- stale-frame queue
- commit stream
- explicit version watermark

### Phase 3: FB tiled frame prototype

Features:

- tile frame table
- bounded memory pool
- publish when required tile set is ready

### Phase 4: integrate with state-local batch mapping

This is needed for Twitter-scale improvement beyond swap.

## Recommendation

If the goal is to stop wasting time on one-off flags and move toward a paperable path, this is the next design to back:

**Versioned Frame Cache with publish-by-remap and delayed commit**

It directly addresses why LJ-style fixes did not transfer, stays compatible with multi-GPU, and avoids the accuracy failures caused by changing the visible negative domain.
