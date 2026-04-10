# State-Compiled Runtime for GE²

## Goal

Beat the current GE² runtime on large datasets by changing **how a q-state is executed**, not by changing the logical `q=4` training semantics.

The design target is:

- keep the current logical visible state exact
- remove per-batch dynamic remapping from the hot path
- remove per-batch CPU negative-planning work from the hot path
- let tiled hidden-frame swap become the storage backend instead of the whole optimization story

This document is a pre-coding design note. It is based on the current code structure and the experiments already recorded in [epoch_time_optimization_record_20260407.md](/home/smansou2/newCode/ge2/dandelion-dev/paper/epoch_time_optimization_record_20260407.md).

## What The Code Already Gives Us

The codebase already materializes a large fraction of the state-local view we need.

### 1. State-local positive edges already exist

In [graph_storage.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/storage/graph_storage.cpp), `GraphModelStorage::initializeInMemorySubGraph()` and `updateInMemorySubGraph_()` build:

- `all_in_memory_edges_`
- `all_in_memory_mapped_edges_`
- `in_memory_edge_bucket_ids_`
- `in_memory_edge_bucket_sizes_`
- `in_memory_edge_bucket_starts_`
- `in_memory_subgraph_`

So GE² already computes state-local positive edges once per visible state.

### 2. The hot path is still dynamic per batch

In [dataloader.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/dataloader.cpp), `DataLoader::edgeSample()` still does, per batch:

1. slice mapped positive edges
2. run `negativeSample(...)`
3. concatenate positive and negative IDs
4. run `map_tensors(...)`
5. rebuild batch-local compact indices

That means the current hot path still pays:

- per-batch negative generation
- per-batch unique/compact mapping
- per-batch remap assignment

even though the positive-edge state itself is already compiled once.

### 3. The current mapper is the wrong granularity

The current fixed-buffer bitmap path in [util.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/common/util.cpp) is still a general-purpose batch mapper. It is not specialized to the current state-local domain and it still redoes the whole unique/inverse construction each batch.

The failed `GEGE_FIXED_BUFFER_BITMAP_LOCAL_DOMAIN=1` experiment proved that "smaller domain in the same mapper" is not enough. The next map path needs a new algorithm, not a smaller parameter.

## Why The Previous Attempts Failed

The earlier experiments were useful because they narrowed the design space:

### 1. Visible-state changes hurt accuracy

Rejected paths:

- visible `q=5/6`
- active-tile negatives
- evict-avoid negatives
- frontloaded negative-tape reorderings

These changed the sampler-visible domain or batch/negative pairing and hurt LJ accuracy.

### 2. Storage-only changes help, but only partially

The hidden-frame + delayed stale-writeback prototype in [buffer.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/storage/buffer.cpp) gave:

- Twitter: `223.230 s -> 215.207 s`
- FB86M: `164.098 s -> 154.289 s`

That proves the storage direction is real, but it is not enough by itself because:

- Twitter still has a large `map_lookup` wall
- FB86M full hidden `k=2` already OOMs

### 3. Tiny safe overlap paths are too small

The async next-state batch-list rebuild path did not help because `initializeBatches()` itself is not the real wall. The real wall is:

- `map_lookup`
- `negative_sample`
- `swap_update`

## Design Thesis

The right design is not "compile everything up front" and not "optimize one batch operator."

The right design is a **layered state-compiled runtime**:

1. **StateLocalMapEngine**
   Replace the current per-batch global-domain unique mapper with a touched-ID, generation-stamped, state-local mapper.

2. **NegativeDescriptorTape**
   Preserve exact batch order and q-state semantics, but move negative planning out of the hot path by storing exact per-batch descriptors or replay seeds bound to a state version.

3. **TiledVersionedFrameCache**
   Use the existing frame-cache direction as the storage backend that hides swap/update work.

Together, these three components turn the runtime into:

```text
compile state-local metadata once
replay batches from state-local tape
generate negatives from exact descriptors
run on a tiled frame cache
```

## Architecture

### Layer A: StateLocalMapEngine

This is the first prototype to build. It has the highest upside-to-risk ratio.

#### Core idea

The current batch mapper constructs `unique_node_indices_` and inverse maps from scratch every batch.

Instead, maintain a state-local generation table:

- `seen_generation[num_nodes_in_memory]`
- `compact_index[num_nodes_in_memory]`

For each batch:

1. flatten touched local IDs
2. if `seen_generation[id] != current_generation`, assign a new compact row
3. write `compact_index[id]`
4. fill inverse tensors from `compact_index[id]`

This avoids:

- global-domain bitmap scans
- clearing large temporary structures
- general-purpose unique-map machinery in the hot path

#### Why this is different from the rejected local-domain bitmap path

The rejected path still used the existing padded bitmap algorithm. It only changed the domain size.

This new engine changes the algorithmic cost model:

- current path: effectively tied to batch-wide unique machinery and large-domain bookkeeping
- proposed path: proportional to **touched IDs in this batch**

#### Data structures

Per device, per visible state:

```text
StateLocalMapWorkspace
  uint32 seen_generation[num_nodes_in_memory]
  int32 compact_index[num_nodes_in_memory]
  int32 unique_ids_buf[max_batch_touched_ids]
  int32 inverse_buf[max_batch_touched_ids]
  uint32 current_generation
```

Notes:

- `num_nodes_in_memory` is already known from `in_memory_subgraph_->num_nodes_in_memory_`
- `max_batch_touched_ids` is bounded by `2 * batch_edges + negatives`
- use `int32` for local IDs where possible

#### First prototype scope

- positives + current negatives
- no change to batch order
- no change to negative distribution
- no storage change required

This isolates the Twitter map wall from all other moving parts.

### Layer B: NegativeDescriptorTape

This is the second prototype, after the state-local mapper.

#### Core idea

The negative sampler today still plans negatives inside `edgeSample()` for every batch.

Instead, bind exact negative descriptors to the state version:

```text
NegativeDescriptorTape
  state_version
  batch_id
  sampler_mode
  num_chunks
  num_uniform
  num_degree
  rng_seed_hi
  rng_seed_lo
  optional exact sampled edge ids
  optional exact uniform ids
```

The crucial difference from the rejected frontloaded tape experiments is:

- batch order does not change
- each descriptor is bound to one exact batch
- descriptors are replayed under the exact same published q-state

This is a compile/replay contract, not a batch reordering optimization.

#### Prototype order

There are two valid prototype levels:

1. **Replay seeds only**
   - preserve exact RNG order
   - replay the same generator path on GPU
   - lower storage cost

2. **Materialized descriptors**
   - store exact sampled edge IDs / uniform IDs
   - simpler replay
   - more storage

The first prototype should start with replay seeds.

### Layer C: TiledVersionedFrameCache

This is the third layer, not the first.

The current hidden-frame work already proved:

- storage overlap is real
- FB responds more strongly than Twitter
- full FB hidden `k=2` partitions do not fit

So the storage backend should become:

```text
logical q4 slots
  -> versioned physical frames
  -> tiled hidden admit frames
  -> tiled stale commit queue
```

This layer should be built after the map/runtime contract is cleaner, not before.

## Proposed Data Structures

### 1. CompiledStateKey

```text
CompiledStateKey
  device_idx
  state_sequence
  logical_partition_ids_hash
  frame_version_hash
```

Purpose:

- makes compiled artifacts invalid if the visible state changes unexpectedly
- prevents replay against the wrong published state

### 2. BatchTapeEntry

```text
BatchTapeEntry
  batch_id
  edge_start
  edge_size
  bucket_id
  positive_unique_offset
  positive_unique_count
  neg_desc_offset
  flags
```

This references existing state-local mapped edges rather than duplicating them.

### 3. PositiveUniqueArena

```text
PositiveUniqueArena
  int32 unique_ids_flat[]
  int64 offsets[]
```

This allows a prototype where only positive compaction is prebuilt, before full negative replay is added.

### 4. NegativeDescriptorArena

```text
NegativeDescriptorArena
  uint64 seed_hi[]
  uint64 seed_lo[]
  int32 num_chunks[]
  int32 num_uniform[]
  int32 num_degree[]
  int64 aux_offset[]
```

### 5. StateLocalMapWorkspace

The per-state generation-stamped workspace described earlier.

### 6. TileFrameVersionTable

Reuse the frame-cache concepts already prototyped:

```text
TileFrameVersionTable
  logical_slot -> frame/tile-set
  frame_id -> {partition_id, tile_range, version, dirty}
```

## Why This Should Work Better Than The Current Frame-Cache-Only Path

Because it attacks all remaining exposed costs together:

### Twitter current-code reference

- epoch: `223.230 s`
- `map_lookup`: `43.508 s`
- `negative_sample`: `9.107 s`
- `swap_update`: `59.414 s`

### FB current-code reference

- epoch: `164.098 s`
- `map_lookup`: `86.620 s`
- `negative_sample`: `0.544 s`
- `swap_update`: `65.485 s`

The current frame-cache prototype only attacks part of `swap_update`.

The state-compiled runtime can attack:

- `map_lookup` directly through `StateLocalMapEngine`
- `negative_sample` through `NegativeDescriptorTape`
- `swap_update` through `TiledVersionedFrameCache`

## Prototype Sequence

### Prototype 0: simulator only

No runtime edits. Estimate lower and upper bounds for:

- residual map cost
- residual negative cost
- residual swap cost

### Prototype 1: StateLocalMapEngine only

Scope:

- keep current negative sampler
- keep current storage
- replace current batch `map_tensors(...)` path under bucket-streaming LP

Acceptance:

- LJ safety gate passes
- Twitter `map_lookup` drops materially
- no visible-state or accuracy semantics change

### Prototype 2: NegativeDescriptorTape

Scope:

- preserve exact batch order
- replay exact per-batch seeds/descriptors
- keep current storage

Acceptance:

- `negative_sample` drops
- LJ accuracy gate passes

### Prototype 3: integrate with tiled frame cache

Scope:

- storage overlap backend only after Layers A and B exist

Acceptance:

- Twitter and FB large-dataset epoch time moves materially

## What To Simulate Before Coding

### 1. Epoch-time residual model

Use:

```text
epoch = base_other + residual_map + residual_neg + residual_swap + publish_penalty
```

where:

- `base_other = epoch - map - neg - swap`
- residuals are fractions of the current measured costs

This is enough to reject weak ideas before runtime changes.

Current simulator:

- script: [state_compiled_runtime_sim.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/state_compiled_runtime_sim.py)
- outputs:
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_sim_20260409/twitter_1gpu.tsv`
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_sim_20260409/fb86m_1gpu.tsv`
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_sim_20260409/fb86m_1gpu_conservative.tsv`

Useful read of the current sweep:

- Twitter aggressive case (`map_residual=0.10`, `neg_residual=0.00`, `swap_residual=0.25`) gives about `130.6 s`
- Twitter more moderate case (`map_residual=0.25`, `neg_residual=0.25`, `swap_residual=0.35`) gives about `145.3 s`
- FB86M conservative case (`map_residual=0.75`, `neg_residual=1.00`, `swap_residual=0.50`) gives about `109-110 s`

These are not claims of achieved runtime. They are residual-cost scenarios that tell us:

- Twitter needs all three layers to move materially
- FB86M can benefit even if only part of the current map bucket is actually removable

### 2. State-local workspace size

For each dataset:

- `seen_generation` bytes
- `compact_index` bytes
- `unique_ids_buf` bytes
- `inverse_buf` bytes

This tells us whether the state-local mapper can live on GPU or should start on CPU.

Current offline analyzer:

- binary: [gege_state_compiled_runtime_analyzer.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/gege_state_compiled_runtime_analyzer.cpp)
- output files:
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_twitter_state0_20260409.txt`
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_twitter_allstates_20260409.txt`
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_fb86m_state0_20260409.txt`
  - `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_fb86m_allstates_20260409.txt`

Key result:

- the positive-only `StateLocalMapEngine` workspace is small and stable enough to prototype first
- fully materialized negative-id tapes are too large to make the first prototype

Twitter all-states summary (`20` states, `CUSTOM p=16 q=4`):

- `positive_edges`: `66.97M` min, `73.42M` avg, `91.35M` max
- `positive_unique_rows`: `8.24M` min, `8.40M` avg, `8.85M` max
- `positive_density`: `0.791x` min, `0.807x` avg, `0.850x` max
- `compile_ms`: `3804.5` min, `4210.3` avg, `5330.7` max
- `workspace.positive_only.total_gib`: `0.078` for every state
- `tape.positive_unique_arena_gib`: `0.173` min, `0.185` avg, `0.229` max
- `tape.materialized_negative_ids_gib`: `0.998` min, `1.094` avg, `1.362` max

FB86M all-states summary (`20` states, `CUSTOM p=16 q=4`):

- `positive_edges`: `12.53M` min, `15.24M` avg, `19.24M` max
- `positive_unique_rows`: `8.26M` min, `9.42M` avg, `11.06M` max
- `positive_density`: `0.384x` min, `0.438x` avg, `0.514x` max
- `compile_ms`: `1475.4` min, `1764.5` avg, `2160.3` max
- `workspace.positive_only.total_gib`: `0.161` for every state
- `tape.positive_unique_arena_gib`: `0.070` min, `0.080` avg, `0.100` max
- `tape.materialized_negative_ids_gib`: `0.187` min, `0.227` avg, `0.287` max

Interpretation:

- Twitter state `0` was representative enough for the workspace decision; it was slightly heavier than average, not a misleading outlier
- FB86M is even more favorable for the positive-only map-engine prototype than Twitter
- both datasets support the same conclusion:
  - implement the positive-only state-local mapper first
  - keep negatives descriptor-based or seeded, not fully materialized

### 3. Tape size

For each dataset:

- `BatchTapeEntry` bytes
- positive unique arena bytes
- negative descriptor arena bytes

This tells us whether a full state tape is reasonable or whether the first prototype should compile only a bounded lookahead window.

## Risks

### 1. Map-space correctness

The state-local mapper must be validated against the current `map_tensors(...)` output exactly.

### 2. RNG equivalence

If replay seeds do not match the current negative sampler exactly, quality comparisons will be noisy or invalid.

### 3. Tape memory

A fully materialized batch tape may be too large if it stores too much per-batch inverse metadata. This is why the prototype sequence starts with the generation-stamped mapper rather than the full tape.

## Recommendation

The next implementation should **not** be another swap optimization and **not** another scheduler variant.

The next implementation should be:

1. `StateLocalMapEngine`
2. then `NegativeDescriptorTape`
3. then integrate with tiled frame cache

This is the cleanest path to a system that can materially beat current GE² without changing q-state semantics.
