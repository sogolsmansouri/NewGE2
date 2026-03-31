# CUSTOM Schedule Limitations, Math, and a Better Scheduler Direction

## Scope

This note connects three things:

1. The `COVER` / `CUSTOM` schedule described in `paper/ge2_technical_report.pdf`.
2. The current runtime implementation in `gege/src/cpp/src/data/ordering.cpp`.
3. The later optimizer in `scripts/optimize_custom_schedule.py`.

The main conclusion is:

- The original `CUSTOM` construction is a valid coverage template.
- It is not a runtime-optimal scheduler.
- The current optimizer improves that template, but it does not replace it.
- A stronger contribution is to synthesize a runtime-aware schedule directly for `1`, `2`, and `4` GPUs.

## What Is True About the Original CUSTOM Algorithm

The original schedule does provide a narrow form of balance:

- It exact-covers partition-pair interactions.
- It groups states into disjoint rounds that fit the memory budget when `buffer_capacity = 4`.

But it does **not** optimize for actual runtime on skewed datasets:

- It does not use real bucket sizes.
- It does not use batch counts.
- It does not maximize partition reuse over time.
- It is not topology-aware.
- It is not peer-relay-aware.
- It does not model inter-GPU communication cost.

So the paper schedule is best understood as a **feasibility construction**, not a cost-optimal execution schedule.

## Important Corrections

Two common simplifications are slightly off:

### It is not hard-coded to exactly 16 partitions

The current builder is not fixed to `16`, but it is restricted:

- `buffer_capacity == 4`
- `num_partitions` must be a power of two

That restriction is encoded in:

- `gege/src/cpp/src/data/ordering.cpp`
- `scripts/optimize_custom_schedule.py`

So the real limitation is not "hard-coded 16"; it is:

- hard-coded `q = 4`
- power-of-two partition construction
- quadratic growth in state count

### The balance claim is combinatorial, not runtime-aware

The construction balances pair coverage structurally, but that is not the same as balancing:

- edges
- batches
- load bytes
- swap cost
- optimizer-state movement
- barrier slack

Those runtime quantities depend on the dataset, not just on the combinatorics.

## Why 16 Partitions and Buffer Size 4 Gives 20 States

Let:

- `p = 16` partitions
- `q = 4` partitions resident in one state

### Pair-counting view

There are:

- `C(16, 2) = 120` unordered partition pairs

A 4-partition state contains:

- `C(4, 2) = 6` unordered pairs

So the minimum number of 4-partition states needed for exact pair coverage is:

- `120 / 6 = 20`

### Ordered-bucket view

If you think in directed edge buckets instead:

- off-diagonal ordered buckets = `16 * 15 = 240`

One 4-partition state covers:

- `4 * 3 = 12` off-diagonal ordered buckets

So:

- `240 / 12 = 20`

Same answer.

## Why 5 Groups / Rounds

Each partition must meet the other `15` partitions exactly once.

In one 4-partition state, a given partition meets:

- `3` other partitions

So each partition must appear in:

- `15 / 3 = 5` groups

The equivalent report formula is:

- states per group = `p / q = 16 / 4 = 4`
- groups = `(p - 1) / (q - 1) = 15 / 3 = 5`

Therefore:

- `4` states per group
- `5` groups
- total `4 * 5 = 20` states

## Hardware Meaning of Those 5 Groups

The construction is naturally aligned to `4` GPUs:

- `4` states can run together in one disjoint round
- `5` rounds complete the epoch schedule

So for `p = 16`, `q = 4`:

- on `4` GPUs: `5` hardware supersteps
- on `2` GPUs: `10` hardware supersteps
- on `1` GPU: `20` sequential state steps

This is the core reason `CUSTOM` feels natural on `4` GPUs but awkward on `1` and `2`.

The structure was designed around disjoint groups of `4`, not around low-GPU-count reuse.

## A Subtle Implementation Imbalance in Current Code

The paper-level balance story is idealized.

The current implementation greedily assigns buckets to the first compatible state. That means:

- off-diagonal buckets are exact-covered once
- diagonal buckets `(p_i, p_i)` get assigned early

For `p = 16`, this makes the first group heavier than later groups if diagonal buckets are counted. So even bucket-count balance is not perfectly uniform in the current runtime implementation.

## What `optimize_custom_schedule.py` Already Improves

The later optimizer is useful. It is not just cosmetic.

It improves the fixed `CUSTOM` template in three real ways:

### 1. Dataset-aware slot relabeling

Instead of random partition placement, it assigns partitions to slots using observed bucket sizes and partition hotness.

### 2. Better balance inside the fixed state family

It scores schedules using:

- worst round spread
- worst batch spread
- worst state weight
- total round spread
- continuity penalties

### 3. Better lane continuity

It does dynamic programming over within-round permutations so that consecutive rounds reuse partitions on the same lane more often.

That logic exists both in:

- `scripts/optimize_custom_schedule.py`
- the integrated runtime path behind `GEGE_OPTIMIZED_CUSTOM_SCHEDULE`

## What It Still Does Not Fix

This is the important limitation.

The optimizer is still **template-preserving**.

It does **not**:

- change the state family
- generate new states
- change the number of states
- change the number of rounds
- optimize a `2`-GPU-native or `1`-GPU-native decomposition
- account for actual transfer bytes
- account for peer-relay cost
- account for topology
- account for optimizer-state movement explicitly

In the current optimizer:

- rounds are still created by chunking the fixed state list into blocks of `active_devices`
- continuity is still modeled by hotness and number of new partitions, not by a measured transfer/caching model

So the current optimizer is best described as:

- a strong **Phase 1**
- good for improving `CUSTOM`
- not yet a new scheduler

## Which Optimizations Are Actually CUSTOM-Specific

Mostly CUSTOM-specific:

- `GEGE_OPTIMIZED_CUSTOM_SCHEDULE`
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM`
- `GEGE_ACCESS_AWARE_STATE_GENERATION`
- `GEGE_ACCESS_AWARE_SCHEDULER`
- `GEGE_PARTITION_BUFFER_PEER_RELAY`

Mostly not CUSTOM-specific:

- `GEGE_FAST_MAP_TENSORS`
- `GEGE_PARTITION_BUFFER_LP_FAST_PATH`
- `GEGE_GPU_ACTIVE_EDGE_SHUFFLE`
- `GEGE_DEG_CHUNK_EXCLUSION`
- `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS`
- `GEGE_UNIQUE_BACKEND=bitmap`
- `GEGE_CSR_GATHER`
- `GEGE_CSR_UPDATE`
- `GEGE_BUCKET_STREAMING_LP`

That matters for the paper framing. Many observed wins are broader system wins, not `CUSTOM`-specific algorithmic wins.

## Why CUSTOM Is Not Scalable in the Right Sense

The issue is not only raw complexity. It is also mismatch to hardware.

### Structural scaling issue

For fixed `q = 4`, state count grows roughly as:

- `O(p^2)`

Examples:

- `p = 8` gives `6` states
- `p = 16` gives `20`
- `p = 32` gives `104`
- `p = 64` gives `336`

### Hardware mismatch issue

The schedule is intrinsically organized around groups of `4` disjoint states.

That means:

- `4` GPUs are the natural target
- `2` GPUs inherit a split version of a `4`-GPU design
- `1` GPU inherits a serialized version of a `4`-GPU design

So even after optimization, `CUSTOM` is not naturally shaped for `1`- or `2`-GPU execution.

## What a Better Scheduler Should Optimize

A stronger scheduler should keep the correctness constraints, but replace the objective.

### Correctness constraints

- every bucket is assigned exactly once
- each state fits memory
- if a round has multiple simultaneously active states, those states are disjoint as needed

### Runtime objective

Minimize a weighted cost that reflects actual runtime:

- bucket compute weight
- batch count
- new partition bytes loaded
- optimizer-state bytes loaded
- transfer / relay cost
- barrier imbalance across GPUs
- transition cost between consecutive supersteps

In other words, optimize:

- state generation
- state ordering
- state-to-device assignment
- handoff cost between rounds

## A Better Formulation for `p = 16, q = 4`

For `16` partitions and capacity `4`, the search space is manageable.

There are only:

- `C(16, 4) = 1820` possible 4-partition states

That means you can precompute for every candidate state:

- covered buckets
- covered edge weight
- batch count
- resident partition set
- resident bytes

And for every pair of states:

- compatibility
- transition bytes
- relay opportunities
- overlap / reuse score

Then scheduling becomes:

- choose a sequence of states
- group them into rounds of size `g = active_devices`
- assign round members to lanes
- minimize total cost while covering all buckets exactly once

That gives the right specialization automatically:

### 4 GPUs

- choose the best `4` disjoint states per round

### 2 GPUs

- choose the best `2` disjoint states per round
- do **not** inherit a `4`-GPU grouping and split it afterward

### 1 GPU

- solve a path problem over states
- maximize reuse and minimize new loads

This is the key conceptual shift:

- from "optimize a fixed schedule"
- to "synthesize a hardware-aware schedule"

## Practical Algorithm Directions

### Option A: Beam search

At each step:

- keep top `K` partial schedules
- extend with compatible next states or next rounds
- score by uncovered weight, balance, and transition cost

Why it is attractive:

- easy to prototype
- easy to specialize for `1`, `2`, or `4` GPUs
- easy to inject heuristics

### Option B: ILP / MIP

Variables:

- whether a state is selected
- which round it belongs to
- which lane it occupies

Objective:

- weighted sum of coverage, balance, and transition costs

Why it is attractive:

- precise formulation
- good for small `p`

Why it is less attractive first:

- harder to integrate quickly
- harder to tune with nonlinear transition logic

### Option C: Greedy + local improvement

1. Greedily form rounds by best cost-benefit ratio.
2. Reorder rounds for continuity.
3. Swap states between rounds if cost improves.

Why it is attractive:

- easiest first prototype
- closer to the current code structure

## Recommended Research Path

If the goal is a publishable novelty, the clean path is:

### Phase 1

Keep the current `CUSTOM` template and improve it further for `1` and `2` GPUs:

- explicit `2`-GPU round synthesis
- explicit `1`-GPU path ordering
- transition cost based on bytes, not just hotness

### Phase 2

Replace the fixed template with synthesized schedules:

- enumerate candidate states
- choose rounds based on runtime cost
- lane-assign with real continuity cost

### Phase 3

Integrate topology and relay:

- peer-relay aware cost
- topology-aware lane assignment
- different policies for PCIe vs NVLink-like systems

## Short Paper Framing

If this becomes the novelty section, the concise framing is:

- Original `CUSTOM` provides an exact-cover construction for bounded-memory multi-GPU training.
- However, it is not runtime-aware and is structurally aligned to `4`-way disjoint execution.
- Later optimizations improve partition relabeling and lane continuity, but still preserve the fixed template.
- We propose a hardware-aware scheduler that directly optimizes state generation, grouping, ordering, and assignment under measured execution cost.

That framing is technically honest and clearly separates:

- the original construction
- your current improvements
- the next real contribution
