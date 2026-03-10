# KGE Optimization Update

Advisor presentation draft.

Date: 2026-03-09

---

# 1. Objective

- Reproduce the GE2 large-scale Twitter result in the current codebase
- Find the real runtime bottlenecks, not just local kernel issues
- Test whether targeted systems changes move end-to-end training time
- Decide what is paper-worthy vs what is only an ablation

---

# 2. Two Workloads We Studied

- `FB15K` / `LiveJournal`
  - used to optimize the decoder and negative-sampling path
  - goal: reduce selected-negative overhead

- `Twitter_16p` on 4 GPUs
  - used to analyze GE2-style partition-buffer training
  - goal: find what still dominates when COVER is already in place

---

# 3. FB15K: Initial Problem

- Early profiling showed that the old fast path still spent meaningful time in:
  - selected-negative materialization
  - selected-negative scoring
  - exact selection
- We also tested whether unique/remap was the main bottleneck.

Initial conclusion:
- unique/remap was not the main end-to-end limiter
- the more important path was decoder-side negative handling

---

# 4. FB15K / LiveJournal: What Worked

## A. Fused selected-negative scoring path

FB15K, fixed-seed 1 epoch:

- baseline:
  - `980 ms`
  - `493002 edges/s`
  - `MRR 0.502476`
- optimized:
  - `638 ms`
  - `757276 edges/s`
  - `MRR 0.502733`

Effect:
- `34.9%` lower epoch time
- `53.6%` higher throughput
- no meaningful accuracy loss

LiveJournal, 1 epoch train-only:

- baseline:
  - `73310 ms`
  - `847011 edges/s`
- optimized:
  - `51654 ms`
  - `1202122 edges/s`

Effect:
- `29.5%` lower epoch time
- `41.9%` higher throughput

---

# 5. LiveJournal Validation

3-epoch fused-path validation:

- epoch 1:
  - `51507 ms`
  - `MRR 0.580266`
- epoch 2:
  - `51656 ms`
  - `MRR 0.643794`
- epoch 3:
  - `51723 ms`
  - `MRR 0.666757`

Conclusion:
- runtime stayed flat
- MRR improved normally
- the decoder-side optimization is stable, not just a microbenchmark artifact

---

# 6. Negative Sampling: What Worked

## Tournament selection

LiveJournal, GAN, 3 epochs:

- top-k baseline:
  - `51479 / 51647 / 51681 ms`
  - final `MRR 0.666753`
- tournament:
  - `49393 / 49509 / 49638 ms`
  - final `MRR 0.666564`

Effect:
- about `4.0%` faster
- negligible MRR change

FB15K, 1 epoch:

- DNS:
  - top-k `633 ms`, `MRR 0.502539`
  - tournament `599 ms`, `MRR 0.506906`
- GAN:
  - top-k `931 ms`, `MRR 0.509810`
  - tournament `883 ms`, `MRR 0.513612`

Conclusion:
- tournament selection is a real win
- kept in the codebase

---

# 7. What We Tried And Rejected

Rejected because they did not move the true critical path, or hurt quality/runtime:

- exact two-stage top-k
- approximate preselect + rerank
- relation-local sampler variants
- hard-negative cache
- narrow tensor-core pad/cast path
- several narrow fused selector kernels
- bucket-backed batch path as implemented
- hot-entity replica cache
- sparse row-flush / naive delta-log writeback

Common pattern:
- many of these reduced one local cost
- but the saved time reappeared elsewhere, or the new path added its own overhead

Key lesson:
- after the decoder optimizations, the remaining big problem was no longer a local kernel issue

---

# 8. Twitter 4-GPU: Reproduction Baseline

We built a paper-like approximation for Table 5:

- 4 GPUs
- `RNS`
- `batch_size = 50000`
- `num_chunks = 50`
- `negatives_per_positive = 1000`
- `output_dim = 100`
- `HOST_MEMORY` edges + `MEM_PARTITION_BUFFER` embeddings
- `CUSTOM` ordering

Important note:
- current codebase does not expose a true config-level `DOT` decoder
- we used the closest available approximation

Reference from GE2 Table 5:
- Twitter, 4 GPUs, Dot model: `181.5 s`

Our steady-state baseline:
- epochs `2-10` average about `167.8 s`

Conclusion:
- the current codebase can reproduce the GE2 Twitter regime

---

# 9. Twitter 4-GPU: Actual Bottleneck

Steady-state baseline, epochs `2-5`:

- epoch runtime mean:
  - `166846 ms`

Mean counters:

- `swap_barrier_wait_ms = 45222`
- `swap_update_ms = 65860`
- `swap_rebuild_ms = 62605`
- `swap_sync_wait_ms = 38022`
- `all_reduce_ms = 0`

Interpretation:

- not decoder compute
- not dense all-reduce
- the main remaining GE2 cost is execution around superstep transitions:
  - swap/update
  - rebuild/remap
  - synchronization / waiting

This is the key unresolved problem in the current codebase.

---

# 10. Twitter 4-GPU: Relay Experiment

## Exact peer relay

Steady-state, epochs `2-5`:

- runtime mean:
  - `158886 ms`
- improvement vs baseline:
  - `4.8%`

Mean counters:

- `swap_barrier_wait_ms = 45400`
- `swap_update_ms = 35167`
- `swap_rebuild_ms = 61388`
- `swap_sync_wait_ms = 37368`

What changed:

- `swap_update_ms` dropped by about `46.6%`
- barrier wait did not improve

Conclusion:
- relay is real and worth keeping
- but transport is not the only remaining problem
- the saved swap time is not turning into proportional end-to-end gain

---

# 11. Twitter 4-GPU: Block Shuffle And Combined Result

## Block shuffle only

Steady-state, epochs `2-5`:

- runtime mean:
  - `187368 ms`
- regression vs baseline:
  - `12.3%`

Conclusion:
- not a useful standalone optimization on ARC

## Relay + block shuffle

Steady-state, epochs `2-5`:

- runtime mean:
  - `158047 ms`
- improvement vs baseline:
  - `5.3%`
- improvement vs relay only:
  - about `0.5%`

Conclusion:
- best current variant
- but almost all of the gain still comes from relay
- block shuffle is an ablation, not a main result

---

# 12. What The Current Data Means

GE2/COVER is not the problem anymore.

Current evidence says:

- partition communication structure is already good enough
- the unresolved cost is **how the next state is prepared and consumed**

The large remaining terms are:

- `swap_barrier_wait_ms`
- `swap_rebuild_ms`
- `swap_sync_wait_ms`

So the next contribution should not be:

- “better ordering than COVER”
- another local decoder/kernel tweak

It should be:

- **better execution around COVER**

---

# 13. Most Relevant Paper Ideas

The three papers most aligned with the current bottleneck:

## `GSplit`
- strongest match for the next step
- useful idea: split-parallel execution instead of global lockstep
- maps to: remove the hard superstep barrier

## `BGL`
- closest systems match
- useful idea: data prep / I/O / preprocessing are first-class bottlenecks
- maps to: move state prep off the critical path

## `PCGCN`
- strongest match for the eventual execution model
- useful idea: partition-centric processing instead of whole-state materialization
- maps to: bucket/block streaming

Second-tier, later:
- `TLPGNN`
- `fuseGNN`
- DLRM embedding communication study

Weak match for the current bottleneck:
- `Tango`
- sparse inference survey

---

# 14. Paper-Worthy Direction

Current optimizations are not enough by themselves for a VLDB paper.

Relay is a useful optimization.
Relay + block shuffle is still only an engineering result.

The paper-worthy contribution should be:

## Rolling COVER + Relay-Streaming KGE

Claim:

- COVER optimizes which partitions move
- it does not optimize how the next state is prepared and consumed
- our system fixes that execution gap

Core mechanisms:

- dependency-driven partition relay
- no global superstep barrier
- streaming bucket/block execution instead of full-state materialization
- overlap next-superstep prep with current compute

---

# 15. Immediate Engineering Plan

## Keep
- peer relay

## Keep only as ablation
- block shuffle

## Next implementation steps

1. per-device, per-superstep finish-time instrumentation
2. Rolling COVER:
   - remove the hard global barrier
   - advance when required relayed partitions are ready
3. relay-streaming execution:
   - stream bucket/block descriptors
   - stop full-state materialization where possible

Goal:
- convert the current `40-55s` barrier wait into useful work

---

# 16. Advisor Takeaway

- We reproduced the GE2 Twitter 4-GPU regime.
- FB15K/LiveJournal decoder work produced strong real gains.
- On the Twitter 4-GPU path, relay is the only proven new optimization so far.
- The remaining bottleneck is not math; it is superstep execution overhead.
- The next research step is **Rolling COVER + relay-streaming**, not more local synchronous tweaks.
