# GE2 Epoch-Time Optimization Record

Date: 2026-04-07

This note records the optimization path we explored after profiling Figure 3's `load -> train -> dump` bottleneck. It is written for paper use: what we tried, why it was plausible, what happened, and what it implies for the next architecture.

## Starting Point

The key observation from the April 2026 profiles was that epoch time was dominated by data movement and batch construction rather than the core score function.

Validated LJ reference before the final stable stack:

- The run was dominated by `batch_fetch`, with large `map_lookup` and `swap_update` costs.
- Negative sampling was not the primary bottleneck for LJ under the then-current RNS path.
- The rigid state machine in Figure 3 exposed swap cost at COVER state boundaries.

The practical target became:

- Remove avoidable per-batch mapping barriers.
- Hide or reduce partition swap movement.
- Preserve exact accuracy before treating any sampler change as publishable.

## Stable Winning Result: LJ 8.5s Stack

Commit: `348d64d Optimize single-GPU LJ training pipeline`

Validated run:

- Log directory: `/dev/shm/smansou2_ge2/lj_streamwait_rerun_30e_eval_20260407`
- Avg epoch excluding epoch 1: `8499.10 ms`
- Last 10 epochs: `8525.00 ms`
- Exact eval: `MRR=0.129537`, `Hits@10=0.3108`

Why it helped:

- Fixed-buffer bitmap mapping removed the old per-batch `torch::unique` style variable-output path from the hot path.
- `map_tensors calls=0` in the validated run, so the previous explicit unique/remap call path was bypassed.
- Stream-wait async swap made the LJ-sized evict/admit staging path safe and partially overlapped without reading stale rows.
- The stream-wait correctness fix ensured the evict copy stream waits for training-stream completion before copying rows out of the GPU buffer.

Key flags for the validated LJ run:

```bash
CUDA_VISIBLE_DEVICES=0
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
LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

Eval-only exact rank flags:

```bash
GEGE_EVAL_CHUNKED_RANKS=1
GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768
```

Remaining LJ steady-state breakdown:

- Epoch: `8499.1 ms`
- `batch_fetch`: `6207.5 ms`
- `swap_update`: `2648.6 ms`
- `map_lookup`: `2201.7 ms`
- `negative_sample`: `959.1 ms`
- Model compute and other non-fetch work: about `2291.6 ms`

Interpretation:

- LJ is now fast enough to serve as a correctness and small-scale sanity benchmark.
- It is not representative of the large-scale memory problem because the full async staging buffers fit in 24 GB HBM for LJ.

## Stable Winning Result: FB86M Padded CUDA DEG Filter

Commit: `c485da6 Add padded CUDA DEG negative filter`

Baseline before this step:

- Run: `/dev/shm/smansou2_ge2/fb86m_exact_old_probe_flags_3e_20260407`
- Avg epoch: `201538 ms`
- `negative_sample`: about `36871 ms`
- `filter`: about `36188 ms`
- `swap_update`: about `65491 ms`
- `map_lookup`: about `1746 ms`

Winning run:

- Run: `/dev/shm/smansou2_ge2/fb86m_deg_filter_padded_3e_20260407`
- Avg epoch: `166393 ms`
- Speedup: `17.44%`
- `negative_sample`: `583.646 ms`
- `filter`: `80.359 ms`
- `swap_update`: `65433.374 ms`
- `map_lookup`: `86723.261 ms`

Why it helped:

- FB86M was dominated by the DEG local negative filter, not by map lookup.
- The padded CUDA filter preserves the same local filter semantics but moves the expensive path into a fixed-shape CUDA implementation.
- Verification runs passed on LJ and FB86M before using the clean timing run.

Important interpretation:

- After this optimization, `map_lookup` became a synchronization bucket that absorbs queued GPU work; it is not evidence that the old map algorithm became worse.
- The true removed cost is the `filter_ms` collapse from about `36.2 s` to about `80 ms`.
- The next FB86M bottleneck is partition swap movement, not the negative filter.

Flags for this FB86M run:

```bash
CUDA_VISIBLE_DEVICES=0
GEGE_BUCKET_STREAMING_LP=1
GEGE_FAST_MAP_TENSORS=0
GEGE_CSR_GATHER=0
GEGE_CSR_UPDATE=0
GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
GEGE_UNIQUE_BACKEND=bitmap
GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
GEGE_DEG_LOCAL_FILTER_PADDED=1
LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
```

Verifier flags used during development:

```bash
GEGE_DEG_LOCAL_FILTER_PADDED_VERIFY=1
GEGE_DEG_LOCAL_FILTER_VERIFY_MAX_CALLS=64
```

## Useful Infrastructure: Swap Dirty Profiling and Async Memory Guard

Commits:

- `7d6d057 Add partition swap dirty profiling`
- `f5b14a3 Guard async swap stage allocations`

Why we added it:

- LJ async swap staging works, but FB86M and Twitter OOM when trying to allocate full shadow evict/admit stages.
- We needed to know whether dirty-only writeback could reduce swap traffic enough and whether full async staging could safely fit.

What it showed:

- FB86M full async staging does not fit on the RTX 3090. Per storage swap, it needs about `12.31 GiB` while free HBM is around `6 GiB`.
- FB86M dirty ratio averages about `66%`. Dirty-only writeback would reduce evict traffic, but not admit traffic, and it requires sparse gather/scatter.
- Twitter dirty ratio is about `99%`, so dirty-only writeback is essentially useless for Twitter.
- The memory guard prevents OOM by falling back when full async staging cannot fit.

Interpretation:

- The guard is useful engineering infrastructure, not a large-scale speedup by itself.
- Dirty profiling ruled out naive dirty-only writeback as the main solution.

## Tried and Rejected: Dirty-Row Writeback

Status: reverted, not committed.

Hypothesis:

- If only a subset of rows changed, write back only dirty rows and their optimizer state.

Why it was plausible:

- FB86M dirty ratio was around `66%`, so evict traffic might fall from about `6.16 GiB` to about `4.07 GiB` per storage swap.

What happened:

- The implementation used sparse `index_select` from GPU plus scattered CPU writes.
- The sparse writeback path was slower than the existing full contiguous copy path.
- In the partial FB86M run, per-swap GPU-to-CPU time frequently rose into the `1100-1450 ms` range and was clearly worse.

Reason rejected:

- It reduced bytes but destroyed memory access regularity.
- For partition-scale tensors, contiguous bulk copy is cheaper than sparse gather plus scattered host update.

Paper implication:

- The issue is not just "copy fewer rows." A scalable design must preserve contiguous transfer structure or change the partition schedule/cache model.

## Tried and Rejected: Full Async Staging on FB86M and Twitter

Hypothesis:

- Reuse the LJ full async evict/admit staging path to overlap swap with compute.

What happened:

- FB86M OOMed without the guard; with the guard it fell back, completing but not improving time.
- Twitter also OOMed with the full LJ async staging stack.

Observed memory limits:

- FB86M needed an additional multi-GiB stage, around `8 GiB` in the earlier OOM and about `12.31 GiB` per storage swap in the detailed profile.
- Twitter needed another `3.88 GiB` in the OOM case.

Reason rejected:

- Full double-buffering is not scalable under 24 GB HBM for large datasets.

Paper implication:

- This is the clearest evidence that the LJ mechanism is not a general large-scale solution. Large-scale GE2 needs a memory-bounded execution model, not larger shadow buffers.

## Tried and Rejected: CPU-Stage Async Evict

Status: reverted, not committed.

Hypothesis:

- Avoid GPU OOM by staging evicted rows through CPU/pinned memory asynchronously.

Result:

- FB86M run: `/dev/shm/smansou2_ge2/fb86m_async_cpu_evict_3e_notiming_20260407`
- Avg epoch was about `209701 ms`, worse than the `201538 ms` baseline.

Reason rejected:

- It avoided the GPU memory pressure but did not reduce the exposed swap cost.
- Additional staging overhead outweighed any overlap benefit.

## Tried and Rejected: Lower Async Memory Margin

Hypothesis:

- The guard margin may be too conservative for FB86M, so reducing it to zero might allow async staging.

Result:

- FB86M margin-0 run OOMed during forward.

Reason rejected:

- The extra memory is genuinely needed for training kernels and allocator headroom.
- Forcing full staging into marginal free memory is unsafe.

## Tried and Rejected: Prefetch CPU Remap Path

Hypothesis:

- Enable prefetch and make the LP fast path compatible with it so swap/build work overlaps with training.

Result:

- Prefetch reduced visible `swap_update` but increased other CPU-side costs.
- LJ prefetch CPU-remap trial: epoch average around `9469 ms`, worse than the validated `8499 ms`.
- `swap_update` dropped, but `negative_sample` and `edge_sample` rose from CPU contention.

Reason rejected:

- It hides one timer bucket by creating contention elsewhere.
- It does not provide a clean end-to-end win.

## Tried and Rejected: Fast-Int DEG Chunk Exclusion

Hypothesis:

- Use a faster negative-sampler path for degree chunk exclusion.

Result:

- Runtime stayed near `8.5 s` on LJ.
- Exact eval collapsed to `MRR=0.050544`, `Hits@10=0.1181`, versus baseline `MRR=0.129537`, `Hits@10=0.3108`.

Reason rejected:

- It changed the negative-sampling distribution or semantics enough to destroy accuracy.

Paper implication:

- For this workload, sampler modifications need exact equivalence or a validated statistical correction. Small-looking sampler shortcuts can invalidate the model.

## Tried and Rejected: Avoid Next-Evict Partitions in RNS

Hypothesis:

- Since RNS is random, avoid sampling negatives from partitions that will be evicted next, creating safe swap windows.

Result:

- Run: `/dev/shm/smansou2_ge2/lj_rns_avoid_evict_clean_30e_eval_20260407`
- Avg epoch excluding epoch 1: `8816.5 ms`, worse than baseline.
- Exact eval: `MRR=0.047135`, `Hits@10=0.1068`.

Reason rejected:

- Avoiding evict partitions concentrates negatives into the remaining partitions and changes the training signal.
- It is not just harmless random noise.

Paper implication:

- Liveness-aware sampling cannot simply remove partitions from the negative domain. It needs exact replay or importance-corrected sampling.

## Tried and Rejected: StateNegativeTape Frontloading

Hypothesis:

- Pre-generate the state-level RNS negative multiset from the original distribution, then frontload negatives that touch soon-to-be-evicted tiles. This would preserve the state-level set while creating early tile liveness.

Sanity that passed:

- State-level multiset equality passed for all checked states: `19/19 sanity_equal=true`.
- Some tile windows appeared: `40/360` evicted tiles became tail-safe in the frontloaded plan.

Result:

- Exact LJ eval collapsed: `MRR=0.049743`, `Hits@10=0.1175`.
- Runtime also worsened to about `9566 ms/epoch` because tape building moved work into `swap_rebuild`.

Reason rejected:

- Preserving the state-level negative multiset is insufficient.
- Moving a negative from one batch to another changes the optimizer sequence and the pairing between positives and negatives.

Paper implication:

- Batch-level pairing matters for KGE training. Any future negative-tape design must preserve per-batch negatives or provide a new algorithmic argument.

## Tried and Rejected: Exact StateNegativeTape Liveness Without Frontloading

Hypothesis:

- Pre-generate RNS negatives from the original distribution, compute exact liveness, and start rolling/tiled swap when a tile becomes truly dead.

Result:

- LJ exact liveness probe checked `360` evicted tiles.
- `0` exact tail tiles became safe under the original full-resident RNS distribution.
- Even when positives had a tail window, negatives touched every evicted tile.

Reason rejected:

- Full-resident RNS keeps the whole buffer live until the state boundary.

Paper implication:

- Rolling swap cannot safely overlap using positive-edge liveness alone.
- Negative-sampler liveness is the central blocker for exact early eviction.

## Tried and Rejected: Resident-Local Direct LP Path

Hypothesis:

- Bypass batch-local unique/remap completely and gather resident embeddings directly by node id. This is a more architectural "dedup-free" path.

FB86M result:

- Run: `/dev/shm/smansou2_ge2/fb86m_resident_direct_padded_1e_20260407`
- Epoch 1: `166921 ms`
- Padded-filter baseline: `166393 ms`
- `map_lookup` became `0`, but total epoch did not improve.

Twitter result:

- Run: `/dev/shm/smansou2_ge2/twitter_resident_direct_1e_20260407`
- Stopped early at 20% because projected epoch was about `300 s` versus the `223 s` baseline.

Reason rejected:

- Removing dedup/remap increases duplicate embedding gathers and pushes work into compute/update.
- For FB86M it is neutral; for Twitter it is worse.

Paper implication:

- Pure "skip unique and use direct duplicate gathers" is not the right general solution for large graph embedding with Adagrad-style state. Deduplication still provides useful compression.

## Tried and Rejected: Twitter Direct-Multi Fixed Buffer Flag

Hypothesis:

- `GEGE_FIXED_BUFFER_BITMAP_DIRECT_MULTI=1` plus output reuse might bypass a large part of Twitter's apparent `map_lookup` time.

Result:

- Run: `/dev/shm/smansou2_ge2/twitter_directmulti_probe_1e_20260407`
- Epoch was about `222401 ms` versus `223085 ms` baseline.
- `map_lookup` dropped from about `44 s` to about `1.68 s`, but total time barely changed.

Reason rejected:

- The old `map_lookup` timer was largely a synchronization bucket. The work moved elsewhere.

Paper implication:

- We need to distinguish measured timer buckets from causal bottlenecks. For Twitter, the map timer alone overstates the gain available from a map-only optimization.

## Current Best Single-GPU Times

| Dataset | Best stable run | Epoch time | Accuracy status |
| --- | ---: | ---: | --- |
| LJ | `/dev/shm/smansou2_ge2/lj_streamwait_rerun_30e_eval_20260407` | `8499.10 ms` excluding epoch 1 | exact eval preserved |
| FB86M | `/dev/shm/smansou2_ge2/fb86m_deg_filter_padded_3e_20260407` | `166393 ms` | train-only timing; exact filter verifier passed |
| Twitter | `/dev/shm/smansou2_ge2/twitter_lj_stack_no_async_stage_3e_20260407` | `223084.67 ms` | train-only timing |

## How The Large-Graph Epoch Time Adds Up

These numbers are useful as accounting, but they should not all be interpreted as causal kernel costs. Several timer buckets, especially `map_lookup` after the fixed-buffer/padded paths, can become the first host synchronization point for queued GPU work.

Twitter q4 single-GPU baseline:

- Run: `/dev/shm/smansou2_ge2/twitter_lj_stack_no_async_stage_3e_20260407`
- Epoch time: `223.085 s`
- This is the current q4 no-async train-only reference. The raw log has since been evicted from `/dev/shm`, so exact perf-line values are not all recoverable; the decomposition below uses the recorded exact epoch/swap values plus the rounded sub-buckets recorded during the analysis session.

Nested accounting:

```text
223.085 s total
= 115.190 s batch_fetch
 + 107.895 s compute/other
```

Expanding `batch_fetch`:

```text
115.190 s batch_fetch
= 61.603 s swap_update
 + 53.530 s edge_sample
 +  0.057 s small batch_fetch overhead
```

Expanding `edge_sample`:

```text
53.530 s edge_sample
= 44.000 s map_lookup
 +  7.930 s negative_sample
 +  1.600 s other edge-sample overhead
```

Fully expanded:

```text
223.085 s total
= 107.895 s compute/other
 + 61.603 s swap_update
 + 44.000 s map_lookup
 +  7.930 s negative_sample
 +  1.600 s other edge-sample overhead
 +  0.057 s small batch_fetch overhead
```

Interpretation:

- Swap-only optimization cannot get Twitter near `100 s`: `223.085 - 61.603 = 161.482 s`.
- Even removing the visible `swap_update + map_lookup + negative_sample` buckets gives only about `109.552 s` before accounting for timer migration and remaining overhead.
- Therefore Twitter needs a real pipeline that hides/removes most of `batch_fetch`, not a swap-only patch.

FB86M current best single-GPU baseline:

- Run: `/dev/shm/smansou2_ge2/fb86m_deg_filter_padded_3e_20260407`
- Epoch time: `166.393 s`
- The raw train log has also been evicted from `/dev/shm`; the exact recorded fields are the stable table values below.

Recorded major buckets:

```text
166.393 s total
= 65.433 s swap_update
 + 86.723 s map_lookup timer bucket
 +  0.584 s negative_sample
 + 13.653 s remaining recorded work
```

Where:

```text
13.653 s remaining
= total - swap_update - map_lookup - negative_sample
= 166.393 - 65.433 - 86.723 - 0.584
```

Interpretation:

- This FB86M accounting is not a clean nested `batch_fetch -> edge_sample` breakdown because the full perf block is no longer available.
- The `map_lookup=86.723 s` value should not be read as "the old map algorithm costs 86 s." After the padded CUDA DEG filter, this timer became a synchronization bucket that absorbs queued GPU work.
- The real confirmed win was `negative_sample` collapse from about `36.871 s` to `0.584 s`, with `filter_ms` specifically dropping from about `36.188 s` to `0.080 s`.
- The next confirmed large FB86M systems target is still partition movement: `swap_update=65.433 s`.

## What We Learned

1. LJ is now a small-scale success case.
   The full async staging mechanism fits in HBM and preserves accuracy.

2. FB86M's first real bottleneck was not map lookup.
   The exact padded CUDA DEG filter removed about `35 s/epoch`, giving the best large-scale single-GPU win so far.

3. Large-scale swap remains the main unresolved systems bottleneck.
   FB86M and Twitter still spend about `60-65 s/epoch` in `swap_update`.

4. Full double-buffering is not the scalable answer.
   It OOMs or falls back on FB86M/Twitter because admitted and evicted partition stages are multi-GiB.

5. Dirty-row writeback is not enough.
   FB86M has only moderate sparsity and sparse gather/scatter is slower than contiguous copy. Twitter is nearly fully dirty.

6. Rolling swap is blocked by negative-sampler liveness.
   Full-resident RNS can touch evicted partitions until the state boundary, so early slot reuse changes semantics unless the sampler is redesigned.

7. Map-only bypasses are not sufficient for Twitter.
   Direct-multi and resident-direct mostly moved time between timers or slowed training.

## Implication for a Novel Pipeline

The next paper-worthy direction should not be "make Marius/Legend again" or "add larger prefetch buffers." The evidence points to a different constraint:

GE2 needs a memory-bounded state executor whose scheduling is aware of both positive-edge coverage and negative-sampler liveness.

A credible new track is:

- Use deterministic state schedules to build a liveness graph over partitions, tiles, positives, and planned negatives.
- Preserve per-batch negative semantics by default; only introduce sampler changes with an explicit correction or validation.
- Reduce whole-partition movement by changing the state schedule/cache policy, not by sparse dirty scatter.
- Keep transfers contiguous and tile-bounded.
- For multi-GPU, treat HBM as a versioned partition/tile cache with exact ownership barriers.

The key novelty opportunity is the coupling of partition scheduling with negative-sampler liveness. The experiments above show why conventional full prefetch, dirty writeback, and naive random-negative rerouting fail on this workload.

## COVER Optimality and the Novelty Boundary

The GE2 report's COVER argument is about the partition scheduling problem (PSP) under the following model:

- A buffer state contains up to `q` node partitions.
- A state covers all positive edge buckets induced by the partitions in that state.
- Every positive edge bucket should be covered once.
- Concurrent multi-GPU states must be independent, i.e., no shared partitions.
- Work should be balanced across independent groups.
- Communication volume is minimized under these constraints.

This is a strong result for COVER's stated objective, but it is not the same as end-to-end epoch-time optimality on current hardware. It does not directly optimize:

- Transfer/compute overlap.
- HBM headroom for staging.
- CUDA stream wait placement.
- Negative-sampler liveness.
- Per-batch synchronization boundaries.
- The fact that a timer bucket such as `map_lookup` can become the first later synchronization point for queued GPU work.

This matters for novelty. A simple "keep 3 partitions and admit 1 partition" schedule is not enough as a paper idea by itself. The GE2 report already discusses BETA/Marius-style behavior: keep `q` partitions and swap one partition to process `q-1` buckets. Reintroducing that idea without a new constraint would look like a known alternative, not a new architecture.

The stronger novelty boundary is to change the object being scheduled. COVER schedules positive edge-bucket coverage. Our evidence says the large-scale bottleneck is caused by the interaction of positive buckets, negative-sampling domains, and partition movement. Therefore, a new scheduler should treat the negative domain as a first-class scheduled object.

## Candidate Novel Track: Contrastive Tiled COVER

Working name: `Contrastive Tiled COVER` or `Negative-Domain-Aware COVER`.

Core idea:

- Keep COVER's positive-bucket coverage as a correctness skeleton.
- Split each state's RNS/DEG negative domain into partition or tile domains.
- Train microtasks of the form `(positive_bucket, negative_tile)` instead of only `(buffer_state)`.
- For RNS, sampling uniformly from a full state can be decomposed exactly in expectation: sample a negative partition/tile with probability proportional to its node count, then sample uniformly within that tile.
- For DEG sampling, sample the negative tile with probability proportional to the tile's degree mass, then sample within the tile by degree.
- This is stratified/importance-corrected negative sampling, not naive exclusion of evict partitions.

Why this is different from Marius/BETA/Legend:

- The scheduled unit is not just a partition swap or future batch.
- The scheduler co-designs positive edge coverage and the negative sampling domain.
- The goal is to create exact or unbiased liveness windows for partition tiles without changing the expected contrastive objective.
- The free HBM slot created by a smaller active negative tile can be used for bounded prefetch/admit without full double buffering.

Why it addresses the failed experiments:

- Unlike `GEGE_RNS_AVOID_NEXT_EVICT_NEGATIVES`, it does not simply remove evict partitions from the negative distribution.
- Unlike frontloaded `StateNegativeTape`, it does not arbitrarily move sampled negatives across positive batches.
- Unlike dirty writeback, it keeps transfers contiguous and scheduled rather than sparse and reactive.
- Unlike resident-local direct gather, it preserves the useful compression from deduplication where that helps.

Expected risks:

- It changes the variance of negative sampling even if the expectation is preserved.
- It requires a careful accuracy study against exact RNS/DEG.
- It may increase kernel launch count unless microtasks are fused or grouped.
- It needs a scheduler that avoids duplicating positive edge work.

Minimal validation path:

1. Implement an offline simulator first: given a COVER state order and a negative-tile decomposition, estimate transfer volume, live tiles, and possible overlap.
2. Add a sampling-only verifier: compare tile-stratified RNS/DEG histograms against the current sampler distribution.
3. Implement a single-GPU `negative_tile` mode for LJ with exact eval after 30 epochs.
4. If LJ accuracy is stable, run FB86M/Twitter 1-epoch timing and check whether `swap_update` starts to fall without full async staging.
5. Only then add multi-GPU independent-group constraints.

The paper claim should be framed as: COVER is optimal for positive bucket coverage under GE2's PSP model, but the measured bottleneck on large datasets is a different problem: contrastive-domain-aware state execution. The new contribution is to schedule negative domains and partition movement together.

## Offline Simulator Added

Script:

- `/home/smansou2/newCode/ge2/dandelion-dev/scripts/contrastive_tiled_cover_sim.py`

Output directory:

- `/dev/shm/smansou2_ge2/contrastive_tiled_cover_sim_20260407`

The simulator is intentionally an estimator, not a runtime predictor. It is used to reject bad design points before modifying the training loop.

It reports:

- COVER-style state lower bounds for candidate `(p, q)`.
- Per-lane transition count for `G=1,2,4`.
- Resident HBM footprint.
- Full evict/admit staging footprint.
- One-partition bounded staging footprint.
- Whether a contrastive microtask with one active negative tile leaves free HBM slots for prefetch.
- A calibrated swap-only epoch estimate using the current measured swap time.

Important caveat:

- For `q=4` and power-of-two `p`, the simulator can use the GE2 CUSTOM template shape and a cheap overlap-greedy ordering approximation.
- For other `q` values, state counts are lower bounds. A real scheduler still needs to be generated and verified.

### Simulator Takeaways for Single GPU

Current single-GPU baselines used for calibration:

- FB86M: `166.393 s`, `swap_update=65.433 s`
- Twitter: `223.085 s`, `swap_update=61.603 s`

FB86M single-GPU candidates:

| Candidate | Resident HBM | Full stage | One-partition stage | Same-style swap-only epoch | One-admit bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current `p=16,q=4` | `16.029 GiB` | `25.309 GiB` | `8.014 GiB` | `166.393 s` | `121.680 s` |
| `p=24,q=6` | `16.029 GiB` | `26.715 GiB` | `5.343 GiB` | `166.393 s` | `114.047 s` |
| `p=24,q=5` | `13.357 GiB` | `21.372 GiB` | `5.343 GiB` | `179.480 s` | `120.590 s` |
| `p=32,q=6` | `12.022 GiB` | `20.036 GiB` | `4.007 GiB` | `190.930 s` | `118.954 s` |

Twitter single-GPU candidates:

| Candidate | Resident HBM | Full stage | One-partition stage | Same-style swap-only epoch | One-admit bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current `p=16,q=4` | `7.758 GiB` | `12.250 GiB` | `3.879 GiB` | `223.085 s` | `180.990 s` |
| `p=16,q=6` | `11.638 GiB` | `19.396 GiB` | `3.879 GiB` | `197.417 s` | `168.669 s` |
| `p=16,q=5` | `9.698 GiB` | `15.517 GiB` | `3.879 GiB` | `206.658 s` | `172.776 s` |
| `p=24,q=6` | `7.758 GiB` | `12.931 GiB` | `2.586 GiB` | `223.085 s` | `173.803 s` |

Interpretation:

- Increasing `p` alone is not enough. It shrinks partitions but can explode state count.
- Increasing `q` alone is not enough. It reduces state count but raises resident HBM pressure.
- The most promising axis is `q=5/6` plus negative-domain tiling, because a microtask using two positive partitions plus one negative tile leaves `q-3` free slots for bounded prefetch.
- The practical first prototype should not use full staging. Full stage remains too large for FB86M and often too large for Twitter.
- The prototype should target a one-partition/tile staging engine driven by a contrastive negative-domain schedule.

Initial design point:

- Try Twitter first with `p=16,q=6` or `p=16,q=5`, because the resident footprint is smaller than FB86M and the one-admit bound is materially better than the current baseline.
- For FB86M, `p=24,q=6` is the most interesting simulator point, but it requires a new scheduler because current GE2 CUSTOM is specialized for `q=4`.

Why this remains novel:

- It is not "BETA with a different buffer size." The simulator suggests changing the scheduled object from state-level positive coverage to contrastive microtasks with explicit negative-domain tiles.
- COVER remains the positive-bucket skeleton; the new machinery optimizes the contrastive domain and transfer liveness that COVER does not model.

### Hypothetical 2/4-GPU Simulator Results

Full outputs:

- `/dev/shm/smansou2_ge2/contrastive_tiled_cover_sim_20260407/fb86m_1_2_4.tsv`
- `/dev/shm/smansou2_ge2/contrastive_tiled_cover_sim_20260407/twitter_1_2_4.tsv`
- `/dev/shm/smansou2_ge2/contrastive_tiled_cover_sim_20260407/fb86m_4gpu_calibrated.tsv`
- `/dev/shm/smansou2_ge2/contrastive_tiled_cover_sim_20260407/twitter_2gpu_calibrated.tsv`
- `/dev/shm/smansou2_ge2/contrastive_tiled_cover_sim_20260407/twitter_4gpu_calibrated.tsv`

FB86M 4-GPU calibrated against measured `49.822 s` epoch and `11.816 s` swap:

| Candidate | States/lane | Resident HBM | One-partition stage | Same-style epoch | One-admit bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p=24,q=6` | `5` | `16.029 GiB` | `5.343 GiB` | `50.478 s` | `40.500 s` |
| `p=32,q=6` | `9` | `12.022 GiB` | `4.007 GiB` | `56.715 s` | `41.748 s` |
| `p=24,q=5` | `7` | `13.357 GiB` | `5.343 GiB` | `52.973 s` | `41.748 s` |
| `p=48,q=6` | `19` | `8.014 GiB` | `2.671 GiB` | `66.069 s` | `43.619 s` |

Twitter 2-GPU calibrated against measured `170.088 s` epoch and `18.644 s` swap:

| Candidate | States/lane | Resident HBM | One-partition stage | Same-style epoch | One-admit bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p=16,q=6` | `4` | `11.638 GiB` | `3.879 GiB` | `161.284 s` | `153.412 s` |
| `p=16,q=5` | `6` | `9.698 GiB` | `3.879 GiB` | `164.564 s` | `154.724 s` |
| `p=24,q=6` | `10` | `7.758 GiB` | `2.586 GiB` | `171.124 s` | `155.380 s` |
| Current `p=16,q=4` | `10` | `7.758 GiB` | `3.879 GiB` | `170.088 s` | `157.348 s` |

Twitter 4-GPU calibrated against measured `74.057 s` epoch and `19.984 s` swap:

| Candidate | States/lane | Resident HBM | One-partition stage | Same-style epoch | One-admit bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p=16,q=6` | `2` | `11.638 GiB` | `3.879 GiB` | `61.983 s` | `55.655 s` |
| `p=16,q=5` | `3` | `9.698 GiB` | `3.879 GiB` | `66.730 s` | `57.237 s` |
| `p=24,q=6` | `5` | `7.758 GiB` | `2.586 GiB` | `75.167 s` | `58.292 s` |
| Current `p=16,q=4` | `5` | `7.758 GiB` | `3.879 GiB` | `74.057 s` | `60.401 s` |

Interpretation for multi-GPU:

- The same candidate family extends to multi-GPU: `q=5/6` with one negative tile per microtask still creates free slots for bounded prefetch.
- `p=16,q=6` looks best for Twitter on 2/4 GPUs because it keeps state count very low and still fits a one-partition stage.
- `p=24,q=6` looks best for FB86M 4-GPU because it makes one-partition staging fit where current `p=16` does not.
- These are not final runtime predictions. They assume a scheduler can realize one-admit/tile movement while preserving the contrastive negative distribution. The next code step must therefore be a scheduler prototype, not another flag.

## Prototype: Greedy Positive-Cover Scheduler

Code:

- `GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING=1`
- Implemented in `getGreedyCoverEdgeBucketOrdering(...)`.
- This is a positive-bucket scheduler skeleton, not the full contrastive negative-tile executor.

What it does:

- Generates candidate buffer states of size `q`.
- Greedily covers all directed positive buckets exactly once through `greedyAssignEdgeBucketsToBuffers`.
- Reorders the selected states to maximize adjacent partition retention.
- Leaves the default COVER path unchanged unless the flag is enabled.

Scheduler sanity:

| Setup | States | Assigned buckets | Avg retained partitions | Multi-GPU disjoint? |
| --- | ---: | ---: | ---: | --- |
| Twitter-style `p=16,q=5` | `16` | `256` | `2.067` after reorder, `1.333` before | No for 2 GPU |
| Twitter-style `p=16,q=6` | `11` | `256` | `2.800` after reorder, `1.600` before | No for 2 GPU |
| FB-style candidate `p=24,q=6` | `25` | `576` | `1.667` before reorder in the first analyzer pass | Not validated |

Training probes on Twitter single GPU:

| Run | Epoch runtime | Swap count | Swap update | Map lookup | Negative sample | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline q4 reference | `223.085 s` avg over 3 epochs | about `19` | `61.603 s` | about `44 s` | about `7.93 s` | Current best Twitter baseline |
| Greedy cover `p=16,q=6`, before overlap reorder | `209.968 s` | `10` | `48.064 s` | `39.899 s` | `15.381 s` | First q6 probe |
| Greedy cover `p=16,q=5` | `217.194 s` | `15` | `55.810 s` | `41.543 s` | `12.080 s` | Slower than q6; log truncated by `/dev/shm` full, values recovered from stdout |
| Greedy cover `p=16,q=6`, after overlap reorder | `200.905 s` | `10` | `38.628 s` | `39.915 s` | `15.489 s` | Best scheduler probe so far |
| Greedy cover `p=16,q=6` plus CUDA DEG chunk-exclusion sampler | `199.014 s` | `10` | `37.751 s` | `24.189 s` | `2.230 s` | Reduces sampler timer strongly, but only small end-to-end gain; changes RNG implementation and needs accuracy validation |

Logs:

- `/dev/shm/smansou2_ge2/twitter_greedy_cover_q6_1e_20260407`
- `/dev/shm/smansou2_ge2/twitter_greedy_cover_q5_1e_20260407`
- `/dev/shm/smansou2_ge2/twitter_greedy_cover_q6_reorder_1e_20260407`
- `/dev/shm/smansou2_ge2/twitter_greedy_cover_q6_negtile_profile_1e_20260407`
- `/dev/shm/smansou2_ge2/twitter_greedy_cover_q6_deg_cuda_sample_clean_1e_20260407`

Negative-tile profile on the q6 reordered run:

- Added default-off profiler flag: `GEGE_NEGATIVE_TILE_PROFILE=1`.
- For Twitter `q=6`, ran with `GEGE_NEGATIVE_TILE_PROFILE_NUM_TILES=6` and `GEGE_NEGATIVE_TILE_PROFILE_MAX_CALLS=8`.
- Profiled run time: `200.427 s`, effectively matching the non-profile q6 run because only the first eight sampler calls were synchronized for histogram logging.
- `negative_sample`: `15.466 s`
- `uniform_randint`: `1.312 s`
- `sample_edge_randint`: `13.335 s`
- `materialize`: `0.688 s`
- `filter`: `0.048 s`

Profiler finding:

- Uniform RNS samples are balanced across the six resident tiles, as expected.
- The DEG-local half is not balanced. In the first sampled batches, the full `25,000` DEG-derived candidates land in a single tile while uniform contributes about `4,100-4,300` samples per tile.
- Example line: `uniform_counts=[4247,4159,4111,4091,4233,4159]`, `deg_counts=[0,0,25000,0,0,0]`.
- This happens because the current non-global DEG path samples edge rows from the active positive batch and then materializes the sampled endpoint. Under a bucketed positive schedule, that endpoint naturally belongs to the positive bucket's tile/domain.

CUDA DEG chunk-exclusion sampler probe:

- Added default-off flag: `GEGE_DEG_CHUNK_EXCLUSION_CUDA_SAMPLE=1`.
- The CUDA path generates the chunk-excluded `sample_edge_ids` with one fixed kernel instead of a small Torch graph of `rand`, multiply, floor, cast, compare, and shift operations.
- Added verifier flag: `GEGE_DEG_CHUNK_EXCLUSION_CUDA_SAMPLE_VERIFY=1`, which checks range and chunk-exclusion invariants for the first configured calls.
- Clean q6 Twitter run with the CUDA sampler reduced `sample_edge_randint` from `13.335 s` to `0.163 s`, and `negative_sample` from `15.466 s` to `2.230 s`.
- End-to-end epoch time only improved from `200.427 s` in the profiled q6 reference and `200.905 s` in the non-profile q6 reference to `199.014 s`.
- Interpretation: this removes a real sampler construction cost, but much of the benefit appears as timer migration into later GPU work. It should be treated as an implementation optimization, not the main new machinery.
- Caveat: the CUDA sampler preserves the intended chunk-exclusion support but uses a different random number implementation than Torch. It requires 30-epoch accuracy validation before being claimed as a stable result.

Interpretation:

- The schedule change is real: fewer states and better partition retention cut Twitter swap from about `61.6 s` to `38.6 s`.
- The current prototype is still not the full proposed machinery. Negative sampling gets more expensive because the resident state is larger (`q=6`), so the next step must tile the negative domain rather than sampling from the whole resident state.
- The current greedy state order is not multi-GPU disjoint for 2 GPUs. Multi-GPU requires group-aware generation, not just single-lane reorder.
- The negative-domain work should target the DEG-local candidate path first. Tiling only the uniform RNS half would not address the measured q6 negative wall, because `sample_edge_randint` dominates.
- This confirms the paper direction: the positive-cover scheduler alone helps, but the actual new machinery must be contrastive-domain-aware.

### Accuracy Gate: LJ q6 Greedy + CUDA DEG Sampler

Run:

- `/dev/shm/smansou2_ge2/lj_q6_greedy_deg_cuda_30e_eval_20260407`

Flags relative to the validated LJ stack:

- `buffer_capacity=6`
- `GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING=1`
- `GEGE_DEG_CHUNK_EXCLUSION_CUDA_SAMPLE=1`

Timing:

| Metric | Value |
| --- | ---: |
| Avg epoch, all 30 | `8.345 s` |
| Avg epoch, excluding epoch 1 | `8.311 s` |
| Last 10 epochs | `8.336 s` |
| `batch_fetch` | `5.239 s` |
| `swap_update` | `2.201 s` |
| `edge_sample` | `3.031 s` |
| `map_lookup` | `2.411 s` |
| `negative_sample` | `0.340 s` |

Exact eval:

| Metric | Validated q4 LJ | q6 greedy + CUDA DEG |
| --- | ---: | ---: |
| MRR | `0.129537` | `0.075410` |
| Hits@10 | `0.3108` | `0.1835` |
| Mean Rank | not recorded here | `353327.874700` |

Conclusion:

- The q6 positive-cover schedule is slightly faster than the validated q4 LJ stack (`8.31 s` vs `8.50 s` steady-state), mostly because the state count and swap work drop.
- The accuracy drop is too large. This is not a valid optimization as-is.
- The failure confirms that changing the resident contrastive domain changes the training signal even when the positive buckets are covered exactly.
- Do not claim `q=6` greedy ordering as an accuracy-preserving result. Use it as evidence that the paper contribution must include negative-domain correction, not just positive-cover rescheduling.

## Visibility-Decoupled Ghost Prefetch

Motivation:

- Visible `q=6` states reduce swap but change the resident negative domain and collapse LJ accuracy.
- The safer idea is to keep the logical sampler-visible state at GE2's validated `q=4`, but add hidden physical GPU frames for future partitions.
- A prefetched partition in a ghost frame is not visible to the sampler until the state boundary. This preserves the q4 contrastive domain while moving some H2D admit traffic off the critical path.

Simulator extension:

- Added `--ghost-capacity-parts` to `scripts/contrastive_tiled_cover_sim.py`.
- A value of `K` models `K` partition-equivalent hidden physical frames.
- The model assumes a ghost frame can hide up to `K` admitted partitions before the boundary and up to `K` evicted partitions after the boundary. This is an optimistic bound for a true physical-frame/page-table design.

Simulator outputs:

- `/dev/shm/smansou2_ge2/ghost_prefetch_sim_20260407/twitter_q4_ghost_single.tsv`
- `/dev/shm/smansou2_ge2/ghost_prefetch_sim_20260407/fb86m_q4_ghost_single.tsv`

Twitter q4-visible simulator result:

| Ghost capacity | Extra HBM | Total resident | Exposed boundary epoch bound |
| ---: | ---: | ---: | ---: |
| `0` | `0.000 GiB` | `7.758 GiB` | `223.085 s` |
| `1` | `1.940 GiB` | `9.698 GiB` | `203.577 s` |
| `2` | `3.879 GiB` | `11.638 GiB` | `184.070 s` |
| `3` | `5.819 GiB` | `13.577 GiB` | `164.562 s` |

FB86M q4-visible simulator result:

| Candidate | Ghost capacity | Extra HBM | Total resident | Exposed boundary epoch bound |
| --- | ---: | ---: | ---: | ---: |
| current `p=16,q=4` | `0` | `0.000 GiB` | `16.029 GiB` | `166.393 s` |
| current `p=16,q=4` | `1` | `4.007 GiB` | `20.036 GiB` | `145.673 s` |
| candidate `p=24,q=4` | `1` | `2.671 GiB` | `13.357 GiB` | `166.393 s` |
| candidate `p=24,q=4` | `0.5` | `1.336 GiB` | `12.022 GiB` | `182.751 s` |

Interpretation:

- Twitter is the better first target for full-partition ghosting. Three q4-invisible ghost partitions fit in the simulator's conservative `6 GiB` hidden-frame budget and give a large boundary bound.
- FB86M current `p=16` can maybe fit one hidden partition in theory, but it is much tighter. A paged/tiled ghost design or a different `p` is safer for FB.

Prototype: limited async admit preload

- Added `GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD_LIMIT_PARTS`.
- This is a partial ghost-admit prototype that reuses the existing hidden GPU admit stage, but only preloads the first `K` admitted partitions.
- It preserves q4 logical visibility and falls back for the remaining admits.
- It is not the full ghost-frame design because it still copies the staged admit data into evict slots at the boundary and still pays evict D2H before those slots can be overwritten.

Twitter q4-visible train-only probes:

| Run | Epoch runtime | Swap update | Batch fetch | Edge sample | Map lookup | Negative sample | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q4 no-async baseline | `223.085 s` | `61.603 s` | not listed here | not listed here | about `44 s` | about `7.93 s` | 3-epoch average |
| q4 ghost-admit `K=1` | `220.592 s` | `57.659 s` | `112.024 s` | `54.267 s` | `43.651 s` | `8.311 s` | 1 epoch, timing logs off |
| q4 ghost-admit `K=3` | `215.726 s` | `49.644 s` | `106.244 s` | `56.503 s` | `43.645 s` | `8.931 s` | 1 epoch, `38` preload consumes, `0` skips |

Conclusion:

- The visibility-decoupled direction is promising: q4-visible `K=3` improves Twitter by about `7.36 s`/epoch and cuts `swap_update` by about `11.96 s`, without changing the visible negative domain.
- The current prototype only hides admit-side work and still pays evict D2H and GPU-stage install copies at the boundary. This explains why it does not reach the simulator's `164.6 s` optimistic bound.
- The next real machinery is a physical-frame/page-table buffer: publish a ghost frame by remapping a logical slot to a physical frame, and let the old physical frame become a delayed-evict ghost. That avoids the boundary GPU-stage install copy and can also move evict writeback off the boundary.

## Bucket-Granular Dirty/Delta Profiling

Motivation:

- A natural alternative to whole-partition swap is to update only rows or tiles touched by each bucket.
- This would be novel if it turned GE2's state-level dump into an asynchronous bucket-level delta stream.
- The risk is that full-resident negative sampling may touch almost every row/tile, making dirty writeback equivalent to full writeback.

Instrumentation:

- Added default-off profiler flags:
  - `GEGE_BUCKET_DELTA_PROFILE=1`
  - `GEGE_BUCKET_DELTA_PROFILE_TILE_ROWS=4096`
  - `GEGE_BUCKET_DELTA_ROW_REUSE_PROFILE=1`
  - `GEGE_BUCKET_DELTA_ROW_REUSE_PROFILE_MAX_STATES=1`
  - `GEGE_BUCKET_DELTA_ROW_REUSE_PROFILE_MAX_CALLS_PER_STATE=0`
- The first profiler estimates per-batch row bytes versus full dirty tile bytes.
- The second profiler coalesces all row IDs within a state to estimate whether a row-level delta log would remain sparse after repeated updates.

Twitter q4-visible run:

- `/dev/shm/smansou2_ge2/twitter_bucket_delta_profile_q4_1e_20260408`
- `/dev/shm/smansou2_ge2/twitter_bucket_delta_row_reuse_q4_state0_1e_20260408`

Results:

| Probe | Result |
| --- | ---: |
| Per-batch profiler calls | `512` |
| Per-batch active rows | `30.69 M` |
| Per-batch row delta bytes | `23.41 GiB` |
| Per-batch dirty-tile bytes (`4096` rows/tile) | `4068.62 GiB` |
| Dirty-tile/full-state equivalent | `1.000x` |
| State-0 row-reuse calls | `1325` |
| State-0 raw row updates | `79.06 M` |
| State-0 unique rows after coalescing | `10.32 M` |
| State-0 unique/full visible state | `0.991x` |
| State-0 unique row delta bytes | `7.87 GiB` |
| State-0 full visible state bytes | `7.94 GiB` |

Interpretation:

- Tile-granular dirty commit is not useful for Twitter q4 with the current negative sampler. A single batch already touches all `4096`-row tiles in the visible state.
- Row-level coalescing helps locally (`79.06 M` raw row references collapse to `10.32 M` unique rows), but the unique set is still `99.1%` of the full visible state after one state.
- Therefore, "write back only changed rows" does not materially reduce Twitter swap volume under the current full-resident RNS/DEG semantics.
- This negative result is important: the swap problem is not primarily sparse dirty-row compression. The current contrastive domain makes almost the whole visible state live and dirty.
- A useful new pipeline needs either visibility-decoupled physical frames, state-local batch-map construction, or sampler/domain machinery that preserves accuracy while reducing liveness. Bucket-level dirty writeback alone is not enough.

## Shared Negative Sampling Track

Source:

- Kochsiek and Gemulla, "Parallel Training of Knowledge Graph Embedding Models: A Comparison of Techniques" (`/home/smansou2/p633-kochsiek.pdf`).

Why we looked at it:

- The paper shows that negative samples can dominate the set of unique entity accesses per batch.
- Shared sampling can reduce unique negative entities from roughly `B * Nneg` to `Nneg`, lowering communication and epoch time.
- Local sampling can improve efficiency, but static local pools can hurt quality; dynamic/repartitioned local pools are safer.
- This matches our GE2 profiling: Twitter q4 touches nearly the whole visible state through negative sampling, so swap/delta optimizations alone do not stay sparse.

Important GE2-specific caveat:

- GE2 already has chunk-shared negative sampling: `uniform_ids` is shaped by chunk rather than by edge.
- Therefore, the next useful mechanism is not simply "turn on shared sampling"; it should be a temporal/materialized shared negative pool that reuses negative entity IDs across a short batch window while remaining dynamic across windows and epochs.
- The existing `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES` path is only a partial mechanism. It reuses the negative plan/pool, but it does not yet implement a fully materialized contrastive pool designed to reduce dirty-row liveness.

LJ gate: existing state negative pool, `W=2`

Runs:

- W=2 run: `/dev/shm/smansou2_ge2/lj_state_neg_pool_w2_30e_eval_20260408`
- Paired no-pool baseline: `/dev/shm/smansou2_ge2/lj_state_neg_pool_baseline_30e_approx_20260408`

Flags:

- W=2 adds `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES=2`.
- Baseline uses `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES=0`.
- Both use the validated LJ q4 stack and keep failed q6/frontload paths disabled.

Fast sampled-eval result:

| Run | Avg epoch runtime | Approx MRR | Approx Hits@10 | State pool hits |
| --- | ---: | ---: | ---: | ---: |
| Baseline no pool | `8517.63 ms` | `0.427579` | `0.551282` | `0` |
| W=2 existing pool | `8527.53 ms` | `0.428350` | `0.552074` | about `450-470` per epoch |

Interpretation:

- W=2 did not collapse the sampled-eval quality, which is a useful sanity signal.
- W=2 also did not improve epoch time; it was about `+10 ms` slower than the paired baseline.
- The current implementation is therefore not the mechanism we need for large-scale swap reduction. It reduces some plan/random generation work but does not materially change the negative entity liveness pattern.
- A full filtered/chunked eval was attempted, but it was too slow for an inner-loop gate. We should reserve it for a candidate that first shows real timing/dirty-liveness improvement under the fast paired protocol.

Next implication:

- If we continue this paper track, implement a new materialized windowed negative pool and profile dirty-row uniqueness, rather than only increasing `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES`.
- The design should expose explicit knobs such as `pool_window_batches`, `pool_size`, and `mix_fraction` so we can retain part of the original per-batch sampler while reducing unique negative row pressure.

### Liveness-Aware Negative Cache Simulator

Motivation:

- The failed dirty/delta writeback probe showed that Twitter q4 dirties `99.1%` of the visible state within one state.
- The important target is therefore not only sampler runtime. It is the unique negative rows/tiles touched by the sampler, because that controls whether bucket-level delta commit or staged swap can remain sparse.
- A cache that still draws a small fresh sample from the full visible state can destroy tile sparsity. The simulator explicitly tests this.

Simulator:

- Added `/home/smansou2/newCode/ge2/dandelion-dev/scripts/liveness_negative_cache_sim.py`.
- Outputs are in `/dev/shm/smansou2_ge2/liveness_negative_cache_sim_20260408`.
- Knobs:
  - `window_batches`
  - `pool_rows`
  - `tile_budget`
  - `fresh_mix`
  - `fresh_domain`
  - `negative_unique_rows_per_batch`

Key simulator results:

| Dataset | Fresh domain | Best shown knobs | Unique rows vs baseline | Dirty tiles vs baseline | Interpretation |
| --- | --- | --- | ---: | ---: | --- |
| Twitter | full visible state | `W=32`, `pool_rows=4096`, `tile_budget=4`, `fresh_mix=0` | `0.0161x` | `0.0705x` | Pure cached/tiled pool can cut liveness strongly, but has no exploration. |
| Twitter | full visible state | any nonzero full-domain fresh mix | varies | about `1.0000x` | Even small full-domain fresh exploration dirties all tiles, so it defeats the liveness goal. |
| Twitter | active tile domain | `W=32`, `pool_rows=4096`, `tile_budget=4`, `fresh_mix=0.05` | `0.0702x` | `0.0705x` | Promising liveness point, but it changes proposal support and needs an accuracy/correction gate. |
| FB86M | active tile domain | `W=32`, `pool_rows=4096`, `tile_budget=4`, `fresh_mix=0.05` | `0.0148x` | `0.0076x` | Strong simulated liveness reduction because the visible state is much larger. |

Interpretation:

- A materialized negative cache can theoretically make row/tile liveness sparse enough for a delta/swap pipeline, but only if both cached and fresh negatives are drawn from the scheduled active domain.
- Full-domain exploration preserves the original support locally, but it touches all tiles quickly and eliminates the systems benefit.
- This creates a clear research problem: preserve ranking quality while using an active-domain proposal. The likely solution is not naive local sampling; it needs mixing, importance/logQ correction, or hardness-aware admission.

### Active-Tile Uniform Sampling Prototype

Implementation:

- Added default-off flags in `negative.cpp`:
  - `GEGE_LIVENESS_UNIFORM_TILE_SAMPLING=1`
  - `GEGE_LIVENESS_UNIFORM_TILE_ROWS=4096`
  - `GEGE_LIVENESS_UNIFORM_TILE_BUDGET=4`
- The first prototype only changes the uniform-RNS component of the existing chunked negative plan. DEG-local candidate sampling is intentionally left unchanged.
- It composes with the existing `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES` path, so the first tested mechanism is `W=32` cached active-tile uniform negatives.

LJ 2-epoch sanity:

| Run | Epoch 2 runtime | Epoch 2 batch fetch | Epoch 2 map lookup | Epoch 2 negative sample | Epoch 2 swap update |
| --- | ---: | ---: | ---: | ---: | ---: |
| W=32 default-off active-tile control | `8379 ms` | `5233.6 ms` | `1239.3 ms` | `746.1 ms` | `2997.2 ms` |
| W=32 active-tile uniform, budget 4 | `8325 ms` | `5132.9 ms` | `1227.3 ms` | `733.5 ms` | `2928.3 ms` |

Interpretation:

- This is only an execution/timing sanity check. It shows the prototype runs and does not create an immediate runtime regression.
- The result is not an accuracy result and should not be used as evidence of correctness.
- The next gate is a longer LJ train plus fast sampled evaluation. If that holds, run dirty-row/tile liveness profiling on Twitter to see whether changing only the uniform component moves the systems bottleneck.

LJ 30-epoch sampled-eval gate:

- Run: `/dev/shm/smansou2_ge2/lj_liveness_uniform_active_tile_w32_budget4_30e_approx_20260408`
- Additional flags over the validated q4 LJ stack:
  - `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES=32`
  - `GEGE_LIVENESS_UNIFORM_TILE_SAMPLING=1`
  - `GEGE_LIVENESS_UNIFORM_TILE_ROWS=4096`
  - `GEGE_LIVENESS_UNIFORM_TILE_BUDGET=4`
  - `GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING=0`
  - `GEGE_DEG_CHUNK_EXCLUSION_CUDA_SAMPLE=0`
  - `GEGE_STATE_NEGATIVE_TAPE=0`

Result:

| Run | Avg epoch all | Avg epoch excluding epoch 1 | Last 10 avg | Approx MRR | Approx Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired no-pool sampled baseline | `8517.63 ms` | not recomputed in this note | not recomputed in this note | `0.427579` | `0.551282` |
| W=2 existing pool | `8527.53 ms` | not recomputed in this note | not recomputed in this note | `0.428350` | `0.552074` |
| W=32 active-tile uniform, budget 4 | `8507.63 ms` | `8475.48 ms` | `8482.90 ms` | `0.439919` | `0.552183` |

Breakdown for W=32 active-tile uniform:

| Metric | Avg all epochs | Avg excluding epoch 1 |
| --- | ---: | ---: |
| `batch_fetch` | `5280.02 ms` | `5261.42 ms` |
| `edge_sample` | `2269.17 ms` | `2249.82 ms` |
| `negative_sample` | `712.73 ms` | `707.20 ms` |
| `map_lookup` | `1308.47 ms` | `1309.66 ms` |
| `swap_update` | `3000.28 ms` | `3001.13 ms` |

Interpretation:

- This active-tile uniform prototype does not hurt the fast sampled-eval metric, and runtime remains within noise of the validated LJ q4 stack.
- It is not a speedup by itself. That is expected because it only changes the uniform component; DEG-local negatives still dominate a large part of row/tile liveness and sample-edge materialization.
- This result is a permission gate to do large-dataset liveness profiling, not a final optimization. The next diagnostic is Twitter q4 row-reuse profiling with the same active-tile uniform path to test whether dirty-row/tile coverage falls below the previous `99.1%` full-state level.

Twitter q4 state-0 row-reuse profile:

- Run: `/dev/shm/smansou2_ge2/twitter_liveness_uniform_active_tile_w32_budget4_row_reuse_q4_state0_1e_20260408`
- This is train-only and profiling-enabled; runtime includes `11.879 s` of row-reuse profiling overhead, so it should not be used as a clean timing result.
- Additional flags over the Twitter q4 stack:
  - `GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES=32`
  - `GEGE_LIVENESS_UNIFORM_TILE_SAMPLING=1`
  - `GEGE_LIVENESS_UNIFORM_TILE_ROWS=4096`
  - `GEGE_LIVENESS_UNIFORM_TILE_BUDGET=4`
  - `GEGE_BUCKET_DELTA_ROW_REUSE_PROFILE=1`
  - `GEGE_BUCKET_DELTA_ROW_REUSE_PROFILE_MAX_STATES=1`
  - `GEGE_BUCKET_DELTA_ROW_REUSE_PROFILE_MAX_CALLS_PER_STATE=0`

Result:

| Probe | State-0 raw rows | State-0 unique rows | Unique/full visible state | Unique row delta | Full visible state |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline q4 current sampler | `79.06 M` | `10.32 M` | `0.991x` | `7.87 GiB` | `7.94 GiB` |
| W=32 active-tile uniform, budget 4 | `63.03 M` | `8.37 M` | `0.804x` | `6.24 GiB` | `7.76 GiB` |

Timing/breakdown from the profiling run:

| Metric | Value |
| --- | ---: |
| Epoch runtime | `225.597 s` |
| `batch_fetch` | `106.738 s` |
| `edge_sample` | `45.146 s` |
| `map_lookup` | `36.152 s` |
| `negative_sample` | `7.256 s` |
| `swap_update` | `61.535 s` |
| row-reuse profiler overhead | `11.879 s` |

Interpretation:

- Restricting only the uniform component to an active tile set gives a real liveness signal: `unique/full` improves from `0.991x` to `0.804x` for Twitter state 0.
- This is still not sparse enough for delta writeback or tile-level swap to become the main machinery. About `80%` of the visible state still becomes dirty.
- The remaining dirtiness is expected because DEG-local negative sampling is unchanged. To make the sampler useful for the systems pipeline, the next prototype must control the DEG/local negative component as well, with an accuracy gate and likely a correction/mixing strategy.

### Active-Tile DEG Sampling Prototype: Rejected

Implementation:

- Added default-off flag in `negative.cpp`:
  - `GEGE_LIVENESS_DEG_TILE_SAMPLING=1`
- This experiment reuses the same active-tile set as `GEGE_LIVENESS_UNIFORM_TILE_SAMPLING=1` and directly samples the DEG/local negative columns from those active tiles.
- It intentionally changes the DEG/local proposal distribution. The goal was to test whether controlling the remaining negative component can make row/tile liveness sparse enough for a later delta/swap pipeline.

LJ 2-epoch mechanical gate:

- Run: `/dev/shm/smansou2_ge2/lj_liveness_uniform_deg_tile_w32_budget4_2e_20260408`
- Additional flags over the W=32 active-tile uniform path:
  - `GEGE_LIVENESS_DEG_TILE_SAMPLING=1`

| Run | Epoch 2 runtime | Epoch 2 batch fetch | Epoch 2 map lookup | Epoch 2 negative sample | Epoch 2 swap update |
| --- | ---: | ---: | ---: | ---: | ---: |
| W=32 active-tile uniform, budget 4 | `8325 ms` | `5132.9 ms` | `1227.3 ms` | `733.5 ms` | `2928.3 ms` |
| W=32 active-tile uniform+DEG, budget 4 | `8775 ms` | `5520.5 ms` | `1328.7 ms` | `521.5 ms` | `3417.9 ms` |

LJ 30-epoch accuracy gate:

- Run: `/dev/shm/smansou2_ge2/lj_liveness_uniform_deg_tile_w32_budget4_30e_approx_20260408`
- Training completed, but unchunked eval OOMed; chunked-rank eval was used for the reported metric.

| Run | Avg epoch all | Eval mode | MRR | Hits@10 |
| --- | ---: | --- | ---: | ---: |
| Validated q4 baseline | `8499.10 ms` excluding epoch 1 | exact/chunked | `0.129537` | `0.3108` |
| W=32 active-tile uniform, budget 4 | `8507.63 ms` | sampled gate only | `0.439919` | `0.552183` |
| W=32 active-tile uniform+DEG, budget 4 | `8903.00 ms` | exact/chunked after unchunked eval OOM | `0.050724` | `0.1177` |

Breakdown for W=32 active-tile uniform+DEG:

| Metric | Avg all epochs |
| --- | ---: |
| `swap_update` | `3448.89 ms` |
| `negative_sample` epoch 30 | `497.65 ms` |
| `map_lookup` epoch 30 | `1403.90 ms` |
| `sample_edge_randint` epoch 30 | `189.82 ms` |
| `uniform_randint` epoch 30 | `42.37 ms` |

Interpretation:

- The stronger active-domain DEG path successfully reduces the measured negative-sampler time, but it makes the epoch slower overall and destroys LJ ranking quality relative to the validated exact q4 baseline.
- This is the same failure mode as the previous visible-domain and evict-avoidance attempts: changing proposal support for negatives is not a safe systems optimization by itself.
- The active-domain simulator remains useful as a liveness target, but the implementation needs a correction/mixing/hardness mechanism before it can be considered a training algorithm. Do not use `GEGE_LIVENESS_DEG_TILE_SAMPLING=1` as a performance result.

### Versioned Frame Cache: First Twitter q4 Remap Prototype

Goal:

- Preserve the exact logical q4 visible state.
- Add one hidden physical GPU frame via `GEGE_FRAME_CACHE_HIDDEN_FRAMES=1`.
- Prefetch one admitted partition into the hidden frame and publish it by remapping a logical slot to that frame instead of installing by copy into the old visible slot.

Important first bug that had to be fixed:

- The initial frame-cache patch changed `getNumInMemory()` to report the physical frame count (`q + hidden`) instead of the logical visible q4 size.
- That leaked hidden frame `4` into the advertised in-memory state, and the first Twitter q4 run failed immediately with:
  - `MemPartitionBuffer::translateLogicalIndicesToPhysical_ (indexRead) received logical slots [0, 4] outside visible range [0, 3]`
- Fix: restore `getNumInMemory()` to the logical visible size only. Hidden frames are physically allocated, but they are not part of the visible in-memory node domain.

After that fix, the remap prototype ran through the first state cleanly and could be measured.

Current-code Twitter q4 baseline, 3 epochs, train-only:

- Run: `/dev/shm/smansou2_ge2/twitter_framecache_measure_baseline_3e_20260408`

| Epoch | Runtime | `batch_fetch` | `edge_sample` | `map_lookup` | `negative_sample` | `swap_update` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `225.427 s` | `116.133 s` | `56.510 s` | `43.296 s` | `9.098 s` | `59.530 s` |
| 2 | `222.056 s` | `114.128 s` | `54.574 s` | `43.483 s` | `9.107 s` | `59.458 s` |
| 3 | `222.208 s` | `114.214 s` | `54.869 s` | `43.744 s` | `9.117 s` | `59.254 s` |
| Avg | `223.230 s` | `114.825 s` | `55.318 s` | `43.508 s` | `9.107 s` | `59.414 s` |

Twitter q4 with one hidden frame, 2 completed epochs, train-only:

- Run: `/dev/shm/smansou2_ge2/twitter_framecache_measure_hidden1_3e_20260408`
- Epoch 3 was intentionally stopped once the regression was already clear.

| Epoch | Runtime | `batch_fetch` | `edge_sample` | `map_lookup` | `negative_sample` | `swap_update` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `228.010 s` | `201.035 s` | `141.712 s` | `66.616 s` | `71.543 s` | `59.237 s` |
| 2 | `224.899 s` | `199.354 s` | `140.122 s` | `66.891 s` | `71.759 s` | `59.146 s` |
| Avg | `226.455 s` | `200.194 s` | `140.917 s` | `66.754 s` | `71.651 s` | `59.192 s` |

Interpretation:

- The prototype is now past the first correctness blocker: it can train through the early Twitter q4 states with hidden-frame publish enabled.
- But it is not a performance win in its current form.
- `swap_update` stays flat at about `59.2 s`, so the current remap path is not yet removing visible boundary work.
- Worse, `edge_sample` explodes from `~55.3 s` to `~140.9 s`, driven by:
  - `map_lookup` rising from `~43.5 s` to `~66.8 s`
  - `negative_sample` rising from `~9.1 s` to `~71.7 s`

Most likely root cause:

- The current frame-cache implementation introduces hot-path logical-to-physical translation and/or state metadata side effects during normal batch construction and reads.
- In other words, it changes the steady-state read/sampling path, not only the boundary publish path.
- That is the wrong shape for the design. A correct versioned frame cache should make hidden-frame handling almost invisible during ordinary q4 batch execution and pay work mainly at prefetch/publish time.

Conclusion from this prototype:

- The architecture direction is still plausible, but the current implementation strategy is wrong.
- Do not proceed by layering more logic into per-read/per-update translation.
- The next version should keep the hot path in pure logical q4 coordinates and move frame indirection to the boundary metadata/publish path only.

### Versioned Frame Cache Second Prototype: Hot-Path Translation Moved Out of Storage

We then reworked the prototype so ordinary storage reads/updates no longer translate
logical q4 local ids to physical frame offsets inside `indexRead` / `indexAdd`.
Instead, the dataloader translates visible local ids once when materializing the
batch tensors that touch embeddings or optimizer state.

Runs:

- Baseline q4 rerun: `/dev/shm/smansou2_ge2/twitter_framecache_rerun_baseline_2e_20260408/train.log`
- Hidden-frame rerun: `/dev/shm/smansou2_ge2/twitter_framecache_rerun_hidden1_2e_20260408/train.log`

Two-epoch comparison:

| Config | Avg runtime | `batch_fetch` | `edge_sample` | `map_lookup` | `negative_sample` | `swap_update` | `runtime - batch_fetch` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q4 baseline | `223.947 s` | `115.259 s` | `55.417 s` | `43.159 s` | `9.236 s` | `59.746 s` | `108.688 s` |
| q4 + hidden frame `k=1` | `225.546 s` | `74.560 s` | `15.210 s` | `4.119 s` | `8.333 s` | `59.254 s` | `150.986 s` |

What changed:

- The catastrophic hot-path regression is gone.
- `edge_sample` falls back from `140.917 s` in the first prototype to `15.210 s`.
- `map_lookup` drops from `66.754 s` in the first prototype to `4.119 s`.
- `negative_sample` returns to baseline range (`8.333 s` vs `9.236 s`).

What did **not** change:

- `swap_update` stays flat: `59.746 s -> 59.254 s`.
- End-to-end epoch time is still not better: `223.947 s -> 225.546 s`.

Interpretation:

- Moving logical-to-physical translation out of the storage hot path fixed the
  original implementation bug.
- But the current hidden-frame path still does not remove visible boundary work.
- Instead, it shifts about `40.7 s` out of `batch_fetch` and into the downstream
  GPU-side `runtime - batch_fetch` bucket, which now rises from `108.688 s` to
  `150.986 s`.
- So the second prototype is mechanically sane, but still architecturally wrong:
  it changes where work is paid, not how much work the epoch performs.

Updated conclusion:

- The frame-cache architecture is still a **go**.
- The second prototype is a **mechanically fixed no-go**:
  - it no longer corrupts the steady-state q4 sampling path,
  - but it still fails to turn hidden-frame publish into a real `swap_update`
    reduction.
- The next implementation must keep the ordinary q4 batch execution path almost
  identical to baseline and push frame-cache work into prefetch, publish, and
  delayed-commit boundaries only.

### Versioned Frame Cache Kill-Test: Did We Actually Remove Boundary Admit Copies?

To avoid more open-ended work on an unproven path, we added explicit boundary
instrumentation:

- `visible_install_rows`: rows copied into visible q4 slots at the boundary
- `hidden_publish_rows`: rows admitted by hidden-frame publish/remap instead of
  visible-slot install

Probe:

- `/dev/shm/smansou2_ge2/twitter_framecache_hidden1_timing_1e_20260408/train.log`
- flags included:
  - `GEGE_FRAME_CACHE_HIDDEN_FRAMES=1`
  - `GEGE_PARTITION_BUFFER_SWAP_TIMING=1`

Epoch summary:

- `Epoch Runtime: 227.073 s`
- `swap_update: 59.682 s`
- `batch_fetch: 75.934 s`
- `edge_sample: 16.158 s`
- `map_lookup: 4.104 s`
- `negative_sample: 8.255 s`

What the boundary instrumentation proved:

- For the first two embedding swaps, hidden-frame publish **did** remove one
  partition from visible install:
  - `admit_parts=4`
  - `visible_install_rows=7,809,795`
  - `hidden_publish_rows=2,603,265`
  - `cpu_to_gpu_ms=0.000`
- Example first embedding swap:
  - `swap 0`
  - `visible_install_mib=2979.200`
  - `hidden_publish_mib=993.067`
  - `preloaded_admit=true`
  - `total_ms=788.624`

- But this succeeded for only `2 / 19` embedding swaps in the epoch:
  - `preloaded_true=2`
  - `fallback=17`

- After that, the preload path was skipped on memory guard:
  - free HBM stayed around `1470.812 MiB`
  - next hidden preloads needed `3010.134 MiB` or `4003.200 MiB`
  - logs show repeated `[partition-buffer-preload-skip]` for both
    `embeddings.bin` and `embeddings_state.bin`

- In fallback swaps, hidden publish disappeared and the normal boundary copy
  path returned:
  - `hidden_publish_rows=0`
  - `visible_install_rows=7,809,795` or `10,413,060`
  - `cpu_to_gpu_ms≈405–675 ms`

Interpretation:

- The architecture is not purely hypothetical anymore: when a hidden frame fits,
  it really does remove visible install bytes from the boundary.
- But on Twitter q4 with one full hidden partition, this happens only at the very
  beginning of the epoch.
- The main reason `swap_update` stayed flat is now known exactly:
  the hidden-frame path runs out of usable free HBM after the first two swaps,
  so almost the entire epoch reverts to the old boundary-copy path.

Decision after the kill-test:

- **No-go** on full hidden-partition `k=1` as a standalone Twitter solution.
- **Go** only if the next version changes granularity:
  - tiled hidden frames, or
  - a stricter delayed-commit policy that actually frees enough HBM for repeated
    preloads across the epoch.

### Versioned Frame Cache Simulator: Calibrated Go / No-Go Check

To avoid pushing further on a broken implementation strategy, we added an offline simulator:

- Script: `scripts/versioned_frame_cache_sim.py`
- Outputs:
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_1gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_2gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_4gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_q4_1gpu_calibrated.tsv`
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_q4_4gpu_calibrated.tsv`

Simulator assumptions:

- logical visible state remains exact q4
- hidden capacity is expressed in partition-equivalents
- future admits are prefetched into hidden frames during compute
- publish at the boundary remaps logical slots instead of installing by copy
- old visible frames become stale backlog and drain during later compute windows
- only unhidden boundary work remains exposed

Calibration inputs:

- Twitter 1 GPU current baseline: `223.230 s`, exposed boundary `59.414 s`
- Twitter 2 GPU best measured baseline: `170.088 s`, exposed boundary `27.071 s = 18.644 s swap_update + 8.427 s swap_barrier_wait`
- Twitter 4 GPU best measured baseline: `74.057 s`, exposed boundary `21.426 s = 19.984 s swap_update + 1.442 s swap_barrier_wait`
- FB86M 1 GPU current best baseline: `166.393 s`, exposed boundary `65.433 s`
- FB86M 4 GPU best measured baseline: `49.822 s`, exposed boundary `18.319 s = 11.816 s swap_update + 6.503 s swap_barrier_wait`

Predicted epoch time if the architecture is implemented cleanly:

| Dataset / GPUs | Hidden capacity | Predicted epoch | Delta vs current |
| --- | ---: | ---: | ---: |
| Twitter 1 GPU | `k=1` | `204.416 s` | `-18.814 s` |
| Twitter 1 GPU | `k=2` | `185.601 s` | `-37.629 s` |
| Twitter 1 GPU | `k=3` | `166.787 s` | `-56.443 s` |
| Twitter 2 GPU | `k=1` | `161.516 s` | `-8.572 s` |
| Twitter 2 GPU | `k=2` | `152.943 s` | `-17.145 s` |
| Twitter 2 GPU | `k=3` | `144.371 s` | `-25.717 s` |
| Twitter 4 GPU | `k=1` | `67.272 s` | `-6.785 s` |
| Twitter 4 GPU | `k=2` | `60.487 s` | `-13.570 s` |
| Twitter 4 GPU | `k=3` | `53.702 s` | `-20.355 s` |
| FB86M 1 GPU | `k=0.25` | `161.213 s` | `-5.180 s` |
| FB86M 1 GPU | `k=0.50` | `156.033 s` | `-10.360 s` |
| FB86M 1 GPU | `k=1.00` | `145.673 s` | `-20.720 s` |
| FB86M 4 GPU | `k=0.25` | `48.372 s` | `-1.450 s` |
| FB86M 4 GPU | `k=0.50` | `46.922 s` | `-2.900 s` |
| FB86M 4 GPU | `k=1.00` | `44.021 s` | `-5.801 s` |

What this says:

- The architecture is still worth pursuing. In the ideal publish-by-remap model, the boundary component shrinks materially on all tested scales.
- Twitter benefits the most because its current q4 boundary is both large and partition-sized hidden frames fit within the modeled `~6 GiB` free-HBM budget.
- FB86M still benefits, but less dramatically under q4 full-partition frames. That reinforces the design choice to move FB86M toward tiled frames rather than full hidden partitions.
- The simulator result is completely inconsistent with the current hidden-frame prototype (`223.230 s -> 226.455 s`). That gap is the strongest evidence that the current implementation is wrong in *where* it pays cost, not wrong in architectural direction.

Decision:

- **Go** on the architecture.
- **No-go** on the current implementation shape.
- The next implementation should only touch prefetch, publish, and commit paths. It should not add frame indirection work inside steady-state q4 batch reads, mapping, or negative sampling.

### Safe Overlap Probe: Async Next-State Batch Preparation

To test the safest possible overlap path before touching admit/evict again, we added:

- `GEGE_ASYNC_NEXT_STATE_BATCHES=1`

What it does:

- While the current state trains, if the next in-memory subgraph state has already been prefetched, a background thread prebuilds the next state's bucket-streaming LP `Batch` list into a side buffer.
- At the swap boundary, `getNextBatch()` tries to consume that prepared list instead of calling `initializeBatches()` synchronously.
- This does **not** change q4 visibility, negative support, embeddings, optimizer state, or admit/evict behavior.

Runs:

- Baseline: `/dev/shm/smansou2_ge2/twitter_framecache_measure_baseline_3e_20260408/train.log`
- Async next-state batches: `/dev/shm/smansou2_ge2/twitter_async_next_state_batches_3e_20260408/train.log`

Three-epoch comparison:

| Config | Avg runtime | Avg runtime excl. epoch 1 | `batch_fetch` | `edge_sample` | `map_lookup` | `negative_sample` | `swap_update` | `swap_rebuild` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q4 baseline | `223.230 s` | `222.132 s` | `114.825 s` | `55.318 s` | `43.508 s` | `9.107 s` | `59.414 s` | `52.962 ms` |
| q4 + async next-state batches | `224.239 s` | `223.237 s` | `115.663 s` | `55.299 s` | `43.563 s` | `9.073 s` | `60.272 s` | `51.547 ms` |
| Delta | `+1.009 s` | `+1.105 s` | `+0.838 s` | `-0.019 s` | `+0.056 s` | `-0.035 s` | `+0.858 s` | `-1.415 ms` |

Interpretation:

- This safe overlap path works mechanically and does not perturb the hot sampling/mapping behavior in any large way.
- But it is not a useful optimization on Twitter q4.
- The only direct win is a tiny `swap_rebuild` reduction, from about `52.96 ms` to `51.55 ms`.
- That does not matter at Twitter scale, and the run actually gets slower overall because `swap_update` drifts upward by about `0.86 s`.

Conclusion:

- Prebuilding next-state `Batch` metadata is safe, but it is too small to matter.
- Twitter's exposed cost is not synchronous batch-list construction. It is still dominated by visible state movement and batch fetch.
- This path should be treated as a low-risk utility, not as the main optimization direction.

## Next Architecture Direction

The current recommendation is no longer "try another local flag." The path forward is the standalone design in:

- `paper/versioned_frame_cache_design_20260408.md`

The reason is simple:

- LJ responded to smaller synchronization and staging fixes because its state footprint is small.
- FB86M and Twitter are still dominated by visible state movement at 1, 2, and 4 GPUs.
- Changing visible state size or negative-domain support has repeatedly damaged accuracy.

So the next paper-grade path should preserve logical q4 visibility while changing the physical movement policy:

- hidden physical frames or tiles
- publish by remapping logical slots to prefetched frames
- delayed commit instead of dump-at-boundary
- later, state-local batch mapping to attack the remaining Twitter map wall

## Versioned Frame Cache Simulator Refresh

Before touching the runtime again, rerun the offline frame-cache simulator with the current measured baselines and explicit stress assumptions.

Outputs:

- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_ideal.tsv`
- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_q4_stress.tsv`
- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_2gpu_q4.tsv`
- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/twitter_4gpu_q4.tsv`
- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_q4_ideal.tsv`
- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_q4_stress.tsv`
- `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/fb86m_4gpu_q4.tsv`

Calibration used:

- Twitter 1 GPU baseline: `223.085 s`, exposed boundary `61.603 s`
- Twitter 2 GPU baseline: `170.088 s`, exposed boundary `27.071 s`
- Twitter 4 GPU baseline: `74.057 s`, exposed boundary `21.426 s`
- FB86M 1 GPU baseline: `166.393 s`, exposed boundary `65.433 s`
- FB86M 4 GPU baseline: `49.822 s`, exposed boundary `18.319 s`

Current q4-template overlap for `p=16,q=4` is lower than the old retained=`1` simplification:

- average retained partitions per transition: about `0.211`
- average swapped partition-equivalents per transition: about `3.789`

This makes the simulator more conservative than the earlier hand estimate.

Twitter q4 versioned-frame results:

| Scale | Hidden capacity | Predicted epoch | Predicted boundary |
| --- | ---: | ---: | ---: |
| 1 GPU ideal | `k=1` | `206.829 s` | `45.347 s` |
| 1 GPU ideal | `k=2` | `190.572 s` | `29.090 s` |
| 1 GPU ideal | `k=3` | `174.316 s` | `12.834 s` |
| 1 GPU stress (`prefetch=0.2`, `commit=0.1`) | `k=1` | `206.829 s` | `45.347 s` |
| 1 GPU stress (`prefetch=0.2`, `commit=0.1`) | `k=2` | `192.307 s` | `30.825 s` |
| 1 GPU stress (`prefetch=0.2`, `commit=0.1`) | `k=3` | `191.451 s` | `29.969 s` |
| 2 GPU | `k=1` | `162.944 s` | `19.927 s` |
| 2 GPU | `k=2` | `155.801 s` | `12.784 s` |
| 2 GPU | `k=3` | `148.657 s` | `5.640 s` |
| 4 GPU | `k=1` | `68.403 s` | `15.772 s` |
| 4 GPU | `k=2` | `62.749 s` | `10.118 s` |
| 4 GPU | `k=3` | `57.095 s` | `4.464 s` |

FB86M q4 versioned-frame results:

| Scale | Hidden capacity | Predicted epoch | Predicted boundary |
| --- | ---: | ---: | ---: |
| 1 GPU ideal | `k=0.25` | `162.076 s` | `61.116 s` |
| 1 GPU ideal | `k=0.50` | `157.759 s` | `56.799 s` |
| 1 GPU ideal | `k=1.00` | `149.126 s` | `48.166 s` |
| 1 GPU stress (`prefetch=0.2`, `commit=0.1`) | `k=0.25` | `162.076 s` | `61.116 s` |
| 1 GPU stress (`prefetch=0.2`, `commit=0.1`) | `k=0.50` | `157.759 s` | `56.799 s` |
| 1 GPU stress (`prefetch=0.2`, `commit=0.1`) | `k=1.00` | `149.126 s` | `48.166 s` |
| 4 GPU | `k=0.25` | `48.613 s` | `17.110 s` |
| 4 GPU | `k=0.50` | `47.405 s` | `15.902 s` |
| 4 GPU | `k=1.00` | `44.988 s` | `13.485 s` |

What this changes:

- The architecture still has real headroom, but the old `~166 s` Twitter q=4 hidden-frame estimate was too optimistic.
- The best realistic Twitter 1-GPU target from q4 full hidden partitions is now closer to `~174-191 s`, depending on commit overlap, not `~160 s`.
- Twitter remains the better first implementation target because full hidden partitions still fit within the modeled `~6 GiB` free-HBM budget:
  - partition size at `p=16`: about `1.94 GiB`
  - `k=3` hidden partitions: about `5.82 GiB`
- FB86M still benefits, but the absolute single-GPU gain is smaller under q4 full hidden partitions:
  - best q4 full-partition frame model: `166.393 -> 149.126 s`
- FB86M 4-GPU still shows meaningful boundary headroom, but it is smaller than Twitter's. This reinforces the earlier conclusion that FB86M should move to tiled hidden frames sooner.

Most important conclusion:

- The simulator still says **go** on versioned frame cache.
- But the right expectation is now:
  - Twitter q4 full hidden-frame publish/remap can plausibly save tens of seconds, not more than `100 s`
  - FB86M q4 full hidden-frame publish/remap is helpful, but not enough by itself; tiled frames remain the right long-term path
- The architecture is worth implementing only if the next prototype keeps all hidden-frame machinery off the steady-state q4 hot path and only changes prefetch, publish, and delayed commit.

### `k=1` Publish/Backlog Penalty Sweep

Before touching runtime code again, refine the simulator with two explicit control-path penalties:

- `publish_transition_ms`: fixed publish/remap control cost per transition
- `stale_backlog_part_ms`: exposed cost per stale partition-equivalent in backlog

This is a sensitivity study, not a calibrated runtime model. It answers a narrower question:

- if the first prototype only implements `k=1`, is it still worth trying once publish/commit overhead is charged explicitly?

Implementation:

- `scripts/versioned_frame_cache_sim.py` now accepts:
  - `--publish-transition-ms`
  - `--stale-backlog-part-ms`
- Sweep output:
  - `/dev/shm/smansou2_ge2/frame_cache_sim_20260408/k1_penalty_sweep.tsv`

Swept values:

- `publish_transition_ms ∈ {0, 50, 100, 250}`
- `stale_backlog_part_ms ∈ {0, 50, 100, 250}`

Summary for `k=1`:

| Dataset / GPUs | Base (`0/0`) | Mid (`50/50`) | Worst (`250/250`) |
| --- | ---: | ---: | ---: |
| Twitter 1 GPU | `206.829 s` | `208.729 s` | `216.329 s` |
| Twitter 2 GPU | `162.944 s` | `163.844 s` | `167.444 s` |
| Twitter 4 GPU | `68.403 s` | `68.803 s` | `70.403 s` |
| FB86M 1 GPU | `149.126 s` | `151.026 s` | `158.626 s` |
| FB86M 4 GPU | `44.988 s` | `45.388 s` | `46.988 s` |

Interpretation:

- The first `k=1` prototype is not hypersensitive to small publish/commit overheads.
- But it is also not transformative on Twitter:
  - even the clean `k=1` model is only `223.085 -> 206.829 s`
  - with explicit `250/250 ms` penalties it weakens to `216.329 s`
- So `k=1` is a good **correctness prototype**, not a likely paper result by itself.
- The simulator still supports the same sequencing:
  1. implement `k=1` only to prove publish/remap + delayed commit is correct
  2. if that works and does not pollute the hot path, move quickly to `k=2/3` on Twitter
  3. keep FB86M on the tiled-frame path rather than betting on full hidden partitions

## Twitter hidden-frame runtime isolation and `k=1` hidden-only preload

To separate frame-capacity effects from preload-traffic effects, I ran three Twitter q4 single-GPU runtime probes on the same codebase:

1. q4 baseline with no hidden frames
2. hidden frames enabled, but `GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=0`
3. hidden frames enabled with a new `GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1` path

The new path only preloads the hidden partition into a hidden physical frame. The remaining admits still use the normal synchronous fallback at the boundary.

Results:

| Variant | Epoch time | `swap_update` | `map_lookup` | `negative_sample` | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline 1e | `225.427 s` | `59.530 s` | `43.296 s` | `9.098 s` | current q4 reference |
| Hidden frames, no admit preload, 1e | `222.397 s` | `60.784 s` | `44.099 s` | `7.776 s` | no hidden publish; extra capacity alone is not the problem |
| Old hidden1 full preload, 2e steady | `224.899 s` | `59.146 s` | `66.891 s` | `71.759 s` | background preload polluted the hot path badly |
| New hidden-only preload, 1e | `219.932 s` | `57.165 s` | `43.923 s` | `8.245 s` | hidden publish active; hot path stays sane |
| New hidden-only preload, 2e steady | `219.384 s` | `56.801 s` | `44.066 s` | `8.334 s` | stable improvement over baseline |

Key runtime evidence:

- The hidden-only prototype logs `visible_install_rows=0` and nonzero `hidden_publish_rows` on every swap, so one admitted partition is being prefetched into a hidden physical frame and published by remap.
- The remaining admitted partitions still fall back through the synchronous visible install path at the boundary.
- The old hidden1 slowdown was therefore not caused by frame indirection itself. It was caused by preloading too much GPU traffic in the background.
- Restricting preload to the hidden partition removed the `map_lookup` / `negative_sample` blow-up and recovered a modest epoch gain.

Interpretation:

- This is the first runtime prototype that improves Twitter without changing visible q4 semantics.
- The gain is modest because evict D2H and the remaining visible admits are still fully exposed.
- The next storage step should keep the hidden-only policy and attack the other boundary costs:
  - delayed commit for stale frames
  - more than one hidden frame only if the hot path remains clean
  - tiled hidden frames for FB86M rather than full hidden partitions

## Twitter `k=1` delayed stale-frame commit

I then implemented the next storage step on top of hidden-only preload:

- keep q4 visible semantics unchanged
- publish one admitted partition by remap into the hidden frame
- do **not** free the old visible frame immediately
- asynchronously write back that stale old frame to host after publish, then release the frame back to the hidden-frame pool

Flag:

- `GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1`

This reused the existing async evict writeback thread, but pointed it directly at the stale frame slice inside `buffer_tensor_gpu_view_` after publish. No extra full-stage GPU tensor was allocated.

Results:

| Variant | Epoch time | `batch_fetch` | `swap_update` | `map_lookup` | `negative_sample` | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Hidden-only preload, epoch 1 | `219.722 s` | `110.822 s` | `56.777 s` | `44.052 s` | `8.231 s` | previous clean runtime reference |
| Hidden-only preload + delayed stale commit, epoch 1 | `215.207 s` | `106.489 s` | `52.560 s` | `43.951 s` | `8.206 s` | one stale frame per swap written back asynchronously after publish |

What changed:

- `swap_update` dropped by about `4.22 s`
- `batch_fetch` dropped by about `4.33 s`
- `map_lookup` and `negative_sample` stayed effectively flat, so the gain is coming from storage execution rather than hot-path pollution

Key runtime evidence:

- Every swap logged `deferred_stale_writeback=true`
- Each swap also logged a follow-up `[partition-buffer-async-evict-writeback] ... release_frames=[...]` event for the stale frame
- The free hidden frame alternated cleanly between the old visible frame ids, proving the stale-frame release path worked
- The run completed a full epoch and entered epoch 2 cleanly before I stopped it

Default-off sanity:

- LJ 1e with `GEGE_FRAME_CACHE_HIDDEN_FRAMES=0` stayed on the normal path:
  - `Epoch Runtime: 11437ms`
  - `swap_update_ms=4966.372`
  - `map_lookup_ms=2334.387`
- No hidden publish or delayed stale writeback was active in that run

Interpretation:

- This is the first storage change after hidden-only preload that moves Twitter epoch time by more than noise without changing q4 visibility.
- The improvement is still modest, but it is real and directly attributable to moving one evict/writeback off the boundary path.
- The next logical step is still the same architecture direction:
  - keep visible q4 exact
  - increase hidden-frame depth only if the hot path stays clean
  - then revisit whether a `k=2` hidden-frame path or tiled FB86M stale-frame commit is the better next prototype

## FB86M `k=1` delayed stale-frame commit probe

I then ran the same storage path on FB86M single GPU using the current best FB stack plus:

- `GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1`
- `GEGE_FRAME_CACHE_HIDDEN_FRAMES=1`
- `GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1`
- `GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1`

The FB baseline on current code, without hidden frames, was:

| Variant | Epoch time | `batch_fetch` | `swap_update` | `map_lookup` | `negative_sample` |
| --- | ---: | ---: | ---: | ---: | ---: |
| FB current best baseline | `164.098 s` | `153.515 s` | `65.485 s` | `86.620 s` | `0.544 s` |
| FB hidden1 + delayed stale commit | `154.289 s` | `139.655 s` | `53.290 s` | `84.604 s` | `0.719 s` |

Delta versus baseline:

- epoch time: `-9.809 s` (`-5.98%`)
- `batch_fetch`: `-13.859 s`
- `swap_update`: `-12.195 s`
- `map_lookup`: `-2.016 s`
- `negative_sample`

The main effect is clearly on storage execution, not the sampler:

- delayed stale-frame writeback is worth much more on FB86M than on Twitter
- the same single-hidden-frame mechanism buys about `10 s` on FB86M vs about `4.5 s` on Twitter

This means the frame-cache path is worth keeping for FB86M, but it is still not enough by itself to be the whole answer.

## FB86M `k=2` full hidden-partition probe

I then extended the hidden-publish bookkeeping from one hidden frame to multiple hidden frames and tried the obvious next FB86M step:

- `GEGE_FRAME_CACHE_HIDDEN_FRAMES=2`
- `GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1`
- `GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1`

Result:

- reject as a full-partition design on 24GB
- the run failed before epoch 1 in `MemPartitionBuffer::ensureBackingTensorsAllocated_()` with CUDA OOM
- the allocator tried to reserve another `12.02 GiB` while about `12.05 GiB` was already allocated

Interpretation:

- the multi-hidden publish mechanics compile and are wired correctly
- but **full hidden partitions do not scale past `k=1` on FB86M single GPU**
- the next storage step for FB86M must be **tiled hidden frames**, not more full hidden partitions

## Twitter state-local bitmap-domain probe

I then tested a narrower Twitter map-path idea: keep the current fixed-buffer padded bitmap mapper, but pass the current in-memory subgraph size as the bitmap domain instead of always falling back to the global `GEGE_UNIQUE_BITMAP_NUM_NODES`.

Code:

- Added `GEGE_FIXED_BUFFER_BITMAP_LOCAL_DOMAIN=1`
- Extended `map_tensors(...)` to accept an explicit `value_domain_size`
- Under the in-memory subgraph LP path, `edgeSample()` passes `current_subgraph_states_[device_idx]->in_memory_subgraph_->num_nodes_in_memory_` into `map_tensors(...)` when the flag is enabled

LJ safety gate:

- Run: `/tmp/smansou2_ge2/lj_local_bitmap_domain_1e_20260409`
- Result: mechanically safe
- `Epoch Runtime: 9493ms`
- `map_lookup_ms=2112.305`
- `swap_update_ms=2578.520`

Twitter q4 single-GPU probe:

- Run: `/dev/shm/smansou2_ge2/twitter_local_bitmap_domain_1e_20260409/train.log`
- Result: reject

| Variant | Epoch time | `batch_fetch` | `edge_sample` | `map_lookup` | `negative_sample` | `swap_update` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current q4 baseline (current-code reference) | `223.230 s` | `114.825 s` | `55.318 s` | `43.508 s` | `9.107 s` | `59.414 s` |
| `GEGE_FIXED_BUFFER_BITMAP_LOCAL_DOMAIN=1` | `387.904 s` | `253.732 s` | `168.352 s` | `101.291 s` | `64.419 s` | `83.132 s` |

Interpretation:

- The simple local-domain bitmap idea is **not** a Twitter win
- It does not reduce the current q4 `map_lookup` wall
- It makes `map_lookup`, `negative_sample`, and `swap_update` all substantially worse
- So the next Twitter map direction should not be "smaller domain in the current bitmap path"; it needs a different design, such as a true state-local mapping/tape path

## New design direction: State-Compiled Runtime

After the local-domain bitmap failure, I stopped short of more runtime changes and wrote a new pre-coding design note:

- [state_compiled_runtime_design_20260409.md](/home/smansou2/newCode/ge2/dandelion-dev/paper/state_compiled_runtime_design_20260409.md)

The core idea is a layered runtime:

1. `StateLocalMapEngine`
   - replace per-batch dynamic `map_tensors(...)` with a generation-stamped state-local mapper
2. `NegativeDescriptorTape`
   - move exact batch-local negative descriptors or replay seeds out of the hot path
3. `TiledVersionedFrameCache`
   - use the frame-cache work as the storage backend instead of the whole optimization

The current simulator for this design is:

- [state_compiled_runtime_sim.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/state_compiled_runtime_sim.py)

Outputs:

- `/dev/shm/smansou2_ge2/state_compiled_runtime_sim_20260409/twitter_1gpu.tsv`
- `/dev/shm/smansou2_ge2/state_compiled_runtime_sim_20260409/fb86m_1gpu.tsv`
- `/dev/shm/smansou2_ge2/state_compiled_runtime_sim_20260409/fb86m_1gpu_conservative.tsv`

Main takeaway:

- Twitter only moves materially if all three layers improve together
- FB86M can still move materially even if only part of the current `map_lookup` timer bucket is truly removable
- So the next real prototype should be `StateLocalMapEngine`, not another swap flag and not another scheduler tweak

## Offline analyzer: state-local tape feasibility

I added an offline analyzer binary:

- [gege_state_compiled_runtime_analyzer.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/gege_state_compiled_runtime_analyzer.cpp)

It uses the real ordering and train-edge partitions, but does **not** change runtime code. It measures, per visible q-state:

- exact positive edges in the state
- exact positive unique local rows
- compile-time for a state-local positive pass
- positive-only workspace size for a generation-stamped `StateLocalMapEngine`
- positive unique-tape size
- upper-bound materialized negative-id tape size

Outputs:

- `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_twitter_state0_20260409.txt`
- `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_twitter_allstates_20260409.txt`
- `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_fb86m_state0_20260409.txt`
- `/dev/shm/smansou2_ge2/state_compiled_runtime_analyzer_fb86m_allstates_20260409.txt`

Twitter all-states summary (`20` states):

- `workspace.positive_only.total_gib = 0.078` for every state
- `tape.positive_unique_arena_gib = 0.173 .. 0.229`
- `tape.materialized_negative_ids_gib = 0.998 .. 1.362`
- `compile_ms = 3804.5 .. 5330.7`

FB86M all-states summary (`20` states):

- `workspace.positive_only.total_gib = 0.161` for every state
- `tape.positive_unique_arena_gib = 0.070 .. 0.100`
- `tape.materialized_negative_ids_gib = 0.187 .. 0.287`
- `compile_ms = 1475.4 .. 2160.3`

Interpretation:

- The positive-only state-local mapper is cheap enough to prototype first on both large datasets.
- Twitter state `0` was slightly heavier than average, so the earlier state-0 result was a safe planning point.
- Fully materialized negative-id tapes are too large to be the first implementation, especially on Twitter.
- So the next implementation should be:
  1. `StateLocalMapEngine`
  2. then seeded / descriptor-based negative replay
  3. then integrate with tiled frame cache

## First `StateLocalMapEngine` runtime prototype: mechanically safe, reject on Twitter

Implemented the first runtime prototype as a default-off path:

- new env flag: `GEGE_STATE_LOCAL_MAP_ENGINE=1`
- verification flags:
  - `GEGE_STATE_LOCAL_MAP_VERIFY=1`
  - `GEGE_STATE_LOCAL_MAP_VERIFY_MAX_CALLS`
- code path:
  - [util.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/common/util.cpp)
  - [dataloader.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/dataloader.cpp)
  - [unique_map_cuda.h](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/include/common/unique_map_cuda.h)
  - [unique_map_cuda.cu](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cuda/src/common/unique_map_cuda.cu)

The new CUDA backend is a generation-stamped state-local unique/inverse mapper:

- input IDs are already in the current q-state local domain
- the backend uses a dense generation table over that local domain
- it records first-touch positions, sorts only touched IDs by first appearance, and builds the inverse map from state-local slots

### LJ safety gate

Used the paper-opt LJ stack as a mechanical gate only.

Logs:
- `/dev/shm/smansou2_ge2/lj_paperopt_baseline_2e_20260409/train.log`
- `/dev/shm/smansou2_ge2/lj_state_local_map_kernel_1e_20260409/train.log`

Result:

| Run | Epoch 2 runtime | `edge_sample` | `negative_sample` | `map_lookup` | `swap_update` |
| --- | ---: | ---: | ---: | ---: | ---: |
| LJ paper-opt baseline | `21253 ms` | `4643.026 ms` | `4257.792 ms` | `291.143 ms` | `5588.304 ms` |
| LJ state-local map | `20985 ms` | `4707.198 ms` | `4304.377 ms` | `318.506 ms` | `5444.427 ms` |

Interpretation:

- No verifier failures.
- No semantic/mechanical break was observed.
- LJ is effectively neutral because `map_lookup` is already small on this path.

### Twitter q4 result

Log:
- `/dev/shm/smansou2_ge2/twitter_state_local_map_kernel_1e_20260409/train.log`

Reference:
- current-code Twitter q4 baseline is `223.230 s` from the 3-epoch train-only table above.

Result:

| Run | Epoch runtime | `batch_fetch` | `edge_sample` | `negative_sample` | `map_lookup` | `swap_update` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Twitter q4 current baseline | `223.230 s` | `114.825 s` | `55.318 s` | `9.107 s` | `43.508 s` | `59.414 s` |
| Twitter q4 + `GEGE_STATE_LOCAL_MAP_ENGINE=1` | `327.359 s` | `284.232 s` | `202.369 s` | `4.905 s` | `196.376 s` | `81.301 s` |

Interpretation:

- The prototype is mechanically safe, but it is not a win.
- `negative_sample` drops, but `map_lookup` explodes and total `edge_sample` becomes much worse.
- `swap_update` also rises because the training loop is now slower overall and the state boundary stays exposed longer.
- This is not timer migration. The first-generation local mapper backend is genuinely slower than the current padded bitmap path on Twitter.

Conclusion:

- Keep the prototype default-off.
- Reject this backend as the first production `StateLocalMapEngine`.
- The state-compiled runtime direction is still valid, but the next real step cannot be "replace `map_tensors(...)` with this kernel and stop."
- The next prototype should move one layer up:
  1. compile/replay state-local positive batch tapes, not just the unique/inverse backend
  2. bind negatives through exact replay seeds/descriptors
  3. use the mapper only as part of that larger compile/replay contract

## `StatePositiveBatchTape` prototype: safe on LJ, reject on Twitter

Implemented the next layered prototype as a default-off path:

- new env flag: `GEGE_STATE_POSITIVE_BATCH_TAPE=1`
- verifier:
  - `GEGE_STATE_POSITIVE_BATCH_TAPE_VERIFY=1`
  - `GEGE_STATE_POSITIVE_BATCH_TAPE_VERIFY_MAX_CALLS`

Code path:

- [batch.h](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/include/data/batch.h)
- [batch.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/batch.cpp)
- [dataloader.cpp](/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/dataloader.cpp)

Mechanism:

- during streamed q-state batch initialization, compile once per batch:
  - positive-only unique local IDs
  - positive src mapping
  - positive dst mapping
- at runtime, keep positive mappings fixed and only remap:
  - `compiled_positive_unique_node_indices_`
  - plus sampled negatives

So this is the first replay-style positive-tape prototype, but it still uses the old runtime `map_tensors(...)` backend to merge positive uniques with negatives.

### LJ safety gate

Log:

- `/dev/shm/smansou2_ge2/lj_state_positive_batch_tape_2e_20260409/train.log`

Result:

| Run | Epoch runtime | `batch_fetch` | `edge_sample` | `negative_sample` | `map_lookup` | `swap_update` | `swap_rebuild` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LJ `StatePositiveBatchTape`, epoch 1 | `18359 ms` | `15695.053 ms` | `9092.949 ms` | `245.286 ms` | `8768.014 ms` | `5582.514 ms` | `305.234 ms` |
| LJ `StatePositiveBatchTape`, epoch 2 | `17825 ms` | `15710.883 ms` | `9143.360 ms` | `244.253 ms` | `8824.118 ms` | `5532.496 ms` | `310.522 ms` |

Safety result:

- No verifier failures in the first 8 checked calls.
- The mechanism is mechanically safe on LJ.
- The per-state positive tape footprint is about `75-101 MiB` on LJ.

Interpretation:

- This path does not validate the final design, but it is a safe replay-style positive-tape prototype.
- It also shows a warning sign: work is shifted from `negative_sample` into `map_lookup`, not removed.

### Twitter q4 result

Log:

- `/dev/shm/smansou2_ge2/twitter_state_positive_batch_tape_1e_20260409/train.log`

Reference:

- current-code Twitter q4 baseline in this note: `223.230 s`

Result:

| Run | Epoch runtime | `batch_fetch` | `edge_sample` | `negative_sample` | `map_lookup` | `swap_update` | `swap_rebuild` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Twitter q4 current baseline | `223.230 s` | `114.825 s` | `55.318 s` | `9.107 s` | `43.508 s` | `59.414 s` | `52.962 ms` |
| Twitter q4 + `GEGE_STATE_POSITIVE_BATCH_TAPE=1` | `333.874 s` | `288.479 s` | `198.360 s` | `5.315 s` | `191.401 s` | `81.335 s` | `6.921 s` |

Additional state-build observations:

- per-state positive tape footprint on Twitter is huge: about `1.35-1.87 GiB`
- the first state compiled `1347` replay batches and `46.28M` positive unique IDs
- later states peaked around `1.87 GiB`

Interpretation:

- This prototype is a clear reject on Twitter.
- It is not enough to "compile positives once" if replay still feeds a very large positive-unique tensor back through the current `map_tensors(...)` path.
- The path also makes `initializeBatches()` expensive enough that `swap_rebuild` jumps from `~53 ms` to `6.9 s`.
- So the positive-tape idea is not dead, but this implementation shape is wrong.

Conclusion:

- Keep the prototype default-off.
- Do not continue the "positive tape + old merge mapper" runtime path.
- The next state-compiled prototype should be:
  1. **compressed positive replay metadata**, not full per-batch `int64` remap tensors
  2. **descriptor/seed-based negative replay**, so replay does not have to merge a huge positive-unique tensor with negatives through the old mapper
  3. a real replay engine that bypasses `map_tensors(...)`, not one that re-enters it with a different input

## `StatePositiveUniqueTape` prototype: smaller metadata, still not a win

Implemented a narrower replay prototype behind:

- `GEGE_STATE_POSITIVE_UNIQUE_TAPE=1`

Mechanism:

- compile only the sorted positive unique local IDs once per streamed batch
- do **not** store positive src/dst remap tensors
- at runtime:
  - remap positives by `searchsorted` into the sorted positive unique set
  - run `map_tensors(...)` only on negative misses

### LJ result

Log:

- `/dev/shm/smansou2_ge2/lj_state_positive_unique_tape_2e_20260409_r2/train.log`

Result:

| Run | Epoch runtime | `batch_fetch` | `edge_sample` | `negative_sample` | `map_lookup` | `swap_update` | `swap_rebuild` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LJ `StatePositiveUniqueTape`, epoch 1 | `18634 ms` | `15878.572 ms` | `9218.596 ms` | `249.695 ms` | `8909.681 ms` | `5562.703 ms` | `434.403 ms` |
| LJ `StatePositiveUniqueTape`, epoch 2 | `18111 ms` | `15958.682 ms` | `9271.080 ms` | `247.136 ms` | `8971.393 ms` | `5583.552 ms` | `436.452 ms` |

Additional observations:

- the per-state unique-only tape shrank to about `31.6-41.0 MiB`, much smaller than the full positive tape
- verifier passed after fixing the sorted-positive replay contract

Interpretation:

- shrinking replay metadata alone is not enough
- runtime still spends almost all of the old mapper time in `searchsorted` + negative miss merge
- this is mechanically safer than the full positive tape, but still far from a useful win

### Twitter result

Log:

- `/dev/shm/smansou2_ge2/twitter_state_positive_unique_tape_1e_20260409/train.log`

Observed state-build footprint before stopping:

- state 0: `353.104 MiB`
- state 1: `362.920 MiB`

This run is not a valid epoch measurement because `/dev/shm` filled and the logger started failing before the first epoch summary was written. Still, the footprint alone is enough to reject this implementation shape on Twitter.

Conclusion:

- keep default-off
- do not continue the "sorted positive unique + negative miss merge" path
- the next replay step must reduce runtime work, not only tape size

## `GEGE_RESIDENT_LOCAL_LP_DIRECT=1`: removes `map_lookup`, but compute/update explodes

There is already an existing direct state-local LP path in:

- `/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/dataloader.cpp`
- `/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/nn/model.cpp`
- `/home/smansou2/newCode/ge2/dandelion-dev/gege/src/cpp/src/data/batch.cpp`

It bypasses batch-local compaction entirely:

- positives use the current state-local IDs directly
- negative tensors are loaded directly from the resident state-local space
- forward/backward operate on resident-local embeddings
- `accumulateResidentLocalGradients()` reduces the resulting duplicate gradients with CSR

### LJ result

Log:

- `/tmp/smansou2_ge2/lj_resident_local_lp_direct_2e_20260409/train.log`

Epoch 1:

| Run | Epoch runtime | `batch_fetch` | `edge_sample` | `negative_sample` | `map_lookup` | `swap_update` | `swap_rebuild` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LJ baseline (validated path) | `~8499 ms` | baseline note above | baseline note above | baseline note above | baseline note above | baseline note above | baseline note above |
| LJ `GEGE_RESIDENT_LOCAL_LP_DIRECT=1` | `17403 ms` | `6589.356 ms` | `256.602 ms` | `228.430 ms` | `0.000 ms` | `6029.280 ms` | `4.384 ms` |

Interpretation:

- this path does exactly what it should on the data-loader side:
  - `map_lookup -> 0`
  - `swap_rebuild -> ~0`
- but total runtime nearly doubles on LJ
- the lost batch-local reuse moves the wall into model compute / embedding-update work

### Twitter result

Log:

- `/tmp/smansou2_ge2/twitter_resident_local_lp_direct_1e_20260409/train.log`

Observed progress before stopping:

- epoch start: `16:19:20`
- `10%` at `16:20:14`

That is roughly `53 s` for the first `10%`, implying an epoch in the `~500 s` range if it continued at the same pace.

Conclusion:

- reject this direct resident-local execution shape
- the insight is still useful: `map_lookup` can be eliminated, but not by giving up all per-batch reuse
- the next viable direction has to keep state-local replay **and** preserve some compact reused view for compute

## Existing `sort` unique backend: reject on Twitter

Tested the existing unique backend switch with:

- `GEGE_UNIQUE_BACKEND=sort`

### Twitter result

Log:

- `/dev/shm/smansou2_ge2/twitter_unique_sort_1e_solo_20260409/train.log`

Observed progress:

- training start: `16:25:25`
- `10%` at `16:25:58`
- `20%` at `16:26:27`

So the first `20%` took about `62 s`, implying an epoch around `~310 s`, much worse than the `223.230 s` bitmap baseline.

Conclusion:

- do not spend more time on alternate existing unique backends
- this is another confirmation that backend-only unique changes are not enough

## Updated conclusion after the replay/runtime probes

The recent failed probes point to one narrow conclusion:

1. backend-only mapper changes (`StateLocalMapEngine`, `sort`) are not enough
2. pure resident-local direct execution kills reuse and becomes compute/update bound
3. full or partial positive tapes that still re-enter the old mapper are not enough

So the next implementation target should be:

- **state-local replay with reused compact views for compute**, not dynamic remap and not pure duplicated direct execution
- the best concrete next step is likely a **manual RNS replay/update path on top of a compact state-local view**, because the direct resident-local experiments show that removing the mapper is easy, but preserving reuse in the compute/update path is the hard part

## Positive overlap and hot-substate analysis on real q-states

To test whether a compact reused working set is even plausible, I extended
`gege_state_compiled_runtime_analyzer` to measure, for a real q-state:

- per-batch positive unique local-ID touches
- overlap between one batch and the previous `1/2/4/8` batches
- hot-substate coverage: how many rows explain `50/70/90%` of batch-unique positive touches

Outputs:

- `/dev/shm/smansou2_ge2/state_compiled_overlap_twitter_state0_20260409.txt`
- `/dev/shm/smansou2_ge2/state_compiled_overlap_fb86m_state0_20260409.txt`

Important scope note:

- this analysis is **positive-only**
- it answers whether there is enough repeated local access structure inside a visible q-state to justify a compact reused compute view
- it does **not** yet solve the negative side; that still needs descriptor/replay handling

### Twitter q4 state 0

Command shape:

```bash
./build_ge2env_ge2py39/gege/gege_state_compiled_runtime_analyzer \
  /home/smansou2/newCode/ge2/dandelion-dev/datasets/twitter_16p_paper_10k_eval \
  CUSTOM 16 4 1 0 0 1 \
  --state-idx 0 \
  --batch-size 50000 \
  --num-chunks 50 \
  --negatives-per-positive 1000 \
  --degree-fraction 0.5
```

Result:

| Metric | Value |
| --- | ---: |
| `positive_edges` | `91,353,525` |
| `batches` | `1,828` |
| `active_partition_rows` | `10,413,060` |
| `positive_unique_rows` | `8,847,213` |
| `positive_density` | `0.849627` |
| `total_batch_unique_touches` | `61,359,267` |
| `positive_batch_touch_reuse` | `6.935435x` |
| `avg_positive_unique_per_batch` | `33,566` |
| `max_positive_unique_per_batch` | `49,341` |
| `compile_ms` | `10,687.495` |
| positive-only workspace | `0.078 GiB` |
| positive unique arena | `0.229 GiB` |
| materialized negative IDs upper bound | `1.362 GiB` |

Window overlap:

| Previous window | Avg overlap with current batch |
| --- | ---: |
| `1` batch | `0.165992` |
| `2` batches | `0.224218` |
| `4` batches | `0.289735` |
| `8` batches | `0.361748` |

Hot-substate coverage:

| Rows kept | Touch coverage |
| --- | ---: |
| top `0.1%` (`10,414` rows) | `4.8610%` |
| top `0.5%` (`52,066` rows) | `14.5105%` |
| top `1.0%` (`104,131` rows) | `21.1624%` |
| top `2.0%` (`208,262` rows) | `28.6578%` |
| top `5.0%` (`520,653` rows) | `40.2369%` |
| top `10.0%` (`1,041,306` rows) | `51.3819%` |

Rows needed to explain the positive batch-unique touches:

| Target | Rows | Fraction of active rows |
| --- | ---: | ---: |
| `50%` | `964,224` | `9.2598%` |
| `70%` | `2,455,964` | `23.5854%` |
| `90%` | `5,205,853` | `49.9935%` |

Interpretation:

- Twitter does have real positive-side reuse inside one q-state.
- The reuse is not tiny: `50%` of positive batch-unique touches come from only about `9.3%` of active rows.
- Adjacent-batch overlap is only moderate (`16.6%`), but short windows matter: the previous `8` batches already cover about `36.2%` of the current batch's positive unique rows on average.
- This is enough to keep the compact-view hypothesis alive.
- But it is not enough to justify pure direct resident-local execution; the working set still has to be compacted and reused intelligently.

### FB86M q4 state 0

Command shape:

```bash
./build_ge2env_ge2py39/gege/gege_state_compiled_runtime_analyzer \
  /home/smansou2/newCode/ge2/dandelion-dev/datasets/freebase86m_16p_paper_10k_eval \
  CUSTOM 16 4 1 0 0 1 \
  --state-idx 0 \
  --batch-size 100000 \
  --num-chunks 50 \
  --negatives-per-positive 1000 \
  --degree-fraction 0.5
```

Result:

| Metric | Value |
| --- | ---: |
| `positive_edges` | `19,238,553` |
| `batches` | `193` |
| `active_partition_rows` | `21,513,540` |
| `positive_unique_rows` | `11,061,734` |
| `positive_density` | `0.514175` |
| `total_batch_unique_touches` | `26,460,464` |
| `positive_batch_touch_reuse` | `2.392072x` |
| `avg_positive_unique_per_batch` | `137,100` |
| `max_positive_unique_per_batch` | `154,979` |
| `compile_ms` | `4,167.166` |
| positive-only workspace | `0.162 GiB` |
| positive unique arena | `0.099 GiB` |
| materialized negative IDs upper bound | `0.144 GiB` |

Window overlap:

| Previous window | Avg overlap with current batch |
| --- | ---: |
| `1` batch | `0.048874` |
| `2` batches | `0.079712` |
| `4` batches | `0.122709` |
| `8` batches | `0.172141` |

Hot-substate coverage:

| Rows kept | Touch coverage |
| --- | ---: |
| top `0.1%` (`21,514` rows) | `2.7456%` |
| top `0.5%` (`107,568` rows) | `7.6670%` |
| top `1.0%` (`215,136` rows) | `11.7344%` |
| top `2.0%` (`430,271` rows) | `17.6754%` |
| top `5.0%` (`1,075,677` rows) | `29.7860%` |
| top `10.0%` (`2,151,354` rows) | `44.4892%` |

Rows needed to explain the positive batch-unique touches:

| Target | Rows | Fraction of active rows |
| --- | ---: | ---: |
| `50%` | `2,637,411` | `12.2593%` |
| `70%` | `5,051,497` | `23.4805%` |
| `90%` | `8,415,688` | `39.1181%` |

Interpretation:

- FB86M has weaker positive-side locality than Twitter.
- Adjacent-batch overlap is low (`4.9%`), but there is still a meaningful window effect (`17.2%` by `8` batches).
- The hot-substate hypothesis is still alive on FB86M, but the positive-side payoff is weaker than on Twitter.

### Overall conclusion from the overlap analysis

The compact reused-view idea survives this offline gate.

What the numbers say:

1. There is enough positive-side redundancy to justify a compact reused compute view, especially on Twitter.
2. The redundancy is **windowed**, not purely adjacent-batch. So the runtime object should be a small rolling hot-substate, not only "reuse the previous batch."
3. Materializing full negative tapes is still too large on Twitter; the negative side must stay descriptor/seed-based.
4. The best next implementation is therefore:
   - compact state-local positive replay metadata
   - rolling hot-substate over local IDs
   - descriptor-based negative replay
   - manual/fused RNS update path over that compact view

This is stronger evidence for the next step than any of the earlier mapper-only or full-tape prototypes.
