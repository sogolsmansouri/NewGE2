# FB86M epoch-2 critical path and overlap details

Source image:
`/home/smansou2/newCode_fed9d052/ge2/dandelion-dev/experiment_logs/fb_epoch_state_t_overlap_timeline_20260507.png`

Source training log:
`/home/smansou2/codex_runs/exp_logs/fb86m_p32_q4_bounded_minmax_10frames_fed9_py312_3e_20260506_200625_train.log`

Run checkout:
`/home/smansou2/newCode_fed9d052/ge2/dandelion-dev`

Commit:
`fed9d052c8d0b31004a4fbda000f5cc7774a5e62`

The concrete epoch used here is epoch 2, because it is the steady-state epoch in the 103 s FB run.

## What this log can and cannot show

The log has enough information to explain the epoch-level critical path, all per-state timings, aggregate swap time, and aggregate frame-cache behavior.

The log does not contain per-transition partition-id lists, hidden-frame ids, or per-swap timing lines, because this run had `GEGE_PARTITION_BUFFER_PIPELINE_TIMING=0`. Therefore, exact partition lists like "transition 45 -> 46 evicted these exact partition ids" cannot be reconstructed from this log alone. The aggregate counters still prove whether the preload path was used, whether fallback happened, and whether stale writeback was delayed.

## Epoch shape

Scheduler summary from epoch 2:

```text
Generating BOUNDED_GREEDY_COVER_Q4 Ordering states=93 transitions=92 total_buckets=1024 max_admits=3 transition_admits=228 overlap_hist=[1:44, 2:48] admit_hist=[2:48, 3:44] edge_total=304727650 edge_min=2476910 edge_max=4349765
BOUNDED_GREEDY_COVER_Q4 retained_avg=1.522 pre_reorder_retained_avg=1.522
Using bounded GREEDY_COVER q4 ordering for CUSTOM schedule with 1 active device(s)
```

Meaning:

- One epoch is 93 visible partition states.
- There are 92 transitions between states.
- Each visible state has 4 resident partitions, because this is q4.
- The schedule covers all 1024 directed edge buckets for 32 partitions.
- Each transition admits at most 3 new partitions.
- 48 transitions retain 2 partitions and admit 2.
- 44 transitions retain 1 partition and admit 3.
- Average retained partitions per transition is 1.522.
- Average admitted partitions per transition is 228 / 92 = 2.478.
- The total planned training edges are 304,727,650.

The epoch progress line reaches the planned total:

```text
Edges processed: [304727650/304727650], 100.00%
Epoch Runtime: 103580ms
```

So for this epoch, the training loop processed the full scheduled edge total.

## Critical path accounting

Epoch-2 wall time:

```text
Epoch Runtime: 103580ms
[perf][epoch 2][logical_lane 0] process_ms_total=103478.968 rebuild_ms_total=63.191 cycle_ms_total=103542.159 states=93
```

For wall-time reconstruction, `process_ms_total=103478.968` is the closest critical-path sum. It is the sum over states of:

```text
batch_fetch + gpu_load + map + compute + embedding_update + embedding_update_g + finalize
```

The printed `rebuild_ms_total=63.191` is a separate tag for swap rebuild work, but that rebuild is also inside the `get_next_swap_path` portion of `batch_fetch`. Do not add `rebuild_ms_total`, `swap_update_ms`, `edge_sample_ms`, or `negative_sampler` totals again when reconstructing epoch wall time. They are nested counters.

Critical-path state-region totals:

| Region | Total ms | Share of process total |
|---|---:|---:|
| batch_fetch | 37900.538 | 36.63% |
| gpu_load | 7152.966 | 6.91% |
| map | 8.773 | 0.008% |
| compute | 49881.649 | 48.20% |
| embedding_update | 8516.075 | 8.23% |
| finalize | 18.968 | 0.018% |

Nested timing inside `batch_fetch`:

```text
[perf][epoch 2][batch_fetch] total_ms=37900.538 get_next_batch_ms=16044.937 get_next_direct_ms=11.305 get_next_swap_path_ms=16024.051 get_next_swap_overhead_ms=0.261 edge_sample_ms=21823.605 node_sample_ms=0.000 load_cpu_parameters_ms=14.638 device_prepare_ms=0.000 perform_map_ms=0.000 overhead_ms=6.572
```

Interpretation:

- `batch_fetch` is the training-loop wait/work before a batch is handed to GPU load.
- `get_next_swap_path_ms=16024.051` is paid at state boundaries. It includes the swap update, state rebuild, and swap synchronization.
- `get_next_direct_ms=11.305` is the tiny direct iterator path for batches that do not cross a state boundary.
- `edge_sample_ms=21823.605` is the work to fetch edge data, generate negatives, collect and map ids, and finalize batch tensors.
- `load_cpu_parameters_ms=14.638` is negligible for this run.

Nested timing inside `edge_sample`:

```text
[perf][epoch 2][edge_sample] total_ms=21823.605 get_edges_ms=3390.773 negative_sample_ms=10727.611 collect_ids_ms=51.556 map_lookup_ms=6698.601 compact_active_ms=0.000 verify_ms=1.970 remap_assign_ms=68.427 finalize_ms=747.534 unaccounted_ms=79.767
```

Interpretation:

- `negative_sample_ms=10727.611` is the largest edge-sample substage.
- `map_lookup_ms=6698.601` is also large; this maps sampled ids into the resident buffer/local view.
- `get_edges_ms=3390.773` fetches positive edges from the active edge buckets.
- `finalize_ms=747.534` finalizes sampled/remapped batch data.

Negative sampler breakdown:

```text
[perf][epoch 2][negative_sampler] calls=12274 call_ms_total=10652.357 plan_lock_calls=0 plan_lock_wait_ms_total=0.000
[perf][epoch 2][negative_sampler_breakdown] uniform_randint_ms=2129.346 sample_edge_randint_ms=2380.215 materialize_ms=3944.914 filter_ms=2166.943 state_pool_hits=0 planned_uniform_fetches=0 cuda_calls=12274 cpu_calls=0
```

Interpretation:

- Negative sampling ran on CUDA for all 12,274 calls.
- There was no plan-lock overhead.
- There was no batched negative plan in this run.
- The sampler time is nested inside `edge_sample`, which is nested inside `batch_fetch`.

## State t in the image

The PNG uses concrete mid-epoch state position 46 as state `t`.

The three adjacent visible training states are:

| Role | state_pos | state_idx | active_buckets | active_edges | batches | process_ms | batch_fetch_ms | gpu_load_ms | map_ms | compute_ms | update_ms | finalize_ms | rebuild_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t-1 | 45 | 45 | 12 | 3513390 | 71 | 946.571 | 336.382 | 62.471 | 0.078 | 464.508 | 82.948 | 0.185 | 0.614 |
| t | 46 | 46 | 12 | 3766635 | 76 | 1150.297 | 386.283 | 77.960 | 0.092 | 589.178 | 96.587 | 0.197 | 0.728 |
| t+1 | 47 | 47 | 10 | 3038032 | 61 | 1218.251 | 439.830 | 90.943 | 0.112 | 590.409 | 96.677 | 0.280 | 0.668 |

For state `t=46`, the critical path is:

1. Enter state 46. If this is the first batch after state 45, the fetch path includes the state-boundary swap into state 46.
2. Fetch and sample 76 batches. Total state `batch_fetch_ms=386.283`.
3. Load GPU batch data and resident parameters. `gpu_load_ms=77.960`.
4. Perform dense graph local id mapping. `map_ms=0.092`.
5. Forward/backward/decoder compute. `compute_ms=589.178`.
6. Apply embedding updates for resident partitions. `embedding_update_ms=96.587`.
7. Clear/finish batch bookkeeping. `finalize_ms=0.197`.

The state-46 `process_ms=1150.297` is the sum of these critical-path regions.

The `rebuild_ms=0.728` shown on the same state line is the rebuild sample associated with entering this state. It is useful for labeling the boundary, but it is also inside the state-boundary fetch path, so it is not an extra wall-time segment to add on top of `batch_fetch`.

## What overlaps around state t

For state `t`, two background partition-side activities can overlap with the visible training work.

### t+1 admit preload

After the current visible state is installed and the current in-memory subgraph is ready, the storage starts an async admit preload for the next state. That preload reads the partitions that will be admitted at transition `t -> t+1`, copies them through host/pinned staging, and places them into hidden GPU frames.

For this FB epoch:

```text
hidden_publish_parts=228
hidden_publish_rows=613135854
fallback_visible_admit_parts=0
fallback_visible_admit_rows=0
preload_miss_swaps=0
partial_preload_swaps=0
async_admit_valid_before_swap_swaps=92
reserved_preload_frames_avg=2.48
```

Interpretation:

- Every one of the 92 state transitions had a valid async admit preload before the swap.
- No transition missed the preload.
- No transition had a partial preload.
- No partition had to be admitted through the visible fallback path.
- All 228 admitted partitions were published from hidden frames.
- The average reserved hidden frames is 2.48, matching the average transition admit count.

In the diagram, this is the lower `t+1` line: while state `t` trains, the next state's admitted partitions are loaded into hidden frames.

### t-1 stale writeback

When state `t` begins, hidden frames that were preloaded during state `t-1` are published into the visible logical slots. The old physical frames that previously occupied those logical slots contain updated embeddings from the just-finished state. Those old frames become stale frames and must be written back to host/storage.

For this FB epoch:

```text
delayed_stale_writeback_swaps=92
async_evict_in_flight_before_swap_swaps=3
stale_backlog_before_swap_max=1
stale_backlog_after_publish_max=3
free_frames_before_swap_avg=5.97
free_frames_after_publish_avg=3.52
```

Interpretation:

- All 92 swaps used delayed stale writeback.
- Only 3 swaps found an async evict/writeback still in flight before the next swap.
- The stale backlog never exceeded 3 frames after publish.
- The system usually had nearly all 6 hidden frames free before the next swap (`free_frames_before_swap_avg=5.97`).
- After publishing hidden frames, about 3.52 hidden frames remained free on average, because about 2.48 were consumed for the admitted partitions.

In the diagram, this is the lower `t-1` line: while state `t` trains, the stale frames from the previous state are written back asynchronously.

## Swap mechanics in this run

Aggregate swap counters:

```text
[perf][epoch 2] swap_count=92 swap_barrier_wait_ms=0.059 swap_update_ms=15960.444 swap_rebuild_ms=63.191 swap_sync_wait_ms=0.096
```

Average per swap:

- `swap_update_ms`: 15960.444 / 92 = 173.483 ms.
- `swap_rebuild_ms`: 63.191 / 92 = 0.687 ms.
- Barrier and sync waits are effectively zero for single GPU.

What happens at a state-boundary swap:

1. The dataloader finishes the last batch of the current state.
2. It enters the swap path in `getNextBatch`.
3. It waits at the swap read barrier. For this single-GPU run, total barrier wait was only 0.059 ms.
4. It resets negative-sampler plan cache.
5. It calls `graph_storage_->updateInMemorySubGraph(device_idx)`.
6. Storage computes the next visible partition state.
7. It determines retained partitions, evicted partitions, and admitted partitions.
8. With q4 and max_admits=3, each transition admits either 2 or 3 partitions.
9. The async admit preload from the previous state is consumed.
10. Because the preload was hidden-only, there are no visible installs.
11. The preloaded hidden frames are published by changing the logical-slot to physical-frame mapping.
12. The old physical frames in those logical slots are kept as stale frames if they contain updated evicted data.
13. Delayed stale writeback is launched asynchronously for those old frames.
14. Partition metadata is updated: `present`, `buffer_idx`, and physical-frame mapping are moved to the next state.
15. Frame-cache tensor views are refreshed.
16. Batches are rebuilt for the new active edge buckets.
17. It leaves the swap path and returns the first batch of the new state.
18. It starts async admit preload for the next transition, so the next state's admitted partitions load while the current state's batches train.

Important consequence:

The partition-side pipeline is active, but the prepared-batch/trainer pipeline is not. The visible training loop still serially executes fetch, GPU load, map, compute, update, and finalize for each batch. The overlap here is specifically partition preload/writeback overlap across states.

## Why the epoch is about 103 s

The epoch is dominated by:

- 49.882 s compute.
- 37.901 s batch fetch.
- 8.516 s embedding update.
- 7.153 s GPU load.

Inside the 37.901 s batch-fetch total:

- 16.024 s is the state-boundary swap path.
- 21.824 s is edge sampling.

Inside the 21.824 s edge-sampling total:

- 10.728 s is negative sampling.
- 6.699 s is map lookup.
- 3.391 s is positive edge fetch.

The async partition preload is working well in this run. We know that because `preload_miss_swaps=0`, `partial_preload_swaps=0`, `fallback_visible_admit_parts=0`, and `async_admit_valid_before_swap_swaps=92`. The remaining 16.024 s swap path is not bulk visible H2D fallback. It is the boundary work that remains after overlap: consuming/publishing hidden frames, metadata, any required writeback coordination, subgraph update, and batch rebuild.

## Edge and bucket coverage details

The scheduler reports:

```text
total_buckets=1024 edge_total=304727650
```

The progress reporter confirms:

```text
Edges processed: [304727650/304727650], 100.00%
```

The per-state perf lines report valid `active_buckets` and `active_edges` for state positions 0 through 91, then print `-1` for state position 92:

```text
[perf][epoch 2][logical_lane 0][state_pos 92] state_idx=92 active_buckets=-1 active_edges=-1 batches=62 ...
```

This is a metadata sampling artifact, not missing training. From the scheduler totals:

- Sum of state positions 0-91 active buckets: 1012.
- Missing final state buckets: 1024 - 1012 = 12.
- Sum of state positions 0-91 active edges: 300,949,880.
- Missing final state edges: 304,727,650 - 300,949,880 = 3,777,770.
- State position 92 still ran 62 batches.

So state position 92 should be treated as the final 12 buckets and 3,777,770 edges, even though the per-state metadata line emits `-1`.

## Full epoch-2 state table

Columns:

- `pos`: position in the visible-state sequence.
- `state`: logged state index.
- `bkt`: active bucket count from the perf sample.
- `edges`: active edge count from the perf sample.
- `batches`: number of batches timed in that state.
- `process`: critical-path process time for that state.
- `fetch`: batch-fetch region.
- `gpu`: GPU load region.
- `map`: local id mapping region.
- `compute`: forward/backward/decoder region.
- `update`: embedding update region.
- `final`: finalize/clear region.
- `rebuild`: swap rebuild tag associated with entering the state.

| pos | state | bkt | edges | batches | process | fetch | gpu | map | compute | update | final | rebuild |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 10 | 3448892 | 69 | 1090.471 | 356.754 | 102.680 | 0.046 | 504.290 | 126.580 | 0.122 | 0.000 |
| 1 | 1 | 9 | 3028107 | 61 | 1177.152 | 419.599 | 84.287 | 0.092 | 572.700 | 100.266 | 0.207 | 0.753 |
| 2 | 2 | 13 | 3490234 | 70 | 1008.181 | 348.199 | 65.745 | 0.075 | 505.317 | 88.682 | 0.163 | 0.680 |
| 3 | 3 | 12 | 4349765 | 87 | 1130.329 | 377.113 | 75.814 | 0.106 | 574.246 | 102.862 | 0.187 | 0.776 |
| 4 | 4 | 12 | 3954065 | 80 | 1429.517 | 518.887 | 107.922 | 0.125 | 673.404 | 128.898 | 0.281 | 0.675 |
| 5 | 5 | 12 | 4319912 | 87 | 1309.103 | 423.680 | 92.711 | 0.122 | 683.037 | 109.277 | 0.276 | 0.793 |
| 6 | 6 | 12 | 3693315 | 74 | 1430.614 | 470.836 | 107.372 | 0.126 | 713.942 | 138.082 | 0.257 | 0.762 |
| 7 | 7 | 7 | 2751386 | 56 | 1220.393 | 417.941 | 89.647 | 0.101 | 609.190 | 103.299 | 0.214 | 0.789 |
| 8 | 8 | 13 | 3581346 | 72 | 958.536 | 357.167 | 67.934 | 0.075 | 449.777 | 83.411 | 0.170 | 0.718 |
| 9 | 9 | 11 | 3172906 | 64 | 1140.872 | 398.359 | 81.806 | 0.077 | 553.125 | 107.329 | 0.174 | 0.669 |
| 10 | 10 | 11 | 3193103 | 64 | 1075.510 | 357.163 | 71.472 | 0.085 | 556.162 | 90.438 | 0.189 | 0.737 |
| 11 | 11 | 13 | 3235090 | 65 | 1120.260 | 426.413 | 81.274 | 0.099 | 522.286 | 89.985 | 0.204 | 0.722 |
| 12 | 12 | 12 | 3519629 | 71 | 1061.182 | 357.749 | 71.070 | 0.124 | 547.652 | 84.369 | 0.218 | 0.715 |
| 13 | 13 | 10 | 3114901 | 63 | 1283.831 | 479.582 | 91.854 | 0.110 | 603.811 | 108.230 | 0.243 | 0.736 |
| 14 | 14 | 14 | 3363668 | 68 | 1048.493 | 382.290 | 73.226 | 0.101 | 509.380 | 83.286 | 0.209 | 0.628 |
| 15 | 15 | 11 | 2909879 | 59 | 1118.718 | 383.789 | 79.449 | 0.082 | 555.106 | 100.117 | 0.174 | 0.663 |
| 16 | 16 | 12 | 3081435 | 62 | 973.213 | 308.531 | 57.851 | 0.081 | 531.337 | 75.194 | 0.221 | 0.747 |
| 17 | 17 | 11 | 3103239 | 63 | 1163.217 | 519.158 | 67.825 | 0.075 | 500.497 | 75.500 | 0.162 | 0.768 |
| 18 | 18 | 13 | 3439983 | 69 | 1005.847 | 331.112 | 66.050 | 0.100 | 528.826 | 79.531 | 0.228 | 0.608 |
| 19 | 19 | 10 | 3245766 | 65 | 1299.299 | 554.055 | 83.687 | 0.114 | 570.237 | 90.973 | 0.233 | 0.675 |
| 20 | 20 | 12 | 3770062 | 76 | 1054.405 | 395.477 | 77.975 | 0.104 | 501.764 | 78.852 | 0.233 | 0.539 |
| 21 | 21 | 11 | 3936477 | 79 | 1474.257 | 662.252 | 96.733 | 0.093 | 610.895 | 104.068 | 0.215 | 0.650 |
| 22 | 22 | 14 | 3460278 | 70 | 1338.727 | 467.098 | 99.321 | 0.104 | 652.889 | 119.102 | 0.213 | 0.761 |
| 23 | 23 | 12 | 3225977 | 65 | 1143.227 | 373.883 | 81.721 | 0.093 | 587.234 | 100.076 | 0.220 | 0.774 |
| 24 | 24 | 13 | 3691161 | 74 | 1040.002 | 378.506 | 76.972 | 0.110 | 505.263 | 78.910 | 0.243 | 0.643 |
| 25 | 25 | 11 | 2884283 | 58 | 1412.225 | 643.836 | 87.309 | 0.101 | 572.702 | 108.060 | 0.218 | 0.656 |
| 26 | 26 | 10 | 2631059 | 53 | 974.216 | 341.450 | 62.290 | 0.084 | 491.087 | 79.122 | 0.184 | 0.635 |
| 27 | 27 | 11 | 2814442 | 57 | 920.544 | 352.300 | 69.549 | 0.080 | 429.289 | 69.161 | 0.166 | 0.663 |
| 28 | 28 | 14 | 3384438 | 68 | 933.297 | 332.550 | 63.212 | 0.078 | 452.802 | 84.484 | 0.172 | 0.676 |
| 29 | 29 | 11 | 2978343 | 60 | 1093.335 | 369.309 | 75.697 | 0.084 | 549.268 | 98.792 | 0.186 | 0.656 |
| 30 | 30 | 8 | 2655235 | 54 | 979.823 | 353.619 | 67.525 | 0.104 | 474.846 | 83.539 | 0.189 | 0.707 |
| 31 | 31 | 8 | 3694142 | 74 | 903.100 | 319.321 | 58.252 | 0.067 | 449.443 | 75.837 | 0.180 | 0.690 |
| 32 | 32 | 12 | 4075327 | 82 | 1212.011 | 454.442 | 94.002 | 0.094 | 550.786 | 112.499 | 0.188 | 0.601 |
| 33 | 33 | 8 | 3093424 | 62 | 1335.584 | 466.740 | 102.735 | 0.092 | 643.590 | 122.202 | 0.225 | 0.686 |
| 34 | 34 | 10 | 2718487 | 55 | 1005.156 | 368.296 | 69.351 | 0.086 | 482.185 | 85.040 | 0.197 | 0.619 |
| 35 | 35 | 12 | 3326922 | 67 | 899.902 | 305.445 | 54.743 | 0.093 | 471.980 | 67.430 | 0.211 | 0.687 |
| 36 | 36 | 10 | 2798452 | 56 | 1363.672 | 640.772 | 82.433 | 0.093 | 546.866 | 93.301 | 0.208 | 0.651 |
| 37 | 37 | 9 | 2476910 | 50 | 958.051 | 366.607 | 63.282 | 0.089 | 456.410 | 71.461 | 0.202 | 0.706 |
| 38 | 38 | 11 | 2794269 | 56 | 848.003 | 345.560 | 53.329 | 0.108 | 386.682 | 62.161 | 0.163 | 0.597 |
| 39 | 39 | 12 | 3043918 | 61 | 948.864 | 347.572 | 59.560 | 0.083 | 469.694 | 71.787 | 0.167 | 0.865 |
| 40 | 40 | 11 | 2844557 | 57 | 1023.278 | 361.438 | 73.277 | 0.074 | 498.853 | 89.460 | 0.177 | 0.723 |
| 41 | 41 | 10 | 2683830 | 54 | 932.063 | 323.185 | 59.575 | 0.074 | 475.921 | 73.158 | 0.150 | 0.642 |
| 42 | 42 | 10 | 2885647 | 58 | 881.346 | 296.959 | 59.242 | 0.079 | 460.115 | 64.789 | 0.163 | 0.678 |
| 43 | 43 | 10 | 3473667 | 70 | 951.751 | 312.507 | 58.196 | 0.110 | 507.886 | 72.880 | 0.173 | 0.595 |
| 44 | 44 | 9 | 2835161 | 57 | 1135.837 | 384.700 | 77.301 | 0.085 | 577.197 | 96.369 | 0.184 | 0.748 |
| 45 | 45 | 12 | 3513390 | 71 | 946.571 | 336.382 | 62.471 | 0.078 | 464.508 | 82.948 | 0.185 | 0.614 |
| 46 | 46 | 12 | 3766635 | 76 | 1150.297 | 386.283 | 77.960 | 0.092 | 589.178 | 96.587 | 0.197 | 0.728 |
| 47 | 47 | 10 | 3038032 | 61 | 1218.251 | 439.830 | 90.943 | 0.112 | 590.409 | 96.677 | 0.280 | 0.668 |
| 48 | 48 | 13 | 3423878 | 69 | 1031.157 | 349.290 | 66.869 | 0.102 | 533.946 | 80.752 | 0.198 | 0.620 |
| 49 | 49 | 10 | 3460262 | 70 | 1133.687 | 407.322 | 89.684 | 0.072 | 529.704 | 106.711 | 0.194 | 0.754 |
| 50 | 50 | 12 | 3816332 | 77 | 1176.995 | 416.769 | 80.475 | 0.108 | 587.010 | 92.415 | 0.218 | 0.716 |
| 51 | 51 | 10 | 3293156 | 66 | 1233.581 | 404.747 | 81.306 | 0.107 | 650.373 | 96.790 | 0.257 | 0.780 |
| 52 | 52 | 12 | 3332981 | 67 | 1137.594 | 406.874 | 83.898 | 0.094 | 550.056 | 96.459 | 0.212 | 0.747 |
| 53 | 53 | 11 | 2944039 | 59 | 1112.126 | 354.175 | 69.558 | 0.077 | 602.824 | 85.316 | 0.176 | 0.647 |
| 54 | 54 | 9 | 3091064 | 62 | 979.494 | 342.033 | 64.720 | 0.093 | 497.080 | 75.375 | 0.193 | 0.711 |
| 55 | 55 | 11 | 3337736 | 67 | 1044.991 | 390.205 | 71.461 | 0.077 | 488.968 | 94.066 | 0.214 | 0.641 |
| 56 | 56 | 12 | 3227986 | 65 | 1092.618 | 361.066 | 72.499 | 0.125 | 567.634 | 91.090 | 0.204 | 0.641 |
| 57 | 57 | 10 | 3035566 | 61 | 1086.671 | 363.576 | 69.329 | 0.130 | 568.933 | 84.451 | 0.253 | 0.759 |
| 58 | 58 | 12 | 3254944 | 66 | 1163.224 | 470.491 | 63.100 | 0.109 | 557.447 | 71.854 | 0.223 | 0.720 |
| 59 | 59 | 12 | 3403571 | 69 | 1266.473 | 575.075 | 81.720 | 0.089 | 517.776 | 91.614 | 0.199 | 0.658 |
| 60 | 60 | 12 | 4305089 | 87 | 1243.654 | 490.260 | 95.528 | 0.109 | 555.594 | 101.946 | 0.217 | 0.668 |
| 61 | 61 | 10 | 3476706 | 70 | 1397.446 | 521.677 | 115.333 | 0.124 | 637.069 | 123.008 | 0.235 | 0.670 |
| 62 | 62 | 9 | 2546562 | 51 | 1128.729 | 406.372 | 87.322 | 0.093 | 543.425 | 91.298 | 0.221 | 0.546 |
| 63 | 63 | 10 | 2974965 | 60 | 867.700 | 289.079 | 50.468 | 0.067 | 462.402 | 65.542 | 0.142 | 0.676 |
| 64 | 64 | 14 | 3593511 | 72 | 954.902 | 368.296 | 71.952 | 0.088 | 433.697 | 80.698 | 0.171 | 0.519 |
| 65 | 65 | 12 | 3536197 | 71 | 1166.231 | 384.066 | 80.518 | 0.092 | 603.328 | 98.027 | 0.201 | 0.755 |
| 66 | 66 | 12 | 3265089 | 66 | 1167.763 | 388.616 | 75.874 | 0.112 | 600.246 | 102.648 | 0.267 | 0.769 |
| 67 | 67 | 12 | 4033378 | 81 | 1114.369 | 344.355 | 74.751 | 0.109 | 613.644 | 81.256 | 0.256 | 0.755 |
| 68 | 68 | 12 | 3420541 | 69 | 1375.206 | 519.286 | 106.775 | 0.110 | 621.640 | 127.182 | 0.214 | 0.760 |
| 69 | 69 | 8 | 2614947 | 53 | 1105.131 | 430.644 | 87.919 | 0.100 | 488.745 | 97.514 | 0.208 | 0.562 |
| 70 | 70 | 10 | 2818438 | 57 | 890.453 | 307.513 | 60.919 | 0.069 | 450.323 | 71.460 | 0.169 | 0.672 |
| 71 | 71 | 11 | 3082763 | 62 | 918.436 | 305.439 | 59.786 | 0.078 | 478.868 | 74.105 | 0.160 | 0.674 |
| 72 | 72 | 10 | 2801485 | 57 | 1044.355 | 337.713 | 71.789 | 0.081 | 548.128 | 86.421 | 0.222 | 0.683 |
| 73 | 73 | 11 | 3211521 | 65 | 1039.100 | 421.970 | 64.501 | 0.099 | 473.972 | 78.367 | 0.190 | 0.613 |
| 74 | 74 | 11 | 3080331 | 62 | 1322.826 | 634.320 | 79.228 | 0.099 | 518.683 | 90.266 | 0.230 | 0.614 |
| 75 | 75 | 9 | 3266030 | 66 | 1018.865 | 353.118 | 72.994 | 0.078 | 500.789 | 91.717 | 0.169 | 0.718 |
| 76 | 76 | 8 | 3035769 | 61 | 1096.302 | 402.880 | 79.962 | 0.112 | 518.765 | 94.403 | 0.180 | 0.608 |
| 77 | 77 | 8 | 2714787 | 55 | 988.108 | 399.136 | 76.417 | 0.074 | 432.594 | 79.741 | 0.146 | 0.609 |
| 78 | 78 | 12 | 3438073 | 69 | 916.605 | 331.001 | 65.869 | 0.080 | 446.291 | 73.210 | 0.153 | 0.592 |
| 79 | 79 | 12 | 3795850 | 76 | 1094.350 | 369.229 | 72.632 | 0.083 | 557.922 | 94.289 | 0.195 | 0.645 |
| 80 | 80 | 13 | 3225295 | 65 | 1269.734 | 450.479 | 89.431 | 0.112 | 628.545 | 100.842 | 0.325 | 0.660 |
| 81 | 81 | 12 | 3424426 | 69 | 1051.932 | 418.962 | 79.717 | 0.104 | 463.486 | 89.451 | 0.212 | 0.686 |
| 82 | 82 | 13 | 3260338 | 66 | 1325.224 | 591.341 | 73.225 | 0.110 | 560.548 | 99.751 | 0.248 | 0.782 |
| 83 | 83 | 12 | 3374737 | 68 | 1090.701 | 341.391 | 69.251 | 0.105 | 592.149 | 87.587 | 0.219 | 0.749 |
| 84 | 84 | 12 | 4099220 | 82 | 1202.296 | 443.172 | 83.121 | 0.104 | 588.862 | 86.791 | 0.247 | 0.769 |
| 85 | 85 | 9 | 3509033 | 71 | 1413.442 | 544.229 | 103.888 | 0.093 | 635.105 | 129.912 | 0.215 | 0.967 |
| 86 | 86 | 10 | 3719138 | 75 | 1208.465 | 439.733 | 93.278 | 0.081 | 555.141 | 120.058 | 0.174 | 0.868 |
| 87 | 87 | 12 | 3871439 | 78 | 1212.598 | 426.458 | 86.013 | 0.101 | 591.259 | 108.530 | 0.236 | 0.636 |
| 88 | 88 | 10 | 2924926 | 59 | 1265.720 | 461.809 | 99.697 | 0.116 | 603.858 | 99.992 | 0.248 | 0.571 |
| 89 | 89 | 9 | 2566925 | 52 | 978.749 | 359.365 | 69.651 | 0.110 | 470.499 | 78.921 | 0.202 | 0.687 |
| 90 | 90 | 10 | 2773227 | 56 | 895.572 | 340.982 | 54.709 | 0.090 | 427.105 | 72.501 | 0.184 | 0.605 |
| 91 | 91 | 12 | 3076484 | 62 | 933.944 | 296.661 | 55.968 | 0.090 | 509.709 | 71.285 | 0.232 | 0.696 |
| 92 | 92 | -1 raw, 12 inferred | -1 raw, 3777770 inferred | 62 | 1198.914 | 624.665 | 75.229 | 0.057 | 403.443 | 95.402 | 0.118 | 0.535 |

## Short answer

For the one real FB epoch, visible training is not missing buckets. The 103.58 s epoch is mostly compute plus batch-fetch work. The partition-side overlap is working: every transition had a valid async preload, all admitted partitions were published from hidden frames, and no visible fallback admit happened. The overlapped work is specifically:

- `t+1`: async host/H2D preload into hidden frames while state `t` trains.
- `t-1`: delayed stale writeback of old frames while state `t` trains.

The remaining swap cost is the non-overlapped state-boundary work inside `get_next_swap_path`, averaging about 173.5 ms per transition.
