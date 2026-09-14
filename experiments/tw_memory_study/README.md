# ARC TW Memory Study, September 14, 2026

Authorized execution: q=4 only. q=2 and q=3 are planning tables, not queued runs.
The runtime changes are inherited from commit 9777b506ac3b286ce03450fb62aa37a12a0f782a,
on top of the stable experiment baseline 57c73b5e0a895eb74db8fb22df300e34cae00b2f.
The enclosing Git commit freezes this harness, inputs, and schedule certificates.

## Workload and Budget

Full TW: 41,652,230 nodes, 1,468,345,182 training edges, real width 100,
FP32 embeddings and Adagrad accumulator. W = 800 * nodes = 31.033330 GiB.
The padded frame footprint is F(p) = 800 * ceil(nodes/p) bytes.

For a selected parameter-frame cap c and total physical frames k:

    p0 = max(q, ceil(k*W/c)); validate k*F(p) <= c.

This is a frame-only lower bound, not a total GPU-memory guarantee or a
prediction of the fastest configuration. c=20 and c=32 GiB in the planning
table are experimental frame caps, not measured free memory. Whole-state
GPU graphs, graph-prefetch coexistence, maps, batch workspace, runtime state,
and allocator requirements must also fit. The ARC launch uses a 0.9 capacity
payload preflight; passing is necessary but insufficient for peak-memory fit.
Runtime OOMs are retained as failed experiments; graph policy is not changed.

With fixed quotas k=q+hp+hs. With one shared hidden pool k=q+h, even when both
the preload and stale-writeback limits equal h. Hidden roles change over time;
they are not two separately allocated h-frame pools.

One all-resident state would have effective visible count 2, not 4 or 10
physical frames. On the checked A6000, entity state plus the full mapped TW
graph alone needs about 52.91 GiB, exceeding the measured 47.40 GiB device.
The p=2 all-resident case is therefore recorded as payload-infeasible under
the unchanged whole-state GPU graph policy; it is not launched.

## Execution Order

1. Matched q=4, k=7 shared-hidden runs at p=16, 11, 10, 8.
2. q=4 shared-pool sweep for every k=4,...,10 using the c=20 GiB p table.
3. q=4 p=16 fixed-quota controls: k=7 with 1+2 and 2+1 hidden splits;
   k=10 with 3+3.
4. Remaining q=4 fixed-quota c=20 GiB sweep. Odd hidden counts test both splits.

Duplicates are removed: 21 candidate cases, each five epochs, batch 50,000,
evaluation disabled. The full dataset is physically repartitioned for each p
and hash-verified. Each candidate first passes a tiny full-training gate and
a GPU value/freshness test, using the same pulled build. Minimum-state and
maximum-overlap certificates are checked before execution. No claim is made
that every candidate will finish within one allocation.

The supervisor runs as a detached srun step inside allocation 281316, stops
starting cases near its deadline, and records unfinished cases explicitly.
It survives the launching SSH connection, but cannot outlive allocation
expiration or cancellation (including release by an interactive salloc owner).
Only GPU 1 is selected. Other-user processes on GPU 0 are not touched; timings
are labeled shared-node, not uncontended-node measurements.

## Planning Other Visible Counts

Run tools/tw_visible_frame_plan.py to generate q=2,3,4 tables for arbitrary
frame caps. At fixed k,c, p often does not change with q; visible versus hidden
capacity and the required cover do. At fixed hidden count h, smaller q reduces
k=q+h and can reduce the frame-derived p. Neither guarantees faster training.
Some listed hidden capacities can exceed what one lookahead transition uses;
the table describes memory candidates, not proven useful pipeline depths.
