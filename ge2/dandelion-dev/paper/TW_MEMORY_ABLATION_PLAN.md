# TW memory and pipeline ablation plan

Date: 2026-09-14. Status: proposed; no ARC training launched for this plan.
The runtime changes in this checkpoint are based on
`57c73b5e0a895eb74db8fb22df300e34cae00b2f`, on branch
`codex/tw-fixed-memory-ablation-20260913`. This is an experimental checkpoint,
not a declaration that every runtime mode is validated.

## Questions and controls

1. With identical code, p, q, schedule, and total frames, does a shared hidden
   pool differ from fixed preload/writeback quotas?
2. At fixed partition count and frame memory, does moving one frame from
   visible residency to hidden capacity improve epoch time?
3. On an A6000, do configurations that exhausted memory on a 3090 become
   feasible, and how do fewer partitions trade off against more hidden frames?

Fixed controls: TW, single GPU, batch 50,000, embedding dimension 100,
float32, Adagrad, same negative-sampling settings and seed, whole-state mapped
GPU graphs, graph prefetch enabled, event-scoped synchronization. Five epochs
per accepted configuration; report all five, average, and mean of epochs 2-5.
No evaluation in this timing study, and no claim of equal final quality.

Use the same full-edge workload as the local ablation: 41,652,230 entities and
1,468,345,182 training edges. The p=16 training-file SHA256 is
`d978663192467e9da277fadd5d374f28076a7a07f46a849ae907cff7902ccd95`.
ARC's cached controlled 90/5/5 dataset has 1,321,528,663 training edges and
must not silently replace this workload. Different p requires actual
repartitioning of the same logical edge multiset and verified bucket counts.

## Memory contract

All capacities below are GiB (2^30 bytes), not decimal GB.

| Quantity | Value | Meaning |
|---|---:|---|
| A6000 nominal VRAM | 48 | Product capacity, not a parameter-frame budget |
| CUDA-reported capacity C on c31 GPU 1 | 47.401794 | 50,897,289,216 bytes, queried in allocation 281316 |
| Local 3090 CUDA-reported capacity | about 23.552 | Earlier measurement, not assumed to equal nominal 24 |
| Complete entity and optimizer storage W | 31.033330 | 41,652,230 * 100 * 4 * 2 bytes |
| One frame at p=16 | 1.939584 | Embeddings and Adagrad state combined |
| Four visible frames at p=16 | 7.758334 | Excludes hidden frames, graphs, workspace |
| Seven total frames at p=16 | 13.577085 | q=4/shared3 or q=3/fixed2+2 |
| Ten total frames at p=16 | 19.395836 | q=4/fixed3+3 |

For n entities, d embedding components, b bytes/component, and one Adagrad
accumulator per component:

    W = n * d * b * 2
    F(p) = ceil(n / p) * d * b * 2

For fixed quotas:

    k = q + hp + hs

For a shared hidden pool of capacity h:

    k = q + h
    incoming_or_ready(t) + stale(t) + free_hidden(t) = h

The shared equation describes non-visible roles outside an in-progress
publication. Physical-frame identities rotate as frames become visible and
old visible frames retire. A shared3 pool is not a permanent 1+2 or 2+1 split:
it can hold three incoming frames, then three outgoing stale frames after
publication. It cannot hold three incoming plus three stale frames at once.
Matching pending writebacks and last-update events still constrain reuse.

Given an explicit parameter-frame cap c:

    p0 = max(q, ceil(k * W / c))
    require k * F(p) <= c after padding

This is necessary but insufficient for total-memory feasibility:

    k * F(p) + max_t [graph(t) + workspace(t) + runtime_overhead(t)] + margin <= C

Equivalently, the safe frame cap depends on the plan:

    c <= C - nonframe_peak(p, q, policy, batch) - margin

Neither 20 GiB for a nominal 24 GiB GPU nor 42 GiB for a nominal 48 GiB GPU
is established as a universally safe cap. The 20 GiB value in Stage B is a
deliberate experimental cap, chosen to revisit the previous sweep unchanged.
It leaves at least 27.402 GiB of the A6000 capacity outside the configured
frame cap, before graphs and all other allocations consume that headroom.

Graph payload must use real bucket counts. For resident state S_t:

    G_t = 16 * sum(bucket_edges[i,j] for i,j in S_t)

This implementation materializes all resident buckets, not only those assigned
to train in this state. Check both one-state payload and current-plus-next
graph payload. At p=16 the measured-schedule largest adjacent graph pair is
2.810369 GiB for q=4 and 1.593416 GiB for q=3. These are payloads, not complete
non-frame peaks. Local sampled usage outside frames includes allocator cache;
it is not a portable estimate of irreducible workspace on another GPU.

## Stage A: isolate the allocation policy first

All rows use p=16 and the same committed runtime rebuilt for ARC. A1-A3 use
the identical 20-state q=4 schedule. A4 uses the certified 43-state q=3
schedule, so it changes more than pipeline capacity. A5 also uses the q=4
schedule but increases total frame memory. No speedup is assumed in advance.

| ID | q | Hidden policy | k | Frame payload (GiB) | Question |
|---|---:|---|---:|---:|---|
| A1 | 4 | 3 shared | 7 | 13.577 | Shared-pool reference on the committed build |
| A2 | 4 | fixed hp=1, hs=2 | 7 | 13.577 | Fixed quota versus A1 |
| A3 | 4 | fixed hp=2, hs=1 | 7 | 13.577 | Reverse fixed quota versus A1 |
| A4 | 3 | fixed hp=2, hs=2 | 7 | 13.577 | Visible-versus-hidden tradeoff, same frame memory |
| A5 | 4 | fixed hp=3, hs=3 | 10 | 19.396 | Does extra capacity help beyond a shared pool? |

A1-A3 are the first priority. Do not interpret A4 or A5 as pipeline-only
causal measurements. Five consecutive epochs are not five independent trials;
confirmation runs with controlled order and an isolated node are needed for
publication-grade variance and uncontended timing claims.

## Stage B: repeat the original frame-cap sweep on A6000

Keep q=4 and c=20 GiB as an explicit frame cap. Both splits in a row are
separate five-epoch cases. Preload/stale capacities may be smaller than the
maximum admissions at a boundary; uncovered movement remains exposed instead
of allocating additional unbudgeted GPU frames.

| k | hp/hs | p0 | Frame size (GiB) | Total frames (GiB) | Certified states | Maximum total overlap |
|---:|---|---:|---:|---:|---:|---:|
| 4 | 0/0 | 7 | 4.4333 | 17.7333 | 5 | 10 |
| 5 | 0/1 and 1/0 | 8 | 3.8792 | 19.3958 | 6 | 10 |
| 6 | 1/1 | 10 | 3.1033 | 18.6200 | 9 | 16 |
| 7 | 1/2 and 2/1 | 11 | 2.8212 | 19.7485 | 11 | 20 |
| 8 | 2/2 | 13 | 2.3872 | 19.0974 | 13 | 12 |
| 9 | 2/3 and 3/2 | 14 | 2.2167 | 19.9500 | 18 | 34 |
| 10 | 3/3 | 16 | 1.9396 | 19.3958 | 20 | 19 |

The archived certificates establish minimum state count, then maximum total
overlap over distinct full-pair covers of that count and their path orders;
they do not establish minimum runtime. Epoch wraparound is not counted in
the overlap objective. Validate and freeze each schedule before use.
There are ten Stage B cases; its p=16,3/3 case duplicates A5 and should reuse
that result when all controls match. Thus A plus B requests 14 distinct
five-epoch cases (70 epochs), not 15 cases.

Only after this sweep and non-frame-memory calibration should we propose a
larger c that exploits more A6000 memory. It would produce smaller p through
the same formula, but larger resident graphs can invalidate the resulting
plan. Do not choose 42 GiB automatically or promise that every formula row fits.

## Versioning, readiness and execution gates

- Commit the pending runtime/source changes and this plan without binaries or
  generated model data. Leave the stable base branch unchanged. Record a new
  source-tree manifest for the commit and the separate ARC build hashes.
- Preserve archived local run provenance; the new commit does not retroactively
  change which binary an older run used.
- The harness currently lives outside this Git repository under
  `tacc_baselines/tools`. Freeze its exact files and hashes for ARC. Its legacy
  base-commit check must be updated to the approved experiment commit rather
  than bypassed. The commit alone is not a turnkey ARC launcher.
- The local ELF library requires GLIBC symbols newer than c31's GLIBC 2.28;
  rebuild the committed source on ARC instead of copying the local binary.
  The node-local environment has Python 3.9, Torch 2.1.2+cu121 and CUDA 12.1.
- Use node-local environment, source and data. The old BeeGFS paths are absent
  on c31. Stage and hash the same full-edge TW data; do not substitute the
  smaller cached training split.
- Before timed runs, perform a relocated-library/import check and GPU value/
  allocation checks for each quota policy, including odd capacities, partial
  partitions, repeated epochs and final host freshness. These ARC checks have
  not yet run for this checkpoint.
- The parameter-pipeline-off control with allocated hidden frames failed its
  previous behavior gate (preload/publication events were observed). Clarify
  boundary-only versus background admission and validate the intended off
  semantics before adding that factorial experiment. It is excluded here.
- Eleven host-side harness unit tests and `git diff --check` passed on
  2026-09-14. They do not replace the GPU correctness checks. Do not disturb
  an ongoing timing run by launching another CUDA test on its GPU.
- Allocation 281316 grants c31 through 2026-09-14 09:20:58 Eastern. GPUs 1-3
  were idle when checked; GPU 0 runs another user's process. Do not touch it.
  Run our timing cases serially on a selected idle GPU, archive node-wide
  occupancy, and label the shared-node results as such. GPU isolation does
  not guarantee uncontended CPU memory bandwidth or storage.
- Prefer a Slurm job step within the existing allocation, with a detached
  supervisor and durable status under HOME. No step may outlive the allocation;
  a detached shell does not extend the six-hour limit. Stop before starting a
  case that cannot fit the remaining allocation time. Retain incomplete logs,
  and never count a partial run as a complete five-epoch measurement.
- Record per-epoch native time, separately measured epoch wall time, exposed
  transition time, background H2D/D2H and graph work, transferred bytes, peak
  allocated/reserved memory, frame counts, edge coverage and foreign processes.
  Overlapping operation durations must not be summed into an epoch total.
- Profile selected states with Nsight in separate diagnostic runs, not inside
  the primary timing rows. No multi-GPU P2P is involved in these single-GPU tests.

## Existing local reference, not an ARC result

The archived q=4, p=16, shared3, batch50K reproduction completed on 2026-09-14
with epochs 184.662, 183.945, 184.613 seconds: average 184.406667 s, steady
184.279 s. Its archived train/lib SHA256 prefixes are e7ec590c31f2 and
f1b2d908ea70. This reproduces the earlier approximately 184-second observation
but does not by itself validate the newer fixed-quota build's shared mode.
That separate local run was already authorized and running before this ARC
planning pause. No new ARC ablation training is authorized by this document.
