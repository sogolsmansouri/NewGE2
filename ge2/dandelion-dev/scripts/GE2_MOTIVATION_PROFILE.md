# GE2 motivation measurement

This diagnostic supplements, and does not replace, clean training + accuracy
measurements. It makes no PipeGE runtime or quality change.

## Accounting contract

- Partition the main training thread's native epoch wall time into disjoint
  sampling, shuffling, mapping, model/update, movement/lookup, and other ranges.
- A host range includes framework work and waits for CUDA. It is not a kernel
  service timer. In particular, model/update wall time is not pure GPU compute.
- Use the union of CUDA kernel, memcpy, and memset intervals to measure device
  activity. Count overlaps once. The complement is **no recorded CUDA work**.
- Intersect that complement with host phases. GPU inactivity is an overlay on
  host work, not an extra additive slice beside that work.
- Report memcpy directions and payload bytes separately. Summed copy service
  time and summed background-worker time are not epoch wall time.
- Preserve nested operation counters for drill-down, but do not sum all nested
  counters. `operation_counters.csv` explicitly labels them as inclusive.

## Full-epoch jobs

On c30 at 300 W, profile TW Dot and FB ComplEx with p=16, q=4, batch 50K and
the saved config from the corresponding clean GE2 parent. Seed, data, optimizer,
negative sampling, and prefetch settings remain the same. Only run length,
output paths, checkpoint/evaluation disabling, and instrumentation differ.

Each job executes three fresh-model passes sequentially on GPU 0:

1. Three uninstrumented epochs using the installed original GE2 binary.
2. Three epochs with host counters, without NVTX capture.
3. Two epochs with Nsight, capturing the whole second native epoch plus a
   separately labeled post-timer finalization/drain interval.

Compare epoch 2 across the passes to expose instrumentation overhead. Save all
epoch times, counters, NVTX/CUDA trace, full configs, data hashes, build hashes,
allocation evidence, and GPU/node isolation samples. Do not rescale profiled
phases to the uninstrumented time. These short runs have no accuracy result;
accuracy is associated with the full clean parent and its saved checkpoint.

The capture adds no per-operation CUDA synchronization. One diagnostic drain
after the native timer closes outstanding device work before ending capture.
The main-thread and device panels use the same native-epoch bounds; report
post-timer work separately rather than hiding it in the residual.

`prepare_ge2_epoch_profile.py` applies a checksummed archived annotation patch
to the pinned Zenodo ZIP, corrects the diagnostic epoch label, and adds capture
ranges. The only dependency-build change skips Git submodule initialization
when the dependency headers are already bundled in the ZIP. The existing build
helper records its CUDA/PyTorch compatibility diff. Never install this build
over the clean runtime.

## Validation and submission

Run `test_ge2_motivation.py` and `smoke_ge2_epoch_profile.py` before submission.
The smoke uses a tiny synthetic graph and validates capture plumbing only.
`stage_ge2_epoch_profile.py` freezes files and references the clean queue ledger.
`submit_ge2_epoch_profile.py` submits serial exclusive jobs after that queue,
with successful-clean-parent dependencies. It refuses duplicate ledgers.

A missing native-epoch marker, missing CUDA kernels, missing finalization marker,
overlapping host phase classifications, changed input, incomplete edge work,
or explicit trace loss rejects the corresponding analysis.

## Historical evidence

The legacy RTX 3090 TW counter run processes 1,468,345,182 edges, not the
1,321,528,663-edge training split of the matched ARC campaign. Its full-epoch
host counters and bounded Nsight cycle are useful preliminary diagnostics but
must not be presented as a whole-epoch A6000 breakdown. The analyzer labels
bounded-cycle input explicitly and will not infer full-epoch idle from it.

The preliminary figure supports investigating preparation, mapping, movement,
and synchronization as well as arithmetic. It does not, on its own, establish
how much of PipeGE's speedup is caused by pipelining. That requires matched
on/off ablations with equal positive-edge work and training semantics.
