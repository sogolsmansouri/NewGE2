Twitter/Dot transfer repair and two-epoch diagnostic record, 2026-10-15

The accompanying source changes add startup transfer-integrity checks, an explicit
CPU-staged initial model broadcast when forced host fallback is selected, host
handoff payload validation, and all-lane preload ordering. They do not repair a
host PCIe/IOMMU problem. Direct P2P remains the default when the force-host flag
is unset, but startup now rejects incorrect payloads. Forced host mode supports
the tested two-column Twitter/Dot path; unvalidated dense NCCL collectives are
rejected. No FB/ComplEx collective correctness claim is made.

configs/ preserves the exact accepted two-epoch workload and paths from this
machine; 50,000 edges PER GPU, h=3, eight Torch intra-op threads, no checkpoints.
The two-GPU replacement changed the controller guard from90GB to95GB after a
controlled stop during reset. It did not change training semantics. GPU0/GPU1
mean logical CUDA_VISIBLE_DEVICES indices, not arbitrary host indices.

controllers/ contains the exact archival controllers used for this experiment,
not portable launchers. They reference the original /workspace artifact tree,
old frozen helper, data receipts and source hashes. Committing this checkout
changes Git HEAD and invalidates their original preflight gate by design.
Do not weaken those checks to rerun them; prepare a separately bound run for a
new commit/protocol. snapshot_manifest.json identifies original files by hash.
Full raw logs/receipts stay in the named workspace and evidence archives.

Validation: Release build passed, two native CPU tests passed, model host-transfer
and expected-direct-rejection tests passed, and two two-epoch fixtures passed.
The validating fixture checked72 host payloads exactly. Production timing did
not enable full payload auditing. All six C++/CMake changes were hash-matched to
the tested build before commit; archiving records did not alter their contents.

Epoch1/epoch2:1GPU209.361/208.861s,2GPU122.283/118.523s. Epoch2 ratio1.762198x.
Full-process walls642.184/528.871s (1.214256x); native epoch timers exclude
initialization/reset/finalization. Each2GPU epoch used36host handoffs/37.487GB,
zeropeer copies, zero descriptor mismatches and zero preload misses. The first
2GPU attempt stopped at its90GB guard; the accepted retry peaked91.375GB with
noOOM. Compared with older1bcaa5c,2GPU is only0.722% faster and1GPU8.075% slower;
the larger scaling ratio is not a large absolute performance improvement.

These measurements are diagnostic: unresolved NVML PID namespace means GPU
exclusivity is unconfirmed, only one warmed epoch is available, no accuracy
or equal-quality claim, and no per-edge execution ledger. See comparison and
performance records for details. The historical ARC table is not a matched run.

The native CUDA diagnostic reproduces failure without Torch:32/32 enabled-peer
copy cases and2/2 remote-read plus2/2 remote-write cases fail with successful
API returns, while controls pass. Both GPU IOMMU groups areDMA-FQ (translated),
a strong suspected host cause per NVIDIA documentation. No host setting was
changed, so causality and a repaired P2P path have not been demonstrated.
Compile/run instructions for the independent probe are in diagnostics/.

This archive records the completed TWO-epoch comparison. Subsequent ten-epoch
accuracy work is a separately authorized experiment, with separate receipts;
it must not be mixed into these timings or presented as already validated.
