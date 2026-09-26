# Matched ARC measurements

The September 25 campaign is a separate c30 / RTX A6000 / 300 W cohort.
It must not replace individual timings in the older c31 / 200 W table.

- Submit one preparation job, then GE2 and PipeGE serially for LJ Dot,
  FB ComplEx, TW Dot, FB DistMult, WK DistMult, and WK ComplEx.
- Use exclusive `dkex` allocations, one GPU per measurement, a node-wide
  idle check, and sampled isolation/power evidence. Never change GPU power.
- Freeze source, binary, helper, data, query, configuration, and schedule hashes.
  GE2 uses the released native implementation with explicit Table 4 overrides.
- Both systems use batch 50K, unweighted loss, the same per-workload seed,
  width, training splits and evaluator. Partition schedules differ by design.
- PipeGE: LJ p2/q2; TW p16/q4/shared-3; FB p32/q4/shared-6;
  WK p30/q4/shared-6. These are not dedicated preload/writeback allocations.
- PipeGE runs a two-epoch correctness gate before fresh final training.
  Final training is 30 epochs for LJ and 10 otherwise.
- Report full-catalog filtered tail accuracy over frozen 10K queries,
  pessimistic ties and TF32 disabled. Recompute reported metrics from saved
  tail ranks. WK queries are public validation, not hidden test.
- Save native epoch mean/steady time and available wall intervals separately.
  Profiling and checkpoint archival are outside native epoch timing.
- Keep execution and environments node-local. Copy final checkpoints plus
  evidence to the persistent BeeGFS archive and verify every file by SHA256.
  Retain source checkpoints; small summaries go to home.
- Refuse duplicate submissions using a persistent job ledger. The preparation
  gate must succeed before any training job starts. Later independent cases
  may proceed after a failed case, but its failure remains visible.

These jobs establish one measured run per system/workload, not three-repeat
statistics or equal-quality speedups. Quality gaps must remain visible.
Batch jobs survive client/VPN disconnection but are bounded by their own Slurm
time limits. A timeout is a failed/incomplete measurement, never a result.
