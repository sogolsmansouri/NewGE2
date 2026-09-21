# PipeGE protocol decision, 2026-09-21

The user requested removal of the negative-softmax mass multiplier. All new
training uses unweighted negative mass (scale 1, log bias 0). The engine rejects
obsolete nonunit flags in both autograd loss and explicit score gradients.
The corrected/fused gradient kernels and runtime optimizations remain enabled.

TW reports **tail-only** filtered ranking on the same pinned 10,000-query subset
as the GE2 TW reproduction. The evaluator may retain both sets of ranks for
audit, but the primary MRR and hits use only the 10,000 tail ranks. LJ, FB and
WK retain their existing head-and-tail protocol. WK remains public validation,
not the unavailable hidden test set. Do not change direction based on accuracy.

Existing FB/WK mass-8 checkpoints are historical tuned-objective results, not
unweighted baselines. They require retraining, not just reevaluation. Preserve
their artifacts unchanged. Existing TW mass-1 checkpoints can be reported
tail-only directly from hash-verified saved ranks; label this a derived report,
not a new GPU scoring run. The eight-batch negative *planning* setting is unrelated
to the removed loss multiplier and remains unchanged.

Use a new frozen campaign, commit, binary hashes, native gradient parity gates,
two-epoch update checks, full training, and evaluation. Do not overwrite old
campaign manifests. Other sampling differences (including degree-chunk
exclusion) remain and still need protocol review before a pipeline-only claim.

When the user explicitly chooses accuracy work on a shared node, launch with
`--allow-shared-node --gpu <physical UUID>`. Monitor the selected GPU and stop
only our child process group upon contention. Preserve whole-node hardware
and job records, and label timings provisional even if accuracy passes.
Remeasure timing later on an exclusive node. This mode never silently replaces
the default isolated-run checks.
