# GE2 Freebase reproduction: repartition control

This is an accuracy diagnostic, not a new final-paper performance measurement.
The Zenodo technical report describes node repartitioning between epochs; the
released `GraphModelStorage::load` leaves `rePartition()` commented out. A separate
driver invokes that original method after epoch flushing and before the next
epoch. The original library, gradients, optimizer, and scoring remain unchanged.
The experiment must be labeled as a paper-described variant, not the unchanged
released execution path. A missing feature is not yet a proven cause of the gap.

| Model | Control | Changed factor | Epochs | Evaluation |
|---|---|---|---:|---|
| DistMult | fixed | None | 10 | Fixed 10K validation, both directions |
| DistMult | repartition | Fresh node partition membership before epochs 2-10 | 10 | Same validation queries |
| ComplEx | fixed | None | 10 | Same validation queries |
| ComplEx | repartition | Fresh node partition membership before epochs 2-10 | 10 | Same validation queries |

Constants: FB86M, 304,727,650 training triples, p=16/q=4, real width 100,
batch 50K, 50 chunks, 1,000 negatives, degree fraction 0.5, SUM softmax loss,
Adagrad 0.1 for entities and relations, inverse relations, one GPU. Additional
RNG consumption by repartitioning means subsequent batches and samples need not
be identical even though both runs start from the same seed.

Validation seed: `ge2-fb-optimizer-control-validation-20260920:v1`.
Query SHA256: `1ace54b772ccd59818befff780fd138f19a2f79d2b4a34f2237e6eb8dfd114cd`.
The panel must be in validation and absent from train/test. All-candidate,
filtered, pessimistic-tie head/tail ranks are retained. No test-score tuning.
The paper's test metrics are not directly compared to validation metrics.

## Gates

- The 803-file Zenodo archive stays untouched; drivers compile separately
  against the installed original library. The archive MD5 and library SHA256
  are checked again on ARC.
- Native three-epoch fixtures audit gradients, updates, entity/state mapping,
  canonical positive-edge coverage, and epoch flush across changing partitions.
- Fixed-mode driver checkpoints must agree with the ordinary GE2 entry point
  on independent DistMult and ComplEx fixtures within FP32 tolerances.
- The known uneven-partition negative-padding defect remains present and
  explicitly reported. Strict fixture status remains false for affected cases;
  the campaign only tolerates this specific failure, not arbitrary audit errors.
- Owned allocation, idle GPU, dataset hashes, checkpoint sizes, exact evaluator
  hashes and native score checks are required. No learning-policy change is
  hidden in the harness.

The existing epoch timer excludes repartitioning; the driver reports its duration
separately. These timings must not replace final timing-table entries.
Large checkpoints stay on node-local storage with manifests and are **not**
declared durably archived. Small configs, logs, ranks and metrics persist in ARC
home. Slurm submission survives the client disconnecting, but does not survive
the allocation expiring or a node failure.

## Current evidence

Uniform negatives already worsened FB accuracy; dense Adam did not consistently
improve validation accuracy. Native/external full-candidate ranks agreed on
predeclared 256-query panels. This control tests a remaining structural mismatch,
not a demonstrated gradient defect. Further stages, if needed, are initialization
and negative-sampling convention checks, controlled seeds, and the exact data
permutation/split recipe. They are not silently swept for the best test score.
