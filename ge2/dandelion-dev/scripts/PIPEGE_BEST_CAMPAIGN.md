# ARC fast-policy replay, 2026-09-20

Historical campaign description. New runs follow
[the 2026-09-21 unweighted protocol](PIPEGE_UNWEIGHTED_PROTOCOL.md): mass 1 for
every dataset and tail-only TW reporting. Existing mass-8 results below are
preserved as tuned-objective measurements, not reused as unweighted results.

Six independent single-GPU RTX A6000 runs, serial and node-exclusive. The
new c30 allocation 284846 can run them while GE2 continues independently on
c31. c30 is configured at 300 W versus c31 at 200 W; record these hardware
differences and do not merge their timings into an unlabeled comparison.
Live inspection found another user's CPU allocation on c30 with no GPU process.
The interactive run therefore provides quality measurements and provisional
shared-node timings, not a claim of exclusive-node paper timing. Batch
continuations request exclusive allocation; all results retain isolation logs.
Each case first runs a two-epoch masked-update correctness gate, then trains
from scratch for the final epoch count and saves entity, optimizer, and
relation checkpoints. Only the final run is timed for the result table.

| Case | Partitions | Visible | Shared hidden | States | Final epochs | Graph prefetch |
|---|---:|---:|---:|---:|---:|---|
| LJ Dot | 2 | 2 | 0 | 1 | 30 | off |
| TW Dot | 16 | 4 | 3 | 20 | 10 | on |
| FB DistMult | 32 | 4 | 6 | 88 | 10 | off |
| FB ComplEx | 32 | 4 | 6 | 88 | 10 | off |
| WK DistMult | 30 | 4 | 6 | 75 | 10 | off |
| WK ComplEx | 30 | 4 | 6 | 75 | 10 | off |

All rows use batch 50K, 1,000 negatives, 50 chunks, degree fraction 0.5,
and Adagrad sparse updates at 0.1. Width is 100 except WK at 80 real values
for both decoders. Snapshots retain their archived seeds, initializers,
dense optimizers, and runtime flags. LJ/TW use negative-plan reuse of eight
batches and loss negative mass 1; FB/WK disable that reuse and use mass 8.
These are not all the baseline training semantics. Quality must be measured,
not copied from a slower recipe or an older buggy-gradient checkpoint.

WK is explicitly an adaptation: its bias-free GE2-style Adagrad recipe is
combined with the archived fast runtime. The old fastest WK YAML enabled
encoder bias and dense Adam; the old manual path silently bypassed that bias.
Replaying that bug to recover a timing is not acceptable. The repaired manual
path rejects non-identity encoders. No claim of exact old-WK reproduction is made.

TW uses the audited 1,321,528,663-edge training split, not the historical full
1.468B-edge timing-only workload. FB and WK are physically repartitioned from
the audited splits into new private p32/p30 views before training. Query IDs
stay in the original entity domain. Sources and filters are never overwritten.

Exact evaluation uses 10,000 pinned queries, both head and tail directions,
all entities as candidates, train/validation/test filtering, pessimistic ties,
and TF32 disabled. WK queries are drawn from public validation because the
official test labels are unavailable; they are not hidden-test paper reproduction.

The repaired engine is built at fa783f4b9ef685fd5bd969d3426ddce3ee324721.
The latest committed campaign code reuses that binary only after proving
the engine source tree is identical, checking binary/library hashes and ldd,
and checking the saved paired-gradient gate. Preparation reruns native
manual/autograd parity at widths 100/80 and masses 1/8. A future engine change
fails closed and requires a fresh build, rather than relabeling an old binary.

`stage_pipege_best_campaign.py` freezes configurations, original flag provenance,
helper code, schedules, and evaluation contracts. `run_arc_pipege_best.py`
implements preparation and one case per Slurm job. `arc_pipege_best.sbatch`
uses normal batch execution, survives client/VPN disconnects, and remains
subject to its allocation deadline. `supervise_arc_pipege_best.py` also runs
detached inside an existing allocation; that mode cannot outlive cancellation
of the allocation itself. Queue continuation jobs after that allocation to
retry unfinished cases and skip already completed, evaluated results.
Preparation failure prevents all training; a case failure does not suppress
later independent cases. Dataset copies on c30 use a private campaign cache;
existing data and the active c31 GE2 campaign are not modified.

Small evidence files live in ARC home. Large checkpoints are retained on the compute node's
local disk and are explicitly marked not durably archived. Completed results
remain paper candidates until protocol, power-limit, and isolation review.

Relocated private dataset copies have their on-disk `dataset.yaml` path
normalized, not only the path in the generated config. Each training launch
first passes the actual GEGE config loader with `save=False` and checks the
resolved dataset, partition count, visible capacity and batch size. Data
counts and other metadata cannot change during relocation.

For a driver-only repair, `handoff_arc_pipege_best.py` pauses the old serial
dispatcher but lets its active training/evaluation child finish. It then
starts a fresh supervisor and preparation workspace. Completed results from
an explicitly allowed older commit can be reused only when frozen inputs
and the engine source tree match; their original result provenance is kept.
The engine, datasets and running checkpoint are not patched in place.
