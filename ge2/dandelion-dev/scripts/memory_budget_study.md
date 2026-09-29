# Memory-budget study v2

This is a timing-only selection study for TW/Dot and FB/ComplEx. It is not
an accuracy result or proof of a universally optimal planner. Source, binary,
config, data and schedule evidence accompanies each result.

## Contract

- Fix q=4, batch 50K, width 100, Adagrad at 0.1 and the corrected independent
  negative draws with negative-mass multiplier 1. No 8-batch sampling reuse.
- Sweep shared h=0..6, so k=4..10. Stale backlog is limited to min(h,3).
- TW graph prefetch stays on; FB graph prefetch stays off in both arms.
- Budgets: 16, 24, 32, 40 GiB and the A6000's actual 49140 MiB capacity.
- Apply rho=0.9 to the declared budget. Reserve another 512 MiB inside that
  limit for non-Torch allocations; enforce the remainder with the allocator.
  Sample total GPU memory as a second check, not a continuous peak guarantee.
- All selection runs contain 10 epochs. Report average of 1..10 and steady
  mean of 2..10. Two-epoch synthetic acceptance tests are never performance data.

Let W=800 times the entity count, F(p)=800 ceil(nodes/p), and c be the memory
left for frames after graph, dense state, workspace and external reserves.
The initial formula remains p0=max(q,ceil(kW/c)); exact frame rounding and the
full footprint must also pass. The screening estimate for graph memory uses
2x the uniform bucket volume (and current plus next views for TW); workspace
is provisionally 4 GiB. These are assumptions, not measured upper bounds.

The frozen selection list tests the first estimated-feasible integer p and
its lower/upper neighbours for each (workload,budget,h). Four known-geometry
24-GiB controls precede them. There are 214 initial selection cases. With the
current estimates, TW at the full A6000 budget and h=0 starts at p=8; p=7
and p=9 are tested too. This does not assert p=8 is feasible or fastest.

Full-catalog all-resident TW exceeds the largest effective budget even before
workspace: approximately 31.03 GiB entity state plus 19.69 GiB training edges.
FB's entity state alone is approximately 64.12 GiB. Thus neither needs an
all-resident special case in this particular budget range.

## Execution and evidence

`memory_budget_study.py plan` writes the manifest; `smoke` checks all h=0..6
at TW p=16/p=8 and FB p=32/p=8 for bitwise pipeline equivalence, plus an
intentional allocator OOM. `sweep` consumes the frozen plan and records
10-epoch selection cases, OOMs and total-memory failures separately.

Each new physical training view is generated from the canonical split and
verified for bucket order and edge count, with input/output hashes. Entity IDs
do not change. Scratch model files and disposable views are removed only after
the case evidence has been saved; existing datasets and checkpoints are never
deleted. Evaluation is disabled.

Schedules are checked for pair coverage and admission bounds. A generic cover
is not called minimum unless its count meets the lower bound; p=32 uses the
native 88-state path. A later schedule-quality audit may require replacing a
non-tight cover and rerunning the affected comparisons.

`package_memory_budget_study.py` freezes a clean commit and the local acceptance
report. `run_arc_memory_budget.py` clones that bundle, builds natively on c30,
repeats the synthetic gate on the deployed binary, and executes one bounded
six-hour block. It refuses foreign GPU work, checks the 300 W power cap and
monitors Slurm isolation. Completed cases can be skipped on a later allocation;
interrupted or unexpected failures require inspection before retrying.

Training and source/env access stay on node-local storage. Complete logs and
small results are mirrored to BeeGFS; only compact status goes into quota-limited
home. A batch job survives SSH/VPN loss, not allocation expiry. The driver stops
launching new cases near its deadline. One six-hour block does not finish the
214-case search. The three independent 10-epoch winner repeats are a subsequent
phase, not silently included in the selection results.

## Figure

Select best tested no-hidden and shared-pipeline configurations using selection
runs, then freeze winners and use independent repeats for plot values/error
bars. Do not label selection minima as independent measurements. Do not infer
accuracy equivalence across p from the small matched-p gates.

The proposed overlap crossover requires separate tau/beta calibration and the
actual admissions. The q-1-admission bound is conditional; do not draw an
unqualified impossibility line for covers with larger overlaps.
