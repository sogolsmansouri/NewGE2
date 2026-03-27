# Main Opt Flag Tables And Ablation Ladder

This file converts the current `main`-branch flag audit into tables and a concrete ablation plan.

Scope:
- Codebase: `main`
- Goal: isolate which `main`-only optimizations actually reduce epoch time
- Measurement target: training `Epoch Runtime`
- Important rule: start from an explicit-off baseline, then turn on one thing at a time

## Core Flags And Knobs

| Flag / knob | Type | Default if unset | Current role | Single GPU evidence | 2 GPU evidence | 4 GPU evidence | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` | env | `0` | Needed, single-GPU only | observed in Freebase train log | not applicable | not applicable | Only matters when `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0` or does not take over |
| `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS` | env | `0` | Needed | observed in Freebase train log | no observed run | no observed run | Logged as `keep_storage_hot=1` at epoch boundary |
| `GEGE_OPTIMIZED_CUSTOM_SCHEDULE` | env | `1` | Implicit | default-on in code, explicit `=0` repro in ablation note | no completed observed run | no completed observed run | Single-GPU alternative to `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM`; do not treat them as additive |
| `GEGE_PARTITION_BUFFER_LP_FAST_PATH` | env | auto-on under LP + in-memory + unfiltered negatives | Implicit | implicit in opt path | implicit if config is run | implicit if config is run | To ablate it, force `0`; otherwise it turns on automatically in the standard LP path |
| `GEGE_FAST_MAP_TENSORS` | env | `1` | Implicit | part of single-GPU best path note | implicit if config is run | implicit if config is run | Default-on |
| `GEGE_GPU_ACTIVE_EDGE_SHUFFLE` | env | `1` | Implicit, strong candidate | part of single-GPU best path note | implicit if config is run | implicit if config is run | Best single-GPU win in current Freebase note |
| `GEGE_CSR_GATHER` | env | `1` | Implicit, but current best path keeps it off | best single-GPU path uses `0` | no completed observed run | logical-4 sim uses `0` | Treat as a regression probe, not as an expected win |
| `GEGE_CSR_UPDATE` | env | `1` | Implicit, but current best path keeps it off | best single-GPU path uses `0` | no completed observed run | no completed observed run | Treat as a regression probe, not as an expected win |
| `GEGE_CSR_DEBUG` | env | `1` | Fixed control | best runs force `0` | should force `0` | logical-4 sim forces `0` | This is debug overhead, not a speed feature; keep it off in all timing experiments |
| `GEGE_DEG_CHUNK_EXCLUSION` | env | `1` | Implicit | implicit in opt path | implicit if config is run | implicit if config is run | Default-on |
| `GEGE_GLOBAL_DEGREE_SAMPLING` | env | `0` | Optional probe, off by default | LJ A/B run shows similar MRR/Hits but faster epoch time | not evidenced | not evidenced | `unset` keeps the old batch-local DegreeNS path; `=1` enables the new global degree-weighted negative sampler |
| `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST` | env | `1` | Implicit | single-GPU ablatable | code forces pinned host on multi-GPU MEM buffer | code forces pinned host on multi-GPU MEM buffer | Do not spend time ablating this on real multi-GPU runs |
| `GEGE_PARTITION_BUFFER_PEER_RELAY` | env | `0` | Needed for relay experiments | not applicable | wired, no completed observed 2-GPU train log | observed in Twitter relay note | Requires real multi-GPU CUDA peer access and CUSTOM ordering |
| `GEGE_UNIQUE_BACKEND` | env | `auto` | Mode-specific | bitmap used in dot-emulation runs | no completed observed run | bitmap used in Twitter relay note; sort used in logical-4 sim | Keep fixed during epoch-time ablations unless unique itself is the thing being tested |
| `GEGE_UNIQUE_BITMAP_NUM_NODES` | env | unset | Mode-specific | used with bitmap backend | no completed observed run | used with bitmap backend | Only matters when `GEGE_UNIQUE_BACKEND=bitmap` |
| `dense_sync_batches` | YAML | `1` | Config knob | default `1` | config explicitly sets `1` | config explicitly sets `1` | Separate sweep; not a `GEGE_*` env flag |
| `logical_active_devices` | YAML | `0` | Config knob | logical-4 smoke uses `4` | config keeps `0` | config keeps `0`; logical-4 sim uses `4` on one physical GPU | Real throughput knob only when you intentionally simulate or reorder for logical lanes |
| `GEGE_PROFILE_LOGICAL_LANE` | env | unset | Simulation-only | not used in real single-GPU opt run | not used | logical-4 sim only | Not a real throughput optimization on its own |
| `GEGE_EVAL_CHUNKED_RANKS` | env | `1` | Eval-only implicit | used in eval driver | not relevant to train epoch time | not relevant to train epoch time | Hold fixed outside training epoch ablations |
| `GEGE_EVAL_NEGATIVE_CHUNK_SIZE` | env | `131072` | Eval-only | eval driver used `32768` | not relevant to train epoch time | not relevant to train epoch time | Hold fixed outside training epoch ablations |
| `GEGE_BUCKET_STREAMING_LP` | env | `0` | Explicit probe | ablation note says `1` speeds up but lowers MRR | not evidenced | not evidenced | Keep off for quality-preserving path; probe separately |

## Wired But Not Yet Part Of Current Opt Runs

| Flag / knob | Type | Default if unset | Status | Why it is not in the current ladder |
| --- | --- | --- | --- | --- |
| `GEGE_ACCESS_AWARE_SCHEDULER` | env | `0` | wired, not evidenced | No completed opt run in the workspace uses it |
| `GEGE_ACCESS_AWARE_STATE_GENERATION` | env | `0` | wired, not evidenced | No completed opt run in the workspace uses it |
| `GEGE_NEGATIVE_TOURNAMENT` | env | `0` | wired, not evidenced | Current opt YAMLs keep tournament selection off |
| `GEGE_TILED_TOURNAMENT_SCORES` | env | `0` | wired, not evidenced | Current opt YAMLs keep tiled tournament scoring off |
| `GEGE_TILED_TOURNAMENT_GROUPS_PER_TILE` | env | `64` when env is used | wired, not evidenced | Only matters if tiled tournament scoring is enabled |
| `training.negative_sampling.superbatch_negative_plan_batches` | YAML | `0` | wired, not evidenced | Current opt YAMLs keep it at `0` |
| `training.negative_sampling.tournament_selection` | YAML | `false` | wired, not evidenced | Current opt YAMLs keep it `false` |
| `training.negative_sampling.tiled_tournament_scores` | YAML | `false` | wired, not evidenced | Current opt YAMLs keep it `false` |
| `training.negative_sampling.tiled_tournament_groups_per_tile` | YAML | `64` | wired, not evidenced | Current opt YAMLs keep the default |
| `GEGE_SELECTED_NEG_CUDA` / `GEGE_DISTMULT_SELECTED_CUDA` | env | `1` | wired, not evidenced in current RNS configs | Current opt YAMLs use `negative_sampling_selected_ratio: 1.0`, so this is not the main experiment axis |

## Explicit-Off Baseline

Use this as the clean starting point for timing ablations. It disables the flags that are otherwise active by default in the opt path.

Keep these fixed for every timing run:

| Setting | Value | Why |
| --- | --- | --- |
| `GEGE_CSR_DEBUG` | `0` | Remove debug overhead from all timing runs |
| `GEGE_BUCKET_STREAMING_LP` | `0` | Keep the quality-preserving path fixed unless testing the speed/quality tradeoff explicitly |
| `GEGE_EVAL_CHUNKED_RANKS` | fixed | Eval-only; do not mix eval tuning into train epoch timing |
| `GEGE_EVAL_NEGATIVE_CHUNK_SIZE` | fixed | Eval-only; do not mix eval tuning into train epoch timing |

### Single-GPU explicit-off baseline

```bash
export CUDA_VISIBLE_DEVICES=0

export GEGE_CSR_DEBUG=0
export GEGE_BUCKET_STREAMING_LP=0

export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=0
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=0

export GEGE_PARTITION_BUFFER_LP_FAST_PATH=0
export GEGE_FAST_MAP_TENSORS=0
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=0
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_DEG_CHUNK_EXCLUSION=0
export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=0
```

### DegreeNS Mode Switch

Use this only for the separate DegreeNS A/B probe. It is not part of the default opt ladder.

```bash
# old/original behavior
unset GEGE_GLOBAL_DEGREE_SAMPLING

# new global-degree behavior
export GEGE_GLOBAL_DEGREE_SAMPLING=1
```

### Multi-GPU explicit-off baseline

Use the same baseline for 2 or 4 GPUs, but do not bother forcing `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=0`; the code forces pinned host buffers on multi-GPU MEM partition-buffer runs.

```bash
export CUDA_VISIBLE_DEVICES=0,1
# or: export CUDA_VISIBLE_DEVICES=0,1,2,3

export GEGE_CSR_DEBUG=0
export GEGE_BUCKET_STREAMING_LP=0

export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=0
export GEGE_FAST_MAP_TENSORS=0
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=0
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_DEG_CHUNK_EXCLUSION=0
export GEGE_PARTITION_BUFFER_PEER_RELAY=0
```

## Single-GPU Ablation Ladder

Important dependency:
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` and `GEGE_OPTIMIZED_CUSTOM_SCHEDULE` are scheduler alternatives.
- Do not enable both and then interpret the result as the sum of two independent wins.

Recommended order:

| Step | Change from previous step | Expected direction | Why this step exists |
| --- | --- | --- | --- |
| `S0` | explicit-off baseline | reference | Cold baseline |
| `S1` | `GEGE_FAST_MAP_TENSORS=1` | likely positive | Cheap mapping improvement, default-on in code |
| `S2` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | likely positive | Re-enables the arithmetic remap fast path |
| `S3` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | strong positive candidate | Best current single-GPU win in the Freebase ablation note |
| `S4` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | likely positive | Removes repeated unload/reload churn across epochs |
| `S5a` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | likely positive | Tests the explicit single-GPU reorder path with optimized schedule still off |
| `S5b` | reset `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=0`; set `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | likely positive | Tests the default optimized CUSTOM scheduler as an alternative to `S5a` |
| `S6` | `GEGE_DEG_CHUNK_EXCLUSION=1` | small or neutral | Restores the default degree-chunk exclusion behavior |
| `S7` | `GEGE_CSR_GATHER=1` | likely neutral or negative | Regression probe; current best path keeps it off |
| `S8` | `GEGE_CSR_UPDATE=1` | likely neutral or negative | Regression probe; current best path keeps it off |
| `S9` | `GEGE_BUCKET_STREAMING_LP=1` | may improve time, may hurt MRR | Explicit speed/quality tradeoff probe |

Scheduler interpretation:

| Comparison | What it tells you |
| --- | --- |
| `S4 -> S5a` | Is explicit single-GPU GPU-aware CUSTOM helping on top of the non-optimized baseline? |
| `S4 -> S5b` | Is the optimized CUSTOM scheduler better than the cold baseline? |
| `S5a -> S5b` | Which scheduler should be the real single-GPU default for this workload? |

## Multi-GPU Ablation Ladder

Important caveats:
- No completed 2-GPU optimized train log exists in the local workspace.
- Real 4-GPU evidence currently comes from the Twitter relay note, not from a completed Freebase 4-GPU train log.
- `logical_active_devices` is not a normal throughput flag for real 2-GPU or 4-GPU training. It is mainly useful for simulation or logical-lane analysis.

Recommended order:

| Step | Change from previous step | Applies to | Expected direction | Why this step exists |
| --- | --- | --- | --- | --- |
| `M0` | explicit-off baseline | 2 GPU, 4 GPU | reference | Cold multi-GPU baseline |
| `M1` | `GEGE_FAST_MAP_TENSORS=1` | 2 GPU, 4 GPU | likely positive | Restores default-on fast tensor mapping |
| `M2` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | 2 GPU, 4 GPU | likely positive | Restores the arithmetic remap fast path |
| `M3` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | 2 GPU, 4 GPU | likely positive | Restores default-on active-edge shuffle |
| `M4` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | 2 GPU, 4 GPU | likely positive | Re-enables the optimized CUSTOM scheduling path |
| `M5` | `GEGE_DEG_CHUNK_EXCLUSION=1` | 2 GPU, 4 GPU | small or neutral | Restores default degree-chunk exclusion |
| `M6` | `GEGE_PARTITION_BUFFER_PEER_RELAY=1` | 2 GPU, 4 GPU | 4 GPU strongest candidate | Tests direct GPU-to-GPU relay instead of CPU-backed swaps |
| `M7` | YAML sweep: `dense_sync_batches = 2, 4, 8` | 2 GPU, 4 GPU | workload-dependent | Separate synchronization-frequency sweep |
| `M8` | `GEGE_CSR_GATHER=1` | 2 GPU, 4 GPU | likely neutral or negative | Regression probe |
| `M9` | `GEGE_CSR_UPDATE=1` | 2 GPU, 4 GPU | likely neutral or negative | Regression probe |
| `M10` | `GEGE_BUCKET_STREAMING_LP=1` | 2 GPU, 4 GPU | may improve time, may hurt MRR | Explicit speed/quality tradeoff probe |

## Knobs To Keep Out Of The Main Epoch-Time Ladder

| Flag / knob | Keep it out because |
| --- | --- |
| `GEGE_CSR_DEBUG` | Debug tax, not an optimization |
| `GEGE_EVAL_CHUNKED_RANKS` | Eval-only |
| `GEGE_EVAL_NEGATIVE_CHUNK_SIZE` | Eval-only |
| `GEGE_PROFILE_LOGICAL_LANE` | Simulation-only |
| `logical_active_devices` | Mostly for logical-lane simulation or special scheduling studies |
| `GEGE_UNIQUE_BACKEND` / `GEGE_UNIQUE_BITMAP_NUM_NODES` | Change backend behavior; keep fixed unless unique itself is the experiment |
| `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST` on multi-GPU | The code forces pinned host buffers anyway |

## Minimal Run Pattern

Run one step at a time and log the result separately.

```bash
LOGDIR=/tmp/gege_flag_ablation
mkdir -p "$LOGDIR"

# Example single-GPU run
./build/gege/gege_train gege/configs/twitter_16p_paper_opt.yaml \
  |& tee "$LOGDIR/s0.log"

rg -n "Epoch Runtime:" "$LOGDIR/s0.log"
```

For each next step:
- change exactly one flag
- write to a new log file
- record `Epoch Runtime`
- do not change evaluation flags during training timing runs

## Bottom Line

If the goal is to find the real epoch-time wins first, focus on this order:

1. `GEGE_GPU_ACTIVE_EDGE_SHUFFLE`
2. `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS`
3. single-GPU scheduler choice: `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` versus `GEGE_OPTIMIZED_CUSTOM_SCHEDULE`
4. `GEGE_PARTITION_BUFFER_LP_FAST_PATH`
5. `GEGE_FAST_MAP_TENSORS`
6. `GEGE_PARTITION_BUFFER_PEER_RELAY` on real multi-GPU runs

Treat `GEGE_CSR_GATHER` and `GEGE_CSR_UPDATE` as regression probes, not as expected wins, and keep `GEGE_CSR_DEBUG=0` in every timing run.
