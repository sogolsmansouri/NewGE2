# 2-GPU Ablation Results Template

Use this file to record the 2-GPU comparison matrix for:
- `livejournal_16p`
- `twitter_16p`
- `freebase86m_16p`

Rules:
- Row 1 is always `main` with all flags off.
- Row 2 is always `baseline` with matched config and matched runtime params.
- Then record one-flag-at-a-time rows on `main`.
- Then record incremental rows on `main`, where each row adds one more flag on top of the previous stack.
- Use the same train/eval protocol for `main` and `baseline`.
- Keep `GEGE_CSR_DEBUG=0` for every timing run.
- Average training metrics over epochs `1..30` for `livejournal_16p`.
- Average training metrics over epochs `1..10` for `twitter_16p` and `freebase86m_16p`.
- Measure inter-epoch gap as the time from `Finished training epoch N` to `Starting training epoch N+1`.
- Use `scripts/summarize_benchmark_logs.py` to extract averaged metrics from the train/eval logs.
- Use the standardized 2-GPU configs under `gege/configs/2gpu/` on both branches.
- Do not assume `rg` exists on the target machine; every command in this file should work with standard `grep`.

## How To Run These Experiments

Use this procedure on any machine. Do not assume the YAMLs are ready as-is for a new environment.

### 0. Sync The Checkout And Do Not Assume A Build Directory Already Exists

Before the first run on a machine, update the branch and create a build directory explicitly.

```bash
cd /path/to/dandelion-dev

git switch <main-or-baseline-branch>
git pull --ff-only origin <main-or-baseline-branch>
```

Use a branch-local build directory and configure it if it does not already exist:

```bash
export REPO_ROOT=$PWD
export BUILD_DIR=$REPO_ROOT/build_2gpu_runs

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$BUILD_DIR" -j --target gege_train gege_eval
```

Do not hardcode `build_ge2env`, `build_gcc9`, or any other local build directory name unless that directory actually exists in your checkout.

If configure fails inside vendored `pybind11` with a message about compatibility with `CMake < 3.5` being removed, rerun configure with:

```bash
-DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

If a previous failed configure left a partial build directory behind, remove it first:

```bash
rm -rf "$BUILD_DIR"
```

### 1. Choose Branch, Build Directory, And Dataset Config

Use the same relative config names on both branches:

| Dataset | Config path | Epochs to average |
| --- | --- | ---: |
| `livejournal_16p` | `gege/configs/2gpu/livejournal_16p.yaml` | `30` |
| `twitter_16p` | `gege/configs/2gpu/twitter_16p.yaml` | `10` |
| `freebase86m_16p` | `gege/configs/2gpu/freebase86m_16p.yaml` | `10` |

Before running on a different machine, update these YAML fields for your environment:
- `storage.dataset.dataset_dir`
- `storage.model_dir`
- `evaluation.checkpoint_dir`

### 2. Set Generic Environment Variables

```bash
export REPO_ROOT=/path/to/ge2/repo
export BUILD_DIR=$REPO_ROOT/<build-directory-containing-gege-binaries>
export CONFIG=$REPO_ROOT/gege/configs/2gpu/livejournal_16p.yaml
export LOG_DIR=$REPO_ROOT/experiment_logs
export RUN_NAME=livejournal_16p_main_2gpu_control
export RUN_ROOT=$REPO_ROOT/experiment_outputs/$RUN_NAME
export EPOCHS=30

mkdir -p "$LOG_DIR" "$REPO_ROOT/experiment_outputs"
```

Examples:
- `main` might use `BUILD_DIR=$REPO_ROOT/build_ge2env`
- `baseline` might use `BUILD_DIR=$REPO_ROOT/build_gcc9`
- if neither exists, create one with `cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_POLICY_VERSION_MINIMUM=3.5`

### 3. Reset Flags And Select Two GPUs

```bash
for v in $(compgen -e | grep '^GEGE_'); do unset "$v"; done
export CUDA_VISIBLE_DEVICES=0,1
```

If you want to use a different physical pair, for example GPUs `2` and `3`, do this instead:

```bash
export CUDA_VISIBLE_DEVICES=2,3
```

Keep the YAML `device_ids` at `[0, 1]`. `CUDA_VISIBLE_DEVICES` remaps the selected physical GPUs to logical devices `0` and `1` inside the process.

For `livejournal_16p` and `twitter_16p`, enable paper-faithful social-dot emulation on branches that support it:

```bash
export GEGE_EMULATE_DOT_SINGLE_RELATION=1
```

For `freebase86m_16p`, leave `GEGE_EMULATE_DOT_SINGLE_RELATION` unset.

### 4. Main-Branch All-Flags-Off Control

Use the standardized 2-GPU YAML plus explicit env overrides.

```bash
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

unset GEGE_GLOBAL_DEGREE_SAMPLING
```

For one-flag and incremental rows, change only the flag or YAML knob listed in the table. Keep every other setting fixed.

### 5. Build, Train, Evaluate, And Summarize

```bash
cmake --build "$BUILD_DIR" -j --target gege_train gege_eval

"$BUILD_DIR/gege/gege_train" "$CONFIG" \
  |& tee "$LOG_DIR/${RUN_NAME}_train.log"

"$BUILD_DIR/gege/gege_eval" "$CONFIG" \
  |& tee "$LOG_DIR/${RUN_NAME}_eval.log"

python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/${RUN_NAME}_train.log" \
  --eval-log "$LOG_DIR/${RUN_NAME}_eval.log" \
  --epochs "$EPOCHS"
```

If exact filtered eval on a large graph runs out of GPU memory and your branch supports chunked eval, keep these on the eval command only:

```bash
GEGE_EVAL_CHUNKED_RANKS=1 \
GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768 \
"$BUILD_DIR/gege/gege_eval" "$CONFIG" \
  |& tee "$LOG_DIR/${RUN_NAME}_eval.log"
```

### 5a. Exact Example: `main` LiveJournal One-Flag Run For 15 Epochs

This example runs `livejournal_16p` on `main` with exactly one optimization flag enabled: `GEGE_FAST_MAP_TENSORS=1`.

```bash
cd "$REPO_ROOT"

git switch main

export BUILD_DIR=$REPO_ROOT/build_2gpu_runs
export CONFIG_SRC=$REPO_ROOT/gege/configs/2gpu/livejournal_16p.yaml
export CONFIG_TMP=${TMPDIR:-/tmp}/livejournal_16p_main_2gpu_fast_map_tensors_15ep.yaml
export LOG_DIR=$REPO_ROOT/experiment_logs
export RUN_NAME=livejournal_16p_main_2gpu_fast_map_tensors_15ep
export RUN_ROOT=$REPO_ROOT/experiment_outputs/$RUN_NAME
export EPOCHS=15

mkdir -p "$LOG_DIR" "$REPO_ROOT/experiment_outputs"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$BUILD_DIR" -j --target gege_train gege_eval

cp "$CONFIG_SRC" "$CONFIG_TMP"

perl -0pi -e "s/num_epochs: 30/num_epochs: 15/;
s|model_dir: .*|model_dir: $RUN_ROOT|;
s|checkpoint_dir: .*|checkpoint_dir: $RUN_ROOT|;" \
"$CONFIG_TMP"

for v in \$(compgen -e | grep '^GEGE_'); do unset \"\$v\"; done
export CUDA_VISIBLE_DEVICES=0,1
export GEGE_EMULATE_DOT_SINGLE_RELATION=1

export GEGE_CSR_DEBUG=0
export GEGE_BUCKET_STREAMING_LP=0
export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=0
export GEGE_FAST_MAP_TENSORS=1
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=0
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_DEG_CHUNK_EXCLUSION=0
export GEGE_PARTITION_BUFFER_PEER_RELAY=0
unset GEGE_GLOBAL_DEGREE_SAMPLING

rm -rf "$RUN_ROOT"

"$BUILD_DIR/gege/gege_train" "$CONFIG_TMP" \
  |& tee "$LOG_DIR/${RUN_NAME}_train.log"

GEGE_EMULATE_DOT_SINGLE_RELATION=1 \
GEGE_EVAL_CHUNKED_RANKS=1 \
GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768 \
"$BUILD_DIR/gege/gege_eval" "$CONFIG_TMP" \
  |& tee "$LOG_DIR/${RUN_NAME}_eval.log"

python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/${RUN_NAME}_train.log" \
  --eval-log "$LOG_DIR/${RUN_NAME}_eval.log" \
  --epochs "$EPOCHS"
```

To run a different one-flag LJ experiment, keep the same command block and change only one `export GEGE_...=` line.

### 5b. Required YAML Updates On A New Machine

The standardized configs are portable only after these fields are updated for the machine where you run:

- `storage.dataset.dataset_dir`
- `storage.model_dir`
- `evaluation.checkpoint_dir`

The safest pattern is to copy the checked-in YAML to a temporary run-specific file and patch only those machine-local paths:

```bash
export CONFIG_SRC=$REPO_ROOT/gege/configs/2gpu/livejournal_16p.yaml
export CONFIG_TMP=${TMPDIR:-/tmp}/livejournal_16p_run.yaml
export RUN_ROOT=$REPO_ROOT/experiment_outputs/livejournal_16p_run
export DATASET_DIR=/path/to/livejournal_16p_10k_eval

cp "$CONFIG_SRC" "$CONFIG_TMP"

perl -0pi -e "s|dataset_dir: .*|dataset_dir: $DATASET_DIR|;
s|model_dir: .*|model_dir: $RUN_ROOT|;
s|checkpoint_dir: .*|checkpoint_dir: $RUN_ROOT|;" \
"$CONFIG_TMP"
```

Use `"$CONFIG_TMP"` for both `gege_train` and `gege_eval`.

### 5c. If The Current Checkout Does Not Have `summarize_benchmark_logs.py`

The summarizer lives in:

- `scripts/summarize_benchmark_logs.py`

If your current checkout does not contain that file, use one of these options:

1. Update the branch so the script exists.
2. Run the script from another local checkout that already has it.

Example:

```bash
python /path/to/checkout/that/has/scripts/summarize_benchmark_logs.py \
  --train-log "$LOG_DIR/${RUN_NAME}_train.log" \
  --eval-log "$LOG_DIR/${RUN_NAME}_eval.log" \
  --epochs "$EPOCHS"
```

If you cannot use the script, at minimum record the final evaluation block and the epoch timing lines directly from the logs with `grep`.

### 6. Baseline-Branch Note

Use the same `gege/configs/2gpu/<dataset>.yaml` paths on `baseline/ge2_original`.

For `livejournal_16p` and `twitter_16p`:
- if your baseline branch includes the social Dot-emulation hook, set `GEGE_EMULATE_DOT_SINGLE_RELATION=1`
- otherwise record the run as a branch-compatible DistMult approximation rather than an exact paper-faithful social baseline

Recommended 2-GPU candidate flags and knobs:
- `GEGE_FAST_MAP_TENSORS`
- `GEGE_PARTITION_BUFFER_LP_FAST_PATH`
- `GEGE_GPU_ACTIVE_EDGE_SHUFFLE`
- `GEGE_OPTIMIZED_CUSTOM_SCHEDULE`
- `GEGE_DEG_CHUNK_EXCLUSION`
- `GEGE_PARTITION_BUFFER_PEER_RELAY`
- `GEGE_CSR_GATHER`
- `GEGE_CSR_UPDATE`
- `GEGE_BUCKET_STREAMING_LP`
- `dense_sync_batches` as a YAML sweep

Important notes:
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` is not part of the 2-GPU ladder.
- `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS` is not part of the current 2-GPU ladder.
- `GEGE_PARTITION_BUFFER_PEER_RELAY` is the main 2-GPU/4-GPU-specific probe.
- `dense_sync_batches` is a config knob, not an env flag.
- Keep YAML paths repo-relative in this table. Machine-specific paths belong inside the YAMLs or in local run wrappers, not in the experiment record.
- If `CUDA_VISIBLE_DEVICES` is set to a non-default physical pair such as `2,3`, keep YAML `device_ids` at `[0, 1]`; the process sees those selected GPUs as logical devices `0` and `1`.
- If exact filtered eval is slow but still making progress, wait for the final summary block. The evaluator prints `MRR` and `Hits@K` only after all evaluation batches finish.

Column definitions:
- `Flags Enabled / YAML Overrides`: explicit env vars and any YAML override for that run.
- `Epochs`: epoch count used for the averaged training metrics in that row.
- `Avg Inter-Epoch Gap`: average time between epoch end and the next epoch start.
- `Eval Notes`: `exact filtered`, `approximate sampled negatives`, or any failure/protocol caveat.

## LiveJournal 16p

| Row | Branch | Config | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `control_baseline_matched` | `baseline` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_peer_relay` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_CSR_GATHER=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_CSR_UPDATE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_2` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `dense_sync_batches=2` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_4` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `dense_sync_batches=4` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_8` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `dense_sync_batches=8` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_01_fast_map_tensors` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_02_lp_fast_path` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_03_active_edge_shuffle` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_04_optimized_custom_schedule` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05_deg_chunk_exclusion` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_06_partition_buffer_peer_relay` | `main` | `gege/configs/2gpu/livejournal_16p.yaml` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Twitter 16p

| Row | Branch | Config | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `control_baseline_matched` | `baseline` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_peer_relay` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_2` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `dense_sync_batches=2` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_4` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `dense_sync_batches=4` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_8` | `main` | `gege/configs/2gpu/twitter_16p.yaml` | `dense_sync_batches=8` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Freebase86M 16p

| Row | Branch | Config | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `control_baseline_matched` | `baseline` | `gege/configs/2gpu/freebase86m_16p.yaml` | `none` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_peer_relay` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_2` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `dense_sync_batches=2` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_4` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `dense_sync_batches=4` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `yaml_dense_sync_batches_8` | `main` | `gege/configs/2gpu/freebase86m_16p.yaml` | `dense_sync_batches=8` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Parsing Command

Use this command shape to summarize any completed run into the averaged table values:

```bash
python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/<run>_train.log" \
  --eval-log "$LOG_DIR/<run>_eval.log" \
  --epochs <30-or-10>
```
