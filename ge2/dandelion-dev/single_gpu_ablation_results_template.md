# Single-GPU Ablation Results Template

Use this file to record the single-GPU comparison matrix for:
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
- Use the standardized single-GPU configs under `gege/configs/single_gpu/` on both branches.

## How To Run These Experiments

Use this procedure on any machine. Do not assume the YAMLs are ready as-is for a new environment.

### 1. Choose Branch, Build Directory, And Dataset Config

Use the same relative config names on both branches:

| Dataset | Config path | Epochs to average |
| --- | --- | ---: |
| `livejournal_16p` | `gege/configs/single_gpu/livejournal_16p.yaml` | `30` |
| `twitter_16p` | `gege/configs/single_gpu/twitter_16p.yaml` | `10` |
| `freebase86m_16p` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `10` |

Before running on a different machine, update these YAML fields for your environment:
- `storage.dataset.dataset_dir`
- `storage.model_dir`
- `evaluation.checkpoint_dir`

### 2. Set Generic Environment Variables

```bash
export REPO_ROOT=/path/to/ge2/repo
export BUILD_DIR=$REPO_ROOT/<build-directory-containing-gege-binaries>
export CONFIG=$REPO_ROOT/gege/configs/single_gpu/livejournal_16p.yaml
export LOG_DIR=$REPO_ROOT/experiment_logs
export RUN_NAME=livejournal_16p_main_control
export EPOCHS=30

mkdir -p "$LOG_DIR"
```

Examples:
- `main` might use `BUILD_DIR=$REPO_ROOT/build_ge2env`
- `baseline` might use `BUILD_DIR=$REPO_ROOT/build_gcc9`

### 3. Reset Flags And Select One GPU

```bash
for v in $(compgen -e | rg '^GEGE_'); do unset "$v"; done
export CUDA_VISIBLE_DEVICES=0
```

For `livejournal_16p` and `twitter_16p`, enable paper-faithful social-dot emulation on branches that support it:

```bash
export GEGE_EMULATE_DOT_SINGLE_RELATION=1
```

For `freebase86m_16p`, leave `GEGE_EMULATE_DOT_SINGLE_RELATION` unset.

### 4. Main-Branch All-Flags-Off Control

Use the standardized YAML plus explicit env overrides. This replaces the older branch-specific `*_paper_allflags_off.yaml` naming.

```bash
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

unset GEGE_GLOBAL_DEGREE_SAMPLING
```

For one-flag and incremental rows, change only the flags listed in the table. Keep every other setting fixed.

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

export BUILD_DIR=$REPO_ROOT/build_ge2env
export CONFIG_SRC=$REPO_ROOT/gege/configs/single_gpu/livejournal_16p.yaml
export CONFIG_TMP=${TMPDIR:-/tmp}/livejournal_16p_main_fast_map_tensors_15ep.yaml
export LOG_DIR=$REPO_ROOT/experiment_logs
export RUN_NAME=livejournal_16p_main_fast_map_tensors_15ep
export RUN_ROOT=$REPO_ROOT/experiment_outputs/$RUN_NAME
export EPOCHS=15

mkdir -p "$LOG_DIR" "$REPO_ROOT/experiment_outputs"

cmake --build "$BUILD_DIR" -j --target gege_train gege_eval

cp "$CONFIG_SRC" "$CONFIG_TMP"

perl -0pi -e "s/num_epochs: 30/num_epochs: 15/;
s|model_dir: .*|model_dir: $RUN_ROOT|;
s|checkpoint_dir: .*|checkpoint_dir: $RUN_ROOT|;" \
"$CONFIG_TMP"

for v in \$(compgen -e | rg '^GEGE_'); do unset \"\$v\"; done
export CUDA_VISIBLE_DEVICES=0
export GEGE_EMULATE_DOT_SINGLE_RELATION=1

export GEGE_CSR_DEBUG=0
export GEGE_BUCKET_STREAMING_LP=0
export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=0
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=0
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=0
export GEGE_FAST_MAP_TENSORS=1
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=0
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_DEG_CHUNK_EXCLUSION=0
export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=0
unset GEGE_GLOBAL_DEGREE_SAMPLING

rm -rf "$RUN_ROOT"

"$BUILD_DIR/gege/gege_train" "$CONFIG_TMP" \
  |& tee "$LOG_DIR/${RUN_NAME}_train.log"

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

### 6. Baseline-Branch Note

Use the same `gege/configs/single_gpu/<dataset>.yaml` paths on `baseline/ge2_original`.

For `livejournal_16p` and `twitter_16p`:
- if your baseline branch includes the social Dot-emulation hook, set `GEGE_EMULATE_DOT_SINGLE_RELATION=1`
- otherwise record the run as a branch-compatible DistMult approximation rather than an exact paper-faithful social baseline

Recommended single-GPU candidate flags:
- `GEGE_FAST_MAP_TENSORS`
- `GEGE_PARTITION_BUFFER_LP_FAST_PATH`
- `GEGE_GPU_ACTIVE_EDGE_SHUFFLE`
- `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS`
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM`
- `GEGE_OPTIMIZED_CUSTOM_SCHEDULE`
- `GEGE_DEG_CHUNK_EXCLUSION`
- `GEGE_GLOBAL_DEGREE_SAMPLING`
- `GEGE_CSR_GATHER`
- `GEGE_CSR_UPDATE`
- `GEGE_BUCKET_STREAMING_LP`

Important notes:
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` and `GEGE_OPTIMIZED_CUSTOM_SCHEDULE` are alternative scheduler paths. In the incremental section, keep two branches if needed instead of stacking them blindly.
- For `livejournal_16p` and `twitter_16p`, `GEGE_EMULATE_DOT_SINGLE_RELATION=1` is paper-required and should stay enabled for every paper-faithful `main` run. Treat it as model selection, not as an optimization flag.
- `GEGE_GLOBAL_DEGREE_SAMPLING` is a separate A/B probe for the DegreeNS implementation. `unset` keeps the old/original batch-local behavior; `=1` enables the new global degree-weighted path.
- For exact paper-style eval on large graphs, use exact filtered ranks on the `*_10k_eval` datasets. If eval needs chunking to fit, keep `GEGE_EVAL_CHUNKED_RANKS=1` and `GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768` on the eval command only.
- Keep YAML paths repo-relative in this table. Machine-specific paths belong inside the YAMLs or in local run wrappers, not in the experiment record.

Column definitions:
- `Flags Enabled`: explicit env vars enabled for that run. For `livejournal_16p` and `twitter_16p`, assume `GEGE_EMULATE_DOT_SINGLE_RELATION=1` on every `main` row even when the cell only lists optimization flags.
- `Config`: YAML used for the run.
- All training metrics are averages over the dataset-specific epoch window above.
- `Avg Inter-Epoch Gap` is the average time between epoch end and the next epoch start.
- `Eval Notes` should record `exact filtered`, `approximate sampled negatives`, or any model/protocol caveat.

## LiveJournal 16p

| Row | Branch | Config | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `30` | `24245.07 ms` | `5150020.55` | `3180.34 ms` | `19.0` | `0.0053` | `7315.8454` | `1968.4273` | `0.0202` | `0.127752` | `0.0372` | `0.1549` | `0.3071` | `exact filtered; 10k eval` | `historical control summary` | `historical control summary` | `Same experiment shape as gege/configs/single_gpu/livejournal_16p.yaml with the explicit all-flags-off env block` |
| `control_baseline_matched` | `baseline` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `30` | `28590.60 ms` | `2171981.09` | `751.52 ms` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `eval failed: checkpoint load reported "Unrecognized data format"` | `/tmp/livejournal_16p_baseline_single_gpu_train.log` | `/tmp/livejournal_16p_baseline_single_gpu_eval.log` | `30-epoch baseline LJ single-GPU training completed; baseline train log does not emit swap breakdown, and eval did not reach final ranking metrics` |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_keep_storage_hot_between_epochs` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_single_gpu_gpu_aware_custom` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_global_degree_sampling` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_GLOBAL_DEGREE_SAMPLING=1` | `30` | `22033.67 ms` | `5642599.71` | `3229.38 ms` | `19.0` | `0.0045` | `7290.2811` | `1813.0670` | `0.0227` | `0.127764` | `0.0368` | `0.1547` | `0.3079` | `exact filtered; 10k eval` | `livejournal_16p_main_global_degree_train.log` | `livejournal_16p_main_global_degree_eval.log` | `Mean Rank=74007.133700; Hits@5=0.215200; Hits@50=0.540000; Hits@100=0.624600. Captured while the global-degree path was active; after the revert, rerun by setting GEGE_GLOBAL_DEGREE_SAMPLING=1 on top of the standard single_gpu YAML` |
| `oneflag_csr_gather` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_CSR_GATHER=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_CSR_UPDATE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_01_fast_map_tensors` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_02_lp_fast_path` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_03_active_edge_shuffle` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_04_keep_hot` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05a_scheduler_gpu_aware_custom` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05b_scheduler_optimized_custom` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_06_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_07_csr_gather` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_CSR_GATHER=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_08_csr_update` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_CSR_UPDATE=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_09_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_BUCKET_STREAMING_LP=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Twitter 16p

| Row | Branch | Config | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `control_baseline_matched` | `baseline` | `gege/configs/single_gpu/twitter_16p.yaml` | `branch-compatible social baseline; not exact DOT unless patched` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_keep_storage_hot_between_epochs` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_single_gpu_gpu_aware_custom` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_01_fast_map_tensors` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_02_lp_fast_path` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_03_active_edge_shuffle` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_04_keep_hot` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05a_scheduler_gpu_aware_custom` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05b_scheduler_optimized_custom` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_06_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_07_csr_gather` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `previous_stack + GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_08_csr_update` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `previous_stack + GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_09_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `previous_stack + GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Freebase86M 16p

| Row | Branch | Config | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `control_baseline_matched` | `baseline` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `none` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_keep_storage_hot_between_epochs` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_single_gpu_gpu_aware_custom` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_01_fast_map_tensors` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_02_lp_fast_path` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_03_active_edge_shuffle` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_04_keep_hot` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05a_scheduler_gpu_aware_custom` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_05b_scheduler_optimized_custom` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_06_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_07_csr_gather` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `previous_stack + GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_08_csr_update` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `previous_stack + GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `incremental_09_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `previous_stack + GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Parsing Command

Use this command shape to summarize any completed run into the averaged table values:

```bash
python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/<run>_train.log" \
  --eval-log "$LOG_DIR/<run>_eval.log" \
  --epochs <30-or-10>
```
