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
- Average training metrics over the epoch window used by the run. Historical `livejournal_16p` control rows below use `30` epochs, but the current checked-in `gege/configs/single_gpu/livejournal_16p.yaml` uses `15`.
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
for v in $(compgen -e | grep '^GEGE_'); do unset "$v"; done
export CUDA_VISIBLE_DEVICES=0
```

For social-dataset paper-faithful runs, `GEGE_EMULATE_DOT_SINGLE_RELATION=1` is often part of the final stack. For one-flag ablations, do not force it globally; only enable it on the dedicated `oneflag_emulate_dot_single_relation` row.

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

for v in \$(compgen -e | grep '^GEGE_'); do unset \"\$v\"; done
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

### 5b. Compact Train-Only LiveJournal Sweep

For the current train-time-only LJ sweep, use the checked-in `gege/configs/single_gpu/livejournal_16p.yaml` as-is. It currently has `num_epochs: 15`.

Run the default remaining one-flag cases:

```bash
cd "$REPO_ROOT"
bash scripts/run_lj_single_gpu_train_only_ablation.sh
```

Run only specific cases:

```bash
cd "$REPO_ROOT"
bash scripts/run_lj_single_gpu_train_only_ablation.sh \
  partition_buffer_lp_fast_path \
  gpu_active_edge_shuffle \
  keep_storage_hot_between_epochs
```

The script:
- uses the current `num_epochs` value from `gege/configs/single_gpu/livejournal_16p.yaml`
- runs train only, no eval
- writes train logs to `experiment_logs/`
- writes per-run summaries with `mrr=None` and the train-time averages filled in

When recording these train-only rows in the table, set:
- `Epochs` to the current config value, which is `15`
- `Eval Notes` to `train only; eval skipped`

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
- `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST`
- `GEGE_UNIQUE_BACKEND=bitmap` with dataset-specific `GEGE_UNIQUE_BITMAP_NUM_NODES`
- `GEGE_EMULATE_DOT_SINGLE_RELATION` on single-relation social datasets only

Important notes:
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` and `GEGE_OPTIMIZED_CUSTOM_SCHEDULE` are alternative scheduler paths. In the incremental section, keep two branches if needed instead of stacking them blindly.
- `GEGE_EMULATE_DOT_SINGLE_RELATION=1` is meaningful only for the single-relation social datasets. Keep it as a dedicated one-flag A/B row instead of silently forcing it on every social run.
- `GEGE_GLOBAL_DEGREE_SAMPLING` is a separate A/B probe for the DegreeNS implementation. `unset` keeps the old/original batch-local behavior; `=1` enables the new global degree-weighted path.
- `GEGE_UNIQUE_BACKEND=bitmap` must be paired with the correct dataset-specific `GEGE_UNIQUE_BITMAP_NUM_NODES`.
- For exact paper-style eval on large graphs, use exact filtered ranks on the `*_10k_eval` datasets. If eval needs chunking to fit, keep `GEGE_EVAL_CHUNKED_RANKS=1` and `GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768` on the eval command only.
- Keep YAML paths repo-relative in this table. Machine-specific paths belong inside the YAMLs or in local run wrappers, not in the experiment record.

Column definitions:
- `Flags Enabled`: explicit env vars enabled for that run. Do not assume hidden social or bitmap settings; list them directly when they are part of the row.
- `Config`: YAML used for the run.
- All training metrics are averages over the dataset-specific epoch window above.
- `Avg Inter-Epoch Gap` is the average time between epoch end and the next epoch start.
- `Eval Notes` should record `exact filtered`, `approximate sampled negatives`, or any model/protocol caveat.

## LiveJournal 16p

Active LJ plan: use only the recovered fast-stack study. For every LJ stack run, hold these fixed:
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

These are already default-off in the current code, so the stack table below focuses only on the flags being added. The old LJ control/one-flag/incremental table is preserved below as a comment for reference.

| Row | Branch | Config | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `stack_00_control` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `all stack flags off; fixed off env block above` | `5` | 23065.60 ms | 5345127.10 | 2989.75 ms | 19.0 | 0.0062 | 6925.1680 | 1561.3062 | 0.0236 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_00_control_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_01_bitmap_unique` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=4847571` | `5` | 22955.80 ms | 5370674.20 | 3033.75 ms | 19.0 | 0.0056 | 6825.0968 | 1551.1044 | 0.0226 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_01_bitmap_unique_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_02_plus_emulate_dot_single_relation` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `5` | 23218.80 ms | 5309795.60 | 3044.75 ms | 19.0 | 0.0054 | 7029.7758 | 1593.1896 | 0.0208 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_02_plus_emulate_dot_single_relation_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_03_plus_mem_partition_buffer_pinned_host` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `5` | 22601.20 ms | 5454938.70 | 3024.75 ms | 19.0 | 0.0042 | 6417.9740 | 1570.1520 | 0.0208 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_03_plus_mem_partition_buffer_pinned_host_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_04_plus_fast_map_tensors` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `5` | 22511.40 ms | 5476642.50 | 3000.25 ms | 19.0 | 0.0048 | 6360.9400 | 1573.1768 | 0.0196 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_04_plus_fast_map_tensors_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_05_plus_partition_buffer_lp_fast_path` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `5` | 21549.60 ms | 5721378.20 | 3001.25 ms | 19.0 | 0.0040 | 5614.4796 | 1437.9546 | 0.0220 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_05_plus_partition_buffer_lp_fast_path_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_06_plus_single_gpu_gpu_aware_custom` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `5` | 20456.00 ms | 6026800.90 | 2975.00 ms | 19.0 | 0.0044 | 4479.8008 | 1483.5512 | 0.0206 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_06_plus_single_gpu_gpu_aware_custom_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_07_plus_keep_storage_hot_between_epochs` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `5` | 20470.80 ms | 6023439.80 | 724.75 ms | 19.0 | 0.0038 | 4505.2704 | 1465.3886 | 0.0198 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_07_plus_keep_storage_hot_between_epochs_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_08_plus_gpu_active_edge_shuffle` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `5` | 19141.80 ms | 6440686.90 | 645.25 ms | 19.0 | 0.0084 | 4519.4866 | 33.3720 | 0.0240 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_08_plus_gpu_active_edge_shuffle_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| `stack_09_plus_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `5` | 15497.80 ms | 7955580.40 | 651.00 ms | 19.0 | 0.0050 | 4509.0698 | 33.1920 | 0.0208 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/livejournal_16p_main_stack_09_plus_deg_chunk_exclusion_5ep_trainonly_train.log` | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |

LJ marginal importance summary, ranked by epoch-time improvement in the cumulative stack study:

| Rank | Added Flag | Source Row | Delta vs Previous Row | Resulting Avg Epoch Runtime | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `GEGE_DEG_CHUNK_EXCLUSION=1` | `stack_09_plus_deg_chunk_exclusion` | `-3644.0 ms` | `15497.80 ms` | Biggest remaining training-side gain; this is the step that reaches the ~15s regime. |
| 2 | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `stack_08_plus_gpu_active_edge_shuffle` | `-1329.0 ms` | `19141.80 ms` | Collapses rebuild from ~1.47s to ~33ms. |
| 3 | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `stack_06_plus_single_gpu_gpu_aware_custom` | `-1093.6 ms` | `20456.00 ms` | Major scheduling/path improvement. |
| 4 | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `stack_05_plus_partition_buffer_lp_fast_path` | `-961.8 ms` | `21549.60 ms` | First large drop in update-path cost. |
| 5 | `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `stack_03_plus_mem_partition_buffer_pinned_host` | `-617.6 ms` | `22601.20 ms` | Moderate host-buffer/storage gain. |
| 6 | `GEGE_UNIQUE_BACKEND=bitmap` + `GEGE_UNIQUE_BITMAP_NUM_NODES=4847571` | `stack_01_bitmap_unique` | `-109.8 ms` | `22955.80 ms` | Small but positive. |
| 7 | `GEGE_FAST_MAP_TENSORS=1` | `stack_04_plus_fast_map_tensors` | `-89.8 ms` | `22511.40 ms` | Small incremental gain after pinned host. |
| 8 | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `stack_07_plus_keep_storage_hot_between_epochs` | `+14.8 ms` | `20470.80 ms` | Neutral on epoch time, but it cuts inter-epoch gap sharply. |
| 9 | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `stack_02_plus_emulate_dot_single_relation` | `+263.0 ms` | `23218.80 ms` | Needed for the intended social-model setup, but not a speed optimization. |

<!--
| Row | Branch | Config | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `control_baseline_matched` | `baseline` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_keep_storage_hot_between_epochs` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_single_gpu_gpu_aware_custom` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_global_degree_sampling` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_GLOBAL_DEGREE_SAMPLING=1` | `30` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_CSR_GATHER=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_CSR_UPDATE=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_mem_partition_buffer_pinned_host` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_unique_backend_bitmap` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=4847571` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_emulate_dot_single_relation` | `main` | `gege/configs/single_gpu/livejournal_16p.yaml` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `5` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
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
-->

## Twitter 16p

| Row | Branch | Config | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | MRR | Hits@1 | Hits@3 | Hits@10 | Eval Notes | Train Log | Eval Log | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `control_main_all_flags_off` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `3` | 506444.33 ms | 6026260.33 | 35162.50 ms | 19.0 | 0.0060 | 85572.0913 | 85771.0893 | 0.0240 | n/a | n/a | n/a | n/a | train only; eval skipped | `experiment_logs/twitter_16p_main_control_main_all_flags_off_3ep_trainonly_train.log` | `n/a` | 3-epoch twitter_16p single-GPU train-only overnight run via run_single_gpu_train_only_ablation_matrix.sh (2026-03-28) |
| `control_baseline_matched` | `baseline` | `gege/configs/single_gpu/twitter_16p.yaml` | `branch-compatible social baseline; not exact DOT unless patched` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_fast_map_tensors` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_partition_buffer_lp_fast_path` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_gpu_active_edge_shuffle` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_keep_storage_hot_between_epochs` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_single_gpu_gpu_aware_custom` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_optimized_custom_schedule` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_deg_chunk_exclusion` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_global_degree_sampling` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_GLOBAL_DEGREE_SAMPLING=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_mem_partition_buffer_pinned_host` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_unique_backend_bitmap` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=41652230` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_emulate_dot_single_relation` | `main` | `gege/configs/single_gpu/twitter_16p.yaml` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
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
| `oneflag_global_degree_sampling` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_GLOBAL_DEGREE_SAMPLING=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_gather` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_csr_update` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_bucket_streaming_lp` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_mem_partition_buffer_pinned_host` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| `oneflag_unique_backend_bitmap` | `main` | `gege/configs/single_gpu/freebase86m_16p.yaml` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151` | `10` |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
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
