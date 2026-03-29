# Single-GPU Ablation Results Template

Use this file to record single-GPU comparisons for:
- `livejournal_16p`
- `twitter_16p`
- `freebase86m_16p`

Core rules:
- Row 1 is `main` with all relevant flags off.
- Row 2 is `baseline` with matched config and matched runtime params.
- One-flag rows change exactly one flag relative to the control row.
- Incremental rows add one more flag on top of the previous stack row.
- Keep `GEGE_CSR_DEBUG=0` for every timing run.
- Measure `Avg Inter-Epoch Gap` as the time from `Finished training epoch N` to `Starting training epoch N+1`.
- Use [summarize_benchmark_logs.py](/home/smansou2/newCode/ge2/dandelion-dev/scripts/summarize_benchmark_logs.py) for averaged metrics.
- Keep YAML paths repo-relative in this file. Machine-specific paths belong in temp YAMLs or local wrappers.

**Canonical runners**

These checked-in scripts are the preferred way to run and update this markdown:

| Dataset | Script | Config | Default epochs | Mode | Auto-updates `.md` |
| --- | --- | --- | ---: | --- | --- |
| `livejournal_16p` | [run_lj_single_gpu_stack_ablation.sh](/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_lj_single_gpu_stack_ablation.sh) | `gege/configs/single_gpu/livejournal_16p.yaml` | `5` | train only | yes |
| `twitter_16p` | [run_twitter_single_gpu_stack_ablation.sh](/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_twitter_single_gpu_stack_ablation.sh) | `gege/configs/single_gpu/twitter_16p.yaml` | `3` | train only | yes |
| `freebase86m_16p` | [run_fb86m_single_gpu_stack_ablation.sh](/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_fb86m_single_gpu_stack_ablation.sh) | `gege/configs/single_gpu/freebase86m_16p.yaml` | `3` | train only | yes |

**Quick start**

Before running on a new machine, patch these YAML fields for your environment:
- `storage.dataset.dataset_dir`
- `storage.model_dir`
- `evaluation.checkpoint_dir`

Build once:

```bash
export REPO_ROOT=/path/to/ge2/repo
export BUILD_DIR=$REPO_ROOT/build_ge2env_ge2py39

cmake --build "$BUILD_DIR" -j --target gege_train
```

Run the LJ stack study:

```bash
cd "$REPO_ROOT"
bash scripts/run_lj_single_gpu_stack_ablation.sh
```

Run the FB86M stack study:

```bash
cd "$REPO_ROOT"
bash scripts/run_fb86m_single_gpu_stack_ablation.sh
```

Run the Twitter stack study:

```bash
cd "$REPO_ROOT"
bash scripts/run_twitter_single_gpu_stack_ablation.sh
```

Both runners:
- patch a temp YAML
- force `save_model: false`
- force `save_state: false`
- write train logs to `experiment_logs/`
- summarize train-time metrics
- write completed rows back into this markdown automatically

**Manual full train+eval runs**

For historical rows or exact eval runs, use the standardized configs under `gege/configs/single_gpu/`, run `gege_train`, then `gege_eval`, then summarize with:

```bash
python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/<run>_train.log" \
  --eval-log "$LOG_DIR/<run>_eval.log" \
  --epochs <N>
```

If exact filtered eval needs chunking, keep these on the eval command only:

```bash
GEGE_EVAL_CHUNKED_RANKS=1 \
GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768 \
"$BUILD_DIR/gege/gege_eval" "$CONFIG" \
  |& tee "$LOG_DIR/${RUN_NAME}_eval.log"
```

Important notes:
- `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM` is single-GPU-only. Do not carry it into multi-GPU studies.
- `GEGE_EMULATE_DOT_SINGLE_RELATION=1` is meaningful only for the single-relation social datasets.
- `GEGE_UNIQUE_BACKEND=bitmap` must be paired with the correct dataset-specific `GEGE_UNIQUE_BITMAP_NUM_NODES`.
- `GEGE_GLOBAL_DEGREE_SAMPLING=1` is a separate DegreeNS A/B probe; `unset` keeps the older batch-local behavior.
- For train-only runs, record that status in `Notes`.

Column definitions:
- `Flags Enabled`: explicit env vars enabled for that run.
- `Config`: YAML used for the run.
- `Avg Epoch Runtime` and other train metrics are averaged over the epoch window used by that run.
- `Notes` should capture any important eval or protocol caveat.

## LiveJournal 16p

Active LJ plan: use only the recovered fast-stack study. For every LJ stack run, hold these fixed:
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

These are already default-off in the current code, so the stack table below focuses only on the flags being added. The old LJ control/one-flag/incremental table is preserved below as a comment for reference.

| Branch | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: stack_00_control --> `main` | `all stack flags off; fixed off env block above` | `5` | 23065.60 ms | 5345127.10 | 2989.75 ms | 19.0 | 0.0062 | 6925.1680 | 1561.3062 | 0.0236 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_01_bitmap_unique --> `main` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=4847571` | `5` | 22955.80 ms | 5370674.20 | 3033.75 ms | 19.0 | 0.0056 | 6825.0968 | 1551.1044 | 0.0226 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_02_plus_emulate_dot_single_relation --> `main` | `previous_stack + GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `5` | 23218.80 ms | 5309795.60 | 3044.75 ms | 19.0 | 0.0054 | 7029.7758 | 1593.1896 | 0.0208 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_03_plus_mem_partition_buffer_pinned_host --> `main` | `previous_stack + GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `5` | 22601.20 ms | 5454938.70 | 3024.75 ms | 19.0 | 0.0042 | 6417.9740 | 1570.1520 | 0.0208 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_04_plus_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `5` | 22511.40 ms | 5476642.50 | 3000.25 ms | 19.0 | 0.0048 | 6360.9400 | 1573.1768 | 0.0196 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_05_plus_partition_buffer_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `5` | 21549.60 ms | 5721378.20 | 3001.25 ms | 19.0 | 0.0040 | 5614.4796 | 1437.9546 | 0.0220 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_06_plus_single_gpu_gpu_aware_custom --> `main` | `previous_stack + GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `5` | 20456.00 ms | 6026800.90 | 2975.00 ms | 19.0 | 0.0044 | 4479.8008 | 1483.5512 | 0.0206 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_07_plus_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `5` | 20470.80 ms | 6023439.80 | 724.75 ms | 19.0 | 0.0038 | 4505.2704 | 1465.3886 | 0.0198 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_08_plus_gpu_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `5` | 19141.80 ms | 6440686.90 | 645.25 ms | 19.0 | 0.0084 | 4519.4866 | 33.3720 | 0.0240 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |
| <!-- row: stack_09_plus_deg_chunk_exclusion --> `main` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `5` | 15497.80 ms | 7955580.40 | 651.00 ms | 19.0 | 0.0050 | 4509.0698 | 33.1920 | 0.0208 | `n/a` | 5-epoch LJ single-GPU train-only stack run via run_lj_single_gpu_stack_ablation.sh |

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

Active Twitter plan: use the same cumulative single-GPU stack structure as LJ. For every Twitter stack run, hold these fixed:
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: stack_00_control --> `main` | `all stack flags off; fixed off env block above` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_01_bitmap_unique --> `main` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=41652230` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_02_plus_emulate_dot_single_relation --> `main` | `previous_stack + GEGE_EMULATE_DOT_SINGLE_RELATION=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_03_plus_mem_partition_buffer_pinned_host --> `main` | `previous_stack + GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_04_plus_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_05_plus_partition_buffer_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_06_plus_single_gpu_gpu_aware_custom --> `main` | `previous_stack + GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_07_plus_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_08_plus_gpu_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |
| <!-- row: stack_09_plus_deg_chunk_exclusion --> `main` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `3` |  |  |  |  |  |  |  |  | `n/a` | 3-epoch Twitter single-GPU train-only stack run via run_twitter_single_gpu_stack_ablation.sh |

## Freebase86M 16p

Active FB86M plan: run the cumulative stack in the importance order inferred from the LJ study. For every FB86M stack run, hold these fixed:
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`
- do not set `GEGE_EMULATE_DOT_SINGLE_RELATION`

| Branch | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: stack_00_control --> `main` | `all stack flags off; fixed off env block above` | `3` | 257982.67 ms | 2293219.13 | 49691.50 ms | 19.0 | 0.0087 | 107258.8313 | 13736.8950 | 0.0237 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_01_plus_deg_chunk_exclusion --> `main` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `3` | 259791.33 ms | 2277254.27 | 49743.50 ms | 19.0 | 0.0077 | 108665.9903 | 13923.4243 | 0.0227 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_02_plus_gpu_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `3` | 245853.00 ms | 2406362.93 | 48911.00 ms | 19.0 | 0.0063 | 108476.2773 | 83.2737 | 0.0217 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_03_plus_single_gpu_gpu_aware_custom --> `main` | `previous_stack + GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `3` | 232719.00 ms | 2542170.17 | 49125.50 ms | 19.0 | 0.0063 | 95445.7277 | 96.6480 | 0.0247 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_04_plus_partition_buffer_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `3` | 221153.00 ms | 2675136.27 | 47795.00 ms | 19.0 | 0.0067 | 83406.1767 | 186.6980 | 0.0217 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_05_plus_mem_partition_buffer_pinned_host --> `main` | `previous_stack + GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `3` | 204814.00 ms | 2888526.53 | 48474.00 ms | 19.0 | 0.0063 | 66941.4303 | 192.2537 | 0.0203 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_06_plus_bitmap_unique --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151` | `3` | 204893.00 ms | 2887412.57 | 48180.50 ms | 19.0 | 0.0050 | 66740.3920 | 187.1983 | 0.0233 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_07_plus_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `3` | 204975.67 ms | 2886255.27 | 48121.50 ms | 19.0 | 0.0043 | 66803.5517 | 198.0767 | 0.0223 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: stack_08_plus_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `3` | 204314.33 ms | 2895591.77 | 8155.00 ms | 19.0 | 0.0047 | 66627.6183 | 168.7187 | 0.0193 | `n/a` | 3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh |
| <!-- row: probe_bucket_gpuaware_hot_lp_bitmap_pinned --> `main` | `GEGE_BUCKET_STREAMING_LP=1, GEGE_FAST_MAP_TENSORS=0, GEGE_CSR_GATHER=0, GEGE_CSR_UPDATE=0, GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1, GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151` | `3` | 202818.67 ms | 2916943.27 | 8194.50 ms | 19.0 | 0.0050 | 66723.5367 | 11.2103 | 0.0207 | `n/a` | 3-epoch FB86M single-GPU custom probe from `experiment_logs/freebase86m_16p_flagprobe_bucket_gpuaware_hot_lp_bitmap_pinned_20260328_201605_train.log` |

| Branch | Flags Enabled | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: control_baseline_matched --> `baseline` | `none` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_fast_map_tensors --> `main` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_partition_buffer_lp_fast_path --> `main` | `GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_gpu_active_edge_shuffle --> `main` | `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_keep_storage_hot_between_epochs --> `main` | `GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_single_gpu_gpu_aware_custom --> `main` | `GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_optimized_custom_schedule --> `main` | `GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_deg_chunk_exclusion --> `main` | `GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_global_degree_sampling --> `main` | `GEGE_GLOBAL_DEGREE_SAMPLING=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_csr_gather --> `main` | `GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_csr_update --> `main` | `GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_bucket_streaming_lp --> `main` | `GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_mem_partition_buffer_pinned_host --> `main` | `GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: oneflag_unique_backend_bitmap --> `main` | `GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_01_fast_map_tensors --> `main` | `GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_02_lp_fast_path --> `main` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_03_active_edge_shuffle --> `main` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_04_keep_hot --> `main` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_05a_scheduler_gpu_aware_custom --> `main` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_05b_scheduler_optimized_custom --> `main` | `GEGE_FAST_MAP_TENSORS=1, GEGE_PARTITION_BUFFER_LP_FAST_PATH=1, GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1, GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1, GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_06_deg_chunk_exclusion --> `main` | `previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_07_csr_gather --> `main` | `previous_stack + GEGE_CSR_GATHER=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_08_csr_update --> `main` | `previous_stack + GEGE_CSR_UPDATE=1` | `10` |  |  |  |  |  |  |  |  |  |  |
| <!-- row: incremental_09_bucket_streaming_lp --> `main` | `previous_stack + GEGE_BUCKET_STREAMING_LP=1` | `10` |  |  |  |  |  |  |  |  |  |  |

## Parsing Command

Use this command shape to summarize any completed run into the averaged table values:

```bash
python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/<run>_train.log" \
  --eval-log "$LOG_DIR/<run>_eval.log" \
  --epochs <30-or-10>
```
