## LiveJournal 16p

Active LiveJournal plan: use the same 4-GPU cumulative stack style as the 2-GPU table. For every LiveJournal stack run, hold these fixed:
- `GEGE_EMULATE_DOT_SINGLE_RELATION=1`
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; all optional stack flags off; fixed off env block above` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_01_deg_chunk_exclusion --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_02_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_03_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_04_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_05_unique_backend_bitmap --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=4847571` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_06_optimized_custom_schedule --> `main` | `previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_07_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |
| <!-- row: incremental_08_partition_buffer_peer_relay --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `30` |  |  |  |  |  |  |  |  | `n/a` | `LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs` |

## Twitter 16p

Active Twitter plan: use the same 4-GPU cumulative stack style as LiveJournal. For every Twitter stack run, hold these fixed:
- `GEGE_EMULATE_DOT_SINGLE_RELATION=1`
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; all optional stack flags off; fixed off env block above` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_01_deg_chunk_exclusion --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_02_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_03_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_04_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_05_unique_backend_bitmap --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=41652230` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_06_optimized_custom_schedule --> `main` | `previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_07_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |
| <!-- row: incremental_08_partition_buffer_peer_relay --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh` |

## Freebase86M 16p

Active Freebase86M plan: use the same 4-GPU cumulative stack style as LiveJournal and Twitter. For every Freebase86M stack run, hold these fixed:
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `all optional stack flags off; fixed off env block above; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_01_deg_chunk_exclusion --> `main` | `previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_02_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_03_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_04_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_05_unique_backend_bitmap --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_06_optimized_custom_schedule --> `main` | `previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_07_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |
| <!-- row: incremental_08_partition_buffer_peer_relay --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `10` |  |  |  |  |  |  |  |  | `n/a` | `Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh` |

## Parsing Command

Use this command shape to summarize any completed run into the averaged table values:

```bash
python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/<run>_train.log" \
  --eval-log "$LOG_DIR/<run>_eval.log" \
  --epochs <30-or-10>
```
