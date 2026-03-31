## LiveJournal 16p

Active LiveJournal plan: use the same 4-GPU cumulative stack style as the 2-GPU table. For every LiveJournal stack run, hold these fixed:
- `GEGE_EMULATE_DOT_SINGLE_RELATION=1`
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; all optional stack flags off; fixed off env block above` | `5` | 6237.40 ms | 19957058.40 | 2475.00 ms | 6.4 | 636.8436 | 1939.5684 | 423.9158 | 276.7780 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_01_deg_chunk_exclusion --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1` | `5` | 5100.60 ms | 24405902.00 | 2481.75 ms | 6.4 | 466.6018 | 1944.2876 | 421.1558 | 282.0488 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_02_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `5` | 4703.60 ms | 26468409.20 | 2346.25 ms | 6.4 | 470.0928 | 1946.6464 | 11.1852 | 49.5436 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_03_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `5` | 4747.80 ms | 26221005.60 | 2292.50 ms | 6.4 | 472.3390 | 1990.8120 | 12.7720 | 45.0830 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_04_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `5` | 4751.40 ms | 26203942.80 | 2293.75 ms | 6.4 | 472.6532 | 2014.4368 | 13.4248 | 52.2466 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_05_unique_backend_bitmap --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=4847571` | `5` | 4745.60 ms | 26233180.80 | 2273.50 ms | 6.4 | 464.8116 | 2016.0624 | 17.7240 | 38.1832 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_06_optimized_custom_schedule --> `main` | `previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `5` | 4443.60 ms | 28009868.80 | 4053.00 ms | 6.4 | 31.9936 | 2002.2204 | 16.3852 | 32.4916 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_07_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `5` | 4441.80 ms | 28021750.40 | 2588.75 ms | 6.4 | 30.5546 | 1988.1764 | 19.1814 | 28.6676 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |
| <!-- row: incremental_08_partition_buffer_peer_relay --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `5` | 3754.20 ms | 33163196.40 | 2595.50 ms | 6.4 | 29.9422 | 928.0190 | 28.8044 | 35.5556 | `n/a` | LJ 4-GPU importance stack run via run_lj_4gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs |

## Twitter 16p

Active Twitter plan: use the same 4-GPU cumulative stack style as LiveJournal. For every Twitter stack run, hold these fixed:
- `GEGE_EMULATE_DOT_SINGLE_RELATION=1`
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; all optional stack flags off; fixed off env block above` | `3` | 130853.33 ms | 20382088.67 | 30686.00 ms | 8.0 | 9515.7027 | 27226.5727 | 31346.5383 | 19658.5007 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_01_deg_chunk_exclusion --> `main` | `GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1` | `3` | 106634.00 ms | 25011027.33 | 30728.00 ms | 8.0 | 7089.6397 | 27215.7327 | 30467.9293 | 20179.0263 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_02_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `3` | 83131.67 ms | 32091324.67 | 23369.50 ms | 8.0 | 7051.7530 | 27330.7423 | 66.1047 | 365.6263 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_03_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `3` | 84924.33 ms | 31412854.00 | 21171.00 ms | 8.0 | 7078.5013 | 30860.5580 | 80.8613 | 562.3387 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_04_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `3` | 85039.67 ms | 31370482.67 | 21028.50 ms | 8.0 | 7057.9233 | 31011.1110 | 93.5830 | 427.1810 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_05_unique_backend_bitmap --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=41652230` | `3` | 84543.33 ms | 31554322.00 | 21283.00 ms | 8.0 | 7035.1410 | 30712.2970 | 105.8590 | 411.8487 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_06_optimized_custom_schedule --> `main` | `previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `3` | 79355.33 ms | 33607820.67 | 22484.00 ms | 8.0 | 1468.2663 | 31218.9700 | 116.0183 | 208.7393 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_07_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `3` | 79415.00 ms | 33582594.67 | 11369.00 ms | 8.0 | 1453.4550 | 31318.1130 | 103.2440 | 455.5700 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |
| <!-- row: incremental_08_partition_buffer_peer_relay --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `3` | 74056.67 ms | 36012506.00 | 11315.00 ms | 8.0 | 1442.1580 | 19984.1710 | 204.6273 | 322.0793 | `n/a` | Twitter 4-GPU importance stack run via run_twitter_4gpu_stack_ablation.sh |

## Freebase86M 16p

Active Freebase86M plan: use the same 4-GPU cumulative stack style as LiveJournal and Twitter. For every Freebase86M stack run, hold these fixed:
- `GEGE_BUCKET_STREAMING_LP=0`
- `GEGE_CSR_GATHER=0`
- `GEGE_CSR_UPDATE=0`
- `GEGE_CSR_DEBUG=0`

| Branch | Flags Enabled / YAML Overrides | Epochs | Avg Epoch Runtime | Avg Edges per Second | Avg Inter-Epoch Gap | Avg swap_count | Avg swap_barrier_wait_ms | Avg swap_update_ms | Avg swap_rebuild_ms | Avg swap_sync_wait_ms | Eval Log | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <!-- row: control_main_all_flags_off --> `main` | `all optional stack flags off; fixed off env block above; unset GEGE_GLOBAL_DEGREE_SAMPLING` | `3` | 68102.00 ms | 10047936.67 | 35464.00 ms | 8.0 | 8324.0640 | 35763.7967 | 5358.0220 | 3608.2047 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_01_deg_chunk_exclusion --> `main` | `previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1` | `3` | 67997.33 ms | 10064005.33 | 35471.50 ms | 8.0 | 8322.4103 | 35371.0533 | 4602.6047 | 4212.0907 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_02_active_edge_shuffle --> `main` | `previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1` | `3` | 63961.33 ms | 10697785.00 | 34158.00 ms | 8.0 | 8323.1580 | 35520.1480 | 66.8550 | 1068.0700 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_03_lp_fast_path --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1` | `3` | 63006.33 ms | 10859859.00 | 33031.00 ms | 8.0 | 8318.1303 | 32894.3463 | 74.6503 | 1276.9123 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_04_fast_map_tensors --> `main` | `previous_stack + GEGE_FAST_MAP_TENSORS=1` | `3` | 62919.67 ms | 10874936.67 | 33242.50 ms | 8.0 | 8322.1627 | 33241.8153 | 77.6890 | 1047.6987 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_05_unique_backend_bitmap --> `main` | `previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151` | `3` | 62960.67 ms | 10867785.00 | 33151.50 ms | 8.0 | 8315.0910 | 33326.1333 | 91.4880 | 1204.5370 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_06_optimized_custom_schedule --> `main` | `previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1` | `3` | 61012.33 ms | 11214540.33 | 34802.00 ms | 8.0 | 6502.6173 | 33371.1437 | 80.8870 | 1105.8490 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_07_keep_storage_hot_between_epochs --> `main` | `previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1` | `3` | 61078.33 ms | 11202343.33 | 12260.00 ms | 8.0 | 6506.8647 | 33157.2530 | 90.2613 | 1097.7793 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |
| <!-- row: incremental_08_partition_buffer_peer_relay --> `main` | `previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1` | `3` | 49822.33 ms | 13733674.00 | 12368.50 ms | 8.0 | 6503.4303 | 11815.9140 | 93.7410 | 325.1633 | `n/a` | Freebase86M 4-GPU importance stack run via run_fb86m_4gpu_stack_ablation.sh |

## Parsing Command

Use this command shape to summarize any completed run into the averaged table values:

```bash
python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
  --train-log "$LOG_DIR/<run>_train.log" \
  --eval-log "$LOG_DIR/<run>_eval.log" \
  --epochs <30-or-10>
```
