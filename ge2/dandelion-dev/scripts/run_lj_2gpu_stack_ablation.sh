#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_2gpu_runs}"
CONFIG_SRC="${CONFIG_SRC:-$REPO_ROOT/gege/configs/2gpu/livejournal_16p.yaml}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dev/shm/smansou2_ge2/lj_2gpu_stack_ablation}"
RESULTS_MD="${RESULTS_MD:-$REPO_ROOT/2gpu_ablation_results_template.md}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$REPO_ROOT/scripts/summarize_benchmark_logs.py}"
UPDATE_SCRIPT="${UPDATE_SCRIPT:-$REPO_ROOT/scripts/update_single_gpu_ablation_table.py}"
EPOCHS="${EPOCHS:-30}"
DATASET_DIR="${DATASET_DIR:-}"

stack_cases=(
  control_main_all_flags_off
  incremental_01_deg_chunk_exclusion
  incremental_02_active_edge_shuffle
  incremental_03_lp_fast_path
  incremental_04_fast_map_tensors
  incremental_05_unique_backend_bitmap
  incremental_06_optimized_custom_schedule
  incremental_07_keep_storage_hot_between_epochs
  incremental_08_partition_buffer_peer_relay
)

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

configure_build_if_needed() {
  if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DLIBNVTOOLSEXT=
  fi
  cmake --build "$BUILD_DIR" -j --target gege_train >/dev/null
}

reset_env() {
  local v
  while IFS= read -r v; do
    unset "$v"
  done < <(compgen -e | grep '^GEGE_' || true)

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

  # Fixed social-model setting for LJ 2-GPU runs.
  export GEGE_EMULATE_DOT_SINGLE_RELATION=1

  # Fixed-off block for timing runs.
  export GEGE_CSR_DEBUG=0
  export GEGE_BUCKET_STREAMING_LP=0
  export GEGE_CSR_GATHER=0
  export GEGE_CSR_UPDATE=0

  # Stack flags default off until enabled by the case.
  export GEGE_DEG_CHUNK_EXCLUSION=0
  export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=0
  export GEGE_PARTITION_BUFFER_LP_FAST_PATH=0
  export GEGE_FAST_MAP_TENSORS=0
  unset GEGE_UNIQUE_BACKEND
  unset GEGE_UNIQUE_BITMAP_NUM_NODES
  export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
  export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=0
  export GEGE_PARTITION_BUFFER_PEER_RELAY=0

  unset GEGE_GLOBAL_DEGREE_SAMPLING
  unset GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM
  unset GEGE_MEM_PARTITION_BUFFER_PINNED_HOST
}

enable_stack_through() {
  case "$1" in
    control_main_all_flags_off)
      ;;
    incremental_01_deg_chunk_exclusion)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      ;;
    incremental_02_active_edge_shuffle)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      ;;
    incremental_03_lp_fast_path)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      ;;
    incremental_04_fast_map_tensors)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_FAST_MAP_TENSORS=1
      ;;
    incremental_05_unique_backend_bitmap)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_FAST_MAP_TENSORS=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
      ;;
    incremental_06_optimized_custom_schedule)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_FAST_MAP_TENSORS=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
      export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
      ;;
    incremental_07_keep_storage_hot_between_epochs)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_FAST_MAP_TENSORS=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
      export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
      export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
      ;;
    incremental_08_partition_buffer_peer_relay)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_FAST_MAP_TENSORS=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
      export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
      export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
      export GEGE_PARTITION_BUFFER_PEER_RELAY=1
      ;;
    *)
      echo "unknown stack case: $1" >&2
      exit 1
      ;;
  esac
}

flags_string_for_case() {
  case "$1" in
    control_main_all_flags_off)
      printf '%s' 'GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; all optional stack flags off; fixed off env block above'
      ;;
    incremental_01_deg_chunk_exclusion)
      printf '%s' 'GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed; previous_stack=control + GEGE_DEG_CHUNK_EXCLUSION=1'
      ;;
    incremental_02_active_edge_shuffle)
      printf '%s' 'previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1'
      ;;
    incremental_03_lp_fast_path)
      printf '%s' 'previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1'
      ;;
    incremental_04_fast_map_tensors)
      printf '%s' 'previous_stack + GEGE_FAST_MAP_TENSORS=1'
      ;;
    incremental_05_unique_backend_bitmap)
      printf '%s' 'previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=4847571'
      ;;
    incremental_06_optimized_custom_schedule)
      printf '%s' 'previous_stack + GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1'
      ;;
    incremental_07_keep_storage_hot_between_epochs)
      printf '%s' 'previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1'
      ;;
    incremental_08_partition_buffer_peer_relay)
      printf '%s' 'previous_stack + GEGE_PARTITION_BUFFER_PEER_RELAY=1'
      ;;
    *)
      printf '%s' 'unknown'
      ;;
  esac
}

row_already_filled() {
  python3 - "$RESULTS_MD" "$1" <<'PY'
from pathlib import Path
import sys

md_path = Path(sys.argv[1])
row = sys.argv[2]
lines = md_path.read_text(encoding="utf-8").splitlines()

try:
    start = lines.index("## LiveJournal 16p")
except ValueError:
    print("0")
    raise SystemExit(0)

end = len(lines)
for idx in range(start + 1, len(lines)):
    if lines[idx].startswith("## "):
        end = idx
        break

needle = f"| `{row}` |"
for idx in range(start + 1, end):
    line = lines[idx]
    if line.startswith(needle):
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 20:
          print("0")
          raise SystemExit(0)
        avg_epoch = parts[5].replace("`", "").strip()
        train_log = parts[18].replace("`", "").strip()
        filled = avg_epoch not in {"", "n/a"} and train_log not in {"", "n/a"}
        print("1" if filled else "0")
        raise SystemExit(0)

print("0")
PY
}

configure_build_if_needed

for case_name in "${stack_cases[@]}"; do
  if [[ "$(row_already_filled "$case_name")" == "1" ]]; then
    echo "skipping completed row: $case_name"
    continue
  fi

  reset_env
  enable_stack_through "$case_name"

  run_name="livejournal_16p_main_${case_name}_${EPOCHS}ep_trainonly"
  config_tmp="${TMPDIR:-/tmp}/${run_name}.yaml"
  run_root="$OUTPUT_ROOT/$run_name"
  train_log="$LOG_DIR/${run_name}_train.log"
  summary_log="$LOG_DIR/${run_name}_summary.txt"

  cp "$CONFIG_SRC" "$config_tmp"

  perl -0pi -e "s|model_dir: .*|model_dir: $run_root|;
s|checkpoint_dir: .*|checkpoint_dir: $run_root|;
s|num_epochs: .*|num_epochs: $EPOCHS|;
s|^[[:space:]]*save_model: .*|  save_model: false|m;
s|^[[:space:]]*save_state: .*|    save_state: false|m;" \
    "$config_tmp"

  if [[ -n "$DATASET_DIR" ]]; then
    perl -0pi -e "s|dataset_dir: .*|dataset_dir: $DATASET_DIR|;" "$config_tmp"
  fi

  rm -rf "$run_root"
  rm -f "$train_log" "$summary_log"

  echo "=== $case_name ==="
  echo "config=$config_tmp"
  echo "epochs=$EPOCHS"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

  "$BUILD_DIR/gege/gege_train" "$config_tmp" |& tee "$train_log"

  python3 "$SUMMARY_SCRIPT" \
    --train-log "$train_log" \
    --epochs "$EPOCHS" \
    > "$summary_log"

  python3 "$UPDATE_SCRIPT" \
    --md "$RESULTS_MD" \
    --dataset livejournal_16p \
    --row "$case_name" \
    --branch main \
    --config "gege/configs/2gpu/livejournal_16p.yaml" \
    --flags "$(flags_string_for_case "$case_name")" \
    --epochs "$EPOCHS" \
    --train-log "experiment_logs/${run_name}_train.log" \
    --eval-log "n/a" \
    --eval-notes "train only; eval skipped" \
    --notes "LJ 2-GPU importance stack run via run_lj_2gpu_stack_ablation.sh with GEGE_EMULATE_DOT_SINGLE_RELATION=1 fixed across runs" \
    --summary "$summary_log"

  rm -rf "$run_root"
  rm -f "$config_tmp"
done

echo "done"
