#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_ge2env_ge2py39}"
CONFIG_SRC="${CONFIG_SRC:-$REPO_ROOT/gege/configs/single_gpu/freebase86m_16p.yaml}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dev/shm/smansou2_ge2/fb86m_single_gpu_stack_ablation}"
RESULTS_MD="${RESULTS_MD:-$REPO_ROOT/single_gpu_ablation_results_template.md}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$REPO_ROOT/scripts/summarize_benchmark_logs.py}"
UPDATE_SCRIPT="${UPDATE_SCRIPT:-$REPO_ROOT/scripts/update_single_gpu_ablation_table.py}"
EPOCHS="${EPOCHS:-3}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ ! -x "$BUILD_DIR/gege/gege_train" ]]; then
  echo "missing binary: $BUILD_DIR/gege/gege_train" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_SRC" ]]; then
  echo "missing config: $CONFIG_SRC" >&2
  exit 1
fi

cmake --build "$BUILD_DIR" -j --target gege_train >/dev/null

stack_cases=(
  stack_00_control
  stack_01_plus_deg_chunk_exclusion
  stack_02_plus_gpu_active_edge_shuffle
  stack_03_plus_single_gpu_gpu_aware_custom
  stack_04_plus_partition_buffer_lp_fast_path
  stack_05_plus_mem_partition_buffer_pinned_host
  stack_06_plus_bitmap_unique
  stack_07_plus_fast_map_tensors
  stack_08_plus_keep_storage_hot_between_epochs
)

reset_env() {
  local v
  while IFS= read -r v; do
    unset "$v"
  done < <(compgen -e | grep '^GEGE_' || true)

  export CUDA_VISIBLE_DEVICES=0

  export GEGE_BUCKET_STREAMING_LP=0
  export GEGE_CSR_GATHER=0
  export GEGE_CSR_UPDATE=0
  export GEGE_CSR_DEBUG=0

  export GEGE_DEG_CHUNK_EXCLUSION=0
  export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=0
  export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=0
  export GEGE_PARTITION_BUFFER_LP_FAST_PATH=0
  export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=0
  export GEGE_UNIQUE_BACKEND=sort
  unset GEGE_UNIQUE_BITMAP_NUM_NODES
  export GEGE_FAST_MAP_TENSORS=0
  export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=0
  unset GEGE_GLOBAL_DEGREE_SAMPLING
  unset GEGE_EMULATE_DOT_SINGLE_RELATION
}

enable_stack_through() {
  case "$1" in
    stack_00_control)
      ;;
    stack_01_plus_deg_chunk_exclusion)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      ;;
    stack_02_plus_gpu_active_edge_shuffle)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      ;;
    stack_03_plus_single_gpu_gpu_aware_custom)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      ;;
    stack_04_plus_partition_buffer_lp_fast_path)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      ;;
    stack_05_plus_mem_partition_buffer_pinned_host)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
      ;;
    stack_06_plus_bitmap_unique)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
      ;;
    stack_07_plus_fast_map_tensors)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
      export GEGE_FAST_MAP_TENSORS=1
      ;;
    stack_08_plus_keep_storage_hot_between_epochs)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
      export GEGE_FAST_MAP_TENSORS=1
      export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
      ;;
    *)
      echo "unknown stack case: $1" >&2
      exit 1
      ;;
  esac
}

flags_string_for_case() {
  case "$1" in
    stack_00_control)
      printf '%s' 'all stack flags off; fixed off env block above'
      ;;
    stack_01_plus_deg_chunk_exclusion)
      printf '%s' 'previous_stack + GEGE_DEG_CHUNK_EXCLUSION=1'
      ;;
    stack_02_plus_gpu_active_edge_shuffle)
      printf '%s' 'previous_stack + GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1'
      ;;
    stack_03_plus_single_gpu_gpu_aware_custom)
      printf '%s' 'previous_stack + GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1'
      ;;
    stack_04_plus_partition_buffer_lp_fast_path)
      printf '%s' 'previous_stack + GEGE_PARTITION_BUFFER_LP_FAST_PATH=1'
      ;;
    stack_05_plus_mem_partition_buffer_pinned_host)
      printf '%s' 'previous_stack + GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1'
      ;;
    stack_06_plus_bitmap_unique)
      printf '%s' 'previous_stack + GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=86054151'
      ;;
    stack_07_plus_fast_map_tensors)
      printf '%s' 'previous_stack + GEGE_FAST_MAP_TENSORS=1'
      ;;
    stack_08_plus_keep_storage_hot_between_epochs)
      printf '%s' 'previous_stack + GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1'
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
    start = lines.index("## Freebase86M 16p")
except ValueError:
    print("0")
    raise SystemExit(0)

end = len(lines)
for idx in range(start + 1, len(lines)):
    if lines[idx].startswith("## "):
        end = idx
        break

header_idx = None
for idx in range(start + 1, end):
    if lines[idx].startswith("| ") and "Branch" in lines[idx] and "Avg Epoch Runtime" in lines[idx]:
        header_idx = idx
        break

if header_idx is None:
    print("0")
    raise SystemExit(0)

headers = [p.strip() for p in lines[header_idx].split("|")[1:-1]]
marker = f"<!-- row: {row} -->"
needle = f"| `{row}` |"
for idx in range(header_idx + 2, end):
    line = lines[idx]
    if marker in line or line.startswith(needle):
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if "Avg Epoch Runtime" not in headers:
            print("0")
            raise SystemExit(0)
        avg_epoch = parts[headers.index("Avg Epoch Runtime")].replace("`", "").strip().lower()
        filled = avg_epoch not in {"", "n/a"}
        print("1" if filled else "0")
        raise SystemExit(0)

print("0")
PY
}

for case_name in "${stack_cases[@]}"; do
  if [[ "$(row_already_filled "$case_name")" == "1" ]]; then
    echo "skipping completed row: $case_name"
    continue
  fi

  reset_env
  enable_stack_through "$case_name"

  run_name="freebase86m_16p_main_${case_name}_${EPOCHS}ep_trainonly"
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

  rm -rf "$run_root"
  rm -f "$train_log" "$summary_log"

  echo "=== $case_name ==="
  echo "config=$config_tmp"
  echo "epochs=$EPOCHS"

  "$BUILD_DIR/gege/gege_train" "$config_tmp" |& tee "$train_log"

  python3 "$SUMMARY_SCRIPT" \
    --train-log "$train_log" \
    --epochs "$EPOCHS" \
    > "$summary_log"

  python3 "$UPDATE_SCRIPT" \
    --md "$RESULTS_MD" \
    --dataset freebase86m_16p \
    --row "$case_name" \
    --branch main \
    --config "gege/configs/single_gpu/freebase86m_16p.yaml" \
    --flags "$(flags_string_for_case "$case_name")" \
    --epochs "$EPOCHS" \
    --train-log "experiment_logs/${run_name}_train.log" \
    --eval-log "n/a" \
    --eval-notes "train only; eval skipped" \
    --notes "3-epoch FB86M single-GPU train-only importance stack run via run_fb86m_single_gpu_stack_ablation.sh" \
    --summary "$summary_log"

  rm -rf "$run_root"
  rm -f "$config_tmp"
done

echo "done"
