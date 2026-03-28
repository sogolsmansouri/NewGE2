#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_ge2env_ge2py39}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dev/shm/smansou2_ge2/single_gpu_ablation_matrix}"
RESULTS_MD="${RESULTS_MD:-$REPO_ROOT/single_gpu_ablation_results_template.md}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$REPO_ROOT/scripts/summarize_benchmark_logs.py}"
UPDATE_SCRIPT="${UPDATE_SCRIPT:-$REPO_ROOT/scripts/update_single_gpu_ablation_table.py}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ ! -x "$BUILD_DIR/gege/gege_train" ]]; then
  echo "missing binary: $BUILD_DIR/gege/gege_train" >&2
  exit 1
fi

if [[ ! -f "$RESULTS_MD" ]]; then
  echo "missing markdown template: $RESULTS_MD" >&2
  exit 1
fi

cmake --build "$BUILD_DIR" -j --target gege_train >/dev/null

declare -A CONFIG_BY_DATASET=(
  [livejournal_16p]="$REPO_ROOT/gege/configs/single_gpu/livejournal_16p.yaml"
  [twitter_16p]="$REPO_ROOT/gege/configs/single_gpu/twitter_16p.yaml"
  [freebase86m_16p]="$REPO_ROOT/gege/configs/single_gpu/freebase86m_16p.yaml"
)

declare -A EPOCHS_BY_DATASET=(
  [livejournal_16p]=5
  [twitter_16p]=3
  [freebase86m_16p]=3
)

declare -A CONFIG_REL_BY_DATASET=(
  [livejournal_16p]="gege/configs/single_gpu/livejournal_16p.yaml"
  [twitter_16p]="gege/configs/single_gpu/twitter_16p.yaml"
  [freebase86m_16p]="gege/configs/single_gpu/freebase86m_16p.yaml"
)

declare -A BITMAP_NUM_NODES_BY_DATASET=(
  [livejournal_16p]=4847571
  [twitter_16p]=41652230
  [freebase86m_16p]=86054151
)

datasets=(livejournal_16p twitter_16p freebase86m_16p)

base_cases=(
  control_main_all_flags_off
  oneflag_fast_map_tensors
  oneflag_partition_buffer_lp_fast_path
  oneflag_gpu_active_edge_shuffle
  oneflag_keep_storage_hot_between_epochs
  oneflag_single_gpu_gpu_aware_custom
  oneflag_optimized_custom_schedule
  oneflag_deg_chunk_exclusion
  oneflag_global_degree_sampling
  oneflag_csr_gather
  oneflag_csr_update
  oneflag_bucket_streaming_lp
  oneflag_mem_partition_buffer_pinned_host
  oneflag_unique_backend_bitmap
)

social_extra_cases=(
  oneflag_emulate_dot_single_relation
)

reset_base_env() {
  local v
  while IFS= read -r v; do
    unset "$v"
  done < <(compgen -e | grep '^GEGE_' || true)

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
  unset GEGE_GLOBAL_DEGREE_SAMPLING
  unset GEGE_EMULATE_DOT_SINGLE_RELATION
  unset GEGE_UNIQUE_BACKEND
  unset GEGE_UNIQUE_BITMAP_NUM_NODES
}

flags_string_for_case() {
  local case_name="$1"
  local dataset="$2"
  local bitmap_nodes="${BITMAP_NUM_NODES_BY_DATASET[$dataset]}"
  case "$case_name" in
    control_main_all_flags_off)
      printf '%s' 'explicit all-flags-off env block; unset GEGE_GLOBAL_DEGREE_SAMPLING'
      ;;
    oneflag_fast_map_tensors)
      printf '%s' 'GEGE_FAST_MAP_TENSORS=1'
      ;;
    oneflag_partition_buffer_lp_fast_path)
      printf '%s' 'GEGE_PARTITION_BUFFER_LP_FAST_PATH=1'
      ;;
    oneflag_gpu_active_edge_shuffle)
      printf '%s' 'GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1'
      ;;
    oneflag_keep_storage_hot_between_epochs)
      printf '%s' 'GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1'
      ;;
    oneflag_single_gpu_gpu_aware_custom)
      printf '%s' 'GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1'
      ;;
    oneflag_optimized_custom_schedule)
      printf '%s' 'GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1'
      ;;
    oneflag_deg_chunk_exclusion)
      printf '%s' 'GEGE_DEG_CHUNK_EXCLUSION=1'
      ;;
    oneflag_global_degree_sampling)
      printf '%s' 'GEGE_GLOBAL_DEGREE_SAMPLING=1'
      ;;
    oneflag_csr_gather)
      printf '%s' 'GEGE_CSR_GATHER=1'
      ;;
    oneflag_csr_update)
      printf '%s' 'GEGE_CSR_UPDATE=1'
      ;;
    oneflag_bucket_streaming_lp)
      printf '%s' 'GEGE_BUCKET_STREAMING_LP=1'
      ;;
    oneflag_mem_partition_buffer_pinned_host)
      printf '%s' 'GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1'
      ;;
    oneflag_unique_backend_bitmap)
      printf '%s' "GEGE_UNIQUE_BACKEND=bitmap, GEGE_UNIQUE_BITMAP_NUM_NODES=${bitmap_nodes}"
      ;;
    oneflag_emulate_dot_single_relation)
      printf '%s' 'GEGE_EMULATE_DOT_SINGLE_RELATION=1'
      ;;
    *)
      printf '%s' 'unknown'
      ;;
  esac
}

enable_case() {
  local case_name="$1"
  local dataset="$2"
  case "$case_name" in
    control_main_all_flags_off)
      ;;
    oneflag_fast_map_tensors)
      export GEGE_FAST_MAP_TENSORS=1
      ;;
    oneflag_partition_buffer_lp_fast_path)
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      ;;
    oneflag_gpu_active_edge_shuffle)
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      ;;
    oneflag_keep_storage_hot_between_epochs)
      export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
      ;;
    oneflag_single_gpu_gpu_aware_custom)
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      ;;
    oneflag_optimized_custom_schedule)
      export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
      ;;
    oneflag_deg_chunk_exclusion)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      ;;
    oneflag_global_degree_sampling)
      export GEGE_GLOBAL_DEGREE_SAMPLING=1
      ;;
    oneflag_csr_gather)
      export GEGE_CSR_GATHER=1
      ;;
    oneflag_csr_update)
      export GEGE_CSR_UPDATE=1
      ;;
    oneflag_bucket_streaming_lp)
      export GEGE_BUCKET_STREAMING_LP=1
      ;;
    oneflag_mem_partition_buffer_pinned_host)
      export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
      ;;
    oneflag_unique_backend_bitmap)
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES="${BITMAP_NUM_NODES_BY_DATASET[$dataset]}"
      ;;
    oneflag_emulate_dot_single_relation)
      export GEGE_EMULATE_DOT_SINGLE_RELATION=1
      ;;
    *)
      echo "unknown case: $case_name" >&2
      return 1
      ;;
  esac
}

row_already_filled() {
  python3 - "$RESULTS_MD" "$1" "$2" <<'PY'
from pathlib import Path
import sys

md_path = Path(sys.argv[1])
dataset = sys.argv[2]
row = sys.argv[3]
section_by_dataset = {
    "livejournal_16p": "## LiveJournal 16p",
    "twitter_16p": "## Twitter 16p",
    "freebase86m_16p": "## Freebase86M 16p",
}
lines = md_path.read_text(encoding="utf-8").splitlines()
section = section_by_dataset[dataset]
try:
    start = lines.index(section)
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
        avg_epoch = parts[5]
        train_log = parts[18]
        filled = avg_epoch not in {"", "` `"} and train_log not in {"", "` `"}
        print("1" if filled else "0")
        raise SystemExit(0)
print("0")
PY
}

for dataset in "${datasets[@]}"; do
  config_src="${CONFIG_BY_DATASET[$dataset]}"
  config_rel="${CONFIG_REL_BY_DATASET[$dataset]}"
  epochs="${EPOCHS_BY_DATASET[$dataset]}"
  if [[ ! -f "$config_src" ]]; then
    echo "missing config for $dataset: $config_src" >&2
    exit 1
  fi

  cases=("${base_cases[@]}")
  if [[ "$dataset" == "livejournal_16p" || "$dataset" == "twitter_16p" ]]; then
    cases+=("${social_extra_cases[@]}")
  fi

  for case_name in "${cases[@]}"; do
    if [[ "$(row_already_filled "$dataset" "$case_name")" == "1" ]]; then
      echo "[$(date '+%F %T')] SKIP dataset=$dataset case=$case_name already recorded in markdown"
      continue
    fi

    reset_base_env
    enable_case "$case_name" "$dataset"

    run_name="${dataset}_main_${case_name}_${epochs}ep_trainonly"
    config_tmp="${TMPDIR:-/tmp}/${run_name}.yaml"
    run_root="$OUTPUT_ROOT/$run_name"
    train_log="$LOG_DIR/${run_name}_train.log"
    summary_log="$LOG_DIR/${run_name}_summary.txt"
    if [[ "$train_log" == "$REPO_ROOT/"* ]]; then
      train_log_rel="${train_log#$REPO_ROOT/}"
    else
      train_log_rel="$train_log"
    fi

    cp "$config_src" "$config_tmp"
    python3 - <<PY
from pathlib import Path
p = Path("$config_tmp")
s = p.read_text()
replacements = {
    "model_dir: /dev/shm/smansou2_ge2/config_matrix/main/single_gpu/${dataset}": "model_dir: $run_root",
    "checkpoint_dir: /dev/shm/smansou2_ge2/config_matrix/main/single_gpu/${dataset}": "checkpoint_dir: $run_root",
}
for old, new in replacements.items():
    s = s.replace(old, new)
s = s.replace("num_epochs: 15", "num_epochs: $epochs")
s = s.replace("num_epochs: 10", "num_epochs: $epochs")
s = s.replace("  save_model: true", "  save_model: false")
s = s.replace("    save_state: true", "    save_state: false")
s = s.replace("    save_best: true", "    save_best: false")
p.write_text(s)
PY

    rm -rf "$run_root"
    rm -f "$train_log" "$summary_log"

    echo "[$(date '+%F %T')] START dataset=$dataset case=$case_name epochs=$epochs"
    echo "[$(date '+%F %T')] config=$config_tmp"
    if "$BUILD_DIR/gege/gege_train" "$config_tmp" |& tee "$train_log"; then
      python3 "$SUMMARY_SCRIPT" \
        --train-log "$train_log" \
        --eval-log /dev/null \
        --epochs "$epochs" > "$summary_log"

      notes="${epochs}-epoch ${dataset} single-GPU train-only overnight run via run_single_gpu_train_only_ablation_matrix.sh ($(date '+%Y-%m-%d'))"
      python3 "$UPDATE_SCRIPT" \
        --md "$RESULTS_MD" \
        --dataset "$dataset" \
        --row "$case_name" \
        --config "$config_rel" \
        --flags "$(flags_string_for_case "$case_name" "$dataset")" \
        --epochs "$epochs" \
        --train-log "$train_log_rel" \
        --eval-log n/a \
        --eval-notes "train only; eval skipped" \
        --notes "$notes" \
        --summary "$summary_log"
      echo "[$(date '+%F %T')] DONE dataset=$dataset case=$case_name"
    else
      echo "[$(date '+%F %T')] FAIL dataset=$dataset case=$case_name" >&2
    fi
  done
done
