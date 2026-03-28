#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_ge2env}"
CONFIG_SRC="${CONFIG_SRC:-$REPO_ROOT/gege/configs/single_gpu/livejournal_16p.yaml}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/experiment_logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_outputs}"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ ! -x "$BUILD_DIR/gege/gege_train" ]]; then
  echo "missing binary: $BUILD_DIR/gege/gege_train" >&2
  echo "build first with: cmake --build \"$BUILD_DIR\" -j --target gege_train" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_SRC" ]]; then
  echo "missing config: $CONFIG_SRC" >&2
  exit 1
fi

EPOCHS_SRC="$(grep -E '^[[:space:]]*num_epochs:' "$CONFIG_SRC" | head -n 1 | awk '{print $2}')"
if [[ -z "$EPOCHS_SRC" ]]; then
  echo "unable to detect num_epochs from $CONFIG_SRC" >&2
  exit 1
fi
DEFAULT_EPOCHS="${DEFAULT_EPOCHS:-5}"
DOTEMU_EPOCHS_DEFAULT="${DOTEMU_EPOCHS_DEFAULT:-10}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"

default_cases=(
  fast_map_tensors
  partition_buffer_lp_fast_path
  gpu_active_edge_shuffle
  keep_storage_hot_between_epochs
  single_gpu_gpu_aware_custom
  optimized_custom_schedule
  deg_chunk_exclusion
  global_degree_sampling
  csr_gather
  csr_update
  bucket_streaming_lp
)

RESULTS_MD="${RESULTS_MD:-$REPO_ROOT/single_gpu_ablation_results_template.md}"
SKIP_COMPLETED_FROM_MD="${SKIP_COMPLETED_FROM_MD:-1}"

completed_cases_from_md() {
  [[ -f "$RESULTS_MD" ]] || return 0

  awk -F'|' '
    function trim(s) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
      return s
    }
    /^## LiveJournal 16p/ { in_lj=1; next }
    /^## / { if (in_lj) exit }
    in_lj && $0 ~ /^\| Row \|/ {
      avg_col = 0
      for (i = 1; i <= NF; i++) {
        header = trim($i)
        if (header == "Avg Epoch Runtime") {
          avg_col = i
          break
        }
      }
      if (avg_col == 0) {
        avg_col = 6
      }
    }
    in_lj && $0 ~ /^\| `oneflag_/ {
      row = $2
      gsub(/^[[:space:]]*`|`[[:space:]]*$/, "", row)
      value = (avg_col > 0 && avg_col <= NF) ? $avg_col : ""
      value = trim(value)
      gsub(/`/, "", value)
      if (value != "" && value != "n/a") {
        sub(/^oneflag_/, "", row)
        print row
      }
    }
  ' "$RESULTS_MD" | sort -u
}

if [[ "$#" -gt 0 ]]; then
  cases=("$@")
else
  cases=("${default_cases[@]}")
fi

if [[ "$SKIP_COMPLETED_FROM_MD" == "1" ]]; then
  mapfile -t completed_cases < <(completed_cases_from_md || true)
  if [[ "${#completed_cases[@]}" -gt 0 ]]; then
    filtered_cases=()
    for case_name in "${cases[@]}"; do
      skip=0
      for completed in "${completed_cases[@]}"; do
        if [[ "$case_name" == "$completed" ]]; then
          skip=1
          break
        fi
      done
      if [[ "$skip" -eq 0 ]]; then
        filtered_cases+=("$case_name")
      fi
    done
    cases=("${filtered_cases[@]}")
  fi
fi

if [[ "${#cases[@]}" -eq 0 ]]; then
  echo "no remaining cases to run"
  exit 0
fi

reset_base_env() {
  local v
  for v in $(compgen -e | grep '^GEGE_'); do
    unset "$v"
  done

  export CUDA_VISIBLE_DEVICES=0
  export GEGE_EMULATE_DOT_SINGLE_RELATION=1

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
}

enable_case() {
  case "$1" in
    fast_map_tensors)
      export GEGE_FAST_MAP_TENSORS=1
      ;;
    partition_buffer_lp_fast_path)
      export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
      ;;
    gpu_active_edge_shuffle)
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      ;;
    keep_storage_hot_between_epochs)
      export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
      ;;
    single_gpu_gpu_aware_custom)
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      ;;
    optimized_custom_schedule)
      export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
      ;;
    deg_chunk_exclusion)
      export GEGE_DEG_CHUNK_EXCLUSION=1
      ;;
    global_degree_sampling)
      export GEGE_GLOBAL_DEGREE_SAMPLING=1
      ;;
    csr_gather)
      export GEGE_CSR_GATHER=1
      ;;
    csr_update)
      export GEGE_CSR_UPDATE=1
      ;;
    bucket_streaming_lp)
      export GEGE_BUCKET_STREAMING_LP=1
      ;;
    dotemu_singlegpu_repro)
      export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
      export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
      export GEGE_FAST_MAP_TENSORS=1
      export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
      export GEGE_UNIQUE_BACKEND=bitmap
      export GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
      ;;
    *)
      echo "unknown case: $1" >&2
      exit 1
      ;;
  esac
}

for case_name in "${cases[@]}"; do
  reset_base_env
  enable_case "$case_name"

  config_src_case="$CONFIG_SRC"
  case_epochs_default="$DEFAULT_EPOCHS"
  case "$case_name" in
    dotemu_singlegpu_repro)
      config_src_case="$REPO_ROOT/gege/configs/livejournal_16p_paper_dotemu_opt.yaml"
      case_epochs_default="$DOTEMU_EPOCHS_DEFAULT"
      ;;
  esac

  if [[ -n "$EPOCHS_OVERRIDE" ]]; then
    EPOCHS="$EPOCHS_OVERRIDE"
  else
    EPOCHS="$case_epochs_default"
  fi

  if [[ ! -f "$config_src_case" ]]; then
    echo "missing config: $config_src_case" >&2
    exit 1
  fi

  run_name="livejournal_16p_main_${case_name}_${EPOCHS}ep_trainonly"
  config_tmp="${TMPDIR:-/tmp}/${run_name}.yaml"
  run_root="$OUTPUT_ROOT/$run_name"
  train_log="$LOG_DIR/${run_name}_train.log"
  summary_log="$LOG_DIR/${run_name}_summary.txt"

  cp "$config_src_case" "$config_tmp"

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

  python "$REPO_ROOT/scripts/summarize_benchmark_logs.py" \
    --train-log "$train_log" \
    --eval-log /dev/null \
    --epochs "$EPOCHS" | tee "$summary_log"
done
