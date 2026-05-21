#!/usr/bin/env bash
set -euo pipefail

# ARC/c32 reproduction wrapper for the single-GPU Freebase86M p32 q4 fast path.
# It stages only per-run config/logs, writes the checkpoint under /dev/shm, and
# then runs the paper-10k exact filtered eval through run_freebase86m_eval.sh.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="${GEGE_RUN_NAME:-fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_${TIMESTAMP}}"
LOG_DIR="${GEGE_LOG_DIR:-/mnt/local/${USER}/ge2_logs/${RUN_NAME}}"
RUN_ROOT="${GEGE_RUN_ROOT:-/dev/shm/${USER}_ge2/${RUN_NAME}}"
PY_ENV="${GEGE_PY_ENV:-/mnt/local/${USER}/ge2_py312_torch_mirror}"
TRAIN_DS="${GEGE_TRAIN_DS:-/mnt/local/${USER}/datasets/freebase86m_32p}"
EVAL_DS="${GEGE_EVAL_DS:-/mnt/local/${USER}/datasets/freebase86m_16p_paper_10k_eval}"
CONFIG_TEMPLATE="${GEGE_CONFIG_TEMPLATE:-${ROOT}/gege/configs/repro/freebase86m_32p_epoch158_20260428.yaml}"
CONFIG_SRC="${LOG_DIR}/${RUN_NAME}_source.yaml"

TRAIN_BIN="${GEGE_TRAIN_BIN:-}"
if [[ -z "$TRAIN_BIN" ]]; then
  for candidate in \
    "${ROOT}/gege/build-cuda-samecode-arc-cu129-sysgcc-sm86/gege_train" \
    "${ROOT}/gege/build-cuda-check-arc-sm86/gege_train" \
    "${ROOT}/gege/build-cuda-check/gege_train"; do
    if [[ -x "$candidate" ]]; then
      TRAIN_BIN="$candidate"
      break
    fi
  done
fi

EVAL_BIN="${GEGE_EVAL_BIN:-}"
if [[ -z "$EVAL_BIN" ]]; then
  for candidate in \
    "${ROOT}/gege/build-cuda-samecode-arc-cu129-sysgcc-sm86/gege_eval" \
    "${ROOT}/gege/build-cuda-check-arc-sm86/gege_eval" \
    "${ROOT}/gege/build-cuda-check/gege_eval"; do
    if [[ -x "$candidate" ]]; then
      EVAL_BIN="$candidate"
      break
    fi
  done
fi

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 2
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "Missing required executable: $1" >&2
    exit 2
  fi
}

require_path "$PY_ENV"
require_path "$TRAIN_DS"
require_path "$EVAL_DS"
require_path "$CONFIG_TEMPLATE"
require_executable "$TRAIN_BIN"
require_executable "$EVAL_BIN"

mkdir -p "$LOG_DIR" "$RUN_ROOT" "$ROOT/datasets"
ln -sfn "$EVAL_DS" "$ROOT/datasets/freebase86m_16p_paper_10k_eval"

cp "$CONFIG_TEMPLATE" "$CONFIG_SRC"
export GEGE_TRAIN_DS="$TRAIN_DS"
perl -0pi -e 's#(^\s*dataset_dir:\s*).*$#$1$ENV{GEGE_TRAIN_DS}/#m' "$CONFIG_SRC"
perl -0pi -e 's#(^  device_ids:\n)(?:[ \t]*-[^\n]*\n)+#$1  - 0\n#m' "$CONFIG_SRC"
perl -0pi -e 's#(^\s*logical_active_devices:\s*)\d+#${1}1#m' "$CONFIG_SRC"

if [[ -f "$EVAL_DS/dataset.yaml" ]]; then
  EVAL_DS="$EVAL_DS" perl -0pi -e 's#(^dataset_dir:\s*).*$#$1$ENV{EVAL_DS}/#m' "$EVAL_DS/dataset.yaml"
fi

export PATH="${PY_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export GEGE_TRAIN_BIN="$TRAIN_BIN"
export GEGE_EVAL_BIN="$EVAL_BIN"
export GEGE_CONFIG_SRC="$CONFIG_SRC"
export GEGE_LOG_DIR="$LOG_DIR"
export GEGE_RUN_ROOT="$RUN_ROOT"
export GEGE_RUN_NAME="$RUN_NAME"
export GEGE_EPOCHS="${GEGE_EPOCHS:-10}"
export GEGE_RUN_EVAL=1
export GEGE_SAVE_MODEL=true

export GEGE_BOUNDED_GREEDY_COVER_Q4="${GEGE_BOUNDED_GREEDY_COVER_Q4:-1}"
export GEGE_BOUNDED_GREEDY_COVER_REVERSE="${GEGE_BOUNDED_GREEDY_COVER_REVERSE:-0}"
export GEGE_FRAME_CACHE_HIDDEN_FRAMES="${GEGE_FRAME_CACHE_HIDDEN_FRAMES:-6}"
export GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD="${GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD:-1}"
export GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK="${GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK:-1}"
export GEGE_FRAME_CACHE_MAX_STALE_BACKLOG="${GEGE_FRAME_CACHE_MAX_STALE_BACKLOG:-3}"
export GEGE_FRAME_CACHE_PRIORITIZED_WRITEBACK="${GEGE_FRAME_CACHE_PRIORITIZED_WRITEBACK:-1}"
export GEGE_FRAME_CACHE_SERIALIZE_ADMIT_H2D="${GEGE_FRAME_CACHE_SERIALIZE_ADMIT_H2D:-1}"
export GEGE_MEM_PARTITION_BUFFER_ASYNC_EVICT_MAX_IN_FLIGHT="${GEGE_MEM_PARTITION_BUFFER_ASYNC_EVICT_MAX_IN_FLIGHT:-2}"
export GEGE_MINMAX_BUCKET_ASSIGNMENT="${GEGE_MINMAX_BUCKET_ASSIGNMENT:-1}"
export GEGE_DEG_CHUNK_EXCLUSION="${GEGE_DEG_CHUNK_EXCLUSION:-0}"
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE="${GEGE_GPU_ACTIVE_EDGE_SHUFFLE:-1}"
export GEGE_PARTITION_BUFFER_LP_FAST_PATH="${GEGE_PARTITION_BUFFER_LP_FAST_PATH:-1}"
export GEGE_FAST_MAP_TENSORS="${GEGE_FAST_MAP_TENSORS:-1}"
export GEGE_FIXED_BUFFER_BITMAP_MAP="${GEGE_FIXED_BUFFER_BITMAP_MAP:-1}"
export GEGE_FIXED_BUFFER_BITMAP_REUSE_OUTPUTS="${GEGE_FIXED_BUFFER_BITMAP_REUSE_OUTPUTS:-1}"
export GEGE_FIXED_BUFFER_MASKED_UPDATE="${GEGE_FIXED_BUFFER_MASKED_UPDATE:-1}"
export GEGE_SOFTMAX_NEGATIVE_MASS_SCALE="${GEGE_SOFTMAX_NEGATIVE_MASS_SCALE:-8}"
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS="${GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS:-1}"
export GEGE_PREPARED_BATCH_PIPELINE="${GEGE_PREPARED_BATCH_PIPELINE:-0}"
export GEGE_PREFETCH_PREPARE_NEXT_PARTITION="${GEGE_PREFETCH_PREPARE_NEXT_PARTITION:-0}"
export GEGE_BATCHED_NEGATIVE_PLAN_BATCHES="${GEGE_BATCHED_NEGATIVE_PLAN_BATCHES:-0}"
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING="${GEGE_PARTITION_BUFFER_PIPELINE_TIMING:-0}"
export GEGE_EVAL_BATCH_SIZE="${GEGE_EVAL_BATCH_SIZE:-250}"
export GEGE_EVAL_NEGATIVE_CHUNK_SIZE="${GEGE_EVAL_NEGATIVE_CHUNK_SIZE:-32768}"
export GEGE_UNIQUE_BITMAP_NUM_NODES="${GEGE_UNIQUE_BITMAP_NUM_NODES:-86054151}"

{
  echo "NODE=$(hostname)"
  echo "RUN_NAME=$RUN_NAME"
  echo "ROOT=$ROOT"
  echo "PY_ENV=$PY_ENV"
  echo "TRAIN_BIN=$TRAIN_BIN"
  echo "EVAL_BIN=$EVAL_BIN"
  echo "TRAIN_DS=$TRAIN_DS"
  echo "EVAL_DS=$EVAL_DS"
  echo "CONFIG_SRC=$CONFIG_SRC"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "LOG_DIR=$LOG_DIR"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  df -h /dev/shm /mnt/local || true
  nvidia-smi || true
} | tee "${LOG_DIR}/${RUN_NAME}_driver.log"

bash "$ROOT/scripts/run_epoch_time_repro_20260424.sh" fb-10e-eval 2>&1 | tee -a "${LOG_DIR}/${RUN_NAME}_driver.log"
