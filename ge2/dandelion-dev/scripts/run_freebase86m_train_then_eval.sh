#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_BIN="/home/smansou2/miniconda/envs/ge2/bin/gege_train"
TRAIN_TEMPLATE="$ROOT/gege/configs/freebase86m_16p_paper_opt.yaml"
EVAL_HELPER="$ROOT/scripts/run_freebase86m_eval.sh"
RUNTIME_ROOT="${GEGE_RUNTIME_ROOT:-/dev/shm/smansou2_ge2/paper_single_gpu}"
CHECKPOINT_ROOT="/mnt/jli256/smansou2_ge2/checkpoints"
LOGDIR="/tmp"
MIN_RUNTIME_FREE_GB="${GEGE_MIN_RUNTIME_FREE_GB:-70}"
EVAL_DATASET_MODE="${GEGE_EVAL_DATASET_MODE:-paper-10k}"
SKIP_ARCHIVE="${GEGE_SKIP_ARCHIVE:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%m/%d/%y %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    log "Missing required file: $path"
    exit 1
  fi
}

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    log "Missing required executable: $path"
    exit 1
  fi
}

require_executable "$TRAIN_BIN"
require_file "$TRAIN_TEMPLATE"
require_file "$EVAL_HELPER"

case "$EVAL_DATASET_MODE" in
  paper-10k|full)
    ;;
  *)
    log "Unsupported GEGE_EVAL_DATASET_MODE: $EVAL_DATASET_MODE"
    log "Supported values: paper-10k, full"
    exit 1
    ;;
esac

runtime_df_target() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf '%s\n' "$path"
  else
    dirname "$path"
  fi
}

runtime_free_gb() {
  local target
  target="$(runtime_df_target "$1")"
  df -Pk "$target" | awk 'NR==2 {printf "%.1f", $4/1024/1024}'
}

require_runtime_space() {
  local target="$1"
  local min_gb="$2"
  local free_gb
  free_gb="$(runtime_free_gb "$target")"

  if awk -v free="$free_gb" -v min="$min_gb" 'BEGIN { exit !(free < min) }'; then
    log "Insufficient free space under runtime root: $target"
    log "Free space: ${free_gb}G; required minimum: ${min_gb}G"
    log "Set GEGE_RUNTIME_ROOT to a larger filesystem or remove old /dev/shm GE2 runtime directories."
    if [[ -d "$target" ]]; then
      log "Current runtime root contents:"
      du -sh "$target"/* 2>/dev/null | sort -h || true
    fi
    exit 1
  fi
}

copy_checkpoint_dir() {
  local src_dir="$1"
  local dst_dir="$2"

  log "Archiving checkpoint from $src_dir to $dst_dir"
  if [[ -d "$src_dir" ]]; then
    log "Checkpoint size summary:"
    find "$src_dir" -maxdepth 1 -type f -printf '%s %f\n' 2>/dev/null | sort -nr | head -n 8 || true
  fi

  mkdir -p "$dst_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --partial --inplace --human-readable --info=progress2,stats2 "$src_dir"/ "$dst_dir"/
  else
    log "rsync not found; falling back to cp -a without progress output"
    cp -a "$src_dir"/. "$dst_dir"/
  fi
}

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_NAME="freebase86m_16p_opt_${TIMESTAMP}"
RUNTIME_DIR="${RUNTIME_ROOT}/${RUN_NAME}"
SAVE_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
META_DIR="${SAVE_DIR}_meta"
TRAIN_CONFIG="${LOGDIR}/${RUN_NAME}_train.yaml"
TRAIN_LOG="${LOGDIR}/${RUN_NAME}_train.log"
EVAL_LABEL="${EVAL_DATASET_MODE}-exact"
EVAL_CONFIG="${LOGDIR}/${RUN_NAME}_eval_${EVAL_LABEL}.yaml"
EVAL_LOG="${LOGDIR}/${RUN_NAME}_eval_${EVAL_LABEL}.log"
EVAL_DRIVER_LOG="${LOGDIR}/${RUN_NAME}_eval_driver.log"

mkdir -p "$RUNTIME_ROOT" "$CHECKPOINT_ROOT" "$META_DIR"
require_runtime_space "$RUNTIME_ROOT" "$MIN_RUNTIME_FREE_GB"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GEGE_PARTITION_BUFFER_LP_FAST_PATH="${GEGE_PARTITION_BUFFER_LP_FAST_PATH:-1}"
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING="${GEGE_PARTITION_BUFFER_PIPELINE_TIMING:-1}"
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX="${GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX:-4096}"
export GEGE_PARTITION_BUFFER_SWAP_TIMING="${GEGE_PARTITION_BUFFER_SWAP_TIMING:-1}"
export GEGE_PARTITION_BUFFER_SWAP_TIMING_MAX="${GEGE_PARTITION_BUFFER_SWAP_TIMING_MAX:-4096}"
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM="${GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM:-1}"
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS="${GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS:-1}"
export GEGE_FAST_MAP_TENSORS="${GEGE_FAST_MAP_TENSORS:-1}"
export GEGE_BUCKET_STREAMING_LP="${GEGE_BUCKET_STREAMING_LP:-1}"
export GEGE_CSR_GATHER="${GEGE_CSR_GATHER:-1}"
export GEGE_CSR_UPDATE="${GEGE_CSR_UPDATE:-1}"
export GEGE_CSR_UPDATE_REDUCE="${GEGE_CSR_UPDATE_REDUCE:-0}"
export GEGE_CSR_DEBUG="${GEGE_CSR_DEBUG:-0}"
export GEGE_STAGE_DEBUG="${GEGE_STAGE_DEBUG:-0}"
export GEGE_DEG_CHUNK_EXCLUSION="${GEGE_DEG_CHUNK_EXCLUSION:-1}"
export GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES="${GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES:-0}"
export GEGE_UNIQUE_BACKEND="${GEGE_UNIQUE_BACKEND:-bitmap}"
export GEGE_UNIQUE_BITMAP_NUM_NODES="${GEGE_UNIQUE_BITMAP_NUM_NODES:-86054151}"
export GEGE_UNIQUE_LOG="${GEGE_UNIQUE_LOG:-0}"

sed \
  -e "s#^  model_dir:.*#  model_dir: ${RUNTIME_DIR}/#" \
  -e "s#^  checkpoint_dir:.*#  checkpoint_dir: ${RUNTIME_DIR}/#" \
  -e "s#^  num_epochs:.*#  num_epochs: 10#" \
  -e "s#^  save_model:.*#  save_model: true#" \
  "$TRAIN_TEMPLATE" > "$TRAIN_CONFIG"

cp -f "$TRAIN_CONFIG" "${META_DIR}/train_config.yaml"
printf '%s\n' "$RUNTIME_DIR" > "${META_DIR}/runtime_checkpoint_dir.txt"
printf '%s\n' "$SAVE_DIR" > "${META_DIR}/persistent_checkpoint_dir.txt"

log "Starting Freebase86M training from scratch"
log "Train config: $TRAIN_CONFIG"
log "Train log: $TRAIN_LOG"
log "Runtime root: $RUNTIME_ROOT"
log "Runtime checkpoint dir: $RUNTIME_DIR"
log "Eval dataset mode: $EVAL_DATASET_MODE"
log "Skip archive: $SKIP_ARCHIVE"
log "Flags: GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=$GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=$GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS"
log "Flags: GEGE_PARTITION_BUFFER_LP_FAST_PATH=$GEGE_PARTITION_BUFFER_LP_FAST_PATH GEGE_BUCKET_STREAMING_LP=$GEGE_BUCKET_STREAMING_LP GEGE_FAST_MAP_TENSORS=$GEGE_FAST_MAP_TENSORS"
log "Flags: GEGE_CSR_GATHER=$GEGE_CSR_GATHER GEGE_CSR_UPDATE=$GEGE_CSR_UPDATE GEGE_CSR_UPDATE_REDUCE=$GEGE_CSR_UPDATE_REDUCE GEGE_DEG_CHUNK_EXCLUSION=$GEGE_DEG_CHUNK_EXCLUSION"
log "Flags: GEGE_PARTITION_BUFFER_SWAP_TIMING=$GEGE_PARTITION_BUFFER_SWAP_TIMING GEGE_PARTITION_BUFFER_PIPELINE_TIMING=$GEGE_PARTITION_BUFFER_PIPELINE_TIMING"
set +e
"$TRAIN_BIN" "$TRAIN_CONFIG" 2>&1 | tee "$TRAIN_LOG"
train_status=${PIPESTATUS[0]}
set -e
log "Training exit status: $train_status"
if [[ "$train_status" -ne 0 ]]; then
  cp -f "$TRAIN_LOG" "${META_DIR}/$(basename "$TRAIN_LOG")"
  exit "$train_status"
fi

cp -f "$TRAIN_LOG" "${META_DIR}/$(basename "$TRAIN_LOG")"

log "Starting Freebase86M paper-comparable filtered eval (10k exact)"
log "Eval helper: $EVAL_HELPER"
log "Eval config: $EVAL_CONFIG"
log "Eval log: $EVAL_LOG"
log "Eval checkpoint dir: $RUNTIME_DIR"
set +e
bash "$EVAL_HELPER" \
  --checkpoint-dir "$RUNTIME_DIR" \
  --dataset-mode "$EVAL_DATASET_MODE" \
  --config-out "$EVAL_CONFIG" \
  --log-out "$EVAL_LOG" 2>&1 | tee "$EVAL_DRIVER_LOG"
eval_status=${PIPESTATUS[0]}
set -e
log "Eval exit status: $eval_status"

case "$SKIP_ARCHIVE" in
  1|true|TRUE|yes|YES)
    copy_status=0
    log "Skipping checkpoint archive because GEGE_SKIP_ARCHIVE=$SKIP_ARCHIVE"
    ;;
  *)
    set +e
    copy_checkpoint_dir "$RUNTIME_DIR" "$SAVE_DIR"
    copy_status=$?
    set -e
    log "Checkpoint archive exit status: $copy_status"
    ;;
esac

cp -f "$EVAL_DRIVER_LOG" "${META_DIR}/$(basename "$EVAL_DRIVER_LOG")"
if [[ -f "$EVAL_LOG" ]]; then
  cp -f "$EVAL_LOG" "${META_DIR}/$(basename "$EVAL_LOG")"
fi

if [[ "$copy_status" -ne 0 ]]; then
  exit "$copy_status"
fi

exit "$eval_status"
