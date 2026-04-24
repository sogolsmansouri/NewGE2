#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  run_epoch_time_repro_20260424.sh twitter-2e
  run_epoch_time_repro_20260424.sh fb-2e
  run_epoch_time_repro_20260424.sh fb-10e-eval

Environment overrides:
  GEGE_TRAIN_BIN       Path to gege_train. Default: gege/build-cuda-check/gege_train
  GEGE_EVAL_BIN        Path to gege_eval. Default: gege/build-cuda-check/gege_eval
  GEGE_RUN_ROOT        Runtime/checkpoint directory. Default: /dev/shm/smansou2_ge2/<run_name>
  GEGE_LOG_DIR         Log directory. Default: /home/smansou2/codex_runs/exp_logs
  GEGE_RUN_NAME        Run name. Default depends on mode and timestamp.
  GEGE_EPOCHS          Override number of training epochs.
EOF
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

TRAIN_BIN="${GEGE_TRAIN_BIN:-$ROOT/gege/build-cuda-check/gege_train}"
EVAL_BIN="${GEGE_EVAL_BIN:-$ROOT/gege/build-cuda-check/gege_eval}"
LOG_DIR="${GEGE_LOG_DIR:-/home/smansou2/codex_runs/exp_logs}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

case "$MODE" in
  twitter-2e)
    DATASET="twitter"
    CONFIG_SRC="$ROOT/gege/configs/repro/twitter_16p_epoch168_20260424.yaml"
    DEFAULT_EPOCHS=2
    DEFAULT_RUN_NAME="twitter_16p_epoch168_repro_2e_${TIMESTAMP}"
    SAVE_MODEL=false
    RUN_EVAL=0
    ;;
  fb-2e)
    DATASET="fb"
    CONFIG_SRC="$ROOT/gege/configs/repro/freebase86m_16p_epoch158_20260424.yaml"
    DEFAULT_EPOCHS=2
    DEFAULT_RUN_NAME="fb86m_complex_ge2_repro_2e_${TIMESTAMP}"
    SAVE_MODEL=false
    RUN_EVAL=0
    ;;
  fb-10e-eval)
    DATASET="fb"
    CONFIG_SRC="$ROOT/gege/configs/repro/freebase86m_16p_epoch158_20260424.yaml"
    DEFAULT_EPOCHS=10
    DEFAULT_RUN_NAME="fb86m_complex_ge2_repro_10e_eval_${TIMESTAMP}"
    SAVE_MODEL=true
    RUN_EVAL=1
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

RUN_NAME="${GEGE_RUN_NAME:-$DEFAULT_RUN_NAME}"
RUN_ROOT="${GEGE_RUN_ROOT:-/dev/shm/smansou2_ge2/$RUN_NAME}"
EPOCHS="${GEGE_EPOCHS:-$DEFAULT_EPOCHS}"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}_train.log"
EVAL_LOG="$LOG_DIR/${RUN_NAME}_eval.log"
EVAL_DRIVER_LOG="$LOG_DIR/${RUN_NAME}_eval_driver.log"
TMP_CFG="$(mktemp "/tmp/${RUN_NAME}_XXXX.yaml")"

cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "Missing required executable: $1" >&2
    exit 2
  fi
}

require_executable "$TRAIN_BIN"
if [[ "$RUN_EVAL" == "1" ]]; then
  require_executable "$EVAL_BIN"
fi
require_file "$CONFIG_SRC"

mkdir -p "$RUN_ROOT" "$LOG_DIR"
cp "$CONFIG_SRC" "$TMP_CFG"
perl -0pi -e "s#num_epochs:\\s*\\d+#num_epochs: $EPOCHS#; s#model_dir:\\s*.*#model_dir: $RUN_ROOT#; s#checkpoint_dir:\\s*.*#checkpoint_dir: $RUN_ROOT#; s#save_model:\\s*(true|false)#save_model: $SAVE_MODEL#g; s#save_state:\\s*true#save_state: false#g" "$TMP_CFG"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GEGE_HYBRID_COVER=1
export GEGE_STATEFLOW_PLANNER=1
export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
export GEGE_FAST_MAP_TENSORS=1
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
export GEGE_DEG_CHUNK_EXCLUSION=1
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_CSR_UPDATE_REDUCE=0
export GEGE_CSR_DEBUG=0
export GEGE_EMPTY_CACHE_AROUND_SWAP=0
export GEGE_SYNC_BEFORE_SWAP=0
export GEGE_MEM_SWAP_EVENT_SYNC=1
export GEGE_PROFILE_LOGICAL_LANE=0
export GEGE_FIXED_BUFFER_BITMAP_MAP=1
export GEGE_FIXED_BUFFER_BITMAP_REUSE_OUTPUTS=1
export GEGE_FIXED_BUFFER_MASKED_UPDATE=1
export GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1
export GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK=0
export GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1
export GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1
export GEGE_PREPARED_BATCH_PIPELINE=0
export GEGE_PREFETCH_PREPARE_NEXT_PARTITION=0
export GEGE_PARTITION_BUFFER_SWAP_TIMING=0
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING=0

if [[ -r /home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 ]]; then
  CUDA_RT=/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
  TCMALLOC=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
  if [[ -r "$TCMALLOC" ]]; then
    export LD_PRELOAD="$CUDA_RT:$TCMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
  else
    export LD_PRELOAD="$CUDA_RT${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

export LD_LIBRARY_PATH="/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cublas/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cufft/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/curand/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "$DATASET" == "twitter" ]]; then
  export GEGE_BUCKET_STREAMING_LP=1
  export GEGE_FRAME_CACHE_HIDDEN_FRAMES=2
  export GEGE_UNIQUE_BACKEND=bitmap
  export GEGE_UNIQUE_BITMAP_NUM_NODES=41652230
  export GEGE_EMULATE_DOT_SINGLE_RELATION=1
  export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=1
else
  export GEGE_BUCKET_STREAMING_LP=0
  export GEGE_FRAME_CACHE_HIDDEN_FRAMES=1
  export GEGE_UNIQUE_BACKEND=bitmap
  export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
  export GEGE_EMULATE_DOT_SINGLE_RELATION=0
  export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=0
fi

echo "run_name=$RUN_NAME"
echo "mode=$MODE"
echo "config=$TMP_CFG"
echo "run_root=$RUN_ROOT"
echo "train_log=$TRAIN_LOG"
echo "train_bin=$TRAIN_BIN"
echo "epochs=$EPOCHS"
echo "save_model=$SAVE_MODEL"
echo "GEGE_SYNC_BEFORE_SWAP=$GEGE_SYNC_BEFORE_SWAP"
echo "GEGE_MEM_SWAP_EVENT_SYNC=$GEGE_MEM_SWAP_EVENT_SYNC"
echo "GEGE_FRAME_CACHE_HIDDEN_FRAMES=$GEGE_FRAME_CACHE_HIDDEN_FRAMES"
echo "GEGE_BUCKET_STREAMING_LP=$GEGE_BUCKET_STREAMING_LP"

# Record pipeline-relevant settings inside the train log (and stdout) so epoch
# wall times are not misattributed to mechanisms that are intentionally off.
{
  echo "--- gege_epoch_repro_pipeline_fingerprint $(date -Is) ---"
  echo "GEGE_PREPARED_BATCH_PIPELINE=${GEGE_PREPARED_BATCH_PIPELINE:-unset}"
  echo "GEGE_PREFETCH_PREPARE_NEXT_PARTITION=${GEGE_PREFETCH_PREPARE_NEXT_PARTITION:-unset}"
  echo "GEGE_FULL_PIPELINE_PREFETCH=${GEGE_FULL_PIPELINE_PREFETCH:-unset}"
  echo "GEGE_PARTITION_BUFFER_PIPELINE_TIMING=${GEGE_PARTITION_BUFFER_PIPELINE_TIMING:-unset}"
  echo "GEGE_BUCKET_STREAMING_LP=${GEGE_BUCKET_STREAMING_LP:-unset}"
  echo "GEGE_FRAME_CACHE_HIDDEN_FRAMES=${GEGE_FRAME_CACHE_HIDDEN_FRAMES:-unset}"
  echo "storage_prefetch_line=$(grep -E '^[[:space:]]*prefetch:' "$TMP_CFG" 2>/dev/null | head -n1 | tr -d '\r' || echo 'missing')"
  echo "embed_prefetching_line=$(grep -E 'prefetching:' "$TMP_CFG" 2>/dev/null | head -n1 | tr -d '\r' || echo 'missing')"
  echo "--- end fingerprint ---"
} | tee "$TRAIN_LOG"

set +e
"$TRAIN_BIN" "$TMP_CFG" 2>&1 | tee -a "$TRAIN_LOG"
train_status=${PIPESTATUS[0]}
set -e
echo "train_status=$train_status"
if [[ "$train_status" -ne 0 ]]; then
  exit "$train_status"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "eval_log=$EVAL_LOG"
  echo "eval_bin=$EVAL_BIN"
  set +e
  GEGE_EVAL_BIN="$EVAL_BIN" bash "$ROOT/scripts/run_freebase86m_eval.sh" \
    --checkpoint-dir "$RUN_ROOT" \
    --dataset-mode paper-10k \
    --config-out "$LOG_DIR/${RUN_NAME}_eval.yaml" \
    --log-out "$EVAL_LOG" 2>&1 | tee "$EVAL_DRIVER_LOG"
  eval_status=${PIPESTATUS[0]}
  set -e
  echo "eval_status=$eval_status"
  exit "$eval_status"
fi
