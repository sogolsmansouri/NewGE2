#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_SRC="${CONFIG_SRC:-$REPO_ROOT/gege/configs/single_gpu/livejournal_16p.yaml}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_ge2env_ge2py39/gege}"
TRAIN_BIN="${GEGE_TRAIN_BIN:-$BUILD_DIR/gege_train}"
EVAL_BIN="${GEGE_EVAL_BIN:-$BUILD_DIR/gege_eval}"
RUN_NAME="${RUN_NAME:-lj_accuracy_gate_30e}"
RUN_ROOT="${RUN_ROOT:-/tmp/smansou2_ge2/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-/tmp/smansou2_ge2/exp_logs}"
TMP_CONFIG="${TMP_CONFIG:-/tmp/${RUN_NAME}.yaml}"
EPOCHS="${EPOCHS:-30}"

mkdir -p "$RUN_ROOT/model" "$LOG_DIR"

if [[ ! -x "$TRAIN_BIN" ]]; then
  echo "missing train binary: $TRAIN_BIN" >&2
  exit 1
fi

if [[ ! -x "$EVAL_BIN" ]]; then
  echo "missing eval binary: $EVAL_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_SRC" ]]; then
  echo "missing config: $CONFIG_SRC" >&2
  exit 1
fi

cp "$CONFIG_SRC" "$TMP_CONFIG"
perl -0pi -e 's#(training:\n(?:.*\n)*?  num_epochs:\s*)\d+#${1}'"$EPOCHS"'#; s#model_dir: .*#model_dir: '"$RUN_ROOT"'/model#; s#checkpoint_dir: .*#checkpoint_dir: '"$RUN_ROOT"'/model#' "$TMP_CONFIG"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GEGE_UNIQUE_BACKEND=bitmap
export GEGE_UNIQUE_BITMAP_NUM_NODES=4847571
export GEGE_EMULATE_DOT_SINGLE_RELATION=1
export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
export GEGE_FAST_MAP_TENSORS=1
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
export GEGE_DEG_CHUNK_EXCLUSION=1
export GEGE_BUCKET_STREAMING_LP=1
export GEGE_BUCKET_BLOCK_EXECUTOR=1
export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=1
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_EMPTY_CACHE_AROUND_SWAP=0
export GEGE_SYNC_BEFORE_SWAP=0
export GEGE_MEM_SWAP_EVENT_SYNC=1
export GEGE_PROFILE_LOGICAL_LANE=0
export GEGE_FIXED_BUFFER_BITMAP_MAP=1
export GEGE_FIXED_BUFFER_MASKED_UPDATE=1
export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=1
export GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1
export GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK=1
export GEGE_NEGATIVE_CHUNK_SAMPLE_CUDA="${GEGE_NEGATIVE_CHUNK_SAMPLE_CUDA:-0}"
export GEGE_NEGATIVE_FUSED_NODE_CHUNK_CUDA="${GEGE_NEGATIVE_FUSED_NODE_CHUNK_CUDA:-0}"
export GEGE_DEG_LOCAL_FILTER_PADDED="${GEGE_DEG_LOCAL_FILTER_PADDED:-0}"
export GEGE_SCORE_FILTER_CUDA="${GEGE_SCORE_FILTER_CUDA:-0}"

if [[ -z "${LD_PRELOAD:-}" && -f /lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]]; then
  export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
fi

echo "[run_lj_single_gpu_accuracy_gate] config=$TMP_CONFIG epochs=$EPOCHS run_root=$RUN_ROOT" >&2
echo "[run_lj_single_gpu_accuracy_gate] target: exact LJ eval on livejournal_16p_10k_eval should be near MRR=0.1295 Hits@10=0.3108" >&2

"$TRAIN_BIN" "$TMP_CONFIG" |& tee "$LOG_DIR/${RUN_NAME}_train.log"

export GEGE_EVAL_CHUNKED_RANKS=1
export GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768

"$EVAL_BIN" "$TMP_CONFIG" |& tee "$LOG_DIR/${RUN_NAME}_eval.log"

echo "[run_lj_single_gpu_accuracy_gate] train_log=$LOG_DIR/${RUN_NAME}_train.log" >&2
echo "[run_lj_single_gpu_accuracy_gate] eval_log=$LOG_DIR/${RUN_NAME}_eval.log" >&2
