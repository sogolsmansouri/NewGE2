#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/smansou2/newCode/ge2/dandelion-dev"
TRAIN_BIN="/home/smansou2/miniconda/envs/ge2/bin/gege_train"
EVAL_BIN="/home/smansou2/miniconda/envs/ge2/bin/gege_eval"

log() {
  printf '[%s] %s\n' "$(date '+%m/%d/%y %H:%M:%S')" "$*"
}

log "Starting Freebase86M opt training"
export CUDA_VISIBLE_DEVICES=0
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
export GEGE_FAST_MAP_TENSORS=1
export GEGE_BUCKET_STREAMING_LP=1
export GEGE_CSR_GATHER=1
export GEGE_CSR_UPDATE=1
export GEGE_CSR_UPDATE_REDUCE=0
export GEGE_CSR_DEBUG=0
export GEGE_DEG_CHUNK_EXCLUSION=1
export GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES=0
export GEGE_UNIQUE_BACKEND=bitmap
export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
export GEGE_UNIQUE_LOG=0

set +e
"$TRAIN_BIN" "$ROOT/gege/configs/freebase86m_16p_paper_opt.yaml"
fb_status=$?
set -e
log "Freebase86M training exit status: $fb_status"

log "Switching back to Twitter eval on the saved /dev/shm checkpoint"
export GEGE_UNIQUE_BITMAP_NUM_NODES=41652230
export GEGE_EVAL_CHUNKED_RANKS=1
export GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768

set +e
"$EVAL_BIN" "/dev/shm/smansou2_ge2/paper_single_gpu/twitter_16p_opt/full_config.yaml"
tw_status=$?
set -e
log "Twitter eval exit status: $tw_status"
