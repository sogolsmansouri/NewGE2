#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/smansou2/newCode/ge2/dandelion-dev"
TRAIN_BIN="/home/smansou2/miniconda/envs/ge2/bin/gege_train"
EVAL_BIN="/home/smansou2/miniconda/envs/ge2/bin/gege_eval"

ARCHROOT="${1:?archive root path required}"

log() {
  printf '[%s] %s\n' "$(date '+%m/%d/%y %H:%M:%S')" "$*"
}

copy_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -rlptD "$src/" "$dst/"
  else
    cp -a "$src/." "$dst/"
  fi
}

log "Stopping current Twitter eval session if present"
tmux kill-session -t twitter_eval_20260322_214341 2>/dev/null || true
pkill -f "gege_eval /dev/shm/smansou2_ge2/paper_single_gpu/twitter_16p_opt/full_config.yaml" || true
sleep 2

log "Archiving checkpoints to $ARCHROOT"
mkdir -p "$ARCHROOT"
for name in livejournal_16p_baseline livejournal_16p_opt twitter_16p_opt; do
  src="/dev/shm/smansou2_ge2/paper_single_gpu/$name"
  if [ -d "$src" ]; then
    log "Copying $name"
    copy_dir "$src" "$ARCHROOT/$name"
  fi
done

TW_ARCHIVE_DIR="$ARCHROOT/twitter_16p_opt"
log "Rewriting archived Twitter config to point at $TW_ARCHIVE_DIR"
sed -i "s#/dev/shm/smansou2_ge2/paper_single_gpu/twitter_16p_opt#$TW_ARCHIVE_DIR#g" \
  "$TW_ARCHIVE_DIR/full_config.yaml"

log "Removing archived checkpoints from /dev/shm to free space"
rm -rf /dev/shm/smansou2_ge2/paper_single_gpu/livejournal_16p_baseline
rm -rf /dev/shm/smansou2_ge2/paper_single_gpu/livejournal_16p_opt
rm -rf /dev/shm/smansou2_ge2/paper_single_gpu/twitter_16p_opt

log "Current /dev/shm usage"
df -h /dev/shm

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

log "Switching back to archived Twitter eval"
export GEGE_UNIQUE_BITMAP_NUM_NODES=41652230
export GEGE_EVAL_CHUNKED_RANKS=1
export GEGE_EVAL_NEGATIVE_CHUNK_SIZE=32768

set +e
"$EVAL_BIN" "$TW_ARCHIVE_DIR/full_config.yaml"
tw_status=$?
set -e
log "Archived Twitter eval exit status: $tw_status"
