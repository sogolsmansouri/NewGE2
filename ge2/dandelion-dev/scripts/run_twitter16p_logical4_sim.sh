#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build/gege"
TRAIN_BIN="$BUILD_DIR/gege_train"
ESTIMATOR="$ROOT/scripts/estimate_logical_lane_epoch.py"
CONFIG="${1:-$ROOT/gege/configs/twitter_16p_table5_sim_logical4.yaml}"
OUT_DIR="${2:-$ROOT/logs/arc/twitter16p_logical4_sim}"
BACKEND="${GEGE_UNIQUE_BACKEND:-sort}"

mkdir -p "$OUT_DIR"

cmake --build "$BUILD_DIR" -j

for lane in 0 1 2 3; do
  log="$OUT_DIR/lane${lane}.log"
  echo "Running logical lane $lane -> $log"
  CUDA_VISIBLE_DEVICES=0 \
  TORCH_SHOW_CPP_STACKTRACES=1 \
  GEGE_CSR_GATHER=0 \
  GEGE_CSR_DEBUG=0 \
  GEGE_STAGE_DEBUG=0 \
  GEGE_UNIQUE_BACKEND="$BACKEND" \
  GEGE_PARTITION_BUFFER_PEER_RELAY=1 \
  GEGE_PROFILE_LOGICAL_LANE="$lane" \
  "$TRAIN_BIN" "$CONFIG" >"$log" 2>&1
done

python3 "$ESTIMATOR" \
  "$OUT_DIR/lane0.log" \
  "$OUT_DIR/lane1.log" \
  "$OUT_DIR/lane2.log" \
  "$OUT_DIR/lane3.log" | tee "$OUT_DIR/estimate.txt"
