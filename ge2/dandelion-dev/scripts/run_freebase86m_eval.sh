#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_EVAL_BIN="/home/smansou2/miniconda/envs/ge2/bin/gege_eval"
EVAL_BIN="${GEGE_EVAL_BIN:-$DEFAULT_EVAL_BIN}"

CHECKPOINT_DIR=""
DATASET_MODE="paper-10k"
EVAL_BATCH_SIZE="${GEGE_EVAL_BATCH_SIZE:-250}"
NEGATIVE_CHUNK_SIZE="${GEGE_EVAL_NEGATIVE_CHUNK_SIZE:-32768}"
CONFIG_OUT=""
LOG_OUT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  run_freebase86m_eval.sh --checkpoint-dir <dir> [options]

Options:
  --checkpoint-dir <dir>   Checkpoint directory containing full_config.yaml.
  --dataset-mode <mode>    One of: paper-10k, full. Default: paper-10k.
  --eval-batch-size <n>    Evaluation batch size. Default: 250.
  --negative-chunk-size <n>
                           Chunk size for exact filtered ranks. Default: 32768.
  --config-out <path>      Output YAML path. Default: /tmp/<ckpt>_eval_<mode>.yaml.
  --log-out <path>         Eval log path. Default: /tmp/<ckpt>_eval_<mode>.log.
  --dry-run                Generate config and print command without running eval.
  -h, --help               Show this help.

Notes:
  - 'paper-10k' matches the GE2 Table 4 protocol: exact filtered MRR on 10^4
    test edges, not the full 5% test split.
  - 'full' points eval at the full Freebase86M 90/5/5 split dataset.
  - The default batch size is conservative for 24 GB GPUs. Increase it only if
    you have already verified your card can hold the eval batches.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-dir)
      CHECKPOINT_DIR="${2:?missing value for --checkpoint-dir}"
      shift 2
      ;;
    --dataset-mode)
      DATASET_MODE="${2:?missing value for --dataset-mode}"
      shift 2
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="${2:?missing value for --eval-batch-size}"
      shift 2
      ;;
    --negative-chunk-size)
      NEGATIVE_CHUNK_SIZE="${2:?missing value for --negative-chunk-size}"
      shift 2
      ;;
    --config-out)
      CONFIG_OUT="${2:?missing value for --config-out}"
      shift 2
      ;;
    --log-out)
      LOG_OUT="${2:?missing value for --log-out}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CHECKPOINT_DIR" ]]; then
  echo "--checkpoint-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 2
fi

FULL_CONFIG="$CHECKPOINT_DIR/full_config.yaml"
if [[ ! -f "$FULL_CONFIG" ]]; then
  echo "Checkpoint full_config.yaml not found: $FULL_CONFIG" >&2
  exit 2
fi

case "$DATASET_MODE" in
  paper-10k)
    DATASET_DIR="$ROOT/datasets/freebase86m_16p_paper_10k_eval"
    ;;
  full)
    DATASET_DIR="$ROOT/datasets/freebase86m_16p"
    ;;
  *)
    echo "Unsupported --dataset-mode: $DATASET_MODE" >&2
    exit 2
    ;;
esac

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Dataset directory does not exist: $DATASET_DIR" >&2
  exit 2
fi

if [[ ! -x "$EVAL_BIN" ]]; then
  echo "gege_eval binary not found or not executable: $EVAL_BIN" >&2
  exit 2
fi

CKPT_BASENAME="$(basename "$CHECKPOINT_DIR")"
CONFIG_OUT="${CONFIG_OUT:-/tmp/${CKPT_BASENAME}_eval_${DATASET_MODE}.yaml}"
LOG_OUT="${LOG_OUT:-/tmp/${CKPT_BASENAME}_eval_${DATASET_MODE}.log}"

python3 - "$FULL_CONFIG" "$DATASET_DIR" "$CHECKPOINT_DIR" "$EVAL_BATCH_SIZE" "$CONFIG_OUT" <<'PY'
import pathlib
import sys

import yaml

full_config_path = pathlib.Path(sys.argv[1])
dataset_dir = pathlib.Path(sys.argv[2]).resolve()
checkpoint_dir = pathlib.Path(sys.argv[3]).resolve()
eval_batch_size = int(sys.argv[4])
config_out = pathlib.Path(sys.argv[5])

with full_config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

encoder = config.setdefault("model", {}).setdefault("encoder", {})
layers = encoder.get("layers", [])
if layers and layers[0] and isinstance(layers[0][0], dict):
    encoder_optimizer = layers[0][0].setdefault("optimizer", {})
    encoder_optimizer["type"] = encoder_optimizer.get("type", "DEFAULT")
    encoder_optimizer["options"] = {
        "learning_rate": 0.1,
        "eps": 1.0e-10,
        "init_value": 0.0,
        "lr_decay": 0.0,
        "weight_decay": 0.0,
    }

storage = config.setdefault("storage", {})
storage["device_type"] = "cuda"
storage["device_ids"] = [0]
storage["model_dir"] = str(checkpoint_dir) + "/"
storage["full_graph_evaluation"] = False

dataset = storage.setdefault("dataset", {})
dataset_yaml = dataset_dir / "dataset.yaml"
with dataset_yaml.open("r", encoding="utf-8") as f:
    dataset_config = yaml.safe_load(f)
dataset.clear()
dataset.update(dataset_config)
dataset["dataset_dir"] = str(dataset_dir) + "/"

evaluation = config.setdefault("evaluation", {})
evaluation["batch_size"] = eval_batch_size
evaluation["epochs_per_eval"] = 1
evaluation["checkpoint_dir"] = str(checkpoint_dir) + "/"

negative_sampling = evaluation.setdefault("negative_sampling", {})
negative_sampling["filtered"] = True

config_out.parent.mkdir(parents=True, exist_ok=True)
with config_out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

echo "Generated config: $CONFIG_OUT"
echo "Checkpoint dir:   $CHECKPOINT_DIR"
echo "Dataset mode:     $DATASET_MODE"
echo "Dataset dir:      $DATASET_DIR"
echo "Eval batch size:  $EVAL_BATCH_SIZE"
echo "Log file:         $LOG_OUT"

echo "Environment:"
echo "  CUDA_VISIBLE_DEVICES=0"
echo "  GEGE_EVAL_CHUNKED_RANKS=1"
echo "  GEGE_EVAL_NEGATIVE_CHUNK_SIZE=$NEGATIVE_CHUNK_SIZE"
echo "  GEGE_PARTITION_BUFFER_PIPELINE_TIMING=1"
echo "  GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX=4096"
echo "  GEGE_PARTITION_BUFFER_SWAP_TIMING=1"
echo "  GEGE_PARTITION_BUFFER_SWAP_TIMING_MAX=4096"
echo "  GEGE_UNIQUE_BITMAP_NUM_NODES=86054151"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run enabled; not starting eval."
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
export GEGE_EVAL_CHUNKED_RANKS=1
export GEGE_EVAL_NEGATIVE_CHUNK_SIZE="$NEGATIVE_CHUNK_SIZE"
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING=1
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX=4096
export GEGE_PARTITION_BUFFER_SWAP_TIMING=1
export GEGE_PARTITION_BUFFER_SWAP_TIMING_MAX=4096
export GEGE_FAST_MAP_TENSORS=1
export GEGE_BUCKET_STREAMING_LP=1
export GEGE_CSR_GATHER=1
export GEGE_CSR_UPDATE=1
export GEGE_CSR_UPDATE_REDUCE=0
export GEGE_CSR_DEBUG=0
export GEGE_STAGE_DEBUG=0
export GEGE_DEG_CHUNK_EXCLUSION=1
export GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES=0
export GEGE_UNIQUE_BACKEND=bitmap
export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
export GEGE_UNIQUE_LOG=0

echo "Running eval..."
"$EVAL_BIN" "$CONFIG_OUT" 2>&1 | tee "$LOG_OUT"
status=${PIPESTATUS[0]}
echo "Eval exit status: $status"
exit "$status"
