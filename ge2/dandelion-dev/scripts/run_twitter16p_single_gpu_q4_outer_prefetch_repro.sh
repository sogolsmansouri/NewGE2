#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${CONFIG_SRC:-$ROOT/gege/configs/twitter_16p_paper_opt.yaml}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-$ROOT/scripts/run_twitter16p_single_gpu_q4_prefetch_baseline.sh}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/home/smansou2/miniconda/bin/activate}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-libkge-clean}"
BUILD_DIR="${GEGE_BUILD_DIR:-$ROOT/gege/build-codex-libkge-clean-targets}"
LEGACY_SITE_PACKAGES="${LEGACY_SITE_PACKAGES:-/home/smansou2/miniconda/envs/ge2/lib/python3.9/site-packages}"
PRINT_ONLY="${PRINT_ONLY:-0}"
KEEP_GENERATED_CONFIG="${KEEP_GENERATED_CONFIG:-0}"

prepend_path() {
  local entry="$1"
  if [[ ! -d "$entry" ]]; then
    return 0
  fi
  case ":${PYTHONPATH:-}:" in
    *":$entry:"*) ;;
    *)
      if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="$entry:$PYTHONPATH"
      else
        export PYTHONPATH="$entry"
      fi
      ;;
  esac
}

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "missing base config: $BASE_CONFIG" >&2
  exit 1
fi

if [[ ! -x "$BASELINE_SCRIPT" ]]; then
  echo "missing baseline launcher: $BASELINE_SCRIPT" >&2
  exit 1
fi

if [[ "${GEGE_SKIP_CONDA_ACTIVATE:-0}" != "1" && -f "$CONDA_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  set +u
  source "$CONDA_ACTIVATE" "$CONDA_ENV_NAME"
  set -u
fi

if [[ -z "${GEGE_TRAIN_BIN:-}" ]]; then
  if [[ -x "$BUILD_DIR/gege_train" ]]; then
    export GEGE_TRAIN_BIN="$BUILD_DIR/gege_train"
  fi
fi

if [[ -z "${GEGE_EVAL_BIN:-}" ]]; then
  if [[ -x "$BUILD_DIR/gege_eval" ]]; then
    export GEGE_EVAL_BIN="$BUILD_DIR/gege_eval"
  fi
fi

prepend_path "$BUILD_DIR"
prepend_path "$LEGACY_SITE_PACKAGES"

TMP_CFG="$(mktemp /tmp/twitter_q4_outer_prefetch_XXXX.yaml)"
cleanup() {
  if [[ "$KEEP_GENERATED_CONFIG" != "1" ]]; then
    rm -f "$TMP_CFG"
  fi
}
trap cleanup EXIT

cp "$BASE_CONFIG" "$TMP_CFG"
perl -0pi -e 's#(^\s*prefetch:\s*).*$#$1true#m' "$TMP_CFG"

export CONFIG_SRC="$TMP_CFG"
export GEGE_STATEFLOW_PLANNER="${GEGE_STATEFLOW_PLANNER:-1}"
export EPOCHS="${EPOCHS:-2}"
export RUN_NAME="${RUN_NAME:-twitter_q4_outer_prefetch_repro_${EPOCHS}e_$(date '+%Y%m%d_%H%M%S')}"
export RUN_ROOT="${RUN_ROOT:-/home/smansou2/codex_runs/$RUN_NAME}"
export LOG_DIR="${LOG_DIR:-/home/smansou2/codex_runs/exp_logs}"
export MIN_FREE_GIB="${MIN_FREE_GIB:-25}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  echo "base_config=$BASE_CONFIG"
  echo "generated_config=$TMP_CFG"
  echo "prefetch_line=$(grep -n '^[[:space:]]*prefetch:' "$TMP_CFG" | head -n 1)"
  echo "GEGE_TRAIN_BIN=${GEGE_TRAIN_BIN:-}"
  echo "GEGE_EVAL_BIN=${GEGE_EVAL_BIN:-}"
  echo "PYTHONPATH=${PYTHONPATH:-}"
  echo "GEGE_STATEFLOW_PLANNER=$GEGE_STATEFLOW_PLANNER"
  echo "EPOCHS=$EPOCHS"
  echo "RUN_ROOT=$RUN_ROOT"
  echo "LOG_DIR=$LOG_DIR"
  echo "launcher=$BASELINE_SCRIPT"
  exit 0
fi

"$BASELINE_SCRIPT"
