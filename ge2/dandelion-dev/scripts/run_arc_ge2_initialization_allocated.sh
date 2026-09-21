#!/bin/bash
# Run the frozen GE2 initialization controls inside an existing allocation.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  printf 'Usage: %s BASE WORK RESULTS DATA\n' "$0" >&2
  exit 2
fi
BASE=$1
WORK=$2
RESULT=$3
DATA=$4
: "${SLURM_JOB_ID:?An active owned allocation is required}"
ENV=/mnt/local/smansou2/ge2-a6000-cuda121
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$ENV/lib/python3.9/site-packages/gege:$ENV/lib/python3.9/site-packages/torch/lib:$ENV/lib"
unset PYTHONPATH PYTHONHOME LD_PRELOAD
test -f "$DATA/dataset.yaml"
test ! -e "$WORK"
test ! -e "$RESULT"
mkdir -p "$WORK" "$RESULT"
cp -a "$BASE/scripts" "$WORK/scripts"
cp -a "$BASE/tools" "$WORK/tools"
"$ENV/bin/python" "$WORK/scripts/test_zenodo_fb_sampling_control.py"
for MODEL in distmult complex; do
  "$ENV/bin/python" "$WORK/scripts/run_zenodo_fb_sampling_control.py" \
    --work "$WORK/$MODEL" --results "$RESULT/$MODEL" --env "$ENV" \
    --tools "$WORK/tools" --reference "$BASE/reference/fb_$MODEL" \
    --data "$DATA" --gpu 0 --job "$SLURM_JOB_ID" --model "$MODEL" \
    --study initialization
done
