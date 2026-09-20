#!/bin/bash
#SBATCH --job-name=ge2_fb_init_control
#SBATCH --partition=a6000
#SBATCH --nodelist=c31
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=04:00:00
#SBATCH --output=ge2_fb_init_control_%j.out
#SBATCH --error=ge2_fb_init_control_%j.err
set -euo pipefail
BASE="${1:-$PWD}"
ENV=/mnt/local/smansou2/ge2-a6000-cuda121
WORK="/mnt/local/smansou2/ge2_fb_init_control_${SLURM_JOB_ID}"
RESULT="$HOME/arc_results/runs/ge2_fb_init_control_${SLURM_JOB_ID}"
DATA=/mnt/local/smansou2/ge2_fb_optimizer_fb_distmult_284744/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ENV/lib/python3.9/site-packages/gege:$ENV/lib/python3.9/site-packages/torch/lib:$ENV/lib"
mkdir -p "$WORK" "$RESULT"
test -f "$DATA/dataset.yaml"
hostname
nvidia-smi
df -h /mnt/local "$HOME"
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
