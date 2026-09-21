#!/bin/bash
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/home/smansou2/arc_results/ge2_accuracy_%j.out
#SBATCH --error=/home/smansou2/arc_results/ge2_accuracy_%j.err
set -euo pipefail
BASE=$1
MODEL=$2
STUDY=$3
ENV=/mnt/local/smansou2/ge2-a6000-cuda121
export PATH="$ENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ENV/lib/python3.9/site-packages/gege:$ENV/lib/python3.9/site-packages/torch/lib:$ENV/lib"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
exec "$ENV/bin/python" "$BASE/scripts/launch_arc_ge2_accuracy.py" --base "$BASE" --model "$MODEL" --study "$STUDY"
