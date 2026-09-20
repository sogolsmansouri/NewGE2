#!/bin/bash
#SBATCH --job-name=pipege_best
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/home/smansou2/arc_results/pipege_best_%j.out
#SBATCH --error=/home/smansou2/arc_results/pipege_best_%j.err
set -euo pipefail
BASE=$1
WORK=$2
RESULTS=$3
COMMIT=$4
CASE=$5
ENV=/mnt/local/smansou2/ge2-a6000-cuda121
export PATH="$ENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/lib/python3.9/site-packages/torch/lib:/usr/lib64"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
exec "$ENV/bin/python" "$BASE/scripts/supervise_arc_pipege_best.py" \
    --base "$BASE" --work "$WORK" --results "$RESULTS" --commit "$COMMIT" --only "$CASE"
