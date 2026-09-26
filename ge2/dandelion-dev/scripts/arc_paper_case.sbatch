#!/bin/bash
#SBATCH --job-name=paper_case
#SBATCH --partition=dkex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@240
set -euo pipefail
BASE=$1
SYSTEM=$2
CASE=$3
ENV=/mnt/local/smansou2/ge2-a6000-cuda121
export PATH="$ENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/lib/python3.9/site-packages/torch/lib:/usr/lib64"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH LD_PRELOAD
exec "$ENV/bin/python" "$BASE/scripts/run_arc_paper_case.py" --base "$BASE" --system "$SYSTEM" --case "$CASE"
