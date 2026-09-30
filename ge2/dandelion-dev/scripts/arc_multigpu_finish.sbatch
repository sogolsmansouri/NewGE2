#!/bin/bash
#SBATCH --job-name=multigpu_final
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/home/smansou2/arc_results/logs/multigpu_final_%j.out
#SBATCH --error=/home/smansou2/arc_results/logs/multigpu_final_%j.err
#SBATCH --open-mode=append

set -euo pipefail
PAYLOAD=${1:?Pass frozen payload directory}
CASE=${2:?Pass one case name}
PREFIX=/mnt/local/$USER/ge2-a6000-cuda121
export PATH="$PREFIX/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/python3.9/site-packages/torch/lib:/usr/lib64"
unset PYTHONPATH PYTHONHOME LD_PRELOAD
ulimit -c 0
test -s "$PAYLOAD/READY"
RESTART=${SLURM_RESTART_COUNT:-0}
[[ "$RESTART" =~ ^(0|[1-9][0-9]*)$ ]] || exit 2
LOCAL_PAYLOAD="/mnt/local/$USER/multigpu_finish_payload_${SLURM_JOB_ID}_r${RESTART}"
mkdir "$LOCAL_PAYLOAD"
cp -a "$PAYLOAD/." "$LOCAL_PAYLOAD/"
exec "$PREFIX/bin/python" -B -u "$LOCAL_PAYLOAD/run_arc_multigpu_finish.py" \
    --payload "$LOCAL_PAYLOAD" --case "$CASE" \
    --base "/mnt/local/$USER/multigpu_finish_$SLURM_JOB_ID"
