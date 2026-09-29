#!/bin/bash
#SBATCH --job-name=fb_multigpu_repair
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/home/smansou2/arc_results/logs/fb_multigpu_repair_%j.out
#SBATCH --error=/home/smansou2/arc_results/logs/fb_multigpu_repair_%j.err

set -euo pipefail
PAYLOAD=${1:?Pass repair payload directory}
PREFIX=/mnt/local/$USER/ge2-a6000-cuda121
export PATH="$PREFIX/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/python3.9/site-packages/torch/lib:/usr/lib64"
unset PYTHONPATH PYTHONHOME LD_PRELOAD
echo "Waiting for the verified repair payload on $(hostname), job $SLURM_JOB_ID"
for ((attempt=0; attempt<180; attempt++)); do
    if [[ -f "$PAYLOAD/READY" ]]; then
        exec "$PREFIX/bin/python" -B -u "$PAYLOAD/run_arc_multigpu_repair.py" \
            --payload "$PAYLOAD" --base "/mnt/local/$USER/fb_multigpu_repair_$SLURM_JOB_ID"
    fi
    sleep 10
done
echo "Repair payload was not ready within 30 minutes" >&2
exit 1
