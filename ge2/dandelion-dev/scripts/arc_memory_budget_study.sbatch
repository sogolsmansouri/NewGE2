#!/bin/bash
#SBATCH --job-name=pipege_memory_10e
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=/home/smansou2/arc_results/logs/memory_budget_%j.out
#SBATCH --error=/home/smansou2/arc_results/logs/memory_budget_%j.err

set -euo pipefail
PAYLOAD=${1:?Pass the verified frozen payload directory}
PREFIX=/mnt/local/$USER/ge2-a6000-cuda121
export PATH="$PREFIX/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/python3.9/site-packages/torch/lib:/usr/lib64"
unset PYTHONPATH PYTHONHOME LD_PRELOAD
# A predecessor's GPU context can outlive its Slurm accounting transition.
# Wait for cleanup, but never terminate another workload or bypass isolation.
idle_samples=0
for ((attempt=0; attempt<40; attempt++)); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    if [[ -z "$apps" ]]; then
        idle_samples=$((idle_samples + 1))
        if ((idle_samples >= 3)); then break; fi
    else
        idle_samples=0
        echo "Waiting for existing GPU contexts to exit: $apps"
    fi
    sleep 15
done
if ((idle_samples < 3)); then
    echo "Node remained busy; refusing competing measurements" >&2
    exit 1
fi
exec "$PREFIX/bin/python" -B -u "$PAYLOAD/run_arc_memory_budget.py" \
    --payload "$PAYLOAD" \
    --root "/mnt/local/$USER/paper_memory_budget_20260929" \
    --results "/mnt/beegfs/$USER/paper_memory_budget_20260929" \
    --index "/home/$USER/arc_results/runs/paper_memory_budget_20260929" \
    --seconds 20700
