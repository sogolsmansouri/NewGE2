#!/usr/bin/env bash
set -euo pipefail
# Invoke under nohup inside a newly verified exclusive allocation. No srun.
: "${SLURM_JOB_ID:?Run inside an owned ARC allocation}"
: "${BUNDLE:?Frozen preparation directory}"
: "${TW_DATA:?Read-only canonical controlled TW p16 dataset}"
: "${GE2_ARCHIVE:?Released ge2.zip}"
: "${POWER_LIMIT_WATTS:?Expected observed power limit, e.g. 200 or 300}"
ENV_DIR="${ENV_DIR:-/mnt/local/smansou2/ge2-a6000-cuda121}"
ENGINE_DIR="${ENGINE_DIR:-/mnt/local/smansou2/pipege_engine_14ec00d_arc}"
TAG="tw_multigpu_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
WORK_ROOT="${WORK_ROOT:-/mnt/local/smansou2/$TAG}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/arc_results/runs/$TAG}"
mkdir -p "$RESULT_ROOT"
printf '%s\n' "$WORK_ROOT" "$RESULT_ROOT"
read -r -a cases <<< "${CASES:-ge2_tw_2gpu pipege_tw_2gpu ge2_tw_4gpu pipege_tw_4gpu}"
for case in "${cases[@]}"; do
    case "$case" in
        ge2_tw_2gpu|pipege_tw_2gpu) ids="${GPU_IDS_2:-0,1}" ;;
        ge2_tw_4gpu|pipege_tw_4gpu) ids="${GPU_IDS_4:-0,1,2,3}" ;;
        *) printf 'Unknown case: %s\n' "$case" >&2; exit 2 ;;
    esac
    common=("$ENV_DIR/bin/python" "$BUNDLE/run_tw_multigpu.py"
        --bundle "$BUNDLE" --case "$case" --env "$ENV_DIR" --engine "$ENGINE_DIR"
        --data "$TW_DATA" --source-archive "$GE2_ARCHIVE" --gpu-ids "$ids"
        --power-limit "$POWER_LIMIT_WATTS")
    "${common[@]}" --phase gate --work "$WORK_ROOT/${case}_gate" --results "$RESULT_ROOT/${case}_gate"
    "${common[@]}" --phase final --work "$WORK_ROOT/$case" --results "$RESULT_ROOT/$case" \
        --gate-result "$RESULT_ROOT/${case}_gate/result.json"
done
