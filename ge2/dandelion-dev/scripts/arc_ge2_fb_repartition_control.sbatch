#!/bin/bash
#SBATCH --job-name=ge2_fb_repartition
#SBATCH --partition=a6000
#SBATCH --nodelist=c31
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=06:00:00
#SBATCH --output=ge2_fb_repartition_%j.out
#SBATCH --error=ge2_fb_repartition_%j.err
set -euo pipefail
BASE="$SLURM_SUBMIT_DIR"
ENV=/mnt/local/smansou2/ge2-a6000-cuda121
WORK="/mnt/local/smansou2/ge2_fb_repartition_${SLURM_JOB_ID}"
RESULT="$HOME/arc_results/runs/ge2_fb_repartition_${SLURM_JOB_ID}"
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
test "$(md5sum "$BASE/ge2.zip" | awk '{print $1}')" = 6de3d9702241a0c822971939752d0834
unzip -q "$BASE/ge2.zip" -d "$WORK/source"
"$ENV/bin/python" "$WORK/scripts/build_zenodo_control.py" \
  --source-root "$WORK/source/dandelion-dev/gege" --out "$WORK/build"
cp -a "$WORK/build/build.json" "$RESULT/build.json"
set +e
"$ENV/bin/python" "$WORK/scripts/run_zenodo_native_fixture.py" \
  --binary "$WORK/build/audit_zenodo_native_training" \
  --template "$BASE/reference/fb_distmult/config.yaml" \
  --work "$WORK/fixture_data" --results "$RESULT/fixtures" --env "$ENV" \
  --gpu 0 --job "$SLURM_JOB_ID" --epochs 3 --repartition --observe-padding
RC=$?
set -e
# Uneven fixtures must expose the known padding bug, not bypass other failures.
"$ENV/bin/python" - "$RESULT/fixtures/manifest.json" "$RC" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert int(sys.argv[2]) == 1 and len(r['cases']) == 6
for c in r['cases']:
    assert c['exit_code'] == 0 and c['math_storage_passed']
    assert c['epochs'] == 3 and c['repartition'] and c['edges'] == 13056
    if c['case'].endswith('_512'):
        assert c['passed'] and c['padding_candidate_draws'] == 0
    else:
        assert not c['passed'] and c['padding_candidate_draws'] > 0
print('Repartition integrity gate passed; uneven-partition padding defect remains disclosed.')
PY
"$ENV/bin/python" "$WORK/scripts/test_zenodo_driver_parity.py" \
  --fixtures "$RESULT/fixtures" --out "$RESULT/driver_parity" \
  --binary "$WORK/build/train_zenodo_repartition_control" --env "$ENV"
for MODEL in distmult complex; do
  "$ENV/bin/python" "$WORK/scripts/run_zenodo_fb_sampling_control.py" \
    --work "$WORK/$MODEL" --results "$RESULT/$MODEL" --env "$ENV" \
    --tools "$WORK/tools" --reference "$BASE/reference/fb_$MODEL" \
    --data "$DATA" --gpu 0 --job "$SLURM_JOB_ID" --model "$MODEL" \
    --study repartition --binary "$WORK/build/train_zenodo_repartition_control"
done
