#!/usr/bin/env python3
"""Validate native graph remapping against storage for a selected schedule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess

import yaml

from run_workstation_fb_accuracy_control import schedule_flags


def validation_flags(original, schedule, retain, pipeline):
    if pipeline and (schedule != 'bounded' or not retain):
        raise ValueError('Pipeline validation requires bounded scheduling and retention')
    flags = schedule_flags(original, schedule)
    flags.update(GEGE_PARTITION_BUFFER_LP_FAST_PATH='1',
                 GEGE_PARTITION_BUFFER_LP_FAST_PATH_VALIDATE='1',
                 GEGE_PARTITION_BUFFER_LP_FAST_PATH_VALIDATE_MAX='100000',
                 GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM='1' if retain else '0',
                 GEGE_FRAME_CACHE_HIDDEN_FRAMES='6' if pipeline else '0',
                 GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='1' if pipeline else '0',
                 GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='1' if pipeline else '0',
                 GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='3' if pipeline else '0',
                 GEGE_PARTITION_BUFFER_PEER_RELAY='0',
                 GEGE_STATEFLOW_ALLOW_PEER_RELAY='0')
    return flags


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--retain', action='store_true', help='Check the stable-slot retention control')
    parser.add_argument('--schedule', choices=('legacy-random', 'bounded'), default='legacy-random')
    parser.add_argument('--pipeline', action='store_true', help='Check hidden-frame publication too')
    args = parser.parse_args()
    config = yaml.safe_load((args.fixture/'config.yaml').read_text())
    flags = validation_flags(json.loads((args.fixture/'flags.json').read_text()),
                             args.schedule, args.retain, args.pipeline)
    apps = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError('Selected GPU is busy: '+apps)
    args.work.mkdir(parents=True, exist_ok=False)
    config['storage']['model_dir'] = str(args.work/'model')+'/'
    config['storage']['device_ids'] = [0]
    # Compare after storage publication; the dense validator has no future-state map.
    config['storage']['prefetch'] = False
    config['evaluation']['checkpoint_dir'] = str(args.work/'model')+'/'
    config['training']['num_epochs'] = 2
    config_path = args.work/'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (args.work/'flags.json').write_text(json.dumps(flags, indent=2)+'\n')
    package = args.work/'python'
    package.mkdir()
    (package/'gege').symlink_to((args.source/'src/python').resolve(), target_is_directory=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('GEGE_', 'PYTHON', 'CONDA', 'SLURM_', 'OMP_', 'MKL_'))}
    env.pop('LD_PRELOAD', None)
    env.update({key: str(value) for key, value in flags.items()})
    env.update(PATH=str(args.prefix/'bin')+':/usr/bin:/bin',
               PYTHONPATH=str(package), GEGE_NO_BINDINGS='1', PYTHONNOUSERSITE='1')
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['LD_LIBRARY_PATH'] = ':'.join([str(args.binary.parent), str(args.prefix/'lib'),
                                      str(args.prefix/'lib/python3.9/site-packages/torch/lib'),
                                      env.get('LD_LIBRARY_PATH', '')])
    env['OMP_NUM_THREADS'] = '4'
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with (args.work/'train.log').open('w') as log:
        run = subprocess.run([str(args.binary.resolve()), str(config_path.resolve())],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (args.work/'train.log').read_text()
    result = dict(exit_code=run.returncode, prefetch=False,
                  retention=args.retain, schedule=args.schedule, pipeline=args.pipeline,
                  binary_sha256=digest(args.binary),
                  library_sha256=digest(args.binary.parent/'libge2.so'),
                  remap_mismatch='LP fast path remap mismatch' in text,
                  validation_checks=len(re.findall(r'LP fast path validation passed', text)),
                  epochs_completed=text.count('Epoch Runtime:'), paper_ready=False)
    (args.work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(run.returncode != 0 or result['epochs_completed'] != 2
                     or result['validation_checks'] < 2)


if __name__ == '__main__':
    main()
