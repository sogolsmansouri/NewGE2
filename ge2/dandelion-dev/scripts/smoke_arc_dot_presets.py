#!/usr/bin/env python3
"""Exercise frozen LJ/TW policies on small synthetic graphs, not benchmark data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import resource

import numpy as np
import yaml

from run_arc_pipege_best import configure, freeze_baseline_sampling, schedule_check, training_check


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'references', 'gege', 'build', 'env', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--fix-sampling', action='store_true')
    args = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.output.mkdir(parents=True, exist_ok=False)
    overlay = args.output/'python'
    overlay.mkdir()
    (overlay/'gege').symlink_to(args.gege.resolve()/'src/python', target_is_directory=True)
    manifest = json.loads(args.manifest.read_text())
    report = dict(scope='synthetic functional smoke only; no timing or accuracy claims', cases={},
                  build={n:digest(args.build/n) for n in ('gege_train', 'libge2.so')})
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'SLURM_', 'PYTHON', 'CONDA'))}
    env.update(GEGE_NO_BINDINGS='1', PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE='1',
               LD_LIBRARY_PATH=f'{args.build}:{args.env}/lib:{args.env}/lib/python3.9/site-packages/torch/lib',
               CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    env.pop('LD_PRELOAD', None)
    scripts = Path(__file__).resolve().parent
    for name in ('lj_dot', 'tw_dot'):
        case = args.output/name
        (case/'data/edges').mkdir(parents=True)
        original = manifest['cases'][name]
        cfg_path = args.references/original['config']
        if digest(cfg_path) != manifest['references'][original['config']]:
            raise ValueError('Not the frozen config: '+name)
        flags = json.loads((args.references/original['flags']).read_text())
        flags.update(GEGE_BASELINE_TRAINING_SEMANTICS='1', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1')
        flags.pop('GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS', None)
        encoded = (json.dumps(flags, indent=2, sort_keys=True)+'\n').encode()
        if hashlib.sha256(encoded).hexdigest() != manifest['references'][original['flags']]:
            raise ValueError('Not the frozen flags: '+name)
        if 'schedule' in original:
            schedule = args.references/original['schedule']
            if digest(schedule) != manifest['references'][original['schedule']]:
                raise ValueError('Not the frozen schedule')
            schedule_check(schedule.read_text(), original)
            flags['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(schedule)
        partitions = original['p']
        rng = np.random.default_rng(123)
        per_bucket = 100000 if name == 'lj_dot' else 1000
        chunks = [np.column_stack((rng.integers(i*128, (i+1)*128, per_bucket),
                                   rng.integers(j*128, (j+1)*128, per_bucket))).astype('<i4')
                  for i in range(partitions) for j in range(partitions)]
        edges = np.concatenate(chunks)
        spec = dict(original, nodes=partitions*128, edges=len(edges))
        data = dict(dataset_dir=str(case/'data')+'/', num_nodes=spec['nodes'], num_relations=1,
                    num_edges=len(edges), num_train=len(edges), num_valid=100, num_test=100)
        (case/'data/dataset.yaml').write_text(yaml.safe_dump(data))
        for split, values in [('train', edges), ('validation', edges[:100]), ('test', edges[:100])]:
            values.tofile(case/'data/edges'/f'{split}_edges.bin')
            ids = (values[:, 0]//128)*partitions+values[:, 1]//128
            counts = np.bincount(ids, minlength=partitions**2)
            np.savetxt(case/'data/edges'/f'{split}_partition_offsets.txt', counts, fmt='%d')
        reference = yaml.safe_load(cfg_path.read_text())
        if args.fix_sampling:
            reference, flags = freeze_baseline_sampling(reference, flags)
        cfg = configure(reference, data, case/'model', spec, True)
        path = case/'config.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        flags.update(GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY='1', GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY_MAX='8')
        (case/'flags.json').write_text(json.dumps(flags, indent=2, sort_keys=True)+'\n')
        case_env = dict(env, **flags)
        commands = [('resolved_config', [str(args.env/'bin/python'), '-c',
            'import json,sys; from run_arc_pipege_best import resolved_config_check; '
            'resolved_config_check(sys.argv[1],sys.argv[2],json.loads(sys.argv[3]))',
            str(path), str(case/'data'), json.dumps(spec)]),
            ('train', [str(args.build/'gege_train'), str(path)])]
        value = dict(status='running', spec=spec, sampling_fix=args.fix_sampling,
                     changed='synthetic graph dimensions/paths; sampling fix if requested')
        report['cases'][name] = value
        for stage, command in commands:
            with (case/(stage+'.log')).open('w') as log:
                rc = subprocess.run(command, env=dict(case_env, PYTHONPATH=str(scripts)+':'+str(overlay)),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
            value.update(stage=stage, exit_code=rc)
            if rc:
                value['status'] = 'failed'
                break
        if value['exit_code'] == 0:
            text = (case/'train.log').read_text()
            try:
                training_check(text, spec, 2)
                value['status'] = 'pass'
            except ValueError as error:
                value.update(status='failed', validation_error=repr(error))
        print(name, json.dumps(value), flush=True)
        (args.output/'result.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if any(v['status'] != 'pass' for v in report['cases'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
