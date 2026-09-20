#!/usr/bin/env python3
"""Integration check: a fixed-mode diagnostic must reproduce the original trainer."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'out', 'binary', 'env'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GEGE_NO_BINDINGS='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    site = args.env / 'lib/python3.9/site-packages'
    env.update(LD_LIBRARY_PATH=f'{site}/gege:{site}/torch/lib:{args.env}/lib', PYTHONPATH=str(site))
    report = dict(cases=[], passed=False)
    for name in ('distmult_512', 'complex_512'):
        template = yaml.safe_load((args.fixtures / (name + '.yaml')).read_text())
        models = []
        for mode in ('original', 'original_repeat', 'fixed'):
            path = args.out / (name + '_' + mode)
            path.mkdir()
            config = copy.deepcopy(template)
            config['storage']['model_dir'] = str(path / 'model') + '/'
            config['storage']['checkpoint_dir'] = config['storage']['model_dir']
            config['training']['save_model'] = True
            config_path = path / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            command = ([str(args.env / 'bin/gege_train'), str(config_path)] if mode.startswith('original')
                       else [str(args.binary), str(config_path), 'fixed'])
            child_env = dict(env)
            if mode.startswith('original'):
                child_env.pop('GEGE_NO_BINDINGS', None)
            with (path / 'train.log').open('w') as log:
                subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            models.append(path / 'model')
        # Multi-step GPU scatter reductions are not bitwise deterministic. Use the
        # independent native-math audit's tolerance, and retain ordinary-repeat noise.
        entry = dict(case=name, comparisons={}, rtol=2e-4, atol=2e-5)
        for label, other in (('ordinary_repeat', models[1]), ('fixed_driver', models[2])):
            checks = {}
            for filename in ('embeddings.bin', 'embeddings_state.bin'):
                a, b = [np.fromfile(m / filename, dtype='<f4') for m in (models[0], other)]
                checks[filename] = float(np.max(np.abs(a - b)))
                np.testing.assert_allclose(a, b, rtol=entry['rtol'], atol=entry['atol'])
            a, b = [torch.jit.load(str(m / 'model.pt_0'), map_location='cpu').state_dict()
                    for m in (models[0], other)]
            assert a.keys() == b.keys()
            for key in a:
                checks[key] = float((a[key] - b[key]).abs().max())
                torch.testing.assert_close(a[key], b[key], rtol=entry['rtol'], atol=entry['atol'])
            entry['comparisons'][label] = checks
        report['cases'].append(entry)
    report['passed'] = True
    (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
