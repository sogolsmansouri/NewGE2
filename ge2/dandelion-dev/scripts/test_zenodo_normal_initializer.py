#!/usr/bin/env python3
"""Smoke-test the native CLI's Normal initializer on existing tiny fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml

from run_zenodo_fb_sampling_control import case_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures', 'out', 'env'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GEGE_', 'PYTHON', 'CONDA'))}
    site = args.env / 'lib/python3.9/site-packages'
    env.update(LD_LIBRARY_PATH=f'{site}/gege:{site}/torch/lib:{args.env}/lib',
               PATH=f'{args.env}/bin:/usr/bin:/bin', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4')
    report = dict(cases=[], passed=False)
    for name in ('distmult_512', 'complex_512'):
        template = yaml.safe_load((args.fixtures / (name + '.yaml')).read_text())
        data = Path(template['storage']['dataset']['dataset_dir'])
        assert yaml.safe_load((data / 'dataset.yaml').read_text())['num_nodes'] == 512
        path = args.out / name
        path.mkdir()
        config = case_config(template, data, path / 'model', .5, 'normal_0001')
        config['training']['num_epochs'] = 1
        config['training']['save_model'] = True
        config_path = path / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config))
        with (path / 'train.log').open('w') as log:
            subprocess.run([str(args.env / 'bin/gege_train'), str(config_path)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        saved = yaml.safe_load((path / 'model/full_config.yaml').read_text())
        init = saved['model']['encoder']['layers'][0][0]['init']
        assert init == dict(type='NORMAL', options=dict(mean=0., std=.001)), init
        assert 'Finished training epoch 1' in (path / 'train.log').read_text()
        for filename in ('embeddings.bin', 'embeddings_state.bin'):
            values = np.fromfile(path / 'model' / filename, dtype='<f4')
            assert values.size == 512 * 10 and np.isfinite(values).all()
        report['cases'].append(dict(case=name, initialization=init, passed=True))
    report['passed'] = True
    (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
