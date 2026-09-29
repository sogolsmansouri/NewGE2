#!/usr/bin/env python3
"""Bounded environment-only controls of the frozen Zenodo GE2 release."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time

import yaml

from arc_job_support import run_logged, save_failure_evidence, write_json
from run_arc_multigpu_campaign import idle_node, native_score_environment
from run_arc_paper_case import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

    def stop(sig, frame):
        raise KeyboardInterrupt('Allocation terminating')

    signal.signal(signal.SIGTERM, stop)
    old = Path('/mnt/local/smansou2/paper_multigpu_293571')
    manifest = json.loads((old/'manifest.json').read_text())
    spec = manifest['cases']['ge2_fb_complex_2gpu']
    prefix = Path(manifest['env'])
    archive = old/'ge2.zip'
    library = prefix/'lib/python3.9/site-packages/gege/libge2.so'
    if hashlib.md5(archive.read_bytes()).hexdigest() != '6de3d9702241a0c822971939752d0834':
        raise ValueError('Zenodo archive identity mismatch')
    if digest(library) != manifest['ge2_library_sha256']:
        raise ValueError('Released library identity mismatch')
    root = args.base/'original_runtime_controls'
    root.mkdir(exist_ok=False)
    summary = Path('/home/smansou2/arc_results/runs')/args.base.name/'original_runtime_controls'
    summary.mkdir(parents=True, exist_ok=True)
    persistent = Path('/mnt/beegfs/smansou2')/args.base.name/'original_runtime_controls'
    persistent.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', paper_ready=False, job=os.environ['SLURM_JOB_ID'],
                 original_library_sha256=digest(library), archive_sha256=digest(archive),
                 script_sha256=digest(Path(__file__)), cases=[])

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(summary/'status.json', state)
        write_json(persistent/'status.json', state)

    try:
        for name, overrides in (
                ('nccl_without_p2p', {'NCCL_P2P_DISABLE': '1'}),
                ('uncached_cuda_allocations', {'PYTORCH_NO_CUDA_MEMORY_CACHING': '1'})):
            idle_node()
            work = root/name
            work.mkdir()
            data = work/'data'
            shutil.copytree(spec['data'], data)
            cfg = yaml.safe_load(Path(spec['config']).read_text())
            meta = yaml.safe_load((data/'dataset.yaml').read_text())
            meta['dataset_dir'] = str(data)+'/'
            (data/'dataset.yaml').write_text(yaml.safe_dump(meta))
            model = str(work/'model')+'/'
            cfg['storage'].update(dataset=meta, model_dir=model, checkpoint_dir=model)
            cfg['evaluation']['checkpoint_dir'] = model
            cfg['training'].update(num_epochs=1, save_model=False)
            config = work/'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            env = {k:v for k,v in os.environ.items()
                   if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'PYTORCH_', 'CUDA_', 'NCCL_'))}
            env.update(PATH=f'{prefix}/bin:/usr/bin:/bin', CUDA_VISIBLE_DEVICES='0,1',
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                       OPENBLAS_NUM_THREADS='1', NCCL_DEBUG='INFO')
            env = native_score_environment(env, prefix)
            env.update(overrides)
            update(case=name, environment_overrides=overrides)
            write_json(work/'provenance.json', dict(
                classification='Unmodified released GE2; environment-only diagnostic, not timing result',
                archive_sha256=state['archive_sha256'], library_sha256=state['original_library_sha256'],
                config_sha256=digest(config), environment_overrides=overrides, ld_preload=None))
            rc = run_logged([str(prefix/'bin/python'), str(prefix/'bin/gege_train'), str(config)],
                            env, work/'train.log', 600, work/'hardware.jsonl')
            times = [int(t)/1000 for t in re.findall(r'Epoch Runtime:\s*(\d+)ms', (work/'train.log').read_text())]
            state['cases'].append(dict(name=name, exit_code=rc, epochs_observed=len(times),
                                      epoch_times_s=times, pass_one_epoch=rc == 0 and len(times) == 1))
            save_failure_evidence(work, summary/name)
            save_failure_evidence(work, persistent/name)
            update()
            time.sleep(15)
        update(status='diagnostics_completed')
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
