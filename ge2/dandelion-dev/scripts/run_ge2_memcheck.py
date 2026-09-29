#!/usr/bin/env python3
"""Capture the first invalid device access without CUDA_LAUNCH_BLOCKING/NCCL hangs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import yaml
from arc_job_support import run_logged, save_failure_evidence, write_json
from run_arc_multigpu_campaign import idle_node, native_score_environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--payload', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    a = p.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    def stop(sig, frame):
        raise KeyboardInterrupt('Allocation terminating')
    signal.signal(signal.SIGTERM, stop)
    a.base.mkdir(exist_ok=False)
    summary = Path('/home/smansou2/arc_results/runs')/a.base.name
    summary.mkdir(parents=True, exist_ok=True)
    old = Path('/mnt/local/smansou2/paper_multigpu_293571')
    manifest = json.loads((old/'manifest.json').read_text())
    spec = manifest['cases']['ge2_fb_complex_2gpu']
    prefix = Path(manifest['env'])
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'PYTHON', 'CUDA_', 'NCCL_'))}
    env.update(PATH=f'{prefix}/bin:/usr/bin:/bin', CUDA_VISIBLE_DEVICES='0,1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    env = native_score_environment(env, prefix)
    idle_node()
    data = a.base/'data'
    shutil.copytree(spec['data'], data)
    cfg = yaml.safe_load(Path(spec['config']).read_text())
    meta = yaml.safe_load((data/'dataset.yaml').read_text())
    meta['dataset_dir'] = str(data)+'/'
    (data/'dataset.yaml').write_text(yaml.safe_dump(meta))
    model = str(a.base/'model')+'/'
    cfg['storage'].update(dataset=meta, model_dir=model, checkpoint_dir=model)
    cfg['evaluation']['checkpoint_dir'] = model
    cfg['training'].update(num_epochs=1, save_model=False)
    config = a.base/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    write_json(summary/'status.json', dict(status='running', stage='compute_sanitizer', paper_ready=False))
    command = ['/usr/local/cuda/bin/compute-sanitizer', '--tool', 'memcheck', '--target-processes', 'all',
               '--error-exitcode', '99', '--print-limit', '12', '--destroy-on-device-error', 'kernel',
               str(prefix/'bin/python'), str(prefix/'bin/gege_train'), str(config)]
    rc = run_logged(command, env, a.base/'memcheck.log', 1200, a.base/'hardware.jsonl')
    save_failure_evidence(a.base, summary/'evidence')
    write_json(summary/'status.json', dict(status='diagnostic_finished', exit_code=rc, paper_ready=False))
    # Keep this diagnostic allocation available for a verified next step.
    until = time.time()+1800
    while time.time() < until:
        ready = a.payload/'NEXT_READY'
        if ready.exists():
            from run_arc_paper_case import digest
            script = a.payload/'next_ge2.py'
            if digest(script) != ready.read_text().strip():
                raise ValueError('Next-step hash mismatch')
            rc = run_logged([str(prefix/'bin/python'), str(script), '--base', str(a.base)], env,
                            a.base/'next.log', 14400)
            write_json(summary/'status.json', dict(status='next_step_finished', exit_code=rc, paper_ready=False))
            if rc:
                raise RuntimeError('Next step failed')
            return
        time.sleep(15)
    raise RuntimeError('Diagnosis complete; no next step staged')


if __name__ == '__main__':
    main()
