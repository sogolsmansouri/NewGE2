#!/usr/bin/env python3
"""Serialize frozen single-GPU FB ComplEx controls on an authorized workstation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

from arc_job_support import write_json


QUEUES = {
    0: [('p16_sync_manual', 16, 'off', 'manual'),
        ('p32_sync_autograd', 32, 'off', 'autograd')],
    1: [('p32_pipeline_manual', 32, 'on', 'manual'),
        ('p32_sync_manual', 32, 'off', 'manual')],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=QUEUES, required=True)
    parser.add_argument('--commit', required=True, help='Tested native source commit')
    parser.add_argument('--launcher-commit', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    payload = Path(__file__).resolve().parent
    host = 'eb2-3224-lin01.csc.ncsu.edu'
    if os.uname().nodename != host:
        raise RuntimeError('This queue is scoped to the authorized A5000 workstation')
    runner = payload/'run_workstation_fb_accuracy_control.py'
    ledger = root/('single_gpu_queue_5521fe0_gpu'+str(args.gpu)+'.json')
    if ledger.exists():
        raise RuntimeError('Refusing to overwrite an existing queue ledger')
    state = dict(status='starting', gpu=args.gpu, decoder='complex', commit=args.commit,
                 launcher_commit=args.launcher_commit,
                 supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                 paper_ready=False, timing_eligible=False, cases={})
    child = None

    def save(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(ledger, state)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=120)
        save(status='interrupted', signal=signum)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    common = [str(root/'ge2-a6000-cuda121/bin/python'), '-B', '-u', str(runner),
              '--binary', str(root/'build_current/gege_train'),
              '--source', str(root/'source_final/ge2/dandelion-dev/gege'),
              '--prefix', str(root/'ge2-a6000-cuda121'),
              '--data16', str(root/'data/full_fb_p16'), '--data32', str(root/'data/full_fb_p32'),
              '--tools', str(root/'tools'), '--template', str(root/'templates/config.yaml'),
              '--flags', str(root/'templates/flags.json'), '--gate', str(root/'single_gpu_gate_5521fe0/result.json'),
              '--workstation-host', host, '--commit', args.commit,
              '--gpu', str(args.gpu), '--visible', '4', '--decoder', 'complex', '--replay-seed', '17']
    save()
    for name, partitions, pipeline, gradients in QUEUES[args.gpu]:
        work = root/('full_fb_complex_'+name+'_5521fe0')
        command = common+['--partitions', str(partitions), '--pipeline', pipeline,
                         '--gradients', gradients, '--work', str(work),
                         '--evidence', str(root/'single_gpu_evidence_5521fe0'/name)]
        if work.exists():
            raise RuntimeError('Refusing to reuse a training directory: '+str(work))
        save(status='running', active_case=name, command=command)
        with (root/(work.name+'.supervisor.log')).open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
            save(child_pid=child.pid)
            code = child.wait()
        state['cases'][name] = dict(exit_code=code, work=str(work))
        child = None
        save(child_pid=None)
    save(status='complete' if all(row['exit_code'] == 0 for row in state['cases'].values()) else 'failed_cases',
         active_case=None)


if __name__ == '__main__':
    main()
