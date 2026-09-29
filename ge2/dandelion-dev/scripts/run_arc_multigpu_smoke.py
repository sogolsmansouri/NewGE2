#!/usr/bin/env python3
"""Detached, non-paper two-epoch gates on idle GPUs of an owned allocation."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from arc_job_support import save_failure_evidence, write_json
from run_arc_multigpu_campaign import idle_devices, run_case
from run_arc_multigpu_finish import freeze_retry
from run_arc_paper_case import digest


CASES = ('ge2_fb_complex_2gpu', 'pipege_fb_complex_2gpu',
         'pipege_tw_dot_2gpu', 'ge2_tw_dot_2gpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

    def stop(sig, frame):
        raise KeyboardInterrupt('Diagnostic interrupted: '+str(sig))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    job = os.environ['SLURM_JOB_ID']
    fields = dict(word.split('=', 1) for word in subprocess.check_output(
        ['scontrol', 'show', 'job', job, '-o'], text=True).split() if '=' in word)
    if (fields['JobState'] != 'RUNNING' or fields['NodeList'] != 'c30'
            or os.uname().nodename.split('.')[0] != 'c30'
            or not fields['UserId'].startswith(os.environ['USER']+'(')):
        raise RuntimeError('Requires the user\'s running c30 allocation')
    deadline = datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-180
    payload = args.payload.resolve()
    metadata = json.loads((payload/'payload_manifest.json').read_text())
    if digest(payload/'payload_manifest.json') != (payload/'READY').read_text().strip():
        raise ValueError('Payload manifest checksum mismatch')
    for name, sha in metadata['files'].items():
        if digest(payload/name) != sha:
            raise ValueError('Payload checksum mismatch: '+name)
    base = args.base.resolve()
    if base.parent != Path('/mnt/local/smansou2'):
        raise ValueError('Dedicated node-local diagnostic directory required')
    base.mkdir(exist_ok=False)
    summary = Path('/home/smansou2/arc_results/runs')/base.name
    archive = Path('/mnt/beegfs/smansou2')/base.name
    summary.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    status = dict(job=job, status='preflight', diagnostic_only=True, paper_ready=False,
                  launcher_commit=metadata['commit'], cases={}, gpus=[0, 1],
                  four_gpu_status='not_tested_requires_four_idle_GPUs')

    def update(**changes):
        status.update(changes, updated=datetime.datetime.now().isoformat())
        for target in (summary, archive):
            write_json(target/'smoke_status.json', status)

    try:
        update()
        with Path('/mnt/local/smansou2/paper_multigpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            idle_devices(2)
            for name in CASES:
                if deadline-time.time() < 600:
                    raise RuntimeError('Insufficient remaining allocation time')
                update(status='running', active_case=name)
                case = base/name
                case.mkdir()
                evidence = summary/name
                evidence.mkdir()
                try:
                    manifest = freeze_retry(Path('/mnt/local/smansou2/paper_multigpu_293571'),
                                            case, payload, name, metadata['commit'])
                    shutil.copy2(case/'manifest.json', evidence/'manifest.json')
                    sys.path.insert(0, str(case/'harness/tools'))
                    run_case(case, manifest, name, 'gate', deadline, archive/name, evidence,
                             diagnostic_gate=True)
                    result = json.loads((case/'results'/name/'gate/status.json').read_text())
                    status['cases'][name] = result
                except Exception as error:
                    status['cases'][name] = dict(status='failed', error=repr(error))
                finally:
                    save_failure_evidence(case/'results', evidence/'evidence', total=32 << 20)
                    save_failure_evidence(case/'results', archive/name/'evidence', total=32 << 20)
                    update()
            passed = all(row.get('status') == 'gate_passed' for row in status['cases'].values())
            update(status='passed' if passed else 'failed', active_case=None)
            if not passed:
                raise RuntimeError('One or more two-GPU gates failed; see smoke_status.json')
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
