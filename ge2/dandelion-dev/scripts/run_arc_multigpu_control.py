#!/usr/bin/env python3
"""Run pinned four-GPU gates and accuracy controls, never final timing."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from arc_job_support import save_failure_evidence, write_json
from run_arc_multigpu_campaign import idle_devices, run_case
from run_arc_multigpu_finish import freeze_retry, source_campaign
from run_arc_paper_case import digest


CASES = ('pipege_tw_dot_4gpu', 'pipege_fb_complex_4gpu',
         'ge2_tw_dot_4gpu', 'ge2_fb_complex_4gpu')


def verify_payload(payload):
    if digest(payload/'payload_manifest.json') != (payload/'READY').read_text().strip():
        raise ValueError('Payload manifest checksum mismatch')
    metadata = json.loads((payload/'payload_manifest.json').read_text())
    for name, expected in metadata['files'].items():
        if digest(payload/name) != expected:
            raise ValueError('Frozen payload file changed: '+name)
    if metadata.get('power_w') not in (200, 300):
        raise ValueError('Explicit power cohort required')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--payload', type=Path, required=True)
    args = parser.parse_args()
    if Path(__file__).resolve().parent != args.payload.resolve():
        raise ValueError('Execute the frozen payload launcher')
    metadata = verify_payload(args.payload)
    source = source_campaign(metadata)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

    def stop(sig, frame):
        raise KeyboardInterrupt('Allocation interrupted: '+str(sig))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    job = os.environ['SLURM_JOB_ID']
    fields = dict(word.split('=', 1) for word in subprocess.check_output(
        ['scontrol', 'show', 'job', job, '-o'], text=True).split() if '=' in word)
    if (fields['JobState'] != 'RUNNING' or fields['NodeList'] != 'c30'
            or os.uname().nodename.split('.')[0] != 'c30'
            or not fields['UserId'].startswith(os.environ['USER']+'(')):
        raise RuntimeError('Requires owned running c30 allocation')
    deadline = datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-180
    base = args.base.resolve()
    if base.parent != Path('/mnt/local/smansou2'):
        raise ValueError('Dedicated node-local run directory required')
    base.mkdir(exist_ok=False)
    summary = Path('/home/smansou2/arc_results/runs')/base.name
    archive = Path('/mnt/beegfs/smansou2')/base.name
    summary.mkdir(exist_ok=False)
    archive.mkdir(exist_ok=False)
    state = dict(job=job, launcher_commit=metadata['commit'], control_only=True,
                 paper_ready=False, timing_eligible=False, status='preparing', cases={},
                 power_w=metadata['power_w'], source_campaign=str(source),
                 payload_manifest_sha256=digest(args.payload/'payload_manifest.json'),
                 protocol='Four-GPU 2-epoch gates then fresh 10-epoch tail-only accuracy controls')

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        for target in (summary, archive):
            write_json(target/'control_status.json', state)

    try:
        update()
        with Path('/mnt/local/smansou2/paper_multigpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            idle_devices(4)
            manifests = {}
            for name in CASES:
                case = base/name
                case.mkdir()
                (summary/name).mkdir()
                manifest = freeze_retry(source, case, args.payload, name, metadata['commit'])
                manifest.update(control_only=True, power_w=metadata['power_w'],
                                cohort='four_gpu_accuracy_control_'+job)
                write_json(case/'manifest.json', manifest)
                shutil.copy2(case/'manifest.json', summary/name/'manifest.json')
                manifests[name] = manifest
                state['cases'][name] = {}
            for phase in ('gate', 'control'):
                for name in CASES:
                    if phase == 'control' and state['cases'][name].get('gate', {}).get('status') != 'gate_passed':
                        continue
                    if deadline-time.time() < (1200 if phase == 'gate' else 1800):
                        update(status='waiting_next_allocation', active_case=None, stage=phase)
                        return
                    case = base/name
                    update(status='running', active_case=name, stage=phase)
                    try:
                        run_case(case, manifests[name], name, phase, deadline, archive/name,
                                 summary/name, diagnostic_gate=(phase == 'gate'),
                                 physical_devices=[0, 1, 2, 3])
                        state['cases'][name][phase] = json.loads(
                            (case/'results'/name/phase/'status.json').read_text())
                    except Exception as error:
                        state['cases'][name][phase] = dict(status='failed', error=repr(error))
                    finally:
                        for target in (summary/name, archive/name):
                            save_failure_evidence(case/'results', target/'evidence', total=64 << 20)
                        update()
            passed = all(state['cases'][name].get('control', {}).get('status') ==
                         'control_complete_not_timing' for name in CASES)
            update(status='completed_controls' if passed else 'incomplete_requires_review', active_case=None)
    except BaseException as error:
        update(status='interrupted_or_failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
