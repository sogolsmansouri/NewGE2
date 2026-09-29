#!/usr/bin/env python3
"""Run one gated final case from the frozen multi-GPU campaign, without engine patches."""
import argparse
import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from arc_job_support import run_logged, save_failure_evidence, write_json
from run_arc_multigpu_campaign import idle_node, native_score_environment, run_case, runtime_environment
from run_arc_paper_case import digest


CASES = ('ge2_fb_complex_2gpu', 'pipege_fb_complex_2gpu', 'pipege_tw_dot_4gpu',
         'ge2_tw_dot_4gpu', 'pipege_fb_complex_4gpu', 'ge2_fb_complex_4gpu')


def freeze_retry(old, base, payload, name, launcher_commit):
    prior = json.loads((old/'manifest.json').read_text())
    if prior.get('ge2_dense_repair'):
        raise ValueError('Final campaign must not load the experimental trainer correction')
    for rel, expected in prior['files'].items():
        if digest(old/rel) != expected:
            raise ValueError('Frozen campaign file changed: '+rel)
    manifest = copy.deepcopy(prior)
    for folder in ('references', 'harness'):
        shutil.copytree(old/folder, base/folder)
    shutil.copytree(payload, base/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(old/'ge2.zip', base/'ge2.zip')
    (base/'engine').symlink_to((old/'engine').resolve(), target_is_directory=True)
    spec = copy.deepcopy(prior['cases'][name])
    for key in ('config', 'flags'):
        spec[key] = str(base/Path(spec[key]).relative_to(old))
    flags = json.loads(Path(spec['flags']).read_text())
    schedule = flags.get('GEGE_BOUNDED_STATE_ORDER_FILE')
    if schedule:
        flags['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(base/Path(schedule).relative_to(old))
    write_json(Path(spec['flags']), flags)
    manifest.update(cases={name: spec}, launcher_commit=launcher_commit,
                    source_campaign=str(old), source_manifest_sha256=digest(old/'manifest.json'),
                    runtime_policy='c30_nccl_shm_v1' if spec['graph'] == 'fb' else 'default',
                    native_build_policy='Reuse the hash-verified training build of the completed campaign')
    manifest['files'] = {str(p.relative_to(base)):digest(p)
                        for folder in ('references', 'harness', 'scripts')
                        for p in (base/folder).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts}
    runtime_environment(manifest)
    write_json(base/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--case', required=True, choices=CASES)
    args = parser.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

    def stop(sig, frame):
        raise KeyboardInterrupt('Allocation termination: '+str(sig))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    job = os.environ['SLURM_JOB_ID']
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if ('JobState=RUNNING ' not in allocation or 'NodeList=c30 ' not in allocation
            or os.uname().nodename.split('.')[0] != 'c30'
            or 'UserId='+os.environ['USER']+'(' not in allocation):
        raise RuntimeError('Requires owned running c30 allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-180
    payload_manifest = json.loads((args.payload/'payload_manifest.json').read_text())
    if digest(args.payload/'payload_manifest.json') != (args.payload/'READY').read_text().strip():
        raise ValueError('Payload manifest checksum mismatch')
    for name, expected in payload_manifest['files'].items():
        if digest(args.payload/name) != expected:
            raise ValueError('Payload checksum mismatch: '+name)
    base = args.base.resolve()
    if base.parent != Path('/mnt/local/smansou2'):
        raise ValueError('Dedicated node-local run directory required')
    base.mkdir(exist_ok=False)
    summary = Path('/home/smansou2/arc_results/runs')/base.name
    archive = Path('/mnt/beegfs/smansou2')/base.name
    summary.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    state = dict(job=job, case=args.case, status='preflight', paper_ready=False,
                 launcher_commit=payload_manifest['commit'])

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(summary/'campaign_status.json', state)
        write_json(archive/'campaign_status.json', state)

    try:
        update()
        with Path('/mnt/local/smansou2/paper_multigpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Allow the preceding allocation's CUDA contexts to disappear.
            samples = 0
            for attempt in range(40):
                try:
                    idle_node()
                    samples += 1
                except RuntimeError:
                    samples = 0
                if samples == 3:
                    break
                time.sleep(15)
            else:
                raise RuntimeError('Node did not become exclusively idle within ten minutes')
            old = Path('/mnt/local/smansou2/paper_multigpu_293571')
            manifest = freeze_retry(old, base, args.payload, args.case, payload_manifest['commit'])
            for target in (summary, archive):
                shutil.copy2(base/'manifest.json', target/'manifest.json')
            sys.path.insert(0, str(base/'harness/tools'))
            if args.case.startswith('pipege_'):
                prefix = Path(manifest['env'])
                env = native_score_environment(dict(os.environ), prefix)
                env['LD_LIBRARY_PATH'] = f'{base}/engine/build_git:'+env['LD_LIBRARY_PATH']
                for test in ('gege_stateflow_validator_tests', 'gege_manual_backward_test',
                             'gege_manual_training_update_test'):
                    update(stage=test)
                    rc = run_logged([str(base/'engine/build_git'/test)], env,
                                    summary/(test+'.log'), min(600, deadline-time.time()))
                    if rc:
                        raise RuntimeError(f'{test} exited {rc}')
            for phase in ('gate', 'final'):
                update(status='running', stage=phase, runtime_policy=manifest['runtime_policy'])
                run_case(base, manifest, args.case, phase, deadline, archive, summary)
            update(status='done_pending_review', stage='complete')
    except BaseException as error:
        update(status='failed', error=repr(error))
        save_failure_evidence(base/'results', summary/'failure')
        save_failure_evidence(base/'results', archive/'failure')
        raise


if __name__ == '__main__':
    main()
