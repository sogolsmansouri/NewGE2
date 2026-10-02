#!/usr/bin/env python3
"""Build pinned PipeGE source and run two-epoch, non-paper multi-GPU gates."""
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

from arc_accuracy_gpu_guard import guarded_run
from arc_job_support import save_failure_evidence, write_json
from run_arc_multigpu_campaign import diagnostic_devices, idle_devices, prepare, run_case
from run_arc_multigpu_finish import freeze_retry


def stage_scripts(source, destination):
    # Do not follow historical machine-local package/build symlinks in scripts/.
    destination.mkdir(exist_ok=False)
    for path in source.glob('*.py'):
        if path.is_symlink():
            raise ValueError('Launcher scripts must be regular source files: '+str(path))
        shutil.copy2(path, destination/path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--devices', required=True, type=int, nargs='+')
    parser.add_argument('--power-w', required=True, type=int)
    args = parser.parse_args()
    devices = diagnostic_devices(len(args.devices), args.devices)
    base = args.base.resolve()
    repo = base/'engine/repo'
    scripts = repo/'ge2/dandelion-dev/scripts'
    if base.parent != Path('/mnt/local/smansou2') or Path(__file__).resolve().parent != scripts:
        raise ValueError('Run the pinned launcher from its dedicated node-local checkout')
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})

    def stop(sig, frame):
        raise KeyboardInterrupt('Diagnostic allocation interrupted: '+str(sig))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    job = os.environ['SLURM_JOB_ID']
    fields = dict(word.split('=', 1) for word in subprocess.check_output(
        ['scontrol', 'show', 'job', job, '-o'], text=True).split() if '=' in word)
    if (fields['JobState'] != 'RUNNING' or fields['NodeList'] != 'c30'
            or os.uname().nodename.split('.')[0] != 'c30'
            or not fields['UserId'].startswith(os.environ['USER']+'(')):
        raise RuntimeError('Requires the owned running c30 allocation')
    deadline = datetime.datetime.fromisoformat(fields['EndTime']).timestamp()-180
    summary = Path('/home/smansou2/arc_results/runs')/base.name
    archive = Path('/mnt/beegfs/smansou2')/base.name
    summary.mkdir(parents=True, exist_ok=False)
    archive.mkdir(parents=True, exist_ok=False)
    state = dict(job=job, commit=args.commit, devices=devices, power_w=args.power_w,
                 diagnostic_only=True, paper_ready=False, status='preparing', cases={})

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        for target in (summary, archive):
            write_json(target/'diagnostic_status.json', state)

    def execute(command, env, label):
        update(stage=label)
        env = dict(env, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=','.join(map(str, devices)))
        env.pop('LD_PRELOAD', None)
        idle_devices(len(devices), devices)
        rc = guarded_run(list(map(str, command)), env, summary/(label+'.log'),
                         deadline-time.time(), ','.join(map(str, devices)),
                         summary/(label+'.gpu_guard.jsonl'))
        if rc:
            raise RuntimeError(f'{label} exited {rc}')

    try:
        update()
        with Path('/mnt/local/smansou2/paper_multigpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            idle_devices(len(devices), devices)
            for gpu in devices:
                power = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=power.limit', '--format=csv,noheader,nounits'], text=True)
                if float(power.strip()) != args.power_w:
                    raise ValueError('Diagnostic power cohort does not match selected GPU')
            stage_scripts(scripts, base/'scripts')
            prepare(base, args.commit, execute)
            update(status='testing', stage='native_tests_passed')
            for workload in ('fb_complex', 'tw_dot'):
                name = f'pipege_{workload}_{len(devices)}gpu'
                if deadline-time.time() < 1200:
                    raise RuntimeError('Not enough allocation time for the next full-data gate')
                case, evidence = base/'cases'/name, summary/name
                case.mkdir(parents=True)
                evidence.mkdir()
                update(active_case=name)
                try:
                    manifest = freeze_retry(base, case, base/'scripts', name, args.commit)
                    manifest.update(power_w=args.power_w, diagnostic_only=True,
                                    native_build_policy='Fresh native build from the pinned source commit')
                    write_json(case/'manifest.json', manifest)
                    shutil.copy2(case/'manifest.json', evidence/'manifest.json')
                    run_case(case, manifest, name, 'gate', deadline, archive/name, evidence,
                             diagnostic_gate=True, physical_devices=devices)
                    state['cases'][name] = json.loads((case/'results'/name/'gate/status.json').read_text())
                except Exception as error:
                    state['cases'][name] = dict(status='failed', error=repr(error))
                finally:
                    for target in (evidence, archive/name):
                        save_failure_evidence(case/'results', target/'evidence', total=64 << 20)
                    update()
            passed = all(row.get('status') == 'gate_passed' for row in state['cases'].values())
            update(status='passed' if passed else 'failed', active_case=None)
            if not passed:
                raise RuntimeError('One or more gates failed; inspect diagnostic_status.json')
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
