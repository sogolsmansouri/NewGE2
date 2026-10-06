#!/usr/bin/env python3
"""Detached FB ComplEx two/four-GPU training with gates and checkpoint archives."""
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
from check_fb_multigpu_training_parity import validate_execution_scope
from run_workstation_fb_accuracy_control import allocation_seconds, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'evidence', 'archive', 'binary', 'source', 'prefix',
                 'data16', 'data32', 'tools', 'template', 'flags', 'fixture'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--job', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--launcher-commit', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, choices=(2, 4), default=[2, 4])
    args = parser.parse_args()
    validate_execution_scope(args.job, None)
    if args.work.resolve().parent != Path('/mnt/local/smansou2'):
        raise ValueError('Use a dedicated node-local work directory')
    if shutil.disk_usage(args.work.parent).free < 160 * 2**30:
        raise RuntimeError('Need 160 GiB free to retain both complete checkpoints')
    for path in (args.work, args.evidence, args.archive):
        path.mkdir(parents=True, exist_ok=False)
    payload = Path(__file__).resolve().parent
    python = args.prefix/'bin/python'
    state = dict(status='starting', job=args.job, host=os.uname().nodename,
                 native_commit=args.commit, launcher_commit=args.launcher_commit,
                 supervisor_pid=os.getpid(), child_pid=None, cases={},
                 decoder='complex', p=32, q=4, shared_hidden_frames=6, max_stale_backlog=3,
                 batch_per_gpu=50000, epochs=10, queries=10000, power_w=200,
                 evaluation='Full-catalog filtered tail-only, pessimistic ties, TF32 disabled',
                 paper_ready=False, timing_status='Requires review of isolation and complete wall times',
                 binary_sha256=digest(args.binary), library_sha256=digest(args.binary.parent/'libge2.so'))
    child = None

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().astimezone().isoformat())
        for path in (args.work, args.evidence, args.archive):
            write_json(path/'status.json', state)

    def stop(signum, frame):
        raise KeyboardInterrupt('Supervisor signal '+str(signum))

    def run(command, label, devices):
        nonlocal child
        update(status='running', active_case=label)
        env = dict(os.environ, SLURM_JOB_ID=args.job,
                   CUDA_VISIBLE_DEVICES=','.join(map(str, range(devices))), CUDA_DEVICE_ORDER='PCI_BUS_ID')
        with (args.evidence/(label+'.supervisor.log')).open('x') as log:
            child = subprocess.Popen(list(map(str, command)), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(child_pid=child.pid)
            try:
                while child.poll() is None:
                    validate_execution_scope(args.job, None)
                    allocation_seconds(args.job)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                child = None
                update(child_pid=None)
        if code:
            raise RuntimeError(label+' exited '+str(code))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    update()
    try:
        with Path('/mnt/local/smansou2/paper_multigpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gates = {}
            for count in args.gpus:
                if allocation_seconds(args.job) < 3600:
                    update(status='waiting_next_allocation', active_case=None)
                    return
                label = 'peer_'+str(count)+'gpu'
                gate_work = args.work/('gate_'+label)
                gate_cmd = [python, '-B', '-u', payload/'check_fb_multigpu_training_parity.py',
                            '--binary', args.binary, '--source', args.source, '--prefix', args.prefix,
                            '--template', args.fixture/'distmult_autograd/config.yaml',
                            '--flags', args.fixture/'distmult_autograd/flags.json',
                            '--data', args.fixture/'data', '--work', gate_work, '--job', args.job,
                            '--gpus', str(count), '--visible', '4', '--decoder', 'COMPLEX',
                            '--transport', 'peer', '--peer-scratch', 'shared']
                try:
                    run(gate_cmd, 'gate_'+label, count)
                finally:
                    save_failure_evidence(gate_work, args.evidence/('gate_'+label))
                gate = gate_work/'result.json'
                if json.loads(gate.read_text())['status'] != 'passed':
                    raise RuntimeError('Failed native parity gate: '+label)
                gates[count] = gate
            for count in args.gpus:
                if allocation_seconds(args.job) < 3600:
                    update(status='waiting_next_allocation', active_case=None)
                    return
                label = 'peer_'+str(count)+'gpu'
                gate = gates[count]
                work = args.work/label
                command = [python, '-B', '-u', payload/'run_workstation_fb_accuracy_control.py',
                           '--binary', args.binary, '--source', args.source, '--prefix', args.prefix,
                           '--data16', args.data16, '--data32', args.data32, '--tools', args.tools,
                           '--template', args.template, '--flags', args.flags, '--gate', gate,
                           '--job', args.job, '--commit', args.commit, '--gpu', '0', '--gpus', str(count),
                           '--partitions', '32', '--visible', '4', '--decoder', 'complex',
                           '--pipeline', 'on', '--gradients', 'manual', '--transport', 'peer',
                           '--peer-scratch', 'shared', '--epochs', '10', '--eval-queries', '10000',
                           '--expected-power', '200', '--work', work, '--evidence', args.evidence/label]
                run(command, label, count)
                progress = json.loads((work/'progress.json').read_text())
                if progress['status'] != 'done':
                    raise RuntimeError('Incomplete training or evaluation: '+label)
                update(status='archiving', active_case=label)
                destination = args.archive/label
                destination.mkdir()
                shutil.copytree(work/'model', destination/'model')
                entries = []
                for source in sorted((work/'model').iterdir()):
                    if not source.is_file():
                        continue
                    target = destination/'model'/source.name
                    expected = digest(source)
                    if expected != digest(target):
                        raise RuntimeError('Checkpoint archive mismatch: '+source.name)
                    entries.append(dict(path=source.name, bytes=source.stat().st_size, sha256=expected))
                receipt = dict(status='verified', source=str(work/'model'), destination=str(destination/'model'),
                               files=entries, native_commit=args.commit, binary_sha256=state['binary_sha256'])
                write_json(args.evidence/label/'archive_receipt.json', receipt)
                save_failure_evidence(work, destination/'evidence')
                for name in ('train.log', 'train.hardware.jsonl', 'evaluate.log'):
                    source = work/name
                    if source.is_file():
                        shutil.copy2(source, destination/'evidence'/name)
                        shutil.copy2(source, args.evidence/label/name)
                shutil.copy2(work/'exact_eval.ranks.npz', destination/'evidence/exact_eval.ranks.npz')
                state['cases'][label] = {key: progress[key] for key in
                    ('epoch_times_s', 'average_epoch_s', 'steady_epoch_s', 'mrr', 'hits_at_10',
                     'workload_audit', 'transport_counts', 'dense_replica_comparison')}
                state['cases'][label]['epoch_timing'] = progress['epoch_timing']
                state['cases'][label]['checkpoint_archive'] = str(destination/'model')
                update()
            update(status='complete', active_case=None)
    except BaseException as error:
        update(status='failed_or_interrupted', error=repr(error))
        raise


if __name__ == '__main__':
    main()
