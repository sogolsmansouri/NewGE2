#!/usr/bin/env python3
"""Detached, gated one/two-GPU FB ComplEx accuracy screen on the authorized host."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from arc_job_support import save_failure_evidence, write_json
from run_workstation_fb_accuracy_control import digest


CASES = (('single', 1, 'host'), ('peer', 2, 'peer'), ('host', 2, 'host'))
HOST = 'eb2-3224-lin01.csc.ncsu.edu'


def summarize(work):
    result = {}
    queries = None
    ranks = {}
    states = {}
    for name, gpus, transport in CASES:
        case = work/name
        state = json.loads((case/'progress.json').read_text())
        if state['status'] != 'done' or state['epochs_completed'] != 3:
            raise ValueError('Incomplete comparison case: '+name)
        with np.load(case/'exact_eval.ranks.npz') as saved:
            current_queries = saved['triples']
            ranks[name] = saved['tail_ranks'].astype(np.float64)
        if queries is not None and not np.array_equal(queries, current_queries):
            raise ValueError('Evaluation query triples differ')
        queries = current_queries
        states[name] = state
        result[name] = dict(gpus=gpus, transport=transport, mrr=float(np.mean(1/ranks[name])),
                            hits_at_10=float(np.mean(ranks[name] <= 10)),
                            epoch_times_s=state['epoch_times_s'],
                            entity_sha256=json.loads((case/'exact_eval.json').read_text())['entity_bin_sha256'])
    result['transport_comparison'] = dict(
        identical_inputs=states['peer']['input_trace'] == states['host']['input_trace'],
        identical_entity_weights=result['peer']['entity_sha256'] == result['host']['entity_sha256'],
        identical_tail_ranks=bool(np.array_equal(ranks['peer'], ranks['host'])),
        peer_minus_single_mrr=result['peer']['mrr']-result['single']['mrr'],
        host_minus_single_mrr=result['host']['mrr']-result['single']['mrr'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'after', 'binary', 'source', 'single-gate', 'fixture'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--native-commit', required=True)
    parser.add_argument('--launcher-commit', required=True)
    args = parser.parse_args()
    if os.uname().nodename != HOST:
        raise RuntimeError('Only the explicitly authorized workstation may run this queue')
    if args.work.parent != Path('/tmp')/os.environ['USER']:
        raise ValueError('Use a dedicated directory on the workstation local scratch filesystem')
    if shutil.disk_usage(args.work.parent).free < 240*2**30:
        raise RuntimeError('Need 240 GiB free to retain all three checkpoints')
    args.work.mkdir(exist_ok=False)
    evidence = args.root/'evidence'/args.work.name
    evidence.mkdir(exist_ok=False)
    payload = Path(__file__).resolve().parent
    prefix = args.root/'ge2-a6000-cuda121'
    python = prefix/'bin/python'
    state = dict(status='starting', native_commit=args.native_commit, launcher_commit=args.launcher_commit,
                 supervisor_pid=os.getpid(), child_pid=None, cases={}, gates={}, paper_ready=False,
                 timing_eligible=False, host=HOST, work=str(args.work), evidence=str(evidence),
                 p=32, q=4, shared_hidden_frames=6, max_stale_backlog=3, decoder='complex',
                 batch_per_gpu=50000, epochs=3, eval_queries=1000, replay_seed=17,
                 protocol='Same data, model seed, optimizer, negative mixture and frozen tail queries; '
                          '50K per GPU means up to 100K per dense synchronization on two GPUs.',
                 driver_sha256=digest(payload/'run_workstation_fb_accuracy_control.py'))
    child = None

    def save(**changes):
        state.update(changes, updated=datetime.datetime.now().astimezone().isoformat())
        for root in (args.work, evidence):
            write_json(root/'status.json', state)

    def stop(signum, frame):
        raise KeyboardInterrupt('Supervisor signal '+str(signum))

    def run(command, label, env=None):
        nonlocal child
        save(status='running', active_case=label, command=list(map(str, command)))
        with (evidence/(label+'.supervisor.log')).open('x') as log:
            child = subprocess.Popen(list(map(str, command)), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(child_pid=child.pid)
            code = child.wait()
        child = None
        save(child_pid=None)
        if code:
            raise RuntimeError(label+' exited '+str(code))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    try:
        save(status='waiting_existing_single_gpu_queue', dependency=str(args.after))
        deadline = time.monotonic()+3*3600
        while True:
            prior = json.loads(args.after.read_text())
            if prior['status'] == 'complete':
                break
            if prior['status'] not in ('running', 'starting'):
                raise RuntimeError('Preceding queue did not complete: '+prior['status'])
            if time.monotonic() > deadline:
                raise TimeoutError('Preceding queue did not finish within three hours')
            time.sleep(15)
        save(status='waiting_idle_gpus')
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader'], text=True).strip():
            if time.monotonic() > deadline:
                raise TimeoutError('GPUs did not become idle; no process was displaced')
            time.sleep(15)
        gate_runner = payload/'check_fb_multigpu_training_parity.py'
        gate_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
        for transport in ('peer', 'host'):
            gate_work = args.work/('gate_'+transport)
            command = [python, '-B', '-u', gate_runner, '--binary', args.binary,
                       '--source', args.source, '--prefix', prefix,
                       '--template', args.fixture/'complex_manual/config.yaml',
                       '--flags', args.fixture/'complex_manual/flags.json',
                       '--data', args.fixture/'data', '--work', gate_work,
                       '--workstation-host', HOST, '--gpus', '2', '--visible', '4',
                       '--decoder', 'COMPLEX', '--transport', transport, '--peer-scratch', 'shared']
            try:
                run(command, 'gate_'+transport, gate_env)
            finally:
                save_failure_evidence(gate_work, evidence/('gate_'+transport))
            gate = json.loads((gate_work/'result.json').read_text())
            if gate['status'] != 'passed':
                raise RuntimeError('Native training-parity gate failed: '+transport)
            state['gates'][transport] = dict(status='passed', path=str(gate_work/'result.json'))
            save()
        for name, gpus, transport in CASES:
            gate = args.single_gate if gpus == 1 else args.work/('gate_'+transport)/'result.json'
            command = [python, '-B', '-u', payload/'run_workstation_fb_accuracy_control.py',
                       '--binary', args.binary, '--source', args.source, '--prefix', prefix,
                       '--data16', args.root/'data/full_fb_p16', '--data32', args.root/'data/full_fb_p32',
                       '--tools', args.root/'tools', '--template', args.root/'templates/config.yaml',
                       '--flags', args.root/'templates/flags.json', '--gate', gate,
                       '--workstation-host', HOST, '--commit', args.native_commit, '--gpu', '0',
                       '--gpus', str(gpus), '--transport', transport, '--visible', '4', '--partitions', '32',
                       '--decoder', 'complex', '--pipeline', 'on', '--gradients', 'manual',
                       '--epochs', '3', '--eval-queries', '1000', '--replay-seed', '17',
                       '--work', args.work/name, '--evidence', evidence/name]
            run(command, name)
            progress = json.loads((args.work/name/'progress.json').read_text())
            if progress['status'] != 'done':
                raise RuntimeError('Incomplete run: '+name)
            state['cases'][name] = {key: progress[key] for key in
                                   ('mrr', 'hits_at_10', 'epoch_times_s', 'epochs_completed')}
            save()
        save(status='complete', active_case=None, comparisons=summarize(args.work))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save(status='failed_or_interrupted', child_pid=None, error=repr(error))
        raise


if __name__ == '__main__':
    main()
