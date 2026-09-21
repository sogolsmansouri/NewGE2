#!/usr/bin/env python3
"""Evaluate an intact Dot checkpoint, then optionally continue its frozen campaign."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(label + ' mismatch')


def verify_checkpoint_files(checkpoint, saved, sha256_file):
    require_equal(Path(saved['source']).resolve(), checkpoint.resolve(), 'Checkpoint location')
    names = [entry['path'] for entry in saved['files']]
    if len(set(names)) != len(names) or not {'embeddings.bin', 'embeddings_state.bin', 'model.pt_0'}.issubset(names):
        raise ValueError('Incomplete or duplicate checkpoint members')
    for entry in saved['files']:
        path = checkpoint/entry['path']
        if checkpoint.resolve() not in path.resolve().parents:
            raise ValueError('Checkpoint member escapes directory')
        require_equal(path.stat().st_size, entry['bytes'], 'Checkpoint size ' + entry['path'])
        require_equal(sha256_file(path), entry['sha256'], 'Checkpoint hash ' + entry['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'work', 'results', 'training-case', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--then-supervise', action='store_true')
    parser.add_argument('--reuse-completed-commit', action='append', default=[])
    args = parser.parse_args()
    sys.path[:0] = [str(args.base/'scripts'), str(args.base/'harness/tools')]
    from run_arc_pipege_best import CORE, ENV, evaluation_check, training_check
    from run_arc_pipege_quality import timing_summary
    from run_arc_ge2_allocated_queue import run_logged, write_json
    from prepare_ge2_partitioned_view import sha256_file

    job, host = os.environ['SLURM_JOB_ID'], os.uname().nodename.split('.')[0]
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if ('JobState=RUNNING ' not in allocation
            or 'UserId=' + os.environ['USER'] + '(' not in allocation
            or re.search(r'\bNodeList=(\S+)', allocation)[1] != host):
        raise RuntimeError('An owned, active allocation on this node is required')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-180
    if deadline-time.time() < 900:
        raise RuntimeError('Insufficient allocation time')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'allocation.txt').write_text(allocation)
    state = dict(status='preflight', job=job, host=host, pid=os.getpid(),
                 training_case=str(args.training_case), recovery_script_sha256=sha256_file(Path(__file__)))
    def update(**values):
        state.update(values, updated=datetime.datetime.now().isoformat())
        write_json(args.output/'status.json', state)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Signal ' + str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    update()
    locks = []
    try:
        # Hold the same locks as training so recovery cannot race another run.
        for path in (args.results/'supervisor.lock', args.work/'campaign.lock'):
            lock = path.open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        case = args.training_case
        previous = json.loads((case.parent/'status.json').read_text())
        contract = json.loads((case/'contract.json').read_text())
        manifest = json.loads((args.base/'manifest.json').read_text())
        prepared = json.loads((args.work/'prepared.json').read_text())
        spec = manifest['cases'][contract['case']]
        require_equal(spec['model'], 'dot', 'Dot model')
        require_equal(previous['host'], host, 'Checkpoint host')
        require_equal(previous['commit'], contract['commit'], 'Training commit')
        require_equal(previous['built_engine_commit'], CORE, 'Training engine')
        require_equal(prepared['commit'], contract['commit'], 'Prepared source')
        require_equal(prepared['status'], 'ready', 'Preparation')
        require_equal(prepared['manifest_sha256'], sha256_file(args.base/'manifest.json'), 'Manifest')
        for key, value in spec.items():
            require_equal(contract.get(key), value, 'Contract ' + key)
        for rel, digest in {**manifest['references'], **manifest['helpers']}.items():
            require_equal(sha256_file(args.base/rel), digest, 'Frozen helper ' + rel)
        for name in ('config', 'flags'):
            suffix = '.yaml' if name == 'config' else '.json'
            require_equal(sha256_file(case/(name+suffix)), contract[name+'_sha256'], name)
        if (case/'result.json').exists() or previous['status'] == 'done':
            raise RuntimeError('Case already evaluated; refusing replacement')
        training_text = (case/'train.log').read_text()
        times = training_check(training_text, spec, spec['epochs'])
        timing = timing_summary(training_text, times)
        saved_timing = json.loads((case/'training_result.json').read_text())
        require_equal(saved_timing['epoch_times_s'], times, 'Completed epochs')
        require_equal(json.loads((case.parent/'gate_2e/result.json').read_text())['status'],
                      'gate_pass', 'Training correctness gate')
        write_json(args.output/'previous_status.json', previous)
        checkpoint = Path(contract['model_dir'])
        saved = json.loads((case/'checkpoint_manifest.json').read_text())
        update(stage='verify_checkpoint', checkpoint=str(checkpoint))
        verify_checkpoint_files(checkpoint, saved, sha256_file)
        write_json(args.output/'checkpoint_manifest.json', saved)
        update(stage='verify_data')
        for split, entry in prepared['data'][spec['graph']]['splits'].items():
            require_equal(sha256_file(Path(spec['source'])/'edges'/f'{split}_edges.bin'),
                          entry['sha256'], 'Filter split ' + split)
        require_equal(sha256_file(Path(spec['query'])), spec['eval_sha'], 'Evaluation queries')
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPUs are in use; refusing overlap with another run')
        env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
        env.pop('LD_PRELOAD', None)
        env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   LD_LIBRARY_PATH=f'{ENV}/lib:{ENV}/lib/python3.9/site-packages/torch/lib:/usr/lib64',
                   OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        command = [str(ENV/'bin/python'), str(args.base/'harness/tools/stream_marius_dot_exact_eval.py'),
                   '--run-dir', str(args.output), '--embedding-file', str(checkpoint/'embeddings.bin'),
                   '--num-nodes', str(spec['nodes']), '--dim', str(spec['width']),
                   '--eval-edge-columns', '2', '--filter-edge-columns', '2', '--expected-num-eval-edges', '10000',
                   '--eval-edges', spec['query'], '--expected-eval-sha256', spec['eval_sha'],
                   '--ge2-data-dir', spec['source'], '--filtered', '--tie-policy', 'pessimistic',
                   '--device', 'cuda:0', '--batch-size', '128', '--candidate-chunk', '250000',
                   '--out', str(args.output/'exact_eval.json')]
        recorded_env = {k: env[k] for k in ('SLURM_JOB_ID', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                        'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
        write_json(args.output/'invocation.json', dict(command=command, environment=recorded_env,
                   training_commit=contract['commit'], evaluation_job=job))
        update(status='running', stage='exact_eval')
        rc = run_logged(command, env, args.output/'exact_eval.log', deadline-time.time(),
                        args.output/'exact_eval.hardware.jsonl')
        if rc:
            raise RuntimeError('Evaluation exit ' + str(rc))
        quality = json.loads((args.output/'exact_eval.json').read_text())
        evaluation_check(quality, spec)
        isolation = [r for r in map(json.loads, (case/'train.hardware.jsonl').read_text().splitlines())
                     if r['other_jobs'] or r['other_processes']]
        result = dict(status='done', train_status=0, exact_eval_status=0, **timing,
                      mrr=quality['mrr'], hits_at_10=quality['hits_at_10'],
                      commit=contract['commit'], built_engine_commit=CORE, scope=spec['scope'], host=host,
                      training_job=previous['job'], evaluation_job=job, training_run_dir=str(case),
                      config_notes=spec['notes'], checkpoint_durable=False, checkpoint=str(checkpoint),
                      isolation_violations=isolation, other_jobs_at_start=previous.get('other_jobs_at_start', []),
                      paper_readiness='shared_node_timing_provisional' if isolation or previous.get('other_jobs_at_start')
                                      else 'pending_protocol_review')
        write_json(args.output/'result.json', result)
        # Link the new evaluation without overwriting the interrupted log or
        # relabeling the source training job, commit, or timing measurements.
        previous.update(status='done', stage='evaluated_after_recovery', evaluation_job=job,
                        final_result=str(args.output/'result.json'), recovery_dir=str(args.output),
                        updated=datetime.datetime.now().isoformat())
        write_json(case.parent/'status.json', previous)
        update(status='done', stage='evaluated', final_result=str(args.output/'result.json'))
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise
    finally:
        for lock in reversed(locks):
            lock.close()
    if args.then_supervise:
        command = [str(ENV/'bin/python'), str(args.base/'scripts/supervise_arc_pipege_best.py'),
                   '--base', str(args.base), '--work', str(args.work), '--results', str(args.results),
                   '--commit', contract['commit']]
        for commit in args.reuse_completed_commit:
            command += ['--reuse-completed-commit', commit]
        write_json(args.output/'continuation.json', dict(command=command, job=job))
        os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
