#!/usr/bin/env python3
"""Replay the archived fast TW settings with repaired gradients and held-out evaluation."""

import argparse
import copy
import datetime
import fcntl
import itertools
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

import yaml

NODES = 41652230
EDGES = 1321528663
CORE = 'fa783f4b9ef685fd5bd969d3426ddce3ee324721'
QUERY_SHA = '93bf1dd7104a2a225800229abb62fbf7477ef006807d0ce523fe6094f3e3ded6'
FAST = dict(GEGE_BUCKET_STREAMING_LP='1', GEGE_BATCHED_NEGATIVE_PLAN_BATCHES='8',
            GEGE_DEG_CHUNK_EXCLUSION='1', GEGE_FRAME_CACHE_HIDDEN_FRAMES='3',
            GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES='-1', GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='3',
            GEGE_FRAME_CACHE_AUTO_PIPELINE_FRAMES='0', GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='1',
            GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='1', GEGE_SYNC_BEFORE_SWAP='0',
            GEGE_MEM_SWAP_EVENT_SYNC='1', GEGE_FIXED_BUFFER_MANUAL_DOT_RNS='1')


def make_config(reference, dataset, model_dir, epochs):
    cfg = copy.deepcopy(reference)
    m, s, t = cfg['model'], cfg['storage'], cfg['training']
    ns, opts = t['negative_sampling'], s['embeddings']['options']
    checks = [opts['num_partitions'] == 16, opts['buffer_capacity'] == 4,
              s['prefetch'] is True, t['batch_size'] == 50000,
              t['negative_sampling_method'] == 'RNS', ns['num_chunks'] == 50,
              ns['negatives_per_positive'] == 1000, ns['degree_fraction'] == .5,
              ns['superbatch_negative_plan_batches'] == 8,
              m['encoder']['layers'][0][0]['output_dim'] == 100,
              m['decoder']['type'] == 'DISTMULT',
              m['random_seed'] == 741135446461071584,
              m['loss']['type'] == 'SOFTMAX_CE', m['loss']['options']['reduction'] == 'SUM']
    for key in ('dense_optimizer', 'sparse_optimizer'):
        checks += [m[key]['type'] == 'ADAGRAD', m[key]['options']['learning_rate'] == .1]
    checks += [dataset['num_nodes'] == NODES, dataset['num_train'] == EDGES,
               dataset['num_relations'] == 1, epochs in (2, 10)]
    if not all(checks):
        raise ValueError('Inputs do not match the archived 50K fast TW and controlled-split contracts')
    s['dataset'] = copy.deepcopy(dataset)
    s['model_dir'] = str(model_dir) + '/'
    t.update(num_epochs=epochs, save_model=epochs == 10, resume_training=False,
             resume_from_checkpoint='')
    cfg['evaluation'].update(epochs_per_eval=1000, checkpoint_dir=str(model_dir) + '/')
    return cfg


def fast_environment(reference, schedule, gate):
    if any(reference.get(key) != value for key, value in FAST.items()):
        raise ValueError('Archived invocation does not match the fast shared-3 configuration')
    flags = {k: str(v) for k, v in reference.items() if k.startswith('GEGE_')}
    for key in ('GEGE_TRAINING_REPLAY_SEED', 'GEGE_TRAINING_INPUT_AUDIT'):
        flags.pop(key, None)
    flags.update(GEGE_BASELINE_TRAINING_SEMANTICS='0', GEGE_STARTUP_TIMING='1',
                 GEGE_BOUNDED_STATE_ORDER_FILE=str(schedule),
                 GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET='1',
                 GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES='0',
                 GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS='1',
                 GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS='1',
                 GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY='1' if gate else '0',
                 GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY_MAX='8')
    flags['PYTORCH_CUDA_ALLOC_CONF'] = reference['PYTORCH_CUDA_ALLOC_CONF']
    return flags


def check_schedule(text):
    states = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'state=(\[.*\])', line.strip())
        if not match:
            raise ValueError('Malformed schedule')
        values = json.loads(match[1])
        if len(values) != 4 or len(set(values)) != 4 or any(type(x) is not int or not 0 <= x < 16 for x in values):
            raise ValueError('Invalid resident state')
        states.append(frozenset(values))
    pairs = {pair for state in states for pair in itertools.combinations(sorted(state), 2)}
    if (len(states) != 20 or len(set(states)) != 20 or len(pairs) != 120
            or any(len(b - a) != 3 for a, b in zip(states, states[1:]))):
        raise ValueError('Expected a full 20-state p16/q4 cover with overlap one')


def check_training(text, epochs):
    times = [int(x) / 1000 for x in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
    finished = [int(x) for x in re.findall(r'Finished training epoch\s+(\d+)', text)]
    counts = re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%', text)
    if (len(times) != epochs or min(times, default=0) <= 0 or finished != list(range(1, epochs + 1))
            or counts != [(str(EDGES), str(EDGES))] * epochs):
        raise ValueError('Incomplete full-edge training')
    states = set(re.findall(r'Generating bounded GREEDY_COVER ordering states=(\d+)', text))
    frames = re.findall(r'deferred backing allocation device=cuda:\d+ visible_rows=(\d+) physical_rows=(\d+) dim=(\d+) pinned=true hidden_frames=(\d+)', text)
    rows = (NODES + 15) // 16
    if states != {'20'} or len(frames) < 2 or any(row != tuple(map(str, (4*rows, 7*rows, 100, 3))) for row in frames):
        raise ValueError('Shared frame allocation or schedule not verified')
    if ('[manual_dot_rns] enabled=1' not in text or 'Using bucket-streaming LP path' not in text
            or re.search(r'\b(nan|inf)\b|CUDA error|device-side assert', text, re.I)):
        raise ValueError('Wrong execution path or numerical failure')
    samples = re.findall(r'\[frame_cache\] swap_samples=(\d+) [^\n]*hidden_publish_parts=(\d+) [^\n]*fallback_visible_admit_parts=(\d+) [^\n]*preload_miss_swaps=(\d+)', text)
    if len(samples) != epochs or any(row != ('19', '57', '0', '0') for row in samples):
        raise ValueError('Full hidden-frame admission was not observed in every epoch')
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('work', 'results', 'engine_work', 'env', 'harness', 'fast_reference', 'control', 'after_status', 'gradient_gate'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.harness / 'tools'))
    from prepare_ge2_partitioned_view import sha256_file
    from run_arc_ge2_allocated_queue import audit_data, run_logged, write_json
    from run_arc_pipege_quality import checkpoint_manifest, timing_summary, validate_gradient_gate, validate_update_checks

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.results.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    lock = (args.work.parent / 'pipege_tw_fast_replay.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='preflight', job=args.job, pid=os.getpid(), core_commit=CORE,
                 gpu=args.gpu, host=os.uname().nodename.split('.')[0], checkpoint_durable=False,
                 purpose=__doc__, interpretation='Fast policy replay on held-out split; not an exact old-workload reproduction')

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(args.results / 'status.json', state)

    def guard():
        alloc = subprocess.check_output(['scontrol', 'show', 'job', args.job, '-o'], text=True, timeout=20)
        if ('JobState=RUNNING ' not in alloc or 'UserId=' + os.environ['USER'] + '(' not in alloc
                or re.search(r'\bNodeList=(\S+)', alloc)[1] != state['host']):
            raise RuntimeError('Owned allocation is not active on this host')
        return alloc

    def interrupted(sig, frame):
        raise KeyboardInterrupt('Signal ' + str(sig))

    signal.signal(signal.SIGTERM, interrupted)
    update()
    try:
        alloc = guard()
        (args.results / 'allocation.txt').write_text(alloc)
        deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', alloc)[1]).timestamp() - 180
        reference = yaml.safe_load((args.fast_reference / 'effective_config.yaml').read_text())
        invocation = json.loads((args.fast_reference / 'invocation.json').read_text())
        controlled = yaml.safe_load((args.control / 'ours_TW_Dot_1gpu.yaml').read_text())
        data = Path(controlled['storage']['dataset']['dataset_dir'])
        schedule = args.results / 'state_order.txt'
        schedule_text = Path(invocation['GEGE_BOUNDED_STATE_ORDER_FILE']).read_text()
        check_schedule(schedule_text)
        schedule.write_text(schedule_text)
        shutil.copyfile(args.fast_reference / 'effective_config.yaml', args.results / 'archived_config.yaml')
        shutil.copyfile(args.fast_reference / 'invocation.json', args.results / 'archived_invocation.json')
        cases = []
        for epochs, label in [(2, 'gate_2e'), (10, 'final_10e')]:
            case = args.results / label
            case.mkdir()
            model = args.work / label / 'model'
            model.parent.mkdir()
            cfg = make_config(reference, controlled['storage']['dataset'], model, epochs)
            flags = fast_environment(invocation, schedule, epochs == 2)
            (case / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            write_json(case / 'flags.json', flags)
            cases.append((case, model, epochs, flags))
        update(status='waiting', stage='waiting_for_FB_campaign', predecessor=str(args.after_status))
        while True:
            guard()
            if deadline - time.time() < 4800:
                update(status='deferred', stage='insufficient_allocation_time')
                return
            previous = json.loads(args.after_status.read_text()) if args.after_status.exists() else {}
            if previous.get('status') in ('done', 'failed', 'partial_failure', 'interrupted'):
                apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
                if not apps:
                    break
            time.sleep(30)

        update(status='running', stage='source_and_data_audit')
        repo, build = args.engine_work / 'repo', args.engine_work / 'build_git'
        if (subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != CORE
                or subprocess.check_output(['git', '-C', str(repo), 'diff', '--binary', 'HEAD'])
                or (args.engine_work / 'build_git_completed_commit.txt').read_text().strip() != CORE):
            raise RuntimeError('Repaired engine source/build attestation changed')
        build_hashes = {name: sha256_file(build / name) for name in ('libge2.so', 'gege_train')}
        validate_gradient_gate(json.loads(args.gradient_gate.read_text()), build_hashes)
        if shutil.disk_usage(args.work).free < 100 * 1024**3:
            raise RuntimeError('Need 100 GiB free for gate and final checkpoint')
        helpers = json.loads((args.control.parent / 'harness_sha256.json').read_text())
        checked = {}
        for path, digest in helpers.items():
            if path.startswith('tools/') and path.endswith('.py'):
                if sha256_file(args.harness / path) != digest:
                    raise RuntimeError('Frozen helper changed: ' + path)
                checked[path] = digest
        query = data / 'exact10000_uniform_v2/edges/test_edges.bin'
        pinned = json.loads((args.control.parent / 'tw_input_audit.json').read_text())
        audit = audit_data(data, 2, query, QUERY_SHA, {k:v['sha256'] for k,v in pinned['splits'].items()})
        write_json(args.results / 'data_audit.json', audit)
        write_json(args.results / 'provenance.json', dict(core_commit=CORE, build=build_hashes,
                   driver_sha256=sha256_file(Path(__file__)), schedule_sha256=sha256_file(schedule), helpers=checked,
                   config_sha256={case.name: sha256_file(case/'config.yaml') for case, _, _, _ in cases},
                   flags_sha256={case.name: sha256_file(case/'flags.json') for case, _, _, _ in cases}))
        env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
        env.pop('LD_PRELOAD', None)
        env.update(PATH=f'{args.env}/bin:/usr/bin:/bin',
                   PYTHONPATH=str(args.engine_work / 'python'), PYTHONDONTWRITEBYTECODE='1',
                   LD_LIBRARY_PATH=f'{build}:{args.env}/lib:{args.env}/lib/python3.9/site-packages/torch/lib:/usr/lib64',
                   GEGE_NO_BINDINGS='1', CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1', SLURM_JOB_ID=args.job)

        def run(command, case, label, extra=None):
            guard()
            if deadline - time.time() < 60:
                raise RuntimeError('Allocation deadline reached')
            update(stage=case.name + ':' + label)
            code = run_logged(list(map(str, command)), dict(env, **(extra or {})), case / (label + '.log'),
                              deadline - time.time(), case / (label + '.hardware.jsonl'))
            if code:
                raise RuntimeError(f'{case.name}/{label} exited {code}')

        for case, model, epochs, flags in cases:
            run([build / 'gege_train', case / 'config.yaml'], case, 'train', flags)
            log = (case / 'train.log').read_text()
            times = check_training(log, epochs)
            timing = timing_summary(log, times)
            if epochs == 2:
                gate = validate_update_checks(log, manual=True, model='Dot')
                write_json(case / 'result.json', dict(status='gate_pass', checks=gate, **timing))
                continue
            write_json(case / 'training_result.json', dict(status='trained', **timing))
            update(stage='checkpoint_hashing')
            receipt = checkpoint_manifest(model, NODES, 100, 1)
            write_json(case / 'checkpoint_manifest.json', receipt)
            python = args.env / 'bin/python'
            run([python, args.harness / 'tools/stream_marius_dot_exact_eval.py', '--run-dir', case,
                 '--embedding-file', model / 'embeddings.bin', '--num-nodes', NODES, '--dim', 100,
                 '--eval-edge-columns', 2, '--filter-edge-columns', 2, '--expected-num-eval-edges', 10000,
                 '--eval-edges', query, '--expected-eval-sha256', QUERY_SHA, '--ge2-data-dir', data,
                 '--filtered', '--tie-policy', 'pessimistic', '--device', 'cuda:0', '--batch-size', 128,
                 '--candidate-chunk', 250000, '--out', case / 'exact_eval.json'], case, 'exact_eval')
            evaluation = json.loads((case / 'exact_eval.json').read_text())
            if (evaluation['num_ranks'] != 20000 or not evaluation['filtered']
                    or evaluation['eval_edges_sha256'] != QUERY_SHA
                    or evaluation['report_directions'] != 'both'
                    or evaluation['tie_policy'] != 'pessimistic' or evaluation['tf32'] is not False
                    or evaluation['embedding_file'] != str(model / 'embeddings.bin')
                    or evaluation['evaluator_sha256'] != checked['tools/stream_marius_dot_exact_eval.py']
                    or evaluation['ranking_helper_sha256'] != checked['tools/exact_eval_ranking.py']):
                raise RuntimeError('Evaluation protocol mismatch')
            violations = []
            for monitor in case.glob('*.hardware.jsonl'):
                for line in monitor.read_text().splitlines():
                    sample = json.loads(line)
                    if sample['other_processes'] or sample['other_jobs']:
                        violations.append(sample)
            result = dict(status='done', train_status=0, exact_eval_status=0, **timing,
                          mrr=evaluation['mrr'], hits_at_10=evaluation['hits_at_10'],
                          source_commit=CORE, checkpoint_durable=False, isolation_violations=violations,
                          embedding_sha256=next(x['sha256'] for x in receipt['files'] if x['path']=='embeddings.bin'),
                          paper_readiness='pending_protocol_and_repeat_review')
            write_json(case / 'result.json', result)
        update(status='done', stage='finished', result=str(cases[-1][0] / 'result.json'))
    except BaseException as error:
        update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
