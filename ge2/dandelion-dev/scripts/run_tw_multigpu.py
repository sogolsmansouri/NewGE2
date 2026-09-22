#!/usr/bin/env python3
"""Run one frozen TW multi-GPU gate or final case inside an exclusive allocation."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

import yaml

from arc_accuracy_gpu_guard import foreign_gpu_pids, guarded_run
from prepare_tw_multigpu import check_config, sha


def check_resource_policy(selected_pids, all_apps, other_jobs, allow_shared):
    if selected_pids:
        raise RuntimeError('Selected GPUs are occupied: '+repr(selected_pids))
    if not allow_shared and (all_apps.strip() or other_jobs):
        raise RuntimeError('Node is busy/shared; no competing runs will be started')


def qualify_timing(samples, uuids, allow_shared):
    if not samples:
        raise RuntimeError('Hardware monitoring is missing')
    for sample in samples:
        if any(row.split(',')[0].strip() in uuids for row in sample['other_processes']):
            raise RuntimeError('Contention on a selected GPU')
    isolated = all(not x['other_jobs'] and not x['other_processes'] for x in samples)
    if not isolated and not allow_shared:
        raise RuntimeError('Contention detected; results are not controlled timings')
    return isolated, 'shared_node_provisional' if allow_shared else 'exclusive_node'


def parse_training(text, system, gpus, epochs, gate=False):
    times = [int(v)/1000 for v in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
    finished = list(map(int, re.findall(r'Finished training epoch\s+(\d+)', text)))
    if len(times) != epochs or min(times, default=0) <= 0 or finished != list(range(1, epochs+1)):
        raise ValueError('Incomplete or duplicate epochs')
    if (f'Broadcasting model to: {gpus} GPUs' not in text or 'SynchronousMultiGPUTrainer' not in text
            or re.search(r'CUDA error|out of memory|Traceback|\b(?:nan|inf)\b', text, re.I)):
        raise ValueError('Wrong GPU trainer, numerical error, or failed run')
    if system == 'pipege':
        selected = re.findall(r'Stateflow multi-GPU selected family=.*gpu_count=(\d+).*lanes=(\d+).*microstates=(\d+)', text)
        if not selected or any(tuple(map(int, row)) != (gpus, gpus, 20) for row in selected):
            raise ValueError('Expected twenty-state multi-GPU plan missing')
        if f'rounds={20//gpus} ' not in text or 'estimated_bucket_edges:1321528663' not in text:
            raise ValueError('Wrong round count or planned positive-edge workload')
        rows = (41652230+15)//16
        frames = re.findall(r'deferred backing allocation device=cuda:(\d+) visible_rows=(\d+) physical_rows=(\d+) dim=(\d+) pinned=true hidden_frames=(\d+)', text)
        if ({int(row[0]) for row in frames} != set(range(gpus)) or len(frames) < 2*gpus
                or any(tuple(map(int, row[1:])) != (4*rows, 7*rows, 100, 3) for row in frames)):
            raise ValueError('Unexpected embedding/optimizer frame allocation')
        if ('[manual_dot_rns] enabled=1' not in text or 'Using bucket-streaming LP path' not in text
                or 'stateflow_scope=all' not in text
                or re.search(r'descriptor_mismatch_count=[1-9]|(?:dst|src)_mismatch_values=[1-9]', text)):
            raise ValueError('Runtime path or peer-copy validation mismatch')
        checks = re.findall(r'\[stateflow-peer-validate \d+\].*dst_mismatch_values=0.*src_mismatch_values=0', text)
        if gate and len(checks) < 16:
            raise ValueError('Peer-copy value checks did not execute; gate cannot pass')
    return dict(epoch_times_s=times, average_epoch_s=statistics.mean(times),
                steady_average_epoch_s_excluding_first=statistics.mean(times[1:]),
                timing_definition='engine Epoch Runtime; excludes initialization, evaluation and some epoch finalization')


def check_evaluation(metrics, expected_sha):
    if (metrics.get('eval_edges_sha256') != expected_sha or metrics.get('num_ranks') != 10000
            or metrics.get('report_directions') != 'tail' or metrics.get('filtered') is not True
            or metrics.get('tf32') is not False or metrics.get('tie_policy') != 'pessimistic'):
        raise ValueError('Evaluation protocol mismatch')
    if any(not math.isfinite(metrics.get(k, float('nan'))) or not 0 <= metrics[k] <= 1
           for k in ('mrr', 'hits_at_10')):
        raise ValueError('Invalid evaluation result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'work', 'results', 'env', 'data', 'source-archive'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=Path('/mnt/local/smansou2/pipege_engine_14ec00d_arc'))
    parser.add_argument('--case', required=True)
    parser.add_argument('--phase', choices=('gate', 'final'), required=True)
    parser.add_argument('--gpu-ids', required=True, help='Physical indices or UUIDs, comma separated')
    parser.add_argument('--gate-result', type=Path)
    parser.add_argument('--power-limit', type=float, required=True, help='Verify only; never changes GPU power')
    parser.add_argument('--allow-shared-node', action='store_true',
                        help='Accuracy only on idle selected GPUs; timings always provisional')
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    manifest = json.loads((args.bundle/'manifest.json').read_text())
    for rel, expected in manifest['files'].items():
        if sha(args.bundle/rel) != expected:
            raise ValueError('Frozen file changed: '+rel)
    spec = manifest['cases'][args.case]
    ids = args.gpu_ids.split(',')
    if len(ids) != spec['gpus'] or len(set(ids)) != len(ids):
        raise ValueError('GPU selection does not match case')
    if args.work.exists() or args.results.exists():
        raise ValueError('Fresh work and result directories are required for every attempt')
    if not str(args.work).startswith('/mnt/local/'):
        raise ValueError('Large data/checkpoints must be on node-local storage')
    sys.path.insert(0, str(args.bundle/'tools'))
    from run_arc_ge2_allocated_queue import audit_data, run_logged, write_json
    from run_arc_ge2_dot_reproduction import check_subset_membership
    job, host = os.environ['SLURM_JOB_ID'], os.uname().nodename.split('.')[0]
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=20)
    if ('JobState=RUNNING' not in allocation or re.search(r'\bNodeList=(\S+)', allocation)[1] != host
            or re.search(r'\bUserId=(\w+)', allocation)[1] != os.environ['USER']):
        raise RuntimeError('Not inside the owned running allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-120
    def exclusive():
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=30).strip()
        jobs = subprocess.check_output(['squeue', '-h', '-w', host, '-o', '%A'], text=True, timeout=20).split()
        check_resource_policy(foreign_gpu_pids(','.join(ids)), apps,
                              [j for j in jobs if j != job], args.allow_shared_node)
    exclusive()
    # A per-node user lock also prevents two detached copies racing the idle check.
    lock = Path('/mnt/local/smansou2/tw_multigpu_followup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    exclusive()
    uuids, hardware = [], []
    for gpu in ids:
        row = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=uuid,name,power.limit', '--format=csv,noheader'], text=True, timeout=30).strip()
        uuid, name, power = (x.strip() for x in row.split(','))
        if name != 'NVIDIA RTX A6000' or abs(float(power.split()[0])-args.power_limit) > .1:
            raise ValueError('Unexpected GPU model or power limit: '+row)
        uuids.append(uuid)
        hardware.append(row)
    if len(set(uuids)) != spec['gpus']:
        raise ValueError('Duplicate physical GPUs')
    args.work.mkdir(parents=True)
    args.results.mkdir(parents=True)
    state = dict(status='preflight', case=args.case, phase=args.phase, host=host, job=job,
                 gpu_uuids=uuids, hardware=hardware, bundle_sha256=sha(args.bundle/'manifest.json'),
                 checkpoint_durable=False, paper_ready=False, allow_shared_node=args.allow_shared_node,
                 timing_status='shared_node_provisional' if args.allow_shared_node else 'exclusive_node')
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(args.results/'status.json', state)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Allocation signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
    env.pop('LD_PRELOAD', None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(uuids),
               OMP_NUM_THREADS='8', PATH=str(args.env/'bin')+':'+os.environ['PATH'])
    python = str(args.env/'bin/python')
    tools = args.bundle/'tools'
    def run(command, name, monitor=False, max_seconds=None):
        exclusive()
        budget = min(deadline-time.time(), max_seconds or float('inf'))
        if budget < 60:
            raise RuntimeError('Allocation time exhausted')
        update(stage=name)
        hardware_log = args.results/(name+'.hardware.jsonl') if monitor else None
        if args.allow_shared_node:
            rc = guarded_run(command, env, args.results/(name+'.log'), budget, ','.join(uuids),
                             args.results/(name+'.gpu_guard.jsonl'), hardware_log)
        else:
            rc = run_logged(command, env, args.results/(name+'.log'), budget, hardware_log)
        if rc:
            raise RuntimeError(f'{name} exited {rc}')
    try:
        update()
        (args.results/'allocation.txt').write_text(allocation)
        if args.phase == 'final':
            if args.gate_result is None:
                raise ValueError('Final training requires an explicit successful gate result')
            gate = json.loads(args.gate_result.read_text())
            for key in ('case', 'bundle_sha256', 'gpu_uuids', 'hardware', 'allow_shared_node'):
                if gate[key] != state[key]:
                    raise ValueError('Gate provenance/hardware mismatch: '+key)
            if gate.get('status') != 'gate_passed' or gate.get('phase') != 'gate':
                raise ValueError('Live multi-GPU gate has not passed')
            state['gate_result_sha256'] = sha(args.gate_result)
        if spec['system'] == 'pipege':
            for name, expected in manifest['engine_hashes'].items():
                if sha(args.engine/'build_git'/name) != expected:
                    raise ValueError('PipeGE engine mismatch: '+name)
            env['PYTHONPATH'] = str(args.engine/'repo/ge2/dandelion-dev/gege/src/python')
            libdir = args.engine/'build_git'
            command = [str(libdir/'gege_train')]
            env.update(json.loads((args.bundle/spec['flags']).read_text()))
            env['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(args.bundle/'schedule.txt')
            if args.phase == 'gate':
                env.update(GEGE_STATEFLOW_PEER_RELAY_VALIDATE='1',
                           GEGE_STATEFLOW_PEER_RELAY_VALIDATE_FAIL_FAST='1',
                           GEGE_STATEFLOW_PEER_RELAY_VALIDATE_MAX_CHECKS='100000',
                           GEGE_STATEFLOW_DEBUG_VALIDATE='1')
        else:
            libdir = args.env/'lib/python3.9/site-packages/gege'
            if sha(libdir/'libge2.so') != manifest['ge2_library_sha256']:
                raise ValueError('Released GE2 library mismatch')
            command = [str(args.env/'bin/gege_train')]
        if sha(args.source_archive) != manifest['ge2_source_sha256']:
            raise ValueError('Released source archive mismatch')
        env['LD_LIBRARY_PATH'] = ':'.join(map(str, [libdir, args.env/'lib/python3.9/site-packages/torch/lib', args.env/'lib']))
        run([python, '-c',
             'import torch,gege,json; n=torch.cuda.device_count(); '
             f'assert n=={spec["gpus"]}; '
             'p=[[i==j or torch.cuda.can_device_access_peer(i,j) for j in range(n)] for i in range(n)]; '
             'print(json.dumps(dict(gpus=n,peer_access=p,gege=gege.__file__,torch=torch.__version__))); '
             'assert all(all(row) for row in p), "Peer access unavailable"'], 'runtime_gate', max_seconds=120)
        topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True, timeout=30)
        (args.results/'topology.txt').write_text(topology)
        if shutil.disk_usage(args.work).free < 80*1024**3:
            raise RuntimeError('Need at least 80 GiB free for private data and checkpoint')
        update(stage='private_data_copy')
        data = args.work/'data'
        shutil.copytree(args.data, data)
        # The native loader re-reads dataset.yaml, so both paths must be relocated.
        meta = yaml.safe_load((data/'dataset.yaml').read_text())
        if (meta['num_nodes'], meta['num_train'], meta['num_valid'], meta['num_test']) != (41652230,1321528663,73418259,73418260):
            raise ValueError('Wrong TW split or full-source workload')
        meta['dataset_dir'] = str(data)+'/'
        (data/'dataset.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
        run([python, str(tools/'validate_ge2_table4_tw_data.py'), '--data-dir', str(data),
             '--protocol-id', 'ge2-table4-v1', '--split-identity', 'controlled_deterministic_ge2_90_5_5_v1',
             '--num-train', '1321528663', '--num-valid', '73418259', '--num-test', '73418260',
             '--num-nodes', '41652230', '--verify-hashes', '--verify-bucket-order',
             '--report', str(args.results/'split_audit.json')], 'data_validation')
        split = json.loads((data/'split_manifest.json').read_text())
        query = data/split['exact_eval']['path']
        report = audit_data(data, 2, query, manifest['eval_sha256'], manifest['split_sha256'])
        report['query_membership'] = check_subset_membership(data/'edges/test_edges.bin', query)
        write_json(args.results/'data_audit.json', report)
        cfg = yaml.safe_load((args.bundle/spec['config']).read_text())
        check_config(cfg, spec['system'], spec['gpus'])
        model = args.work/'model'
        cfg['storage'].update(dataset=meta, model_dir=str(model)+'/', checkpoint_dir=str(model)+'/')
        cfg['evaluation']['checkpoint_dir'] = str(model)+'/'
        epochs = 2 if args.phase == 'gate' else 10
        cfg['training'].update(num_epochs=epochs, save_model=args.phase == 'final')
        config = args.results/'config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(args.results/'flags.json', {k:v for k,v in env.items() if k.startswith(('GEGE_', 'CUDA_', 'OMP_', 'PYTORCH_'))})
        run(command+[str(config)], 'train', monitor=True,
            max_seconds=2700 if args.phase == 'gate' else None)
        training = parse_training((args.results/'train.log').read_text(), spec['system'], spec['gpus'], epochs, args.phase == 'gate')
        samples = [json.loads(line) for line in (args.results/'train.hardware.jsonl').read_text().splitlines()]
        isolated, timing_status = qualify_timing(samples, uuids, args.allow_shared_node)
        for sample in samples:
            observed = {}
            for row in sample['gpu'].splitlines():
                fields = [x.strip() for x in row.split(',')]
                observed[fields[1]] = float(fields[2].split()[0])
            if any(uuid not in observed or abs(observed[uuid]-args.power_limit) > .1 for uuid in uuids):
                raise RuntimeError('GPU power setting changed or monitoring is incomplete')
        state.update(training, isolation_passed=isolated, timing_status=timing_status)
        if args.phase == 'gate':
            update(status='gate_passed', scope='Two-epoch runtime and sampled peer-value checks, not a proof of convergence')
            write_json(args.results/'result.json', state)
            return
        checkpoint = model/'embeddings.bin'
        if checkpoint.stat().st_size != manifest['nodes']*100*4:
            raise ValueError('Incomplete final entity checkpoint')
        state.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        write_json(args.results/'training_result.json', state)
        run([python, str(tools/'stream_marius_dot_exact_eval.py'), '--run-dir', str(model),
             '--embedding-file', str(checkpoint), '--num-nodes', str(manifest['nodes']), '--dim', '100',
             '--eval-edges', str(query), '--expected-eval-sha256', manifest['eval_sha256'],
             '--expected-num-eval-edges', '10000', '--filtered', '--ge2-data-dir', str(data),
             '--tie-policy', 'pessimistic', '--report-directions', 'tail', '--device', 'cuda:0',
             '--batch-size', '128', '--candidate-chunk', '250000',
             '--out', str(args.results/'exact_eval.json')], 'eval', monitor=True)
        metrics = json.loads((args.results/'exact_eval.json').read_text())
        check_evaluation(metrics, manifest['eval_sha256'])
        update(status='done_pending_review', mrr=metrics['mrr'], hits_at_10=metrics['hits_at_10'],
               review='Compare quality and timing definitions before publication; checkpoint remains node-local')
        write_json(args.results/'result.json', state)
    except BaseException as exc:
        update(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
