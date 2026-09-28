#!/usr/bin/env python3
"""Read-only tail-rank replay of frozen FB checkpoints; never retrain or retime."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from arc_accuracy_gpu_guard import guarded_run
from arc_job_support import write_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            value.update(block)
    return value.hexdigest()


def checked_hash(path, expected):
    observed = digest(path)
    if observed != expected:
        raise ValueError('Frozen input changed: ' + str(path))
    return observed


def validate_protocol(old):
    required = dict(report_directions='tail', filtered=True, tie_policy='pessimistic',
                    tf32=False, mapping_contract='identity', num_eval_edges=10000,
                    num_ranks=10000, num_nodes=86054151, num_relations=14824,
                    score_contract='ge2_forward_inverse_relation_embeddings',
                    ranking_contract='fp32_query_fp64_dot_refinement_v1')
    for key, expected in required.items():
        if old.get(key) != expected:
            raise ValueError('Unexpected saved protocol: ' + key)
    if old['score'] not in ('distmult', 'complex') or old['entity_shape'] != [86054151, 100]:
        raise ValueError('Unexpected FB model dimensions')
    if old.get('sample_seed') is not None:
        raise ValueError('Replay requires the saved fixed query panel')


def command_for(old, paths, tools, output, count):
    validate_protocol(old)
    command = [sys.executable, '-u', str(tools/'eval_marius_kge_exact10k.py')]
    for key in ('entity_bin', 'src_relation_bin', 'dst_relation_bin', 'ge2_data_dir', 'eval_edges'):
        command += ['--'+key.replace('_', '-'), str(paths[key])]
    for key in ('score', 'num_nodes', 'num_relations', 'batch_size', 'candidate_chunk',
                'filter_chunk', 'tie_policy', 'score_contract', 'report_directions'):
        command += ['--'+key.replace('_', '-'), str(old[key])]
    return command + ['--filtered', '--num-test', str(count), '--embedding-dim', '100',
        '--expected-eval-sha256', old['eval_edges_sha256'], '--device', 'cuda:0',
        '--progress-every', '10', '--evaluator-contract', 'arc_fb_tail_recheck_20260928',
        '--out', str(output)]


def verify_result(old, result, previous_ranks, count):
    import numpy as np
    for key in ('entity_bin_sha256', 'src_relation_bin_sha256', 'dst_relation_bin_sha256',
                'eval_edges_sha256', 'report_directions', 'filtered', 'tie_policy',
                'tf32', 'score', 'score_contract', 'ranking_contract', 'entity_shape'):
        if result.get(key) != old[key]:
            raise ValueError('Reevaluation contract/identity changed: ' + key)
    if result['num_ranks'] != count or result['num_eval_edges'] != count:
        raise ValueError('Expected exactly one tail rank per query')
    checked_hash(result['ranks_file'], result['ranks_sha256'])
    checked_hash(previous_ranks, old['ranks_sha256'])
    with np.load(result['ranks_file']) as current, np.load(previous_ranks) as previous:
        ranks = current['tail_ranks']
        if ranks.shape != (count,) or not np.all(np.isfinite(ranks)) or np.any(ranks < 1):
            raise ValueError('Invalid tail ranks')
        if np.any(ranks > old['num_nodes']):
            raise ValueError('Rank exceeds the candidate population')
        if not np.array_equal(current['triples'], previous['triples'][:count]):
            raise ValueError('Query identity/order changed')
        metrics = dict(mrr=float(np.mean(1.0/ranks)), hits_at_10=float(np.mean(ranks <= 10)))
        for key, value in metrics.items():
            if not math.isclose(value, result[key], abs_tol=1e-12, rel_tol=0):
                raise ValueError('Saved rank metrics do not reproduce: '+key)
        return dict(**metrics, changed_ranks=int(np.count_nonzero(ranks != previous['tail_ranks'][:count])),
                    num_ranks=count, old_mrr=float(np.mean(1.0/previous['tail_ranks'][:count])),
                    old_hits_at_10=float(np.mean(previous['tail_ranks'][:count] <= 10)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', training_modified=False, new_training_timing=False,
                 cases=[], host=os.uname().nodename.split('.')[0], job=os.environ['SLURM_JOB_ID'])

    def update(**extra):
        state.update(extra, updated=datetime.datetime.now().astimezone().isoformat())
        write_json(args.out/'status.json', state)

    def interrupted(sig, frame):
        raise RuntimeError('Allocation signal '+str(sig))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    try:
        update()
        allocation = subprocess.check_output(['scontrol', 'show', 'job', state['job'], '-o'],
                                             text=True, timeout=20)
        if ('JobState=RUNNING' not in allocation or 'UserId='+os.environ['USER']+'(' not in allocation
                or re.search(r'\bNodeList=(\S+)', allocation)[1] != state['host']):
            raise RuntimeError('Expected an owned active single-node allocation')
        (args.out/'allocation.txt').write_text(allocation)
        deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-120
        for relative, sha in spec['payload_sha256'].items():
            checked_hash(root/relative, sha)
        write_json(args.out/'manifest.json', spec)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        tools = root/'tools'

        def run(command, name, output):
            remaining = deadline-time.time()
            if remaining < 60:
                raise RuntimeError('Insufficient allocation time')
            write_json(output/(name+'.command.json'), command)
            rc = guarded_run(command, env, output/(name+'.log'), remaining, args.gpu,
                             output/(name+'.guard.jsonl'), output/(name+'.hardware.jsonl'))
            if rc:
                raise RuntimeError(name+' exited '+str(rc))

        # Keep the parent CUDA-free so the isolation guard sees only its child.
        run([sys.executable, '-c',
             'import json, sys, torch; '
             'assert torch.cuda.is_available(), "CUDA unavailable"; '
             'print(json.dumps(dict(python=sys.version, torch=torch.__version__, '
             'cuda=torch.version.cuda, device=torch.cuda.get_device_name(0))))'],
            'runtime', args.out)
        split_hashes = {}
        for case in spec['cases']:
            old = json.loads((root/case['previous_eval']).read_text())
            validate_protocol(old)
            output = args.out/case['name']
            output.mkdir()
            cell = dict(name=case['name'], status='preflight')
            state['cases'].append(cell)
            update(status='running')
            if case.get('flags'):
                flags = json.loads((root/case['flags']).read_text())
                for key, expected in case['required_flags'].items():
                    if str(flags.get(key)) != str(expected):
                        raise ValueError('Checkpoint was not trained with required fix: '+key)
            paths = {key: Path(value) for key, value in case['paths'].items()}
            for key in ('entity_bin', 'eval_edges'):
                checked_hash(paths[key], old[key+'_sha256'])
            for split in ('train', 'validation', 'test'):
                path = paths['ge2_data_dir']/'edges'/(split+'_edges.bin')
                if str(path) not in split_hashes:
                    split_hashes[str(path)] = digest(path)
                expected = case.get('filter_hashes', {}).get(split)
                if expected and split_hashes[str(path)] != expected:
                    raise ValueError('Filtering split hash mismatch: '+split)
            write_json(args.out/'filter_inputs.json', split_hashes)
            # Relation tensors are extracted into the new output, never the checkpoint.
            checked_hash(case['model'], case['model_sha256'])
            paths.update(src_relation_bin=output/'src_relations.bin', dst_relation_bin=output/'dst_relations.bin')
            run([sys.executable, str(tools/'extract_ge2_relation_embeddings.py'),
                 '--model', case['model'], '--src-out', str(paths['src_relation_bin']),
                 '--dst-out', str(paths['dst_relation_bin']), '--expected-relations', '14824',
                 '--expected-dim', '100', '--report', str(output/'relations.json')], 'relations', output)
            for key in ('src_relation_bin', 'dst_relation_bin'):
                checked_hash(paths[key], old[key+'_sha256'])
            for name, count in (('canary64', 64), ('full', 10000)):
                cell.update(status=name)
                update()
                target = output/(name+'.json')
                run(command_for(old, paths, tools, target, count), name, output)
                result = json.loads(target.read_text())
                comparison = verify_result(old, result, root/case['previous_ranks'], count)
                write_json(output/(name+'.comparison.json'), comparison)
            checked_hash(paths['entity_bin'], old['entity_bin_sha256'])
            checked_hash(case['model'], case['model_sha256'])
            cell.update(status='done', **comparison, checkpoint_and_queries_match=True)
            write_json(output/'result.json', cell)
            update()
        update(status='done')
    except BaseException as error:
        if state['cases'] and state['cases'][-1]['status'] != 'done':
            state['cases'][-1].update(status='failed', error=repr(error))
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
