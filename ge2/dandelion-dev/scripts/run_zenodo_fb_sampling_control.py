#!/usr/bin/env python3
"""Prespecified original-GE2 FB accuracy controls with frozen evaluation panels."""

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
import statistics
import subprocess
import sys
import time

import yaml


def check_reference(config, model):
    m, s, t = config['model'], config['storage'], config['training']
    options = s['embeddings']['options']
    checks = [m['decoder']['type'].lower() == model,
              m['encoder']['layers'][0][0]['output_dim'] == 100,
              m['encoder']['layers'][0][0]['init']['type'] == 'GLOROT_UNIFORM',
              m['decoder']['options']['input_dim'] == 100,
              m['decoder']['options']['inverse_edges'] is True,
              m['random_seed'] == 741135446461071584,
              m['loss']['type'] == 'SOFTMAX_CE', m['loss']['options']['reduction'] == 'SUM',
              options['num_partitions'] == 16, options['buffer_capacity'] == 4,
              s['embeddings']['type'] == 'MEM_PARTITION_BUFFER',
              t['batch_size'] == 50000, t['num_epochs'] == 10,
              t['negative_sampling_method'] == 'RNS',
              t['negative_sampling']['degree_fraction'] == .5,
              t['negative_sampling']['num_chunks'] == 50,
              t['negative_sampling']['negatives_per_positive'] == 1000,
              config['evaluation']['epochs_per_eval'] > 10]
    for name in ('dense_optimizer', 'sparse_optimizer'):
        checks.extend([m[name]['type'] == 'ADAGRAD', m[name]['options']['learning_rate'] == .1])
    if not all(checks):
        raise ValueError('Reference no longer matches the declared FB control')


def case_config(reference, data, model_dir, fraction, initialization=None):
    if fraction not in (0., .5):
        raise ValueError('Only the two prespecified sampling conditions are allowed')
    config = copy.deepcopy(reference)
    config['storage']['dataset']['dataset_dir'] = str(data) + '/'
    config['storage']['model_dir'] = str(model_dir) + '/'
    config['storage']['checkpoint_dir'] = str(model_dir) + '/'
    config['training']['negative_sampling']['degree_fraction'] = fraction
    if initialization is not None:
        if initialization != 'normal_0001':
            raise ValueError('Only the prespecified Normal(0, 0.001) initializer is allowed')
        config['model']['encoder']['layers'][0][0]['init'] = dict(
            type='NORMAL', options=dict(mean=0., std=.001))
    return config


def training_times(log):
    times = [int(v) / 1000 for v in re.findall(r'Epoch Runtime:\s*(\d+)ms', log)]
    completed = [int(v) for v in re.findall(r'Finished training epoch (\d+)', log)]
    edge_totals = [(int(a), int(b)) for a, b in re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\]', log)]
    full_epochs = sum(a == b == 304727650 for a, b in edge_totals)
    if len(times) != 10 or completed != list(range(1, 11)) or full_epochs != 10 or min(times) <= 0:
        raise ValueError('Training did not complete ten full FB epochs')
    return times


def conditions(study):
    if study == 'sampling':
        return [('degree_00', 0., None), ('degree_05', .5, None)]
    if study == 'repartition':
        return [('fixed', .5, 'fixed'), ('repartition', .5, 'repartition')]
    if study == 'initialization':
        return [('normal_0001', .5, None)]
    raise ValueError('Unknown study: ' + study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'results', 'env', 'tools', 'reference', 'data'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--model', choices=('distmult', 'complex'), required=True)
    parser.add_argument('--study', choices=('sampling', 'repartition', 'initialization'), default='sampling')
    parser.add_argument('--binary', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tools))
    from prepare_ge2_partitioned_view import sha256_file
    from run_arc_ge2_allocated_queue import run_logged, write_json
    from run_arc_ge2_kge_final import inspect_data, checkpoint_names, SPECS, query_membership
    from run_arc_ge2_fb_reproduction import LIB_SHA

    args.results.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    lock = (args.work.parent / f'ge2_accuracy_gpu_{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='preflight', job=args.job, host=os.uname().nodename.split('.')[0],
                 gpu=args.gpu, pid=os.getpid(), model=args.model,
                 factors=[c[1] for c in conditions(args.study)], cases=[],
                 purpose={'sampling': 'Uniform/mixed negative sampling sensitivity',
                          'repartition': 'Disabled epoch repartitioning sensitivity',
                          'initialization': 'Original-Marius entity initialization sensitivity'}[args.study],
                 study=args.study, paper_timing_eligible=False,
                 checkpoint_durable=False, checkpoint_storage='node_local_retained',
                 prediction_direction='both', test_set_tuning=False,
                 interpretation='Prespecified LJ-derived hypothesis; not an authors FB86M recipe')
    state['paper_reference'] = dict(zip(('mrr', 'hits_at_10'),
                                       (.404, .604) if args.model == 'distmult' else (.438, .612)))

    def update(**items):
        state.update(items, updated=datetime.datetime.now().isoformat())
        write_json(args.results / 'progress.json', state)

    def guard():
        alloc = subprocess.check_output(['scontrol', 'show', 'job', args.job, '-o'], text=True, timeout=20)
        if ('JobState=RUNNING ' not in alloc or 'UserId=' + os.environ['USER'] + '(' not in alloc
                or re.search(r'\bNodeList=(\S+)', alloc)[1] != state['host']):
            raise RuntimeError('Owned running allocation on this node is required')
        return alloc

    def gpu_idle():
        apps = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True, timeout=20).strip()
        if apps:
            raise RuntimeError('Selected GPU is occupied: ' + apps)

    def interrupted(sig, frame):
        raise KeyboardInterrupt('Supervisor signal ' + str(sig))

    signal.signal(signal.SIGTERM, interrupted)
    update()
    try:
        if args.study == 'repartition' and (args.binary is None or not args.binary.is_file()):
            raise ValueError('Repartition study requires the separately tested native driver')
        allocation = guard()
        (args.results / 'allocation.txt').write_text(allocation)
        deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp() - 180
        if shutil.disk_usage(args.work).free < 220 * 1024**3:
            raise RuntimeError('Need 220 GiB free for private datasets and retained checkpoints')
        reference = yaml.safe_load((args.reference / 'config.yaml').read_text())
        check_reference(reference, args.model)
        frozen = json.loads((args.reference / 'frozen_inputs.json').read_text())
        library = args.env / 'lib/python3.9/site-packages/gege/libge2.so'
        if sha256_file(library) != frozen['library_sha256'] or frozen['library_sha256'] != LIB_SHA:
            raise RuntimeError('Original GE2 library changed')
        helper_names = ('eval_marius_kge_exact10k.py', 'eval_dglke_kge_exact10k.py',
                        'exact_eval_ranking.py', 'extract_ge2_relation_embeddings.py',
                        'verify_ge2_native_checkpoint_scores.py', 'prepare_ge2_partitioned_view.py',
                        'prepare_ge2_relation_mapping_view.py', 'run_arc_ge2_allocated_queue.py',
                        'run_arc_ge2_kge_final.py', 'run_arc_ge2_fb_reproduction.py')
        for name in helper_names:
            if sha256_file(args.tools / name) != frozen['tools'][name]:
                raise RuntimeError('Previously audited helper changed: ' + name)
        if sha256_file(args.reference / 'config.yaml') != frozen['config_sha256']:
            raise RuntimeError('Reference config changed')
        shutil.copyfile(args.reference / 'result.json', args.results / 'previous_result.json')
        write_json(args.results / 'frozen_inputs.json', dict(library_sha256=LIB_SHA,
                   reference_config_sha256=frozen['config_sha256'], driver_sha256=sha256_file(Path(__file__)),
                   native_driver_sha256=sha256_file(args.binary) if args.binary else None,
                   helpers={name: sha256_file(args.tools / name) for name in helper_names}))
        update(stage='source_data_audit')
        write_json(args.results / 'source_data_audit.json', inspect_data(args.data, 'FB', True))
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
        env.pop('LD_PRELOAD', None)
        env.update(SLURM_JOB_ID=args.job, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu),
                   PATH=f'{args.env}/bin:/usr/bin:/bin', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   LD_LIBRARY_PATH=':'.join(str(args.env / x) for x in (
                       'lib/python3.9/site-packages/gege', 'lib/python3.9/site-packages/torch/lib', 'lib')))
        python, spec = args.env / 'bin/python', SPECS['FB']
        query_path, query_hash = None, spec['eval_sha']
        if args.study != 'sampling':
            import numpy as np
            from deterministic_eval_subset import stable_sample_indices
            seed = 'ge2-fb-optimizer-control-validation-20260920:v1'
            source = args.data / 'edges/validation_edges.bin'
            rows = np.memmap(source, '<i4', mode='r').reshape(-1, 3)
            indices = stable_sample_indices(len(rows), 10000, seed)
            queries = np.asarray(rows[indices.astype(np.int64)])
            membership = query_membership(args.data, queries, spec['nodes'])
            if membership != dict(train=0, validation=10000, test=0):
                raise RuntimeError('Validation membership/leakage mismatch')
            query_path = args.results / 'validation_queries.bin'
            queries.tofile(query_path)
            query_hash = sha256_file(query_path)
            if query_hash != '1ace54b772ccd59818befff780fd138f19a2f79d2b4a34f2237e6eb8dfd114cd':
                raise RuntimeError('Predeclared validation query panel changed')
            indices.astype('<u8').tofile(args.results / 'validation_row_indices.bin')
            write_json(args.results / 'validation_manifest.json', dict(seed=seed, rows=10000,
                       queries_sha256=query_hash, source_sha256=sha256_file(source), membership=membership,
                       selector_sha256=sha256_file(args.tools / 'deterministic_eval_subset.py')))
            interpretation = ('Paper-described repartition path; not unchanged release' if args.study == 'repartition'
                              else 'Single-factor original-Marius initialization; not an authors GE2 FB86M recipe')
            update(evaluation_scope='validation_only', interpretation=interpretation,
                   conditions=[c[0] for c in conditions(args.study)])

        def run(command, name, case):
            guard()
            if deadline - time.time() < 60:
                raise RuntimeError('Allocation deadline reached')
            update(status='running', stage=case.name + ':' + name)
            child_env = dict(env)
            if args.binary and Path(command[0]) == args.binary:
                child_env['GEGE_NO_BINDINGS'] = '1'
            rc = run_logged(list(map(str, command)), child_env, case / (name + '.log'), deadline - time.time(),
                            case / (name + '.hardware.jsonl') if name in ('train', 'evaluate') else None)
            if rc:
                raise RuntimeError(f'{case.name}/{name} failed: {rc}')

        for label, fraction, native_mode in conditions(args.study):
            guard()
            if deadline - time.time() < 4500:
                update(status='deferred', stage='insufficient_allocation_time', next_condition=label)
                return
            gpu_idle()
            work, case = args.work / label, args.results / label
            work.mkdir()
            case.mkdir()
            data, model = work / 'data', work / 'model'
            row = dict(degree_fraction=fraction, condition=label, status='preparing', checkpoint=str(model), run_dir=str(case))
            state['cases'].append(row)
            update(stage=label + ':private_data_copy')
            shutil.copytree(args.data, data)
            meta = yaml.safe_load((data / 'dataset.yaml').read_text())
            meta['dataset_dir'] = str(data) + '/'
            (data / 'dataset.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
            write_json(case / 'data_audit.json', inspect_data(data, 'FB'))
            initializer = label if args.study == 'initialization' else None
            config = case_config(reference, data, model, fraction, initializer)
            row['entity_initialization'] = config['model']['encoder']['layers'][0][0]['init']
            config_path = case / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            row.update(status='training', config_sha256=sha256_file(config_path))
            run([python, '-c', 'import gege,torch; assert torch.cuda.is_available(); '
                 'print(gege.__file__); print(torch.__version__); print(torch.cuda.get_device_name(0))'], 'runtime_gate', case)
            gpu_idle()
            command = ([args.binary, config_path, native_mode] if native_mode
                       else [args.env / 'bin/gege_train', config_path])
            run(command, 'train', case)
            if native_mode:
                text = (case / 'train.log').read_text()
                if 'CONTROL_COMPLETE mode=' + native_mode + ' epochs=10' not in text:
                    raise RuntimeError('Missing native control completion')
                costs = [float(x) for x in re.findall(r'REPARTITION before_epoch=\d+ seconds=([0-9.e+-]+)', text)]
                if len(costs) != (9 if native_mode == 'repartition' else 0):
                    raise RuntimeError('Wrong number of epoch repartitions')
                row['repartition_times_s'] = costs
                row['repartition_total_s'] = sum(costs)
            times = training_times((case / 'train.log').read_text())
            row.update(status='checkpoint_hashing', epoch_times_s=times, average_epoch_s=statistics.mean(times),
                       steady_epoch_s=statistics.mean(times[1:]), paper_timing_eligible=False)
            update(stage=label + ':checkpoint_hashing')
            files = []
            for name in checkpoint_names('FB', data):
                path = model / name
                if not path.is_file():
                    raise RuntimeError('Missing checkpoint component: ' + name)
                if name.startswith('embeddings') and path.stat().st_size != spec['nodes'] * spec['dim'] * 4:
                    raise RuntimeError('Incomplete checkpoint: ' + name)
                files.append(dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path)))
            write_json(case / 'checkpoint_manifest.json', dict(status='ready', source=str(model),
                       host=state['host'], files=files, checkpoint_durable=False))
            shutil.copyfile(model / 'full_config.yaml', case / 'full_config.yaml')
            full_config = yaml.safe_load((case / 'full_config.yaml').read_text())
            observed_init = full_config['model']['encoder']['layers'][0][0]['init']
            if observed_init['type'] != row['entity_initialization']['type']:
                raise RuntimeError('Saved entity initialization type differs from requested configuration')
            if initializer and observed_init['options'] != dict(mean=0., std=.001):
                raise RuntimeError('Saved normal initializer parameters differ from requested configuration')
            queries = query_path or data / 'exact10000_uniform_v1/edges/test_edges.bin'
            row['status'] = 'evaluating'
            run([python, args.tools / 'verify_ge2_native_checkpoint_scores.py', '--run', model,
                 '--eval-edges', queries, '--score', args.model, '--nodes', spec['nodes'],
                 '--relations', spec['relations'], '--width', spec['dim'], '--out', case / 'native_score_check.json'],
                'native_score_gate', case)
            src, dst = work / 'forward_relations.bin', work / 'inverse_relations.bin'
            run([python, args.tools / 'extract_ge2_relation_embeddings.py', '--model', model / 'model.pt_0',
                 '--src-out', src, '--dst-out', dst, '--expected-relations', spec['relations'],
                 '--expected-dim', spec['dim'], '--report', case / 'relation_extract.json'], 'relations', case)
            run([python, args.tools / 'eval_marius_kge_exact10k.py', '--entity-bin', model / 'embeddings.bin',
                 '--src-relation-bin', src, '--dst-relation-bin', dst, '--ge2-data-dir', data,
                 '--eval-edges', queries, '--expected-eval-sha256', query_hash, '--score', args.model,
                 '--filtered', '--tie-policy', 'pessimistic', '--num-test', '10000',
                 '--num-nodes', spec['nodes'], '--num-relations', spec['relations'], '--embedding-dim', spec['dim'],
                 '--batch-size', '32', '--candidate-chunk', '500000', '--filter-chunk', '5000000',
                 '--device', 'cuda:0', '--evaluator-contract', 'ge2_fb_' + args.study + '_control_20260920',
                 '--score-contract', 'ge2_forward_inverse_relation_embeddings', '--out', case / 'exact_eval.json'],
                'evaluate', case)
            quality = json.loads((case / 'exact_eval.json').read_text())
            if (quality['num_ranks'] != 20000 or not quality['filtered']
                    or quality['eval_edges_sha256'] != query_hash
                    or quality['entity_bin_sha256'] != files[0]['sha256']
                    or quality['tie_policy'] != 'pessimistic'
                    or quality['evaluator_sha256'] != frozen['tools']['eval_marius_kge_exact10k.py']
                    or quality['ranking_helper_sha256'] != frozen['tools']['exact_eval_ranking.py']):
                raise RuntimeError('Evaluation contract mismatch')
            for metric in ('mrr', 'hits_at_10'):
                if abs(quality[metric] - (quality['head_' + metric] + quality['tail_' + metric]) / 2) > 1e-12:
                    raise RuntimeError('Pooled metric mismatch')
            row.update(status='done', evaluation=str(case / 'exact_eval.json'), prediction_direction='both')
            for direction in ('', 'head_', 'tail_'):
                for metric in ('mrr', 'hits_at_10'):
                    row[direction + metric] = quality[direction + metric]
            row['evaluation_scope'] = 'test' if args.study == 'sampling' else 'validation'
            if args.study == 'sampling':
                row['difference_from_paper'] = {metric: quality[metric] - value
                                                for metric, value in state['paper_reference'].items()}
            write_json(case / 'result.json', row)
            update()
        update(status='done', stage='complete')
    except BaseException as exc:
        update(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
