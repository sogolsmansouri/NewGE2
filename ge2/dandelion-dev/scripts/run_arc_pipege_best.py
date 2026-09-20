#!/usr/bin/env python3
"""Frozen, serial fast-policy replay with repaired gradients and checkpoint evaluation."""
import argparse
import copy
import datetime
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

CORE = 'fa783f4b9ef685fd5bd969d3426ddce3ee324721'
ENGINE = Path('/mnt/local/smansou2/pipege_manual_verified_284744')
ENV = Path('/mnt/local/smansou2/ge2-a6000-cuda121')
TREE = 'ge2/dandelion-dev/gege'
MODELS = dict(dot='Dot', distmult='DistMult', complex='ComplEx')
BUILD_HASHES = dict(libge2_so='00b3fa5288e8f685127238613c58916adb9250e125ff9cfc9b9ddf9c7079d4d9',
                    gege_train='51e870c4cf4d31b5813cf69a61084f364b15c321e2d0e7df8709ba169f1ce0be')
SPLIT_HASHES = {
    'fb': dict(train='fcbf8d7fc859221e1e95d3ed8c3917c2a68738d1ffbdeec3a0d92517474173ab',
               validation='a9922866ed3c1995bab4e3eff0d98060f01ee1f3424badc3801c73334d3eaedf',
               test='d8a318ec2aa4e3bd088123161c349bf7ac1d31c8f2ef1624415c80509c382aee'),
    'wk': dict(train='807bf4957ca7f8c7accefa2d06e2359af99d996ea82911df1fe5fcb6a099bd22',
               validation='0e361b81f024069fc13a223d20bc011db9abcb04e07dce55ecf2af34b11c10fd',
               test='0e361b81f024069fc13a223d20bc011db9abcb04e07dce55ecf2af34b11c10fd'),
}


def schedule_check(text, spec):
    states = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'state=(\[.*\])', line.strip())
        if match:
            row = json.loads(match[1])
        elif re.fullmatch(r'\d+(?:\s+\d+)*', line.strip()):
            row = list(map(int, line.split()))
        else:
            raise ValueError('Malformed schedule row')
        if (len(row) != spec['q'] or len(set(row)) != spec['q']
                or any(type(x) is not int or not 0 <= x < spec['p'] for x in row)):
            raise ValueError('Invalid resident state')
        states.append(frozenset(row))
    pairs = {pair for s in states for pair in itertools.combinations(sorted(s), 2)}
    if (len(states) != spec['states'] or len(set(states)) != len(states)
            or len(pairs) != math.comb(spec['p'], 2)
            or any(len(b-a) > 3 for a, b in zip(states, states[1:]))):
        raise ValueError('Incomplete cover, duplicate state, or invalid admission bound')


def configure(reference, dataset, model_dir, spec, gate):
    cfg = copy.deepcopy(reference)
    model, storage, train = cfg['model'], cfg['storage'], cfg['training']
    opts, ns = storage['embeddings']['options'], train['negative_sampling']
    layer = model['encoder']['layers'][0][0]
    if (opts['num_partitions'] != spec['p'] or opts['buffer_capacity'] != spec['q']
            or layer['output_dim'] != spec['width'] or layer.get('bias', False)
            or layer.get('activation', 'NONE') != 'NONE' or train['batch_size'] != 50000
            or ns['num_chunks'] != 50 or ns['negatives_per_positive'] != 1000
            or ns['degree_fraction'] != .5 or train['negative_sampling_method'] != 'RNS'
            or model['decoder']['type'] != ('COMPLEX' if spec['model'] == 'complex' else 'DISTMULT')
            or model['sparse_optimizer']['type'] != 'ADAGRAD'
            or model['sparse_optimizer']['options']['learning_rate'] != .1
            or bool(storage['prefetch']) != (spec['graph'] == 'tw')):
        raise ValueError('Preset violates the frozen fast/identity-encoder contract')
    if (dataset['num_nodes'], dataset['num_relations'], dataset['num_train']) != (
            spec['nodes'], spec['relations'], spec['edges']):
        raise ValueError('Dataset dimensions do not match the preset')
    storage['dataset'] = copy.deepcopy(dataset)
    storage.update(model_dir=str(model_dir)+'/', device_ids=[0], device_type='cuda')
    train.update(num_epochs=2 if gate else spec['epochs'], save_model=not gate,
                 resume_training=False, resume_from_checkpoint='')
    train['checkpoint'] = dict(interval=-1, save_best=False, save_state=False)
    cfg['evaluation'].update(epochs_per_eval=1000, checkpoint_dir=str(model_dir)+'/')
    return cfg


def training_check(text, spec, epochs):
    times = [int(x)/1000 for x in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
    finished = [int(x) for x in re.findall(r'Finished training epoch\s+(\d+)', text)]
    edges = re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%', text)
    if (len(times) != epochs or min(times, default=0) <= 0 or finished != list(range(1, epochs+1))
            or edges != [(str(spec['edges']), str(spec['edges']))]*epochs):
        raise ValueError('Incomplete training or wrong positive-edge workload')
    states = set(re.findall(r'Generating bounded GREEDY_COVER ordering states=(\d+)', text))
    frames = re.findall(r'deferred backing allocation device=cuda:\d+ visible_rows=(\d+) physical_rows=(\d+) dim=(\d+) pinned=true hidden_frames=(\d+)', text)
    rows = (spec['nodes']+spec['p']-1)//spec['p']
    expected = tuple(map(str, (spec['q']*rows, (spec['q']+spec['hidden'])*rows, spec['width'], spec['hidden'])))
    if states != {str(spec['states'])} or len(frames) < 2 or any(row != expected for row in frames):
        raise ValueError('Actual schedule or parameter/optimizer frame allocation does not match')
    if (f"[manual_{spec['model']}_rns] enabled=1" not in text
            or re.search(r'\b(nan|inf)\b|CUDA error|device-side assert', text, re.I)):
        raise ValueError('Manual path missing or numerical/CUDA error')
    if spec['graph'] == 'tw' and 'Using bucket-streaming LP path' not in text:
        raise ValueError('TW fast bucket-streaming execution missing')
    return times


def evaluation_check(value, spec):
    if (value.get('num_ranks') != 20000 or value.get('filtered') is not True
            or value.get('eval_edges_sha256') != spec['eval_sha']
            or value.get('tie_policy') != 'pessimistic' or value.get('tf32') is not False):
        raise ValueError('Incomplete or mismatched exact evaluation')
    for key in ('mrr', 'hits_at_10'):
        x = value.get(key, float('nan'))
        if not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Invalid quality metric')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'work', 'results'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.base/'harness/tools'))
    from prepare_ge2_partitioned_view import sha256_file, verify_bucket_order
    from run_arc_ge2_allocated_queue import audit_data, run_logged, write_json
    from run_arc_pipege_quality import checkpoint_manifest, timing_summary, validate_gradient_gate, validate_update_checks
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    manifest = json.loads((args.base/'manifest.json').read_text())
    if args.case != 'prepare' and args.case not in manifest['cases']:
        raise ValueError('Unknown case')
    result_dir = args.results/args.case
    result_dir.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job, host = os.environ['SLURM_JOB_ID'], os.uname().nodename.split('.')[0]
    state = dict(status='preflight', job=job, host=host, case=args.case, commit=args.commit,
                 built_engine_commit=CORE, pid=os.getpid(), checkpoint_durable=False)
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(result_dir/'status.json', state)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Slurm signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    update()
    repo, build, python = args.work/'repo', ENGINE/'build_git', ENV/'bin/python'
    tools = args.base/'harness/tools'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
    env.pop('LD_PRELOAD', None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', PYTHONPATH=str(args.work/'python'),
               LD_LIBRARY_PATH=f'{build}:{ENV}/lib:{ENV}/lib/python3.9/site-packages/torch/lib:/usr/lib64',
               PYTHONDONTWRITEBYTECODE='1', GEGE_NO_BINDINGS='1', CUDA_VISIBLE_DEVICES='0',
               CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1')
    try:
        alloc = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=20)
        if (host not in ('c30', 'c31') or 'JobState=RUNNING ' not in alloc
                or 'UserId='+os.environ['USER']+'(' not in alloc
                or re.search(r'\bNodeList=(\S+)', alloc)[1] != host):
            raise RuntimeError('Must run inside an owned A6000 allocation')
        (result_dir/'allocation.txt').write_text(alloc)
        deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', alloc)[1]).timestamp()-150
        def run(command, log, extra=None, monitor=False):
            update(stage=log.stem, status='running')
            if time.time() >= deadline:
                raise RuntimeError('Allocation deadline reached')
            rc = run_logged(list(map(str, command)), dict(env, **(extra or {})), log, deadline-time.time(),
                            log.with_suffix('.hardware.jsonl') if monitor else None)
            if rc:
                raise RuntimeError(f'Exit {rc}: {log}')
        for rel, digest in {**manifest['references'], **manifest['helpers']}.items():
            if sha256_file(args.base/rel) != digest:
                raise RuntimeError('Frozen file changed: '+rel)
        if args.case == 'prepare':
            if shutil.disk_usage(args.work).free < 800*1024**3:
                raise RuntimeError('Need 800 GiB for gate/final state and partitioned views')
            if host != 'c31':
                ENGINE.mkdir(parents=True, exist_ok=True)
                for item in ('repo', 'build_git', 'build_git_completed_commit.txt'):
                    run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                         f'c31:{ENGINE}/{item}', str(ENGINE)+'/'], result_dir/f'engine_copy_{item}.log')
            run(['git', 'clone', '--no-hardlinks', ENGINE/'repo', repo], result_dir/'clone.log')
            run(['git', '-C', repo, 'fetch', args.base/'source.bundle', args.commit], result_dir/'fetch.log')
            run(['git', '-C', repo, 'checkout', '--detach', args.commit], result_dir/'checkout.log')
            (args.work/'python').mkdir()
            (args.work/'python/gege').symlink_to(repo/TREE/'src/python', target_is_directory=True)
        def git(*cmd):
            return subprocess.check_output(['git', '-C', str(repo), *cmd], text=True).strip()
        if (git('rev-parse', 'HEAD') != args.commit or git('diff', 'HEAD')
                or git('rev-parse', args.commit+':'+TREE) != git('rev-parse', CORE+':'+TREE)
                or (ENGINE/'build_git_completed_commit.txt').read_text().strip() != CORE):
            raise RuntimeError('Source tree/build attestation mismatch; rebuild required')
        if sha256_file(Path(__file__)) != sha256_file(repo/'ge2/dandelion-dev/scripts'/Path(__file__).name):
            raise RuntimeError('Staged driver differs from committed driver')
        hashes = {n:sha256_file(build/n) for n in ('libge2.so', 'gege_train')}
        if hashes != {'libge2.so':BUILD_HASHES['libge2_so'], 'gege_train':BUILD_HASHES['gege_train']}:
            raise RuntimeError('Repaired binary changed')
        validate_gradient_gate(json.loads(Path('/home/smansou2/arc_results/runs/pipege_manual_verified_284744/paired_gradient_gate/result.json').read_text()), hashes)
        links = subprocess.check_output(['ldd', str(build/'gege_train')], env=env, text=True)
        if 'not found' in links or f'libge2.so => {build}/libge2.so' not in links:
            raise RuntimeError('Runtime libraries do not resolve to the verified engine')
        (result_dir/'ldd.txt').write_text(links)
        write_json(result_dir/'provenance.json', dict(commit=args.commit, built_engine_commit=CORE,
                   engine_tree=git('rev-parse', args.commit+':'+TREE), build=hashes,
                   manifest_sha256=sha256_file(args.base/'manifest.json'), driver_sha256=sha256_file(Path(__file__))))
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        jobs = subprocess.check_output(['squeue', '-h', '-w', host, '-o', '%A'], text=True).split()
        if apps.strip() or jobs != [job]:
            raise RuntimeError('Exclusive idle node required before timing/gates')

        if args.case == 'prepare':
            native = build/'gege_manual_training_update_test'
            for width, mass in [(100,1), (100,8), (80,8)]:
                log = result_dir/f'native_w{width}_m{mass}.log'
                run([native, 'manual', '--width', width, '--mass', mass], log)
                records = [json.loads(s) for s in log.read_text().splitlines() if s.startswith('{"mode":')]
                if len(records) != 42 or not all(x['pass'] and x['exact'] for x in records):
                    raise RuntimeError('Native gradient parity gate failed')
            audits = {}
            for name in ('lj_dot', 'tw_dot', 'fb_distmult', 'wk_distmult'):
                spec = manifest['cases'][name]
                source, query = Path(spec['source']), Path(spec['query'])
                if 'source_origin' in spec:
                    source.mkdir(parents=True, exist_ok=False)
                    run(['rsync', '-aL', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                         f"c31:{spec['source_origin']}/", str(source)+'/'], result_dir/f"{spec['graph']}_copy.log")
                    original_source, original_query = Path(spec['source_origin']), Path(spec['query_origin'])
                    if original_source not in original_query.parents:
                        query.parent.parent.mkdir(parents=True, exist_ok=True)
                        run(['rsync', '-aL', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                             f'c31:{original_query.parent.parent}/', str(query.parent.parent)+'/'],
                            result_dir/f"{spec['graph']}_query_copy.log")
                update(stage=spec['graph']+':data_audit')
                if spec['graph'] != 'lj':
                    pins = SPLIT_HASHES.get(spec['graph'])
                    if spec['graph'] == 'tw':
                        splits = json.loads((source/'split_manifest.json').read_text())['splits']
                        pins = {k:v['sha256'] for k,v in splits.items()}
                    audit = audit_data(source, spec['columns'], query, spec['eval_sha'], pins)
                else:
                    md = yaml.safe_load((source/'dataset.yaml').read_text())
                    audit = dict(splits={})
                    for split, key in [('train','num_train'),('validation','num_valid'),('test','num_test')]:
                        path = source/'edges'/f'{split}_edges.bin'
                        if path.stat().st_size != md[key]*8:
                            raise RuntimeError('LJ file size mismatch')
                        audit['splits'][split] = dict(count=md[key], sha256=sha256_file(path))
                    if query.stat().st_size != 80000 or sha256_file(query) != spec['eval_sha']:
                        raise RuntimeError('LJ query hash mismatch')
                view = source
                if spec['graph'] in ('fb','wk'):
                    view = args.work/'data'/f"{spec['graph']}_p{spec['p']}"
                    run([python, tools/'prepare_ge2_partitioned_view.py', '--source-data-dir', source,
                         '--output-dir', view, '--num-partitions', spec['p'], '--edge-columns', spec['columns']],
                        result_dir/f"{spec['graph']}_repartition.log")
                counts = np.atleast_1d(np.loadtxt(view/'edges/train_partition_offsets.txt', dtype=np.int64))
                if len(counts) != spec['p']**2 or int(counts.sum()) != spec['edges']:
                    raise RuntimeError('Wrong physical training partition count')
                verify_bucket_order(view/'edges/train_edges.bin', spec['columns'],
                                    (spec['nodes']+spec['p']-1)//spec['p'], spec['p'], counts, 1_000_000)
                md = yaml.safe_load((view/'dataset.yaml').read_text())
                md['dataset_dir'] = str(view)+'/'
                audit.update(view=str(view), metadata=md, query_sha256=sha256_file(query),
                             train_view_sha256=sha256_file(view/'edges/train_edges.bin'),
                             train_offsets_sha256=sha256_file(view/'edges/train_partition_offsets.txt'))
                audits[spec['graph']] = audit
            for spec in manifest['cases'].values():
                if 'schedule' in spec:
                    schedule_check((args.base/spec['schedule']).read_text(), spec)
                configure(yaml.safe_load((args.base/spec['config']).read_text()),
                          audits[spec['graph']]['metadata'], args.work/'placeholder_model', spec, True)
            write_json(args.work/'prepared.json', dict(status='ready', commit=args.commit,
                       manifest_sha256=sha256_file(args.base/'manifest.json'), data=audits))
            write_json(result_dir/'data_audits.json', audits)
            update(status='done', stage='prepared')
            return

        prepared = json.loads((args.work/'prepared.json').read_text())
        if (prepared['status'] != 'ready' or prepared['commit'] != args.commit
                or prepared['manifest_sha256'] != sha256_file(args.base/'manifest.json')):
            raise RuntimeError('Preparation identity mismatch')
        spec = manifest['cases'][args.case]
        audit = prepared['data'][spec['graph']]
        view = Path(audit['view'])
        if (sha256_file(view/'edges/train_edges.bin') != audit['train_view_sha256']
                or sha256_file(view/'edges/train_partition_offsets.txt') != audit['train_offsets_sha256']
                or sha256_file(Path(spec['query'])) != spec['eval_sha']):
            raise RuntimeError('Prepared inputs changed')
        for split, entry in audit['splits'].items():
            if sha256_file(Path(spec['source'])/'edges'/f'{split}_edges.bin') != entry['sha256']:
                raise RuntimeError('Source/filter split changed: '+split)
        flags = json.loads((args.base/spec['flags']).read_text())
        if 'schedule' in spec:
            schedule = args.base/spec['schedule']
            schedule_check(schedule.read_text(), spec)
            flags['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(schedule)
        for gate in (True, False):
            epochs = 2 if gate else spec['epochs']
            case = result_dir/('gate_2e' if gate else f'final_{epochs}e')
            case.mkdir()
            model_dir = args.work/'models'/job/args.case/case.name
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            if model_dir.exists():
                raise RuntimeError('Refusing to overwrite a checkpoint directory')
            config = configure(yaml.safe_load((args.base/spec['config']).read_text()),
                               audit['metadata'], model_dir, spec, gate)
            (case/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
            effective = dict(flags, GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY='1' if gate else '0',
                             GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY_MAX='8')
            write_json(case/'flags.json', effective)
            write_json(case/'contract.json', dict(spec, commit=args.commit, model_dir=str(model_dir),
                       gradient_mode='repaired_manual', hidden_policy='shared', epochs=epochs,
                       config_sha256=sha256_file(case/'config.yaml'), flags_sha256=sha256_file(case/'flags.json')))
            run([build/'gege_train', case/'config.yaml'], case/'train.log', effective, monitor=True)
            text = (case/'train.log').read_text()
            times = training_check(text, spec, epochs)
            if gate:
                checks = validate_update_checks(text, True, MODELS[spec['model']])
                write_json(case/'result.json', dict(status='gate_pass', epoch_times_s=times, checks=checks))
                continue
            timing = timing_summary(text, times)
            write_json(case/'training_result.json', dict(status='trained', **timing))
            checkpoints = checkpoint_manifest(model_dir, spec['nodes'], spec['width'], spec['relations'])
            write_json(case/'checkpoint_manifest.json', checkpoints)
            if spec['model'] == 'dot':
                command = [python, tools/'stream_marius_dot_exact_eval.py', '--run-dir', case,
                           '--embedding-file', model_dir/'embeddings.bin', '--num-nodes', spec['nodes'],
                           '--dim', spec['width'], '--eval-edge-columns', 2, '--filter-edge-columns', 2,
                           '--expected-num-eval-edges', 10000]
            else:
                run([python, tools/'extract_ge2_relation_embeddings.py', '--model', model_dir/'model.pt_0',
                     '--src-out', model_dir/'src_relations.bin', '--dst-out', model_dir/'dst_relations.bin',
                     '--expected-relations', spec['relations'], '--expected-dim', spec['width'],
                     '--report', case/'relation_extract.json'], case/'relation_extract.log')
                command = [python, tools/'eval_marius_kge_exact10k.py', '--entity-bin', model_dir/'embeddings.bin',
                           '--src-relation-bin', model_dir/'src_relations.bin', '--dst-relation-bin', model_dir/'dst_relations.bin',
                           '--score', spec['model'], '--num-nodes', spec['nodes'], '--num-relations', spec['relations'],
                           '--embedding-dim', spec['width'], '--num-test', 10000,
                           '--evaluator-contract', 'pipege_fast_a6000_20260920',
                           '--score-contract', 'ge2_forward_inverse_relation_embeddings']
            command += ['--eval-edges', spec['query'], '--expected-eval-sha256', spec['eval_sha'],
                        '--ge2-data-dir', spec['source'], '--filtered', '--tie-policy', 'pessimistic',
                        '--device', 'cuda:0', '--batch-size', 128, '--candidate-chunk', 250000,
                        '--out', case/'exact_eval.json']
            run(command, case/'exact_eval.log', monitor=True)
            quality = json.loads((case/'exact_eval.json').read_text())
            evaluation_check(quality, spec)
            isolation = [r for r in map(json.loads, (case/'train.hardware.jsonl').read_text().splitlines())
                         if r['other_jobs'] or r['other_processes']]
            write_json(case/'result.json', dict(status='done', train_status=0, exact_eval_status=0,
                       **timing, mrr=quality['mrr'], hits_at_10=quality['hits_at_10'],
                       commit=args.commit, built_engine_commit=CORE, scope=spec['scope'],
                       config_notes=spec['notes'], checkpoint_durable=False,
                       checkpoint=str(model_dir), isolation_violations=isolation,
                       paper_readiness='pending_protocol_review' if not isolation else 'isolation_failed'))
        update(status='done', stage='evaluated', final_result=str(case/'result.json'))
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
