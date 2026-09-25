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
import shlex
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


def engine_contract(manifest):
    contract = manifest.get('engine')
    if contract is None:
        return dict(commit=CORE, root=str(ENGINE), hashes=BUILD_HASHES,
                    gradient_gate='/home/smansou2/arc_results/runs/pipege_manual_verified_284744/paired_gradient_gate/result.json')
    if (not re.fullmatch(r'[0-9a-f]{40}', contract.get('commit', ''))
            or not Path(contract.get('root', '')).is_absolute()
            or set(contract.get('hashes', {})) != {'libge2_so', 'gege_train'}
            or any(not re.fullmatch(r'[0-9a-f]{64}', x) for x in contract['hashes'].values())
            or not Path(contract.get('gradient_gate', '')).is_absolute()
            or not Path(contract.get('gate_template', '')).is_absolute()):
        raise ValueError('New engine requires a pinned build and paired gradient gate')
    return contract


def reused_data_contract(prepared, reference):
    if (prepared.get('status') != 'ready' or prepared.get('commit') != reference['commit']
            or prepared.get('manifest_sha256') != reference['manifest_sha256']
            or not all(x in prepared.get('data', {}) for x in ('lj', 'tw', 'fb', 'wk'))):
        raise ValueError('Reused preparation identity mismatch')
    return prepared['data']


def preparation_cases(cases):
    """Audit each selected graph once; models must share its physical view."""
    selected = {}
    keys = ('source', 'query', 'eval_sha', 'p', 'columns', 'nodes', 'relations', 'edges')
    for spec in cases.values():
        previous = selected.setdefault(spec['graph'], spec)
        if any(previous[key] != spec[key] for key in keys):
            raise ValueError('Selected models disagree on dataset: '+spec['graph'])
    if not selected:
        raise ValueError('No cases selected')
    return list(selected.values())


def normalize_dataset_metadata(view, expected=None):
    """Relocate a private dataset copy without changing its data contract."""
    view = Path(view).resolve()
    path = view/'dataset.yaml'
    metadata = yaml.safe_load(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError('Dataset metadata must be a mapping')
    metadata['dataset_dir'] = str(view)+'/'
    if expected is not None and metadata != expected:
        raise ValueError('Dataset metadata changed beyond its location')
    if not (view/'edges').is_dir():
        raise ValueError('Dataset edge directory is missing')
    original = yaml.safe_load(path.read_text())
    if metadata != original:
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError('Refusing to rewrite shared dataset metadata')
        temporary = path.with_name(path.name+'.relocate.tmp')
        with temporary.open('x') as output:
            output.write(yaml.safe_dump(metadata, sort_keys=False))
        temporary.replace(path)
    return metadata


def resolved_config_check(config_path, dataset_path, spec):
    # Use the same loader as gege_train: it re-reads dataset.yaml and may
    # override the path embedded in the generated training configuration.
    from gege.tools.configuration.gege_config import load_config
    config = load_config(str(config_path), save=False)
    actual = config.storage.dataset
    if (Path(actual.dataset_dir).resolve() != Path(dataset_path).resolve()
            or (actual.num_nodes, actual.num_relations, actual.num_train) != (
                spec['nodes'], spec['relations'], spec['edges'])
            or config.storage.embeddings.options.num_partitions != spec['p']
            or config.storage.embeddings.options.buffer_capacity != spec['q']
            or config.training.batch_size != 50000):
        raise ValueError('Resolved training configuration disagrees with the audited dataset/preset')
    print(json.dumps(dict(status='pass', dataset_dir=actual.dataset_dir,
                          nodes=actual.num_nodes, edges=actual.num_train,
                          partitions=spec['p'], visible=spec['q'])), flush=True)


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


def loss_contract(flags):
    for name, expected in [('GEGE_SOFTMAX_NEGATIVE_MASS_SCALE', 1.0),
                           ('GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS', 0.0)]:
        if float(flags.get(name, expected)) != expected:
            raise ValueError('Only unweighted softmax is allowed: '+name)


def report_directions(spec):
    directions = spec.get('report_directions', 'tail' if spec['graph'] == 'tw' else 'both')
    if directions not in ('tail', 'both') or (spec['graph'] == 'tw' and directions != 'tail'):
        raise ValueError('Evaluation direction violates the frozen reporting protocol')
    return directions


def evaluation_check(value, spec):
    directions = report_directions(spec)
    if (value.get('num_ranks') != (10000 if directions == 'tail' else 20000)
            or value.get('report_directions', 'both') != directions or value.get('filtered') is not True
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
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--allow-shared-node', action='store_true', help='Quality runs only; timing is provisional')
    args = parser.parse_args()
    sys.path.insert(0, str(args.base/'harness/tools'))
    from prepare_ge2_partitioned_view import sha256_file, verify_bucket_order
    from run_arc_ge2_allocated_queue import audit_data, run_logged, write_json
    from run_arc_pipege_quality import checkpoint_manifest, timing_summary, validate_gradient_gate, validate_update_checks
    from arc_accuracy_gpu_guard import foreign_gpu_pids, guarded_run
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    manifest = json.loads((args.base/'manifest.json').read_text())
    engine_spec = engine_contract(manifest)
    core, engine, build_hashes = engine_spec['commit'], Path(engine_spec['root']), engine_spec['hashes']
    gradient_gate = Path(engine_spec['gradient_gate'])
    if args.case != 'prepare' and args.case not in manifest['cases']:
        raise ValueError('Unknown case')
    result_dir = args.results/args.case
    result_dir.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=True)
    lock = (args.work/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job, host = os.environ['SLURM_JOB_ID'], os.uname().nodename.split('.')[0]
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if not re.fullmatch(r'GPU-[0-9a-f-]+', gpu_uuid):
        raise ValueError('Exactly one physical GPU is required')
    state = dict(status='preflight', job=job, host=host, case=args.case, commit=args.commit,
                 built_engine_commit=core, pid=os.getpid(), checkpoint_durable=False, gpu_uuid=gpu_uuid,
                 allow_shared_node=args.allow_shared_node)
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(result_dir/'status.json', state)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Slurm signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    update()
    repo, build, python = args.work/'repo', engine/'build_git', ENV/'bin/python'
    tools = args.base/'harness/tools'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
    env.pop('LD_PRELOAD', None)
    env.update(PATH=f'{ENV}/bin:/usr/bin:/bin', PYTHONPATH=str(args.work/'python'),
               LD_LIBRARY_PATH=f'{build}:{ENV}/lib:{ENV}/lib/python3.9/site-packages/torch/lib:/usr/lib64',
               PYTHONDONTWRITEBYTECODE='1', GEGE_NO_BINDINGS='1', CUDA_VISIBLE_DEVICES=gpu_uuid,
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
            if args.allow_shared_node:
                rc = guarded_run(list(map(str, command)), dict(env, **(extra or {})), log, deadline-time.time(),
                                 gpu_uuid, log.with_suffix('.gpu_guard.jsonl'),
                                 hardware_monitor=log.with_suffix('.hardware.jsonl') if monitor else None)
            else:
                rc = run_logged(list(map(str, command)), dict(env, **(extra or {})), log, deadline-time.time(),
                                log.with_suffix('.hardware.jsonl') if monitor else None)
            if rc:
                raise RuntimeError(f'Exit {rc}: {log}')
        for rel, digest in {**manifest['references'], **manifest['helpers']}.items():
            if sha256_file(args.base/rel) != digest:
                raise RuntimeError('Frozen file changed: '+rel)
        if args.case == 'prepare':
            reused = None
            if 'reuse_prepared' in manifest:
                reference = manifest['reuse_prepared']
                reused = reused_data_contract(json.loads(Path(reference['path']).read_text()), reference)
            required_space = (max(s['nodes']*s['width']*16 for s in manifest['cases'].values())
                              + 16*1024**3) if reused else 800*1024**3
            if shutil.disk_usage(args.work).free < required_space:
                raise RuntimeError('Insufficient disk for gate/final state and dataset preparation')
            installed = all((engine/item).exists() for item in (
                'repo', 'build_git/libge2.so', 'build_git/gege_train', 'build_git_completed_commit.txt'))
            if host != 'c31' and not installed:
                engine.mkdir(parents=True, exist_ok=True)
                for item in ('repo', 'build_git', 'build_git_completed_commit.txt'):
                    run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                         f'c31:{engine}/{item}', str(engine)+'/'], result_dir/f'engine_copy_{item}.log')
            if not repo.exists():
                run(['git', 'clone', '--no-hardlinks', engine/'repo', repo], result_dir/'clone.log')
            elif subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD']):
                raise RuntimeError('Refusing to replace modified campaign source')
            run(['git', '-C', repo, 'fetch', args.base/'source.bundle', args.commit], result_dir/'fetch.log')
            run(['git', '-C', repo, 'checkout', '--detach', args.commit], result_dir/'checkout.log')
            (args.work/'python').mkdir(exist_ok=True)
            package = args.work/'python/gege'
            if not package.exists():
                package.symlink_to(repo/TREE/'src/python', target_is_directory=True)
            if package.resolve() != (repo/TREE/'src/python').resolve():
                raise RuntimeError('Wrong Python overlay')
        def git(*cmd):
            return subprocess.check_output(['git', '-C', str(repo), *cmd], text=True).strip()
        if (git('rev-parse', 'HEAD') != args.commit or git('diff', 'HEAD')
                or git('rev-parse', args.commit+':'+TREE) != git('rev-parse', core+':'+TREE)
                or (engine/'build_git_completed_commit.txt').read_text().strip() != core):
            raise RuntimeError('Source tree/build attestation mismatch; rebuild required')
        if sha256_file(Path(__file__)) != sha256_file(repo/'ge2/dandelion-dev/scripts'/Path(__file__).name):
            raise RuntimeError('Staged driver differs from committed driver')
        hashes = {n:sha256_file(build/n) for n in ('libge2.so', 'gege_train')}
        if hashes != {'libge2.so':build_hashes['libge2_so'], 'gege_train':build_hashes['gege_train']}:
            raise RuntimeError('Repaired binary changed')
        links = subprocess.check_output(['ldd', str(build/'gege_train')], env=env, text=True)
        if 'not found' in links or f'libge2.so => {build}/libge2.so' not in links:
            raise RuntimeError('Runtime libraries do not resolve to the verified engine')
        (result_dir/'ldd.txt').write_text(links)
        write_json(result_dir/'provenance.json', dict(commit=args.commit, built_engine_commit=core,
                   engine_tree=git('rev-parse', args.commit+':'+TREE), build=hashes,
                   manifest_sha256=sha256_file(args.base/'manifest.json'), driver_sha256=sha256_file(Path(__file__))))
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        jobs = subprocess.check_output(['squeue', '-h', '-t', 'RUNNING,COMPLETING', '-w', host, '-o', '%A'], text=True).split()
        if foreign_gpu_pids(gpu_uuid):
            raise RuntimeError('Selected GPU is occupied; not launching')
        if apps.strip() and not args.allow_shared_node:
            raise RuntimeError('Idle GPUs required before correctness gates or training')
        other_jobs_at_start = [x for x in jobs if x != job]
        update(other_jobs_at_start=other_jobs_at_start,
               timing_status='shared_node_provisional' if other_jobs_at_start or args.allow_shared_node else 'isolation_monitor_required')

        if args.case == 'prepare':
            native = build/'gege_manual_training_update_test'
            for width, mass in [(100,1), (80,1)]:
                log = result_dir/f'native_w{width}_m{mass}.log'
                run([native, 'manual', '--width', width, '--mass', mass], log)
                records = [json.loads(s) for s in log.read_text().splitlines() if s.startswith('{"mode":')]
                if len(records) != 42 or not all(x['pass'] and x['exact'] for x in records):
                    raise RuntimeError('Native gradient parity gate failed')
            for index, rejected in enumerate((['--mass', '8'], ['--log-mass', '2.0794415416798357'])):
                run([native, 'manual', '--expect-unweighted-rejection']+rejected,
                    result_dir/f'rejected_weight_{index}.log')
            if 'engine' in manifest:
                run([build/'gege_manual_backward_test'], result_dir/'native_backward.log')
                if not gradient_gate.exists():
                    run([python, tools/'run_local_gradient_regression.py', '--build', build,
                         '--gege', repo/TREE, '--env', ENV,
                         '--template-case', engine_spec['gate_template'],
                         '--gpu', gpu_uuid,
                         '--output', gradient_gate.parent], result_dir/'paired_gradient_gate.log')
            validate_gradient_gate(json.loads(gradient_gate.read_text()), hashes)
            audits = {}
            for spec in preparation_cases(manifest['cases']):
                source, query = Path(spec['source']), Path(spec['query'])
                if 'source_origin' in spec and reused is None:
                    source.mkdir(parents=True, exist_ok=True)
                    original_source, original_query = Path(spec['source_origin']), Path(spec['query_origin'])
                    required = ['dataset.yaml'] + [f'edges/{split}_{suffix}' for split in ('train','validation','test')
                                                   for suffix in ('edges.bin','partition_offsets.txt')]
                    optional = ['split_manifest.json','partitioned_view_manifest.json',
                                'nodes/node_mapping.txt','edges/relation_mapping.txt']
                    if original_source in original_query.parents:
                        required.append(str(original_query.relative_to(original_source)))
                        prefix = original_query.parent.parent.relative_to(original_source)
                        optional += [str(prefix/x) for x in ('selection_manifest.json','selected_test_row_indices_u64.bin')]
                    # Copy only the actual split contract, not obsolete raw-data
                    # and evaluation-view symlinks in the surrounding cache.
                    inspect = ('import json; from pathlib import Path; '
                               f'p=Path({str(original_source)!r}); required={required!r}; optional={optional!r}; '
                               'missing=[x for x in required if not (p/x).is_file()]; '
                               'assert not missing, missing; '
                               'print(json.dumps(required+[x for x in optional if (p/x).is_file()]))')
                    selected = json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                        'c31','python3 -c '+shlex.quote(inspect)], text=True, timeout=30))
                    file_list = result_dir/f"{spec['graph']}_copy_files.txt"
                    file_list.write_text('\n'.join(selected)+'\n')
                    run(['rsync', '-aL', '--files-from='+str(file_list), '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                         f"c31:{spec['source_origin']}/", str(source)+'/'], result_dir/f"{spec['graph']}_copy.log")
                    if original_source not in original_query.parents:
                        query.parent.mkdir(parents=True, exist_ok=True)
                        run(['rsync', '-aL', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                             f'c31:{original_query}', str(query)],
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
                if reused is not None:
                    view = Path(reused[spec['graph']]['view'])
                elif spec['graph'] in ('fb','wk'):
                    view = args.work/'data'/f"{spec['graph']}_p{spec['p']}"
                    run([python, tools/'prepare_ge2_partitioned_view.py', '--source-data-dir', source,
                         '--output-dir', view, '--num-partitions', spec['p'], '--edge-columns', spec['columns']],
                        result_dir/f"{spec['graph']}_repartition.log")
                counts = np.atleast_1d(np.loadtxt(view/'edges/train_partition_offsets.txt', dtype=np.int64))
                if len(counts) != spec['p']**2 or int(counts.sum()) != spec['edges']:
                    raise RuntimeError('Wrong physical training partition count')
                verify_bucket_order(view/'edges/train_edges.bin', spec['columns'],
                                    (spec['nodes']+spec['p']-1)//spec['p'], spec['p'], counts, 1_000_000)
                md = normalize_dataset_metadata(view)
                audit.update(view=str(view), metadata=md, query_sha256=sha256_file(query),
                             train_view_sha256=sha256_file(view/'edges/train_edges.bin'),
                             train_offsets_sha256=sha256_file(view/'edges/train_partition_offsets.txt'))
                if reused is not None and audit != reused[spec['graph']]:
                    raise RuntimeError('Reused dataset changed since its prior audit: '+spec['graph'])
                audits[spec['graph']] = audit
            for spec in manifest['cases'].values():
                loss_contract(json.loads((args.base/spec['flags']).read_text()))
                report_directions(spec)
                if 'schedule' in spec:
                    schedule_check((args.base/spec['schedule']).read_text(), spec)
                configure(yaml.safe_load((args.base/spec['config']).read_text()),
                          audits[spec['graph']]['metadata'], args.work/'placeholder_model', spec, True)
            write_json(args.work/'prepared.json', dict(status='ready', commit=args.commit,
                       manifest_sha256=sha256_file(args.base/'manifest.json'), data=audits))
            write_json(result_dir/'data_audits.json', audits)
            update(status='done', stage='prepared')
            return

        validate_gradient_gate(json.loads(gradient_gate.read_text()), hashes)
        prepared = json.loads((args.work/'prepared.json').read_text())
        if (prepared['status'] != 'ready' or prepared['commit'] != args.commit
                or prepared['manifest_sha256'] != sha256_file(args.base/'manifest.json')):
            raise RuntimeError('Preparation identity mismatch')
        spec = manifest['cases'][args.case]
        audit = prepared['data'][spec['graph']]
        view = Path(audit['view'])
        normalize_dataset_metadata(view, audit['metadata'])
        if (sha256_file(view/'edges/train_edges.bin') != audit['train_view_sha256']
                or sha256_file(view/'edges/train_partition_offsets.txt') != audit['train_offsets_sha256']
                or sha256_file(Path(spec['query'])) != spec['eval_sha']):
            raise RuntimeError('Prepared inputs changed')
        for split, entry in audit['splits'].items():
            if sha256_file(Path(spec['source'])/'edges'/f'{split}_edges.bin') != entry['sha256']:
                raise RuntimeError('Source/filter split changed: '+split)
        flags = json.loads((args.base/spec['flags']).read_text())
        loss_contract(flags)
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
            run([python, '-c', 'import json,sys; from run_arc_pipege_best import resolved_config_check; '
                 'resolved_config_check(sys.argv[1],sys.argv[2],json.loads(sys.argv[3]))',
                 case/'config.yaml', view, json.dumps(spec)], case/'resolved_config_gate.log',
                dict(effective, PYTHONPATH=str(args.base/'scripts')+':'+env['PYTHONPATH']))
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
                           '--expected-num-eval-edges', 10000, '--report-directions', report_directions(spec)]
            else:
                run([python, tools/'extract_ge2_relation_embeddings.py', '--model', model_dir/'model.pt_0',
                     '--src-out', model_dir/'src_relations.bin', '--dst-out', model_dir/'dst_relations.bin',
                     '--expected-relations', spec['relations'], '--expected-dim', spec['width'],
                     '--report', case/'relation_extract.json'], case/'relation_extract.log')
                command = [python, tools/'eval_marius_kge_exact10k.py', '--entity-bin', model_dir/'embeddings.bin',
                           '--src-relation-bin', model_dir/'src_relations.bin', '--dst-relation-bin', model_dir/'dst_relations.bin',
                           '--score', spec['model'], '--num-nodes', spec['nodes'], '--num-relations', spec['relations'],
                           '--embedding-dim', spec['width'], '--num-test', 10000,
                           '--report-directions', report_directions(spec),
                           '--evaluator-contract', 'pipege_unweighted_a6000_20260921',
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
                       negative_mass_scale=1, report_directions=report_directions(spec), num_ranks=quality['num_ranks'],
                       gpu_uuid=gpu_uuid,
                       commit=args.commit, built_engine_commit=core, scope=spec['scope'], host=host,
                       config_notes=spec['notes'], checkpoint_durable=False,
                       checkpoint=str(model_dir), isolation_violations=isolation, other_jobs_at_start=other_jobs_at_start,
                       paper_readiness='pending_protocol_review' if not isolation and not other_jobs_at_start and not args.allow_shared_node else 'shared_node_timing_provisional'))
        update(status='done', stage='evaluated', final_result=str(case/'result.json'))
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
