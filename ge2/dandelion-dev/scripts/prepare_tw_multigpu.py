#!/usr/bin/env python3
"""Freeze TW-only 2/4-GPU follow-ups without launching or submitting jobs."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import zipfile

import yaml

ENGINE_COMMIT = '14ec00d64ab50090401067425dea2ac5857aac7a'
ENGINE_TREE = '5b666f7f937a18af1fe43c91d0bfc669a012b6d2'
ENGINE_HASHES = {
    'libge2.so': '4a67f7214b52999afeee9349c6180231a3d759fc4404d9295d43ef3938e43d1b',
    'gege_train': 'ece07ff3199787cf90864aff16c6eec7b11185c291f6e0dbf80acd978ec0d9ad',
}
GE2_LIB = '9dfa5dc17ab5fee3874d8d449260e0fe17557e23a4691becbe5883fa55c53cf5'
ARCHIVE_MD5 = '6de3d9702241a0c822971939752d0834'
EVAL_SHA = '93bf1dd7104a2a225800229abb62fbf7477ef006807d0ce523fe6094f3e3ded6'
SPLIT_HASHES = dict(
    train='a9c40f65bece6538a142ae64d01dbfa46435522326e388c12648d4c097da63b0',
    validation='460ac9dc5f4ef3861315e6603b7f690bc37fc2c7e830a6bd8b712699156d015a',
    test='df2e2da8c0784f22b02b680c7864a73f0b2957eb3113917f71e6039e0fabaf9a')
NODES, EDGES = 41652230, 1321528663


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def cover_check(text):
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    parsed = [json.loads(row[len('state='):]) if row.startswith('state=') else list(map(int, row.split()))
              for row in rows]
    if any(len(row) != 4 or any(type(x) is not int for x in row) for row in parsed):
        raise ValueError('Malformed state row')
    states = [set(row) for row in parsed]
    if len(states) != 20 or any(len(s) != 4 or not s <= set(range(16)) for s in states):
        raise ValueError('TW requires twenty four-partition states')
    pairs = [pair for s in states for pair in itertools.combinations(sorted(s), 2)]
    if len(set(pairs)) != 120 or len(pairs) != 120:
        raise ValueError('Cover must contain each unordered pair exactly once')
    # Each affine parallel class has four disjoint states. The runtime chooses
    # lane order; this check establishes that a 5-round four-GPU packing exists.
    remaining = list(range(20))
    groups = []
    while remaining:
        first = remaining[0]
        group = [i for i in remaining if i == first or not states[i] & states[first]]
        if len(group) != 4 or set.union(*(states[i] for i in group)) != set(range(16)):
            raise ValueError('Cover cannot be packed into the required disjoint rounds')
        groups.append(group)
        remaining = [i for i in remaining if i not in group]
    return dict(states=20, unique_pairs=120, disjoint_parallel_classes=groups,
                possible_rounds={'2': 10, '4': 5},
                runtime_lane_order='must be checked in the live gate; not predetermined here')


def make_pipege(reference, gpu_count):
    cfg = copy.deepcopy(reference)
    if cfg['training']['batch_size'] != 50000:
        raise ValueError('Only batch 50K is authorized')
    cfg['storage'].update(device_ids=list(range(gpu_count)), prefetch=True)
    cfg['training'].update(logical_active_devices=gpu_count, num_epochs=10,
                           save_model=True, resume_training=False, resume_from_checkpoint='')
    cfg['evaluation']['epochs_per_eval'] = 1000
    return cfg


def flags_for(reference):
    flags = {k: str(v) for k, v in reference.items()}
    if float(flags.get('GEGE_SOFTMAX_NEGATIVE_MASS_SCALE', '1')) != 1:
        raise ValueError('Refusing a weighted-loss reference')
    flags.update({
        'GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD': '0',
        'GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD': '1',
        'GEGE_PARTITION_BUFFER_PEER_RELAY': '1',
        'GEGE_STATEFLOW_ALLOW_PEER_RELAY': '1',
        'GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME': '1',
        'GEGE_STATEFLOW_LANE_MATCHING': '1',
        'GEGE_STATEFLOW_PEER_RUNTIME_SCOPE': 'all',
        'GEGE_STATEFLOW_PEER_RELAY_INDEPENDENT_SCRATCH': '1',
        'GEGE_STATEFLOW_PEER_RELAY_WAIT_HOST_READY': '1',
        'GEGE_SOFTMAX_NEGATIVE_MASS_SCALE': '1',
        'GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS': '0',
    })
    for name in ('GEGE_BOUNDED_STATE_ORDER_FILE', 'GEGE_STATEFLOW_FORCE_FAMILY', 'GEGE_STATEFLOW_FORCE_VARIANT'):
        flags.pop(name, None)
    return flags


def check_config(cfg, system, gpus):
    storage, train, model = cfg['storage'], cfg['training'], cfg['model']
    options, ns = storage['embeddings']['options'], train['negative_sampling']
    if (storage['device_ids'] != list(range(gpus)) or gpus not in (2, 4)
            or (options['num_partitions'], options['buffer_capacity']) != (16, 4)
            or train['batch_size'] != 50000 or train['num_epochs'] != 10
            or train['negative_sampling_method'] != 'RNS'
            or (ns['num_chunks'], ns['negatives_per_positive'], ns['degree_fraction']) != (50, 1000, .5)
            or model['encoder']['layers'][0][0]['output_dim'] != 100
            or bool(storage['prefetch']) != (system == 'pipege')):
        raise ValueError('Config violates TW follow-up contract')
    for name in ('dense_optimizer', 'sparse_optimizer'):
        if model[name]['type'] != 'ADAGRAD' or model[name]['options']['learning_rate'] != .1:
            raise ValueError('Unexpected optimizer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'harness', 'pipege-reference', 'source-archive'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new directory; never overwrite frozen inputs')
    sys.path.insert(0, str(args.harness/'tools'))
    from run_arc_ge2_dot_reproduction import make_config
    archive = args.source_archive.read_bytes()
    if hashlib.md5(archive).hexdigest() != ARCHIVE_MD5:
        raise ValueError('Released archive identity mismatch')
    with zipfile.ZipFile(args.source_archive) as source:
        released = yaml.safe_load(source.read('dandelion-dev/gege/configs/twitter_16p.yaml'))
    reference = yaml.safe_load((args.pipege_reference/'config.yaml').read_text())
    flags = flags_for(json.loads((args.pipege_reference/'flags.json').read_text()))
    schedule = (args.pipege_reference/'schedule.txt').read_text()
    cover = cover_check(schedule)
    args.out.mkdir(parents=True)
    scripts = Path(__file__).resolve().parent
    for name in ('prepare_tw_multigpu.py', 'run_tw_multigpu.py', 'run_tw_multigpu_queue.sh',
                 'arc_accuracy_gpu_guard.py'):
        shutil.copyfile(scripts/name, args.out/name)
    tools = args.out/'tools'
    tools.mkdir()
    for path in (args.harness/'tools').glob('*.py'):
        shutil.copyfile(path, tools/path.name)
    (args.out/'schedule.txt').write_text(schedule)
    cases = {}
    for system, gpus in itertools.product(('ge2', 'pipege'), (2, 4)):
        name = f'{system}_tw_{gpus}gpu'
        # Paths are bound to a fresh node-local directory by the execution driver.
        cfg = (make_pipege(reference, gpus) if system == 'pipege' else
               make_config(released, Path('/REQUIRES_PRIVATE_DATA'), Path('/REQUIRES_PRIVATE_MODEL'),
                           10, 5650194872900178194))
        cfg['storage']['device_ids'] = list(range(gpus))
        cfg['storage']['dataset'] = dict(dataset_dir='/REQUIRES_PRIVATE_DATA/', num_nodes=NODES,
            num_edges=EDGES, num_relations=1, num_train=EDGES, num_valid=73418259, num_test=73418260)
        cfg['storage']['model_dir'] = '/REQUIRES_PRIVATE_MODEL/'
        cfg['storage']['checkpoint_dir'] = '/REQUIRES_PRIVATE_MODEL/'
        cfg['evaluation'].update(checkpoint_dir='/REQUIRES_PRIVATE_MODEL/', epochs_per_eval=1000)
        check_config(cfg, system, gpus)
        (args.out/(name+'.yaml')).write_text(yaml.safe_dump(cfg, sort_keys=False))
        (args.out/(name+'.flags.json')).write_text(json.dumps(flags if system == 'pipege' else {}, indent=2)+'\n')
        cases[name] = dict(system=system, gpus=gpus, config=name+'.yaml', flags=name+'.flags.json',
                           epochs=10, gate_epochs=2, batch_per_gpu=50000,
                           states=20 if system == 'pipege' else None,
                           rounds=20//gpus if system == 'pipege' else None,
                           hidden_shared_per_gpu=3 if system == 'pipege' else 0,
                           seed=cfg['model']['random_seed'], readiness='prepared; live multi-GPU gate pending')
    manifest = dict(cases=cases, cover=cover, nodes=NODES, edges=EDGES, eval_sha256=EVAL_SHA,
        split_sha256=SPLIT_HASHES,
        engine_commit=ENGINE_COMMIT, engine_tree=ENGINE_TREE, engine_hashes=ENGINE_HASHES,
        ge2_library_sha256=GE2_LIB, ge2_source_md5=ARCHIVE_MD5,
        ge2_source_sha256=hashlib.sha256(archive).hexdigest(),
        evaluation='10,000 fixed held-out queries; filtered exact ranks; tail only; pessimistic ties; TF32 off',
        data_scope='controlled 90/5/5 split, not original-paper split identity',
        batch_semantics='50K per GPU; do not relabel as a fixed-global-batch experiment',
        seed_scope='Preserves each system single-GPU seed; not a shared-seed experiment',
        submission='not submitted; prepare only; do not compete with current accuracy jobs')
    manifest['files'] = {str(p.relative_to(args.out)):sha(p) for p in args.out.rglob('*') if p.is_file()}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='prepared_not_submitted', cases=list(cases), cover=cover), indent=2))


if __name__ == '__main__':
    main()
