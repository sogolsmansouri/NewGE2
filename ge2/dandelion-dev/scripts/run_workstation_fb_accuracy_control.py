#!/usr/bin/env python3
"""Frozen full-FB controls on an authorized workstation or owned ARC allocation."""
import argparse
import copy
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from arc_job_support import save_failure_evidence
from check_fb_multigpu_training_parity import (comparison_passed, tensor_comparison,
                                             validate_execution_scope)


NODES, RELATIONS, WIDTH, EDGES = 86054151, 14824, 100, 304727650
QUERY_SHA = 'a4f3bf65bdff5ce735946982f57950f55f64a52470eca1eeff9c41fa01912685'
ZENODO_LIBRARY_SHA = '9dfa5dc17ab5fee3874d8d449260e0fe17557e23a4691becbe5883fa55c53cf5'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def control_flags(original, partitions, pipeline=None, gradients=None):
    flags = dict(original)
    if (flags.get('GEGE_BASELINE_TRAINING_SEMANTICS') != '1'
            or flags.get('GEGE_SOFTMAX_NEGATIVE_MASS_SCALE') != '1'
            or flags.get('GEGE_BOUNDED_COVER_EPOCH_RELABEL') != '1'):
        raise ValueError('Expected the corrected, unweighted, epoch-relabeled recipe')
    if partitions not in (16, 32):
        raise ValueError('This paired control supports p16 and p32 only')
    enabled = partitions == 32 if pipeline is None else pipeline
    hidden = '6' if enabled else '0'
    flags.update(GEGE_FRAME_CACHE_HIDDEN_FRAMES=hidden,
                 GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='3' if enabled else '0',
                 GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='1' if enabled else '0',
                 GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='1' if enabled else '0',
                 GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='0',
                 GEGE_PARTITION_BUFFER_PEER_RELAY='0', GEGE_STATEFLOW_ALLOW_PEER_RELAY='0',
                 GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME='0',
                 GEGE_TRAINING_INPUT_AUDIT='0', GEGE_TRAINING_PARAMETER_AUDIT='0',
                 GEGE_STATEFLOW_PEER_RELAY_VALIDATE='0',
                 GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE='0', GEGE_PREPARED_BATCH_PIPELINE='0')
    flags.pop('GEGE_TRAINING_REPLAY_SEED', None)
    if gradients is not None:
        if gradients not in ('manual', 'autograd'):
            raise ValueError('Unknown gradient control')
        for decoder in ('DISTMULT', 'COMPLEX'):
            flags['GEGE_FIXED_BUFFER_MANUAL_'+decoder+'_RNS'] = '1' if gradients == 'manual' else '0'
    return flags


def execution_flags(original, gpus, transport, peer_scratch='shared'):
    if gpus not in (1, 2) or transport not in ('peer', 'host') or peer_scratch not in ('shared', 'independent'):
        raise ValueError('Unsupported GPU count or transport')
    flags = dict(original)
    if gpus == 2:
        flags.update(GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='0',
                     GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='1',
                     GEGE_PARTITION_BUFFER_PEER_RELAY='1',
                     GEGE_STATEFLOW_ALLOW_PEER_RELAY='1',
                     GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME='1',
                     GEGE_STATEFLOW_PEER_RUNTIME='on', GEGE_STATEFLOW_PEER_RUNTIME_SCOPE='all',
                     GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK='1' if transport == 'host' else '0',
                     GEGE_STATEFLOW_PEER_RELAY_INDEPENDENT_SCRATCH='1' if peer_scratch == 'independent' else '0',
                     GEGE_STATEFLOW_PEER_RELAY_WAIT_HOST_READY='1',
                     GEGE_STATEFLOW_SERIALIZE_MEM_SWAPS='1',
                     GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET='0' if peer_scratch == 'independent' else '1')
    return flags


def check_execution_gate(gate, gpus, visible, transport, peer_scratch='shared'):
    if (gate.get('gpus') != gpus or gate.get('visible_frames') != visible
            or (gpus > 1 and (gate.get('transport') != transport
                             or gate.get('peer_scratch', 'independent') != peer_scratch))):
        raise RuntimeError('Training-parity gate does not match GPU count, capacity or transport')


def transport_counts(text):
    rows = re.findall(r'\[perf\]\[epoch \d+\]\[peer_relay\][^\n]*', text)
    return {key: sum(int(value) for row in rows for value in re.findall(key+r'=(\d+)', row))
            for key in ('peer_bytes_executed', 'host_fallback_bytes', 'descriptor_mismatch_count')}


def check_dense_replicas(model, gpus):
    import torch
    reference = torch.jit.load(str(model/'model.pt_0'), map_location='cpu').state_dict()
    if not reference:
        raise RuntimeError('Missing dense checkpoint tensors')
    result = {}
    for lane in range(1, gpus):
        current = torch.jit.load(str(model/('model.pt_'+str(lane))), map_location='cpu').state_dict()
        if reference.keys() != current.keys():
            raise RuntimeError('Dense replica checkpoint keys differ')
        result[str(lane)] = {key: tensor_comparison(reference[key], current[key]) for key in reference}
    return result


def control_config(template, decoder, partitions, visible, data, model, epochs=10, degree_fraction=None, gpus=1):
    if decoder not in ('distmult', 'complex'):
        raise ValueError('Unsupported single-GPU decoder')
    if epochs < 1:
        raise ValueError('Control needs at least one complete epoch')
    if gpus not in (1, 2) or (gpus == 2 and (partitions != 32 or visible != 4)):
        raise ValueError('Two-GPU controls require p32/q4')
    if degree_fraction is not None and (not math.isfinite(degree_fraction) or not 0 <= degree_fraction <= 1):
        raise ValueError('Diagnostic degree fraction must be finite and in [0, 1]')
    if (template['model']['decoder']['type'] not in ('DISTMULT', 'COMPLEX')
            or template['training']['batch_size'] != 50000
            or template['model']['encoder']['embedding_dim'] != WIDTH):
        raise ValueError('Expected a full-FB/50K/width100 control template')
    config = copy.deepcopy(template)
    config['model']['decoder']['type'] = decoder.upper()
    config['storage']['dataset']['dataset_dir'] = str(data)+'/'
    config['storage']['embeddings']['options'].update(num_partitions=partitions, buffer_capacity=visible)
    config['storage'].update(device_ids=list(range(gpus)), model_dir=str(model)+'/')
    config['evaluation']['checkpoint_dir'] = str(model)+'/'
    config['training'].update(logical_active_devices=gpus, num_epochs=epochs, save_model=True)
    if degree_fraction is not None:
        config['training']['negative_sampling']['degree_fraction'] = degree_fraction
    return config


def evaluation_panel(source, count, seed, work):
    """Use a fixed random subpanel, not an order-dependent prefix of test rows."""
    if not 1 <= count <= 10000 or seed < 0:
        raise ValueError('Invalid diagnostic query count or selection seed')
    if source.stat().st_size != 10000 * 3 * np.dtype('<i4').itemsize:
        raise ValueError('Expected the frozen 10000-query int32 triple file')
    if count == 10000:
        return source, dict(selection='full_frozen_10000', num_queries=count,
                            query_sha256=digest(source))
    rows = np.fromfile(source, dtype='<i4').reshape(10000, 3)
    indices = np.sort(np.random.default_rng(seed).choice(10000, count, replace=False))
    query = work/'diagnostic_queries.bin'
    index_path = work/'diagnostic_query_indices_u64.bin'
    rows[indices].tofile(query)
    indices.astype('<u8').tofile(index_path)
    return query, dict(selection='uniform_without_replacement_from_frozen_10000',
                       selection_seed=seed, num_queries=count,
                       parent_query_sha256=digest(source), query_sha256=digest(query),
                       selected_indices_path=str(index_path), selected_indices_sha256=digest(index_path))


def schedule_flags(original, schedule):
    """Separate the schedule control from movement and optimizer controls."""
    if schedule == 'bounded':
        return dict(original)
    if schedule != 'legacy-random':
        raise ValueError('Unknown partition schedule control')
    flags = dict(original)
    for key in ('GEGE_BOUNDED_GREEDY_COVER', 'GEGE_BOUNDED_GREEDY_COVER_Q4',
                'GEGE_BOUNDED_COVER_EPOCH_RELABEL', 'GEGE_BOUNDED_Q4_OPTIMAL88',
                'GEGE_BOUNDED_GREEDY_COVER_REVERSE', 'GEGE_OPTIMIZED_CUSTOM_SCHEDULE',
                'GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING', 'GEGE_HYBRID_COVER',
                'GEGE_STATEFLOW_PLANNER', 'GEGE_STATEFLOW_LANE_MATCHING',
                'GEGE_ACCESS_AWARE_STATE_GENERATION', 'GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM'):
        flags[key] = '0'
    flags.pop('GEGE_BOUNDED_STATE_ORDER_FILE', None)
    # Randomized paths may replace all four frames; these controls are synchronous.
    flags['GEGE_STATEFLOW_MAX_ADMITS'] = '4'
    return flags


def engine_flags(original, engine):
    if engine == 'optimized':
        return dict(original)
    if engine != 'zenodo':
        raise ValueError('Unknown training engine')
    # Do not expose PipeGE feature switches to the released-library reference.
    return {}


def data_path_flags(original, mode):
    """Disable shared fast paths without changing the sampler or numerical recipe."""
    if mode == 'current':
        return dict(original)
    if mode != 'reference':
        raise ValueError('Unknown data-path control')
    flags = dict(original)
    for key in ('GEGE_FAST_MAP_TENSORS', 'GEGE_FIXED_BUFFER_BITMAP_MAP',
                'GEGE_FIXED_BUFFER_BITMAP_REUSE_OUTPUTS', 'GEGE_FIXED_BUFFER_MASKED_UPDATE',
                'GEGE_FIXED_BUFFER_COMPACT_ACTIVE', 'GEGE_FIXED_BUFFER_COMPACT_ACTIVE_PREFIX',
                'GEGE_PARTITION_BUFFER_LP_FAST_PATH', 'GEGE_GPU_ACTIVE_EDGE_SHUFFLE',
                'GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS', 'GEGE_MEM_SWAP_EVENT_SYNC',
                'GEGE_SCORE_FILTER_CUDA', 'GEGE_DEG_LOCAL_FILTER_PADDED',
                'GEGE_RESIDENT_LOCAL_LP_DIRECT', 'GEGE_CSR_GATHER', 'GEGE_CSR_UPDATE',
                'GEGE_CSR_UPDATE_REDUCE'):
        flags[key] = '0'
    flags['GEGE_UNIQUE_BACKEND'] = 'sort'
    flags['GEGE_SYNC_BEFORE_SWAP'] = '1'
    return flags


def audited_input_trace(text):
    rows = sorted(re.findall(r'\[training-input\] (.*)', text))
    if not rows:
        raise ValueError('Requested replay has no audited input evidence')
    return dict(batches=len(rows), sha256=hashlib.sha256('\n'.join(rows).encode()).hexdigest())


def allocation_seconds(job, safety_seconds=300):
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'],
                                         text=True, timeout=20)
    match = re.search(r'\bEndTime=(\S+)', allocation)
    if not match or match[1] == 'Unknown':
        raise RuntimeError('Allocation must have a known deadline')
    remaining = datetime.datetime.fromisoformat(match[1]).timestamp()-time.time()-safety_seconds
    if remaining <= 0:
        raise RuntimeError('Allocation safety deadline reached')
    return remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'prefix', 'data16', 'data32', 'tools', 'template', 'flags', 'gate', 'work'):
        parser.add_argument('--'+name, required=True, type=Path)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument('--workstation-host')
    execution.add_argument('--job')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    parser.add_argument('--gpus', type=int, choices=(1, 2), default=1)
    parser.add_argument('--transport', choices=('peer', 'host'), default='peer')
    parser.add_argument('--peer-scratch', choices=('shared', 'independent'), default='shared')
    parser.add_argument('--partitions', type=int, choices=(16, 32), required=True)
    parser.add_argument('--visible', type=int, choices=(4, 8), default=4)
    parser.add_argument('--pipeline', choices=('default', 'on', 'off'), default='default')
    parser.add_argument('--gradients', choices=('manual', 'autograd'), default='manual',
                        help='Autograd is a diagnostic reference, not a production fallback')
    parser.add_argument('--decoder', choices=('distmult', 'complex'), default='distmult')
    parser.add_argument('--engine', choices=('optimized', 'zenodo'), default='optimized',
                        help='Zenodo is the frozen released GE2 library, without PipeGE feature flags')
    parser.add_argument('--data-path', choices=('current', 'reference'), default='current',
                        help='Reference disables mapping/update/storage fast paths for diagnosis only')
    parser.add_argument('--schedule', choices=('bounded', 'legacy-random'), default='bounded',
                        help='Legacy randomized CUSTOM isolates partition-dependent schedule effects')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval-queries', type=int, default=10000,
                        help='Smaller fixed random query panel for early accuracy diagnostics only')
    parser.add_argument('--eval-seed', type=int, default=17)
    parser.add_argument('--degree-fraction', type=float,
                        help='Diagnostic-only sampling intervention; default preserves the template recipe')
    parser.add_argument('--replay-seed', type=int,
                        help='Audit reproducible batch inputs; diagnostic timings only')
    parser.add_argument('--expected-power', type=float)
    parser.add_argument('--evidence', type=Path, help='Small persistent evidence outside the checkpoint directory')
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.eval_queries <= 10000 or args.eval_seed < 0:
        parser.error('Need positive epochs, 1..10000 queries, and nonnegative selection seed')
    if args.replay_seed is not None and args.replay_seed < 0:
        parser.error('Replay seed must be nonnegative')
    if args.degree_fraction is not None and (not math.isfinite(args.degree_fraction) or not 0 <= args.degree_fraction <= 1):
        parser.error('Degree fraction must be finite and in [0, 1]')
    validate_execution_scope(args.job, args.workstation_host)
    if args.job:
        allocation_seconds(args.job)
    if args.evidence and (args.evidence.resolve() == args.work.resolve()
                          or args.work.resolve() in args.evidence.resolve().parents):
        raise ValueError('Evidence must be outside the run directory')
    pipeline = None if args.pipeline == 'default' else args.pipeline == 'on'
    if args.gpus == 2 and (args.engine != 'optimized' or args.gpu != 0 or args.visible != 4
                          or args.partitions != 32 or pipeline is False or args.schedule != 'bounded'
                          or args.data_path != 'current'):
        parser.error('Two-GPU control requires optimized p32/q4, GPUs 0,1, bounded schedule and pipeline')
    if args.schedule == 'legacy-random' and (args.visible != 4 or pipeline is not False):
        parser.error('Legacy randomized schedule control requires q=4 and --pipeline off')
    if args.engine == 'zenodo' and (args.schedule != 'legacy-random' or pipeline is not False
                                   or args.gradients != 'autograd' or args.replay_seed is not None):
        parser.error('Released GE2 requires legacy-random, pipeline off, autograd, and no PipeGE replay seed')
    if args.data_path == 'reference' and (args.engine != 'optimized' or pipeline is not False
                                         or args.gradients != 'autograd' or args.schedule != 'legacy-random'):
        parser.error('Reference data path requires optimized engine, legacy-random, pipeline off, and autograd')
    flags = control_flags(json.loads(args.flags.read_text()), args.partitions, pipeline, args.gradients)
    flags = schedule_flags(flags, args.schedule)
    flags = data_path_flags(flags, args.data_path)
    flags = execution_flags(flags, args.gpus, args.transport, args.peer_scratch)
    if args.replay_seed is not None:
        flags.update(GEGE_TRAINING_REPLAY_SEED=str(args.replay_seed), GEGE_TRAINING_INPUT_AUDIT='1')
    flags = engine_flags(flags, args.engine)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    released_package = args.prefix/'lib/python3.9/site-packages/gege'
    if args.engine == 'optimized':
        gate = json.loads(args.gate.read_text())
        hashes = dict(binary_sha256=digest(args.binary), library_sha256=digest(args.binary.parent/'libge2.so'))
        if gate['status'] != 'passed' or any(gate[key] != value for key, value in hashes.items()):
            raise RuntimeError('The native binary/library must match a passing training-parity gate')
        check_execution_gate(gate, args.gpus, args.visible, args.transport, args.peer_scratch)
        for name, expected in gate['source_sha256'].items():
            if digest(args.source/name) != expected:
                raise RuntimeError('Source changed after the passing gate: '+name)
    else:
        hashes = dict(entrypoint_sha256=digest(args.prefix/'bin/gege_train'),
                      library_sha256=digest(released_package/'libge2.so'))
        if hashes['library_sha256'] != ZENODO_LIBRARY_SHA:
            raise RuntimeError('Unexpected released GE2 library; refusing an unverified baseline')
    devices = [args.gpu] if args.gpus == 1 else [0, 1]
    gpu_uuids = [subprocess.check_output(['nvidia-smi', '-i', str(device), '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip() for device in devices]
    locks = []
    for uuid in sorted(gpu_uuids):
        lock = Path('/home/'+os.environ['USER'])/('.fb_accuracy_gpu_'+uuid+'.lock')
        locks.append(lock.open('a'))
        fcntl.flock(locks[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_uuid = gpu_uuids[0]

    def apps():
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                          '--format=csv,noheader'], text=True)
        return [row for row in csv.reader(io.StringIO(output), skipinitialspace=True) if row and row[0] in gpu_uuids]

    if apps():
        raise RuntimeError('Selected GPU is busy; no process will be displaced')
    if shutil.disk_usage(args.work.parent).free < 80 * 2**30:
        raise RuntimeError('Need 80 GiB free for this preserved checkpoint')
    args.work.mkdir(exist_ok=False)
    state = dict(status='preflight', host=os.uname().nodename, job=args.job,
                 gpu=args.gpu, gpu_uuid=gpu_uuid, partitions=args.partitions,
                 gpus=args.gpus, physical_devices=devices, gpu_uuids=gpu_uuids,
                 transport=args.transport if args.gpus > 1 else 'single',
                 batch_per_gpu=50000, maximum_global_batch=50000*args.gpus,
                 peer_scratch_outside_hidden_pool=args.gpus > 1 and args.peer_scratch == 'independent',
                 visible_frames=args.visible, hidden_frames=int(flags.get('GEGE_FRAME_CACHE_HIDDEN_FRAMES', '0')),
                 commit=args.commit, data_path=args.data_path, paper_ready=False,
                 timing_status='Accuracy diagnostic; timing requires isolation review',
                 purpose='Partition, visible-negative-domain and pipeline controls; not hyperparameter selection',
                 expected_power_w=args.expected_power, foreign_gpu_observations=[], **hashes)
    state.update(gradients=args.gradients, engine=args.engine, driver_sha256=digest(__file__))
    state.update(decoder=args.decoder, replay_seed=args.replay_seed, schedule=args.schedule,
                 requested_epochs=args.epochs, requested_eval_queries=args.eval_queries,
                 requested_degree_fraction=args.degree_fraction,
                 evaluation_status='early diagnostic' if args.epochs < 10 or args.eval_queries < 10000 else 'full control')

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        temporary = args.work/'progress.json.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.work/'progress.json')
        if args.evidence:
            args.evidence.mkdir(parents=True, exist_ok=True)
            temporary = args.evidence/'progress.json.tmp'
            temporary.write_text(json.dumps(state, indent=2)+'\n')
            temporary.replace(args.evidence/'progress.json')

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor signal '+str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    update()
    try:
        selected = args.data32 if args.partitions == 32 else args.data16
        metadata = yaml.safe_load((selected/'dataset.yaml').read_text())
        parent_query = args.data16/'exact10000_uniform_v1/edges/test_edges.bin'
        source_audit = json.loads((args.data16/'source_audit.json').read_text())
        partition_audit = json.loads((args.data32/'data_audit.json').read_text())
        if (digest(parent_query) != QUERY_SHA or source_audit['status'] != 'valid_raw_reconstruction'
                or partition_audit['status'] != 'pass' or not partition_audit['exact_train_match']):
            raise RuntimeError('Frozen FB data/query identity gate failed')
        data_hashes = {}
        for split in ('train', 'validation', 'test'):
            for count_data, p in ((source_audit['splits'][split], args.data16),
                                  (partition_audit['partition_view']['splits'][split], args.data32)):
                path = p/'edges'/(split+'_edges.bin')
                observed = digest(path)
                if observed != count_data['output_sha256']:
                    raise RuntimeError('Staged split changed: '+str(path))
                data_hashes[str(path)] = observed
        for name, expected in source_audit['mappings'].items():
            if digest(args.data16/name) != expected or digest(args.data32/name) != expected:
                raise RuntimeError('ID mappings must match between partition controls')
        if metadata['num_nodes'] != NODES or metadata['num_relations'] != RELATIONS or metadata['num_train'] != EDGES:
            raise RuntimeError('Wrong full-Freebase dataset')
        query, panel = evaluation_panel(parent_query, args.eval_queries, args.eval_seed, args.work)
        query_sha = panel['query_sha256']
        (args.work/'data_identity.json').write_text(json.dumps(dict(evaluation_panel=panel, query_sha256=query_sha,
                                                                  parent_query_sha256=QUERY_SHA, splits=data_hashes), indent=2)+'\n')
        update(evaluation_panel=panel)
        data = args.work/'data'
        data.mkdir()
        for directory in ('edges', 'nodes'):
            (data/directory).symlink_to((selected/directory).resolve(), target_is_directory=True)
        metadata['dataset_dir'] = str(data)+'/'
        (data/'dataset.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
        model = args.work/'model'
        config = control_config(yaml.safe_load(args.template.read_text()), args.decoder,
                                args.partitions, args.visible, data, model, args.epochs, args.degree_fraction, args.gpus)
        update(negative_sampling=config['training']['negative_sampling'],
               sampling_status='diagnostic intervention' if args.degree_fraction is not None else 'template recipe')
        config_path = args.work/'config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        (args.work/'flags.json').write_text(json.dumps(flags, indent=2)+'\n')
        package = args.work/'python'
        package.mkdir()
        if args.engine == 'optimized':
            (package/'gege').symlink_to((args.source/'src/python').resolve(), target_is_directory=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GEGE_', 'PYTHON', 'CONDA', 'SLURM_', 'OMP_', 'MKL_'))}
        env.pop('LD_PRELOAD', None)
        env.update(flags)
        env.update(PATH=str(args.prefix/'bin')+':/usr/bin:/bin',
                   LD_LIBRARY_PATH=f'{args.binary.parent}:{args.prefix}/lib:{args.prefix}/lib/python3.9/site-packages/torch/lib',
                   PYTHONPATH=f'{package}:{args.tools}', GEGE_NO_BINDINGS='1', PYTHONNOUSERSITE='1',
                   CUDA_VISIBLE_DEVICES=','.join(map(str, devices)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='12', MKL_NUM_THREADS='12', OPENBLAS_NUM_THREADS='1')
        if args.gpus > 1:
            env.update(NCCL_P2P_DISABLE='1', NCCL_DEBUG='WARN')
        if args.engine == 'zenodo':
            env.pop('GEGE_NO_BINDINGS', None)
            env.update(LD_LIBRARY_PATH=f'{released_package}:{args.prefix}/lib/python3.9/site-packages/torch/lib:{args.prefix}/lib',
                       PYTHONPATH=str(args.prefix/'lib/python3.9/site-packages')+':'+str(args.tools))
        if args.job:
            env['SLURM_JOB_ID'] = args.job

        def command(argv, stage, timeout):
            update(status=stage, command=list(map(str, argv)))
            log = args.work/(stage+'.log')
            deadline = time.monotonic()+min(timeout, allocation_seconds(args.job) if args.job else timeout)
            with log.open('x') as output:
                child = subprocess.Popen(list(map(str, argv)), env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                update(child_pid=child.pid)
                try:
                    while child.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError(stage)
                        if args.job:
                            validate_execution_scope(args.job, None)
                        if args.expected_power is not None:
                            powers = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, devices)),
                                          '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                                          text=True, timeout=20).splitlines()
                            if any(abs(float(power)-args.expected_power) > .1 for power in powers):
                                raise RuntimeError('GPU power cap changed; refusing a mixed timing cohort')
                        foreign = []
                        all_apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                                            '--format=csv,noheader'], text=True, timeout=20)
                        observed_foreign = []
                        for row in csv.reader(io.StringIO(all_apps), skipinitialspace=True):
                            if not row:
                                continue
                            try:
                                if os.getpgid(int(row[1])) != child.pid:
                                    observed_foreign.append(row)
                                    if row[0] in gpu_uuids:
                                        foreign.append(row)
                            except ProcessLookupError:
                                pass
                        if observed_foreign and not state['foreign_gpu_observations']:
                            update(foreign_gpu_observations=observed_foreign)
                        if foreign:
                            raise RuntimeError('Foreign GPU workload appeared: '+repr(foreign))
                        if stage == 'train':
                            times = [int(v)/1000 for v in re.findall(r'Epoch Runtime:\s*(\d+)ms', log.read_text())]
                            update(epoch_times_s=times, epochs_completed=len(times))
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                    if child.returncode:
                        raise RuntimeError(stage+' exit '+str(child.returncode))
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    update(child_pid=None)

        if args.engine == 'zenodo':
            probe = ('import gege, json, pathlib, sys, torch; '
                     'expected=pathlib.Path(sys.argv[1]).resolve(); '
                     'package=pathlib.Path(gege.__file__).resolve().parent; '
                     'libraries={pathlib.Path(line.split()[-1]).resolve() for line in '
                     'pathlib.Path("/proc/self/maps").read_text().splitlines() if "libge2.so" in line}; '
                     'assert package == expected, (package, expected); '
                     'assert libraries == {expected/"libge2.so"}, libraries; '
                     'print(json.dumps(dict(package=str(package), libraries=sorted(map(str, libraries)), torch=torch.__version__)))')
            command([args.prefix/'bin/python', '-c', probe, released_package], 'released_import', 120)
        train_command = ([args.binary, config_path] if args.engine == 'optimized' else
                         [args.prefix/'bin/python', args.prefix/'bin/gege_train', config_path])
        command(train_command, 'train', 12*3600)
        text = (args.work/'train.log').read_text()
        times = [int(v)/1000 for v in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
        edge_counts = re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%', text)
        if len(times) != args.epochs or edge_counts != [(str(EDGES), str(EDGES))]*args.epochs:
            raise RuntimeError('Expected '+str(args.epochs)+' complete epochs of the full positive-edge workload')
        if args.replay_seed is not None:
            update(input_trace=audited_input_trace(text))
        if args.gpus > 1:
            movement = transport_counts(text)
            update(transport_counts=movement)
            if (movement['descriptor_mismatch_count'] or
                    (args.transport == 'peer' and not movement['peer_bytes_executed']) or
                    (args.transport == 'host' and (movement['peer_bytes_executed'] or not movement['host_fallback_bytes']))):
                raise RuntimeError('The requested coordinated transport was not verified in the training log')
            replicas = check_dense_replicas(model, args.gpus)
            update(dense_replica_comparison=replicas)
            if not comparison_passed(replicas):
                raise RuntimeError('Dense relation replicas diverged; do not silently evaluate only replica zero')
        for name in ('embeddings.bin', 'embeddings_state.bin'):
            if (model/name).stat().st_size != NODES*WIDTH*4:
                raise RuntimeError('Incomplete checkpoint: '+name)
        python = args.prefix/'bin/python'
        command([python, args.tools/'extract_ge2_relation_embeddings.py', '--model', model/'model.pt_0',
                 '--src-out', model/'forward_relations.bin', '--dst-out', model/'inverse_relations.bin',
                 '--expected-relations', RELATIONS, '--expected-dim', WIDTH, '--report', args.work/'relations.json'],
                'extract', 600)
        command([python, args.tools/'eval_marius_kge_exact10k.py', '--entity-bin', model/'embeddings.bin',
                 '--src-relation-bin', model/'forward_relations.bin', '--dst-relation-bin', model/'inverse_relations.bin',
                 '--ge2-data-dir', args.data16, '--eval-edges', query, '--expected-eval-sha256', query_sha,
                 '--score', args.decoder, '--report-directions', 'tail', '--filtered', '--tie-policy', 'pessimistic',
                 '--num-test', args.eval_queries, '--num-nodes', NODES, '--num-relations', RELATIONS, '--embedding-dim', WIDTH,
                 '--batch-size', 32, '--candidate-chunk', 500000, '--filter-chunk', 2000000, '--device', 'cuda:0',
                 '--evaluator-contract', 'fb_partition_control_early_20261005' if args.epochs < 10 or args.eval_queries < 10000
                     else 'fb_partition_control_20261004', '--score-contract',
                 'ge2_forward_inverse_relation_embeddings', '--out', args.work/'exact_eval.json'], 'evaluate', 12*3600)
        quality = json.loads((args.work/'exact_eval.json').read_text())
        with np.load(args.work/'exact_eval.ranks.npz') as saved:
            ranks = saved['tail_ranks']
            ranked_queries = saved['triples']
        if (quality['eval_edges_sha256'] != query_sha or quality['report_directions'] != 'tail'
                or quality['score'] != args.decoder
                or not quality['filtered'] or quality['tf32'] is not False or ranks.shape != (args.eval_queries,)
                or not np.array_equal(ranked_queries, np.fromfile(query, dtype='<i4').reshape(-1, 3))
                or np.any(ranks < 1) or np.any(ranks > NODES)
                or not np.isclose(quality['mrr'], np.mean(1/ranks), atol=1.e-12, rtol=0)
                or not np.isclose(quality['hits_at_10'], np.mean(ranks <= 10), atol=1.e-12, rtol=0)):
            raise RuntimeError('Saved ranks do not reproduce the frozen tail-only evaluator contract')
        update(status='done', epoch_times_s=times, epochs_completed=args.epochs, mrr=quality['mrr'],
               hits_at_10=quality['hits_at_10'], average_epoch_s=float(np.mean(times)),
               steady_epoch_s=float(np.mean(times[1:])) if len(times) > 1 else None, checkpoint=str(model))
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise
    finally:
        if args.evidence:
            save_failure_evidence(args.work, args.evidence)
            ranks = args.work/'exact_eval.ranks.npz'
            if ranks.is_file():
                shutil.copy2(ranks, args.evidence/ranks.name)


if __name__ == '__main__':
    main()
