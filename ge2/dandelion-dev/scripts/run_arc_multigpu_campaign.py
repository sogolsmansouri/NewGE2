#!/usr/bin/env python3
"""Frozen TW/FB 2/4-GPU gates, training, tail evaluation and durable archives."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import yaml

from arc_job_support import run_logged, save_failure_evidence, write_json
from prepare_tw_multigpu import GE2_LIB, cover_check, flags_for
from run_arc_paper_case import archive_checkpoint, digest, hardware_check, verify_evaluation_artifacts
from run_tw_multigpu import check_evaluation, pipege_python_overlay


def multigpu_config(reference, count, system):
    cfg = copy.deepcopy(reference)
    if count not in (2, 4) or cfg['training']['batch_size'] != 50000:
        raise ValueError('Requires 2/4 GPUs and the 50K single-GPU reference')
    cfg['storage']['device_ids'] = list(range(count))
    cfg['training'].update(num_epochs=10, save_model=True, resume_training=False,
                           resume_from_checkpoint='')
    if system == 'pipege':
        cfg['training']['logical_active_devices'] = count
    cfg['evaluation']['epochs_per_eval'] = 1000
    return cfg


def split_hash_for_view(split, canonical, view, partitions):
    if not view:
        return canonical['splits'][split]['sha256']
    item = view['splits'][split]
    if view['num_partitions'] != partitions or item['source_sha256'] != canonical['splits'][split]['sha256']:
        raise ValueError('Partitioned view does not derive from the audited canonical split')
    return item['output_sha256']


def multigpu_flags(reference):
    if int(reference.get('GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES', '-1')) >= 0:
        raise ValueError('Dedicated fixed-frame multi-GPU execution is not implemented')
    flags = flags_for(reference)
    # The existing peer path has independent source snapshots. Do not advertise
    # the single-GPU no-extra-staging contract for that implementation.
    flags['GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET'] = '0'
    return flags


def state_workload_check(text, spec, count, epochs):
    # State initialization records precede Starting epoch, including epoch 1.
    segments = re.split(r'Finished training epoch\s+\d+', text)
    evidence = []
    for epoch, segment in enumerate(segments[:epochs], 1):
        rows = re.findall(r'\[initializeBatches\] device=(\d+) prepare_encode=false task_id=1 items=(\d+) batches=(\d+)', segment)
        if len(rows) != spec['states'] or {int(row[0]) for row in rows} != set(range(count)):
            raise ValueError('Incomplete state workload records')
        items = sum(int(row[1]) for row in rows)
        if items != spec['edges']:
            raise ValueError('State workload differs from frozen edge count')
        prepared = [sum(int(row[2]) for row in rows if int(row[0]) == gpu) for gpu in range(count)]
        actual = re.findall(rf'\[perf\]\[epoch {epoch}\]\[gpu (\d+)\] batches=(\d+)', text)
        if (len(actual) != count or {int(row[0]) for row in actual} != set(range(count))
                or any(int(batches) != prepared[int(gpu)] for gpu, batches in actual)):
            raise ValueError('Completed batches differ from initialized state batches')
        evidence.append(dict(epoch=epoch, state_items=items, completed_batches_by_gpu=prepared))
    return evidence


def training_check(text, spec, count, epochs, gate):
    from run_arc_pipege_quality import timing_summary
    times = [int(x)/1000 for x in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
    finished = list(map(int, re.findall(r'Finished training epoch\s+(\d+)', text)))
    if len(times) != epochs or min(times, default=0) <= 0 or finished != list(range(1, epochs+1)):
        raise ValueError('Missing, duplicate or invalid epochs')
    if (f'Broadcasting model to: {count} GPUs' not in text or 'SynchronousMultiGPUTrainer' not in text
            or re.search(r'CUDA error|out of memory|Traceback|\b(?:nan|inf)\b', text, re.I)):
        raise ValueError('Wrong trainer or numerical/runtime failure')
    workload = re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%', text)
    if workload and workload != [(str(spec['edges']), str(spec['edges']))]*epochs:
        raise ValueError('Incomplete or unexpected positive-edge workload: '+repr(workload))
    state_workload = []
    if not workload and spec['system'] == 'pipege':
        state_workload = state_workload_check(text, spec, count, epochs)
    elif not workload:
        rates = [float(x) for x in re.findall(r'Edges per Second:\s*([\d.eE+-]+)', text)]
        if len(rates) != epochs or any(not math.isclose(rate*duration, spec['edges'], rel_tol=1e-5)
                                      for rate, duration in zip(rates, times)):
            raise ValueError('Declared epoch cardinality does not match the frozen dataset')
    if spec['system'] == 'pipege':
        plans = re.findall(r'Stateflow multi-GPU selected family=.*gpu_count=(\d+).*lanes=(\d+).*microstates=(\d+)', text)
        if not plans or any(tuple(map(int, row)) != (count, count, spec['states']) for row in plans):
            raise ValueError('Unexpected multi-GPU cover or active lanes')
        rows = (spec['nodes']+spec['p']-1)//spec['p']
        frames = re.findall(r'deferred backing allocation device=cuda:(\d+) visible_rows=(\d+) physical_rows=(\d+) dim=(\d+) pinned=true hidden_frames=(\d+)', text)
        expected = (4*rows, (4+spec['hidden'])*rows, spec['width'], spec['hidden'])
        if ({int(row[0]) for row in frames} != set(range(count)) or len(frames) < 2*count
                or any(tuple(map(int, row[1:])) != expected for row in frames)):
            raise ValueError('Unexpected embedding/optimizer frame allocation')
        if re.search(r'descriptor_mismatch_count=[1-9]|(?:dst|src)_mismatch_values=[1-9]', text):
            raise ValueError('Peer-copy descriptor or value mismatch')
        if gate:
            checks = re.findall(r'\[stateflow-peer-validate \d+\].*dst_mismatch_values=0.*src_mismatch_values=0', text)
            if len(checks) < 16:
                raise ValueError('Insufficient live peer-copy value checks')
        if spec['graph'] == 'fb':
            observed = set(map(int, re.findall(r'\[bounded-cover-relabel\] epoch=(\d+) seed=17', text)))
            if not set(range(epochs)) <= observed:
                raise ValueError('FB epoch relabeling did not execute for every epoch')
    return dict(timing_summary(text, times), observed_edge_progress=bool(workload), state_workload=state_workload,
                throughput_counter_valid=not bool(state_workload),
                workload_evidence=('progress totals' if workload else
                    'state edge totals and completed per-GPU batch counts; legacy throughput bucket-pair counter is invalid'
                    if state_workload else
                    'frozen data and completed epochs; throughput reports configured cardinality, not independently counted edges'))


def check_replicas(model, count):
    import torch
    paths = [model/f'model.pt_{i}' for i in range(count)]
    reference = torch.jit.load(str(paths[0]), map_location='cpu').state_dict()
    if not reference:
        raise ValueError('Empty dense checkpoint')
    for path in paths[1:]:
        other = torch.jit.load(str(path), map_location='cpu').state_dict()
        if reference.keys() != other.keys() or any(not torch.equal(value, other[key]) for key, value in reference.items()):
            raise ValueError('Dense model replicas differ; do not evaluate only replica zero')
    if any(not torch.isfinite(value).all() for value in reference.values()):
        raise ValueError('Nonfinite dense weights')
    return dict(replicas=count, equal=True, hashes={p.name:digest(p) for p in paths})


def native_score_environment(training_env, prefix):
    """Score with released bindings, not the training-only Python overlay."""
    env = dict(training_env)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'GEGE_NO_BINDINGS', 'LD_PRELOAD'):
        env.pop(key, None)
    lib = prefix/'lib/python3.9/site-packages'
    env['LD_LIBRARY_PATH'] = f'{lib}/gege:{lib}/torch/lib:{prefix}/lib'
    return env


def prepare(base, commit, execute, reuse_build=None):
    old = Path('/mnt/local/smansou2/paper_matched_300w_20260925')
    old_manifest = json.loads((old/'manifest.json').read_text())
    audited = json.loads((old/'work/prepared.json').read_text())
    if audited['status'] != 'ready':
        raise ValueError('Canonical data preparation is incomplete')
    scripts = Path(__file__).resolve().parent
    repo, build = base/'engine/repo', base/'engine/build_git'
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('Source commit mismatch')
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', 'HEAD'], check=True)
    if digest(Path(__file__)) != digest(repo/'ge2/dandelion-dev/scripts'/Path(__file__).name):
        raise ValueError('Launcher is not from the pinned commit')
    for source in (old/'harness').rglob('*.py'):
        rel = str(source.relative_to(old))
        if digest(source) != old_manifest['helpers'][rel]:
            raise ValueError('Frozen evaluation helper changed: '+rel)
        target = base/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(old/'ge2.zip', base/'ge2.zip')
    if hashlib.md5((base/'ge2.zip').read_bytes()).hexdigest() != '6de3d9702241a0c822971939752d0834':
        raise ValueError('Released GE2 source archive identity mismatch')
    sys.path.insert(0, str(base/'harness/tools'))
    refs = {
        ('ge2', 'tw_dot'): old/'results/290699_ge2_tw_dot',
        ('ge2', 'fb_complex'): old/'results/290697_ge2_fb_complex',
        ('pipege', 'tw_dot'): Path('/mnt/local/smansou2/paper_matched_300w_repair_v2_20260926/results/291518_pipege_tw_dot/tw_dot/final_10e'),
        ('pipege', 'fb_complex'): Path('/home/smansou2/arc_results/runs/pipege_fb_relabel_292745/fb_complex/final_10e'),
    }
    cases = {}
    for (system, case), reference in refs.items():
        cell = copy.deepcopy(old_manifest['cases'][case])
        cfg = yaml.safe_load((reference/'config.yaml').read_text())
        flags = json.loads((reference/'flags.json').read_text()) if system == 'pipege' else {}
        if system == 'pipege' and case == 'fb_complex' and flags.get('GEGE_BOUNDED_COVER_EPOCH_RELABEL') != '1':
            raise ValueError('FB must preserve the latest accuracy fix')
        data = Path(audited['data'][cell['graph']]['view'] if system == 'pipege' else cell['source'])
        expected = dict(audited['data'][cell['graph']])
        expected_train = expected['train_view_sha256'] if system == 'pipege' else expected['splits']['train']['sha256']
        view_path = data/'partitioned_view_manifest.json'
        view = json.loads(view_path.read_text()) if view_path.exists() and system == 'pipege' and cell['p'] != 16 else {}
        data_hashes = {}
        for split in ('train', 'validation', 'test'):
            path = data/'edges'/f'{split}_edges.bin'
            value = digest(path)
            wanted = split_hash_for_view(split, expected, view, cfg['storage']['embeddings']['options']['num_partitions'])
            if split == 'train' and wanted != expected_train:
                raise ValueError('Training view disagrees with its previously verified fingerprint')
            if value != wanted:
                raise ValueError('Prepared input changed: '+str(path))
            data_hashes[str(path.relative_to(data))] = value
        for split in ('train', 'validation', 'test'):
            offsets = data/'edges'/f'{split}_partition_offsets.txt'
            data_hashes[str(offsets.relative_to(data))] = digest(offsets)
            if view and data_hashes[str(offsets.relative_to(data))] != view['splits'][split]['partition_offsets_sha256']:
                raise ValueError('Partitioned view bucket counts changed')
        if system == 'pipege' and data_hashes['edges/train_partition_offsets.txt'] != expected['train_offsets_sha256']:
            raise ValueError('Audited bucket counts changed')
        if digest(Path(cell['query'])) != cell['eval_sha']:
            raise ValueError('Query identity mismatch')
        ref = base/'references'/f'{system}_{case}'
        ref.mkdir(parents=True)
        shutil.copy2(reference/'config.yaml', ref/'single_gpu.yaml')
        write_json(ref/'single_gpu.flags.json', flags)
        if system == 'pipege':
            flags = multigpu_flags(flags)
            if case == 'tw_dot':
                schedule = old/old_manifest['cases'][case]['schedule']
                cover_check(schedule.read_text())
                shutil.copy2(schedule, ref/'schedule.txt')
                flags['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(ref/'schedule.txt')
        for count in (2, 4):
            name = f'{system}_{case}_{count}gpu'
            multi = multigpu_config(cfg, count, system)
            path = ref/f'{count}gpu.yaml'
            path.write_text(yaml.safe_dump(multi, sort_keys=False))
            write_json(ref/f'{count}gpu.flags.json', flags)
            cellspec = dict(cell, system=system, gpus=count, reference=str(reference),
                config=str(path), flags=str(ref/f'{count}gpu.flags.json'), data=str(data),
                data_hashes=data_hashes, batch_semantics='50K per GPU, not fixed global batch',
                memory_contract=('unchanged visible/shared cache plus separate P2P snapshot workspace; '
                                 'not a strict total-frame-budget experiment' if system == 'pipege' else 'released GE2'),
                p=multi['storage']['embeddings']['options']['num_partitions'])
            cases[name] = cellspec
    prefix = Path('/mnt/local/smansou2/ge2-a6000-cuda121')
    compiler = prefix/'bin/x86_64-conda-linux-gnu-c++'
    env = dict(os.environ, PATH=f'{prefix}/bin:/usr/bin:/bin', CUDA_HOME=str(prefix), CUDA_PATH=str(prefix),
        CXX=str(compiler), CUDACXX=str(prefix/'bin/nvcc'),
        LD_LIBRARY_PATH=f'{prefix}/lib:{prefix}/lib/python3.9/site-packages/torch/lib:/usr/lib64')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    build_commit = commit
    if reuse_build is not None:
        prior = json.loads((reuse_build/'manifest.json').read_text())
        old_tree = subprocess.check_output(['git', '-C', str(reuse_build/'engine/repo'), 'rev-parse',
                                           prior['commit']+':ge2/dandelion-dev/gege'], text=True).strip()
        new_tree = subprocess.check_output(['git', '-C', str(repo), 'rev-parse',
                                           commit+':ge2/dandelion-dev/gege'], text=True).strip()
        if old_tree != new_tree:
            raise ValueError('Cannot reuse native build: engine source tree differs')
        for name, expected in prior['engine_hashes'].items():
            if digest(reuse_build/'engine/build_git'/name) != expected:
                raise ValueError('Reused native binary changed')
        build.symlink_to((reuse_build/'engine/build_git').resolve(), target_is_directory=True)
        build_commit = prior.get('built_engine_commit', prior['commit'])
    else:
        execute([prefix/'bin/cmake', '-S', repo/'ge2/dandelion-dev/gege', '-B', build,
        '-DUSE_CUDA=ON', '-DUSE_OMP=OFF', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_ARCHITECTURES=86', '-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc'),
        '-DCMAKE_CXX_COMPILER='+str(compiler), '-DCMAKE_CUDA_HOST_COMPILER='+str(compiler),
        '-DCUDA_HOST_COMPILER='+str(compiler), '-DCUDA_TOOLKIT_ROOT_DIR='+str(prefix),
        '-DCUDAToolkit_ROOT='+str(prefix), '-DCUDA_CUDA_LIBRARY=/usr/lib64/libcuda.so',
        '-DCUDA_CUDA_LIB=/usr/lib64/libcuda.so', '-DLIBNVTOOLSEXT='+str(prefix/'lib/libnvToolsExt.so'),
        f'-DCMAKE_LIBRARY_PATH={prefix}/lib;{prefix}/targets/x86_64-linux/lib;/usr/lib64',
        '-DPYTHON_EXECUTABLE='+sys.executable, '-DPython3_EXECUTABLE='+sys.executable,
            f'-DCMAKE_BUILD_RPATH={build};{prefix}/lib;{prefix}/lib/python3.9/site-packages/torch/lib'], env, 'configure')
        execute([prefix/'bin/cmake', '--build', build, '--target', 'gege_train', 'gege_stateflow_validator_tests',
                 'gege_manual_training_update_test', 'gege_manual_backward_test', '-j', '4'], env, 'build')
    for target in ('gege_stateflow_validator_tests', 'gege_manual_backward_test', 'gege_manual_training_update_test'):
        execute([build/target], env, target)
    manifest = dict(commit=commit, built_engine_commit=build_commit, cases=cases, env=str(prefix), power_w=300,
        engine_hashes={name:digest(build/name) for name in ('libge2.so', 'gege_train')},
        ge2_library_sha256=GE2_LIB, ge2_source_sha256=digest(base/'ge2.zip'),
        files={str(p.relative_to(base)):digest(p) for directory in ('references', 'harness', 'scripts')
               for p in (base/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        evaluation='full-catalog filtered tail-only 10000 queries, pessimistic ties, TF32 off')
    write_json(base/'manifest.json', manifest)
    return manifest


def run_case(base, manifest, name, phase, deadline, archive_root, summary):
    from run_arc_pipege_quality import checkpoint_manifest
    spec = manifest['cases'][name]
    count = spec['gpus']
    results = base/'results'/name/phase
    results.mkdir(parents=True, exist_ok=False)
    work = base/'work'/name/phase
    work.mkdir(parents=True, exist_ok=False)
    model, data = work/'model', work/'data'
    state = dict(case=name, phase=phase, status='preflight', commit=manifest['commit'],
                 built_engine_commit=manifest.get('built_engine_commit', manifest['commit']),
                 memory_contract=spec['memory_contract'],
                 manifest_sha256=digest(base/'manifest.json'), paper_ready=False, checkpoint_durable=False)
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(results/'status.json', state)
        write_json(summary/(name+'_'+phase+'.json'), state)
    envdir = Path(manifest['env'])
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
    env.pop('LD_PRELOAD', None)
    env.update(PATH=f'{envdir}/bin:/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES=','.join(map(str, range(count))), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    def execute(command, label, monitor=False, timeout=None, process_env=None):
        update(stage=label)
        budget = min(deadline-time.time(), timeout or float('inf'))
        if budget < 60:
            raise RuntimeError('Allocation safety deadline reached')
        rc = run_logged(list(map(str, command)), process_env or env, results/(label+'.log'), budget,
                        results/(label+'.hardware.jsonl') if monitor else None)
        if rc:
            raise RuntimeError(f'{label} exited {rc}')
    try:
        update()
        idle_node()
        for rel, value in manifest['files'].items():
            if digest(base/rel) != value:
                raise ValueError('Frozen file changed: '+rel)
        if phase == 'final':
            gate = json.loads((base/'results'/name/'gate/status.json').read_text())
            if gate['status'] != 'gate_passed' or gate['manifest_sha256'] != state['manifest_sha256']:
                raise ValueError('Requires a successful gate for this frozen configuration')
        uuids = []
        for gpu in range(count):
            row = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,name,power.limit',
                                           '--format=csv,noheader,nounits'], text=True).strip().split(',')
            if row[1].strip() != 'NVIDIA RTX A6000' or float(row[2]) != manifest['power_w']:
                raise ValueError('Wrong GPU/power cohort')
            uuids.append(row[0].strip())
        env['CUDA_VISIBLE_DEVICES'] = ','.join(uuids)
        update(gpu_uuids=uuids, power_w=manifest['power_w'])
        lib = envdir/'lib/python3.9/site-packages/gege'
        if digest(lib/'libge2.so') != manifest['ge2_library_sha256'] or digest(base/'ge2.zip') != manifest['ge2_source_sha256']:
            raise ValueError('Released GE2 identity changed')
        binary = envdir/'bin/gege_train'
        if spec['system'] == 'pipege':
            engine = base/'engine'
            if subprocess.check_output(['git', '-C', str(engine/'repo'), 'rev-parse', 'HEAD'], text=True).strip() != manifest['commit']:
                raise ValueError('Pinned source commit changed')
            subprocess.run(['git', '-C', str(engine/'repo'), 'diff', '--exit-code', 'HEAD'], check=True)
            lib = engine/'build_git'
            for file, expected in manifest['engine_hashes'].items():
                if digest(lib/file) != expected:
                    raise ValueError('PipeGE binary changed')
            env.update(pipege_python_overlay(engine, work))
            env.update(json.loads(Path(spec['flags']).read_text()))
            binary = lib/'gege_train'
            if phase == 'gate':
                env.update(GEGE_STATEFLOW_PEER_RELAY_VALIDATE='1', GEGE_STATEFLOW_PEER_RELAY_VALIDATE_FAIL_FAST='1',
                    GEGE_STATEFLOW_PEER_RELAY_VALIDATE_MAX_CHECKS='100000', GEGE_STATEFLOW_DEBUG_VALIDATE='1')
        env['LD_LIBRARY_PATH'] = f'{lib}:{envdir}/lib/python3.9/site-packages/torch/lib:{envdir}/lib'
        repair = manifest.get('ge2_dense_repair') if spec['system'] == 'ge2' else None
        if repair:
            if (repair['original_library_sha256'] != manifest['ge2_library_sha256']
                    or digest(Path(repair['binary'])) != repair['binary_sha256']):
                raise ValueError('Released-library correction identity mismatch')
            env['LD_PRELOAD'] = repair['binary']
            update(ge2_dense_repair=repair, baseline_classification='released GE2 with disclosed dense-sync correction')
        python = envdir/'bin/python'
        execute([python, '-c', 'import torch,gege,json; n=torch.cuda.device_count(); '
            f'assert n=={count}; '
            'p=[[i==j or torch.cuda.can_device_access_peer(i,j) for j in range(n)] for i in range(n)]; '
            'print(json.dumps(dict(gpus=n,peer_access=p,torch=torch.__version__,gege=gege.__file__))); '
            'assert all(all(row) for row in p)'], 'runtime_gate')
        (results/'topology.txt').write_text(subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))
        if shutil.disk_usage(base).free < 150*2**30:
            raise RuntimeError('Insufficient disk for a fresh private dataset/checkpoint')
        update(stage='private_data_copy')
        shutil.copytree(spec['data'], data)
        for rel, expected in spec['data_hashes'].items():
            if digest(data/rel) != expected:
                raise ValueError('Private data identity mismatch: '+rel)
        cfg = yaml.safe_load(Path(spec['config']).read_text())
        meta = yaml.safe_load((data/'dataset.yaml').read_text())
        if (meta['num_nodes'], meta['num_train']) != (spec['nodes'], spec['edges']):
            raise ValueError('Dataset metadata mismatch')
        meta['dataset_dir'] = str(data)+'/'
        (data/'dataset.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
        cfg['storage'].update(dataset=meta, model_dir=str(model)+'/', checkpoint_dir=str(model)+'/')
        cfg['evaluation']['checkpoint_dir'] = str(model)+'/'
        epochs = 2 if phase == 'gate' else 10
        cfg['training']['num_epochs'] = epochs
        config = results/'config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(results/'flags.json', {k:v for k,v in env.items() if k.startswith(('GEGE_', 'CUDA_', 'OMP_', 'PYTORCH_'))})
        idle_node()
        execute([binary, config], 'train', monitor=True, timeout=2700 if phase == 'gate' else None)
        if repair and '[ge2-repair] dense_barrier=generation_two_phase_v1' not in (results/'train.log').read_text():
            raise ValueError('Dense-sync correction did not execute')
        timing = training_check((results/'train.log').read_text(), spec, count, epochs, phase == 'gate')
        samples = [json.loads(line) for line in (results/'train.hardware.jsonl').read_text().splitlines()]
        timing_error = None
        try:
            for uuid in uuids:
                hardware_check(samples, manifest['power_w'], uuid)
        except ValueError as error:
            timing_error = str(error)
        update(**timing, timing_eligible=timing_error is None, timing_error=timing_error)
        if spec['model'] != 'dot':
            write_json(results/'replica_check.json', check_replicas(model, count))
        if phase == 'gate':
            update(status='gate_passed', stage='runtime_and_peer_validation_passed')
            # Only this campaign's disposable gate artifacts are removed.
            shutil.rmtree(model)
            shutil.rmtree(data)
            return
        checkpoints = checkpoint_manifest(model, spec['nodes'], spec['width'], spec['relations'])
        ckpt = results/'checkpoint_manifest.json'
        write_json(ckpt, checkpoints)
        update(status='archiving', stage='archive_before_eval')
        receipt = archive_checkpoint(ckpt, archive_root/name/'checkpoint')
        write_json(results/'archive_receipt.json', receipt)
        update(checkpoint_durable=True)
        helpers = base/'harness/tools'
        if spec['model'] == 'dot':
            command = [python, helpers/'stream_marius_dot_exact_eval.py', '--run-dir', results,
                '--embedding-file', model/'embeddings.bin', '--num-nodes', spec['nodes'], '--dim', spec['width'],
                '--eval-edge-columns', 2, '--filter-edge-columns', 2, '--expected-num-eval-edges', 10000]
        else:
            execute([python, helpers/'verify_ge2_native_checkpoint_scores.py', '--run', model,
                '--eval-edges', spec['query'], '--score', spec['model'], '--nodes', spec['nodes'],
                '--relations', spec['relations'], '--width', spec['width'], '--out', results/'native_score.json'],
                'native_score', process_env=native_score_environment(env, envdir))
            execute([python, helpers/'extract_ge2_relation_embeddings.py', '--model', model/'model.pt_0',
                '--src-out', model/'src_relations.bin', '--dst-out', model/'dst_relations.bin',
                '--expected-relations', spec['relations'], '--expected-dim', spec['width'],
                '--report', results/'relation_extract.json'], 'extract_relations')
            command = [python, helpers/'eval_marius_kge_exact10k.py', '--entity-bin', model/'embeddings.bin',
                '--src-relation-bin', model/'src_relations.bin', '--dst-relation-bin', model/'dst_relations.bin',
                '--score', spec['model'], '--num-nodes', spec['nodes'], '--num-relations', spec['relations'],
                '--embedding-dim', spec['width'], '--num-test', 10000,
                '--score-contract', 'ge2_forward_inverse_relation_embeddings']
        command += ['--report-directions', 'tail', '--eval-edges', spec['query'], '--expected-eval-sha256', spec['eval_sha'],
            '--ge2-data-dir', spec['source'], '--filtered', '--tie-policy', 'pessimistic', '--device', 'cuda:0',
            '--batch-size', 128, '--candidate-chunk', 250000, '--out', results/'exact_eval.json']
        execute(command, 'eval')
        quality = json.loads((results/'exact_eval.json').read_text())
        check_evaluation(quality, spec['eval_sha'])
        identity = verify_evaluation_artifacts(quality, checkpoints)
        write_json(results/'evaluation_identity.json', identity)
        update(status='done_pending_review', stage='complete', mrr=quality['mrr'], hits_at_10=quality['hits_at_10'],
               review='One run; verify multi-GPU accuracy and paired timing before paper inclusion')
        write_json(results/'result.json', state)
        shutil.copytree(results, archive_root/name/'evidence')
        shutil.rmtree(data)
    except BaseException as error:
        update(status='failed', error=repr(error))
        save_failure_evidence(results, summary/(name+'_'+phase+'_evidence'))
        raise


def idle_node():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    jobs = subprocess.check_output(['squeue', '-h', '-t', 'RUNNING,COMPLETING', '-w', os.uname().nodename.split('.')[0],
                                    '-o', '%A'], text=True).split()
    if apps or any(job != os.environ['SLURM_JOB_ID'] for job in jobs):
        raise RuntimeError('Whole idle node required; refusing contended training')


def main():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Supervisor interrupted by signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--reuse-build', type=Path)
    parser.add_argument('--wait-for-lock', action='store_true')
    args = parser.parse_args()
    base = args.base.resolve()
    if not str(base).startswith('/mnt/local/smansou2/'):
        raise ValueError('Execute only on the user node-local tree')
    job = os.environ['SLURM_JOB_ID']
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if ('JobState=RUNNING ' not in allocation or 'NodeList=c30 ' not in allocation
            or os.uname().nodename.split('.')[0] != 'c30'
            or 'UserId='+os.environ['USER']+'(' not in allocation):
        raise RuntimeError('Requires the owned running c30 allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-180
    summary = Path('/home/smansou2/arc_results/runs')/base.name
    summary.mkdir(parents=True, exist_ok=True)
    archive = Path('/mnt/beegfs/smansou2')/base.name
    archive.mkdir(parents=True, exist_ok=True)
    lock = Path('/mnt/local/smansou2/paper_multigpu.lock').open('a')
    state = dict(job=job, commit=args.commit, status='preparing', completed=[], failures={}, remaining=[])
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(summary/'campaign_status.json', state)
    def execute(command, env, name):
        update(stage=name)
        idle_node()
        if deadline-time.time() < 60:
            raise RuntimeError('Allocation safety deadline reached')
        rc = run_logged(list(map(str, command)), env, summary/(name+'.log'), deadline-time.time())
        if rc:
            raise RuntimeError(f'{name} exited {rc}')
    try:
        update()
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not args.wait_for_lock:
                    raise
                if deadline-time.time() < 1800:
                    update(status='waiting_next_allocation', stage='node_lock_not_acquired')
                    return
                update(status='queued', stage='waiting_for_serial_node_lock', remaining=args.cases or [])
                time.sleep(30)
        idle_node()
        if (base/'manifest.json').exists():
            manifest = json.loads((base/'manifest.json').read_text())
            if manifest['commit'] != args.commit:
                raise ValueError('Cannot reuse a campaign with a different commit')
        else:
            manifest = prepare(base, args.commit, execute, args.reuse_build)
        sys.path.insert(0, str(base/'harness/tools'))
        shutil.copy2(base/'manifest.json', summary/'manifest.json')
        order = args.cases or [f'{system}_{case}_{count}gpu' for count in (2, 4)
                              for case in ('tw_dot', 'fb_complex') for system in ('pipege', 'ge2')]
        update(status='running', remaining=order)
        for name in order:
            if deadline-time.time() < 1800:
                update(status='waiting_next_allocation', stage='preserve_remaining_cases')
                break
            update(case=name)
            try:
                for phase in ('gate', 'final'):
                    existing = base/'results'/name/phase/'status.json'
                    if existing.exists():
                        saved = json.loads(existing.read_text())
                        if saved['status'] not in ('gate_passed', 'done_pending_review'):
                            raise ValueError('Previous failed attempt retained; use a fresh campaign for retries')
                        continue
                    run_case(base, manifest, name, phase, deadline, archive, summary)
                state['completed'].append(name)
            except Exception as error:
                state['failures'][name] = repr(error)
            state['remaining'] = [case for case in order if case not in state['completed'] and case not in state['failures']]
            update()
        else:
            update(status='done_with_failures' if state['failures'] else 'done_pending_review')
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
