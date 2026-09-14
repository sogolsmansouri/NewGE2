#!/usr/bin/env python3
"""Allocation-bounded ARC TW study using the validated fixed/shared-frame runner."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from run_local_tw_fixed_frames import (NODES, TRAIN, cases, digest, environment, execute,
    gpu_processes, schedule_info, validate_prepared_data, write_json)


def study_rows():
    rows = []
    def add(row):
        row = dict(row)
        row['case'] = f"q{row['q']}_" + row['case']
        if row['case'] not in {r['case'] for r in rows}:
            rows.append(row)
    # Matched p comparison, followed by every k in the original sweep.
    for p in (16, 11, 10, 8):
        for row in cases(q=4, min_frames=7, max_frames=7, partitions=p,
                         frame_budget_gib=32, frame_policy='shared'):
            add(row)
    for row in cases(frame_policy='shared'):
        add(row)
    for q, k in ((4, 7), (4, 10)):
        for row in cases(q=q, min_frames=k, max_frames=k, partitions=16):
            add(row)
    for row in cases():
        add(row)
    return rows


def all_resident_payload(nodes, edges, cuda_bytes, fraction=0.9):
    # One mapped int64 endpoint pair per edge; no second prefetched state.
    entity = nodes * 800
    graph = edges * 16
    return dict(p_min=2, configured_q=4, effective_visible_frames=2, hidden_frames=0,
        resident_states=1, entity_bytes=entity, graph_bytes=graph,
        minimum_payload_bytes=entity+graph, capacity_bytes=cuda_bytes,
        budget_fraction=fraction, allowed_bytes=int(fraction*cuda_bytes),
        status='infeasible_payload' if entity+graph > fraction*cuda_bytes else 'requires_runtime_memory_gate',
        excludes='batch, remapping, dense parameters, CUDA context and allocator reserve',
        graph_policy='whole_state_gpu_graph')


def relocate_dataset_metadata(data):
    """Relocate copied dataset metadata without touching edges or node IDs."""
    path = data / 'dataset.yaml'
    metadata = yaml.safe_load(path.read_text())
    target = str(data.resolve()) + '/'
    if metadata.get('dataset_dir') == target:
        return
    original_sha = digest(path)
    backup = data / ('dataset.before_relocation.' + original_sha + '.yaml')
    if not backup.exists():
        shutil.copy2(path, backup)
    previous = metadata.get('dataset_dir')
    metadata['dataset_dir'] = target
    temp = data / 'dataset.yaml.relocating'
    temp.write_text(yaml.safe_dump(metadata, sort_keys=False))
    temp.replace(path)
    write_json(data / 'metadata_relocation.json', dict(previous_dataset_dir=previous,
        dataset_dir=target, previous_sha256=original_sha, sha256=digest(path),
        backup=str(backup), edge_payload_changed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--end-epoch', type=float, required=True)
    args = parser.parse_args()
    work, results = args.work.resolve(), args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    if os.environ.get('SLURM_JOB_ID') != args.job:
        raise RuntimeError('This supervisor must run inside the authorized allocation')
    if 'TW_PHYSICAL_GPU' not in os.environ or 'TW_RUNTIME_ENV' not in os.environ:
        raise RuntimeError('Select the allocated physical GPU and node-local runtime explicitly')
    deadline = args.end_epoch - 300
    os.environ['TW_DEADLINE_EPOCH'] = str(deadline)
    rows = study_rows()
    build = work / 'build_git'
    repo = work / 'repo'
    write_json(results / 'plan.json', dict(commit=args.commit, job=args.job,
        batch_size=50000, epochs=5, eval=False, cases=rows,
        physical_gpu=os.environ['TW_PHYSICAL_GPU'], allocation_end=args.end_epoch,
        note='Shared-node timing; foreign GPUs may use CPU memory/storage. No uncontended-node claim.'))
    completed = []

    def status(state, **extra):
        write_json(results / 'status.json', dict(status=state, pid=os.getpid(), updated=time.time(), **extra))

    def collect(row, state, **extra):
        completed.append(dict(row, status=state, **extra))
        write_json(results / 'results.json', completed)
        fields = ['case', 'p', 'q', 'k', 'hp', 'hs', 'shared_hidden', 'status', 'average_epoch_s', 'steady_epoch_s', 'run_dir']
        with (results / 'manifest.tsv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(completed)

    def freeze(run, row):
        artifacts = run / 'artifacts'
        if artifacts.exists():
            raise RuntimeError(f'Existing run artifacts at {run}; preserve them')
        artifacts.mkdir(parents=True)
        for name in ('gege_train', 'libge2.so', 'gege_fixed_frame_buffer_test', 'CMakeCache.txt'):
            shutil.copy2(build / name, artifacts / name)
        for src, dst in [('reference.yaml', 'reference.yaml'), ('reference_flags.sh', 'reference_flags.sh'),
                         ('source_files.json', 'source_files.json')]:
            shutil.copy2(work / 'input' / src, artifacts / dst)
        for name in ('run_local_tw_fixed_frames.py', 'prepare_ge2_partitioned_view.py', 'run_arc_tw_memory_study.py'):
            shutil.copy2(work / 'input' / name, artifacts / name)
        source = work / 'input/schedules' / f"q{row['q']}" / f"p{row['p']}"
        row.update(schedule_info(source / 'states.txt', source / 'cover.json', row['q']))
        for name, suffix in (('states.txt', 'txt'), ('cover.json', 'json')):
            shutil.copy2(source / name, artifacts / f"p{row['p']}.{suffix}")
        (artifacts / 'source_commit.txt').write_text(args.commit + '\n')
        write_json(artifacts / 'hashes.json', {p.name: digest(p) for p in artifacts.iterdir()})

    def checked(command, log, env=None, timeout=900):
        with log.open('w') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            try:
                return child.wait(timeout=min(timeout, max(1, deadline-time.time())))
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise

    try:
        while not (work / 'build_git_completed_commit.txt').exists() or not (work / 'input/data_transfer_complete').exists():
            status('waiting_for_build_and_data')
            if time.time() > deadline:
                status('needs_next_allocation', remaining=[r['case'] for r in rows])
                return
            time.sleep(15)
        if (work / 'build_git_completed_commit.txt').read_text().strip() != args.commit:
            raise RuntimeError('Build/source commit mismatch')
        if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() != args.commit:
            raise RuntimeError('Pulled repository differs from approved commit')
        source_files = json.loads((work / 'input/source_files.json').read_text())
        if not all(digest(repo / name) == sha for name, sha in source_files.items()):
            raise RuntimeError('Extracted source differs from committed source')
        status('verifying_full_source')
        validate_prepared_data(work / 'data/p16', 16)
        relocate_dataset_metadata(work / 'data/p16')
        frozen_hashes = {name: digest(build / name) for name in ('gege_train', 'libge2.so', 'gege_fixed_frame_buffer_test')}
        write_json(results / 'build.json', dict(commit=args.commit, hashes=frozen_hashes,
            environment=os.environ['TW_RUNTIME_ENV'], pythonpath=os.environ.get('TW_RUNTIME_PYTHONPATH')))
        fixture = work / 'fixture'
        (fixture / 'edges').mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1987)
        for split, count in (('train',32768),('validation',256),('test',256)):
            rng.integers(0,1024,size=(count,2),dtype=np.int32).tofile(fixture / 'edges' / f'{split}_edges.bin')
        (fixture / 'dataset.yaml').write_text(yaml.safe_dump(dict(num_nodes=1024,num_edges=32768,
            num_train=32768,num_valid=256,num_test=256,num_relations=1)))
        memory_env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ['TW_PHYSICAL_GPU'])
        capacity = int(subprocess.check_output([sys.executable,'-c',
            'import torch; print(torch.cuda.get_device_properties(0).total_memory)'],env=memory_env,text=True))
        write_json(results / 'all_resident_p2_preflight.json', all_resident_payload(NODES,TRAIN,capacity))
        for row in rows:
            if time.time() + 1800 > deadline:
                status('needs_next_allocation', remaining=[r['case'] for r in rows if r['case'] not in {c['case'] for c in completed}])
                return
            while gpu_processes():
                status('waiting_for_idle_gpu', case=row['case'])
                if time.time() + 1800 > deadline:
                    status('needs_next_allocation', remaining=[r['case'] for r in rows if r['case'] not in {c['case'] for c in completed}])
                    return
                time.sleep(15)
            if any(digest(build / name) != sha for name,sha in frozen_hashes.items()):
                raise RuntimeError('Build changed mid-study')
            run = results / row['case']
            run.mkdir()
            freeze(run, row)
            status('correctness_gate', case=row['case'])
            tiny_data = work / f"fixture_p{row['p']}"
            if not (tiny_data / 'partitioned_view_manifest.json').exists():
                rc = checked([sys.executable, str(work / 'input/prepare_ge2_partitioned_view.py'),
                    '--source-data-dir',str(fixture),'--output-dir',str(tiny_data),
                    '--num-partitions',str(row['p']),'--edge-columns','2'],run / 'fixture_prepare.log')
                if rc: raise RuntimeError('Tiny dataset conversion failed')
            value_env = environment(work,run,row,1024)
            rc = checked([str(run / 'artifacts/gege_fixed_frame_buffer_test'),str(row['p']),
                str(run / 'artifacts' / f"p{row['p']}.txt"),str(work / 'value_test.bin')],
                run / 'value_test.log',env=value_env,timeout=120)
            if rc:
                collect(row,'value_gate_failed',run_dir=str(run))
                continue
            gate_row = dict(row,case='gate')
            gate = execute(work,run,gate_row,tiny_data,work / 'models' / (row['case']+'_gate'),5,tiny=True)
            if gate['status'] != 'valid_timing':
                collect(row,'training_gate_failed',run_dir=str(run))
                continue
            data = work / 'data' / f"p{row['p']}"
            if not (data / 'partitioned_view_manifest.json').exists():
                status('repartitioning',case=row['case'],p=row['p'])
                rc = checked([sys.executable,str(work / 'input/prepare_ge2_partitioned_view.py'),
                    '--source-data-dir',str(work / 'data/p16'),'--output-dir',str(data),
                    '--num-partitions',str(row['p']),'--edge-columns','2'],run / 'prepare.log',timeout=3600)
                if rc:
                    collect(row,'data_prepare_failed',run_dir=str(run))
                    continue
            validate_prepared_data(data,row['p'])
            relocate_dataset_metadata(data)
            if time.time()+1800 > deadline:
                status('needs_next_allocation', remaining=[r['case'] for r in rows if r['case'] not in {c['case'] for c in completed}])
                return
            subprocess.run(['nvidia-smi','-q'],stdout=(run / 'node_gpu_before.log').open('w'),check=True)
            status('training',case=row['case'],detail=str(run / 'status.json'))
            result = execute(work,run,dict(row,case='full'),data,work / 'models' / row['case'],5)
            subprocess.run(['nvidia-smi','-q'],stdout=(run / 'node_gpu_after.log').open('w'),check=True)
            collect(row,result['status'],average_epoch_s=result.get('average_epoch_s'),
                    steady_epoch_s=result.get('steady_epoch_s'),run_dir=str(run / 'full'))
        status('finished',total=len(completed),valid=sum(r['status']=='valid_timing' for r in completed))
    except BaseException as error:
        status('stopped',error=str(error))
        raise


if __name__ == '__main__':
    main()
