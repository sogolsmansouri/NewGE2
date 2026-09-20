#!/usr/bin/env python3
"""Prespecified LJ uniform-versus-mixed negative sampling accuracy control."""

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import time

import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def allocation(job):
    output = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if ('JobState=RUNNING' not in output or 'UserId=' + os.environ['USER'] + '(' not in output
            or re.search(r'\bNodeList=(\S+)', output)[1] != socket.gethostname().split('.')[0]):
        raise RuntimeError('Owned running allocation on this node is required')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'results', 'env', 'tools', 'template', 'partitioned', 'logical', 'queries'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', host=socket.gethostname(), job=args.job,
                 purpose=__doc__, paper_timing_eligible=False, cases=[],
                 factors=[.5, 0.], seed=5650194872900178194,
                 scope='Only degree_fraction changes; paths and checkpoint saving are administrative overrides.')

    def save(**kw):
        state.update(kw, updated=datetime.datetime.now().isoformat())
        tmp = args.results / 'progress.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(args.results / 'progress.json')

    save()
    child = None
    try:
        alloc = allocation(args.job)
        (args.results / 'allocation.txt').write_text(alloc)
        deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', alloc)[1]).timestamp() - 180
        library = args.env / 'lib/python3.9/site-packages/gege/libge2.so'
        if digest(library) != '9dfa5dc17ab5fee3874d8d449260e0fe17557e23a4691becbe5883fa55c53cf5':
            raise ValueError('Unexpected released GE2 library')
        train_hash = digest(args.partitioned / 'edges/train_edges.bin')
        if train_hash != 'c01844c5ba9d3e7780e07a2406f3a1e0822cb2f3bb5fb9324d545bdb804d2ade':
            raise ValueError('Training input differs from prior seed control: ' + train_hash)
        query_hash = digest(args.queries)
        if query_hash != '1d1669b2fa920c8492341989a0c0ff2c741d97e74ace67ef76735ed29fb5691e':
            raise ValueError('Query set changed')
        for path in ('nodes/node_mapping.txt',):
            if digest(args.partitioned / path) != digest(args.logical / path):
                raise ValueError('Logical/partitioned entity ID mismatch')
        reference = yaml.safe_load(args.template.read_text())
        assert reference['model']['random_seed'] == state['seed']
        assert reference['training']['num_epochs'] == 30
        assert reference['training']['batch_size'] == 50000
        assert reference['training']['negative_sampling']['degree_fraction'] == .5
        if shutil.disk_usage(args.work.parent).free < 18 * 1024**3:
            raise RuntimeError('Need 18 GiB for private data and two checkpoints')
        args.work.mkdir(parents=True, exist_ok=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
        env.pop('LD_PRELOAD', None)
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='8',
                   PYTHONUNBUFFERED='1', LD_LIBRARY_PATH=':'.join(str(args.env / x) for x in
                   ('lib/python3.9/site-packages/gege', 'lib/python3.9/site-packages/torch/lib', 'lib')))
        save(status='running', library_sha256=digest(library), driver_sha256=digest(__file__),
             template_sha256=digest(args.template), train_sha256=train_hash, queries_sha256=query_hash,
             evaluator_sha256=digest(args.tools / 'audit_ge2_lj_native_accuracy.py'))

        def run(command, name):
            nonlocal child
            allocation(args.job)
            save(stage=name)
            with (args.results / (name + '.log')).open('w') as log, (args.results / (name + '.hardware.jsonl')).open('w') as hw:
                child = subprocess.Popen(list(map(str, command)), env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    if time.time() > deadline:
                        raise TimeoutError('Allocation deadline reached')
                    info = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                        '--query-gpu=uuid,utilization.gpu,memory.used,power.limit', '--format=csv,noheader'], text=True)
                    jobs = subprocess.check_output(['squeue', '-h', '-w', socket.gethostname().split('.')[0],
                                                     '-o', '%A %u'], text=True)
                    hw.write(json.dumps(dict(time=time.time(), gpu=info, jobs=jobs)) + '\n')
                    hw.flush()
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(name + ' failed with exit ' + str(child.returncode))
                child = None

        signal.signal(signal.SIGTERM, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        for fraction in state['factors']:
            label = 'degree_' + str(fraction).replace('.', '')
            apps = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip()
            if apps:
                raise RuntimeError('Selected GPU is occupied: ' + apps)
            work = args.work / label
            work.mkdir()
            data, model = work / 'data', work / 'model'
            shutil.copytree(args.partitioned, data)
            metadata = yaml.safe_load((data / 'dataset.yaml').read_text())
            metadata['dataset_dir'] = str(data) + '/'
            (data / 'dataset.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
            cfg = copy.deepcopy(reference)
            cfg['training']['negative_sampling']['degree_fraction'] = fraction
            cfg['storage']['dataset']['dataset_dir'] = str(data) + '/'
            cfg['storage']['model_dir'] = str(model) + '/'
            cfg['storage'].pop('checkpoint_dir', None)
            cfg['training']['save_model'] = True
            cfg['evaluation']['epochs_per_eval'] = 31
            cfg['evaluation']['checkpoint_dir'] = ''
            config = args.results / (label + '.yaml')
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            row = dict(degree_fraction=fraction, status='training', checkpoint=str(model),
                       config_sha256=digest(config), config=str(config))
            state['cases'].append(row)
            run([args.env / 'bin/gege_train', config], label + '_train')
            times = [int(t) / 1000 for t in re.findall(r'Epoch Runtime:\s*(\d+)ms',
                         (args.results / (label + '_train.log')).read_text())]
            if len(times) != 30:
                raise RuntimeError('Expected 30 epochs; got ' + str(len(times)))
            row.update(status='evaluating', epoch_times_s=times, average_epoch_s=statistics.mean(times),
                       steady_epoch_s=statistics.mean(times[1:]), checkpoint_sha256=digest(model / 'embeddings.bin'))
            run([args.env / 'bin/python', args.tools / 'audit_ge2_lj_native_accuracy.py',
                 '--data', args.logical, '--embedding', model / 'embeddings.bin', '--legacy-queries', args.queries,
                 '--output', args.results / (label + '_eval')], label + '_eval_driver')
            row.update(status='done', evaluation=json.loads((args.results / (label + '_eval/result.json')).read_text()))
            save()
        save(status='done', stage='complete')
    except BaseException as exc:
        save(status='failed', error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
