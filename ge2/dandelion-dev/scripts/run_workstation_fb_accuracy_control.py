#!/usr/bin/env python3
"""Full FB partition controls on an explicitly authorized, idle workstation GPU."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import io
import json
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


NODES, RELATIONS, WIDTH, EDGES = 86054151, 14824, 100, 304727650
QUERY_SHA = 'a4f3bf65bdff5ce735946982f57950f55f64a52470eca1eeff9c41fa01912685'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def control_flags(original, partitions):
    flags = dict(original)
    if (flags.get('GEGE_BASELINE_TRAINING_SEMANTICS') != '1'
            or flags.get('GEGE_SOFTMAX_NEGATIVE_MASS_SCALE') != '1'
            or flags.get('GEGE_BOUNDED_COVER_EPOCH_RELABEL') != '1'):
        raise ValueError('Expected the corrected, unweighted, epoch-relabeled recipe')
    if partitions not in (16, 32):
        raise ValueError('This paired control supports p16 and p32 only')
    hidden = '6' if partitions == 32 else '0'
    flags.update(GEGE_FRAME_CACHE_HIDDEN_FRAMES=hidden,
                 GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='3' if partitions == 32 else '0',
                 GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='1' if partitions == 32 else '0',
                 GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='1' if partitions == 32 else '0',
                 GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='0',
                 GEGE_PARTITION_BUFFER_PEER_RELAY='0', GEGE_STATEFLOW_ALLOW_PEER_RELAY='0',
                 GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME='0',
                 GEGE_TRAINING_INPUT_AUDIT='0', GEGE_TRAINING_PARAMETER_AUDIT='0',
                 GEGE_STATEFLOW_PEER_RELAY_VALIDATE='0',
                 GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE='0', GEGE_PREPARED_BATCH_PIPELINE='0')
    flags.pop('GEGE_TRAINING_REPLAY_SEED', None)
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'prefix', 'data16', 'data32', 'tools', 'template', 'flags', 'gate', 'work'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--workstation-host', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--partitions', type=int, choices=(16, 32), required=True)
    args = parser.parse_args()
    if os.uname().nodename != args.workstation_host:
        raise RuntimeError('Wrong explicitly authorized workstation')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    gate = json.loads(args.gate.read_text())
    hashes = dict(binary_sha256=digest(args.binary), library_sha256=digest(args.binary.parent/'libge2.so'))
    if gate['status'] != 'passed' or any(gate[key] != value for key, value in hashes.items()):
        raise RuntimeError('The native binary/library must match the passing real-P2P gate')
    for name, expected in gate['source_sha256'].items():
        if digest(args.source/name) != expected:
            raise RuntimeError('Source changed after the passing gate: '+name)
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
    lock = Path('/home/'+os.environ['USER'])/('.fb_accuracy_gpu_'+gpu_uuid+'.lock')
    lock_file = lock.open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def apps():
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                          '--format=csv,noheader'], text=True)
        return [row for row in csv.reader(io.StringIO(output), skipinitialspace=True) if row and row[0] == gpu_uuid]

    if apps():
        raise RuntimeError('Selected GPU is busy; no process will be displaced')
    if shutil.disk_usage(args.work.parent).free < 80 * 2**30:
        raise RuntimeError('Need 80 GiB free for this preserved checkpoint')
    args.work.mkdir(exist_ok=False)
    state = dict(status='preflight', host=args.workstation_host, gpu=args.gpu, gpu_uuid=gpu_uuid,
                 partitions=args.partitions, visible_frames=4, hidden_frames=6 if args.partitions == 32 else 0,
                 commit=args.commit, paper_ready=False, timing_status='A5000 diagnostic; not A6000 paper timing',
                 purpose='Fixed-recipe partition/negative-domain control, not hyperparameter selection', **hashes)

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        temporary = args.work/'progress.json.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.work/'progress.json')

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor signal '+str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    update()
    try:
        selected = args.data32 if args.partitions == 32 else args.data16
        metadata = yaml.safe_load((selected/'dataset.yaml').read_text())
        query = args.data16/'exact10000_uniform_v1/edges/test_edges.bin'
        source_audit = json.loads((args.data16/'source_audit.json').read_text())
        partition_audit = json.loads((args.data32/'data_audit.json').read_text())
        if (digest(query) != QUERY_SHA or source_audit['status'] != 'valid_raw_reconstruction'
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
        (args.work/'data_identity.json').write_text(json.dumps(dict(query_sha256=QUERY_SHA, splits=data_hashes), indent=2)+'\n')
        data = args.work/'data'
        data.mkdir()
        for directory in ('edges', 'nodes'):
            (data/directory).symlink_to((selected/directory).resolve(), target_is_directory=True)
        metadata['dataset_dir'] = str(data)+'/'
        (data/'dataset.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
        model = args.work/'model'
        config = yaml.safe_load(args.template.read_text())
        if (config['model']['decoder']['type'] != 'DISTMULT' or config['training']['batch_size'] != 50000
                or config['model']['encoder']['embedding_dim'] != WIDTH):
            raise RuntimeError('Wrong paired DistMult/50K/width100 recipe')
        config['storage']['dataset']['dataset_dir'] = str(data)+'/'
        config['storage']['embeddings']['options'].update(num_partitions=args.partitions, buffer_capacity=4)
        config['storage'].update(device_ids=[0], model_dir=str(model)+'/')
        config['evaluation']['checkpoint_dir'] = str(model)+'/'
        config['training'].update(logical_active_devices=1, num_epochs=10, save_model=True)
        config_path = args.work/'config.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        flags = control_flags(json.loads(args.flags.read_text()), args.partitions)
        (args.work/'flags.json').write_text(json.dumps(flags, indent=2)+'\n')
        package = args.work/'python'
        package.mkdir()
        (package/'gege').symlink_to((args.source/'src/python').resolve(), target_is_directory=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GEGE_', 'PYTHON', 'CONDA', 'SLURM_', 'OMP_', 'MKL_'))}
        env.pop('LD_PRELOAD', None)
        env.update(flags)
        env.update(PATH=str(args.prefix/'bin')+':/usr/bin:/bin',
                   LD_LIBRARY_PATH=f'{args.binary.parent}:{args.prefix}/lib:{args.prefix}/lib/python3.9/site-packages/torch/lib',
                   PYTHONPATH=f'{package}:{args.tools}', GEGE_NO_BINDINGS='1', PYTHONNOUSERSITE='1',
                   CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='12', MKL_NUM_THREADS='12', OPENBLAS_NUM_THREADS='1')

        def command(argv, stage, timeout):
            update(status=stage, command=list(map(str, argv)))
            log = args.work/(stage+'.log')
            with log.open('x') as output:
                child = subprocess.Popen(list(map(str, argv)), env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                update(child_pid=child.pid)
                deadline = time.monotonic()+timeout
                try:
                    while child.poll() is None:
                        if time.monotonic() > deadline:
                            raise TimeoutError(stage)
                        foreign = []
                        for row in apps():
                            try:
                                if os.getpgid(int(row[1])) != child.pid:
                                    foreign.append(row)
                            except ProcessLookupError:
                                pass
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

        command([args.binary, config_path], 'train', 12*3600)
        text = (args.work/'train.log').read_text()
        times = [int(v)/1000 for v in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
        if len(times) != 10:
            raise RuntimeError('Expected ten complete epochs')
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
                 '--ge2-data-dir', args.data16, '--eval-edges', query, '--expected-eval-sha256', QUERY_SHA,
                 '--score', 'distmult', '--report-directions', 'tail', '--filtered', '--tie-policy', 'pessimistic',
                 '--num-test', 10000, '--num-nodes', NODES, '--num-relations', RELATIONS, '--embedding-dim', WIDTH,
                 '--batch-size', 32, '--candidate-chunk', 500000, '--filter-chunk', 2000000, '--device', 'cuda:0',
                 '--evaluator-contract', 'fb_partition_control_20261004', '--score-contract',
                 'ge2_forward_inverse_relation_embeddings', '--out', args.work/'exact_eval.json'], 'evaluate', 12*3600)
        quality = json.loads((args.work/'exact_eval.json').read_text())
        with np.load(args.work/'exact_eval.ranks.npz') as saved:
            ranks = saved['tail_ranks']
        if (quality['eval_edges_sha256'] != QUERY_SHA or quality['report_directions'] != 'tail'
                or not quality['filtered'] or quality['tf32'] is not False or ranks.shape != (10000,)
                or np.any(ranks < 1) or np.any(ranks > NODES)
                or not np.isclose(quality['mrr'], np.mean(1/ranks), atol=1.e-12, rtol=0)
                or not np.isclose(quality['hits_at_10'], np.mean(ranks <= 10), atol=1.e-12, rtol=0)):
            raise RuntimeError('Saved ranks do not reproduce the frozen tail-only evaluator contract')
        update(status='done', epoch_times_s=times, epochs_completed=10, mrr=quality['mrr'],
               hits_at_10=quality['hits_at_10'], average_epoch_s=float(np.mean(times)),
               steady_epoch_s=float(np.mean(times[1:])), checkpoint=str(model))
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
