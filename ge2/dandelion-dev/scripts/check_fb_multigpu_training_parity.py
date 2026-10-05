#!/usr/bin/env python3
"""Compare manual/autograd FB-shaped training with the production two-GPU runtime."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import torch
import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_comparison(left, right):
    if left.shape != right.shape:
        raise ValueError('Tensor shapes differ')
    return dict(equal=bool(torch.equal(left, right)),
                max_abs_error=float((left - right).abs().max()),
                finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all()))


def comparison_passed(row):
    if 'equal' in row:
        return row['equal'] and row['finite']
    return bool(row) and all(comparison_passed(value) for value in row.values())


def relay_validation_counts(text):
    checks = re.findall(r'\[stateflow-peer-validate \d+\][^\n]*', text)
    return dict(peer_checks=len(checks),
                negative_handoff_key_checks=sum(bool(re.search(r'pending_key=-\d+', line)) for line in checks),
                validation_mismatch_lines=sum(bool(re.search(r'(?:dst|src)_mismatch_values=[1-9]', line))
                                              for line in checks))


def compare_models(left, right, gpus):
    output = {}
    filenames = [f'model.pt_{lane}' for lane in range(gpus)] + ['model_state.pt']
    for filename in filenames:
        a = torch.jit.load(str(left/filename), map_location='cpu').state_dict()
        b = torch.jit.load(str(right/filename), map_location='cpu').state_dict()
        if not a or a.keys() != b.keys():
            raise ValueError('Missing or mismatched checkpoint tensors: '+filename)
        output[filename] = {key: tensor_comparison(a[key], b[key]) for key in a}
    for filename in ('embeddings.bin', 'embeddings_state.bin'):
        a = torch.from_numpy(np.fromfile(left/filename, dtype='<f4').copy())
        b = torch.from_numpy(np.fromfile(right/filename, dtype='<f4').copy())
        output[filename] = tensor_comparison(a, b)
    return output


def validate_execution_scope(job, workstation_host):
    host = os.uname().nodename
    if workstation_host is not None:
        if host != workstation_host:
            raise RuntimeError('The explicitly authorized workstation host does not match this node')
        return
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if ('JobState=RUNNING ' not in allocation or 'NodeList='+host.split('.')[0]+' ' not in allocation
            or 'UserId='+os.environ['USER']+'(' not in allocation):
        raise RuntimeError('A running owned allocation on this node is required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'prefix', 'template', 'flags', 'data', 'work'):
        parser.add_argument('--'+name, required=True, type=Path)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument('--job')
    execution.add_argument('--workstation-host', help='Exact hostname explicitly authorized for this test')
    parser.add_argument('--transport', choices=('host', 'peer'), default='peer')
    parser.add_argument('--peer-scratch', choices=('shared', 'independent'), default='independent')
    parser.add_argument('--gpus', type=int, choices=(1, 2), default=2)
    parser.add_argument('--visible', type=int, choices=(4, 8), default=4)
    parser.add_argument('--decoder', choices=('DISTMULT', 'COMPLEX', 'both'), default='both')
    parser.add_argument('--parameter-audit', action='store_true')
    parser.add_argument('--prepared-batches', action='store_true')
    parser.add_argument('--observe-validation-mismatches', action='store_true',
                        help='Complete diagnostic checkpoints despite relay mismatches; final gate still fails')
    args = parser.parse_args()
    if args.visible != 4 and args.gpus != 1:
        parser.error('Non-q4 controls currently support one GPU only')
    validate_execution_scope(args.job, args.workstation_host)
    host = os.uname().nodename
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError('Fixture requires idle GPUs: '+apps)
    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError('The selected GPU count does not match --gpus')
    if args.transport == 'peer' and (args.gpus != 2 or not all(torch.cuda.can_device_access_peer(a, b)
                                            for a, b in ((0, 1), (1, 0)))):
        raise RuntimeError('Peer transport requires actual bidirectional CUDA peer access')
    args.work.mkdir(parents=True, exist_ok=False)
    overlay = args.work/'python'
    overlay.mkdir()
    (overlay/'gege').symlink_to((args.source/'src/python').resolve(), target_is_directory=True)
    data = args.work/'data'
    shutil.copytree(args.data, data)
    metadata = yaml.safe_load((data/'dataset.yaml').read_text())
    metadata['dataset_dir'] = str(data)+'/'
    (data/'dataset.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
    template = yaml.safe_load(args.template.read_text())
    flags = json.loads(args.flags.read_text())
    # Changes only execution mode. Replay keys isolate RNG from host-thread scheduling.
    flags.update(GEGE_TRAINING_REPLAY_SEED='17', GEGE_TRAINING_INPUT_AUDIT='1',
                 GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='1' if args.gpus == 1 else '0',
                 GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='1' if args.gpus == 2 else '0',
                 GEGE_PARTITION_BUFFER_PEER_RELAY='1', GEGE_STATEFLOW_ALLOW_PEER_RELAY='1',
                 GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME='1',
                 GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK='0',
                 GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET='0',
                 GEGE_STATEFLOW_PEER_RELAY_VALIDATE='1',
                 GEGE_STATEFLOW_PEER_RELAY_VALIDATE_MAX_CHECKS='100000',
                 GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE='1' if args.prepared_batches else '0',
                 GEGE_PREPARED_BATCH_PIPELINE='1' if args.prepared_batches else '0')
    if args.gpus > 1:
        flags.update(GEGE_STATEFLOW_PEER_RELAY_INDEPENDENT_SCRATCH='1' if args.peer_scratch == 'independent' else '0',
                     GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET='0' if args.peer_scratch == 'independent' else '1')
    if args.parameter_audit:
        flags['GEGE_TRAINING_PARAMETER_AUDIT'] = '1'
    if args.observe_validation_mismatches:
        flags['GEGE_STATEFLOW_PEER_RELAY_VALIDATE_FAIL_FAST'] = '0'
    if args.transport == 'host':
        # Disable the physical peer copy, not the required handoff dependencies.
        flags.update(GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK='1')
    report = dict(status='running', job=args.job, host=host, paper_ready=False,
                  transport=args.transport, gpus=args.gpus,
                  peer_scratch=args.peer_scratch,
                  visible_frames=args.visible,
                  workstation_host=args.workstation_host, prepared_batches=args.prepared_batches,
                  observe_validation_mismatches=args.observe_validation_mismatches,
                  scope='small correctness fixture, not full-data accuracy or timing',
                  binary_sha256=digest(args.binary), library_sha256=digest(args.binary.parent/'libge2.so'),
                  source_sha256={str(path): digest(args.source/path) for path in
                      map(Path, ('src/cpp/src/engine/trainer.cpp', 'src/cpp/src/storage/storage.cpp',
                                 'src/cpp/include/storage/storage.h'))},
                  cases={}, comparisons={})
    result = args.work/'result.json'

    def save():
        result.write_text(json.dumps(report, indent=2)+'\n')

    try:
        save()
        decoders = ('DISTMULT', 'COMPLEX') if args.decoder == 'both' else (args.decoder,)
        for decoder in decoders:
            for mode in ('autograd', 'manual', 'manual_sync'):
                name = decoder.lower()+'_'+mode
                case = args.work/name
                case.mkdir()
                model = case/'model'
                config = copy.deepcopy(template)
                config['model']['decoder']['type'] = decoder
                config['storage']['dataset']['dataset_dir'] = str(data)+'/'
                config['storage']['device_ids'] = list(range(args.gpus))
                config['storage']['embeddings']['options']['buffer_capacity'] = args.visible
                config['storage']['model_dir'] = str(model)+'/'
                config['evaluation']['checkpoint_dir'] = str(model)+'/'
                config['training'].update(logical_active_devices=args.gpus, num_epochs=2, save_model=True)
                config_path = case/'config.yaml'
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))
                current_flags = dict(flags)
                enabled = '0' if mode == 'autograd' else '1'
                current_flags.update(GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS=enabled,
                                     GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS=enabled)
                if mode == 'manual_sync':
                    current_flags.update(GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='0',
                                         GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='0',
                                         GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='0',
                                         GEGE_FRAME_CACHE_HIDDEN_FRAMES='0',
                                         GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='0',
                                         GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE='0',
                                         GEGE_PREPARED_BATCH_PIPELINE='0',
                                         GEGE_SYNC_BEFORE_SWAP='1')
                (case/'flags.json').write_text(json.dumps(current_flags, indent=2)+'\n')
                env = {k: v for k, v in os.environ.items()
                       if not k.startswith(('GEGE_', 'PYTHON', 'CONDA'))}
                env.pop('LD_PRELOAD', None)
                env.update(current_flags)
                env.update(PATH=str(args.prefix/'bin')+':/usr/bin:/bin',
                           LD_LIBRARY_PATH=f'{args.binary.parent}:{args.prefix}/lib/python3.9/site-packages/torch/lib:{args.prefix}/lib',
                           PYTHONPATH=str(overlay), GEGE_NO_BINDINGS='1',
                           NCCL_P2P_DISABLE='1', NCCL_DEBUG='WARN',
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
                report['active_case'] = name
                save()
                with (case/'train.log').open('w') as log:
                    process = subprocess.run([str(args.binary), str(config_path)], env=env,
                                             stdout=log, stderr=subprocess.STDOUT, timeout=600)
                text = (case/'train.log').read_text()
                trace = sorted(re.findall(r'\[training-input\] (.*)', text))
                relay_counts = relay_validation_counts(text)
                host_bytes = sum(int(value) for group in re.findall(r'host_fallback_bytes=([\d,]+)', text)
                                 for value in group.split(','))
                peer_bytes = sum(int(value) for group in re.findall(r'peer_bytes_executed=([\d,]+)', text)
                                 for value in group.split(','))
                entry = dict(exit_code=process.returncode, input_batches=len(trace),
                             trace_sha256=hashlib.sha256('\n'.join(trace).encode()).hexdigest(),
                             **relay_counts,
                             host_handoff_bytes=host_bytes, peer_bytes=peer_bytes,
                             completed_epochs=list(map(int, re.findall(r'Finished training epoch\s+(\d+)', text))))
                report['cases'][name] = entry
                save()
                if (process.returncode or entry['completed_epochs'] != [1, 2] or not trace
                        or (args.transport == 'peer' and not entry['negative_handoff_key_checks'])
                        or (args.transport == 'host' and args.gpus == 2 and (not host_bytes or peer_bytes))
                        or (entry['validation_mismatch_lines'] and not args.observe_validation_mismatches)):
                    raise RuntimeError('Training/peer gate failed: '+name)
            for reference in ('autograd', 'manual_sync'):
                left = args.work/(decoder.lower()+'_'+reference)
                right = args.work/(decoder.lower()+'_manual')
                if report['cases'][left.name]['trace_sha256'] != report['cases'][right.name]['trace_sha256']:
                    raise RuntimeError('Inputs differ; checkpoints cannot establish update parity')
                comparison = compare_models(left/'model', right/'model', args.gpus)
                report['comparisons'][decoder+'_'+reference] = comparison
                save()
                if not comparison_passed(comparison):
                    raise RuntimeError('Checkpoint mismatch: '+decoder+'_'+reference)
        if args.gpus == 2:
            case = args.work/'unsafe_host_rejection'
            case.mkdir()
            config['storage']['model_dir'] = str(case/'model')+'/'
            config['evaluation']['checkpoint_dir'] = str(case/'model')+'/'
            config_path = case/'config.yaml'
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            env.update(GEGE_STATEFLOW_PEER_RUNTIME='off',
                       GEGE_STATEFLOW_ENABLE_UNVERIFIED_PEER_RELAY_RUNTIME='0')
            with (case/'train.log').open('w') as log:
                process = subprocess.run([str(args.binary), str(config_path)], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=120)
            text = (case/'train.log').read_text()
            rejected = bool(process.returncode and
                            'cross-lane handoffs require coordinated storage transitions' in text)
            report['unsafe_host_rejection'] = dict(passed=rejected, exit_code=process.returncode)
            save()
            if not rejected:
                raise RuntimeError('Unsafe host transport was not rejected')
        if any(case['validation_mismatch_lines'] for case in report['cases'].values()):
            raise RuntimeError('Relay validation mismatches observed; diagnostic is not a passing correctness gate')
        report.update(status='passed', active_case=None)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
