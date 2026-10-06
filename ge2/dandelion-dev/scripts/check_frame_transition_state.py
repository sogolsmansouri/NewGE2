#!/usr/bin/env python3
"""Check frame transitions against a separate global sparse-update mirror."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess

import numpy as np
import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'work', 'audit_library', 'deterministic_library', 'order_library'):
        parser.add_argument('--' + name.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=int, choices=(1, 2), default=1)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--order-window', type=int)
    args = parser.parse_args()
    selected_gpus = [args.gpu] if args.gpus == 1 else list(range(args.gpus))
    apps = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, selected_gpus)),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError('Correctness fixture requires idle selected GPUs: ' + apps)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.root
    prefix = root / 'ge2-a6000-cuda121'
    source = root / 'source_mapping_6b0e8be/ge2/dandelion-dev/gege'
    binary = root / 'build_mapping_6b0e8be/gege_train'
    source = args.source or source
    binary = args.binary or binary
    if args.gpus == 2:
        from run_workstation_fb_accuracy_control import execution_flags
    template = yaml.safe_load((root / 'evidence/schedule_order_fixture_20261006/config.yaml').read_text())
    flags = json.loads((root / 'templates/flags.json').read_text())
    flags.update(GEGE_TRAINING_REPLAY_SEED='17', GEGE_TRAINING_INPUT_AUDIT='1',
        GEGE_DIAGNOSTIC_SMALL_FIXTURE='1', GEGE_GLOBAL_DEGREE_SAMPLING='0',
        GEGE_BATCHED_NEGATIVE_PLAN_BATCHES='0', GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES='0',
        GEGE_FIXED_BUFFER_MANUAL_COMPLEX_RNS='1', GEGE_FIXED_BUFFER_MANUAL_DISTMULT_RNS='1',
        GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY='1', GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY_MAX='100000',
        GEGE_PARTITION_BUFFER_LP_FAST_PATH_VALIDATE='1',
        GEGE_PARTITION_BUFFER_LP_FAST_PATH_VALIDATE_MAX='100000')
    if args.gpus == 2:
        flags = execution_flags(flags, 2, 'peer')
    if args.order_window is not None:
        flags['GEGE_STATEFLOW_ORDER_WINDOW'] = str(args.order_window)
    args.work.mkdir(exist_ok=False)
    package = args.work / 'python'
    package.mkdir()
    (package / 'gege').symlink_to(source / 'src/python', target_is_directory=True)
    report = dict(status='running', paper_ready=False, timing_eligible=False,
        scope='Small deterministic storage/update-placement audit; not an independent gradient derivation',
        native_sha256=digest(binary), native_library_sha256=digest(binary.parent / 'libge2.so'),
        audit_library_sha256=digest(args.audit_library), cases={})
    report['gpus'] = args.gpus
    report['order_window'] = args.order_window
    report_path = args.work / 'result.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    cases = [('original_native', False, False, False, False),
             ('original_audit', False, True, False, False),
             ('ordered_native', True, False, False, False),
             ('ordered_audit', True, True, False, False),
             ('original_sync_audit', False, True, True, False),
             ('ordered_sync_audit', True, True, True, False),
             ('injected_mismatch', True, True, False, True)]
    if args.gpus == 2:
        cases = [('peer_native', False, False, False, False),
                 ('peer_audit', False, True, False, False),
                 ('host_audit', False, True, False, False),
                 ('injected_mismatch', False, True, False, True)]
    try:
        for label, reorder, audit, synchronous, inject in cases:
            work = args.work / label
            work.mkdir()
            config = copy.deepcopy(template)
            config['storage']['model_dir'] = str(work / 'model') + '/'
            config['evaluation']['checkpoint_dir'] = str(work / 'model') + '/'
            config['training']['num_epochs'] = 2
            config['training']['logical_active_devices'] = args.gpus
            config['storage']['device_ids'] = list(range(args.gpus))
            path = work / 'config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            current = dict(flags)
            if label == 'host_audit':
                current['GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK'] = '1'
            if synchronous:
                current.update(GEGE_FRAME_CACHE_HIDDEN_FRAMES='0',
                    GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='0',
                    GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='0',
                    GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='0',
                    GEGE_MEM_SWAP_EVENT_SYNC='0', GEGE_SYNC_BEFORE_SWAP='1')
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(('GEGE_', 'PYTHON', 'CONDA', 'OMP_', 'MKL_'))}
            env.pop('LD_PRELOAD', None)
            env.update(current)
            libraries = [str(args.deterministic_library)]
            if args.gpus == 1:
                libraries.insert(0, str(args.order_library))
            if audit:
                libraries.insert(0, str(args.audit_library))
            env.update(PATH=f'{prefix}/bin:/usr/bin:/bin',
                CUDA_VISIBLE_DEVICES=str(args.gpu) if args.gpus == 1 else '0,1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTHONHOME=str(prefix), PYTHONPATH=str(package),
                PYTHONNOUSERSITE='1', GEGE_NO_BINDINGS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                LD_LIBRARY_PATH=f'{binary.parent}:{prefix}/lib/python3.9/site-packages/torch/lib:{prefix}/lib',
                LD_PRELOAD=':'.join(libraries), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                GEGE_DIAGNOSTIC_REORDER=str(int(reorder)),
                GEGE_FRAME_AUDIT_INJECT_CORRUPTION=str(int(inject)))
            (work / 'flags.json').write_text(json.dumps(current, indent=2) + '\n')
            report['active_case'] = label
            save()
            with (work / 'train.log').open('x') as log:
                run = subprocess.run([str(binary), str(path)], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=180)
            text = (work / 'train.log').read_text()
            row = dict(exit_code=run.returncode,
                states_checked=len(re.findall(r'\[frame-audit-state\]', text)),
                host_checks=len(re.findall(r'\[frame-audit-host\]', text)),
                completed_epochs=list(map(int, re.findall(r'Finished training epoch\s+(\d+)', text))),
                mismatch_detected='FRAME_AUDIT_MISMATCH' in text,
                manual_backward_used='[manual_complex_rns] enabled=1' in text)
            if args.gpus == 2 and not inject:
                from run_workstation_fb_accuracy_control import transport_counts
                row['transport'] = transport_counts(text)
            report['cases'][label] = row
            save()
            if inject:
                assert run.returncode != 0 and row['mismatch_detected'], 'Injected corruption was not detected'
            else:
                assert run.returncode == 0 and row['completed_epochs'] == [1, 2], label
                assert row['manual_backward_used'], 'Manual optimized backward did not run'
                if audit:
                    assert row['states_checked'] == 176 and row['host_checks'] >= 4, label
                if args.gpus == 2:
                    movement = row['transport']
                    assert movement['descriptor_mismatch_count'] == 0, label
                    if label == 'host_audit':
                        assert movement['peer_bytes_executed'] == 0 and movement['host_fallback_bytes'] > 0, label
                    else:
                        assert movement['peer_bytes_executed'] > 0 and movement['host_fallback_bytes'] == 0, label
        report['instrumentation_equivalence'] = {}
        for family in ('original', 'ordered') if args.gpus == 1 else ('peer',):
            errors = {}
            for filename in ('embeddings.bin', 'embeddings_state.bin'):
                a = np.fromfile(args.work / (family + '_native') / 'model' / filename, dtype='<f4')
                b = np.fromfile(args.work / (family + '_audit') / 'model' / filename, dtype='<f4')
                assert a.shape == b.shape and np.array_equal(a, b), family + '/' + filename
                errors[filename] = dict(bitwise_equal=True, max_abs_error=0.0)
            report['instrumentation_equivalence'][family] = errors
        if args.gpus == 2:
            report['peer_host_equivalence'] = {}
            for filename in ('embeddings.bin', 'embeddings_state.bin'):
                a = np.fromfile(args.work / 'peer_audit/model' / filename, dtype='<f4')
                b = np.fromfile(args.work / 'host_audit/model' / filename, dtype='<f4')
                assert a.shape == b.shape and np.array_equal(a, b), 'peer/host/' + filename
                report['peer_host_equivalence'][filename] = dict(bitwise_equal=True, max_abs_error=0.0)
        report.update(status='passed', active_case=None)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
