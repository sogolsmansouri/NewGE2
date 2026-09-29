#!/usr/bin/env python3
"""Diagnose the failed GE2 gate and recover saved PipeGE FB evaluation."""
import argparse
import datetime
import json
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
from run_arc_multigpu_campaign import check_replicas, idle_node, native_score_environment
from run_arc_paper_case import digest, verify_evaluation_artifacts
from run_tw_multigpu import check_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Allocation termination: '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    job = os.environ['SLURM_JOB_ID']
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if ('JobState=RUNNING ' not in allocation or 'NodeList=c30 ' not in allocation
            or os.uname().nodename.split('.')[0] != 'c30'
            or 'UserId='+os.environ['USER']+'(' not in allocation):
        raise RuntimeError('Requires owned c30 allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-180
    payload_manifest = json.loads((args.payload/'manifest.json').read_text())
    for name, expected in payload_manifest['files'].items():
        if digest(args.payload/name) != expected:
            raise ValueError('Repair payload changed: '+name)
    base = args.base.resolve()
    if not str(base).startswith('/mnt/local/smansou2/'):
        raise ValueError('Node-local work directory required')
    base.mkdir(exist_ok=False)
    summary = Path('/home/smansou2/arc_results/runs')/base.name
    archive = Path('/mnt/beegfs/smansou2')/base.name
    summary.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    old = Path('/mnt/local/smansou2/paper_multigpu_293571')
    manifest = json.loads((old/'manifest.json').read_text())
    helpers = old/'harness/tools'
    sys.path.insert(0, str(helpers))
    for rel, expected in manifest['files'].items():
        if digest(old/rel) != expected:
            raise ValueError('Original frozen helper changed: '+rel)
    prefix = Path(manifest['env'])
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'PYTHON', 'CONDA', 'CUDA_', 'NCCL_'))}
    env.update(PATH=f'{prefix}/bin:/usr/bin:/bin', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    env = native_score_environment(env, prefix)
    if digest(prefix/'lib/python3.9/site-packages/gege/libge2.so') != manifest['ge2_library_sha256']:
        raise ValueError('Released GE2 binary changed')
    python = prefix/'bin/python'
    state = dict(job=job, status='running', source_campaign=str(old), paper_ready=False,
                 payload_manifest=payload_manifest)
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(summary/'status.json', state)
        write_json(archive/'status.json', state)
    def execute(command, directory, label, process_env, timeout, allow_failure=False):
        idle_node()
        update(stage=label)
        rc = run_logged(list(map(str, command)), process_env, directory/(label+'.log'),
                        min(timeout, deadline-time.time()), directory/(label+'.hardware.jsonl'))
        save_failure_evidence(directory, summary/directory.name)
        if rc and not allow_failure:
            raise RuntimeError(label+' exited '+str(rc))
        return rc
    try:
        update()
        idle_node()
        diag = base/'ge2_diagnostic'
        diag.mkdir()
        spec = manifest['cases']['ge2_fb_complex_2gpu']
        data = diag/'data'
        shutil.copytree(spec['data'], data)
        for rel, expected in spec['data_hashes'].items():
            if digest(data/rel) != expected:
                raise ValueError('Diagnostic data mismatch: '+rel)
        cfg = yaml.safe_load(Path(spec['config']).read_text())
        meta = yaml.safe_load((data/'dataset.yaml').read_text())
        meta['dataset_dir'] = str(data)+'/'
        (data/'dataset.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
        model = diag/'model'
        cfg['storage'].update(dataset=meta, model_dir=str(model)+'/', checkpoint_dir=str(model)+'/')
        cfg['training'].update(num_epochs=1, save_model=False)
        cfg['evaluation']['checkpoint_dir'] = str(model)+'/'
        config = diag/'config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        diag_env = dict(env, CUDA_VISIBLE_DEVICES='0,1', CUDA_LAUNCH_BLOCKING='1', TORCH_SHOW_CPP_STACKTRACES='1')
        write_json(diag/'flags.json', {k:v for k,v in diag_env.items() if k.startswith(('CUDA_', 'NCCL_', 'GEGE_'))})
        rc = execute([prefix/'bin/gege_train', config], diag, 'train_launch_blocking', diag_env, 1800, True)
        update(diagnostic_exit_code=rc)
        eval_dir = base/'pipege_fb_complex_2gpu_eval'
        eval_dir.mkdir()
        spec = manifest['cases']['pipege_fb_complex_2gpu']
        original = old/'results/pipege_fb_complex_2gpu/final'
        model = old/'work/pipege_fb_complex_2gpu/final/model'
        checkpoints = json.loads((original/'checkpoint_manifest.json').read_text())
        write_json(eval_dir/'checkpoint_manifest.json', checkpoints)
        write_json(eval_dir/'replica_check.json', check_replicas(model, 2))
        eval_env = dict(env, CUDA_VISIBLE_DEVICES='0')
        execute([python, '-c', 'import gege; print(gege.__file__); '
                 'assert hasattr(gege.nn.decoders.edge, "ComplexHadamardOperator")'],
                eval_dir, 'native_binding_gate', eval_env, 120)
        execute([python, helpers/'verify_ge2_native_checkpoint_scores.py', '--run', model,
                 '--eval-edges', spec['query'], '--score', spec['model'], '--nodes', spec['nodes'],
                 '--relations', spec['relations'], '--width', spec['width'], '--out', eval_dir/'native_score.json'],
                eval_dir, 'native_score', eval_env, 300)
        execute([python, helpers/'extract_ge2_relation_embeddings.py', '--model', model/'model.pt_0',
                 '--src-out', eval_dir/'src_relations.bin', '--dst-out', eval_dir/'dst_relations.bin',
                 '--expected-relations', spec['relations'], '--expected-dim', spec['width'],
                 '--report', eval_dir/'relation_extract.json'], eval_dir, 'extract_relations', eval_env, 120)
        execute([python, helpers/'eval_marius_kge_exact10k.py', '--entity-bin', model/'embeddings.bin',
                 '--src-relation-bin', eval_dir/'src_relations.bin', '--dst-relation-bin', eval_dir/'dst_relations.bin',
                 '--score', spec['model'], '--num-nodes', spec['nodes'], '--num-relations', spec['relations'],
                 '--embedding-dim', spec['width'], '--num-test', 10000,
                 '--score-contract', 'ge2_forward_inverse_relation_embeddings', '--report-directions', 'tail',
                 '--eval-edges', spec['query'], '--expected-eval-sha256', spec['eval_sha'],
                 '--ge2-data-dir', spec['source'], '--filtered', '--tie-policy', 'pessimistic', '--device', 'cuda:0',
                 '--batch-size', 128, '--candidate-chunk', 250000, '--out', eval_dir/'exact_eval.json'],
                eval_dir, 'eval', eval_env, 3600)
        quality = json.loads((eval_dir/'exact_eval.json').read_text())
        check_evaluation(quality, spec['eval_sha'])
        write_json(eval_dir/'evaluation_identity.json', verify_evaluation_artifacts(quality, checkpoints))
        result = json.loads((original/'status.json').read_text())
        result.update(status='done_pending_review', stage='evaluation_recovered', paper_ready=False,
                      mrr=quality['mrr'], hits_at_10=quality['hits_at_10'], evaluation_retry=str(eval_dir),
                      original_training=str(original), evaluation_only=True)
        write_json(eval_dir/'result.json', result)
        shutil.copytree(eval_dir, archive/eval_dir.name)
        save_failure_evidence(eval_dir, summary/eval_dir.name)
        update(pipege_eval='passed', mrr=quality['mrr'], hits_at_10=quality['hits_at_10'])
        # A separately verified retry can be staged while the preserved checkpoint is evaluated.
        update(stage='waiting_for_ge2_retry')
        retry_deadline = min(deadline-3600, time.time()+1800)
        while time.time() < retry_deadline:
            ready = args.payload/'RETRY_READY'
            if ready.exists():
                retry = args.payload/'retry_ge2.py'
                if digest(retry) != ready.read_text().strip():
                    raise ValueError('GE2 retry script hash mismatch')
                execute([python, retry, '--base', base, '--deadline', deadline], base,
                        'ge2_retry', env, deadline-time.time())
                update(status='done_pending_review')
                break
            time.sleep(15)
        else:
            update(status='evaluation_done_retry_pending')
    except BaseException as error:
        update(status='failed', error=repr(error))
        save_failure_evidence(base, summary/'failure')
        raise


if __name__ == '__main__':
    main()
