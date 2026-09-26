#!/usr/bin/env python3
"""Run one frozen, isolated ARC measurement and verify its persistent archive."""
import argparse
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
import zipfile

from arc_job_support import run_logged, save_failure_evidence


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def hardware_check(samples, expected_power, gpu_uuid):
    if not samples:
        raise ValueError('Missing hardware samples')
    for row in samples:
        if row.get('probe_errors'):
            raise ValueError('Monitoring gaps: clean timing remeasurement required')
        if row.get('other_jobs') or row.get('other_processes'):
            raise ValueError('Contended node: timing rejected')
        matches = [r.split(',') for r in row['gpu'].splitlines()
                   if len(r.split(',')) >= 3 and r.split(',')[1].strip() == gpu_uuid]
        power = float(matches[0][2].strip().split()[0]) if len(matches) == 1 else float('nan')
        if not math.isfinite(power) or abs(power - expected_power) > .1:
            raise ValueError('Missing GPU or changed power limit')


def verify_evaluation_artifacts(quality, checkpoints):
    import numpy as np
    entity = next(e for e in checkpoint_entries(checkpoints) if e['path'] == 'embeddings.bin')
    path = Path(checkpoints['source'])/'embeddings.bin'
    if 'entity_bin_sha256' in quality:
        if quality['entity_bin_sha256'] != entity['sha256']:
            raise ValueError('Evaluated embedding hash differs from checkpoint')
    elif Path(quality['embedding_file']).resolve() != path.resolve():
        raise ValueError('Evaluated embedding path differs from checkpoint')
    # The Dot evaluator records a path, not a weight hash; bracket it with hashes.
    if digest(path) != entity['sha256']:
        raise ValueError('Checkpoint changed during evaluation')
    ranks_path = Path(quality['ranks_file'])
    if digest(ranks_path) != quality['ranks_sha256']:
        raise ValueError('Saved ranks changed')
    with np.load(ranks_path, allow_pickle=False) as saved:
        ranks = saved['tail_ranks']
    if ranks.shape != (10000,) or not np.all(np.isfinite(ranks) & (ranks >= 1) & (ranks == np.floor(ranks))):
        raise ValueError('Invalid tail ranks')
    for key, value in [('mrr', np.mean(1.0/ranks)), ('hits_at_10', np.mean(ranks <= 10))]:
        if not math.isclose(float(value), quality[key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Metric does not reproduce from saved tail ranks: '+key)
    return dict(embedding_sha256=entity['sha256'], ranks_sha256=quality['ranks_sha256'],
                recomputed_tail_metrics=True)


def checkpoint_entries(manifest):
    entries = manifest.get('files', [])
    if not entries:
        raise ValueError('Empty checkpoint manifest')
    for entry in entries:
        p = Path(entry['path'])
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 1:
            raise ValueError('Unsafe checkpoint entry')
    return entries


def archive_checkpoint(manifest_path, destination):
    manifest = json.loads(manifest_path.read_text())
    entries = checkpoint_entries(manifest)
    source = Path(manifest['source'])
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        src, dst = source/entry['path'], destination/entry['path']
        if src.stat().st_size != entry['bytes']:
            raise ValueError('Incomplete source checkpoint')
        if not (dst.exists() and dst.stat().st_size == entry['bytes']
                and digest(dst) == entry['sha256']):
            temp = dst.with_name(dst.name+'.partial')
            shutil.copyfile(src, temp)
            if temp.stat().st_size != entry['bytes'] or digest(temp) != entry['sha256']:
                raise ValueError('Archive checksum mismatch: '+str(temp))
            temp.replace(dst)
    return dict(status='verified', source=str(source), destination=str(destination),
                manifest_sha256=digest(manifest_path), source_deleted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--system', choices=('prepare', 'pipege', 'ge2'), required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    spec = json.loads((base/'campaign.json').read_text())
    sys.path[:0] = [str(base/'scripts'), str(base/'harness/tools')]
    from run_arc_ge2_allocated_queue import write_json
    from run_arc_pipege_best import evaluation_check, normalize_dataset_metadata
    from run_arc_pipege_quality import checkpoint_manifest, timing_summary
    import yaml

    job = os.environ['SLURM_JOB_ID']
    host = os.uname().nodename.split('.')[0]
    alloc = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if (host != spec['node'] or 'JobState=RUNNING ' not in alloc
            or 'UserId='+os.environ['USER']+'(' not in alloc
            or re.search(r'\bNodeList=(\S+)', alloc)[1] != host):
        raise RuntimeError('Not inside the expected owned allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', alloc)[1]).timestamp()-180
    lock = (base/'serial.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    results = base/'results'/f'{job}_{args.system}_{args.case}'
    results.mkdir(parents=True, exist_ok=False)
    summary = Path(spec['summary'])/results.name
    summary.mkdir(parents=True, exist_ok=False)
    archive = Path(spec['archive'])/results.name
    state = dict(job=job, host=host, system=args.system, case=args.case,
                 status='preflight', checkpoint_durable=False, results=str(results),
                 expected_power_w=spec['power_w'], commit=spec['commit'])
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(results/'status.json', state)
        write_json(summary/'status.json', state)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Allocation signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    update()
    envdir = Path(spec['env'])
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
    env.pop('LD_PRELOAD', None)
    env.update(PATH=f'{envdir}/bin:/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               LD_LIBRARY_PATH=f'{envdir}/lib/python3.9/site-packages/gege:{envdir}/lib/python3.9/site-packages/torch/lib:{envdir}/lib',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    python = envdir/'bin/python'
    def run(command, name, monitor=False):
        update(stage=name, status='running')
        if deadline-time.time() < 60:
            raise RuntimeError('Insufficient allocation time')
        rc = run_logged(list(map(str, command)), env, results/(name+'.log'), deadline-time.time(),
                        results/(name+'.hardware.jsonl') if monitor else None)
        if rc:
            raise RuntimeError(f'{name} exited {rc}')
    try:
        for rel, expected in spec['files'].items():
            if digest(base/rel) != expected:
                raise ValueError('Frozen input changed: '+rel)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        jobs = subprocess.check_output(['squeue', '-h', '-t', 'RUNNING,COMPLETING', '-w', host, '-o', '%A'], text=True).split()
        if apps or any(j != job for j in jobs):
            raise RuntimeError('Whole idle node required; refusing contended timing')
        gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid,power.limit', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        gpu_uuid, power = gpu[0].strip(), float(gpu[1])
        if abs(power-spec['power_w']) > .1:
            raise ValueError('Wrong power cohort')
        # Verify storage before long training; all execution remains node-local.
        archive.mkdir(parents=True, exist_ok=False)
        probe = archive/'storage_probe.json'
        write_json(probe, dict(job=job, purpose='archive_write_read_probe'))
        if json.loads(probe.read_text())['job'] != job:
            raise RuntimeError('Archive storage readback failed')
        manifest = json.loads((base/'manifest.json').read_text())
        work = base/'work'
        if args.system in ('prepare', 'pipege'):
            case = 'prepare' if args.system == 'prepare' else args.case
            run([python, base/'scripts/run_arc_pipege_best.py', '--base', base, '--work', work,
                 '--results', results, '--commit', spec['commit'], '--case', case], 'driver')
            if args.system == 'prepare':
                shutil.copytree(results, archive/'evidence')
                update(status='done', stage='prepared')
                return
            cell = manifest['cases'][case]
            final = results/case/f"final_{cell['epochs']}e"
            value = json.loads((final/'result.json').read_text())
            quality = json.loads((final/'exact_eval.json').read_text())
            evaluation_check(quality, cell)
            hardware_check([json.loads(s) for s in (final/'train.hardware.jsonl').read_text().splitlines()],
                           spec['power_w'], gpu_uuid)
            ckpt = final/'checkpoint_manifest.json'
            identity = verify_evaluation_artifacts(quality, json.loads(ckpt.read_text()))
        else:
            from run_arc_ge2_dot_reproduction import make_config, LIB_SHA
            from run_arc_ge2_kge_final import case_config
            cell = manifest['cases'][args.case]
            library = envdir/'lib/python3.9/site-packages/gege/libge2.so'
            if digest(library) != LIB_SHA:
                raise ValueError('Original GE2 library changed')
            source_zip = base/'ge2.zip'
            if hashlib.md5(source_zip.read_bytes()).hexdigest() != '6de3d9702241a0c822971939752d0834':
                raise ValueError('Original GE2 archive changed')
            prepared = json.loads((work/'prepared.json').read_text())
            if prepared['commit'] != spec['commit'] or prepared['manifest_sha256'] != digest(base/'manifest.json'):
                raise ValueError('Prepared data identity mismatch')
            audit = prepared['data'][cell['graph']]
            for split, info in audit['splits'].items():
                if digest(Path(cell['source'])/'edges'/f'{split}_edges.bin') != info['sha256']:
                    raise ValueError('Canonical GE2 input changed')
            if digest(Path(cell['query'])) != cell['eval_sha']:
                raise ValueError('GE2 evaluation query changed')
            private = base/'ge2_work'/job
            private.mkdir(parents=True, exist_ok=False)
            if shutil.disk_usage(private).free < 150*1024**3:
                raise RuntimeError('Need 150 GiB free for private data/checkpoint')
            data, model = private/'data', private/'model'
            run([python, base/'harness/tools/prepare_ge2_partitioned_view.py',
                 '--source-data-dir', cell['source'], '--output-dir', data,
                 '--num-partitions', '16', '--edge-columns', cell['columns']], 'partition')
            normalize_dataset_metadata(data)
            template_name = {'lj':'livejournal_16p.yaml', 'tw':'twitter_16p.yaml'}.get(cell['graph'], 'fb15k_16p.yaml')
            with zipfile.ZipFile(source_zip) as z:
                template = yaml.safe_load(z.read('dandelion-dev/gege/configs/'+template_name))
            seed = yaml.safe_load((base/cell['config']).read_text())['model']['random_seed']
            if cell['model'] == 'dot':
                cfg = make_config(template, data, model, cell['epochs'], seed)
            else:
                cfg = case_config(template, data, model, cell['graph'].upper(), cell['model'], seed)
            cfg_path = results/'config.yaml'
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            run([python, '-c', 'import gege,torch; assert torch.cuda.is_available(); print(gege.__file__)',], 'runtime_gate')
            run([python, '-c', 'import sys; from pathlib import Path; from gege.tools.configuration.gege_config import load_config; c=load_config(sys.argv[1],save=False); assert Path(c.storage.dataset.dataset_dir).resolve()==Path(sys.argv[2]).resolve(); assert c.training.batch_size==50000; print(c)', cfg_path, data], 'resolved_config')
            run([envdir/'bin/gege_train', cfg_path], 'train', monitor=True)
            text = (results/'train.log').read_text()
            times = [int(v)/1000 for v in re.findall(r'Epoch Runtime:\s*(\d+)ms', text)]
            if len(times) != cell['epochs'] or re.findall(r'Finished training epoch\s+(\d+)', text) != [str(i) for i in range(1,cell['epochs']+1)]:
                raise ValueError('Incomplete GE2 training')
            if re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%', text) != [(str(cell['edges']),str(cell['edges']))]*cell['epochs']:
                raise ValueError('Incorrect GE2 positive-edge workload')
            samples = [json.loads(s) for s in (results/'train.hardware.jsonl').read_text().splitlines()]
            hardware_check(samples, spec['power_w'], gpu_uuid)
            ckpt = results/'checkpoint_manifest.json'
            checkpoints = checkpoint_manifest(model, cell['nodes'], cell['width'], cell['relations'])
            write_json(ckpt, checkpoints)
            tools = base/'harness/tools'
            if cell['model'] == 'dot':
                command = [python, tools/'stream_marius_dot_exact_eval.py', '--run-dir', results,
                    '--embedding-file', model/'embeddings.bin', '--num-nodes', cell['nodes'],
                    '--dim', cell['width'], '--eval-edge-columns', 2, '--filter-edge-columns', 2,
                    '--expected-num-eval-edges', 10000]
            else:
                run([python, tools/'verify_ge2_native_checkpoint_scores.py', '--run', model,
                    '--eval-edges', cell['query'], '--score', cell['model'], '--nodes', cell['nodes'],
                    '--relations', cell['relations'], '--width', cell['width'], '--out', results/'native_score.json'], 'native_score')
                run([python, tools/'extract_ge2_relation_embeddings.py', '--model', model/'model.pt_0',
                    '--src-out', model/'src_relations.bin', '--dst-out', model/'dst_relations.bin',
                    '--expected-relations', cell['relations'], '--expected-dim', cell['width'],
                    '--report', results/'relation_extract.json'], 'extract')
                command = [python, tools/'eval_marius_kge_exact10k.py', '--entity-bin', model/'embeddings.bin',
                    '--src-relation-bin', model/'src_relations.bin', '--dst-relation-bin', model/'dst_relations.bin',
                    '--score', cell['model'], '--num-nodes', cell['nodes'], '--num-relations', cell['relations'],
                    '--embedding-dim', cell['width'], '--num-test', 10000,
                    '--score-contract', 'ge2_forward_inverse_relation_embeddings']
            command += ['--report-directions', 'tail', '--eval-edges', cell['query'], '--expected-eval-sha256', cell['eval_sha'],
                '--ge2-data-dir', cell['source'], '--filtered', '--tie-policy', 'pessimistic', '--device', 'cuda:0',
                '--batch-size', 128, '--candidate-chunk', 250000, '--out', results/'exact_eval.json']
            run(command, 'eval')
            quality = json.loads((results/'exact_eval.json').read_text())
            evaluation_check(quality, cell)
            identity = verify_evaluation_artifacts(quality, checkpoints)
            value = dict(status='done', system='ge2', train_status=0, exact_eval_status=0,
                         **timing_summary(text, times), mrr=quality['mrr'], hits_at_10=quality['hits_at_10'],
                         report_directions='tail', source_archive_sha256=digest(source_zip), library_sha256=LIB_SHA)
            final = results
            write_json(results/'result.json', value)
        write_json(results/'evaluation_identity.json', identity)
        update(status='archiving', stage='verify_checkpoint_archive', timing_eligible=True)
        receipt = archive_checkpoint(ckpt, archive/'checkpoint')
        write_json(results/'archive_receipt.json', receipt)
        shutil.copytree(results, archive/'evidence')
        qualified = dict(value, timing_eligible=True, hardware=host, power_limit_watts=spec['power_w'],
                         checkpoint_durable=True, archive_receipt=receipt,
                         publication_scope='single_run_native_epoch_tail_filtered; not equal-quality or repeated-run claim')
        write_json(summary/'result.json', qualified)
        for path in final.iterdir():
            if path.is_file() and path.suffix in ('.json', '.yaml', '.npz') and path.stat().st_size < 2*1024**2:
                shutil.copyfile(path, summary/('raw_'+path.name))
        write_json(archive/'qualified_result.json', qualified)
        update(status='done', stage='archived', checkpoint_durable=True, result=str(summary/'result.json'))
    except BaseException as error:
        update(status='failed', error=repr(error))
        try:
            evidence = save_failure_evidence(results, summary/'failure_evidence')
            update(failure_evidence=str(summary/'failure_evidence'),
                   evidence_errors=evidence['errors'])
        except Exception as evidence_error:
            print('Could not preserve failure evidence: '+repr(evidence_error), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
