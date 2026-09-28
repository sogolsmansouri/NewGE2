#!/usr/bin/env python3
"""Run only PipeGE FB controls, reusing audited ARC data and frozen evaluation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from arc_job_support import write_json
from run_arc_paper_case import archive_checkpoint, digest, verify_evaluation_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    job = os.environ['SLURM_JOB_ID']
    host = os.uname().nodename.split('.')[0]
    allocation = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    if (host != 'c30' or 'JobState=RUNNING ' not in allocation or 'NodeList=c30 ' not in allocation
            or 'UserId='+os.environ['USER']+'(' not in allocation):
        raise RuntimeError('Requires an owned running c30 allocation')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', allocation)[1]).timestamp()-150
    base = Path('/mnt/local/smansou2')/('pipege_fb_relabel_'+job)
    base.mkdir(exist_ok=False)
    results = Path('/home/smansou2/arc_results/runs')/base.name
    results.mkdir(exist_ok=False)
    state = dict(status='preparing', job=job, commit=args.commit, models=['complex', 'distmult'],
                 timing_status='shared_node_provisional', paper_ready=False)

    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(results/'campaign_status.json', state)

    def command(argv, name, env=None):
        import time
        update(stage=name)
        remaining = deadline-time.time()
        if remaining < 120:
            raise RuntimeError('Allocation deadline reached')
        with (results/(name+'.log')).open('x') as output:
            subprocess.run(list(map(str, argv)), env=env, stdout=output, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=remaining, check=True)

    try:
        update()
        old_base = Path('/mnt/local/smansou2/paper_matched_300w_20260925')
        old = json.loads((old_base/'manifest.json').read_text())
        prepared_path = old_base/'work/prepared.json'
        prepared = json.loads(prepared_path.read_text())
        if prepared.get('status') != 'ready':
            raise RuntimeError('Original dataset preparation incomplete')
        build_info = json.loads((args.payload/'build.json').read_text())
        if build_info['source_commit'] != args.commit:
            raise RuntimeError('Build/source mismatch')
        for rel, expected in build_info['payload'].items():
            if digest(args.payload/rel) != expected:
                raise RuntimeError('Payload changed: '+rel)
        if shutil.disk_usage(base).free < 300 * 2**30:
            raise RuntimeError('Need 300 GiB for both new controls and their correctness gates')
        for directory in ('harness', 'references'):
            for source in (old_base/directory).rglob('*'):
                if not source.is_file() or '__pycache__' in source.parts:
                    continue
                relative = str(source.relative_to(old_base))
                if digest(source) != {**old['references'], **old['helpers']}[relative]:
                    raise RuntimeError('Original frozen helper changed: '+relative)
                target = base/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        shutil.copy2(args.payload/'source.bundle', base/'source.bundle')
        work = base/'work'
        engine = work/'engine'
        engine.mkdir(parents=True)
        command(['git', 'clone', base/'source.bundle', engine/'repo'], 'clone')
        command(['git', '-C', engine/'repo', 'checkout', '--detach', args.commit], 'checkout')
        scripts = engine/'repo/ge2/dandelion-dev/scripts'
        (base/'scripts').mkdir()
        for name in ('run_arc_pipege_best.py', 'arc_job_support.py', 'arc_accuracy_gpu_guard.py'):
            shutil.copy2(scripts/name, base/'scripts'/name)
        if digest(Path(__file__)) != digest(scripts/Path(__file__).name):
            raise RuntimeError('Control driver is not from the pinned commit')
        build = engine/'build_git'
        prefix = Path('/mnt/local/smansou2/ge2-a6000-cuda121')
        compiler = prefix/'bin/x86_64-conda-linux-gnu-c++'
        build_env = dict(os.environ, PATH=f'{prefix}/bin:/usr/bin:/bin',
                         CUDA_HOME=str(prefix), CUDA_PATH=str(prefix),
                         CXX=str(compiler), CUDACXX=str(prefix/'bin/nvcc'),
                         LD_LIBRARY_PATH=f'{prefix}/lib:{prefix}/lib/python3.9/site-packages/torch/lib:/usr/lib64')
        build_env.pop('PYTHONPATH', None)
        build_env.pop('PYTHONHOME', None)
        command([prefix/'bin/cmake', '-S', engine/'repo/ge2/dandelion-dev/gege', '-B', build,
                 '-DUSE_CUDA=ON', '-DUSE_OMP=OFF', '-DBUILD_TESTING=ON', '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CUDA_ARCHITECTURES=86', '-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc'),
                 '-DCMAKE_CXX_COMPILER='+str(compiler), '-DCMAKE_CUDA_HOST_COMPILER='+str(compiler),
                 '-DCUDA_TOOLKIT_ROOT_DIR='+str(prefix), '-DPYTHON_EXECUTABLE='+sys.executable,
                 '-DPython3_EXECUTABLE='+sys.executable,
                 f'-DCMAKE_BUILD_RPATH={build};{prefix}/lib;{prefix}/lib/python3.9/site-packages/torch/lib'],
                'configure_arc', build_env)
        command([prefix/'bin/cmake', '--build', build, '--target', 'gege_train',
                 'gege_stateflow_validator_tests', 'gege_manual_training_update_test',
                 'gege_manual_backward_test', '-j', '4'], 'build_arc', build_env)
        build_info['local_verified_binaries'] = build_info['binaries']
        build_info['binaries'] = {name: digest(build/name) for name in build_info['binaries']}
        build_info['deployment_build'] = 'Compiled natively on c30 from the pinned source'
        (engine/'build_git_completed_commit.txt').write_text(args.commit+'\n')
        cases = {}
        for name in ('fb_complex', 'fb_distmult'):
            spec = dict(old['cases'][name], epoch_relabel=True, relabel_seed=17,
                        notes='PipeGE-only epoch-label control; other training settings fixed; provisional timing')
            flags_path = base/spec['flags']
            flags = json.loads(flags_path.read_text())
            if (flags.get('GEGE_BASELINE_TRAINING_SEMANTICS') != '1'
                    or flags.get('GEGE_SOFTMAX_NEGATIVE_MASS_SCALE') != '1'):
                raise RuntimeError('Expected corrected sampling and unweighted objective')
            flags.update(GEGE_BOUNDED_COVER_EPOCH_RELABEL='1', GEGE_BOUNDED_COVER_RELABEL_SEED='17')
            write_json(flags_path, flags)
            cases[name] = spec
        manifest = dict(cases=cases, source_commit=args.commit,
            engine=dict(commit=args.commit, root=str(engine),
                hashes=dict(libge2_so=build_info['binaries']['libge2.so'],
                            gege_train=build_info['binaries']['gege_train']),
                gradient_gate=str(work/'paired_gradient_gate/result.json'),
                gate_template=old['engine']['gate_template']),
            reuse_prepared=dict(path=str(prepared_path), commit=prepared['commit'],
                                manifest_sha256=prepared['manifest_sha256']),
            references={}, helpers={})
        for directory, group in (('references', 'references'), ('harness', 'helpers'), ('scripts', 'helpers')):
            for path in (base/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    manifest[group][str(path.relative_to(base))] = digest(path)
        manifest['helpers']['source.bundle'] = digest(base/'source.bundle')
        write_json(base/'manifest.json', manifest)
        write_json(results/'manifest.json', manifest)
        write_json(results/'build.json', build_info)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        command([build/'gege_stateflow_validator_tests'], 'scheduler_tests', env)
        archive = Path('/mnt/beegfs/smansou2')/base.name
        archive.mkdir(exist_ok=False)
        for case in ('prepare', 'fb_complex', 'fb_distmult'):
            command([sys.executable, base/'scripts/run_arc_pipege_best.py', '--base', base,
                     '--work', work, '--results', results, '--commit', args.commit,
                     '--case', case, '--gpu', '0', '--allow-shared-node'], case, env)
            if case == 'prepare':
                continue
            final = results/case/'final_10e'
            quality = json.loads((final/'exact_eval.json').read_text())
            checkpoint = json.loads((final/'checkpoint_manifest.json').read_text())
            identity = verify_evaluation_artifacts(quality, checkpoint)
            update(stage=case+':archive')
            receipt = archive_checkpoint(final/'checkpoint_manifest.json', archive/case/'checkpoint')
            write_json(final/'archive_receipt.json', dict(receipt, identity=identity))
            shutil.copytree(results/case, archive/case/'evidence')
            update(last_completed=case)
        update(status='done', stage='evaluated_and_archived')
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
