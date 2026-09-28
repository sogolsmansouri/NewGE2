#!/usr/bin/env python3
"""Freeze a committed local build for the PipeGE-only ARC relabeling control."""
import argparse
from pathlib import Path
import shutil
import subprocess

from arc_job_support import write_json
from run_arc_paper_case import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'build', 'out', 'patchelf'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', '-C', str(args.repo), 'diff', 'HEAD']):
        raise RuntimeError('Commit tracked source changes before packaging')
    commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'build').mkdir()
    scripts = args.repo/'ge2/dandelion-dev/scripts'
    for name in ('run_arc_fb_relabel_control.py', 'arc_job_support.py', 'run_arc_paper_case.py'):
        shutil.copy2(scripts/name, args.out/name)
    before = {}
    binaries = {}
    for name in ('gege_train', 'libge2.so', 'gege_manual_training_update_test',
                 'gege_manual_backward_test', 'gege_stateflow_validator_tests'):
        before[name] = digest(args.build/name)
        target = args.out/'build'/name
        shutil.copy2(args.build/name, target)
        subprocess.run([str(args.patchelf), '--set-rpath',
            '$ORIGIN:/mnt/local/smansou2/ge2-a6000-cuda121/lib:'
            '/mnt/local/smansou2/ge2-a6000-cuda121/lib/python3.9/site-packages/torch/lib', str(target)], check=True)
        binaries[name] = digest(target)
    subprocess.run(['git', '-C', str(args.repo), 'bundle', 'create',
                    str(args.out.resolve()/'source.bundle'), 'HEAD'], check=True)
    files = {str(p.relative_to(args.out)): digest(p) for p in args.out.rglob('*') if p.is_file()}
    tree = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse',
                                    'HEAD:ge2/dandelion-dev/gege'], text=True).strip()
    write_json(args.out/'build.json', dict(source_commit=commit, engine_tree=tree,
        original_binaries=before, binaries=binaries, relocation='RUNPATH only', payload=files))
    print(commit)


if __name__ == '__main__':
    main()
