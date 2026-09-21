"""Launch frozen FB accuracy controls on c30 after checking GPU availability."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from arc_accuracy_gpu_guard import foreign_gpu_pids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--model', choices=('distmult', 'complex'), required=True)
    parser.add_argument('--study', choices=('initialization', 'seeds'), required=True)
    args = parser.parse_args()
    job = os.environ['SLURM_JOB_ID']
    host = os.uname().nodename.split('.')[0]
    alloc = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=20)
    if (host != 'c30' or 'JobState=RUNNING ' not in alloc
            or 'UserId='+os.environ['USER']+'(' not in alloc
            or re.search(r'\bNodeList=(\S+)', alloc)[1] != host):
        raise RuntimeError('Owned running c30 allocation required')
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)', alloc)[1]).timestamp()-180
    manifest = json.loads((args.base/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((args.base/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen campaign input changed: '+name)
    work = Path('/mnt/local/smansou2')/('ge2_fb_accuracy_'+job)
    results = Path.home()/'arc_results/runs'/('ge2_fb_accuracy_'+job)
    results.mkdir(parents=True, exist_ok=False)
    state = dict(job=job, host=host, model=args.model, study=args.study, commit=manifest['commit'],
                 status='waiting_for_idle_gpu', gpu=0, paper_timing_eligible=False)
    minimum = 9000 if args.study == 'seeds' else 4500
    while True:
        foreign = foreign_gpu_pids(0)
        state.update(foreign_gpu_pids=foreign, updated=datetime.datetime.now().isoformat())
        (results/'launch.json').write_text(json.dumps(state, indent=2)+'\n')
        if deadline-time.time() < minimum:
            raise RuntimeError('Insufficient allocation time remaining for this control')
        if not foreign:
            break
        time.sleep(15)
    runtime = work/'runtime'
    runtime.mkdir(parents=True, exist_ok=False)
    for name in ('scripts', 'tools', 'reference'):
        shutil.copytree(args.base/name, runtime/name)
    python = Path('/mnt/local/smansou2/ge2-a6000-cuda121/bin/python')
    for test in ('test_zenodo_fb_sampling_control.py', 'test_arc_accuracy_gpu_guard.py'):
        subprocess.run([str(python), str(runtime/'scripts'/test)], check=True, timeout=60)
    state.update(status='launching', control_results=str(results/'control'))
    (results/'launch.json').write_text(json.dumps(state, indent=2)+'\n')
    command = [str(python), str(runtime/'scripts/run_zenodo_fb_sampling_control.py'),
               '--work', str(work/'control'), '--results', str(results/'control'),
               '--env', str(python.parent.parent), '--tools', str(runtime/'tools'),
               '--reference', str(runtime/'reference'/('fb_'+args.model)),
               '--data', manifest['data'], '--job', job, '--gpu', '0',
               '--model', args.model, '--study', args.study]
    os.execv(str(python), command)


if __name__ == '__main__':
    main()
