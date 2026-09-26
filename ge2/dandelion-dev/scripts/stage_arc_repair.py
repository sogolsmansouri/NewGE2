#!/usr/bin/env python3
"""Stage a new repair cohort without modifying a completed campaign."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from arc_job_support import save_failure_evidence
from run_arc_paper_case import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-base', 'base', 'payload', 'summary', 'archive'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.uname().nodename.split('.')[0] != 'c30':
        raise RuntimeError('Stage only inside the c30 batch allocation')
    if args.base.exists():
        raise ValueError('Use a new base; never overwrite a frozen campaign')
    payload = json.loads((args.payload/'payload.json').read_text())
    previous = json.loads((args.old_base/'campaign.json').read_text())
    manifest = json.loads((args.old_base/'manifest.json').read_text())
    for root, entries in ((args.old_base, previous['files']), (args.payload, payload['files'])):
        for rel, expected in entries.items():
            path = Path(rel)
            if path.is_absolute() or '..' in path.parts or digest(root/path) != expected:
                raise ValueError('Frozen deployment file mismatch: '+rel)
    for name in ('290696_pipege_lj_dot', '290700_pipege_tw_dot', '290703_ge2_wk_distmult'):
        source = args.old_base/'results'/name
        if not source.is_dir():
            raise RuntimeError('Missing original failure evidence: '+str(source))
        save_failure_evidence(source, args.summary/'prior_failures'/name)
    args.base.mkdir(parents=True)
    for folder in ('scripts', 'harness'):
        shutil.copytree(args.old_base/folder, args.base/folder, ignore=shutil.ignore_patterns('__pycache__'))
    for rel in payload['files']:
        target = args.base/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.payload/rel, target)
    subprocess.run([previous['env']+'/bin/python', str(args.base/'scripts/stage_arc_paper_campaign.py'),
        '--base', str(args.base), '--old-base', str(args.old_base), '--old-work', str(args.old_base/'work'),
        '--engine', manifest['engine']['root'], '--archive', str(args.archive), '--summary', str(args.summary),
        '--ge2-zip', str(args.old_base/'ge2.zip'), '--commit', payload['commit']], check=True)


if __name__ == '__main__':
    main()
