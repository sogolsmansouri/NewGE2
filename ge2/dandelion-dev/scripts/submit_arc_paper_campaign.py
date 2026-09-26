#!/usr/bin/env python3
"""Submit one preparation gate and twelve serial measurements; persist job IDs."""
import argparse
import json
from pathlib import Path
import re
import subprocess


CASES = ('lj_dot', 'fb_complex', 'tw_dot', 'fb_distmult', 'wk_distmult', 'wk_complex')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--ledger', required=True, type=Path)
    p.add_argument('--batch-script', required=True, type=Path)
    args = p.parse_args()
    if args.ledger.exists():
        raise ValueError('Queue ledger exists; inspect it instead of submitting duplicates')
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    def save():
        tmp = args.ledger.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(base=args.base, jobs=jobs), indent=2)+'\n')
        tmp.replace(args.ledger)
    order = [('prepare','all')] + [(system,case) for case in CASES for system in ('ge2','pipege')]
    previous = gate = None
    for system, case in order:
        cmd = ['sbatch','--parsable','--hold','--nodelist=c30',
               '--job-name=p300_'+system+'_'+case,
               '--output='+str(args.ledger.parent/'job_%j.out'),
               '--error='+str(args.ledger.parent/'job_%j.err')]
        dependency = None
        if previous:
            dependency = 'afterok:'+gate
            if previous != gate:
                dependency += ',afterany:'+previous
            cmd += ['--dependency='+dependency]
        cmd += [str(args.batch_script),args.base,system,case]
        output = subprocess.check_output(cmd, text=True)
        match = re.search(r'^([0-9]+)(?:;[^\n]+)?$', output, re.M)
        if not match:
            raise RuntimeError('Could not parse sbatch result; inspect queue before retry: '+output)
        job = match[1]
        jobs.append(dict(job=job, system=system, case=case, dependency=dependency, released=False))
        save()
        gate = gate or job
        previous = job
    # Partial submission leaves held jobs and an explicit ledger, never a hidden retry.
    for item in jobs:
        subprocess.run(['scontrol','release',item['job']], check=True)
        item['released'] = True
        save()
    print(args.ledger.read_text())


if __name__ == '__main__':
    main()
