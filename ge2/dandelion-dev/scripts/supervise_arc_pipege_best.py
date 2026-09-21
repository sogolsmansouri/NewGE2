#!/usr/bin/env python3
"""Allocation-bound serial campaign; later batch jobs skip verified completed cells."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base','work','results'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--only')
    selection.add_argument('--cases', nargs='+')
    parser.add_argument('--reuse-completed-commit', action='append', default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(args.base/'harness/tools'))
    from run_arc_ge2_allocated_queue import run_logged, write_json
    from prepare_ge2_partitioned_view import sha256_file
    from run_arc_pipege_best import evaluation_check
    job = os.environ['SLURM_JOB_ID']
    args.results.mkdir(parents=True, exist_ok=True)
    lock = (args.results/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = args.results/('attempt_'+job+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    attempt.mkdir(exist_ok=False)
    state = dict(job=job, commit=args.commit, status='starting', completed=[], failed=[])
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(attempt/'supervisor.json', state)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('Signal '+str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    manifest = json.loads((args.base/'manifest.json').read_text())
    alloc = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    deadline = datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)',alloc)[1]).timestamp()-180
    command = [sys.executable, str(args.base/'scripts/run_arc_pipege_best.py'),
               '--base',str(args.base),'--work',str(args.work),'--results',str(attempt),'--commit',args.commit]
    def run(case):
        update(status='running', case=case)
        return run_logged(command+['--case',case],dict(os.environ),attempt/(case+'.supervisor.log'),deadline-time.time())
    try:
        prepared_path = args.work/'prepared.json'
        if not prepared_path.exists():
            if run('prepare'):
                raise RuntimeError('Shared preparation failed; training blocked')
        prepared = json.loads(prepared_path.read_text())
        if (prepared['commit'] != args.commit or prepared['status'] != 'ready'
                or prepared['manifest_sha256'] != sha256_file(args.base/'manifest.json')):
            raise RuntimeError('Preparation mismatch')
        cases = ['lj_dot','tw_dot','fb_distmult','fb_complex','wk_distmult','wk_complex']
        if args.only:
            if args.only not in cases:
                raise ValueError('Unknown case')
            cases = [args.only]
        elif args.cases:
            if len(set(args.cases)) != len(args.cases) or any(case not in cases for case in args.cases):
                raise ValueError('Unknown or duplicate case')
            cases = args.cases
        for case in cases:
            done = False
            for status in args.results.glob('attempt_*/'+case+'/status.json'):
                value = json.loads(status.read_text())
                if value.get('status') == 'done' and value.get('commit') in [args.commit]+args.reuse_completed_commit:
                    result = Path(value['final_result'])
                    provenance = json.loads((status.parent/'provenance.json').read_text())
                    current_tree = subprocess.check_output(['git','-C',str(args.work/'repo'),'rev-parse',
                        args.commit+':ge2/dandelion-dev/gege'],text=True).strip()
                    if (provenance['manifest_sha256'] != sha256_file(args.base/'manifest.json')
                            or provenance['engine_tree'] != current_tree):
                        raise RuntimeError('Completed result has different inputs or engine')
                    row = json.loads(result.read_text())
                    if row['status'] == 'done' and row['train_status'] == 0 and row['exact_eval_status'] == 0:
                        evaluation_check(json.loads((result.parent/'exact_eval.json').read_text()), manifest['cases'][case])
                        state['completed'].append(dict(case=case, result=str(result), previous=True))
                        done = True
                        break
            if done:
                update()
                continue
            # Leave a full case to its queued six-hour continuation rather than
            # start an expensive run just before the interactive lease expires.
            minimum = 1200 if case == 'lj_dot' else 5400
            if deadline-time.time() < minimum:
                update(status='deferred', case=case)
                return
            rc = run(case)
            state['failed' if rc else 'completed'].append(dict(case=case, exit_code=rc,
                status_file=str(attempt/case/'status.json')))
        update(status='partial_failure' if state['failed'] else 'done')
        if state['failed']:
            raise SystemExit(1)
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
