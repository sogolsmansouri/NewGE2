#!/usr/bin/env python3
"""Retry the failed GE2 FB cell, gated and labelled with its trainer correction."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).with_name('retry_support')))

from arc_job_support import save_failure_evidence, write_json
from run_arc_multigpu_campaign import run_case
from run_arc_paper_case import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    old = Path('/mnt/local/smansou2/paper_multigpu_293571')
    base = args.base/'ge2_retry'
    base.mkdir(exist_ok=False)
    manifest = copy.deepcopy(json.loads((old/'manifest.json').read_text()))
    repair = json.loads((args.base/'ge2_dense_build/build.json').read_text())
    if repair['status'] != 'compiled_and_cpu_tested':
        raise ValueError('Correction is not built/tested')
    for directory in ('harness', 'scripts', 'references'):
        shutil.copytree(old/directory, base/directory)
    shutil.copy2(old/'ge2.zip', base/'ge2.zip')
    manifest.update(ge2_dense_repair=repair, recovery_source=str(old),
                    retry_launcher_sha256=digest(Path(__file__)),
                    retry_runner_sha256=digest(Path(__file__).with_name('retry_support')/'run_arc_multigpu_campaign.py'))
    write_json(base/'manifest.json', manifest)
    sys.path.insert(0, str(base/'harness/tools'))
    summary = Path('/home/smansou2/arc_results/runs')/args.base.name/'ge2_retry'
    archive = Path('/mnt/beegfs/smansou2')/args.base.name/'ge2_retry'
    summary.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base/'manifest.json', archive/'manifest.json')
    state = dict(status='running', case='ge2_fb_complex_2gpu', job=os.environ['SLURM_JOB_ID'],
                 paper_ready=False, repair=repair)
    def update(**changes):
        state.update(changes, updated=datetime.datetime.now().isoformat())
        write_json(summary/'status.json', state)
        write_json(archive/'status.json', state)
    try:
        for phase in ('gate', 'final'):
            update(stage=phase)
            run_case(base, manifest, state['case'], phase, args.deadline, archive, summary)
        update(status='done_pending_review')
    except BaseException as error:
        update(status='failed', error=repr(error))
        save_failure_evidence(base/'results', archive/'failure')
        raise


if __name__ == '__main__':
    main()
