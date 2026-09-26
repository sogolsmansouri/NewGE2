#!/usr/bin/env python3
"""Freeze profiling-only inputs separately from the clean measurement campaign."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--harness',type=Path,required=True)
    p.add_argument('--paper-jobs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--remote-base',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent
    def copy(src,rel):
        target=a.out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    for name in ('analyze_ge2_motivation.py','prepare_ge2_epoch_profile.py','run_arc_ge2_epoch_profile.py',
                 'smoke_ge2_epoch_profile.py','test_ge2_motivation.py','arc_ge2_epoch_profile.sbatch',
                 'submit_ge2_epoch_profile.py'):
        copy(here/name,'scripts/'+name)
    copy(a.harness/'tools/build_local_ge2_control.sh','build_local_ge2_control.sh')
    copy(a.harness/'tools/summarize_local_tw_ge2_analysis.py','tools/summarize_local_tw_ge2_analysis.py')
    copy(a.harness/'runs/local_ge2_control_build_20260831_analysis_v2/instrumentation.patch','instrumentation.patch')
    ledger=json.loads(a.paper_jobs.read_text());parents={}
    for row in ledger['jobs']:
        if row['system']=='ge2' and row['case'] in ('tw_dot','fb_complex'):
            parents[row['case']]=row['job']
    if len(parents)!=2:raise ValueError('Missing clean parents')
    spec=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip(),
        base=a.remote_base,paper_base=ledger['base'],parents=parents,after_job=ledger['jobs'][-1]['job'],
        env='/mnt/local/smansou2/ge2-a6000-cuda121',nsys='/usr/local/cuda/bin/nsys',
        summary='/home/smansou2/arc_results/runs/'+Path(a.remote_base).name,
        archive='/mnt/beegfs/smansou2/'+Path(a.remote_base).name,
        files={str(f.relative_to(a.out)):hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted(a.out.rglob('*')) if f.is_file()})
    (a.out/'profile.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(spec,indent=2))


if __name__=='__main__':
    main()
