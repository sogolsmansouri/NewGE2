#!/usr/bin/env python3
"""Submit two held, serial profiling jobs after the clean measurement chain."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--batch-script',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args()
    if a.ledger.exists():raise ValueError('Existing ledger; refusing duplicate jobs')
    a.ledger.parent.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.manifest.read_text());jobs=[];previous=spec['after_job']
    def save():
        temp=a.ledger.with_suffix('.tmp');temp.write_text(json.dumps(dict(base=spec['base'],jobs=jobs),indent=2)+'\n')
        temp.replace(a.ledger)
    for case in ('tw_dot','fb_complex'):
        dependency='afterok:'+spec['parents'][case]+',afterany:'+previous
        cmd=['sbatch','--parsable','--hold','--nodelist=c30','--job-name=ge2_profile_'+case,
             '--dependency='+dependency,'--output='+str(a.ledger.parent/'job_%j.out'),
             '--error='+str(a.ledger.parent/'job_%j.err'),str(a.batch_script),spec['base'],case]
        output=subprocess.check_output(cmd,text=True)
        match=re.search(r'^([0-9]+)(?:;[^\n]+)?$',output,re.M)
        if not match:raise RuntimeError('Cannot parse submission; inspect queue: '+output)
        previous=match[1];jobs.append(dict(job=previous,case=case,dependency=dependency,released=False));save()
    for row in jobs:
        subprocess.run(['scontrol','release',row['job']],check=True)
        row['released']=True;save()
    print(a.ledger.read_text())


if __name__=='__main__':
    main()
