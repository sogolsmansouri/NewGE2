#!/usr/bin/env python3
"""Freeze committed source, input references, and local acceptance evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from arc_job_support import write_json
from memory_budget_study import certified_schedule, make_plan
from run_arc_paper_case import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','out','references','sources','helpers','gate','schedules'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    import sys
    sys.path.insert(0,str(args.helpers.resolve()))
    plan=make_plan(49140)
    for p in sorted({point[2] for point in plan['selection_points']}):
        certified_schedule(args.schedules,p)
    if subprocess.check_output(['git','-C',str(args.repo),'diff','HEAD']):
        raise RuntimeError('Commit tracked changes before packaging')
    commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    gate=json.loads(args.gate.read_text())
    if gate.get('status')!='pass':
        raise RuntimeError('Local synthetic acceptance did not pass')
    args.out.mkdir(parents=True,exist_ok=False)
    refs=json.loads(args.references.read_text())
    normalized={}
    for workload,files in refs.items():
        normalized[workload]={}
        target=args.out/'references'/workload
        target.mkdir(parents=True)
        for kind,name in [('config','config.yaml'),('flags','flags.json')]:
            shutil.copy2(files[kind],target/name)
            normalized[workload][kind]=str((target/name).relative_to(args.out))
    write_json(args.out/'references.json',normalized)
    shutil.copy2(args.sources,args.out/'sources.json')
    shutil.copy2(args.gate,args.out/'local_gate.json')
    write_json(args.out/'plan.json',plan)
    for p in sorted({point[2] for point in plan['selection_points']}):
        target=args.out/'schedules'/f'p{p}'
        target.mkdir(parents=True)
        for name in ('states.txt','cover.json'):
            shutil.copy2(args.schedules/f'p{p}'/name,target/name)
    helpers=args.out/'helpers'
    helpers.mkdir()
    for name in ('plan_pipege_cover.py','prepare_ge2_partitioned_view.py'):
        shutil.copy2(args.helpers/name,helpers/name)
    scripts=args.repo/'ge2/dandelion-dev/scripts'
    for name in ('run_arc_memory_budget.py','arc_job_support.py','arc_memory_budget_study.sbatch'):
        shutil.copy2(scripts/name,args.out/name)
    subprocess.run(['git','-C',str(args.repo),'bundle','create',str(args.out.resolve()/'source.bundle'),'HEAD'],check=True)
    files={str(p.relative_to(args.out)):digest(p) for p in args.out.rglob('*') if p.is_file()}
    write_json(args.out/'manifest.json',dict(source_commit=commit,files=files,
        local_gate='synthetic bitwise parameter/optimizer equivalence; not full-data timings',
        source_tree=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD:ge2/dandelion-dev/gege'],text=True).strip()))
    print(json.dumps(dict(commit=commit,payload=str(args.out),files=len(files))))


if __name__=='__main__':
    main()
