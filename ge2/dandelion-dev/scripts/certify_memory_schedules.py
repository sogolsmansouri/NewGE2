#!/usr/bin/env python3
"""Certify every planned geometry before allowing its small-GPU acceptance test.

Bounds are not assumed attainable. Solver limits produce unresolved entries,
never heuristic schedules disguised as optimal training plans.
"""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from arc_job_support import run_logged, write_json
from memory_budget_study import certified_schedule, make_env
from run_arc_paper_case import digest


def required_geometries(plan):
    groups={}
    for workload,budget,p,h in plan['selection_points']:
        groups.setdefault(p,[]).append(dict(workload=workload,budget_mib=budget,h=h,k=4+h))
    return dict(sorted(groups.items()))


def native_witness(args,p,folder):
    from plan_pipege_cover import summarize_cover, write_schedule
    env=make_env(args.env,args.build,folder/'unused_overlay','0')
    env.update(GEGE_BOUNDED_GREEDY_COVER_Q4='1',GEGE_STATEFLOW_MAX_ADMITS='3',
               GEGE_BOUNDED_Q4_OPTIMAL88=str(int(p==32)))
    command=[str(args.build/'gege_cover_schedule_analyzer'),'CUSTOM',str(p),'4','1','0','0','1','--dump-buckets']
    with (folder/'native.log').open('w') as out:
        subprocess.run(command,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=300)
    text=(folder/'native.log').read_text()
    matches=re.findall(r'^state \d+ partitions=\[([^]]+)\] assigned_buckets=\d+ buckets=\[([^\n]*)\]$',text,re.M)
    states=[tuple(map(int,a.split(','))) for a,b in matches]
    buckets=[]
    for (raw,assigned),state in zip(matches,states):
        for a,b in re.findall(r'\((\d+),(\d+)\)',assigned):
            pair=(int(a),int(b))
            if any(x not in state for x in pair):
                raise ValueError('Native bucket endpoints are not resident')
            buckets.append(pair)
    if sorted(buckets)!=[(a,b) for a in range(p) for b in range(p)]:
        raise ValueError('Native bucket assignment is not exactly once')
    summary=summarize_cover(states,p,4)
    write_schedule(folder/'incumbent.txt',states)
    write_json(folder/'incumbent.json',asdict(summary))
    return states


def solve_one(args):
    from plan_pipege_cover import (build_minimum_cover, order_cover, maximize_cover_overlap,
        parse_schedule, summarize_cover, write_schedule, MinimumNotProven, OverlapNotProven)
    folder=args.output/f'p{args.one}'
    states=parse_schedule(folder/'incumbent.txt')
    proof={}
    try:
        initial=summarize_cover(states,args.one,4)
        if initial.state_count==initial.covering_lower_bound:
            proof=dict(state_count_optimal=True,minimum_state_count=initial.state_count,
                covering_lower_bound=initial.covering_lower_bound,feasible_upper_bound=initial.state_count,
                scope='full_pair_cover_without_transition_constraints',
                method='feasible_cover_matches_lower_bound')
        else:
            states,proof=build_minimum_cover(args.one,4,incumbent=states,restarts=1,
                time_limit_s=args.solver_seconds,max_candidates=args.max_candidates)
        ordered=order_cover(states,4,allow_bridges=False,overlap_first=True)
        if summarize_cover(ordered,args.one,4).total_overlap>summarize_cover(states,args.one,4).total_overlap:
            states=ordered
        write_schedule(folder/'minimum.txt',states)
        write_json(folder/'minimum.json',dict(asdict(summarize_cover(states,args.one,4)),optimality=proof))
        states,overlap=maximize_cover_overlap(args.one,4,states,time_limit_s=args.solver_seconds,
            workers=4,max_variables=args.max_variables)
        summary=summarize_cover(states,args.one,4)
        if summary.max_admits>3:
            raise ValueError('Optimal cover requires an admission limit above the current runtime contract')
        proof.update(requested_solver='optimal',schedule_sha256=summary.schedule_sha256,
            overlap_optimality='proven',maximum_overlap=summary.total_overlap,
            overlap_scope='all_minimum_covers_and_path_orders',overlap=overlap)
        write_schedule(folder/'states.txt',states)
        write_json(folder/'cover.json',dict(asdict(replace(summary,optimality=proof)),
            source='minimum_states_maximum_overlap'))
        certified_schedule(args.output,args.one)
        write_json(folder/'attempt.json',dict(status='certified',optimality=proof))
    except (MinimumNotProven,OverlapNotProven,ValueError) as error:
        write_json(folder/'attempt.json',dict(status='unresolved',error=str(error),
            minimum_proof=proof,solver_evidence=getattr(error,'certificate',None)))


def audit_seed(source,output,p):
    states,summary,evidence=certified_schedule(source,p)
    from plan_pipege_cover import write_schedule
    write_schedule(output/'states.txt',states)
    write_json(output/'cover.json',evidence)
    write_json(output/'seed_source.json',dict(path=str(source/f'p{p}'),
        cover_sha256=digest(source/f'p{p}'/'cover.json'),states_sha256=digest(source/f'p{p}'/'states.txt')))


def campaign(args):
    args.output.mkdir(parents=True,exist_ok=True)
    geometries=required_geometries(json.loads(args.plan.read_text()))
    state=dict(status='running',scope='schedule certificates and synthetic correctness; no timing measurements',
        runner=str(Path(__file__).resolve()),runner_sha256=digest(Path(__file__)),
        plan_sha256=digest(args.plan),helper_sha256=digest(args.helpers/'plan_pipege_cover.py'),
        binaries={n:digest(args.build/n) for n in ('gege_train','libge2.so','gege_cover_schedule_analyzer')},
        counts=len(geometries),results={})
    versions=subprocess.check_output([sys.executable,'-c',
        'import scipy,ortools,json; print(json.dumps(dict(scipy=scipy.__version__,ortools=ortools.__version__)))'],text=True)
    state['solvers']=json.loads(versions)
    previous=args.output/'status.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        for key in ('plan_sha256','helper_sha256','binaries','solvers'):
            if old.get(key)!=state[key]:
                raise ValueError('Resume identity changed: '+key)
    # Existing certified witnesses allow useful runtime checks before harder solves.
    ordered=sorted(geometries,key=lambda p:(not (args.seeds/f'p{p}'/'cover.json').exists(),p))
    write_json(args.output/'status.json',state)
    for p in ordered:
        folder=args.output/f'p{p}'
        folder.mkdir(exist_ok=True)
        state.update(current_p=p,stage='certificate',updated=time.time())
        write_json(args.output/'status.json',state)
        try:
            if (folder/'cover.json').exists():
                certified_schedule(args.output,p)
            elif (args.seeds/f'p{p}'/'cover.json').exists():
                audit_seed(args.seeds,folder,p)
            else:
                native_witness(args,p,folder)
                command=[sys.executable,str(Path(__file__).resolve()),'--one',str(p),
                    '--output',str(args.output),'--helpers',str(args.helpers),
                    '--solver-seconds',str(args.solver_seconds),'--max-candidates',str(args.max_candidates),
                    '--max-variables',str(args.max_variables)]
                with (folder/'solver.log').open('w') as log:
                    try:
                        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                            timeout=2*args.solver_seconds+180)
                    except subprocess.TimeoutExpired:
                        write_json(folder/'attempt.json',dict(status='unresolved',error='Solver subprocess time limit'))
            if not (folder/'cover.json').exists():
                state['results'][str(p)]=dict(status='unresolved',cases=geometries[p],
                    evidence=str(folder/'attempt.json'))
                continue
            _,summary,_=certified_schedule(args.output,p)
            result=dict(status='certified',states=summary.state_count,overlap=summary.total_overlap,
                admissions=summary.total_admissions,max_admits=summary.max_admits,cases=geometries[p])
            state['results'][str(p)]=result
            state.update(stage='synthetic_gpu',updated=time.time())
            write_json(args.output/'status.json',state)
            gate=folder/'gpu'
            if not (gate/'result.json').exists():
                workloads=sorted({row['workload'] for row in geometries[p]})
                command=[str(args.env/'bin/python'),str(Path(__file__).with_name('memory_budget_study.py')),
                    'smoke','--output',str(gate),'--references',str(args.references),'--gege',str(args.gege),
                    '--build',str(args.build),'--env',str(args.env),'--helpers',str(args.helpers),
                    '--schedule-root',str(args.output),'--geometries',json.dumps([[w,p] for w in workloads])]
                code=run_logged(command,dict(os.environ),folder/'gpu.log',3600)
                if code:
                    raise RuntimeError('Synthetic GPU gate exited with status '+str(code))
            checked=json.loads((gate/'result.json').read_text())
            if checked.get('status')!='pass' or checked.get('binaries')!={n:state['binaries'][n] for n in ('gege_train','libge2.so')}:
                raise ValueError('GPU gate failed or belongs to a different build')
            result.update(status='certified_gpu_pass',gpu=str(gate/'result.json'))
        except Exception as error:
            if isinstance(error,InterruptedError):
                state.update(status='interrupted',error=repr(error))
                raise
            state['results'][str(p)]=dict(status='failed',error=repr(error),cases=geometries[p])
        finally:
            write_json(args.output/'status.json',state)
    state.update(status='complete' if all(r['status']=='certified_gpu_pass' for r in state['results'].values())
                 else 'incomplete_requires_review',updated=time.time())
    write_json(args.output/'status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--helpers',type=Path,required=True)
    parser.add_argument('--one',type=int)
    parser.add_argument('--solver-seconds',type=float,default=120)
    parser.add_argument('--max-candidates',type=int,default=1000000)
    parser.add_argument('--max-variables',type=int,default=1000000)
    for name in ('plan','seeds','references','env','gege','build'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.solver_seconds<=0 or args.max_candidates<1 or args.max_variables<1:
        parser.error('Solver limits must be positive')
    sys.path.insert(0,str(args.helpers.resolve()))
    if args.one:
        solve_one(args)
    else:
        if any(getattr(args,n) is None for n in ('plan','seeds','references','env','gege','build')):
            parser.error('Campaign requires plan, seeds, references, env, gege and build')
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'campaign.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def stop(signum,frame):
                raise InterruptedError('Schedule campaign interrupted')
            signal.signal(signal.SIGTERM,stop)
            campaign(args)


if __name__=='__main__':
    main()
