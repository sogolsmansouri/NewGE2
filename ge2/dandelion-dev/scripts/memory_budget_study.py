#!/usr/bin/env python3
"""Frozen q=4 memory-budget plans and small real-GPU acceptance tests.

The smoke results are correctness evidence, never benchmark timings. The
allocator cap excludes CUDA/library allocations; reserve explicit headroom
and reject sampled total-device memory violations as a second check.
"""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time

from arc_job_support import stop_child, write_json
from run_arc_paper_case import digest

MIB = 2**20
GIB = 2**30
PARTITIONS = tuple(range(4, 97))
HIDDEN_COUNTS = tuple(range(7))
WORKLOADS = {
    'tw': dict(model='dot', nodes=41652230, edges=1321528663, relations=1,
               columns=2, graph_prefetch=True),
    'fb': dict(model='complex', nodes=86054151, edges=304727650, relations=14824,
               columns=3, graph_prefetch=False),
}


def limits(budget_mib, rho=.9, external_guard_mib=512):
    if budget_mib <= 0 or not 0 < rho <= 1 or external_guard_mib < 0:
        raise ValueError('Invalid budget, rho or external guard')
    total_limit = math.floor(budget_mib*rho)
    allocator = total_limit-external_guard_mib
    if allocator <= 0:
        raise ValueError('No allocator space remains')
    return dict(budget_mib=budget_mib, rho=rho, total_limit_mib=total_limit,
                external_guard_mib=external_guard_mib, allocator_limit_mib=allocator)


def estimate(workload, p, hidden, budget_mib):
    """Screening estimate only: actual graphs, dense state and peaks must pass."""
    s = WORKLOADS[workload]
    frame = 800*math.ceil(s['nodes']/p)
    # Use all within-state buckets; never assume each state gets uniform work.
    # 2x skew is a screening assumption, not a validated upper bound.
    graph = s['edges'] * (4/p)**2 * s['columns']*8 * 2
    graph *= 2 if s['graph_prefetch'] else 1
    workspace = 4*GIB
    dense = s['relations']*100*4*2*3
    value = dict(p=p, q=4, hidden=hidden, k=4+hidden, frame_bytes=frame,
                 graph_estimate_bytes=math.ceil(graph), workspace_estimate_bytes=workspace,
                 dense_estimate_bytes=dense)
    value['estimated_bytes'] = frame*(4+hidden)+math.ceil(graph)+workspace+dense
    value['parameter_budget_estimate_bytes'] = limits(budget_mib)['allocator_limit_mib']*MIB-math.ceil(graph)-workspace-dense
    c=value['parameter_budget_estimate_bytes']
    value['p0_for_this_reserve']=max(4,math.ceil((4+hidden)*800*s['nodes']/c)) if c>0 else None
    value['screen_pass'] = value['estimated_bytes'] <= limits(budget_mib)['allocator_limit_mib']*MIB
    return value


def candidates(workload, budget_mib):
    out = []
    for hidden in HIDDEN_COUNTS:
        for p in PARTITIONS:
            out.append(dict(estimate(workload, p, hidden, budget_mib),
                            case=f'{workload}_m{budget_mib}_p{p}_q4_h{hidden}',
                            arm='no_hidden' if hidden == 0 else 'shared_pipeline'))
    return out


def make_plan(maximum_mib):
    if maximum_mib < 16384:
        raise ValueError('Study maximum must be at least 16 GiB')
    budgets = [n for n in (16384, 24576, 32768, 40960) if n < maximum_mib]+[maximum_mib]
    return dict(protocol='pipege-memory-budget-v2', phase='plan_not_measurements',
                rho=.9, external_guard_mib=512, q=4, batch_size=50000, width=100,
                optimizer='Adagrad', learning_rate=.1, negative_mass=1,
                chunks=50, negatives=1000, degree_fraction=.5,
                hidden_counts=list(HIDDEN_COUNTS), graph_policy='TW on, FB off, same within each workload',
                pilot_epochs=2, measurement_epochs=10, measurement_repeats=3,
                repeats_status='proposed independent repeats; each run contains 10 epochs',
                selection='best tested feasible per arm; freeze winners before independent repeats',
                evaluation='disabled for timing study; no quality-equivalence claim',
                bound='q-1 formula is a one-overlap reference, not universal impossibility',
                budgets_mib=budgets, workloads=WORKLOADS, selection_points=selection_points(maximum_mib),
                rows=[dict(workload=w, limits=limits(b), candidates=candidates(w,b))
                      for w in WORKLOADS for b in budgets])


def selection_points(maximum_mib):
    """Freeze screening candidates before seeing their timings.

    Include the first estimated-feasible p and its immediate neighbours.
    The lower neighbour tests whether the deliberately conservative screening
    model overestimates memory. This is a finite search, not global optimality.
    """
    points=[]
    if maximum_mib>=24576:
        points=[('tw',24576,16,h) for h in (0,3)]+[('fb',24576,32,h) for h in (0,6)]
    budgets=list(dict.fromkeys([maximum_mib]+[b for b in (24576,16384,32768,40960) if b<=maximum_mib]))
    for b in budgets:
        for w in WORKLOADS:
            for h in HIDDEN_COUNTS:
                feasible=[p for p in PARTITIONS if estimate(w,p,h,b)['screen_pass']]
                if not feasible:
                    raise ValueError(f'Extend partition search range for {w}, {b}, h={h}')
                first=feasible[0]
                points.extend((w,b,p,h) for p in (first,first-1,first+1) if p in PARTITIONS)
    return list(dict.fromkeys(points))


def policy(flags, workload, p, hidden, nodes, cap, profile=False, gate=False):
    out = {k: str(v) for k,v in flags.items() if k.startswith(('GEGE_', 'PYTORCH_'))}
    out.pop('GEGE_BOUNDED_STATE_ORDER_FILE', None)
    out.pop('GEGE_SOFTMAX_NEGATIVE_LOG_MASS_BIAS', None)
    out.update(GEGE_BASELINE_TRAINING_SEMANTICS='1', GEGE_BATCHED_NEGATIVE_PLAN_BATCHES='0',
        GEGE_STATE_NEGATIVE_POOL_REFRESH_BATCHES='0', GEGE_SOFTMAX_NEGATIVE_MASS_SCALE='1',
        GEGE_FRAME_CACHE_AUTO_PIPELINE_FRAMES='0', GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES='-1',
        GEGE_FRAME_CACHE_HIDDEN_FRAMES=str(hidden), GEGE_FRAME_CACHE_MAX_STALE_BACKLOG=str(min(hidden,3)),
        GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET='1',
        GEGE_STATEFLOW_PLANNER='0', GEGE_PARTITION_BUFFER_PEER_RELAY='0',
        GEGE_STATEFLOW_ALLOW_PEER_RELAY='0', GEGE_SYNC_BEFORE_SWAP='0', GEGE_MEM_SWAP_EVENT_SYNC='1',
        GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=str(int(hidden>0)),
        GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=str(int(hidden>0)),
        GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK='0', GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD='0',
        GEGE_BOUNDED_GREEDY_COVER_Q4='1', GEGE_BOUNDED_Q4_OPTIMAL88=str(int(p==32)),
        GEGE_BOUNDED_COVER_EPOCH_RELABEL=str(int(workload=='fb')),
        GEGE_BOUNDED_COVER_RELABEL_SEED='17', GEGE_STATEFLOW_MAX_ADMITS='3',
        GEGE_UNIQUE_BITMAP_NUM_NODES=str(nodes), GEGE_CUDA_MEMORY_STATS='1',
        GEGE_CUDA_ALLOCATOR_LIMIT_MIB=str(cap['allocator_limit_mib']),
        GEGE_PARTITION_BUFFER_PIPELINE_TIMING=str(int(profile)),
        GEGE_PARTITION_BUFFER_SWAP_TIMING=str(int(profile)),
        GEGE_PARTITION_BUFFER_REMAP_BREAKDOWN_TIMING=str(int(profile)),
        GEGE_TRAINING_DETERMINISTIC_GATE=str(int(gate)), GEGE_TRAINING_INPUT_AUDIT=str(int(gate)),
        GEGE_FIXED_BUFFER_MASKED_UPDATE_VERIFY='0', GEGE_STARTUP_TIMING='1')
    return out


def configuration(template, data, model_dir, workload, p, epochs, gate=False):
    cfg = copy.deepcopy(template)
    cfg['storage'].update(dataset=data, model_dir=str(model_dir)+'/', device_ids=[0],
                          device_type='cuda', prefetch=WORKLOADS[workload]['graph_prefetch'])
    cfg['storage']['embeddings']['options'].update(num_partitions=p, buffer_capacity=4, prefetching=False)
    cfg['training'].update(batch_size=50000, num_epochs=epochs, save_model=gate,
                            resume_training=False, resume_from_checkpoint='')
    cfg['training']['checkpoint'] = dict(interval=-1, save_best=False, save_state=False)
    cfg['training']['negative_sampling']['superbatch_negative_plan_batches'] = 0
    cfg['evaluation'].update(epochs_per_eval=0, checkpoint_dir=str(model_dir)+'/')
    return cfg


def make_env(prefix, build, overlay, gpu):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GEGE_', 'OURS_', 'PYTHON', 'CONDA'))}
    env.pop('LD_PRELOAD', None)
    env.update(PATH=f'{prefix}/bin:/usr/bin:/bin',
               LD_LIBRARY_PATH=f'{build}:{prefix}/lib:{prefix}/lib/python3.9/site-packages/torch/lib',
               PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE='1', GEGE_NO_BINDINGS='1',
               CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    return env


def run_training(command, env, directory, cap, timeout, expected_failure=False, power_w=None):
    """Reject any GPU compute contention, cap violation or monitoring gap."""
    apps_cmd = ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']
    if subprocess.check_output(apps_cmd, text=True, timeout=20).strip():
        raise RuntimeError('GPU compute process present; refusing test/timing')
    samples, errors = [], []
    last_jobs=0
    child = None
    deadline = time.monotonic()+timeout
    with (directory/'train.log').open('x') as stream:
        try:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            while child.poll() is None:
                try:
                    row = subprocess.check_output(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=memory.used,power.limit,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                    used,power,util = map(float,row.split(','))
                    if power_w is not None and abs(power-power_w)>.5:
                        raise RuntimeError('GPU power cap differs from requested cohort')
                    apps = subprocess.check_output(apps_cmd, text=True, timeout=10).split()
                    foreign = [pid for pid in apps if int(pid)!=child.pid]
                    job=env.get('SLURM_JOB_ID')
                    if job and time.monotonic()-last_jobs>15:
                        jobs=subprocess.check_output(['squeue','-h','-w',os.uname().nodename.split('.')[0],
                            '-t','RUNNING,COMPLETING','-o','%A'],text=True,timeout=10).split()
                        if not jobs or any(j!=job for j in jobs):
                            raise RuntimeError('Allocation lost or node has another active job: '+repr(jobs))
                        last_jobs=time.monotonic()
                    samples.append(dict(time=time.time(), memory_mib=used,power_w=power,util=util))
                    if foreign:
                        raise RuntimeError('GPU contention: '+repr(foreign))
                    if used > cap['total_limit_mib']:
                        raise RuntimeError('Total GPU footprint exceeded the declared limit')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Test/allocation deadline reached')
                    try:
                        child.wait(timeout=.5)
                    except subprocess.TimeoutExpired:
                        pass
                except BaseException as error:
                    errors.append(repr(error))
                    raise
        finally:
            if child is not None:
                stop_child(child)
            write_json(directory/'hardware.json', dict(samples=samples, errors=errors,
                sampling_interval_s=.5, total_peak_is_sampled_not_continuous=True))
    text = (directory/'train.log').read_text()
    if expected_failure:
        if child.returncode == 0 or not re.search(r'out of memory|Invalid allocator limit|positive integer|exceeds device capacity', text, re.I):
            raise ValueError('Negative cap test did not fail as expected')
        return dict(status='expected_rejection', exit_code=child.returncode)
    if child.returncode:
        raise RuntimeError('Training failed: '+str(child.returncode))
    recorded=re.search(r'\[memory-budget\].*allocator_limit_bytes=(\d+)',text)
    if recorded is None or int(recorded[1])!=cap['allocator_limit_mib']*MIB:
        raise ValueError('Training did not use the requested allocator cap')
    return text


def check_log(text, epochs, edges, hidden, p, nodes):
    times = [int(x)/1000 for x in re.findall(r'Epoch Runtime:\s*(\d+)ms',text)]
    counts = re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%',text)
    peaks = re.findall(r'\[memory-budget-peak\] epoch=(\d+) device=0 allocated_peak_bytes=(\d+) reserved_peak_bytes=(\d+)',text)
    if len(times)!=epochs or counts!=[(str(edges),str(edges))]*epochs or len(peaks)!=epochs:
        raise ValueError('Epoch/edge/peak evidence incomplete')
    if [int(x[0]) for x in peaks]!=list(range(1,epochs+1)):
        raise ValueError('Memory peaks are not from the requested consecutive epochs')
    frames = re.findall(r'deferred backing allocation device=cuda:\d+ visible_rows=(\d+) physical_rows=(\d+) dim=(\d+) pinned=true hidden_frames=(\d+)',text)
    rows = math.ceil(nodes/p)
    expected = tuple(map(str,(4*rows,(4+hidden)*rows,100,hidden)))
    if len(frames)<2 or any(f!=expected for f in frames):
        raise ValueError('Frame allocation differs from request')
    if '[memory-budget]' not in text:
        raise ValueError('Binary lacks allocator-limit support')
    if re.search(r'\b(nan|inf)\b|CUDA error|device-side assert',text,re.I):
        raise ValueError('Numerical or CUDA error')
    allocator = re.search(r'\[memory-budget\].*allocator_limit_bytes=(\d+)', text)
    if allocator is None or any(int(x[2]) > int(allocator[1]) for x in peaks):
        raise ValueError('Allocator peak exceeds cap or cap evidence is missing')
    return dict(epoch_times_s=times, average_epoch_s=statistics.mean(times),
                steady_epoch_s=statistics.mean(times[1:]) if len(times)>1 else None,
                allocated_peak_bytes=max(int(x[1]) for x in peaks),
                reserved_peak_bytes=max(int(x[2]) for x in peaks))


def schedule_evidence(text, epochs, edges, p, expected=None):
    records=re.findall(r'ordering states=(\d+) transitions=(\d+) total_buckets=(\d+) '
        r'max_admits=(\d+) transition_admits=(\d+).*? edge_total=(\d+)',text)
    if len(records)<epochs:
        raise ValueError('Runtime schedule evidence missing')
    geometry={tuple(map(int,row)) for row in records}
    if len(geometry)!=1:
        raise ValueError('Runtime schedule geometry changed between epochs')
    n,transitions,buckets,max_admits,admissions,count=next(iter(geometry))
    if (transitions!=n-1 or buckets!=p*p or count!=edges or max_admits>3
            or (expected is not None and (n,admissions)!=(expected.state_count,expected.total_admissions))):
        raise ValueError('Executed schedule differs from validated plan')
    lower=math.ceil(p*math.ceil((p-1)/3)/4)
    return dict(states=n,transitions=transitions,admissions=admissions,max_admits=max_admits,
                state_lower_bound=lower,state_count_certified_minimum=n==lower)


def smoke(args):
    import numpy as np
    import yaml
    from plan_pipege_cover import plan_cover, write_schedule
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    overlay = root/'python'
    overlay.mkdir()
    (overlay/'gege').symlink_to(args.gege.resolve()/'src/python', target_is_directory=True)
    manifest = json.loads(args.references.read_text())
    cap = limits(16384)
    env = make_env(args.env, args.build, overlay, args.gpu)
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    report = dict(status='running', scope='synthetic correctness only; not memory-fit or benchmark results',
                  binaries={n:digest(args.build/n) for n in ('gege_train','libge2.so')}, cases={})
    write_json(root/'result.json',report)
    try:
        for workload,p in [('tw',16),('fb',32),('tw',8),('fb',8)]:
            spec = WORKLOADS[workload]
            prefix = f'{workload}_p{p}'
            data_dir = root/(prefix+'_data')
            (data_dir/'edges').mkdir(parents=True)
            rng = np.random.default_rng(1987)
            size = 128
            chunks = []
            for i in range(p):
                for j in range(p):
                    src = rng.integers(i*size,(i+1)*size,32)
                    dst = rng.integers(j*size,(j+1)*size,32)
                    columns = (src,dst) if spec['columns']==2 else (src,rng.integers(0,7,32),dst)
                    chunks.append(np.column_stack(columns).astype('<i4'))
            edges = np.concatenate(chunks)
            edges.tofile(data_dir/'edges/train_edges.bin')
            np.savetxt(data_dir/'edges/train_partition_offsets.txt',np.full(p*p,32),fmt='%d')
            if workload=='fb':
                np.savetxt(data_dir/'edges/relation_mapping.txt',
                           np.column_stack((np.arange(7),np.arange(7))),fmt='%d',delimiter=',')
            data = dict(dataset_dir=str(data_dir)+'/',num_nodes=p*size,num_edges=len(edges),
                        num_train=len(edges),num_relations=1 if workload=='tw' else 7,
                        num_valid=-1,num_test=-1)
            (data_dir/'dataset.yaml').write_text(yaml.safe_dump(data))
            schedule,summary,_ = plan_cover(p,4,max_admits=3,restarts=2)
            schedule_path = root/(prefix+'_states.txt')
            # FB p32 uses the engine's certified dynamic 88-state path.
            write_schedule(schedule_path,schedule)
            outputs = []
            for hidden in HIDDEN_COUNTS:
                case = root/f'{prefix}_h{hidden}'
                case.mkdir()
                reference = manifest[workload]
                template = yaml.safe_load(Path(reference['config']).read_text())
                flags = json.loads(Path(reference['flags']).read_text())
                cfg = configuration(template,data,case/'model',workload,p,2,gate=True)
                flags = policy(flags,workload,p,hidden,p*size,cap,gate=True)
                if p!=32:
                    flags['GEGE_BOUNDED_STATE_ORDER_FILE'] = str(schedule_path)
                (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                write_json(case/'flags.json',flags)
                text = run_training([str(args.build/'gege_train'),str(case/'config.yaml')],
                                    dict(env,**flags),case,cap,180)
                metrics = check_log(text,2,len(edges),hidden,p,p*size)
                metrics['schedule']=schedule_evidence(text,2,len(edges),p)
                if workload=='fb':
                    from run_arc_pipege_best import relabel_check
                    relabel_check(text,p,2,17)
                weights = {n:digest(case/'model'/n) for n in ('embeddings.bin','embeddings_state.bin')}
                if workload=='fb':
                    # TorchScript zip hashes can vary despite identical tensors.
                    import torch
                    module = torch.jit.load(str(case/'model/model.pt_0'),map_location='cpu')
                    tensors = {k:v.detach().numpy() for k,v in module.state_dict().items()}
                    np.savez(case/'relations.npz',**tensors)
                outputs.append((case,weights))
                report['cases'][case.name]=dict(status='pass',weights=weights,**metrics)
                write_json(root/'result.json',report)
            if any(outputs[0][1]!=item[1] for item in outputs[1:]):
                raise ValueError(prefix+': pipeline changed entity/optimizer values')
            if workload=='fb':
                for item in outputs[1:]:
                    with np.load(outputs[0][0]/'relations.npz') as a, np.load(item[0]/'relations.npz') as b:
                        if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):
                            raise ValueError('Pipeline changed relation parameters')
            report['cases'][prefix+'_equivalence']=dict(status='pass',bitwise_equal=True)
        case = root/'allocator_oom'
        case.mkdir()
        cfg['storage']['model_dir'] = str(case/'model')+'/'
        (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        flags['GEGE_CUDA_ALLOCATOR_LIMIT_MIB']='1'
        report['cases']['allocator_oom'] = run_training(
            [str(args.build/'gege_train'),str(case/'config.yaml')],dict(env,**flags),case,cap,90,True)
        report['status']='pass'
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        write_json(root/'result.json',report)


def prepare_view(source, expected_sha, spec, p, target):
    import yaml
    from prepare_ge2_partitioned_view import write_partitioned_split
    record = target/'manifest.json'
    if record.exists():
        saved=json.loads(record.read_text())
        if (saved['p'] != p or saved['source_sha256'] != expected_sha
                or digest(target/'edges/train_edges.bin') != saved['train']['output_sha256']
                or digest(target/'edges/train_partition_offsets.txt') != saved['train']['partition_offsets_sha256']):
            raise ValueError('Cached physical partition view changed')
        return saved['dataset']
    target.mkdir(parents=True,exist_ok=False)
    (target/'edges').mkdir()
    train=source/'edges/train_edges.bin'
    if digest(train)!=expected_sha:
        raise ValueError('Canonical training split changed')
    if shutil.disk_usage(target).free < spec['nodes']*800+spec['edges']*spec['columns']*4+5*GIB:
        raise RuntimeError('Insufficient scratch for physical partition view and model state')
    evidence=write_partitioned_split(train,target/'edges/train_edges.bin',
        target/'edges/train_partition_offsets.txt',spec['columns'],math.ceil(spec['nodes']/p),
        p,spec['edges'],2000000)
    data=dict(dataset_dir=str(target)+'/',num_nodes=spec['nodes'],num_relations=spec['relations'],
              num_edges=spec['edges'],num_train=spec['edges'],num_valid=-1,num_test=-1)
    if spec['relations']>1:
        mapping=source/'edges/relation_mapping.txt'
        if not mapping.is_file():
            raise ValueError('Source relation map missing')
        shutil.copy2(mapping,target/'edges/relation_mapping.txt')
    (target/'dataset.yaml').write_text(yaml.safe_dump(data))
    write_json(record,dict(p=p,dataset=data,train=evidence,source_sha256=expected_sha,
                           scope='training-only physical view; logical IDs and split preserved'))
    return data


def sweep(args):
    import yaml
    from plan_pipege_cover import plan_cover, write_schedule
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'serial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        refs=json.loads(args.references.read_text())
        sources=json.loads(args.sources.read_text())
        if json.loads(args.gate.read_text())['status']!='pass':
            raise RuntimeError('Local/ARC synthetic acceptance gate has not passed')
        overlay=root/'python'
        overlay.mkdir(exist_ok=True)
        if not (overlay/'gege').exists():
            (overlay/'gege').symlink_to(args.gege.resolve()/'src/python',target_is_directory=True)
        env=make_env(args.env,args.build,overlay,args.gpu)
        deadline=time.monotonic()+args.seconds
        if args.case:
            selected=[tuple(json.loads(args.case))]
        else:
            selected=json.loads(args.plan.read_text())['selection_points']
        state=dict(status='running',phase='10_epoch_configuration_selection_not_final_repeats',completed={},
                   source_commit=args.commit,points=selected)
        write_json(root/'status.json',state)
        try:
            for workload,budget,p,hidden in selected:
                name=f'{workload}_m{budget}_p{p}_q4_h{hidden}'
                case=root/name
                if (case/'result.json').exists():
                    old=json.loads((case/'result.json').read_text())
                    if (old.get('status') in ('pass','infeasible_oom','infeasible_total_memory')
                            and old.get('source_commit')==args.commit and old.get('epochs_requested')==args.epochs):
                        state['completed'][name]=old['status']
                        continue
                    raise RuntimeError('Previous case requires review: '+name)
                if deadline-time.monotonic()<3600:
                    state['status']='allocation_budget_reached'
                    break
                case.mkdir()
                state['case']=name
                write_json(root/'status.json',state)
                cap=limits(budget)
                spec=WORKLOADS[workload]
                source=sources[workload]
                view=root/'data'/f'{workload}_p{p}'
                data=prepare_view(Path(source['path']),source['train_sha256'],spec,p,view)
                ref=refs[workload]
                cfg=configuration(yaml.safe_load(Path(ref['config']).read_text()),data,
                                  case/'model',workload,p,args.epochs)
                flags=policy(json.loads(Path(ref['flags']).read_text()),workload,p,hidden,
                             spec['nodes'],cap)
                summary=None
                if p!=32:
                    schedule,summary,origin=plan_cover(p,4,max_admits=3,restarts=8)
                    write_schedule(case/'states.txt',schedule)
                    flags['GEGE_BOUNDED_STATE_ORDER_FILE']=str(case/'states.txt')
                    write_json(case/'schedule.json',dict(asdict(summary),origin=origin,
                               note='Minimum only certified when state_gap=0; no global optimality claim'))
                (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                write_json(case/'flags.json',flags)
                result=dict(status='running',source_commit=args.commit,workload=workload,
                            epochs_requested=args.epochs,
                            p=p,q=4,h=hidden,k=4+hidden,limits=cap,
                            frame_bytes=800*math.ceil(spec['nodes']/p),
                            graph_prefetch=spec['graph_prefetch'],paper_ready=False,
                            source_sha256=source['train_sha256'],
                            binaries={n:digest(args.build/n) for n in ('gege_train','libge2.so')},
                            config_sha256=digest(case/'config.yaml'),flags_sha256=digest(case/'flags.json'))
                write_json(case/'result.json',result)
                try:
                    text=run_training([str(args.build/'gege_train'),str(case/'config.yaml')],
                        dict(env,**flags),case,cap,min(6000,deadline-time.monotonic()-180),power_w=args.power_w)
                    result.update(check_log(text,args.epochs,spec['edges'],hidden,p,spec['nodes']),status='pass')
                    result['schedule']=schedule_evidence(text,args.epochs,spec['edges'],p,summary)
                    if workload=='fb':
                        from run_arc_pipege_best import relabel_check
                        relabel_check(text,p,args.epochs,17)
                except Exception as error:
                    text=(case/'train.log').read_text() if (case/'train.log').exists() else ''
                    hardware=json.loads((case/'hardware.json').read_text()) if (case/'hardware.json').exists() else {}
                    # Other failures must stop the campaign, not become memory datapoints.
                    oom=re.search(r'CUDA out of memory|CUDA error: out of memory',text,re.I)
                    result.update(status='infeasible_oom' if oom and not hardware.get('errors') else 'failed',
                                  error=repr(error))
                    if str(error)=='Total GPU footprint exceeded the declared limit':
                        result['status']='infeasible_total_memory'
                    if result['status']=='failed':
                        raise
                finally:
                    write_json(case/'result.json',result)
                    # These are timing-only scratch runs; no checkpoint is promised.
                    if result['status'] in ('pass','infeasible_oom','infeasible_total_memory') and (case/'model').exists():
                        shutil.rmtree(case/'model')
                    if result['status'] in ('pass','infeasible_oom','infeasible_total_memory'):
                        # Keep hashes/counts but regenerate disposable edge views as
                        # needed, so the sweep does not accumulate terabytes.
                        shutil.copy2(view/'manifest.json',case/'data_manifest.json')
                        shutil.rmtree(view)
                state['completed'][name]=result['status']
                write_json(root/'status.json',state)
            else:
                state['status']='selection_complete'
        except BaseException as error:
            state.update(status='failed',error=repr(error))
            raise
        finally:
            write_json(root/'status.json',state)


def main():
    def terminate(signum, frame):
        raise InterruptedError('Study received termination signal; stopping its child')
    signal.signal(signal.SIGTERM,terminate)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','smoke','sweep'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-mib',type=int,default=49140)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--gege',type=Path)
    parser.add_argument('--build',type=Path)
    parser.add_argument('--env',type=Path)
    parser.add_argument('--helpers',type=Path)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--sources',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--gate',type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--seconds',type=int,default=18000)
    parser.add_argument('--power-w',type=float,default=300)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--case',help='Optional JSON [workload,budget_mib,p,hidden] for one measurement')
    args=parser.parse_args()
    if args.mode=='plan':
        args.output.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.output,make_plan(args.maximum_mib))
    else:
        if any(getattr(args,k) is None for k in ('references','gege','build','env','helpers')):
            parser.error('Smoke requires references, gege, build, env and helpers')
        sys.path.insert(0,str(args.helpers.resolve()))
        if args.mode=='smoke':
            smoke(args)
        else:
            if any(getattr(args,k) is None for k in ('sources','gate','commit')):
                parser.error('Sweep also requires sources, gate and commit')
            if args.epochs!=10 or (args.plan is None and args.case is None):
                parser.error('Sweep requires 10 epochs and a frozen plan (or one explicit case)')
            sweep(args)


if __name__=='__main__':
    main()
