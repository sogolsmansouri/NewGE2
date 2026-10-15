#!/usr/bin/env python3
"""Latest-commit TW host-transport gates and matched timing; no automatic long run."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import traceback
import yaml

ROOT=Path(__file__).resolve().parent
BASE=Path('/workspace/tw_1bcaa5c_scaling_20261015')
OLD=BASE/'validation_50k_2epoch'
COMMIT='b1a92d1131e6dba37b6ce8b994caf72c033e6ad9'
UUIDS=['GPU-0761238c-da94-7c93-1fce-739e46be8784','GPU-d2bca02c-5b52-54c6-3fd9-9ae95284bb02']
MEMORY_LIMIT=90000000000
LIVE=None

def require(ok,message):
    if not ok: raise RuntimeError(message)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def freeze(path,value):
    require(not path.exists() or read(path)==value,'Existing file differs: '+str(path))
    if not path.exists(): write(path,value)
def load_helpers():
    path=OLD/'run_scaling.py'
    require(sha(path)=='1842097656dc1467199cbb8862429e663a879feb686f3e287ae3e77bb50296bc','Frozen baseline helper changed')
    spec=importlib.util.spec_from_file_location('baseline_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
H=load_helpers()
def memory():
    cg=Path('/sys/fs/cgroup')
    stat={k:int(v) for k,v in (line.split() for line in (cg/'memory.stat').read_text().splitlines())}
    events={k:int(v) for k,v in (line.split() for line in (cg/'memory.events').read_text().splitlines())}
    return {'current':int((cg/'memory.current').read_text()),'anon':stat.get('anon',0),'shmem':stat.get('shmem',0),'anon_plus_shmem':stat.get('anon',0)+stat.get('shmem',0),'events':events}
def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
def interrupted(signum,frame):
    stop_owned(LIVE);raise RuntimeError('Interrupted by signal '+str(signum))
def data_gate():
    receipt=BASE/'data_validation.json'
    require(sha(receipt)=='11767249b6216288a220b82869085eddd28d0530b36370968c003d77a76149eb','Data validation receipt changed')
    data=read(receipt)
    require(data['status']=='passed' and data['ready_for_training'],'Data gate failed')
    directory=Path(data['data_dir'])
    checked={}
    for rel,entry in data['checked'].items():
        st=(directory/rel).stat()
        current={'bytes':st.st_size,'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
        require(all(current[k]==entry[k] for k in current),'Data stat binding changed: '+rel)
        checked[rel]=current
    require(sha(directory/'dataset.yaml')==data['metadata_rewrite']['view_sha256'],'Data metadata changed')
    return {'receipt_sha256':sha(receipt),'data_dir':str(directory),'stat_bindings':checked,'full_hash_reused':True}
def build_gate():
    b=read(ROOT/'build_receipt.json')
    require(b['status']=='completed' and b.get('commit',b.get('base_commit',b.get('parent_commit')))==COMMIT,'Latest build not completed or wrong base commit')
    require(b['source_manifest_sha256']==sha(ROOT/'build_source_manifest.json'),'Build source receipt changed')
    require(b['patch_manifest_sha256']==sha(ROOT/'patch_manifest.json'),'Declared patch receipt changed')
    source=read(ROOT/'build_source_manifest.json')['files']
    actual={str(p.relative_to(ROOT/'source')):sha(p) for p in sorted((ROOT/'source').rglob('*')) if p.is_file() and '.git' not in p.relative_to(ROOT/'source').parts and not ('__pycache__' in p.parts and p.suffix=='.pyc')}
    require(source==actual,'Source differs from the frozen patched build')
    current=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip()
    require(current==COMMIT,'Git HEAD changed')
    for entry in b['artifacts'].values(): require(sha(entry['path'])==entry['sha256'],'Built artifact changed: '+entry['path'])
    return b

def runtime_environment(plan,flags,uuids):
    env=H.runtime_environment(plan,flags,uuids)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        env[key]=str(plan['cpu_threads'])
    return env

def snapshot_runtime(pid,case,plan):
    runtime=H.snapshot_process(pid,case,plan)
    if runtime is None: return None
    try:
        actual=dict(item.split('=',1) for item in (Path('/proc')/str(pid)/'environ').read_bytes().decode().split('\0') if '=' in item)
    except (FileNotFoundError,ProcessLookupError): return None
    selected={k:actual.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
    require(all(v==str(plan['cpu_threads']) for v in selected.values()),'Training process thread environment mismatch')
    runtime['cpu_thread_environment']=selected
    write(case/'runtime.json',runtime)
    return runtime

def prepare(cpu_threads):
    require(cpu_threads==8,'This comparison fixes eight CPU threads for both GPU counts')
    b=build_gate();data=data_gate()
    recipe=ROOT/'historical_recipe'
    if not recipe.exists(): shutil.copytree(OLD/'historical_recipe',recipe)
    for p in (OLD/'historical_recipe').rglob('*'):
        if p.is_file(): require(sha(recipe/p.relative_to(OLD/'historical_recipe'))==sha(p),'Copied recipe changed')
    fixture=ROOT/'fixture'
    if not fixture.exists():
        subprocess.run(['/usr/local/bin/python',str(recipe/'prepare_fixture.py'),'--destination',str(fixture)],check=True,timeout=120)
    fm=read(fixture/'fixture_manifest.json')
    for rel,entry in fm['files'].items(): require(sha(fixture/rel)==entry['sha256'],'Fixture input changed')
    descriptors=[]
    for name,gpus,synthetic,verify in [('fixture_validate',2,True,True),('fixture_production',2,True,False),('1gpu',1,False,False),('2gpu',2,False,False)]:
        case=ROOT/'cases'/name;case.mkdir(parents=True,exist_ok=True)
        config=yaml.safe_load((fixture/'config.yaml' if synthetic else recipe/(str(gpus)+'gpu.template.yaml')).read_text())
        config['storage']['dataset']['dataset_dir']=str(fixture/'data' if synthetic else Path(data['data_dir']))+'/'
        config['storage']['model_dir']=str(case/'model')+'/'
        config['evaluation']['checkpoint_dir']=str(case/'model')+'/'
        config['training']['save_model']=False
        config['training']['checkpoint'].update(interval=-1,save_best=False,save_state=False)
        config['training']['num_epochs']=2
        flags=read(fixture/'flags.json' if synthetic else recipe/'flags.template.json')
        flags['GEGE_BOUNDED_STATE_ORDER_FILE']=str(fixture/'states.txt' if synthetic else recipe/'states.txt')
        flags['GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK']='1'
        if verify:
            flags.update(GEGE_STATEFLOW_PEER_RELAY_VALIDATE='1',GEGE_STATEFLOW_PEER_RELAY_VALIDATE_FAIL_FAST='1',GEGE_STATEFLOW_PEER_RELAY_VALIDATE_MAX_CHECKS='4096')
        # Retain the original algorithm flags, with intentional host transport enabled.
        if not verify:
            expected=read(OLD/'cases/2gpu/flags.json')
            expected['GEGE_BOUNDED_STATE_ORDER_FILE']=flags['GEGE_BOUNDED_STATE_ORDER_FILE']
            expected['GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK']='1'
            if synthetic: expected['GEGE_UNIQUE_BITMAP_NUM_NODES']='65536'
            require(expected==flags,'Unexpected production flag change')
        text=yaml.safe_dump(config)
        cp=case/'config.yaml';require(not cp.exists() or cp.read_text()==text,'Case config differs')
        if not cp.exists(): cp.write_text(text)
        freeze(case/'flags.json',flags)
        descriptors.append({'name':name,'directory':str(case),'config_source':str(cp),'flags_source':str(case/'flags.json'),'gpu_count':gpus,'epochs':2,'edges':3840000 if synthetic else 1321528663,'synthetic':synthetic,'host_payload_validation':verify,'timeout':300 if synthetic else 1800,'config_sha256':sha(cp),'flags_sha256':sha(case/'flags.json')})
    rt=b['runtime']
    rt=dict(rt,python_paths=rt.get('python_paths') or rt['pythonpath'].split(':'),library_paths=rt.get('library_paths') or rt['ld_library_path'].split(':'))
    plan={'protocol':'latest_commit_host_transport_50k_two_epochs','base_commit':COMMIT,'batch_size_per_gpu':50000,'full_epochs':2,'run_root':str(ROOT),'train_binary':rt['binary'],'runtime_library':rt['library'],'python_paths':rt['python_paths'],'library_paths':rt['library_paths'],'runtime_flags':{'GEGE_NO_BINDINGS':'1'},'gpu_uuids':UUIDS,'build_receipt_sha256':sha(ROOT/'build_receipt.json'),'source_manifest_sha256':sha(ROOT/'build_source_manifest.json'),'patch_manifest_sha256':sha(ROOT/'patch_manifest.json'),'source_changed':True,'cpu_threads':cpu_threads,'runtime_change':{'changes':['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK'],'before':'Historical1bcaa5c build: thread variables unset; direct peer transport enabled.','after':dict({k:str(cpu_threads) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},GEGE_STATEFLOW_PEER_RELAY_FORCE_HOST_FALLBACK='1')},'comparison_scope':'Matched1GPU/2GPU latest patched build with8CPUthreads andhosttransport; oldbaseline differs in source and runtime, so changes cannot be attributed tothreads alone.','runner_sha256':sha(__file__),'data_gate':data,'cases':descriptors,'memory_anon_plus_shmem_limit_bytes':MEMORY_LIMIT,'checkpoint_saving':False,'accuracy_evaluation':False,'automatic_followup':'none'}
    freeze(ROOT/'execution_plan.json',plan)
    return plan

def fields(line):
    answer={}
    for key,value in re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)=([-+0-9.,]+)',line):
        def convert(v): return float(v) if '.' in v else int(v)
        try: answer[key]=[convert(v) for v in value.split(',')] if ',' in value else convert(value)
        except ValueError: pass
    return answer

def counters(text):
    answer={str(n):{} for n in [1,2]}
    for line in text.splitlines():
        m=re.search(r'\[perf\]\[epoch ([12])\]((?:\[[^\]]+\])*) (.*)',line)
        if m:
            epoch,section,tail=m.groups()
            if section in ['','[peer_relay]','[device]','[frame_cache]','[frame_cache][device]','[gpu 0]','[gpu 1]','[gpu 0][batch_fetch]','[gpu 1][batch_fetch]']:
                answer[epoch].setdefault(section or 'total',{}).update(fields(tail))
    return answer

def run_cpu_gate(plan):
    receipt=ROOT/'cpu_gate.json';plan_hash=sha(ROOT/'execution_plan.json')
    if receipt.exists():
        old=read(receipt);require(old['status']=='passed' and old['plan_sha256']==plan_hash and sha(ROOT/'cpu_thread_probe.json')==old['probe_sha256'] and sha(ROOT/'cpu_tests.log')==old['tests_log_sha256'],'Existing CPU gate differs');return old
    env=runtime_environment(plan,{},[])
    code="import json,os,torch; print(json.dumps({'torch_version':torch.__version__,'intraop_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'affinity':sorted(os.sched_getaffinity(0)),'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}))"
    cmd=['/usr/local/bin/python','-c',code]
    p=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
    require(p.returncode==0,'Thread probe failed: '+p.stderr[-2000:])
    probe=json.loads(p.stdout);freeze(ROOT/'cpu_thread_probe.json',probe)
    require(probe['intraop_threads']==plan['cpu_threads'],'Torch intra-op count does not match selected thread count')
    require(all(v==str(plan['cpu_threads']) for v in probe['environment'].values()),'Probe thread environment mismatch')
    result={'status':'passed','completed_utc':now(),'plan_sha256':plan_hash,'probe_sha256':sha(ROOT/'cpu_thread_probe.json'),'intraop_threads':probe['intraop_threads'],'interop_threads':probe['interop_threads'],'scope':'CPU runtime thread configuration and native CPU tests for latest patched build.'}
    cmd=['ctest','--test-dir',str(ROOT/'build'),'--output-on-failure','--timeout','120','-R','^(gege_stateflow_validator_tests|gege_order_window_tests)$']
    for name in ['gege_stateflow_validator_tests','gege_order_window_tests']:
        require((ROOT/'build'/name).is_file(),'Required native CPU test binary missing: '+name)
    with (ROOT/'cpu_tests.log').open('x') as log:
        test=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    result.update(status='passed' if test.returncode==0 else 'failed',test_exit_code=test.returncode,tests_log_sha256=sha(ROOT/'cpu_tests.log'),test_command=cmd)
    write(receipt,result);require(test.returncode==0,'Native CPU test gate failed');return result

def transfer_checks(stats,text,d):
    checks={}
    for epoch in ['1','2']:
        frame=stats[epoch].get('[frame_cache]',{})
        checks['epoch_'+epoch+'_preload_miss_zero']=frame.get('preload_miss_swaps')==0
        checks['epoch_'+epoch+'_expected_partial_preload_swaps']=frame.get('partial_preload_swaps')==(18 if d['gpu_count']==2 else 0)
        checks['epoch_'+epoch+'_expected_visible_host_admit_parts']=frame.get('fallback_visible_admit_parts')==(18 if d['gpu_count']==2 else 0)
    if d['gpu_count']==2:
        for epoch in ['1','2']:
            peer=stats[epoch].get('[peer_relay]',{})
            checks['epoch_'+epoch+'_peer_counters_present']=all(k in peer for k in ['peer_copy_count','peer_bytes_executed','host_fallback_count','host_fallback_bytes','descriptor_mismatch_count'])
            checks['epoch_'+epoch+'_zero_peer_copies']=peer.get('peer_copy_count')==0
            checks['epoch_'+epoch+'_zero_peer_bytes']=peer.get('peer_bytes_executed')==0
            checks['epoch_'+epoch+'_36_host_handoffs']=peer.get('host_fallback_count')==36
            checks['epoch_'+epoch+'_expected_host_handoff_bytes']=peer.get('host_fallback_bytes')==(58982400 if d['synthetic'] else 37487000000)
            checks['epoch_'+epoch+'_no_descriptor_mismatch']=peer.get('descriptor_mismatch_count')==0
    if d['host_payload_validation']:
        validation_lines=[line for line in text.splitlines() if '[stateflow-host-payload-validate ' in line]
        verifications=[fields(line) for line in validation_lines]
        validation_ids=[int(x) for line in validation_lines for x in re.findall(r'\[stateflow-host-payload-validate (\d+)\]',line)]
        checks['all_72_host_payloads_validated']=len(verifications)==72 and sorted(validation_ids)==list(range(72))
        checks['host_payload_validation_exact_match']=all(v.get('dst_mismatch_values')==v.get('src_mismatch_values')==v.get('dst_vs_source_max_abs')==v.get('dst_vs_host_max_abs')==0 for v in verifications)
    else: verifications=[]
    return checks,len(verifications)

def run_case(plan,d,state):
    global LIVE
    case=Path(d['directory']);rp=case/'result.json';plan_hash=sha(ROOT/'execution_plan.json')
    if rp.exists():
        prior=read(rp);require(prior['status']=='completed' and prior['plan_sha256']==plan_hash,'Existing case failed/different; preserve and inspect')
        require(prior['train_log_sha256']==sha(case/'train.log'),'Completed log changed');return prior
    require(not (case/'train.log').exists(),'Partial case exists; preserve it and inspect')
    require(sha(d['config_source'])==d['config_sha256'] and sha(d['flags_source'])==d['flags_sha256'],'Case input changed')
    config=yaml.safe_load(Path(d['config_source']).read_text())
    require(config['training']['batch_size']==50000 and config['training']['num_epochs']==2,'Wrong workload')
    require(config['training']['save_model'] is False and config['training']['checkpoint']['interval']==-1,'Checkpoint saving must remain off')
    require(config['storage']['device_ids']==list(range(d['gpu_count'])),'Wrong logical GPUs')
    require(config['training']['dense_sync_batches']==1,'Dense sync changed')
    uuids=UUIDS[:d['gpu_count']];write(case/'gpu_before.json',H.require_idle(uuids))
    (case/'topology.txt').write_text(H.command(['nvidia-smi','topo','-m']))
    env=runtime_environment(plan,read(d['flags_source']),uuids)
    invocation={'command':[plan['train_binary'],d['config_source']],'cwd':str(case),'started_utc':now(),'gpu_uuids':uuids,'environment':{k:v for k,v in env.items() if k.startswith('GEGE_') or k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'config_sha256':d['config_sha256'],'flags_sha256':d['flags_sha256'],'plan_sha256':plan_hash,'timeout_seconds':d['timeout']}
    write(case/'execution.json',invocation)
    state.update(status='running',active=d['name'],updated_utc=now());write(ROOT/'status.json',state)
    before=memory();require(before['anon_plus_shmem']<MEMORY_LIMIT,'Memory already above limit')
    peak=before['anon_plus_shmem'];monitor=None;runtime=None;error=None;code=None;foreign=[];unknown=[];owned=set();start=time.monotonic();last_sample=-10
    with (case/'train.log').open('x') as log,(case/'gpu_monitor.csv').open('x') as gpu_log,(case/'resource_monitor.jsonl').open('x') as resource_log:
        try:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem','--format=csv','-l','5'],stdout=gpu_log,stderr=subprocess.STDOUT,start_new_session=True)
            LIVE=subprocess.Popen(invocation['command'],cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            invocation['pid']=LIVE.pid;write(case/'execution.json',invocation)
            while LIVE.poll() is None:
                elapsed=time.monotonic()-start
                if runtime is None: runtime=snapshot_runtime(LIVE.pid,case,plan)
                mem=memory();peak=max(peak,mem['anon_plus_shmem'])
                require(mem['anon_plus_shmem']<MEMORY_LIMIT,'Anon+shmem memory limit reached')
                require(mem['events'].get('oom_kill',0)==before['events'].get('oom_kill',0),'Cgroup OOM kill occurred')
                require(elapsed<d['timeout'],'Case timeout reached')
                if elapsed-last_sample>=5:
                    procs=H.selected_gpu_processes(uuids);classification=H.classify_gpu_processes(procs,LIVE.pid,plan['train_binary'])
                    foreign.extend(classification['foreign']);unknown.extend(classification['unknown_namespace']);owned.update(x['gpu_uuid'] for x in classification['owned'])
                    sample={'captured_utc':now(),'elapsed_seconds':elapsed,'memory':mem,'gpu_processes':procs,'namespace_classification':classification}
                    resource_log.write(json.dumps(sample)+'\n');resource_log.flush()
                    require(not foreign,'Foreign GPU process detected')
                    state.update(active_pid=LIVE.pid,active_elapsed_seconds=round(elapsed,1),anon_plus_shmem_bytes=mem['anon_plus_shmem'],updated_utc=now());write(ROOT/'status.json',state)
                    last_sample=elapsed
                time.sleep(.25 if runtime is None else 1)
            code=LIVE.wait()
        except BaseException as exc: error=repr(exc)
        finally:
            stop_owned(LIVE);stop_owned(monitor)
            if LIVE is not None: code=LIVE.poll()
    elapsed=time.monotonic()-start;LIVE=None
    text=(case/'train.log').read_text(errors='replace');parsed=H.parse_log(text,2,d['gpu_count'],d['edges']);stats=counters(text)
    checks=dict(parsed['checks'])
    checks.update(process_exit_zero=code==0,no_controller_failure=error is None,runtime_library_verified=runtime is not None,no_foreign_gpu_processes=not foreign)
    payload_checks,validation_count=transfer_checks(stats,text,d)
    checks.update(payload_checks)
    accepted=all(checks.values());exclusive=not foreign and not unknown and owned==set(uuids)
    result={'status':'completed' if accepted else 'failed','started_utc':invocation['started_utc'],'finished_utc':now(),'wall_seconds':elapsed,'exit_code':code,'controller_error':error,'checks':checks,'timing':parsed,'epoch_counters':stats,'host_payload_validation_checks':validation_count,'memory_before':before,'peak_anon_plus_shmem_bytes':peak,'descriptor':d,'plan_sha256':plan_hash,'train_log_sha256':sha(case/'train.log'),'runtime_maps_captured':runtime is not None,'gpu_process_exclusivity_confirmed':exclusive,'timing_eligible':accepted and exclusive and not d['synthetic'],'foreign_gpu_processes':foreign,'unknown_namespace_gpu_processes':unknown,'checkpoint_saving':False,'accuracy_evaluated':False,'limitations':['Counters describe scheduled/executed peer transfers; no per-edge execution ledger.','Two epochs provide one post-warmup timing, not a long-run mean.','Native epoch timers exclude initialization, reset, and finalization.','GPU process exclusivity is unconfirmed if host PIDs cannot be resolved.','Synthetic transfer checks do not establish full-data model accuracy.','Host fallback is intentional transport; its count is not an error.','TwoGPU partial_preload_swaps=18 and fallback_visible_admit_parts=18 reflect deliberate visible host handoffs alongside ready hidden preloads; embeddings-only frame counters differ from36combined embeddings+optimizer handoffs.','Historical baseline uses another commit and transport, so source and runtime effects cannot be separated.']}
    write(rp,result);require(accepted,'Case failed: '+str(rp));return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','fixture','timing'],required=True);parser.add_argument('--cpu-threads',type=int,required=True);args=parser.parse_args()
    lock=(ROOT/'validation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    state={'status':'preflight','stage':args.stage,'controller_pid':os.getpid(),'started_utc':now(),'active':None,'completed':[]};write(ROOT/'status.json',state)
    results=[]
    try:
        plan=prepare(args.cpu_threads);plan_hash=sha(ROOT/'execution_plan.json')
        if args.stage=='prepare':
            state.update(status='prepared',updated_utc=now());return
        if args.stage=='fixture':
            run_cpu_gate(plan);descriptors=plan['cases'][:2]
        else:
            gate=read(ROOT/'fixture_review.json')
            require(gate['status']=='passed' and gate['plan_sha256']==plan_hash,'Fixture gate missing or from different build')
            require(sha(ROOT/'cpu_gate.json')==gate['cpu_gate_sha256'],'CPU gate changed')
            for path,digest in gate['result_hashes'].items(): require(sha(path)==digest,'Fixture result changed')
            require(shutil.disk_usage(ROOT).free>=75*1024**3,'Need75GiB free for two temporary model stores')
            descriptors=plan['cases'][2:]
        H.require_idle(UUIDS)
        for d in descriptors:
            result=run_case(plan,d,state);results.append(result);state['completed'].append(d['name']);state.update(active=None,updated_utc=now());write(ROOT/'status.json',state)
        if args.stage=='fixture':
            review={'status':'passed','completed_utc':now(),'plan_sha256':plan_hash,'cpu_gate_sha256':sha(ROOT/'cpu_gate.json'),'result_hashes':{str(Path(d['directory'])/'result.json'):sha(Path(d['directory'])/'result.json') for d in descriptors},'next_action':'Stop for review; timing requires an explicit --stage timing --cpu-threads N invocation.'}
            write(ROOT/'fixture_review.json',review)
        else:
            one=results[0]['timing']['epoch_seconds'];two=results[1]['timing']['epoch_seconds']
            review={'status':'completed','completed_utc':now(),'parent_commit':COMMIT,'source_changed':True,'patch_manifest_sha256':sha(ROOT/'patch_manifest.json'),'cpu_threads':plan['cpu_threads'],'runtime_change':plan['runtime_change'],'plan_sha256':plan_hash,'epoch_times_seconds':{'1gpu':one,'2gpu':two},'epoch2_speedup':one[1]/two[1],'epoch2_parallel_efficiency':one[1]/(2*two[1]),'baseline_epoch_times_seconds':{'1gpu':[195.367,193.255],'2gpu':[125.299,119.385]},'two_gpu_epoch2_reduction_vs_baseline':1-two[1]/119.385,'timing_eligible':all(r['timing_eligible'] for r in results),'results':results,'accuracy_evaluated':False,'checkpoint_saving':False,'automatic_followup':'none; stop for review before any ten-epoch run'}
            write(ROOT/'comparison_review.json',review)
        state.update(status='completed',active=None,updated_utc=now())
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),updated_utc=now());raise
    finally: write(ROOT/'status.json',state)

if __name__=='__main__': main()
