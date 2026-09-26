#!/usr/bin/env python3
"""Separate GE2 clean/counter/Nsight diagnostic; never overwrite paper runs."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--case',choices=('tw_dot','fb_complex'),required=True)
    a=p.parse_args(); b=a.base.resolve()
    spec=json.loads((b/'profile.json').read_text())
    paper=Path(spec['paper_base'])
    sys.path[:0]=[str(b/'scripts'),str(b/'tools'),str(paper/'scripts'),str(paper/'harness/tools')]
    from run_arc_paper_case import digest,hardware_check
    from run_arc_ge2_dot_reproduction import LIB_SHA
    from run_arc_pipege_best import normalize_dataset_metadata
    from run_arc_ge2_allocated_queue import run_logged,write_json
    from summarize_local_tw_ge2_analysis import parse_counter_log
    from analyze_ge2_motivation import counter_phases,trace_summary
    import yaml
    job=os.environ['SLURM_JOB_ID']
    root=b/'runs'/job; root.mkdir(parents=True,exist_ok=False)
    home=Path(spec['summary'])/job; home.mkdir(parents=True,exist_ok=False)
    lock=(b/'serial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(job=job,case=a.case,status='preflight',publication_timing=False)
    def update(**kw):
        state.update(kw,updated=datetime.datetime.now().isoformat())
        write_json(root/'status.json',state);write_json(home/'status.json',state)
    def interrupted(sig,frame):
        raise KeyboardInterrupt('Signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGUSR1,interrupted)
    update()
    alloc=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    (root/'allocation.txt').write_text(alloc)
    deadline=datetime.datetime.fromisoformat(re.search(r'\bEndTime=(\S+)',alloc)[1]).timestamp()-180
    envdir=Path(spec['env']); python=envdir/'bin/python'
    env={k:v for k,v in os.environ.items() if not k.startswith(('GEGE_','GE2_','PYTHON','CONDA','OURS_'))}
    env.pop('LD_PRELOAD',None)
    env.update(PATH=f'{envdir}/bin:/usr/local/cuda/bin:/usr/bin:/bin',OMP_NUM_THREADS='8',
        MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        LD_LIBRARY_PATH=f'{envdir}/lib/python3.9/site-packages/gege:{envdir}/lib/python3.9/site-packages/torch/lib:{envdir}/lib')
    def run(cmd,name,extra=None,monitor=False):
        update(status='running',stage=name)
        if deadline-time.time()<60:
            raise RuntimeError('Allocation deadline')
        rc=run_logged(list(map(str,cmd)),dict(env,**(extra or {})),root/(name+'.log'),deadline-time.time(),
                      root/(name+'.hardware.jsonl') if monitor else None)
        if rc:raise RuntimeError(name+' exited '+str(rc))
    try:
        if os.uname().nodename.split('.')[0]!='c30' or 'JobState=RUNNING ' not in alloc:
            raise ValueError('Expected active c30 allocation')
        if 'UserId='+os.environ['USER']+'(' not in alloc:
            raise ValueError('Allocation ownership mismatch')
        for rel,sha in spec['files'].items():
            if digest(b/rel)!=sha:raise ValueError('Changed profile input '+rel)
        campaign=json.loads((paper/'campaign.json').read_text())
        for rel,sha in campaign['files'].items():
            if digest(paper/rel)!=sha:raise ValueError('Changed frozen paper input '+rel)
        if digest(envdir/'lib/python3.9/site-packages/gege/libge2.so')!=LIB_SHA:
            raise ValueError('Original GE2 library changed')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        jobs=subprocess.check_output(['squeue','-h','-t','RUNNING,COMPLETING','-w','c30','-o','%A'],text=True).split()
        if apps or any(j!=job for j in jobs):raise ValueError('Exclusive idle node required')
        gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,power.limit','--format=csv,noheader,nounits'],text=True).split(',')
        if abs(float(gpu[1])-300)>.1:raise ValueError('Wrong power cohort')
        if shutil.disk_usage(b).free<300*1024**3:raise ValueError('Need 300 GiB local scratch')
        parent=paper/'results'/f"{spec['parents'][a.case]}_ge2_{a.case}"
        if json.loads((parent/'result.json').read_text())['status']!='done':raise ValueError('Clean parent incomplete')
        cfg=yaml.safe_load((parent/'config.yaml').read_text())
        build=b/'build'
        if not (build/'artifacts.sha256').exists():
            run([python,b/'scripts/prepare_ge2_epoch_profile.py','--archive',paper/'ge2.zip',
                 '--patch',b/'instrumentation.patch','--out',b/'source'],'instrument')
            run(['bash',b/'build_local_ge2_control.sh'],'build',dict(
                ROOT=str(b),GE2_SOURCE_ROOT=str(b/'source'),GE2_CONTROL_SOURCE_VARIANT='epoch_profile',
                GE2_CONTROL_ENV_PREFIX=str(envdir),GE2_CONTROL_BUILD_RUN_DIR=str(build),
                GE2_CONTROL_TMP_ROOT=str(b/'build_tmp'),GE2_CONTROL_BUILD_META_OUT=str(b/'build.env'),BUILD_JOBS='8'))
        run(['sha256sum','--check',build/'artifacts.sha256'],'verify_build')
        run([python,b/'scripts/smoke_ge2_epoch_profile.py','--base',b,'--paper-base',paper,
             '--job',job,'--out',root/'capture_smoke'],'capture_smoke')
        smoke=json.loads((root/'capture_smoke/result.json').read_text())
        if smoke['status']!='passed' or smoke['build_manifest_sha256']!=digest(build/'artifacts.sha256'):
            raise ValueError('Capture smoke did not verify this build')
        write_json(root/'capture_smoke_result.json',smoke)
        (root/'nsys_version.txt').write_text(subprocess.check_output([spec['nsys'],'--version'],text=True))
        cell=json.loads((paper/'manifest.json').read_text())['cases'][a.case]
        data=root/'data'
        run([python,paper/'harness/tools/prepare_ge2_partitioned_view.py','--source-data-dir',cell['source'],
             '--output-dir',data,'--num-partitions','16','--edge-columns',cell['columns']],'partition')
        normalize_dataset_metadata(data)
        data_hashes={str(f.relative_to(data)):digest(f) for f in (data/'edges').glob('*.bin')}
        write_json(root/'data_identity.json',data_hashes)
        cfg['storage']['dataset']['dataset_dir']=str(data)+'/'
        cfg['training'].update(num_epochs=3,save_model=False)
        cfg['evaluation']['epochs_per_eval']=1000
        cfg['storage'].pop('save_model',None)
        instrument_env=dict(PYTHONHOME=str(envdir),PYTHONPATH=str(build/'python_package'),
            LD_LIBRARY_PATH=f'{build}:{envdir}/lib:{envdir}/lib/python3.9/site-packages/torch/lib')
        timings={}
        for mode in ('clean','counter','nsight'):
            if any(digest(data/f)!=sha for f,sha in data_hashes.items()):
                raise ValueError('Private edge data changed between diagnostic passes')
            model=root/(mode+'_model'); config=root/(mode+'.yaml')
            cfg['training']['num_epochs']=2 if mode=='nsight' else 3
            cfg['storage'].update(model_dir=str(model)+'/',checkpoint_dir=str(model)+'/')
            config.write_text(yaml.safe_dump(cfg,sort_keys=False))
            command=[envdir/'bin/gege_train',config]
            extra={}
            if mode!='clean':
                extra=dict(instrument_env,GE2_ANALYSIS='1',GE2_ANALYSIS_NVTX='0')
                command=[build/'gege_train',config]
            if mode=='nsight':
                extra.update(GE2_ANALYSIS_NVTX='1',GE2_PROFILE_EPOCH='2',NSYS_NVTX_PROFILER_REGISTER_ONLY='0')
                command=[spec['nsys'],'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                         '--capture-range=nvtx','--nvtx-capture=ge2.profile.epoch_cycle',
                         '--capture-range-end=stop','--wait=all',
                         '--show-output=true','--output='+str(root/'trace')]+command
            run(command,mode,extra,monitor=True)
            text=(root/(mode+'.log')).read_text()
            times=[int(n)/1000 for n in re.findall(r'Epoch Runtime:\s*(\d+)ms',text)]
            if len(times)!=cfg['training']['num_epochs']:raise ValueError('Incomplete '+mode+' epochs')
            counts=re.findall(r'Edges processed:\s*\[(\d+)/(\d+)\],\s*100\.00%',text)
            if counts!=[(str(cell['edges']),str(cell['edges']))]*len(times):raise ValueError('Incorrect edge work')
            if mode!='nsight':
                hardware_check([json.loads(s) for s in (root/(mode+'.hardware.jsonl')).read_text().splitlines()],300,gpu[0].strip())
            timings[mode]=times
        if any(digest(data/f)!=sha for f,sha in data_hashes.items()):
            raise ValueError('Private edge data changed during profiling')
        run([spec['nsys'],'export','--type=sqlite','--lazy=false','--output='+str(root/'trace.sqlite'),root/'trace.nsys-rep'],'export')
        counter_epoch=parse_counter_log(root/'counter.log')[2]
        counter=counter_phases(counter_epoch)
        trace=trace_summary(root/'trace.sqlite')
        if abs(trace['window_seconds']-timings['nsight'][1])>.05:
            raise ValueError('Trace epoch range does not match native epoch timer')
        write_json(root/'epoch2_counters.json',counter_epoch)
        if re.search(r'(dropped|lost)\s+[1-9][0-9]*\s+(events|records)',(root/'nsight.log').read_text(),re.I):
            raise ValueError('Trace reports lost events')
        result=dict(status='done',case=a.case,host='c30',power_w=300,clean_parent=str(parent),
            profile_commit=spec['commit'],profile_manifest_sha256=digest(b/'profile.json'),
            clean_library_sha256=LIB_SHA,profile_build_sha256=digest(build/'artifacts.sha256'),
            parent_config_sha256=digest(parent/'config.yaml'),
            epoch_times_s=timings,counter_epoch2=counter,nsight_epoch2=trace,
            evaluation='not requested; accuracy belongs to clean parent',
            counter_overhead_ratio=timings['counter'][1]/timings['clean'][1],
            nsight_overhead_ratio=timings['nsight'][1]/timings['clean'][1],
            scope='diagnostic only; do not substitute profiled timing in main table')
        write_json(root/'result.json',result);write_json(home/'result.json',result)
        archive=Path(spec['archive'])/job; archive.mkdir(parents=True,exist_ok=False)
        for source in root.iterdir():
            if source.is_file() and source.suffix!='.sqlite':
                dst=archive/source.name;shutil.copyfile(source,dst)
                if digest(source)!=digest(dst):raise ValueError('Archive mismatch '+source.name)
        update(status='done',stage='archived',archive=str(archive))
    except BaseException as e:
        update(status='failed',error=repr(e));raise


if __name__=='__main__':
    main()
