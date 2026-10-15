#!/usr/bin/env python3
"""Build isolated TW scaling patch; never launch training."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = Path('/workspace/tw_1bcaa5c_scaling_20261015')
COMMIT = 'b1a92d1131e6dba37b6ce8b994caf72c033e6ad9'
LIVE = None

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()
def write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)
def require(ok, message):
    if not ok: raise RuntimeError(message)
def snapshot():
    return {str(p.relative_to(ROOT/'source')): sha(p) for p in sorted((ROOT/'source').rglob('*')) if p.is_file() and '.git' not in p.relative_to(ROOT/'source').parts and '__pycache__' not in p.parts}
def stop_owned():
    if LIVE is not None and LIVE.poll() is None:
        os.killpg(LIVE.pid, signal.SIGTERM)
        try: LIVE.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(LIVE.pid, signal.SIGKILL); LIVE.wait(timeout=10)
def signal_stop(signum, frame):
    stop_owned(); raise RuntimeError('Interrupted by signal %s' % signum)

def main():
    global LIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra-target', action='append', default=[])
    args = parser.parse_args()
    lock=(ROOT/'build.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (ROOT/'build_receipt.json').exists(), 'Build receipt exists; preserve this build')
    require(not (ROOT/'configure.log').exists(), 'Build attempt exists; inspect before a new build attempt')
    signal.signal(signal.SIGTERM, signal_stop); signal.signal(signal.SIGINT, signal_stop)
    source = snapshot()
    base=json.loads((ROOT/'base_source_manifest.json').read_text())
    require(base['commit']==COMMIT, 'Wrong parent source')
    changes={rel: {'before':base['files'].get(rel),'after':source.get(rel)} for rel in sorted(set(base['files'])|set(source)) if base['files'].get(rel)!=source.get(rel)}
    require(changes, 'No isolated patch present; refusing duplicate baseline build')
    write(ROOT/'patch_manifest.json', {'parent_commit':COMMIT, 'created_utc':now(), 'files':changes})
    write(ROOT/'build_source_manifest.json', {'parent_commit':COMMIT, 'created_utc':now(), 'files':source})
    state={'status':'starting','started_utc':now(),'pid':os.getpid(),'parent_commit':COMMIT,'source_changed':True,'patch_manifest_sha256':sha(ROOT/'patch_manifest.json')}
    write(ROOT/'build_status.json',state)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GEGE_','CUDA_','TORCH_','NCCL_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','CUDACXX']}
    env.update(PATH='/usr/local/cuda-12.8/bin:'+env['PATH'],CUDA_HOME='/usr/local/cuda-12.8',CUDACXX='/usr/local/cuda-12.8/bin/nvcc')
    configure=['cmake','-S',str(ROOT/'source/ge2/dandelion-dev/gege'),'-B',str(ROOT/'build'),'-DCMAKE_BUILD_TYPE=Release','-DUSE_CUDA=ON','-DUSE_OMP=OFF','-DBUILD_TESTING=ON','-DCMAKE_CUDA_ARCHITECTURES=86','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DPython3_EXECUTABLE=/usr/local/bin/python','-DPYTHON_EXECUTABLE=/usr/local/bin/python','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    targets=['gege_train','gege_stateflow_validator_tests','gege_order_window_tests','gege_model_transfer_test']+args.extra_target
    commands=[('configure',configure),('compile',['cmake','--build',str(ROOT/'build'),'--target']+targets+['--parallel','4'])]
    try:
        for label,cmd in commands:
            state.update(status=label,command=cmd,updated_utc=now());write(ROOT/'build_status.json',state)
            with (ROOT/(label+'.log')).open('x') as log:
                LIVE=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['active_pid']=LIVE.pid;write(ROOT/'build_status.json',state)
                try: code=LIVE.wait(timeout=1800)
                except subprocess.TimeoutExpired: stop_owned();raise RuntimeError(label+' timed out')
                require(code==0,label+' exited '+str(code))
            LIVE=None
        require(snapshot()==source,'Source changed during build; build not accepted')
        libs=[str(ROOT/'build'),'/usr/local/lib/python3.12/dist-packages/torch/lib','/usr/local/cuda-12.8/lib64']
        rt={'python':'/usr/local/bin/python','binary':str(ROOT/'build/gege_train'),'library':str(ROOT/'build/libge2.so'),'python_paths':[str(ROOT/'python'),str(BASE/'python_deps')],'library_paths':libs,'GEGE_NO_BINDINGS':'1'}
        checkenv=dict(env,LD_LIBRARY_PATH=':'.join(libs),PYTHONPATH=':'.join(rt['python_paths']),GEGE_NO_BINDINGS='1')
        ldd=subprocess.check_output(['ldd',rt['binary']],env=checkenv,text=True)
        (ROOT/'binary_ldd.txt').write_text(ldd)
        require('not found' not in ldd,'Unresolved binary dependency')
        require(str(ROOT/'build/libge2.so') in ldd,'Binary resolved wrong library')
        subprocess.run([rt['python'],'-c','import gege.tools.configuration.gege_config'],env=checkenv,check=True,timeout=30)
        artifacts={name:{'path':str(ROOT/'build'/name),'sha256':sha(ROOT/'build'/name),'bytes':(ROOT/'build'/name).stat().st_size} for name in ['gege_train','libge2.so','gege_stateflow_validator_tests','gege_order_window_tests','gege_model_transfer_test']+args.extra_target}
        state.update(status='completed',commit=COMMIT,patch_manifest_sha256=sha(ROOT/'patch_manifest.json'),completed_utc=now(),runtime=rt,artifacts=artifacts,configure=configure,compiler_workers=4,source_manifest_sha256=sha(ROOT/'build_source_manifest.json'),source_files_checked=len(source),base_commit=COMMIT,exact_parent_source=False)
        write(ROOT/'build_receipt.json',state)
    except BaseException as exc:
        stop_owned();state.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally: write(ROOT/'build_status.json',state)

if __name__=='__main__': main()
