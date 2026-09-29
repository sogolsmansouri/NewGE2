#!/usr/bin/env python3
"""Build the frozen study on c30, gate it, then run an isolated resumable block."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from arc_job_support import run_logged, write_json


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def archive(root, destination, index=None):
    if index is not None:
        index.mkdir(parents=True,exist_ok=True)
        for name in ('campaign_status.json','native_build.json','hardware.txt','sweep/status.json','smoke/result.json'):
            source=root/name
            if source.is_file():
                target=index/name
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.tmp')
                shutil.copyfile(source,temporary)
                temporary.replace(target)
    subprocess.run(['rsync','-a','--exclude=/repo/','--exclude=/build/',
        '--exclude=model/','--exclude=data/','--exclude=*_data/',
        '--exclude=python/','--exclude=__pycache__/','--exclude=*.bin','--exclude=*.npz',
        '--exclude=*.tmp',str(root)+'/',str(destination)+'/'],check=True,timeout=90)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('payload','root','results'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--prefix',type=Path,default=Path('/mnt/local/smansou2/ge2-a6000-cuda121'))
    parser.add_argument('--seconds',type=int,default=20700)
    parser.add_argument('--index',type=Path,help='Small status mirror on ARC home; full logs go to --results')
    args=parser.parse_args()
    job=os.environ.get('SLURM_JOB_ID')
    if not job or os.uname().nodename.split('.')[0]!='c30':
        raise RuntimeError('Run inside the allocated c30 batch job')
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGCHLD})
    deadline=time.monotonic()+args.seconds
    args.root.mkdir(parents=True,exist_ok=True)
    args.results.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((args.payload/'manifest.json').read_text())
    for path,expected in manifest['files'].items():
        if digest(args.payload/path)!=expected:
            raise RuntimeError('Frozen payload changed: '+path)
    commit=manifest['source_commit']
    state=dict(status='starting',source_commit=commit,job=job,stage='verify',
               started=datetime.datetime.now().isoformat())
    write_json(args.root/'campaign_status.json',state)
    (args.root/'logs').mkdir(exist_ok=True)
    halt=threading.Event()
    archive_errors=[]

    def mirror():
        while not halt.wait(60):
            try:
                archive(args.root,args.results,args.index)
            except Exception as error:
                archive_errors.append(repr(error))

    worker=threading.Thread(target=mirror,daemon=True)
    worker.start()

    def command(argv,name,env=None,maximum=3600):
        state.update(stage=name,updated=datetime.datetime.now().isoformat())
        write_json(args.root/'campaign_status.json',state)
        if archive_errors:
            raise RuntimeError('Persistent evidence mirror failed: '+repr(archive_errors))
        code=run_logged(list(map(str,argv)),env or dict(os.environ),
            args.root/'logs'/f'{name}_{job}.log',min(maximum,deadline-time.monotonic()-120))
        if code:
            raise RuntimeError(f'{name} exited with status {code}; inspect its saved log')
        return code

    try:
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('c30 has a GPU workload; not starting competing measurements')
        others=subprocess.check_output(['squeue','-h','-w','c30','-t','RUNNING','-o','%i'],text=True).split()
        if any(x!=job for x in others):
            raise RuntimeError('c30 is not isolated from other allocations')
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,power.limit','--format=csv,noheader,nounits'],text=True)
        (args.root/'hardware.txt').write_text(gpus)
        for row in gpus.splitlines():
            name,memory,power=map(str.strip,row.split(','))
            if 'A6000' not in name or int(memory)<49140 or abs(float(power)-300)>.5:
                raise RuntimeError('Expected A6000 49140 MiB at 300 W')
        if shutil.disk_usage(args.root).free<120*2**30:
            raise RuntimeError('Need 120 GiB scratch; existing datasets/checkpoints are not deleted')
        repo=args.root/'repo'
        if not repo.exists():
            command(['git','clone',args.payload/'source.bundle',repo],'clone')
            command(['git','-C',repo,'checkout','--detach',commit],'checkout')
        if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=commit:
            raise RuntimeError('Resume root belongs to another commit')
        if subprocess.check_output(['git','-C',str(repo),'diff','HEAD']):
            raise RuntimeError('Pinned ARC source was modified')
        scripts=repo/'ge2/dandelion-dev/scripts'
        if digest(Path(__file__))!=digest(scripts/Path(__file__).name):
            raise RuntimeError('Bootstrap does not match committed source')
        prefix=args.prefix
        build=args.root/'build'
        compiler=prefix/'bin/x86_64-conda-linux-gnu-c++'
        env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','GEGE_','OURS_'))}
        env.update(PATH=f'{prefix}/bin:/usr/bin:/bin',CUDA_HOME=str(prefix),CUDA_PATH=str(prefix),
            CXX=str(compiler),CUDACXX=str(prefix/'bin/nvcc'),
            LD_LIBRARY_PATH=f'{prefix}/lib:{prefix}/lib/python3.9/site-packages/torch/lib:/usr/lib64',
            PYTHONDONTWRITEBYTECODE='1')
        ready=args.root/'native_build.json'
        if not ready.exists():
            command([prefix/'bin/cmake','-S',repo/'ge2/dandelion-dev/gege','-B',build,
                '-DUSE_CUDA=ON','-DUSE_OMP=OFF','-DBUILD_TESTING=ON','-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_CUDA_ARCHITECTURES=86','-DCMAKE_CUDA_COMPILER='+str(prefix/'bin/nvcc'),
                '-DCMAKE_CXX_COMPILER='+str(compiler),'-DCMAKE_CUDA_HOST_COMPILER='+str(compiler),
                '-DCUDA_HOST_COMPILER='+str(compiler),'-DCUDA_TOOLKIT_ROOT_DIR='+str(prefix),
                '-DCUDAToolkit_ROOT='+str(prefix),'-DCUDA_CUDA_LIBRARY=/usr/lib64/libcuda.so',
                '-DCUDA_CUDA_LIB=/usr/lib64/libcuda.so','-DLIBNVTOOLSEXT='+str(prefix/'lib/libnvToolsExt.so'),
                f'-DCMAKE_LIBRARY_PATH={prefix}/lib;{prefix}/targets/x86_64-linux/lib;/usr/lib64',
                '-DPYTHON_EXECUTABLE='+sys.executable,'-DPython3_EXECUTABLE='+sys.executable,
                f'-DCMAKE_BUILD_RPATH={build};{prefix}/lib;{prefix}/lib/python3.9/site-packages/torch/lib'],
                'configure',env)
            command([prefix/'bin/cmake','--build',build,'--target','gege_train','-j','4'],'build',env,5400)
            write_json(ready,dict(commit=commit,binaries={n:digest(build/n) for n in ('gege_train','libge2.so')}))
        native=json.loads(ready.read_text())
        if native['commit']!=commit or any(digest(build/n)!=v for n,v in native['binaries'].items()):
            raise RuntimeError('Native build changed')
        refs=json.loads((args.payload/'references.json').read_text())
        for files in refs.values():
            for key,value in files.items():
                files[key]=str(args.payload/value)
        write_json(args.root/'references.json',refs)
        common=['--references',args.root/'references.json','--gege',repo/'ge2/dandelion-dev/gege',
                '--build',build,'--env',prefix,'--helpers',args.payload/'helpers']
        gate=args.root/'smoke/result.json'
        if not gate.exists():
            command([prefix/'bin/python',scripts/'memory_budget_study.py','smoke',
                     '--output',args.root/'smoke',*common],'smoke',env,1800)
        checked=json.loads(gate.read_text())
        if checked['status']!='pass' or checked['binaries']!=native['binaries']:
            raise RuntimeError('Native-build acceptance gate is missing or stale')
        remaining=int(deadline-time.monotonic()-180)
        if remaining<3600:
            state['status']='ready_for_next_allocation'
        else:
            command([prefix/'bin/python',scripts/'memory_budget_study.py','sweep',
                '--output',args.root/'sweep',*common,'--sources',args.payload/'sources.json',
                '--plan',args.payload/'plan.json','--gate',gate,'--commit',commit,
                '--seconds',remaining,'--epochs','10','--power-w','300'],
                'measure',env,remaining+60)
            state['status']=json.loads((args.root/'sweep/status.json').read_text())['status']
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        halt.set()
        worker.join(timeout=100)
        state.update(ended=datetime.datetime.now().isoformat(),archive_errors=archive_errors)
        write_json(args.root/'campaign_status.json',state)
        archive(args.root,args.results,args.index)
        lock.close()


if __name__=='__main__':
    main()
