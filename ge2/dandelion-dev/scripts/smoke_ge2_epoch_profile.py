#!/usr/bin/env python3
"""Check epoch-2 Nsight capture on a tiny synthetic graph in an owned ARC allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--paper-base',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args(); b=a.base.resolve()
    job=subprocess.check_output(['scontrol','show','job',a.job,'-o'],text=True)
    if ('JobState=RUNNING ' not in job or 'UserId='+os.environ['USER']+'(' not in job
            or 'NodeList='+os.uname().nodename.split('.')[0]+' ' not in job):
        raise ValueError('Need an owned running allocation on this node')
    apps=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise ValueError('GPU 0 is occupied')
    root=a.out or b/'smoke';root.mkdir(exist_ok=False)
    (root/'allocation.txt').write_text(job)
    sys.path[:0]=[str(b/'scripts'),str(b/'tools'),str(a.paper_base/'harness/tools')]
    from analyze_ge2_motivation import trace_summary,counter_phases
    from summarize_local_tw_ge2_analysis import parse_counter_log
    with zipfile.ZipFile(a.paper_base/'ge2.zip') as z:
        cfg=yaml.safe_load(z.read('dandelion-dev/gege/configs/fb15k_16p.yaml'))
    data=root/'data';(data/'edges').mkdir(parents=True);(data/'nodes').mkdir()
    rng=np.random.default_rng(1930);blocks=[]
    for src in range(16):
        for dst in range(16):
            blocks.append(np.column_stack((rng.integers(src*32,(src+1)*32,17),
                                           rng.integers(7,size=17),rng.integers(dst*32,(dst+1)*32,17))))
    edges=np.concatenate(blocks).astype('<i4');edges.tofile(data/'edges/train_edges.bin')
    (data/'edges/train_partition_offsets.txt').write_text('17\n'*256)
    (data/'nodes/node_mapping.txt').write_text(''.join(f'{i},{i}\n' for i in range(512)))
    (data/'edges/relation_mapping.txt').write_text(''.join(f'{i},{i}\n' for i in range(7)))
    (data/'dataset.yaml').write_text(yaml.safe_dump(dict(dataset_dir=str(data)+'/',num_nodes=512,
        num_relations=7,num_edges=len(edges),num_train=len(edges),num_valid=-1,num_test=-1,
        node_feature_dim=-1,rel_feature_dim=-1,num_classes=-1,initialized=False)))
    cfg['model']['random_seed']=123
    cfg['model']['encoder']['layers'][0][0]['output_dim']=10
    cfg['model']['decoder']['options']['input_dim']=10
    cfg['model']['decoder']['type']='COMPLEX'
    cfg['storage']['dataset']=dict(dataset_dir=str(data)+'/')
    cfg['storage'].update(model_dir=str(root/'model')+'/',checkpoint_dir=str(root/'model')+'/')
    cfg['storage'].pop('save_model',None)
    cfg['training'].update(num_epochs=2,batch_size=50,save_model=False)
    cfg['training']['negative_sampling'].update(num_chunks=5,negatives_per_positive=20)
    cfg['evaluation']['epochs_per_eval']=1000
    config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    envdir=Path('/mnt/local/smansou2/ge2-a6000-cuda121')
    env={k:v for k,v in os.environ.items() if not k.startswith(('GEGE_','GE2_','PYTHON','CONDA'))}
    env.pop('LD_PRELOAD',None)
    env.update(PATH=f'{envdir}/bin:/usr/local/cuda/bin:/usr/bin:/bin',PYTHONHOME=str(envdir),
        PYTHONPATH=str(b/'build/python_package'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',
        GE2_ANALYSIS='1',GE2_ANALYSIS_NVTX='1',GE2_PROFILE_EPOCH='2',NSYS_NVTX_PROFILER_REGISTER_ONLY='0',
        LD_LIBRARY_PATH=f'{b}/build:{envdir}/lib:{envdir}/lib/python3.9/site-packages/torch/lib')
    cmd=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
         '--capture-range=nvtx','--nvtx-capture=ge2.profile.epoch_cycle','--capture-range-end=stop','--wait=all',
         '--output='+str(root/'trace'),str(b/'build/gege_train'),str(config)]
    with (root/'train.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    subprocess.run(['/usr/local/cuda/bin/nsys','export','--type=sqlite','--lazy=false',
                    '--output='+str(root/'trace.sqlite'),str(root/'trace.nsys-rep')],check=True,timeout=180)
    trace=trace_summary(root/'trace.sqlite')
    counters=parse_counter_log(root/'train.log')
    if len(counters)!=2 or counters[2]['emitted_epoch_label']!=2:
        raise ValueError('Missing epochs or incorrect epoch label')
    if counters[2]['boundary_count']!=19:raise ValueError('Unexpected transition count')
    if re.findall(r'Finished training epoch\s+(\d+)',(root/'train.log').read_text())!=['1','2']:
        raise ValueError('Incomplete training')
    if abs(trace['window_seconds']-counters[2]['official_epoch_ms']/1000)>.05:
        raise ValueError('Epoch NVTX range and native timer differ')
    result=dict(status='passed',purpose='capture plumbing only; not a performance measurement',
                build_manifest_sha256=hashlib.sha256((b/'build/artifacts.sha256').read_bytes()).hexdigest(),
                trace=trace,counter=counter_phases(counters[2]))
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
