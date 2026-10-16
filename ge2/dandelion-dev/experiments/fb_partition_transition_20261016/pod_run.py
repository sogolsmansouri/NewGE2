#!/usr/bin/env python3
"""Pod attribution: build, then FB ComplEx p32/24GiB vs p24/32GiB, H=6, 2 epochs, all swap timers on.

Usage on the pod:  python3 /workspace/src/pod_run.py [build|run|summary|all]
Mirrors ARC run_case.py prepare() (same config edits and GEGE_* flags), without ARC-only checks.
"""
import json, os, re, subprocess, sys, collections
from pathlib import Path
import yaml

W = Path('/workspace'); SRC = W / 'src'; BUILD = W / 'build'; RECIPE = W / 'recipe'; RUNS = W / 'runs'
CASES = [('p32_b24_h6', 32, 24), ('p24_b32_h6', 24, 32)]
EPOCHS = 2


def sh(cmd, **kw):
    print('+', cmd, flush=True); subprocess.run(cmd, shell=True, check=True, **kw)


def build():
    import torch
    tp = Path(torch.__file__).parent
    arch = '%d%d' % torch.cuda.get_device_capability(0)  # 86 on A6000, 89 on L40S
    BUILD.mkdir(exist_ok=True)
    # CMakeLists references test sources in ../scripts that are not shipped; gege_train does not use them.
    for stub in ('check_dense_local_steps', 'check_frame_transition_state', 'check_parallel_numerics', 'test_dense_local_steps',
                 'test_order_window', 'test_parallel_checkpoint_resume', 'test_resident_negative_domain'):
        f = SRC / 'scripts' / f'{stub}.cpp'; f.parent.mkdir(exist_ok=True)
        if not f.exists(): f.write_text('int main(){return 0;}\n')
    # gege_train embeds Python and imports the `gege` config package (needs omegaconf).
    (W / 'pylib').mkdir(exist_ok=True)
    if not (W / 'pylib/gege').exists(): (W / 'pylib/gege').symlink_to(SRC / 'gege/src/python')
    sh(f'{sys.executable} -m pip install -q --break-system-packages omegaconf pyyaml || {sys.executable} -m pip install -q omegaconf pyyaml')
    sh(f'cmake -S {SRC}/gege -B {BUILD} -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} '
       f'-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DPython3_EXECUTABLE={sys.executable} '
       f'-DPython_EXECUTABLE={sys.executable} > {BUILD}/configure.log 2>&1 || (tail -40 {BUILD}/configure.log; false)')
    sh(f'cmake --build {BUILD} -j 64 --target gege_train > {BUILD}/build.log 2>&1 || (grep -m20 -E "error|Error" {BUILD}/build.log; false)')
    sh(f'g++ -O2 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())} {SRC}/algorithm_policy.cpp '
       f'-I{tp}/include -L{tp}/lib -lc10 -ltorch_cpu -Wl,-rpath,{tp}/lib -o {BUILD}/libalgorithm_policy.so')
    print('BUILD_OK', flush=True)


def prepare(name, p, budget):
    case = RUNS / name; case.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((RECIPE / 'inputs/config.yaml').read_text())
    data_dir = W / 'data' / f'p{p}'
    data = yaml.safe_load((data_dir / 'dataset.yaml').read_text())
    data['dataset_dir'] = str(data_dir) + '/'
    # The loader re-reads dataset.yaml from the data folder, so its dataset_dir must point here too.
    (data_dir / 'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    counts = [int(v) for v in (data_dir / 'edges/train_partition_offsets.txt').read_text().split()]
    assert len(counts) == p * p and sum(counts) == data['num_train']
    cfg['storage'].update(dataset=data, device_ids=[0], prefetch=False, model_dir=str(case / 'model') + '/')
    cfg['storage']['embeddings']['options'].update(num_partitions=p, buffer_capacity=4, prefetching=False)
    cfg['training'].update(batch_size=50000, num_epochs=EPOCHS, save_model=False, logical_active_devices=1, dense_sync_batches=1)
    cfg['evaluation'].update(epochs_per_eval=0, checkpoint_dir=str(case / 'model') + '/')
    phys = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).split()[0])
    flags = json.loads((RECIPE / 'inputs/flags.json').read_text())
    flags.update(GEGE_FRAME_CACHE_HIDDEN_FRAMES='6', GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES='3', GEGE_FRAME_CACHE_MAX_STALE_BACKLOG='3',
        GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD='1', GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK='1', GEGE_FRAME_CACHE_SERIALIZE_ADMIT_H2D='1',
        GEGE_BOUNDED_Q4_OPTIMAL88='0', GEGE_BOUNDED_STATE_ORDER_FILE=str(RECIPE / 'inputs/schedules' / f'p{p}' / 'states.txt'),
        GEGE_UNIQUE_BITMAP_NUM_NODES=str(data['num_nodes']), GEGE_CUDA_ALLOCATOR_LIMIT_MIB=str(min(budget * 1024, phys) - 512),
        GEGE_CUDA_MEMORY_STATS='1', GEGE_DETERMINISTIC_ALGORITHMS='0', GEGE_TRAINING_INPUT_AUDIT='0',
        GEGE_PARTITION_BUFFER_LP_FAST_PATH_VALIDATE='0', GEGE_PREPARED_BATCH_PIPELINE='0', GEGE_MULTI_GPU_PREPARED_BATCH_PIPELINE='0',
        GEGE_FUSED_COMPLEX_ELEMENTWISE='0', GEGE_TRANSITION_DIAGNOSTIC='1', GEGE_PARTITION_BUFFER_PIPELINE_TIMING='1',
        GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX='1000000', GEGE_PARTITION_BUFFER_SWAP_TIMING='1',
        GEGE_PARTITION_BUFFER_REMAP_BREAKDOWN_TIMING='1')
    (case / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (case / 'flags.json').write_text(json.dumps(flags, indent=2))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GEGE_', 'LD_PRELOAD'))}
    env.update(flags, LD_PRELOAD=str(BUILD / 'libalgorithm_policy.so'), GEGE_NO_BINDINGS='1', PYTHONPATH='/workspace/pylib', OMP_NUM_THREADS='16',
               MKL_NUM_THREADS='16', NVIDIA_TF32_OVERRIDE='0', CUDA_VISIBLE_DEVICES='0')
    return case, env


def gpu_local_cpus(n=16):
    """First n CPUs on the GPU's NUMA node (nvidia-smi topo 'CPU Affinity'), e.g. '32-63,96-127' -> 32..47."""
    topo = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    row = next(l for l in topo.splitlines() if l.startswith('GPU0'))
    span = re.search(r'\s(\d+-\d+(?:,\d+-\d+)*)\s', row).group(1)
    cpus = [c for part in span.split(',') for c in range(int(part.split('-')[0]), int(part.split('-')[1]) + 1)]
    return ','.join(map(str, cpus[:n]))


def snapshot(case, tag):
    """Host-memory headroom and co-tenant GPU processes, before and after each case."""
    def read(p):
        try: return Path(p).read_text().strip()
        except OSError: return None
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
    (case / f'host_{tag}.json').write_text(json.dumps(dict(
        cgroup_memory_max=read('/sys/fs/cgroup/memory.max'), cgroup_memory_peak=read('/sys/fs/cgroup/memory.peak'),
        cgroup_memory_current=read('/sys/fs/cgroup/memory.current'), gpu_apps=apps.splitlines(),
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu,power.limit',
                                      '--format=csv,noheader'], text=True).splitlines()), indent=1))
    return apps.strip()


def run():
    cpus = os.environ.get('POD_CPUS') or gpu_local_cpus()
    for name, p, budget in CASES:
        case, env = prepare(name, p, budget)
        others = snapshot(case, 'before')
        # A co-running job on the same host shares host RAM, page cache, CPU and PCIe: refuse to time beside it.
        if others and not os.environ.get('POD_ALLOW_BUSY'):
            raise SystemExit(f'GPU processes already running, not timing {name}:\n{others}')
        print(f'== running {name} (CPUs {cpus}, {EPOCHS} epochs)', flush=True)
        with open(case / 'train.log', 'w') as log:
            rc = subprocess.call(['taskset', '-c', cpus, str(BUILD / 'gege_train'), str(case / 'config.yaml')],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        snapshot(case, 'after')
        print(name, 'rc', rc, flush=True)


def summarize(case):
    t = (case / 'train.log').read_text(errors='replace')
    out = dict(case=case.name, epochs_s=[int(x) / 1000 for x in re.findall(r'Epoch Runtime: (\d+)ms', t)])
    def acc(pattern):
        rows = re.findall(pattern, t); c = collections.Counter()
        for r in rows:
            for k, v in re.findall(r'(\w+_ms)=([\d.]+)', r): c[k] += float(v)
        return dict(n=len(rows), **{k: round(v / 1000, 2) for k, v in c.items()})
    out['outer_update_s'] = acc(r'\[outer-update \d+\][^\n]*')
    out['graph_update_s'] = acc(r'\[partition-buffer-pipeline\]\[update \d+\][^\n]*')
    for tag in ('first-preload-join', 'matching-writeback', 'free-frame-drain', 'hidden-h2d'):
        out[tag] = acc(r'\[transition-diagnostic\]\[%s\][^\n]*' % tag)
    out['not_ready_at_join'] = len(re.findall(r'first-preload-join\][^\n]*ready_at_entry=false', t))
    out['perf_epoch2'] = [l[l.find('[perf]'):][:600] for l in t.splitlines()
                          if re.search(r'\[perf\]\[epoch 2\](\[batch_fetch\]| swap_count|\[frame_cache\] )', l)]
    out['memory'] = re.findall(r'\[memory-budget-peak\] epoch=2[^\n]*', t)[-1:]
    return out


def summary():
    res = [summarize(RUNS / n) for n, _, _ in CASES if (RUNS / n / 'train.log').exists()]
    text = json.dumps(res, indent=1); (RUNS / 'summary.json').write_text(text); print(text)


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if step in ('build', 'all'): build()
    if step in ('run', 'all'): run()
    if step in ('summary', 'run', 'all'): summary()
