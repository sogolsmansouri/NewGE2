# FB ComplEx p32 vs p24 (H=6) — replication guide

## Result being replicated (RunPod L40S, 2026-10-16)

| | p32 / 24 GiB | p24 / 32 GiB |
|---|---:|---:|
| Epoch 1 / epoch 2 | 95.32 s / **92.56 s** | 92.44 s / **89.86 s** |
| Swaps per epoch | 87 | 47 |
| Blocking swap time per epoch (`swap_update_ms`, epoch 2) | 19.05 s | 14.05 s |
| ... of which graph rebuild (edge materialize + remap) | ~96% | ~96% |
| ... of which parameter swap (`perform_swap`) | ~7 ms/swap | ~12 ms/swap |
| Preload not ready at transition / free-frame drains | 0 / 0 | 0 / 0 |
| GPU allocator peak (epoch 2) | 22.96 GB alloc / 23.58 GB reserved | 30.19 GB / 30.70 GB |

Fewer partitions was faster. Hidden-frame preload was fully hidden. What is left in each
swap is the synchronous graph rebuild, which is smaller per epoch at p24.

Raw evidence: `l40s_pod_run_20261016/` (summary.json, per-case config.yaml, flags.json, train.log.gz).

## Exact setup used

- Hardware: NVIDIA L40S 48 GB (350 W), 2× AMD EPYC 9374F, 2 NUMA nodes; GPU on NUMA node 1 (CPUs 32-63,96-127).
  CPUs were pinned to 0-15, which is **cross-NUMA**. The updated runner now pins to the GPU's own node.
- Host-memory limit (cgroup): 188 GB. `memory.peak` reached it (this includes page cache).
- Software: RunPod "Pytorch 2.8.0" template: Torch 2.8.0+cu128, CUDA 12.8 (nvcc 12.8.93), gcc 13.3, Python 3.12, driver 580.126.09.
- Source: `pod_bundle/gege` = `src/ours_memory_ablation/ge2/dandelion-dev/gege` (b1a92d1 + its working-tree changes),
  overlaid with the ARC-port `storage.h`, `storage.cpp` and `dataloader.cpp`, and the timer-instrumented `buffer.cpp` and
  `graph_storage.cpp` from `diagnostic_build/`. Built with `-DUSE_CUDA=ON`, Release.
  Built hashes (sha256 prefix): gege_train 6ea78c9cf1227cf2, libge2.so b0e6344ca0867a30, libalgorithm_policy.so b768cac830373433.
  NOTE: this is NOT the composed RunPod binary (e0261ab…) behind the historical 106.1 s vs 116.6 s numbers; that binary no longer exists.
- Data: ARC views `fb_single_gpu_memory_sweep_313488_20261016/data/p32` and `p24` (symlinks dereferenced), with
  304,727,650 train edges, 86,054,151 nodes and 14,824 relations. Offsets: 32² / 24² entries.
- Recipe: `fb_single_gpu_memory_sweep_313488_20261016/inputs` (config.yaml, flags.json, schedules/pNN/states.txt).
  ComplEx-100, Adagrad lr 0.1, 50,000-edge batches, 50 chunks × 1000 negatives, degree fraction 0.5, buffer_capacity 4,
  graph prefetch OFF, H=6 (3 preload + 3 stale frames), serialized hidden H2D, allocator cap = budget − 512 MiB,
  GEGE_DETERMINISTIC_ALGORITHMS=0, all swap/remap/transition timers on. 2 epochs, no eval.

## Host requirements (important for A6000 pods)

Host RAM must hold embeddings plus Adagrad state (86M × 100 × 4 B ≈ 34 GB each) plus edges, pinned staging and page cache.
Plan for **≥ 128 GB of real host RAM**, ideally ≥ 188 GB as here. Both the historical 251 GB pod and this 188 GB pod hit their cgroup cap.

If you rent a multi-GPU A6000 pod only to get enough RAM:
1. **Run one timing job at a time on the whole pod.** Jobs on the other GPUs share host RAM (one cgroup limit for the pod),
   page cache, memory bandwidth and CPUs. A neighbour job silently changes the timings. The runner now refuses to start
   if any GPU process exists (override with POD_ALLOW_BUSY=1, attribution only).
2. **Pin CPUs to the GPU's own NUMA node.** Check with `nvidia-smi topo -m` (CUDA_VISIBLE_DEVICES selects the GPU;
   the runner uses GPU0 of the visible set and its first 16 local CPUs, or set POD_CPUS=...).
3. Record `cat /sys/fs/cgroup/memory.max` and `memory.peak` (the runner writes host_before.json / host_after.json per case).

## Steps

On the A6000 pod (RunPod "Pytorch 2.8.0" template, /workspace volume ≥ 100 GB, your SSH key added):

```bash
# from eb2-3224-lin00: code + recipe
B=/home/smansou2/ge2_patch_work/tacc_baselines/results
rsync -a -e "ssh -p PORT -i ~/.ssh/id_ed25519" $B/fb_transition_fix_20261016/pod_bundle/ root@IP:/workspace/src/
rsync -a -e "ssh -p PORT -i ~/.ssh/id_ed25519" $B/fb_single_gpu_memory_sweep_313488_20261016/inputs root@IP:/workspace/recipe/
# data (needs an ARC compute allocation, e.g. salloc -p normal ... --no-shell; replace c12 by that node)
ssh arc.csc.ncsu.edu "ssh c12 'cd /mnt/beegfs/smansou2/fb_single_gpu_memory_sweep_313488_20261016/data && tar chf - p32 p24'" \
  | ssh -p PORT -i ~/.ssh/id_ed25519 root@IP "mkdir -p /workspace/data && tar xf - --no-same-owner -C /workspace/data"
# build + run both cases + summary (≈ 15 min build, ≈ 8–12 min per case)
ssh -p PORT -i ~/.ssh/id_ed25519 root@IP "nohup python3 /workspace/src/pod_run.py all > /workspace/pod_run.log 2>&1 &"
# results
ssh -p PORT -i ~/.ssh/id_ed25519 root@IP "cat /workspace/runs/summary.json"
```

`rsync` "chown … Invalid argument" warnings on /workspace are harmless.

## What to compare

- Epoch 2 time, p32 vs p24. Expect p24 ≤ p32.
- `outer_update_s.subgraph_update_ms` vs `perform_swap_ms`: graph rebuild vs parameter swap.
- `first-preload-join.wait_ms` and `not_ready_at_join`: should be ~0. If they are not, preload is exposed.
- `host_after.json`: memory.peak at memory.max means host-memory pressure, so rerun with more RAM before trusting the timings.
