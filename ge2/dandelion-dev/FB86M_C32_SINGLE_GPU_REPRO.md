# Freebase86M c32 Single-GPU Reproduction

This records the ARC c32 run used for the single-GPU Freebase86M p32 q4
fast-path timing and paper-10k evaluation.

## Verified Code

- branch: `codex/stateflow-repro-20260421`
- c32 repro-wrapper fix commit: `de5cae0`
- training binary used on c32:
  `/mnt/local/smansou2/code_mirror/dandelion-dev_samecode_20260521_165108/gege/build-cuda-samecode-arc-cu129-sysgcc-sm86/gege_train`
- eval binary used on c32:
  `/mnt/local/smansou2/code_mirror/dandelion-dev_samecode_20260521_165108/gege/build-cuda-samecode-arc-cu129-sysgcc-sm86/gege_eval`

Source-code hash check between this checkout and the c32 local mirror matched
for the GE2 C++/CUDA files. The c32 build/runtime environment is different
from the local workstation environment.

## c32 Runtime Paths

```bash
ROOT=/mnt/local/smansou2/code_mirror/dandelion-dev_samecode_20260521_165108
PY_ENV=/mnt/local/smansou2/ge2_py312_torch_mirror
TRAIN_DS=/mnt/local/smansou2/datasets/freebase86m_32p
RAW_EVAL_DS=/mnt/local/smansou2/datasets/freebase86m_16p_paper_10k_eval
MERGED_EVAL_DS=/mnt/local/smansou2/datasets/freebase86m_32p_paper_10k_eval_merged
```

The merged eval dataset is intentional: the paper-10k eval split provides the
10k test edges, while exact filtered eval still needs the train/validation graph
edges for filtering and sorting.

## Full 10-Epoch Train + Eval

Run inside an ARC allocation on c32. Use `tmux` so the run survives VPN/SSH
drops.

```bash
ssh -J arc c32
cd /mnt/local/smansou2/code_mirror/dandelion-dev_samecode_20260521_165108

tmux new-session -d -s fb86m_c32_10e_eval_rerun \
  'GEGE_RUN_NAME=fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_rerun_$(date +%Y%m%d_%H%M%S) \
   bash scripts/run_fb86m_c32_single_gpu_fastpath_10e_eval.sh'
```

Attach/check:

```bash
tmux attach -t fb86m_c32_10e_eval_rerun
pgrep -af 'gege_train|gege_eval'
nvidia-smi
```

The wrapper records logs under:

```bash
/mnt/local/smansou2/ge2_logs/<run_name>/
```

## Eval-Only From Existing Checkpoint

Use this when training already completed and the checkpoint is still in
`/dev/shm`. This avoids starting from scratch.

Current checkpoint from the recorded run:

```bash
CKPT=/dev/shm/smansou2_ge2/fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154
```

Eval-only command:

```bash
ssh -J arc c32
cd /mnt/local/smansou2/code_mirror/dandelion-dev_samecode_20260521_165108

export PATH=/mnt/local/smansou2/ge2_py312_torch_mirror/bin:$PATH
export LD_LIBRARY_PATH=/mnt/local/smansou2/ge2_py312_torch_mirror/lib:${LD_LIBRARY_PATH:-}
export GEGE_NO_BINDINGS=1
export PYTHONPATH=$PWD/build_ge2env_ge2py39/package/build/lib:${PYTHONPATH:-}
export GEGE_EVAL_BIN=$PWD/gege/build-cuda-samecode-arc-cu129-sysgcc-sm86/gege_eval

bash scripts/run_freebase86m_eval.sh \
  --checkpoint-dir "$CKPT" \
  --dataset-mode paper-10k \
  --config-out /mnt/local/smansou2/ge2_logs/fb86m_eval_retry.yaml \
  --log-out /mnt/local/smansou2/ge2_logs/fb86m_eval_retry.log
```

For long eval runs, launch the eval-only command inside `tmux`:

```bash
tmux new-session -d -s fb86m_c32_eval_retry \
  'cd /mnt/local/smansou2/code_mirror/dandelion-dev_samecode_20260521_165108 && \
   export PATH=/mnt/local/smansou2/ge2_py312_torch_mirror/bin:$PATH && \
   export LD_LIBRARY_PATH=/mnt/local/smansou2/ge2_py312_torch_mirror/lib:${LD_LIBRARY_PATH:-} && \
   export GEGE_NO_BINDINGS=1 && \
   export PYTHONPATH=$PWD/build_ge2env_ge2py39/package/build/lib:${PYTHONPATH:-} && \
   export GEGE_EVAL_BIN=$PWD/gege/build-cuda-samecode-arc-cu129-sysgcc-sm86/gege_eval && \
   bash scripts/run_freebase86m_eval.sh \
     --checkpoint-dir /dev/shm/smansou2_ge2/fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154 \
     --dataset-mode paper-10k \
     --config-out /mnt/local/smansou2/ge2_logs/fb86m_eval_retry.yaml \
     --log-out /mnt/local/smansou2/ge2_logs/fb86m_eval_retry.log'
```

## Recorded Training Result

Run:

```text
fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154
```

Epoch runtimes:

```text
epoch 1: 132.412s
epoch 2: 129.212s
epoch 3: 129.181s
epoch 4: 129.246s
epoch 5: 129.348s
epoch 6: 129.312s
epoch 7: 129.400s
epoch 8: 129.200s
epoch 9: 129.235s
epoch 10: 129.264s
avg: 129.581s
avg excluding epoch 1: 129.266s
```

Train log:

```bash
/mnt/local/smansou2/ge2_logs/fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154/fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154_train.log
```

CSV record:

```bash
/home/smansou2/fb86m_training_eval_results.csv
```

## Recorded Eval Result

Eval retry:

```text
fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154_eval_retry_merged_20260522_113420
```

Paper-10k exact filtered result:

```text
Link Prediction: 20000 edges evaluated
Mean Rank: 3562151.470450
MRR: 0.406830
Hits@1: 0.336400
Hits@3: 0.443900
Hits@5: 0.488600
Hits@10: 0.541750
Hits@50: 0.637800
Hits@100: 0.666650
```

Eval log:

```bash
/mnt/local/smansou2/ge2_logs/fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154/fb86m_p32_q4_1gpu_c32_fastpath_10e_eval_55a63a7_20260521_182154_eval_retry_merged_20260522_113420.log
```

## Notes

- The checkpoint in `/dev/shm` is volatile. If the allocation or node state is
  lost, rerun the full train wrapper.
- BeegFS was unreliable during this run, so the c32 execution used the
  `/mnt/local` code mirror and staged datasets.
- The c32 source mirror matched the committed GE2 source code, but the binary
  was built with the c32 runtime toolchain and `sm_86`.
