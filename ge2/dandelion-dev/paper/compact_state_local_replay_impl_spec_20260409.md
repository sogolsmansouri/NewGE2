# Current-Tree Fast-Path Reproduction Note

## Latest-Code Base-Stack Reproduction

This section records the exact recipe that lets the latest code reproduce the
old fast single-GPU Hybrid-Cover regime on the same machine.

The important discovery was that the missing piece was not Hybrid-Cover logic.
It was the build/runtime stack:

- build the latest tree against `base`, not `libkge-clean`
- force project CUDA sources on with `-DUSE_CUDA=ON -DUSE_OMP=ON`
- set `TORCH_CUDA_ARCH_LIST=8.6`
- run on a shell that can actually see the GPU
- keep `model_dir` / `checkpoint_dir` on a clean `/dev/shm`

The current working binary is:

- `/tmp/gege_current_base_build3/gege_train`

The working build command is:

```bash
source /home/smansou2/miniconda/etc/profile.d/conda.sh
conda activate base
export TORCH_CUDA_ARCH_LIST=8.6

cmake -S /home/smansou2/newCode/ge2/dandelion-dev/gege \
  -B /tmp/gege_current_base_build3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_OMP=ON \
  -DCMAKE_C_COMPILER=/home/smansou2/miniconda/bin/x86_64-conda-linux-gnu-cc \
  -DCMAKE_CXX_COMPILER=/home/smansou2/miniconda/bin/x86_64-conda-linux-gnu-c++ \
  -DTorch_DIR=/home/smansou2/miniconda/lib/python3.12/site-packages/torch/share/cmake/Torch \
  -DPYTHON_EXECUTABLE=/home/smansou2/miniconda/bin/python \
  -DPython3_EXECUTABLE=/home/smansou2/miniconda/bin/python \
  -DCMAKE_PREFIX_PATH=/home/smansou2/miniconda:/home/smansou2/miniconda/x86_64-conda-linux-gnu/sysroot/usr

cmake --build /tmp/gege_current_base_build3 --target gege_train -j 8
```

Sanity checks after build:

```bash
ldd /tmp/gege_current_base_build3/gege_train | head -n 20
nm -D /tmp/gege_current_base_build3/libge2.so | c++filt | rg 'selected_neg_scores_cuda_(forward|backward)'
nvidia-smi -L
```

Expected characteristics:

- `gege_train` links against the `base` Python/Torch stack
- `libcudart.so.12` comes from the `base` stack
- `selected_neg_scores_cuda_forward/backward` show up as defined (`T`)
- `nvidia-smi -L` reports the RTX 3090

### Shared Runtime Env

All successful reproductions below used the same base runtime shell:

```bash
source /home/smansou2/miniconda/etc/profile.d/conda.sh
conda activate base

export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
export LD_LIBRARY_PATH=/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cublas/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cusolver/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cusparse/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/cufft/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/curand/lib:/home/smansou2/miniconda/lib/python3.12/site-packages/nvidia/nccl/lib:

export GEGE_HYBRID_COVER=1
export GEGE_MEM_PARTITION_BUFFER_PINNED_HOST=1
export GEGE_FAST_MAP_TENSORS=1
export GEGE_PARTITION_BUFFER_LP_FAST_PATH=1
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
export GEGE_OPTIMIZED_CUSTOM_SCHEDULE=0
export GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
export GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
export GEGE_DEG_CHUNK_EXCLUSION=1
export GEGE_BUCKET_STREAMING_LP=0
export GEGE_CSR_GATHER=0
export GEGE_CSR_UPDATE=0
export GEGE_CSR_DEBUG=0
export GEGE_EMPTY_CACHE_AROUND_SWAP=0
export GEGE_SYNC_BEFORE_SWAP=0
export GEGE_MEM_SWAP_EVENT_SYNC=1
export GEGE_PROFILE_LOGICAL_LANE=0
export GEGE_FIXED_BUFFER_BITMAP_MAP=1
export GEGE_FIXED_BUFFER_BITMAP_REUSE_OUTPUTS=1
export GEGE_FIXED_BUFFER_MASKED_UPDATE=1
export GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1
export GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK=0
```

Dataset-specific overrides go on top of that base env.

### Freebase86M Reproduction

Archived fast config:

- `/home/smansou2/codex_runs/tmp/fb_8107cc36_8k_1024_r64_1e_20260417.yaml`

Dataset-specific env on top of the shared runtime env:

```bash
export GEGE_ACTIVE_BUCKET_SUBGRAPH=0
export GEGE_FRAME_CACHE_HIDDEN_FRAMES=0
export GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1
export GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1
export GEGE_FRAME_CACHE_STORAGE_FILTER=embeddings_only
export GEGE_FRAME_CACHE_TILE_ROWS=8192
export GEGE_FRAME_CACHE_HIDDEN_TILES=1024
export GEGE_FRAME_CACHE_RESERVED_STAGING_TILES=64
export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=0
export GEGE_UNIQUE_BACKEND=bitmap
export GEGE_UNIQUE_BITMAP_NUM_NODES=86054151
export GEGE_VIRTUAL_ACTIVE_EDGES=1
```

One-epoch current-tree repro:

```bash
RUN_ROOT=/dev/shm/smansou2_ge2/fb_current_baseenv_repro_1e_20260421
CONFIG=/tmp/fb_current_baseenv_repro_1e_20260421.yaml
LOG=/home/smansou2/codex_runs/exp_logs/fb_current_baseenv_repro_1e_20260421_train.log

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp /home/smansou2/codex_runs/tmp/fb_8107cc36_8k_1024_r64_1e_20260417.yaml "$CONFIG"
perl -0pi -e 's#model_dir:\\s*.*#model_dir: '"$RUN_ROOT"'#; s#checkpoint_dir:\\s*.*#checkpoint_dir: '"$RUN_ROOT"'#' "$CONFIG"

/tmp/gege_current_base_build3/gege_train "$CONFIG" | tee "$LOG"
```

Observed current-tree result:

- `Epoch Runtime: 186329ms`
- log:
  `/home/smansou2/codex_runs/exp_logs/fb_current_baseenv_repro_1e_20260421_train.log`

### Twitter16P Reproduction

Archived fast config:

- `/home/smansou2/codex_runs/tmp/twitter_main_hc_direct_2e_20260419.yaml`

Dataset-specific env on top of the shared runtime env:

```bash
export GEGE_UNIQUE_BACKEND=bitmap
export GEGE_UNIQUE_BITMAP_NUM_NODES=41652230
export GEGE_EMULATE_DOT_SINGLE_RELATION=1
export GEGE_FIXED_BUFFER_MANUAL_DOT_RNS=1
export GEGE_FRAME_CACHE_HIDDEN_FRAMES=1
export GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1
export GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1
```

Two-epoch current-tree repro command:

```bash
RUN_ROOT=/dev/shm/smansou2_ge2/twitter_current_baseenv_repro_2e
CONFIG=/tmp/twitter_current_baseenv_repro_2e.yaml
LOG=/home/smansou2/codex_runs/exp_logs/twitter_current_baseenv_repro_2e_train.log

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp /home/smansou2/codex_runs/tmp/twitter_main_hc_direct_2e_20260419.yaml "$CONFIG"
perl -0pi -e 's#model_dir:\\s*.*#model_dir: '"$RUN_ROOT"'#; s#checkpoint_dir:\\s*.*#checkpoint_dir: '"$RUN_ROOT"'#' "$CONFIG"

/tmp/gege_current_base_build3/gege_train "$CONFIG" | tee "$LOG"
```

Archived reference result on the old run:

- epoch 1: `198785ms`
- epoch 2: `197817ms`
- log:
  `/home/smansou2/codex_runs/exp_logs/twitter_8107cc36_hybridcover_on_2e_20260419_train.log`

Observed current-tree result on the same machine with the base-built binary:

- epoch 1: `201594ms`
- epoch 2: `198001ms`
- log:
  `/home/smansou2/codex_runs/exp_logs/twitter_current_baseenv_repro_2e_clean_20260421_train.log`

That means the latest code on the recovered fast stack is effectively back in
the old Twitter HC regime. The main difference is that epoch 1 is a few
seconds slower, while epoch 2 is within about `0.2s` of the archived April 19
steady-state run.

### Twitter16P Prefetch-True Reproduction

Use the same env as the Twitter16P reproduction above, but flip only:

```bash
prefetch: true
```

Two-epoch current-tree repro command:

```bash
RUN_ROOT=/dev/shm/smansou2_ge2/twitter_current_baseenv_prefetch_true_2e
CONFIG=/tmp/twitter_current_baseenv_prefetch_true_2e.yaml
LOG=/home/smansou2/codex_runs/exp_logs/twitter_current_baseenv_prefetch_true_2e_train.log

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp /home/smansou2/codex_runs/tmp/twitter_main_hc_direct_2e_20260419.yaml "$CONFIG"
perl -0pi -e 's#prefetch:\\s*false#prefetch: true#; s#model_dir:\\s*.*#model_dir: '"$RUN_ROOT"'#; s#checkpoint_dir:\\s*.*#checkpoint_dir: '"$RUN_ROOT"'#' "$CONFIG"

/tmp/gege_current_base_build3/gege_train "$CONFIG" | tee "$LOG"
```

Observed current-tree result:

- epoch 1: `189002ms`
- epoch 2: `185683ms`
- log:
  `/home/smansou2/codex_runs/exp_logs/twitter_current_baseenv_prefetch_true_2e_20260421_train.log`

This beats both:

- the archived non-prefetch HC steady-state result (`197817ms`)
- the earlier prefetch-true current-tree run on the other stack (`188624ms`)

### Operational Notes

- Do not run these on a shell where `nvidia-smi` fails or `torch.cuda` sees `0`
  devices.
- Keep `/dev/shm` clean. The Twitter run can fill tens of GiB.
- If `/home` is nearly full, `tee` may stop logging even if the training
  process itself continues.

# Twitter q4 Outer-Prefetch Reproduction Note

## Scope

This note records the single-GPU Twitter `16p / q=4` reproduction for the
outer-subgraph prefetch path on commit
`1921d1aa94ae8c61b47f07a27050b449a566a048`, plus the required runtime fix in
`GraphModelStorage`.

The purpose is to reproduce the post-fix result later without re-deriving the
steps from logs or chat history.

## Root Cause

With `storage.prefetch: true`, the initial prefetch thread was launched before
the current subgraph state was published to `current_subgraph_states_[device]`.

That created a race:

- `initializeInMemorySubGraph(...)` built `current_subgraph_state_`
- `getNextSubGraph()` immediately spawned a detached worker
- the worker entered `updateInMemorySubGraph_(...)`
- that worker read `current_subgraph_states_[device_idx]` as its source state
- but the main thread had not published it yet

The result was a broken/unstable prefetch path. In practice this showed up as:

- extremely slow startup on `prefetch=true`
- failure to reach epoch 1 in earlier runs

## Code Fix

Two changes were required.

1. Publish `current_subgraph_states_[device_idx]` before launching the initial
   background prefetch build.

2. Thread `device_idx` through `getNextSubGraph(...)` instead of hard-coding
   device `0`.

Relevant code:

- `gege/src/cpp/include/storage/graph_storage.h:117`
- `gege/src/cpp/src/storage/graph_storage.cpp:1170`
- `gege/src/cpp/src/storage/graph_storage.cpp:1260`
- `gege/src/cpp/src/storage/graph_storage.cpp:1344`

The fixed behavior is:

- current state is published first
- the prefetch worker reads a valid source snapshot
- foreground `outer-update` logs show `prefetch=true` and
  `subgraph_update_ms=0.000`

## Build

Use the local build against `libkge-clean`.

```bash
source /home/smansou2/miniconda/bin/activate libkge-clean
cmake --build /home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets --target gege_train -j 8
```

At run time, use:

```bash
export PYTHONPATH=/home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets:/home/smansou2/miniconda/envs/ge2/lib/python3.9/site-packages
export GEGE_TRAIN_BIN=/home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets/gege_train
```

One-command wrapper:

- `scripts/run_twitter16p_single_gpu_q4_outer_prefetch_repro.sh`

That script:

- activates `libkge-clean` unless `GEGE_SKIP_CONDA_ACTIVATE=1`
- points `GEGE_TRAIN_BIN` / `GEGE_EVAL_BIN` at the local build if present
- prepends the build dir and legacy Python deps to `PYTHONPATH`
- generates a temporary config with `prefetch: true`
- delegates to the existing q4 baseline launcher

Useful dry run:

```bash
PRINT_ONLY=1 /home/smansou2/newCode/ge2/dandelion-dev/scripts/run_twitter16p_single_gpu_q4_outer_prefetch_repro.sh
```

## Config

Base config:

- `gege/configs/twitter_16p_paper_opt.yaml`

That file keeps outer prefetch disabled:

- `prefetch: false` at `gege/configs/twitter_16p_paper_opt.yaml:58`

To test the fixed prefetch path, create a throwaway copy and flip only:

- `prefetch: true`
- `model_dir: <large writable path>`

Example:

```bash
cp /home/smansou2/newCode/ge2/dandelion-dev/gege/configs/twitter_16p_paper_opt.yaml \
   /home/smansou2/codex_runs/twitter_16p_paper_opt_prefetch_true.yaml

perl -0pi -e 's#prefetch:\\s*false#prefetch: true#; s#model_dir:\\s*.*#model_dir: /home/smansou2/codex_runs/twitter_q4_prefetch_true#' \
  /home/smansou2/codex_runs/twitter_16p_paper_opt_prefetch_true.yaml
```

## Baseline Reproduction

This is the non-prefetch q4 baseline using the existing launcher.

```bash
source /home/smansou2/miniconda/bin/activate libkge-clean
export PYTHONPATH=/home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets:/home/smansou2/miniconda/envs/ge2/lib/python3.9/site-packages
export GEGE_TRAIN_BIN=/home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets/gege_train
export GEGE_STATEFLOW_PLANNER=1
export CONFIG_SRC=/home/smansou2/newCode/ge2/dandelion-dev/gege/configs/twitter_16p_paper_opt.yaml
export RUN_NAME=twitter_q4_prefetch_baseline_2e_repro
export RUN_ROOT=/home/smansou2/codex_runs/twitter_q4_prefetch_baseline_2e_repro
export LOG_DIR=/home/smansou2/codex_runs/exp_logs
export EPOCHS=2
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_twitter16p_single_gpu_q4_prefetch_baseline.sh
```

Expected steady-state range from the clean April 20 runs:

- epoch 1: about `201.3s` to `201.6s`
- epoch 2: about `200.9s`

Reference logs:

- `/home/smansou2/codex_runs/exp_logs/twitter_q4_prefetch_baseline_2e_20260420_131935_train.log`
- `/home/smansou2/codex_runs/exp_logs/twitter_q4_prefetch_baseline_2e_20260420_133731_train.log`

## Prefetch-True Reproduction

Run the same launcher, but override the config copy:

```bash
source /home/smansou2/miniconda/bin/activate libkge-clean
export PYTHONPATH=/home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets:/home/smansou2/miniconda/envs/ge2/lib/python3.9/site-packages
export GEGE_TRAIN_BIN=/home/smansou2/newCode/ge2/dandelion-dev/gege/build-codex-libkge-clean-targets/gege_train
export GEGE_STATEFLOW_PLANNER=1
export CONFIG_SRC=/home/smansou2/codex_runs/twitter_16p_paper_opt_prefetch_true.yaml
export RUN_NAME=twitter_q4_prefetch_true_2e_repro
export RUN_ROOT=/home/smansou2/codex_runs/twitter_q4_prefetch_true_2e_repro
export LOG_DIR=/home/smansou2/codex_runs/exp_logs
export EPOCHS=2
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_twitter16p_single_gpu_q4_prefetch_baseline.sh
```

Equivalent wrapper:

```bash
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_twitter16p_single_gpu_q4_outer_prefetch_repro.sh
```

## Expected Results

### One-Epoch Validation

From the fixed `prefetch=true` validation run:

- init: `133.638s`
- epoch 1: `194011ms`

Reference log:

- `/home/smansou2/codex_runs/exp_logs/twitter_q4_prefetch_true_startup_20260420b_train.log`

Useful checkpoints in that log:

- `Initialization Complete: 133.638s`
- `################ Starting training epoch 1 ################`
- `Epoch Runtime: 194011ms`
- `swap_update_ms=13817.712`
- `map_lookup_ms=37616.226`

### Two-Epoch Validation

From the fixed `prefetch=true` two-epoch run:

- init: `43.54s`
- epoch 1: `193383ms`
- epoch 2: `188624ms`

The saved log for that run filled up after epoch 1 because the filesystem ran
out of space, but the terminal output completed epoch 2 successfully.

Epoch-2 comparison against the clean non-prefetch baselines:

- versus `200860ms`: `-12236ms` (`6.09%` faster)
- versus `200922ms`: `-12298ms` (`6.12%` faster)

Epoch-2 counter comparison versus the April 20 baseline logs:

- `swap_update_ms`: `12936.946` vs about `33115ms`
- `map_lookup_ms`: `37669.018` vs about `41524ms`

So the steady-state gain is real and is mainly coming from much lower
foreground `swap_update_ms`.

## What To Check In Logs

For a successful fixed prefetch run, you should see all of the following:

1. The planner still chooses the same family:
   `Stateflow planner selected family=HYBRID_COVER:legacy_rotated`

2. The run reaches epoch 1 cleanly.

3. `outer-update` lines show:
   - `prefetch=true`
   - `subgraph_update_ms=0.000`

4. Foreground `swap_update_ms` drops substantially relative to the
   non-prefetch baseline.

5. Epoch 2 should land around `188s` to `194s` on the tested machine,
   depending on startup / filesystem noise.

## Disk-Space Requirements

This path is sensitive to free space.

Observed during the April 20 runs:

- one `prefetch=true` 2-epoch run directory consumed about `21G`
- one startup-timing prefetch run directory consumed about `32G`

Practical guidance:

- leave at least `25G` free for a plain 2-epoch reproduction
- leave materially more if `GEGE_STARTUP_TIMING=1`
- avoid running this with a nearly full `/home`
- avoid `/dev/shm` if it is already near full

If the filesystem fills up, `tee` and the Gege logger may stop writing even if
the training process itself is still making progress.

## Optional Debug Mode

If the path regresses again, enable both:

```bash
export GEGE_STARTUP_TIMING=1
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING=1
export GEGE_PARTITION_BUFFER_PIPELINE_TIMING_MAX=64
```

That should show:

- `GraphModelStorage::load` timing
- `initializeInMemorySubGraph`
- `initializeBatches`
- `outer-update` lines with `prefetch=true`

## Bottom Line

The fixed outer-prefetch path is now:

- functionally correct for Twitter single-GPU q4
- faster in steady state than the old non-prefetch q4 baseline

The reproducible headline to remember is:

- baseline epoch 2: about `200.9s`
- fixed `prefetch=true` epoch 2: `188.6s`

That is roughly a `6%` epoch-time win on the tested setup.
