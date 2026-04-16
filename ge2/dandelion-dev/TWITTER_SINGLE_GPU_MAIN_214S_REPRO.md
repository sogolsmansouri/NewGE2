# Twitter Single-GPU 214s Repro On `main`

This documents the current verified Twitter single-GPU q4/frame-cache path on:

- commit: `947234e3e2578f2c2dfe120eea1ffb15dafecadc`
- branch: `main`

## Verified Result

Fresh verification on `2026-04-16` with the launcher below produced:

- epoch 1: `214179 ms`
- epoch 2: `213285 ms`

Log:

- `~/codex_runs/exp_logs/twitter_q4_main_recheck_2e_20260416_train.log`

Important steady-state counters from epoch 2:

- `map_lookup_ms=43010.244`
- `swap_update_ms=43917.694`

## Why Earlier Attempts Missed

The earlier miss on `main` was not a stable code regression. It came from reproduction drift:

1. The exact q4/frame-cache launcher from the older `b2cc2be` path was not yet present on `main`.
2. One earlier `main` rerun hit a one-off epoch-1 compute spike in logical lane 0, state 0, which pushed epoch 1 to `279369 ms`.
3. Using a disk-backed run root also makes the benchmark more fragile because Twitter writes large transient files and low free space can distort timings.

The current `main` tree can reproduce the `~214s` class when run with the exact q4 launcher and a RAM-backed run root.

## Reproduction

Build:

```bash
cd /home/smansou2/newCode/ge2/dandelion-dev
cmake --build build_ge2env_ge2py39 --target gege_train -j4
```

Run:

```bash
mkdir -p /dev/shm/smansou2_ge2 /home/smansou2/codex_runs/exp_logs

RUN_NAME=twitter_q4_main_recheck_2e_20260416 \
EPOCHS=2 \
RUN_EVAL=0 \
RUN_ROOT=/dev/shm/smansou2_ge2/twitter_q4_main_recheck_2e_20260416 \
LOG_DIR=/home/smansou2/codex_runs/exp_logs \
GEGE_TRAIN_BIN=/home/smansou2/newCode/ge2/dandelion-dev/build_ge2env_ge2py39/gege/gege_train \
/home/smansou2/newCode/ge2/dandelion-dev/scripts/run_twitter16p_single_gpu_q4_prefetch_baseline.sh
```

## Required Notes

- Use `scripts/run_twitter16p_single_gpu_q4_prefetch_baseline.sh`. It carries the exact env flags for the q4/frame-cache path.
- Prefer `RUN_ROOT` under `/dev/shm/smansou2_ge2` to avoid disk-pressure artifacts.
- The benchmark should be judged on epoch 2+ if epoch 1 is an obvious outlier.
- If you see a bad epoch 1, check whether `state_pos 0` has an abnormally large `compute_ms`. That was the signature of the earlier false negative.
