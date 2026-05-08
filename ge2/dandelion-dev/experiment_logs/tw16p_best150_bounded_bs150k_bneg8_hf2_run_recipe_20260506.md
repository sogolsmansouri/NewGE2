# TW16P single-GPU best-known epoch-time recipe

Date recorded: 2026-05-06

Commit:

```text
fed9d052c8d0b31004a4fbda000f5cc7774a5e62
```

Purpose:

```text
Reproduce the best-known Twitter16P single-GPU epoch-time path, around 150-160 s per epoch.
The best prior log found locally is 156.400 s for one epoch.
```

Reference prior run:

```text
/home/smansou2/newCode/ge2/dandelion-dev/experiment_logs/tw16p_latest_bounded_feasible_hf2_bs150k_20260504_173451_train.log
Epoch Runtime: 156400ms
Edges per Second: 19513366
```

Current 10-epoch run:

```text
tmux session: tw16p_best150_231842
run_name: tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842
run_root: /dev/shm/smansou2_ge2/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842
train_log: /home/smansou2/codex_runs/exp_logs/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842_train.log
driver_log: /home/smansou2/codex_runs/exp_logs/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842_driver.log
effective_config: /home/smansou2/codex_runs/configs/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842.effective.yaml
```

Important: the script copies `GEGE_CONFIG_SRC` to a temp YAML and patches it at runtime. The
persisted `effective_config` above is a copy of that runtime-patched temp YAML. It contains:

```text
training.batch_size: 150000
training.num_epochs: 10
training.save_model: true
storage.model_dir: /dev/shm/smansou2_ge2/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842
evaluation.checkpoint_dir: /dev/shm/smansou2_ge2/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842
```

Exact command pattern:

```sh
cd /home/smansou2/newCode_fed9d052/ge2/dandelion-dev

RUN_NAME="tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_$(date +%Y%m%d_%H%M%S)"

env \
  GEGE_TRAIN_BIN=/home/smansou2/newCode_fed9d052/ge2/dandelion-dev/gege/build-cuda-check/gege_train \
  GEGE_EVAL_BIN=/home/smansou2/newCode_fed9d052/ge2/dandelion-dev/gege/build-cuda-check/gege_eval \
  GEGE_CONFIG_SRC=/home/smansou2/codex_runs/configs/tw16p_best150_bounded_bs150k_bneg8_hf2_fed9_10e_20260506_231842.effective.yaml \
  GEGE_RUN_NAME="$RUN_NAME" \
  GEGE_EPOCHS=10 \
  GEGE_RUN_EVAL=0 \
  GEGE_SAVE_MODEL=true \
  CUDA_VISIBLE_DEVICES=0 \
  GEGE_HYBRID_COVER=0 \
  GEGE_STATEFLOW_PLANNER=0 \
  GEGE_BOUNDED_GREEDY_COVER_Q4=1 \
  GEGE_BOUNDED_GREEDY_COVER_REVERSE=0 \
  GEGE_STATEFLOW_MAX_ADMITS=3 \
  GEGE_MINMAX_BUCKET_ASSIGNMENT=1 \
  GEGE_BATCHED_NEGATIVE_PLAN_BATCHES=8 \
  GEGE_FRAME_CACHE_HIDDEN_FRAMES=2 \
  GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1 \
  GEGE_FRAME_CACHE_PRIORITIZED_WRITEBACK=1 \
  GEGE_FRAME_CACHE_SERIALIZE_ADMIT_H2D=1 \
  GEGE_PREPARED_BATCH_PIPELINE=0 \
  GEGE_PREFETCH_PREPARE_NEXT_PARTITION=0 \
  GEGE_FULL_PIPELINE_PREFETCH=0 \
  GEGE_PARTITION_BUFFER_PIPELINE_TIMING=0 \
  GEGE_BUCKET_STREAMING_LP=1 \
  bash scripts/run_epoch_time_repro_20260424.sh twitter-2e \
  2>&1 | tee "/home/smansou2/codex_runs/exp_logs/${RUN_NAME}_driver.log"
```

Expected scheduler fingerprint:

```text
Min-max bucket assignment moved 4 buckets max_load 74714934 -> 70599092
Generating BOUNDED_GREEDY_COVER_Q4 Ordering states=24 transitions=23 total_buckets=256 max_admits=3 transition_admits=52
Using bounded GREEDY_COVER q4 ordering for CUSTOM schedule with 1 active device(s)
Using bucket-streaming LP path for in-memory partition-buffer batches
```

Flags that matter for the 150-160 s path:

```text
training.batch_size=150000
GEGE_BATCHED_NEGATIVE_PLAN_BATCHES=8
GEGE_BOUNDED_GREEDY_COVER_Q4=1
GEGE_MINMAX_BUCKET_ASSIGNMENT=1
GEGE_HYBRID_COVER=0
GEGE_STATEFLOW_PLANNER=0
GEGE_FRAME_CACHE_HIDDEN_FRAMES=2
GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1
GEGE_FRAME_CACHE_PRIORITIZED_WRITEBACK=1
GEGE_FRAME_CACHE_SERIALIZE_ADMIT_H2D=1
GEGE_BUCKET_STREAMING_LP=1
GEGE_PREPARED_BATCH_PIPELINE=0
GEGE_PREFETCH_PREPARE_NEXT_PARTITION=0
GEGE_BATCHED_NEGATIVE_PLAN_BATCHES=8
```

Do not confuse this with the slower frozen `epoch168` path. That path used
`GEGE_BATCHED_NEGATIVE_PLAN_BATCHES=0` and selected `HYBRID_COVER:legacy_rotated`;
the observed epoch 1 runtime in the aborted 2026-05-06 rerun was 171.976 s.
