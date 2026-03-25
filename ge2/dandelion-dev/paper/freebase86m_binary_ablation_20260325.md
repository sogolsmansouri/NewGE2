# Freebase86M Binary Optimization Summary

All runs below use the paper-style exact `paper-10k` evaluation unless noted otherwise.

| Run | Epochs | GPU-aware CUSTOM | Keep Hot | Bucket Streaming | Fast Map | CSR Gather | CSR Update | GPU Active Shuffle | Epoch Runtime | swap_rebuild_ms | MRR | Hits@10 | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `old_slow_20260323` | 10 | 0 | 0 | ? | ? | ? | ? | 0 | `242529ms` | `13.873` | `0.145355` | `0.290750` | pre-fix baseline |
| `old_fast_20260323` | 10 | 1 | ? | 1 | ? | ? | ? | 0 | `210110ms` | `15.750` | `0.144498` | `0.293000` | fast but bad quality |
| `bucket1_fast0_bad_3ep` | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | `199694ms` | `20.681` | `0.318159` | `0.446700` | bucket streaming hurts MRR |
| `bucket0_fast1_good_3ep` | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | `213655ms` | `13217.114` | `0.377921` | `0.513950` | good-quality non-bucket reference |
| `bucket1_fast0_fix_3ep` | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | `199121ms` | `11.028` | `0.317873` | `0.445850` | fixed stale bucket-layout bug, MRR still low |
| `std_dev_shuffle_3ep` | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | `200484ms` | `186.042` | `0.381545` | `0.510400` | current winning binary path on `main` |

## Current Best Binary Path

The current best quality-preserving path for Freebase86M binary LP is:

```bash
GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
GEGE_KEEP_STORAGE_HOT_BETWEEN_EPOCHS=1
GEGE_BUCKET_STREAMING_LP=0
GEGE_FAST_MAP_TENSORS=1
GEGE_CSR_GATHER=0
GEGE_CSR_UPDATE=0
GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1
```

## Interpretation

- `GEGE_BUCKET_STREAMING_LP=1` reduces epoch time but consistently lowers MRR.
- The main win on the standard path comes from `GEGE_GPU_ACTIVE_EDGE_SHUFFLE=1`, which collapses `swap_rebuild_ms` from about `13.2s` to `0.19s` per epoch while preserving model quality.
- This makes the non-bucket path nearly as fast as the bucket-streaming path, without the MRR drop.
