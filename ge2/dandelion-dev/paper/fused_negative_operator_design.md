# Fused Negative Operator Design

## Goal

Replace the current link-prediction negative path

1. candidate score build
2. row-wise negative selection
3. selected-negative score
4. backward scatter-add into negative embeddings

with a single operator contract that makes the dataflow explicit and minimizes
intermediate materialization.

## Current DistMult DNS/GAN Path

For `DISTMULT` in `DNS`/`GAN`, the hot path today is:

- build candidate scores for each chunk row against all sampled negatives
- select a subset of negatives per row
- score the selected negatives
- backpropagate into query embeddings and selected negative embeddings

The current optimized path already fuses the selected-negative score forward and
backward, but it still pays two important costs:

- a full candidate score tensor for selection
- atomic-heavy gradient accumulation into negative embeddings

## Operator Contract

Stage-2 target operator:

`FusedScoreSelectReduce(query_sel, neg_sel, query_score, neg_score) -> (selected_scores, selected_indices)`

Forward:

- input `query_sel`: selector queries shaped `[chunk, row, dim]`
- input `neg_sel`: selector negatives shaped `[chunk, neg, dim]`
- input `query_score`: scorer queries shaped `[chunk, row, dim]`
- input `neg_score`: scorer negatives shaped `[chunk, neg, dim]`
- output `selected_indices`: exact row-wise selected negative ids
- output `selected_scores`: scorer values for those selected ids

Backward:

- `d query_score` is a row-wise dense reduction over selected negatives
- `d neg_score` is a sparse reduction by selected negative id inside each chunk
- selector tensors receive no gradient in the current DNS/GAN execution because
  selection runs under `NoGradGuard`

## Stage-1 Implementation

Stage-1 keeps the current exact forward path and replaces only the negative-side
gradient update with a segmented reduction:

- keep existing selected-negative forward
- keep existing query-gradient kernel
- replace negative-gradient atomic accumulation with:
  1. flatten selected indices per chunk
  2. sort selected ids and carry pair order
  3. build CSR segment boundaries for identical negative ids
  4. reduce weighted query rows with `segment_sum_csr`
  5. scatter reduced rows into the dense `[chunk, neg, dim]` gradient tensor

This stage maps directly to the `SpMM` intuition:

- sparse structure: selected negative ids
- values: per-pair gradient scalars
- dense right-hand side: query embeddings
- result: negative embedding gradient rows

## Why Start Here

- exact semantics are easy to validate against the current path
- it attacks a real source of duplication: many rows select the same negatives
- it does not require changing sampler semantics or decoder math
- it is a clean stepping stone toward a fuller fused `SDDMM-Select-SpMM`
  operator

## Validation Requirements

Any guarded implementation must satisfy:

- exact equality of selected indices with the current selector
- `allclose` on selected scores
- `allclose` on gradients for query and negative embeddings
- identical loss values on the same batch

If any of those fail, the guarded path stays off and is removed rather than
kept as dormant complexity.
