# Padded DEG Local Filter

This note covers the CUDA DEG-local negative-filter path added by `Add padded CUDA DEG negative filter`.

## Flags

Enable the padded path:

```text
GEGE_DEG_LOCAL_FILTER_PADDED=1
```

Optional verifier:

```text
GEGE_DEG_LOCAL_FILTER_PADDED_VERIFY=1
```

The verifier compares the padded CUDA output against the existing reference implementation and throws if they differ.

## What It Changes

The padded path accelerates the DEG-local filter in `negative.cpp` by returning a padded `(row, col)` filter tensor directly from CUDA and then masking scores in-place on device.

This matters only when DEG-local filtering is actually materialized. On the current single-GPU LJ and Twitter paper stack, `GEGE_DEG_CHUNK_EXCLUSION=1` is left on by default, and that often prevents `deg_sample_indices` from being built for the current chunk. In that common case, the padded path is effectively bypassed.

## Validated Behavior

On the current LJ 30-epoch accuracy gate with the default single-GPU stack:

- baseline: `MRR 0.129557`, `Hits@10 0.310600`, `Hits@100 0.614000`, epochs 2-30 average `8396.5 ms`
- padded on: `MRR 0.129557`, `Hits@10 0.310700`, `Hits@100 0.614000`, epochs 2-30 average `8381.6 ms`

On the current Twitter 2-epoch single-GPU stack:

- baseline: epoch 1 `222703 ms`, epoch 2 `220032 ms`
- padded on: epoch 1 `222603 ms`, epoch 2 `219489 ms`

Conclusion: on the default paper stack, the padded path is correct but effectively neutral.

## When It Helps

The padded path does help when chunk exclusion is turned off and the DEG-local filter becomes active:

```text
GEGE_DEG_CHUNK_EXCLUSION=0
GEGE_DEG_LOCAL_FILTER_PADDED=1
```

Twitter single GPU, 1 epoch, same stack except `GEGE_DEG_CHUNK_EXCLUSION=0`:

- baseline: `227793 ms`
- padded on: `220366 ms`
- end-to-end gain: about `3.26%`

The improvement comes from the negative-filter subphase collapsing:

- `filter_ms`: `117481.149 ms -> 2479.968 ms`
- `negative_sample_ms`: `119981.556 ms -> 32307.375 ms`

Some of that gain is offset elsewhere in the pipeline:

- `map_lookup_ms`: `3272.576 ms -> 36887.352 ms`

So the padded path is a real optimization when the DEG-local filter is hot, but it is not a blanket epoch-time win for every configuration.
