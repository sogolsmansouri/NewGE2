#!/usr/bin/env python3
"""Simulate liveness-aware materialized negative caches for GE2.

This is a design simulator, not a training replacement.  It estimates whether a
windowed negative cache can reduce the unique rows/tiles dirtied inside a GE2
visible q=4 state.  The important question is not just sampler runtime; it is
whether the negative sampler stops making the whole resident state live/dirty.

The model intentionally keeps GE2's visible state unchanged.  For every state,
we estimate expected unique negative rows under:

  baseline: fresh chunk-shared negative rows each batch
  cache:    a materialized negative pool refreshed every W batches, optionally
            restricted to K active tiles, plus a fresh exploration fraction

The cache model reports a ratio against the baseline and full visible state.
Promising configurations should then be implemented behind a training flag and
validated with LJ accuracy before using them on Twitter/FB86M.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contrastive_tiled_cover_sim import (  # noqa: E402
    build_custom_template_states_q4,
    greedy_overlap_order,
)


def parse_csv_ints(value: str) -> List[int]:
    out: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(x) for x in chunk.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"Bad int range: {chunk}")
            out.extend(range(start, stop + 1, step))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def parse_csv_floats(value: str) -> List[float]:
    out: List[float] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    return sorted(set(out))


def parse_dataset_yaml(dataset_dir: Path) -> dict[str, int]:
    text = (dataset_dir / "dataset.yaml").read_text(encoding="utf-8")
    values: dict[str, int] = {}
    for key in ("num_nodes", "num_train", "num_edges"):
        match = re.search(rf"^{key}:\s*([0-9]+)", text, re.MULTILINE)
        if match:
            values[key] = int(match.group(1))
    if "num_nodes" not in values:
        raise ValueError(f"num_nodes not found in {dataset_dir / 'dataset.yaml'}")
    return values


def read_bucket_sizes(dataset_dir: Path, num_partitions: int) -> List[int]:
    path = dataset_dir / "edges" / "train_partition_offsets.txt"
    sizes = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = num_partitions * num_partitions
    if len(sizes) != expected:
        raise ValueError(f"{path} has {len(sizes)} buckets, expected {expected}")
    return sizes


def assign_bucket_edges_to_states(states: Sequence[Sequence[int]],
                                  bucket_sizes: Sequence[int],
                                  num_partitions: int) -> List[int]:
    state_sets = [set(state) for state in states]
    state_edges = [0 for _ in states]
    for src in range(num_partitions):
        for dst in range(num_partitions):
            size = bucket_sizes[src * num_partitions + dst]
            for state_idx, state in enumerate(state_sets):
                if src in state and dst in state:
                    state_edges[state_idx] += size
                    break
            else:
                raise ValueError(f"No state covers bucket ({src}, {dst})")
    return state_edges


def expected_uniform_unique(domain_rows: float, draws: float) -> float:
    if domain_rows <= 0 or draws <= 0:
        return 0.0
    # Stable approximation for domain_rows * (1 - (1 - 1/domain_rows)^draws).
    return domain_rows * (1.0 - math.exp(-draws / domain_rows))


def expected_uniform_tiles(num_tiles: float, draws: float) -> float:
    if num_tiles <= 0 or draws <= 0:
        return 0.0
    return num_tiles * (1.0 - math.exp(-draws / num_tiles))


def simulate_cache_unique_rows(num_tiles: int,
                               tile_rows: int,
                               windows: int,
                               pool_rows: int,
                               tile_budget: int,
                               fresh_draws: float,
                               fresh_domain: str,
                               rng: random.Random) -> tuple[float, float]:
    """Return expected unique rows and dirty tiles for the cache proposal."""

    if num_tiles <= 0:
        return 0.0, 0.0

    if tile_budget <= 0 or tile_budget >= num_tiles:
        total_cache_draws = float(windows * pool_rows)
        total_draws = total_cache_draws + fresh_draws
        domain_rows = float(num_tiles * tile_rows)
        return (
            expected_uniform_unique(domain_rows, total_draws),
            expected_uniform_tiles(float(num_tiles), total_draws),
        )

    survival = [1.0] * num_tiles
    if fresh_domain not in {"visible", "active"}:
        raise ValueError(f"Unsupported fresh domain: {fresh_domain}")

    fresh_draws_per_window = fresh_draws / float(windows) if windows > 0 else 0.0
    active_fresh_per_window = fresh_draws_per_window if fresh_domain == "active" else 0.0
    draws_per_selected_tile = (float(pool_rows) + active_fresh_per_window) / float(tile_budget)
    selected_survival = math.exp(-draws_per_selected_tile / float(tile_rows)) if draws_per_selected_tile > 0 else 1.0

    for _ in range(windows):
        for tile in rng.sample(range(num_tiles), tile_budget):
            survival[tile] *= selected_survival

    visible_fresh_draws = fresh_draws if fresh_domain == "visible" else 0.0
    fresh_per_tile = visible_fresh_draws / float(num_tiles)
    fresh_survival = math.exp(-fresh_per_tile / float(tile_rows)) if fresh_draws > 0 else 1.0

    unique_rows = 0.0
    dirty_tiles = 0.0
    for cache_survival in survival:
        untouched = cache_survival * fresh_survival
        if untouched < 0.999999:
            dirty_tiles += 1.0
        unique_rows += float(tile_rows) * (1.0 - untouched)
    return unique_rows, dirty_tiles


@dataclass
class SimRow:
    dataset: str
    num_partitions: int
    buffer_capacity: int
    states: int
    window_batches: int
    pool_rows: int
    tile_budget: int
    fresh_mix: float
    fresh_domain: str
    total_batches: int
    baseline_unique_rows: float
    baseline_dirty_tiles: float
    cache_unique_rows: float
    cache_dirty_tiles: float
    full_visible_state_rows: float
    unique_ratio_vs_baseline: float
    tile_ratio_vs_baseline: float
    unique_ratio_vs_full_state: float
    tile_ratio_vs_full_state: float


def run_sim(args: argparse.Namespace) -> List[SimRow]:
    dataset_dir = Path(args.dataset_dir)
    dataset = dataset_dir.name
    metadata = parse_dataset_yaml(dataset_dir)
    num_nodes = metadata["num_nodes"]
    bucket_sizes = read_bucket_sizes(dataset_dir, args.num_partitions)

    if args.buffer_capacity != 4:
        raise ValueError("This first simulator version models q=4 GE2 states only")
    states = build_custom_template_states_q4(args.num_partitions)
    if args.greedy_overlap_order:
        states = greedy_overlap_order(states)
    state_edges = assign_bucket_edges_to_states(states, bucket_sizes, args.num_partitions)

    partition_rows = math.ceil(num_nodes / args.num_partitions)
    visible_rows = float(args.buffer_capacity * partition_rows)
    visible_tiles = math.ceil(visible_rows / args.tile_rows)
    full_visible_state_tiles = float(visible_tiles)

    window_batches_values = parse_csv_ints(args.window_batches)
    pool_rows_values = parse_csv_ints(args.pool_rows)
    tile_budget_values = parse_csv_ints(args.tile_budget)
    fresh_mix_values = parse_csv_floats(args.fresh_mix)

    rows: List[SimRow] = []
    for window_batches in window_batches_values:
        for pool_rows in pool_rows_values:
            for tile_budget in tile_budget_values:
                for fresh_mix in fresh_mix_values:
                    rng = random.Random(args.seed)
                    baseline_unique = 0.0
                    baseline_tiles = 0.0
                    cache_unique = 0.0
                    cache_tiles = 0.0
                    total_batches = 0

                    for state_idx, edges in enumerate(state_edges):
                        batches = math.ceil(edges / args.batch_size)
                        if batches <= 0:
                            continue
                        total_batches += batches
                        baseline_draws = float(batches * args.negative_unique_rows_per_batch)
                        baseline_unique += expected_uniform_unique(visible_rows, baseline_draws)
                        baseline_tiles += expected_uniform_tiles(full_visible_state_tiles, baseline_draws)

                        windows = math.ceil(batches / window_batches)
                        fresh_draws = baseline_draws * fresh_mix
                        state_cache_unique, state_cache_tiles = simulate_cache_unique_rows(
                            visible_tiles,
                            args.tile_rows,
                            windows,
                            pool_rows,
                            tile_budget,
                            fresh_draws,
                            args.fresh_domain,
                            rng,
                        )
                        cache_unique += state_cache_unique
                        cache_tiles += state_cache_tiles

                    full_visible_rows_all_states = visible_rows * float(len(states))
                    full_visible_tiles_all_states = full_visible_state_tiles * float(len(states))
                    rows.append(SimRow(
                        dataset=dataset,
                        num_partitions=args.num_partitions,
                        buffer_capacity=args.buffer_capacity,
                        states=len(states),
                        window_batches=window_batches,
                        pool_rows=pool_rows,
                        tile_budget=tile_budget,
                        fresh_mix=fresh_mix,
                        fresh_domain=args.fresh_domain,
                        total_batches=total_batches,
                        baseline_unique_rows=baseline_unique,
                        baseline_dirty_tiles=baseline_tiles,
                        cache_unique_rows=cache_unique,
                        cache_dirty_tiles=cache_tiles,
                        full_visible_state_rows=full_visible_rows_all_states,
                        unique_ratio_vs_baseline=cache_unique / baseline_unique if baseline_unique > 0 else 0.0,
                        tile_ratio_vs_baseline=cache_tiles / baseline_tiles if baseline_tiles > 0 else 0.0,
                        unique_ratio_vs_full_state=cache_unique / full_visible_rows_all_states if full_visible_rows_all_states > 0 else 0.0,
                        tile_ratio_vs_full_state=cache_tiles / full_visible_tiles_all_states if full_visible_tiles_all_states > 0 else 0.0,
                    ))
    return rows


def print_tsv(rows: Iterable[SimRow]) -> None:
    headers = [
        "dataset",
        "p",
        "q",
        "states",
        "window_batches",
        "pool_rows",
        "tile_budget",
        "fresh_mix",
        "fresh_domain",
        "total_batches",
        "baseline_unique_rows_m",
        "cache_unique_rows_m",
        "unique_ratio_vs_baseline",
        "unique_ratio_vs_full_state",
        "baseline_dirty_tiles_k",
        "cache_dirty_tiles_k",
        "tile_ratio_vs_baseline",
        "tile_ratio_vs_full_state",
    ]
    print("\t".join(headers))
    for row in rows:
        values = [
            row.dataset,
            str(row.num_partitions),
            str(row.buffer_capacity),
            str(row.states),
            str(row.window_batches),
            str(row.pool_rows),
            str(row.tile_budget),
            f"{row.fresh_mix:.3f}",
            row.fresh_domain,
            str(row.total_batches),
            f"{row.baseline_unique_rows / 1_000_000.0:.3f}",
            f"{row.cache_unique_rows / 1_000_000.0:.3f}",
            f"{row.unique_ratio_vs_baseline:.4f}",
            f"{row.unique_ratio_vs_full_state:.4f}",
            f"{row.baseline_dirty_tiles / 1_000.0:.3f}",
            f"{row.cache_dirty_tiles / 1_000.0:.3f}",
            f"{row.tile_ratio_vs_baseline:.4f}",
            f"{row.tile_ratio_vs_full_state:.4f}",
        ]
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate GE2 liveness-aware negative cache designs.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-partitions", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--negative-unique-rows-per-batch", type=int, default=50000,
                        help="Approximate unique negative entity rows touched by one current batch.")
    parser.add_argument("--tile-rows", type=int, default=4096)
    parser.add_argument("--window-batches", default="2,4,8,16,32")
    parser.add_argument("--pool-rows", default="4096,8192,16384,32768,65536")
    parser.add_argument("--tile-budget", default="0,4,8,16,32",
                        help="0 means unrestricted cache pool over the whole visible state.")
    parser.add_argument("--fresh-mix", default="0,0.05,0.1,0.25",
                        help="Fraction of current fresh negative rows retained per batch.")
    parser.add_argument("--fresh-domain", choices=("visible", "active"), default="visible",
                        help="visible keeps fresh exploration over the full resident state; active confines it to scheduled tiles.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--greedy-overlap-order", action="store_true")
    args = parser.parse_args()
    print_tsv(run_sim(args))


if __name__ == "__main__":
    main()
