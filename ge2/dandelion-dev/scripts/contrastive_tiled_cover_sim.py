#!/usr/bin/env python3
"""Estimate memory/communication tradeoffs for Contrastive Tiled COVER.

This is an offline design simulator, not a runtime predictor.  It is meant to
reject bad (p, q, negative tile) choices before changing the training loop.

The simulator reports:
  * COVER-like state counts.
  * Per-lane swap traffic for G GPUs.
  * HBM resident and staging pressure.
  * Whether a contrastive microtask leaves at least one free slot for prefetch.
  * A simple swap-only epoch estimate when the measured baseline is provided.

For q=4 and power-of-two p, the script can use the same deterministic template
shape as GE2's CUSTOM ordering, without the runtime random state permutation.
For other q values, it reports lower-bound state counts and an explicit retained
partition assumption.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Set, Tuple


GIB = 1024 ** 3
MIB = 1024 ** 2


def int_pow(base: int, exp: int) -> int:
    out = 1
    for _ in range(exp):
        out *= base
    return out


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def parse_ints(values: Optional[Sequence[str]], default: Sequence[int]) -> List[int]:
    if not values:
        return list(default)
    out: List[int] = []
    for value in values:
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
                    raise ValueError(f"Bad range expression: {chunk}")
                out.extend(range(start, stop + 1, step))
            else:
                out.append(int(chunk))
    return sorted(set(out))


def parse_floats(values: Optional[Sequence[str]], default: Sequence[float]) -> List[float]:
    if not values:
        return list(default)
    out: List[float] = []
    for value in values:
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            out.append(float(chunk))
    return sorted(set(out))


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def build_custom_template_states_q4(num_partitions: int) -> List[List[int]]:
    """Port GE2's q=4 CUSTOM template state construction.

    This intentionally omits the later runtime randperm over partition IDs and
    randperm over state order.  The purpose here is to study the structure and
    lower-bound pressure, not reproduce one seeded training run exactly.
    """

    buffer_capacity = 4
    if num_partitions % buffer_capacity != 0:
        raise ValueError("q=4 template requires p divisible by 4")
    if not is_power_of_two(num_partitions):
        raise ValueError("q=4 template requires p to be a power of two")

    sub_chunk_per_perm = num_partitions // buffer_capacity
    log2l = 0
    while int_pow(2, log2l) < num_partitions:
        log2l += 1
    if int_pow(2, log2l) != num_partitions:
        raise ValueError("q=4 template requires p to be a power of two")

    offset_supergroup = [
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
        [[0, 2, 3, 1], [1, 3, 2, 0], [2, 0, 1, 3], [3, 1, 0, 2]],
        [[0, 3, 1, 2], [1, 2, 0, 3], [2, 1, 3, 0], [3, 0, 2, 1]],
    ]
    states_by_round = [[[0, 1, 2, 3]]]

    for log4l_pre in range(1, log2l // 2):
        prev = states_by_round
        states_by_round = []
        for super_state in prev:
            current_super: List[List[int]] = []
            for offset in range(0, int_pow(4, log4l_pre + 1), int_pow(4, log4l_pre)):
                for group in super_state:
                    current_super.append([x + offset for x in group])
            states_by_round.append(current_super)

        length = len(prev)
        for idx in range(length - int_pow(4, log4l_pre - 1), length):
            super_state = prev[idx]
            for offset_super in offset_supergroup:
                current_super = []
                for group in super_state:
                    for offset_group in offset_super:
                        current_super.append([group[j] * 4 + offset_group[j] for j in range(4)])
                states_by_round.append(current_super)

    pairing_chunks = [
        [[0, 2], [1, 3]],
        [[0, 3], [1, 2]],
    ]

    if log2l % 2 == 1:
        length_chunk = sub_chunk_per_perm
        prev = states_by_round
        states_by_round = []

        for super_state in prev:
            current_super = []
            for offset in range(0, int_pow(2, log2l), int_pow(2, log2l - 1)):
                for group in super_state:
                    current_super.append([x + offset for x in group])
            states_by_round.append(current_super)

        length = len(prev)
        for idx in range(length - int_pow(2, log2l - 3), length):
            super_state = prev[idx]
            for pairing_super in pairing_chunks:
                current_super = []
                for chunk_index in pairing_super:
                    for group in super_state:
                        current_group = [
                            chunk_index[x // length_chunk] * length_chunk + x % length_chunk
                            for x in group
                        ]
                        current_super.append(current_group)
                states_by_round.append(current_super)

    states: List[List[int]] = []
    for super_state in states_by_round:
        states.extend(super_state)
    return states


def avg_overlap(states: Sequence[Sequence[int]]) -> float:
    if len(states) <= 1:
        return 0.0
    total = 0
    for prev, cur in zip(states, states[1:]):
        total += len(set(prev).intersection(cur))
    return total / (len(states) - 1)


def greedy_overlap_order(states: Sequence[Sequence[int]]) -> List[List[int]]:
    """Cheaply approximate GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM without edge hotness.

    The production code ranks by overlap first, then shared edge hotness.  The
    simulator usually does not have bucket-size vectors for every candidate p/q,
    so this unweighted version still captures the key machinery: prefer the next
    state that retains at least one partition.

    This is intentionally a single-start greedy pass.  The production helper
    tries every start and has a lookahead tie-break; doing that for p=64 in a
    sweep makes the simulator slower than it needs to be.
    """

    if len(states) <= 1:
        return [list(state) for state in states]

    state_sets = [set(state) for state in states]
    order = [0]
    used = {0}
    while len(order) < len(states):
        prev_idx = order[-1]
        best_next = None
        best_next_key = (-1, -1)
        for candidate_idx in range(len(states)):
            if candidate_idx in used:
                continue
            overlap = len(state_sets[prev_idx].intersection(state_sets[candidate_idx]))
            key = (overlap, -candidate_idx)
            if key > best_next_key:
                best_next_key = key
                best_next = candidate_idx
        if best_next is None:
            break
        used.add(best_next)
        order.append(best_next)

    return [list(states[idx]) for idx in order]


def greedy_cover_states(num_partitions: int, buffer_capacity: int) -> List[List[int]]:
    """Match the C++ greedy positive-bucket cover prototype."""

    if buffer_capacity <= 0 or buffer_capacity > num_partitions:
        return []
    states = [list(candidate) for candidate in combinations(range(num_partitions), buffer_capacity)]
    uncovered: Set[Tuple[int, int]] = {
        (src, dst) for src in range(num_partitions) for dst in range(num_partitions)
    }
    selected: List[List[int]] = []

    while uncovered:
        best_state: Optional[List[int]] = None
        best_key: Optional[Tuple[int, int, int]] = None
        prev = set(selected[-1]) if selected else set()
        for state in states:
            gain = 0
            for src in state:
                for dst in state:
                    if (src, dst) in uncovered:
                        gain += 1
            if gain == 0:
                continue
            overlap = len(prev.intersection(state))
            balance_score = -sum(state)
            key = (gain, overlap, balance_score)
            if best_key is None or key > best_key:
                best_key = key
                best_state = state

        if best_state is None:
            break
        selected.append(best_state)
        for src in best_state:
            for dst in best_state:
                uncovered.discard((src, dst))

    return selected


def state_count_lower_bound(num_partitions: int, buffer_capacity: int) -> int:
    positive_directed = num_partitions * (num_partitions - 1)
    positive_per_state = buffer_capacity * (buffer_capacity - 1)
    diagonal = num_partitions
    diagonal_per_state = buffer_capacity
    return max(
        ceil_div(positive_directed, positive_per_state),
        ceil_div(diagonal, diagonal_per_state),
    )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    node_count: int
    emb_dim: int
    bytes_per_value: int
    optimizer_states: int
    current_epoch_s: Optional[float]
    current_swap_s: Optional[float]
    baseline_partitions: int
    baseline_buffer_capacity: int
    baseline_gpus: int


@dataclass(frozen=True)
class Candidate:
    dataset: str
    gpus: int
    num_partitions: int
    buffer_capacity: int
    state_count: int
    states_per_lane: int
    transitions_per_lane: int
    retained_avg: float
    retained_source: str
    evict_parts_per_transition: float
    resident_gib: float
    full_stage_gib: float
    one_partition_stage_gib: float
    current_like_swap_gib_per_lane: float
    one_admit_swap_gib_per_lane: float
    ghost_capacity_parts: float
    ghost_resident_gib: float
    ghost_total_resident_gib: float
    ghost_boundary_stage_gib: float
    ghost_boundary_swap_gib_per_lane: float
    ghost_fits_free: bool
    full_stage_fits_free: bool
    one_partition_stage_fits_free: bool
    contrastive_active_parts: int
    contrastive_free_slots: int
    contrastive_prefetch_possible: bool
    negative_tiles_per_state: int
    swap_only_epoch_s: Optional[float]
    one_admit_epoch_s: Optional[float]
    ghost_boundary_epoch_s: Optional[float]
    notes: str


def estimate_candidate(
    spec: DatasetSpec,
    num_partitions: int,
    buffer_capacity: int,
    gpus: int,
    negative_tile_parts: int,
    retained_assumption: float,
    free_hbm_gib: float,
    ghost_capacity_parts: float,
    use_template_q4: bool,
    greedy_overlap: bool,
    greedy_cover: bool,
) -> Candidate:
    partition_rows = ceil_div(spec.node_count, num_partitions)
    storage_multiplier = 1 + spec.optimizer_states
    partition_bytes_one_storage = partition_rows * spec.emb_dim * spec.bytes_per_value
    partition_bytes_all_storage = partition_bytes_one_storage * storage_multiplier

    state_count = state_count_lower_bound(num_partitions, buffer_capacity)
    retained_source = "assumed"
    retained_avg = retained_assumption
    notes: List[str] = []

    if greedy_cover:
        states = greedy_cover_states(num_partitions, buffer_capacity)
        state_count = len(states)
        retained_avg = avg_overlap(states)
        retained_source = "greedy_cover"
        notes.append("greedy-cover")
    elif use_template_q4 and buffer_capacity == 4 and is_power_of_two(num_partitions):
        states = build_custom_template_states_q4(num_partitions)
        if greedy_overlap:
            states = greedy_overlap_order(states)
            retained_source = "q4_template_greedy_overlap"
        else:
            retained_source = "q4_template_no_randperm"
        state_count = len(states)
        retained_avg = avg_overlap(states)
        notes.append("q4-template")
    else:
        notes.append("lower-bound-states")

    states_per_lane = ceil_div(state_count, max(gpus, 1))
    transitions_per_lane = max(0, states_per_lane - 1)
    evict_parts_per_transition = max(0.0, buffer_capacity - retained_avg)

    resident_bytes = buffer_capacity * partition_bytes_all_storage
    full_stage_bytes = 2.0 * evict_parts_per_transition * partition_bytes_all_storage
    one_partition_stage_bytes = 2.0 * partition_bytes_all_storage
    current_like_swap_bytes = transitions_per_lane * full_stage_bytes
    one_admit_swap_bytes = transitions_per_lane * one_partition_stage_bytes

    # Hidden physical frames are not visible to the sampler.  They can prefetch
    # admitted partitions before a state boundary, then hold evicted partitions
    # for delayed writeback after the boundary.  This reduces exposed boundary
    # traffic without changing the logical q-visible COVER state.
    ghost_capacity_parts = max(0.0, ghost_capacity_parts)
    ghost_bytes = ghost_capacity_parts * partition_bytes_all_storage
    ghost_total_resident_bytes = resident_bytes + ghost_bytes
    ghost_exposed_parts = max(0.0, evict_parts_per_transition - ghost_capacity_parts)
    ghost_boundary_stage_bytes = 2.0 * ghost_exposed_parts * partition_bytes_all_storage
    ghost_boundary_swap_bytes = transitions_per_lane * ghost_boundary_stage_bytes

    free_hbm_bytes = free_hbm_gib * GIB
    full_stage_fits_free = full_stage_bytes <= free_hbm_bytes
    one_partition_stage_fits_free = one_partition_stage_bytes <= free_hbm_bytes
    ghost_fits_free = ghost_bytes <= free_hbm_bytes

    contrastive_active_parts = 2 + negative_tile_parts
    contrastive_free_slots = buffer_capacity - contrastive_active_parts
    contrastive_prefetch_possible = contrastive_free_slots >= 1
    negative_tiles_per_state = ceil_div(buffer_capacity, max(negative_tile_parts, 1))
    if not contrastive_prefetch_possible:
        notes.append("no-free-slot")
    if not full_stage_fits_free:
        notes.append("full-stage-hbm-risk")
    if one_partition_stage_fits_free and not full_stage_fits_free:
        notes.append("tiled-stage-fits")
    if ghost_capacity_parts > 0:
        notes.append("ghost-prefetch")
        notes.append("ghost-fits" if ghost_fits_free else "ghost-hbm-risk")

    swap_only_epoch_s: Optional[float] = None
    one_admit_epoch_s: Optional[float] = None
    ghost_boundary_epoch_s: Optional[float] = None
    if spec.current_epoch_s is not None and spec.current_swap_s is not None:
        baseline = estimate_candidate(
            DatasetSpec(
                name=spec.name,
                node_count=spec.node_count,
                emb_dim=spec.emb_dim,
                bytes_per_value=spec.bytes_per_value,
                optimizer_states=spec.optimizer_states,
                current_epoch_s=None,
                current_swap_s=None,
                baseline_partitions=spec.baseline_partitions,
                baseline_buffer_capacity=spec.baseline_buffer_capacity,
                baseline_gpus=spec.baseline_gpus,
            ),
            spec.baseline_partitions,
            spec.baseline_buffer_capacity,
            spec.baseline_gpus,
            negative_tile_parts,
            retained_assumption,
            free_hbm_gib,
            0.0,
            use_template_q4,
            greedy_overlap,
            greedy_cover,
        )
        baseline_bytes = baseline.current_like_swap_gib_per_lane
        if baseline_bytes > 0:
            predicted_swap = spec.current_swap_s * (current_like_swap_bytes / GIB) / baseline_bytes
            swap_only_epoch_s = spec.current_epoch_s - spec.current_swap_s + predicted_swap
            predicted_one_admit_swap = spec.current_swap_s * (one_admit_swap_bytes / GIB) / baseline_bytes
            one_admit_epoch_s = spec.current_epoch_s - spec.current_swap_s + predicted_one_admit_swap
            predicted_ghost_swap = spec.current_swap_s * (ghost_boundary_swap_bytes / GIB) / baseline_bytes
            ghost_boundary_epoch_s = spec.current_epoch_s - spec.current_swap_s + predicted_ghost_swap

    return Candidate(
        dataset=spec.name,
        gpus=gpus,
        num_partitions=num_partitions,
        buffer_capacity=buffer_capacity,
        state_count=state_count,
        states_per_lane=states_per_lane,
        transitions_per_lane=transitions_per_lane,
        retained_avg=retained_avg,
        retained_source=retained_source,
        evict_parts_per_transition=evict_parts_per_transition,
        resident_gib=resident_bytes / GIB,
        full_stage_gib=full_stage_bytes / GIB,
        one_partition_stage_gib=one_partition_stage_bytes / GIB,
        current_like_swap_gib_per_lane=current_like_swap_bytes / GIB,
        one_admit_swap_gib_per_lane=one_admit_swap_bytes / GIB,
        ghost_capacity_parts=ghost_capacity_parts,
        ghost_resident_gib=ghost_bytes / GIB,
        ghost_total_resident_gib=ghost_total_resident_bytes / GIB,
        ghost_boundary_stage_gib=ghost_boundary_stage_bytes / GIB,
        ghost_boundary_swap_gib_per_lane=ghost_boundary_swap_bytes / GIB,
        ghost_fits_free=ghost_fits_free,
        full_stage_fits_free=full_stage_fits_free,
        one_partition_stage_fits_free=one_partition_stage_fits_free,
        contrastive_active_parts=contrastive_active_parts,
        contrastive_free_slots=contrastive_free_slots,
        contrastive_prefetch_possible=contrastive_prefetch_possible,
        negative_tiles_per_state=negative_tiles_per_state,
        swap_only_epoch_s=swap_only_epoch_s,
        one_admit_epoch_s=one_admit_epoch_s,
        ghost_boundary_epoch_s=ghost_boundary_epoch_s,
        notes=",".join(notes),
    )


def write_candidates(candidates: Iterable[Candidate], output_format: str) -> None:
    rows = list(candidates)
    field_names = [
        "dataset",
        "gpus",
        "p",
        "q",
        "states",
        "states_per_lane",
        "transitions_per_lane",
        "retained_avg",
        "retained_source",
        "evict_parts_per_transition",
        "resident_gib",
        "full_stage_gib",
        "one_partition_stage_gib",
        "current_like_swap_gib_per_lane",
        "one_admit_swap_gib_per_lane",
        "ghost_capacity_parts",
        "ghost_resident_gib",
        "ghost_total_resident_gib",
        "ghost_boundary_stage_gib",
        "ghost_boundary_swap_gib_per_lane",
        "ghost_fits_free",
        "full_stage_fits_free",
        "one_partition_stage_fits_free",
        "contrastive_active_parts",
        "contrastive_free_slots",
        "contrastive_prefetch_possible",
        "negative_tiles_per_state",
        "swap_only_epoch_s",
        "one_admit_epoch_s",
        "ghost_boundary_epoch_s",
        "notes",
    ]

    def row_to_dict(row: Candidate) -> dict:
        return {
            "dataset": row.dataset,
            "gpus": row.gpus,
            "p": row.num_partitions,
            "q": row.buffer_capacity,
            "states": row.state_count,
            "states_per_lane": row.states_per_lane,
            "transitions_per_lane": row.transitions_per_lane,
            "retained_avg": fmt_float(row.retained_avg),
            "retained_source": row.retained_source,
            "evict_parts_per_transition": fmt_float(row.evict_parts_per_transition),
            "resident_gib": fmt_float(row.resident_gib),
            "full_stage_gib": fmt_float(row.full_stage_gib),
            "one_partition_stage_gib": fmt_float(row.one_partition_stage_gib),
            "current_like_swap_gib_per_lane": fmt_float(row.current_like_swap_gib_per_lane),
            "one_admit_swap_gib_per_lane": fmt_float(row.one_admit_swap_gib_per_lane),
            "ghost_capacity_parts": fmt_float(row.ghost_capacity_parts),
            "ghost_resident_gib": fmt_float(row.ghost_resident_gib),
            "ghost_total_resident_gib": fmt_float(row.ghost_total_resident_gib),
            "ghost_boundary_stage_gib": fmt_float(row.ghost_boundary_stage_gib),
            "ghost_boundary_swap_gib_per_lane": fmt_float(row.ghost_boundary_swap_gib_per_lane),
            "ghost_fits_free": int(row.ghost_fits_free),
            "full_stage_fits_free": int(row.full_stage_fits_free),
            "one_partition_stage_fits_free": int(row.one_partition_stage_fits_free),
            "contrastive_active_parts": row.contrastive_active_parts,
            "contrastive_free_slots": row.contrastive_free_slots,
            "contrastive_prefetch_possible": int(row.contrastive_prefetch_possible),
            "negative_tiles_per_state": row.negative_tiles_per_state,
            "swap_only_epoch_s": "" if row.swap_only_epoch_s is None else fmt_float(row.swap_only_epoch_s),
            "one_admit_epoch_s": "" if row.one_admit_epoch_s is None else fmt_float(row.one_admit_epoch_s),
            "ghost_boundary_epoch_s": "" if row.ghost_boundary_epoch_s is None else fmt_float(row.ghost_boundary_epoch_s),
            "notes": row.notes,
        }

    if output_format == "markdown":
        print("| " + " | ".join(field_names) + " |")
        print("| " + " | ".join(["---"] * len(field_names)) + " |")
        for row in rows:
            values = row_to_dict(row)
            print("| " + " | ".join(str(values[name]) for name in field_names) + " |")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=field_names, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset label for the output table.")
    parser.add_argument("--node-count", required=True, type=int)
    parser.add_argument("--emb-dim", type=int, default=100)
    parser.add_argument("--bytes-per-value", type=int, default=4)
    parser.add_argument("--optimizer-states", type=int, default=1,
                        help="Number of sparse optimizer state tensors stored with embeddings. Adagrad=1.")
    parser.add_argument("--num-partitions", nargs="*", default=["16,24,32"],
                        help="List or ranges, e.g. '16,24,32' or '16:64:16'.")
    parser.add_argument("--buffer-capacity", nargs="*", default=["3,4,5,6"])
    parser.add_argument("--gpus", nargs="*", default=["1"])
    parser.add_argument("--negative-tile-parts", type=int, default=1,
                        help="How many partitions are exposed as the active negative domain per microtask.")
    parser.add_argument("--ghost-capacity-parts", nargs="*", default=["0"],
                        help=(
                            "Partition-equivalent hidden physical capacity for visibility-decoupled ghost "
                            "prefetch. Supports comma lists such as '0,0.25,0.5,1,2,3'."
                        ))
    parser.add_argument("--retained-assumption", type=float, default=1.0,
                        help="Average retained partitions per transition for non-template estimates.")
    parser.add_argument("--free-hbm-gib", type=float, default=6.0,
                        help="Conservative free HBM budget available for staging.")
    parser.add_argument("--use-template-q4", action="store_true",
                        help="Use GE2's q=4 CUSTOM template shape when possible.")
    parser.add_argument("--greedy-overlap-order", action="store_true",
                        help="Approximate GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM for q=4 template states.")
    parser.add_argument("--greedy-cover-states", action="store_true",
                        help="Use the same generic greedy positive-bucket cover as GEGE_CONTRASTIVE_GREEDY_COVER_ORDERING.")
    parser.add_argument("--current-epoch-s", type=float)
    parser.add_argument("--current-swap-s", type=float)
    parser.add_argument("--baseline-partitions", type=int, default=16)
    parser.add_argument("--baseline-buffer-capacity", type=int, default=4)
    parser.add_argument("--baseline-gpus", type=int, default=1)
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    spec = DatasetSpec(
        name=args.dataset,
        node_count=args.node_count,
        emb_dim=args.emb_dim,
        bytes_per_value=args.bytes_per_value,
        optimizer_states=args.optimizer_states,
        current_epoch_s=args.current_epoch_s,
        current_swap_s=args.current_swap_s,
        baseline_partitions=args.baseline_partitions,
        baseline_buffer_capacity=args.baseline_buffer_capacity,
        baseline_gpus=args.baseline_gpus,
    )

    candidates: List[Candidate] = []
    for gpus in parse_ints(args.gpus, [1]):
        for num_partitions in parse_ints(args.num_partitions, [16, 24, 32]):
            for buffer_capacity in parse_ints(args.buffer_capacity, [3, 4, 5, 6]):
                if buffer_capacity < 3:
                    continue
                if buffer_capacity >= num_partitions:
                    continue
                for ghost_capacity_parts in parse_floats(args.ghost_capacity_parts, [0.0]):
                    candidates.append(
                        estimate_candidate(
                            spec=spec,
                            num_partitions=num_partitions,
                            buffer_capacity=buffer_capacity,
                            gpus=gpus,
                            negative_tile_parts=args.negative_tile_parts,
                            retained_assumption=args.retained_assumption,
                            free_hbm_gib=args.free_hbm_gib,
                            ghost_capacity_parts=ghost_capacity_parts,
                            use_template_q4=args.use_template_q4,
                            greedy_overlap=args.greedy_overlap_order,
                            greedy_cover=args.greedy_cover_states,
                        )
                    )

    candidates.sort(
        key=lambda row: (
            row.gpus,
            not row.ghost_fits_free,
            not row.contrastive_prefetch_possible,
            not row.one_partition_stage_fits_free,
            row.ghost_boundary_epoch_s if row.ghost_boundary_epoch_s is not None else row.ghost_boundary_swap_gib_per_lane,
            row.swap_only_epoch_s if row.swap_only_epoch_s is not None else row.current_like_swap_gib_per_lane,
            row.one_admit_epoch_s if row.one_admit_epoch_s is not None else row.one_admit_swap_gib_per_lane,
            row.resident_gib,
        )
    )
    write_candidates(candidates, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
