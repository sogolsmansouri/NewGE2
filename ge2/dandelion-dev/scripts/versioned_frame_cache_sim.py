#!/usr/bin/env python3
"""Estimate Versioned Frame Cache gains before implementation.

This is an offline simulator for the design in
`paper/versioned_frame_cache_design_20260408.md`.

Model:
  * Logical visible state remains fixed at q.
  * Extra hidden/stale frame capacity is expressed in partition-equivalents.
  * Future admits can be prefetched into free hidden capacity during current-state compute.
  * Published hidden frames turn old visible frames into stale backlog.
  * Stale backlog drains asynchronously during later compute windows.
  * Only the remaining synchronous boundary work contributes to exposed boundary time.

The simulator is intentionally simple.  It does not predict total runtime from
first principles; it scales the *measured* exposed boundary cost for a current
configuration and estimates how much of that boundary could disappear if the
frame-cache pipeline works as intended.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from contrastive_tiled_cover_sim import (  # type: ignore
    avg_overlap,
    build_custom_template_states_q4,
    ceil_div,
    fmt_float,
    greedy_cover_states,
    is_power_of_two,
    parse_floats,
    parse_ints,
    state_count_lower_bound,
)


GIB = 1024 ** 3


@dataclass(frozen=True)
class FrameCacheCandidate:
    dataset: str
    gpus: int
    p: int
    q: int
    states: int
    states_per_lane: int
    transitions_per_lane: int
    retained_avg: float
    retained_source: str
    parts_per_transition: float
    partition_gib: float
    visible_resident_gib: float
    hidden_capacity_parts: float
    hidden_resident_gib: float
    total_resident_gib: float
    hidden_fits_free: bool
    current_epoch_s: Optional[float]
    current_boundary_s: Optional[float]
    boundary_part_time_ms: Optional[float]
    compute_window_s: Optional[float]
    prefetch_service_parts_per_state: Optional[float]
    commit_service_parts_per_state: Optional[float]
    avg_prefetched_parts: Optional[float]
    avg_boundary_parts: Optional[float]
    avg_stale_backlog_parts: Optional[float]
    publish_transition_ms: float
    stale_backlog_part_ms: float
    predicted_publish_s: Optional[float]
    predicted_backlog_s: Optional[float]
    predicted_boundary_s: Optional[float]
    predicted_epoch_s: Optional[float]
    notes: str


def compute_state_shape(
    num_partitions: int,
    buffer_capacity: int,
    gpus: int,
    retained_assumption: float,
    use_template_q4: bool,
    greedy_overlap: bool,
    greedy_cover: bool,
) -> tuple[int, int, int, float, str, str]:
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
            # import locally to avoid a wide dependency surface here
            from contrastive_tiled_cover_sim import greedy_overlap_order  # type: ignore

            states = greedy_overlap_order(states)
            retained_source = "q4_template_greedy_overlap"
        else:
            retained_source = "q4_template_no_randperm"
        state_count = len(states)
        retained_avg = avg_overlap(states)
        notes.append("q4-template")
    else:
        state_count = state_count_lower_bound(num_partitions, buffer_capacity)
        notes.append("lower-bound-states")

    states_per_lane = ceil_div(state_count, max(gpus, 1))
    transitions_per_lane = max(0, states_per_lane - 1)
    return state_count, states_per_lane, transitions_per_lane, retained_avg, retained_source, ",".join(notes)


def simulate_boundary_pipeline(
    transitions_per_lane: int,
    parts_per_transition: float,
    hidden_capacity_parts: float,
    prefetch_service_parts_per_state: float,
    commit_service_parts_per_state: float,
) -> tuple[float, float, float]:
    if transitions_per_lane <= 0 or parts_per_transition <= 0:
        return 0.0, 0.0, 0.0

    stale_backlog = 0.0
    total_prefetched = 0.0
    total_boundary_parts = 0.0
    total_backlog = 0.0

    for _ in range(transitions_per_lane):
        stale_backlog = max(0.0, stale_backlog - commit_service_parts_per_state)
        free_hidden = max(0.0, hidden_capacity_parts - stale_backlog)
        prefetched = min(parts_per_transition, free_hidden, prefetch_service_parts_per_state)

        # Remaining parts still pay synchronous evict + admit at the boundary.
        total_boundary_parts += 2.0 * max(0.0, parts_per_transition - prefetched)
        total_prefetched += prefetched

        # Publishing a prefetched frame converts the replaced visible frame into stale backlog.
        stale_backlog += prefetched
        total_backlog += stale_backlog

    avg_prefetched = total_prefetched / transitions_per_lane
    avg_boundary_parts = total_boundary_parts / transitions_per_lane
    avg_backlog = total_backlog / transitions_per_lane
    return avg_prefetched, avg_boundary_parts, avg_backlog


def estimate_candidate(
    dataset: str,
    node_count: int,
    emb_dim: int,
    bytes_per_value: int,
    optimizer_states: int,
    p: int,
    q: int,
    gpus: int,
    hidden_capacity_parts: float,
    free_hbm_gib: float,
    retained_assumption: float,
    use_template_q4: bool,
    greedy_overlap: bool,
    greedy_cover: bool,
    current_epoch_s: Optional[float],
    current_boundary_s: Optional[float],
    prefetch_overlap_efficiency: float,
    commit_overlap_efficiency: float,
    publish_transition_ms: float,
    stale_backlog_part_ms: float,
) -> FrameCacheCandidate:
    state_count, states_per_lane, transitions_per_lane, retained_avg, retained_source, notes = compute_state_shape(
        p, q, gpus, retained_assumption, use_template_q4, greedy_overlap, greedy_cover
    )

    rows_per_partition = ceil_div(node_count, p)
    bytes_per_row = emb_dim * bytes_per_value * (1 + optimizer_states)
    partition_bytes = rows_per_partition * bytes_per_row
    visible_resident_bytes = q * partition_bytes
    hidden_bytes = hidden_capacity_parts * partition_bytes
    total_resident_bytes = visible_resident_bytes + hidden_bytes
    hidden_fits_free = hidden_bytes <= free_hbm_gib * GIB

    parts_per_transition = max(0.0, q - retained_avg)

    boundary_part_time_ms: Optional[float] = None
    compute_window_s: Optional[float] = None
    prefetch_service_parts_per_state: Optional[float] = None
    commit_service_parts_per_state: Optional[float] = None
    avg_prefetched_parts: Optional[float] = None
    avg_boundary_parts: Optional[float] = None
    avg_stale_backlog_parts: Optional[float] = None
    predicted_publish_s: Optional[float] = None
    predicted_backlog_s: Optional[float] = None
    predicted_boundary_s: Optional[float] = None
    predicted_epoch_s: Optional[float] = None

    if current_epoch_s is not None and current_boundary_s is not None and transitions_per_lane > 0 and parts_per_transition > 0:
        baseline_boundary_parts = transitions_per_lane * (2.0 * parts_per_transition)
        boundary_part_time_s = current_boundary_s / baseline_boundary_parts
        boundary_part_time_ms = boundary_part_time_s * 1000.0

        compute_window_s = max(0.0, (current_epoch_s - current_boundary_s) / states_per_lane)
        prefetch_service_parts_per_state = (
            0.0 if boundary_part_time_s <= 0 else (compute_window_s * prefetch_overlap_efficiency) / boundary_part_time_s
        )
        commit_service_parts_per_state = (
            0.0 if boundary_part_time_s <= 0 else (compute_window_s * commit_overlap_efficiency) / boundary_part_time_s
        )

        avg_prefetched_parts, avg_boundary_parts, avg_stale_backlog_parts = simulate_boundary_pipeline(
            transitions_per_lane=transitions_per_lane,
            parts_per_transition=parts_per_transition,
            hidden_capacity_parts=hidden_capacity_parts,
            prefetch_service_parts_per_state=prefetch_service_parts_per_state,
            commit_service_parts_per_state=commit_service_parts_per_state,
        )

        predicted_publish_s = (publish_transition_ms / 1000.0) * transitions_per_lane
        predicted_backlog_s = (stale_backlog_part_ms / 1000.0) * avg_stale_backlog_parts * transitions_per_lane
        predicted_boundary_s = (
            boundary_part_time_s * avg_boundary_parts * transitions_per_lane +
            predicted_publish_s +
            predicted_backlog_s
        )
        predicted_epoch_s = current_epoch_s - current_boundary_s + predicted_boundary_s

    return FrameCacheCandidate(
        dataset=dataset,
        gpus=gpus,
        p=p,
        q=q,
        states=state_count,
        states_per_lane=states_per_lane,
        transitions_per_lane=transitions_per_lane,
        retained_avg=retained_avg,
        retained_source=retained_source,
        parts_per_transition=parts_per_transition,
        partition_gib=partition_bytes / GIB,
        visible_resident_gib=visible_resident_bytes / GIB,
        hidden_capacity_parts=hidden_capacity_parts,
        hidden_resident_gib=hidden_bytes / GIB,
        total_resident_gib=total_resident_bytes / GIB,
        hidden_fits_free=hidden_fits_free,
        current_epoch_s=current_epoch_s,
        current_boundary_s=current_boundary_s,
        boundary_part_time_ms=boundary_part_time_ms,
        compute_window_s=compute_window_s,
        prefetch_service_parts_per_state=prefetch_service_parts_per_state,
        commit_service_parts_per_state=commit_service_parts_per_state,
        avg_prefetched_parts=avg_prefetched_parts,
        avg_boundary_parts=avg_boundary_parts,
        avg_stale_backlog_parts=avg_stale_backlog_parts,
        publish_transition_ms=publish_transition_ms,
        stale_backlog_part_ms=stale_backlog_part_ms,
        predicted_publish_s=predicted_publish_s,
        predicted_backlog_s=predicted_backlog_s,
        predicted_boundary_s=predicted_boundary_s,
        predicted_epoch_s=predicted_epoch_s,
        notes=notes + (",hidden-fits" if hidden_fits_free else ",hidden-hbm-risk"),
    )


def rows_to_dict(row: FrameCacheCandidate) -> dict[str, object]:
    def maybe(value: Optional[float]) -> str:
        return "" if value is None else fmt_float(value)

    return {
        "dataset": row.dataset,
        "gpus": row.gpus,
        "p": row.p,
        "q": row.q,
        "states": row.states,
        "states_per_lane": row.states_per_lane,
        "transitions_per_lane": row.transitions_per_lane,
        "retained_avg": fmt_float(row.retained_avg),
        "retained_source": row.retained_source,
        "parts_per_transition": fmt_float(row.parts_per_transition),
        "partition_gib": fmt_float(row.partition_gib),
        "visible_resident_gib": fmt_float(row.visible_resident_gib),
        "hidden_capacity_parts": fmt_float(row.hidden_capacity_parts),
        "hidden_resident_gib": fmt_float(row.hidden_resident_gib),
        "total_resident_gib": fmt_float(row.total_resident_gib),
        "hidden_fits_free": int(row.hidden_fits_free),
        "current_epoch_s": maybe(row.current_epoch_s),
        "current_boundary_s": maybe(row.current_boundary_s),
        "boundary_part_time_ms": maybe(row.boundary_part_time_ms),
        "compute_window_s": maybe(row.compute_window_s),
        "prefetch_service_parts_per_state": maybe(row.prefetch_service_parts_per_state),
        "commit_service_parts_per_state": maybe(row.commit_service_parts_per_state),
        "avg_prefetched_parts": maybe(row.avg_prefetched_parts),
        "avg_boundary_parts": maybe(row.avg_boundary_parts),
        "avg_stale_backlog_parts": maybe(row.avg_stale_backlog_parts),
        "publish_transition_ms": fmt_float(row.publish_transition_ms),
        "stale_backlog_part_ms": fmt_float(row.stale_backlog_part_ms),
        "predicted_publish_s": maybe(row.predicted_publish_s),
        "predicted_backlog_s": maybe(row.predicted_backlog_s),
        "predicted_boundary_s": maybe(row.predicted_boundary_s),
        "predicted_epoch_s": maybe(row.predicted_epoch_s),
        "notes": row.notes,
    }


def write_output(rows: Iterable[FrameCacheCandidate], output_format: str) -> None:
    rows = list(rows)
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
        "parts_per_transition",
        "partition_gib",
        "visible_resident_gib",
        "hidden_capacity_parts",
        "hidden_resident_gib",
        "total_resident_gib",
        "hidden_fits_free",
        "current_epoch_s",
        "current_boundary_s",
        "boundary_part_time_ms",
        "compute_window_s",
        "prefetch_service_parts_per_state",
        "commit_service_parts_per_state",
        "avg_prefetched_parts",
        "avg_boundary_parts",
        "avg_stale_backlog_parts",
        "publish_transition_ms",
        "stale_backlog_part_ms",
        "predicted_publish_s",
        "predicted_backlog_s",
        "predicted_boundary_s",
        "predicted_epoch_s",
        "notes",
    ]

    if output_format == "markdown":
        print("| " + " | ".join(field_names) + " |")
        print("| " + " | ".join(["---"] * len(field_names)) + " |")
        for row in rows:
            values = rows_to_dict(row)
            print("| " + " | ".join(str(values[name]) for name in field_names) + " |")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=field_names, dialect="excel-tab")
        writer.writeheader()
        for row in rows:
            writer.writerow(rows_to_dict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--node-count", required=True, type=int)
    parser.add_argument("--emb-dim", type=int, default=100)
    parser.add_argument("--bytes-per-value", type=int, default=4)
    parser.add_argument("--optimizer-states", type=int, default=1)
    parser.add_argument("--num-partitions", nargs="*", default=["16"])
    parser.add_argument("--buffer-capacity", nargs="*", default=["4"])
    parser.add_argument("--gpus", nargs="*", default=["1"])
    parser.add_argument("--hidden-capacity-parts", nargs="*", default=["0,0.5,1,2,3"])
    parser.add_argument("--free-hbm-gib", type=float, default=6.0)
    parser.add_argument("--retained-assumption", type=float, default=1.0)
    parser.add_argument("--use-template-q4", action="store_true")
    parser.add_argument("--greedy-overlap-order", action="store_true")
    parser.add_argument("--greedy-cover-states", action="store_true")
    parser.add_argument("--current-epoch-s", type=float)
    parser.add_argument("--current-boundary-s", type=float,
                        help="Measured exposed boundary time to scale (swap, or swap+barrier).")
    parser.add_argument("--prefetch-overlap-efficiency", type=float, default=1.0)
    parser.add_argument("--commit-overlap-efficiency", type=float, default=1.0)
    parser.add_argument("--publish-transition-ms", type=float, default=0.0,
                        help="Fixed publish/remap control cost paid per transition.")
    parser.add_argument("--stale-backlog-part-ms", type=float, default=0.0,
                        help="Exposed backlog cost per stale partition-equivalent per transition.")
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    rows: List[FrameCacheCandidate] = []
    for gpus in parse_ints(args.gpus, [1]):
        for p in parse_ints(args.num_partitions, [16]):
            for q in parse_ints(args.buffer_capacity, [4]):
                if q < 2 or q >= p:
                    continue
                for hidden_capacity_parts in parse_floats(args.hidden_capacity_parts, [0.0, 1.0]):
                    rows.append(
                        estimate_candidate(
                            dataset=args.dataset,
                            node_count=args.node_count,
                            emb_dim=args.emb_dim,
                            bytes_per_value=args.bytes_per_value,
                            optimizer_states=args.optimizer_states,
                            p=p,
                            q=q,
                            gpus=gpus,
                            hidden_capacity_parts=hidden_capacity_parts,
                            free_hbm_gib=args.free_hbm_gib,
                            retained_assumption=args.retained_assumption,
                            use_template_q4=args.use_template_q4,
                            greedy_overlap=args.greedy_overlap_order,
                            greedy_cover=args.greedy_cover_states,
                            current_epoch_s=args.current_epoch_s,
                            current_boundary_s=args.current_boundary_s,
                            prefetch_overlap_efficiency=args.prefetch_overlap_efficiency,
                            commit_overlap_efficiency=args.commit_overlap_efficiency,
                            publish_transition_ms=args.publish_transition_ms,
                            stale_backlog_part_ms=args.stale_backlog_part_ms,
                        )
                    )

    rows.sort(
        key=lambda row: (
            row.gpus,
            row.p,
            row.q,
            not row.hidden_fits_free,
            row.predicted_epoch_s if row.predicted_epoch_s is not None else float("inf"),
            row.hidden_capacity_parts,
        )
    )
    write_output(rows, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
