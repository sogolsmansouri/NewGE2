#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Baseline:
    name: str
    epoch_s: float
    swap_s: float
    map_s: float
    neg_s: float
    states: int

    @property
    def compute_s(self) -> float:
        return self.epoch_s - self.swap_s - self.map_s - self.neg_s

    @property
    def prep_s(self) -> float:
        return self.map_s + self.neg_s


BASELINES = {
    "twitter_1gpu": Baseline(
        name="twitter_1gpu",
        epoch_s=223.230,
        swap_s=59.414,
        map_s=43.508,
        neg_s=9.107,
        states=19,
    ),
    "twitter_1gpu_framecache": Baseline(
        name="twitter_1gpu_framecache",
        epoch_s=215.207,
        swap_s=52.560,
        map_s=43.951,
        neg_s=8.206,
        states=19,
    ),
    "fb86m_1gpu": Baseline(
        name="fb86m_1gpu",
        epoch_s=164.098,
        swap_s=65.485,
        map_s=86.620,
        neg_s=0.544,
        states=19,
    ),
    "fb86m_1gpu_framecache": Baseline(
        name="fb86m_1gpu_framecache",
        epoch_s=154.289,
        swap_s=53.290,
        map_s=84.604,
        neg_s=0.719,
        states=19,
    ),
}


@dataclass(frozen=True)
class Scenario:
    label: str
    overlap_swap: bool
    overlap_prep: bool
    swap_efficiency: float
    prep_efficiency: float


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def schedule_ready_time(current_compute_start: float,
                        resource_ready: float,
                        task_duration: float,
                        can_overlap: bool,
                        efficiency: float) -> float:
    if task_duration <= 0.0:
        return current_compute_start

    if not can_overlap or efficiency <= 0.0:
        return -1.0

    overlappable_duration = task_duration / efficiency
    start = max(current_compute_start, resource_ready)
    finish = start + overlappable_duration
    return finish


def simulate_pipeline(baseline: Baseline,
                      publish_ms_per_state: float,
                      scenario: Scenario) -> dict[str, float | str]:
    states = baseline.states
    compute_per_state = baseline.compute_s / states
    swap_per_state = baseline.swap_s / states
    prep_per_state = baseline.prep_s / states
    publish_s = publish_ms_per_state / 1000.0

    transfer_ready = 0.0
    cpu_ready = 0.0

    swap_ready = [0.0 for _ in range(states)]
    prep_ready = [0.0 for _ in range(states)]

    # State 0 must be prepared before the first compute can start.
    swap_ready[0] = swap_per_state
    prep_ready[0] = prep_per_state
    transfer_ready = swap_ready[0]
    cpu_ready = prep_ready[0]

    compute_start = [0.0 for _ in range(states)]
    compute_end = [0.0 for _ in range(states)]
    state_wait_before_compute = [0.0 for _ in range(states)]

    first_ready = 0.0
    if scenario.overlap_swap:
        first_ready = max(first_ready, swap_ready[0])
    else:
        first_ready += swap_per_state
    if scenario.overlap_prep:
        first_ready = max(first_ready, prep_ready[0])
    else:
        first_ready += prep_per_state

    compute_start[0] = first_ready + publish_s
    state_wait_before_compute[0] = compute_start[0]
    compute_end[0] = compute_start[0] + compute_per_state

    for state_idx in range(states - 1):
        next_idx = state_idx + 1
        swap_ready[next_idx] = schedule_ready_time(
            current_compute_start=compute_start[state_idx],
            resource_ready=transfer_ready,
            task_duration=swap_per_state,
            can_overlap=scenario.overlap_swap,
            efficiency=scenario.swap_efficiency,
        )
        prep_ready[next_idx] = schedule_ready_time(
            current_compute_start=compute_start[state_idx],
            resource_ready=cpu_ready,
            task_duration=prep_per_state,
            can_overlap=scenario.overlap_prep,
            efficiency=scenario.prep_efficiency,
        )
        if scenario.overlap_swap:
            transfer_ready = swap_ready[next_idx]
        if scenario.overlap_prep:
            cpu_ready = prep_ready[next_idx]

        boundary_ready = compute_end[state_idx]
        if scenario.overlap_swap:
            boundary_ready = max(boundary_ready, swap_ready[next_idx])
        else:
            boundary_ready += swap_per_state
        if scenario.overlap_prep:
            boundary_ready = max(boundary_ready, prep_ready[next_idx])
        else:
            boundary_ready += prep_per_state

        compute_start[next_idx] = boundary_ready + publish_s
        state_wait_before_compute[next_idx] = compute_start[next_idx] - compute_end[state_idx]
        compute_end[next_idx] = compute_start[next_idx] + compute_per_state

    total_epoch = compute_end[-1]
    total_publish = states * publish_s
    total_wait = sum(state_wait_before_compute)
    exposed_swap_wait = 0.0
    exposed_prep_wait = 0.0
    for i in range(1, states):
        boundary_ready = compute_end[i - 1]
        if scenario.overlap_swap:
            swap_wait = max(swap_ready[i] - boundary_ready, 0.0)
            boundary_ready = max(boundary_ready, swap_ready[i])
        else:
            swap_wait = swap_per_state
            boundary_ready += swap_per_state
        if scenario.overlap_prep:
            prep_wait = max(prep_ready[i] - boundary_ready, 0.0)
        else:
            prep_wait = prep_per_state
        exposed_swap_wait += swap_wait
        exposed_prep_wait += prep_wait

    return {
        "dataset": baseline.name,
        "scenario": scenario.label,
        "epoch_s": round(total_epoch, 3),
        "baseline_epoch_s": round(baseline.epoch_s, 3),
        "delta_s": round(total_epoch - baseline.epoch_s, 3),
        "compute_s": round(baseline.compute_s, 3),
        "swap_s": round(baseline.swap_s, 3),
        "prep_s": round(baseline.prep_s, 3),
        "compute_per_state_s": round(compute_per_state, 3),
        "swap_per_state_s": round(swap_per_state, 3),
        "prep_per_state_s": round(prep_per_state, 3),
        "publish_s": round(total_publish, 3),
        "total_wait_s": round(total_wait, 3),
        "exposed_swap_wait_s": round(exposed_swap_wait, 3),
        "exposed_prep_wait_s": round(exposed_prep_wait, 3),
        "swap_overlap": int(scenario.overlap_swap),
        "prep_overlap": int(scenario.overlap_prep),
        "swap_efficiency": scenario.swap_efficiency,
        "prep_efficiency": scenario.prep_efficiency,
        "states": states,
    }


def write_tsv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def scenario_grid(efficiencies: Iterable[float]) -> list[Scenario]:
    rows = [
        Scenario("serialized_reference", False, False, 1.0, 1.0),
        Scenario("compute_plus_swap", True, False, 1.0, 1.0),
        Scenario("compute_plus_prep", False, True, 1.0, 1.0),
        Scenario("compute_plus_both", True, True, 1.0, 1.0),
    ]
    for efficiency in efficiencies:
        rows.append(Scenario(f"compute_plus_both_eff_{efficiency:.2f}", True, True, efficiency, efficiency))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="State pipeline overlap simulator for GE2 next-state preparation.")
    parser.add_argument("--dataset", choices=sorted(BASELINES.keys()), required=True)
    parser.add_argument("--publish-ms-per-state", type=float, default=10.0)
    parser.add_argument("--efficiencies", default="1.0,0.75,0.5")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = BASELINES[args.dataset]
    scenarios = scenario_grid(parse_float_list(args.efficiencies))
    rows = [simulate_pipeline(baseline, args.publish_ms_per_state, scenario) for scenario in scenarios]
    rows.sort(key=lambda row: float(row["epoch_s"]))
    write_tsv(args.out, rows)

    print(f"dataset={baseline.name}")
    print(
        f"baseline_epoch_s={baseline.epoch_s:.3f} compute_s={baseline.compute_s:.3f} "
        f"swap_s={baseline.swap_s:.3f} prep_s={baseline.prep_s:.3f} states={baseline.states}"
    )
    print("top_scenarios:")
    for row in rows[:5]:
        print(
            f"  scenario={row['scenario']} epoch_s={row['epoch_s']} delta_s={row['delta_s']} "
            f"wait_s={row['total_wait_s']} exposed_swap_wait_s={row['exposed_swap_wait_s']} "
            f"exposed_prep_wait_s={row['exposed_prep_wait_s']}"
        )


if __name__ == "__main__":
    main()
