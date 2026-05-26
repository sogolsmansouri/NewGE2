#!/usr/bin/env python3
"""CP-SAT checker/searcher for Opt88 multi-GPU lane schedules.

This works over a fixed set of q=4 states. It assigns each state to a
(round, lane), enforces that states in the same round are partition-disjoint,
and optimizes transition residency metrics:

  same-lane overlap -> lane admits = g*q*(rounds-1) - same_overlap
  host admits       -> next partition not in any previous-round lane
  peer admits       -> lane_admits - host_admits

The model is exact for these schedule metrics. A FEASIBLE result is a valid
schedule but not a proof of optimality; OPTIMAL is a certificate for the chosen
objective.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from ortools.sat.python import cp_model


STATE_RE = re.compile(r"state\s+\d+:\s*\[([^\]]+)\]")
RESIDENT_RE = re.compile(r"resident=\[([^\]]+)\]")


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.split(r"[,\s]+", raw.strip()) if part)


def load_states(path: Path) -> list[tuple[int, ...]]:
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        states: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for lane in obj.get("lanes", []):
            for microstate in lane.get("microstates", []):
                state = tuple(microstate["resident_partitions"])
                if state not in seen:
                    seen.add(state)
                    states.append(state)
        if states:
            return states

    states = []
    seen = set()
    for line in path.read_text().splitlines():
        match = STATE_RE.search(line) or RESIDENT_RE.search(line)
        if match is None:
            continue
        state = parse_ints(match.group(1))
        if state not in seen:
            seen.add(state)
            states.append(state)
    if not states:
        raise ValueError(f"no states found in {path}")
    return states


def compute_metrics(schedule: list[list[int]], states: list[tuple[int, ...]]) -> dict[str, int]:
    state_sets = [set(state) for state in states]
    gpus = len(schedule[0])
    same = 0
    host = 0
    peer = 0
    lane = 0
    bad_rounds = 0
    for round_states in schedule:
        union: set[int] = set()
        for state_idx in round_states:
            if union & state_sets[state_idx]:
                bad_rounds += 1
                break
            union.update(state_sets[state_idx])

    for round_idx in range(1, len(schedule)):
        prev_union = set().union(*(state_sets[schedule[round_idx - 1][lane_idx]]
                                   for lane_idx in range(gpus)))
        for lane_idx in range(gpus):
            prev = state_sets[schedule[round_idx - 1][lane_idx]]
            cur = state_sets[schedule[round_idx][lane_idx]]
            same += len(prev & cur)
            for partition in cur - prev:
                lane += 1
                if partition in prev_union:
                    peer += 1
                else:
                    host += 1
    return {
        "same_overlap": same,
        "lane_admits": lane,
        "host_admits": host,
        "peer_admits": peer,
        "bad_rounds": bad_rounds,
    }


def solve(states: list[tuple[int, ...]], gpus: int, objective: str, time_limit: float,
          workers: int, min_same: int | None, anchor: bool) -> tuple[str, float, float, list[list[int]] | None]:
    if len(states) % gpus != 0:
        raise ValueError(f"{len(states)} states is not divisible by {gpus} GPUs")

    rounds = len(states) // gpus
    state_sets = [set(state) for state in states]
    partitions = sorted(set().union(*state_sets))
    q = len(states[0])

    model = cp_model.CpModel()
    x = {
        (state_idx, round_idx, lane_idx): model.NewBoolVar(f"x_{state_idx}_{round_idx}_{lane_idx}")
        for state_idx in range(len(states))
        for round_idx in range(rounds)
        for lane_idx in range(gpus)
    }

    for state_idx in range(len(states)):
        model.AddExactlyOne(x[state_idx, round_idx, lane_idx]
                            for round_idx in range(rounds)
                            for lane_idx in range(gpus))

    for round_idx in range(rounds):
        for lane_idx in range(gpus):
            model.AddExactlyOne(x[state_idx, round_idx, lane_idx]
                                for state_idx in range(len(states)))

    incompatible_pairs = [
        (lhs, rhs)
        for lhs in range(len(states))
        for rhs in range(lhs + 1, len(states))
        if state_sets[lhs] & state_sets[rhs]
    ]
    for round_idx in range(rounds):
        for lhs, rhs in incompatible_pairs:
            model.Add(sum(x[lhs, round_idx, lane_idx] + x[rhs, round_idx, lane_idx]
                          for lane_idx in range(gpus)) <= 1)

    if anchor:
        model.Add(x[0, 0, 0] == 1)

    same_terms = []
    for round_idx in range(rounds - 1):
        for lane_idx in range(gpus):
            for lhs, lhs_parts in enumerate(state_sets):
                for rhs, rhs_parts in enumerate(state_sets):
                    if lhs == rhs:
                        continue
                    overlap = len(lhs_parts & rhs_parts)
                    if overlap == 0:
                        continue
                    y = model.NewBoolVar(f"y_{lhs}_{rhs}_{round_idx}_{lane_idx}")
                    model.AddImplication(y, x[lhs, round_idx, lane_idx])
                    model.AddImplication(y, x[rhs, round_idx + 1, lane_idx])
                    same_terms.append(overlap * y)

    host_terms = []
    if objective in {"same-host", "host"}:
        u = {}
        v = {}
        for partition in partitions:
            containing = [state_idx for state_idx, parts in enumerate(state_sets) if partition in parts]
            for round_idx in range(rounds):
                u[partition, round_idx] = model.NewBoolVar(f"u_{partition}_{round_idx}")
                model.Add(u[partition, round_idx] ==
                          sum(x[state_idx, round_idx, lane_idx]
                              for state_idx in containing
                              for lane_idx in range(gpus)))
                for lane_idx in range(gpus):
                    v[partition, round_idx, lane_idx] = model.NewBoolVar(
                        f"v_{partition}_{round_idx}_{lane_idx}")
                    model.Add(v[partition, round_idx, lane_idx] ==
                              sum(x[state_idx, round_idx, lane_idx]
                                  for state_idx in containing))
        for partition in partitions:
            for round_idx in range(rounds - 1):
                for lane_idx in range(gpus):
                    h = model.NewBoolVar(f"h_{partition}_{round_idx}_{lane_idx}")
                    model.Add(h >= v[partition, round_idx + 1, lane_idx] - u[partition, round_idx])
                    model.Add(h <= v[partition, round_idx + 1, lane_idx])
                    model.Add(h <= 1 - u[partition, round_idx])
                    host_terms.append(h)

    if min_same is not None:
        model.Add(sum(same_terms) >= min_same)

    max_host = gpus * q * (rounds - 1)
    if objective == "same":
        model.Maximize(sum(same_terms))
    elif objective == "same-host":
        model.Maximize((max_host + 1) * sum(same_terms) - sum(host_terms))
    elif objective == "host":
        model.Minimize(sum(host_terms))
    else:
        raise ValueError(f"unknown objective {objective}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    start = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver.StatusName(status), elapsed, solver.BestObjectiveBound(), None

    schedule = [[-1 for _ in range(gpus)] for _ in range(rounds)]
    for state_idx in range(len(states)):
        for round_idx in range(rounds):
            for lane_idx in range(gpus):
                if solver.BooleanValue(x[state_idx, round_idx, lane_idx]):
                    schedule[round_idx][lane_idx] = state_idx

    return solver.StatusName(status), elapsed, solver.BestObjectiveBound(), schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path, help="Plan JSON or text dump containing the 88 states")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--objective", choices=["same", "same-host", "host"], default="same-host")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-same", type=int, default=None)
    parser.add_argument("--no-anchor", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    states = load_states(args.states)
    status, elapsed, bound, schedule = solve(
        states,
        args.gpus,
        args.objective,
        args.time_limit,
        args.workers,
        args.min_same,
        not args.no_anchor,
    )
    payload = {
        "status": status,
        "elapsed_s": elapsed,
        "best_bound": bound,
        "states": len(states),
        "gpus": args.gpus,
        "objective": args.objective,
    }
    if schedule is not None:
        payload.update(compute_metrics(schedule, states))
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.out is not None and schedule is not None:
        with args.out.open("w") as out:
            out.write("# Opt88 CP-SAT multi-GPU schedule\n")
            out.write("# " + json.dumps(payload, sort_keys=True) + "\n")
            for round_idx, round_states in enumerate(schedule):
                parts = [f"gpu{lane}=state{state_idx}:{list(states[state_idx])}"
                         for lane, state_idx in enumerate(round_states)]
                out.write(f"round {round_idx:02d}: " + " ".join(parts) + "\n")

    return 0 if schedule is not None else 1


if __name__ == "__main__":
    sys.exit(main())
