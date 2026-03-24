#!/usr/bin/env python3

import argparse
import re
import sys
from collections import defaultdict


STATE_RE = re.compile(
    r"\[logical_lane (?P<lane>\d+)\]\[state_pos (?P<state_pos>\d+)\].*?process_ms=(?P<process_ms>[0-9.]+)"
)
EPOCH_RE = re.compile(r"Epoch Runtime:\s+(?P<epoch_ms>\d+)ms")


def parse_log(path: str):
    lane_states = defaultdict(dict)
    lane_epoch_ms = {}
    epoch_values = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            state_match = STATE_RE.search(line)
            if state_match:
                lane = int(state_match.group("lane"))
                state_pos = int(state_match.group("state_pos"))
                process_ms = float(state_match.group("process_ms"))
                lane_states[lane][state_pos] = process_ms

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                epoch_values.append(int(epoch_match.group("epoch_ms")))

    if epoch_values:
        lanes = sorted(lane_states.keys())
        if len(epoch_values) == 1 and len(lanes) == 1:
            lane_epoch_ms[lanes[0]] = epoch_values[0]
        else:
            for lane, epoch_ms in zip(lanes, epoch_values):
                lane_epoch_ms[lane] = epoch_ms

    return lane_states, lane_epoch_ms


def main():
    parser = argparse.ArgumentParser(
        description="Estimate 4-GPU logical-lane epoch time and barrier from GEGE logical-lane replay logs."
    )
    parser.add_argument("logs", nargs="+", help="Lane replay log files")
    args = parser.parse_args()

    merged_states = defaultdict(dict)
    merged_epoch_ms = {}
    for path in args.logs:
        lane_states, lane_epoch_ms = parse_log(path)
        for lane, states in lane_states.items():
            merged_states[lane].update(states)
        merged_epoch_ms.update(lane_epoch_ms)

    if not merged_states:
        print("No logical-lane state timing lines found.", file=sys.stderr)
        return 1

    lanes = sorted(merged_states.keys())
    all_state_positions = sorted({state_pos for states in merged_states.values() for state_pos in states.keys()})

    print("Lane epoch runtimes (ms):")
    for lane in lanes:
        epoch_ms = merged_epoch_ms.get(lane)
        epoch_label = f"{epoch_ms}" if epoch_ms is not None else "n/a"
        print(f"lane {lane}: {epoch_label}")

    print("\nPer-round estimate:")
    predicted_epoch_ms = 0.0
    for state_pos in all_state_positions:
        round_values = {}
        for lane in lanes:
            value = merged_states[lane].get(state_pos)
            if value is not None:
                round_values[lane] = value

        if not round_values:
            continue

        round_max = max(round_values.values())
        predicted_epoch_ms += round_max
        barriers = ", ".join(
            f"lane {lane}: {round_max - value:.3f}"
            for lane, value in sorted(round_values.items())
        )
        values = ", ".join(
            f"lane {lane}: {value:.3f}"
            for lane, value in sorted(round_values.items())
        )
        print(
            f"state_pos {state_pos}: round_time_ms={round_max:.3f}; "
            f"lane_process_ms=({values}); barrier_ms=({barriers})"
        )

    print(f"\nPredicted 4-GPU epoch time (ms): {predicted_epoch_ms:.3f}")
    print(f"Predicted 4-GPU epoch time (s): {predicted_epoch_ms / 1000.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
