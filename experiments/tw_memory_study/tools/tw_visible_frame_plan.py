#!/usr/bin/env python3
"""Frame-only TW planning table. No training or GPU-feasibility claims."""
import argparse
import csv
import math
from pathlib import Path

NODES = 41652230
ROW_BYTES = 800


def plan(q, k, budget_gib):
    if q < 2 or k < q or budget_gib <= 0:
        raise ValueError("Require k >= q >= 2 and a positive frame budget")
    capacity = int(budget_gib * 2**30)
    p = max(q, math.ceil(k * NODES * ROW_BYTES / capacity))
    while k * math.ceil(NODES / p) * ROW_BYTES > capacity:
        p += 1
    frame = math.ceil(NODES / p) * ROW_BYTES
    return dict(frame_budget_gib=budget_gib, q=q, k=k, hidden=k-q, p=p,
                frame_gib=frame/2**30, frame_total_gib=k*frame/2**30,
                state_mode="one_resident_state" if p == q else "scheduled",
                status="frame_bound_only_not_runtime_validated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets-gib", type=float, nargs="+", default=[20, 32])
    parser.add_argument("--visible", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [plan(q, k, c) for c in args.budgets_gib for q in args.visible
            for k in range(q, args.max_frames+1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("c is the chosen parameter-frame cap, NOT total GPU memory.")
    print("All rows still require graph/workspace, schedule and runtime validation.")
    for c in args.budgets_gib:
        print(f"\nFrame cap c={c:g} GiB")
        print("k\t" + "\t".join(f"q={q}: p/hidden" for q in args.visible))
        for k in range(min(args.visible), args.max_frames+1):
            values = []
            for q in args.visible:
                r = plan(q,k,c) if k >= q else None
                values.append(f"{r['p']}/{r['hidden']}" if r else "--")
            print(f"{k}\t" + "\t".join(values))


if __name__ == "__main__":
    main()
