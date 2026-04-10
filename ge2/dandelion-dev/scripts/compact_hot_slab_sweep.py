#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path


def parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_report(stdout: str) -> dict[str, str]:
    report = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        report[key.strip()] = value.strip()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hot/cold positive slab settings using the compact replay validator.")
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ordering", default="CUSTOM")
    parser.add_argument("--num-partitions", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=4)
    parser.add_argument("--fine-to-coarse-ratio", type=int, default=1)
    parser.add_argument("--num-cache-partitions", type=int, default=0)
    parser.add_argument("--randomly-assign-edge-buckets", type=int, default=0)
    parser.add_argument("--active-devices", type=int, default=1)
    parser.add_argument("--state-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--window-batches", default="2,4,8")
    parser.add_argument("--hot-touch-targets", default="0.50,0.75,0.90")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows = parse_list(args.window_batches, int)
    targets = parse_list(args.hot_touch_targets, float)

    rows: list[dict[str, object]] = []
    for window in windows:
        for target in targets:
            cmd = [
                str(args.validator),
                str(args.dataset_dir),
                args.ordering,
                str(args.num_partitions),
                str(args.buffer_capacity),
                str(args.fine_to_coarse_ratio),
                str(args.num_cache_partitions),
                str(args.randomly_assign_edge_buckets),
                str(args.active_devices),
                "--batch-size",
                str(args.batch_size),
                "--state-idx",
                str(args.state_idx),
                "--window-batches",
                str(window),
                "--hot-touch-target",
                str(target),
            ]
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            report = parse_report(completed.stdout)

            row = {
                "dataset": args.dataset_name,
                "state_idx": args.state_idx,
                "window_batches": window,
                "hot_touch_target": target,
                "active_partition_rows": int(report["active_partition_rows"]),
                "positive_unique_rows": int(report["positive_unique_rows"]),
                "positive_batch_touch_reuse": float(report["positive_batch_touch_reuse"]),
                "plan_total_gib": float(report["plan.total_gib"]),
                "workspace_positive_only_total_gib": float(report["workspace.positive_only.total_gib"]),
                "avg_hot_rows_fraction": float(report["hot_window.avg_hot_rows_fraction"]),
                "avg_hot_rows": float(report["hot_window.avg_hot_rows"]),
                "avg_covered_touch_fraction": float(report["hot_window.avg_covered_touch_fraction"]),
                "avg_batch_hot_unique_fraction": float(report["hot_window.avg_batch_hot_unique_fraction"]),
                "avg_batch_cold_unique_fraction": float(report["hot_window.avg_batch_cold_unique_fraction"]),
                "avg_batch_cold_unique_rows": float(report["hot_window.avg_batch_cold_unique_rows"]),
                "max_batch_cold_unique_rows": float(report["hot_window.max_batch_cold_unique_rows"]),
                "compile_validate_ms": float(report["compile_validate_ms"]),
            }
            rows.append(row)

    rows.sort(key=lambda row: (
        row["avg_batch_cold_unique_rows"],
        row["avg_hot_rows_fraction"],
        -row["avg_covered_touch_fraction"],
        row["window_batches"],
        row["hot_touch_target"],
    ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"dataset={args.dataset_name}")
    print(f"rows={len(rows)}")
    print("top_scenarios:")
    for row in rows[:5]:
        print(
            f"  W={row['window_batches']} target={row['hot_touch_target']:.2f} "
            f"avg_hot_rows_fraction={row['avg_hot_rows_fraction']:.6f} "
            f"avg_batch_cold_unique_rows={row['avg_batch_cold_unique_rows']:.1f} "
            f"avg_batch_cold_unique_fraction={row['avg_batch_cold_unique_fraction']:.6f} "
            f"avg_batch_hot_unique_fraction={row['avg_batch_hot_unique_fraction']:.6f} "
            f"avg_covered_touch_fraction={row['avg_covered_touch_fraction']:.6f}"
        )


if __name__ == "__main__":
    main()
