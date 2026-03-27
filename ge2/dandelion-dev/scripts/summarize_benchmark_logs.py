#!/usr/bin/env python3
import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


TIMESTAMP_FORMAT = "%m/%d/%y %H:%M:%S.%f"
APPROX_EVAL_WARNING = "reported ranks/MRR/Hits are approximate"


def mean_or_none(values):
    return float(statistics.mean(values)) if values else None


def parse_epoch_metrics(train_log_text: str, epochs: int):
    runtime_ms = [int(value) for value in re.findall(r"Epoch Runtime:\s+(\d+)ms", train_log_text)[:epochs]]
    edges_per_second = [float(value) for value in re.findall(r"Edges per Second:\s+([0-9.]+)", train_log_text)[:epochs]]

    perf_matches = re.findall(
        r"\[perf\]\[epoch (\d+)\] swap_count=(\d+) "
        r"swap_barrier_wait_ms=([0-9.]+) "
        r"swap_update_ms=([0-9.]+) "
        r"swap_rebuild_ms=([0-9.]+) "
        r"swap_sync_wait_ms=([0-9.]+)",
        train_log_text,
    )[:epochs]

    finish_matches = re.findall(
        r"\[([^\]]+)\] ################ Finished training epoch (\d+) ################",
        train_log_text,
    )
    start_matches = re.findall(
        r"\[([^\]]+)\] ################ Starting training epoch (\d+) ################",
        train_log_text,
    )

    finish_times = {
        int(epoch): datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        for timestamp, epoch in finish_matches
    }
    start_times = {
        int(epoch): datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        for timestamp, epoch in start_matches
    }

    gap_ms = []
    for epoch in range(1, epochs):
        finish_time = finish_times.get(epoch)
        next_start = start_times.get(epoch + 1)
        if finish_time is None or next_start is None:
            continue
        gap_ms.append((next_start - finish_time).total_seconds() * 1000.0)

    swap_count = [int(match[1]) for match in perf_matches]
    swap_barrier_wait_ms = [float(match[2]) for match in perf_matches]
    swap_update_ms = [float(match[3]) for match in perf_matches]
    swap_rebuild_ms = [float(match[4]) for match in perf_matches]
    swap_sync_wait_ms = [float(match[5]) for match in perf_matches]

    return {
        "epochs_requested": epochs,
        "epochs_used": min(len(runtime_ms), len(edges_per_second), len(perf_matches)),
        "avg_epoch_runtime_ms": mean_or_none(runtime_ms),
        "avg_edges_per_second": mean_or_none(edges_per_second),
        "avg_inter_epoch_gap_ms": mean_or_none(gap_ms),
        "avg_swap_count": mean_or_none(swap_count),
        "avg_swap_barrier_wait_ms": mean_or_none(swap_barrier_wait_ms),
        "avg_swap_update_ms": mean_or_none(swap_update_ms),
        "avg_swap_rebuild_ms": mean_or_none(swap_rebuild_ms),
        "avg_swap_sync_wait_ms": mean_or_none(swap_sync_wait_ms),
    }


def parse_eval_metrics(eval_log_text: str):
    def last_float(pattern: str):
        matches = re.findall(pattern, eval_log_text)
        return float(matches[-1]) if matches else None

    return {
        "approximate_eval": APPROX_EVAL_WARNING in eval_log_text,
        "mrr": last_float(r"MRR:\s+([0-9.]+)"),
        "hits_at_1": last_float(r"Hits@1:\s+([0-9.]+)"),
        "hits_at_3": last_float(r"Hits@3:\s+([0-9.]+)"),
        "hits_at_10": last_float(r"Hits@10:\s+([0-9.]+)"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize GEGE benchmark train/eval logs.")
    parser.add_argument("--train-log", required=True, help="Path to the gege_train log")
    parser.add_argument("--eval-log", required=True, help="Path to the gege_eval log")
    parser.add_argument("--epochs", type=int, required=True, help="Number of leading epochs to average")
    parser.add_argument(
        "--format",
        choices=("json", "kv"),
        default="kv",
        help="Output format. 'kv' prints a readable summary, 'json' prints machine-readable output.",
    )
    args = parser.parse_args()

    train_log = Path(args.train_log).read_text(encoding="utf-8")
    eval_log = Path(args.eval_log).read_text(encoding="utf-8")

    summary = {
        "train_log": str(Path(args.train_log).resolve()),
        "eval_log": str(Path(args.eval_log).resolve()),
        "train": parse_epoch_metrics(train_log, args.epochs),
        "eval": parse_eval_metrics(eval_log),
    }

    if args.format == "json":
        print(json.dumps(summary, indent=2))
        return

    train = summary["train"]
    eval_metrics = summary["eval"]
    print(f"epochs_requested={train['epochs_requested']}")
    print(f"epochs_used={train['epochs_used']}")
    print(f"avg_epoch_runtime_ms={train['avg_epoch_runtime_ms']}")
    print(f"avg_edges_per_second={train['avg_edges_per_second']}")
    print(f"avg_inter_epoch_gap_ms={train['avg_inter_epoch_gap_ms']}")
    print(f"avg_swap_count={train['avg_swap_count']}")
    print(f"avg_swap_barrier_wait_ms={train['avg_swap_barrier_wait_ms']}")
    print(f"avg_swap_update_ms={train['avg_swap_update_ms']}")
    print(f"avg_swap_rebuild_ms={train['avg_swap_rebuild_ms']}")
    print(f"avg_swap_sync_wait_ms={train['avg_swap_sync_wait_ms']}")
    print(f"approximate_eval={eval_metrics['approximate_eval']}")
    print(f"mrr={eval_metrics['mrr']}")
    print(f"hits_at_1={eval_metrics['hits_at_1']}")
    print(f"hits_at_3={eval_metrics['hits_at_3']}")
    print(f"hits_at_10={eval_metrics['hits_at_10']}")


if __name__ == "__main__":
    main()
