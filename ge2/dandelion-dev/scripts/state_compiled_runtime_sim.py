#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class Baseline:
    name: str
    epoch_s: float
    swap_s: float
    map_s: float
    neg_s: float
    states: int

    @property
    def other_s(self) -> float:
        return self.epoch_s - self.swap_s - self.map_s - self.neg_s


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


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def gib(bytes_value: int) -> float:
    return bytes_value / float(1024 ** 3)


def workspace_rows_summary(local_nodes: Optional[int], max_batch_touched: Optional[int]) -> dict[str, float]:
    if local_nodes is None or max_batch_touched is None:
        return {}

    seen_generation_bytes = local_nodes * 4
    compact_index_bytes = local_nodes * 4
    unique_ids_buf_bytes = max_batch_touched * 4
    inverse_buf_bytes = max_batch_touched * 4
    total_bytes = seen_generation_bytes + compact_index_bytes + unique_ids_buf_bytes + inverse_buf_bytes

    return {
        "local_nodes": float(local_nodes),
        "max_batch_touched": float(max_batch_touched),
        "seen_generation_gib": gib(seen_generation_bytes),
        "compact_index_gib": gib(compact_index_bytes),
        "unique_ids_buf_gib": gib(unique_ids_buf_bytes),
        "inverse_buf_gib": gib(inverse_buf_bytes),
        "workspace_total_gib": gib(total_bytes),
    }


def scenario_rows(
    baseline: Baseline,
    map_residuals: Iterable[float],
    neg_residuals: Iterable[float],
    swap_residuals: Iterable[float],
    publish_ms_per_state: float,
):
    for map_residual in map_residuals:
        for neg_residual in neg_residuals:
            for swap_residual in swap_residuals:
                publish_s = baseline.states * publish_ms_per_state / 1000.0
                residual_map_s = baseline.map_s * map_residual
                residual_neg_s = baseline.neg_s * neg_residual
                residual_swap_s = baseline.swap_s * swap_residual
                epoch_s = baseline.other_s + residual_map_s + residual_neg_s + residual_swap_s + publish_s
                yield {
                    "dataset": baseline.name,
                    "epoch_s": round(epoch_s, 3),
                    "other_s": round(baseline.other_s, 3),
                    "residual_map_s": round(residual_map_s, 3),
                    "residual_neg_s": round(residual_neg_s, 3),
                    "residual_swap_s": round(residual_swap_s, 3),
                    "publish_s": round(publish_s, 3),
                    "map_residual": map_residual,
                    "neg_residual": neg_residual,
                    "swap_residual": swap_residual,
                    "states": baseline.states,
                }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual-cost simulator for the state-compiled GE2 runtime design.")
    parser.add_argument("--dataset", choices=sorted(BASELINES.keys()), required=True)
    parser.add_argument("--map-residuals", default="1.0,0.5,0.25,0.1")
    parser.add_argument("--neg-residuals", default="1.0,0.5,0.25,0.0")
    parser.add_argument("--swap-residuals", default="1.0,0.75,0.5,0.35,0.25")
    parser.add_argument("--publish-ms-per-state", type=float, default=10.0)
    parser.add_argument("--local-nodes", type=int, default=None)
    parser.add_argument("--max-batch-touched", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = BASELINES[args.dataset]
    rows = list(
        scenario_rows(
            baseline,
            parse_float_list(args.map_residuals),
            parse_float_list(args.neg_residuals),
            parse_float_list(args.swap_residuals),
            args.publish_ms_per_state,
        )
    )
    rows.sort(key=lambda row: (row["epoch_s"], row["map_residual"], row["neg_residual"], row["swap_residual"]))

    workspace = workspace_rows_summary(args.local_nodes, args.max_batch_touched)
    if workspace:
        rows.insert(
            0,
            {
                "dataset": baseline.name,
                "epoch_s": "",
                "other_s": "",
                "residual_map_s": "",
                "residual_neg_s": "",
                "residual_swap_s": "",
                "publish_s": "",
                "map_residual": "workspace",
                "neg_residual": "",
                "swap_residual": "",
                "states": "",
                **{k: round(v, 3) for k, v in workspace.items()},
            },
        )
    write_tsv(args.out, rows)

    top_rows = [row for row in rows if row.get("epoch_s") != ""][:5]
    print(f"dataset={baseline.name}")
    print(
        f"baseline_epoch_s={baseline.epoch_s:.3f} other_s={baseline.other_s:.3f} "
        f"map_s={baseline.map_s:.3f} neg_s={baseline.neg_s:.3f} swap_s={baseline.swap_s:.3f} states={baseline.states}"
    )
    if workspace:
        print(
            "workspace_gib="
            f"seen={workspace['seen_generation_gib']:.3f} compact={workspace['compact_index_gib']:.3f} "
            f"unique_buf={workspace['unique_ids_buf_gib']:.3f} inverse_buf={workspace['inverse_buf_gib']:.3f} "
            f"total={workspace['workspace_total_gib']:.3f}"
        )
    print("top_scenarios:")
    for row in top_rows:
        print(
            f"  epoch_s={row['epoch_s']} map_residual={row['map_residual']} "
            f"neg_residual={row['neg_residual']} swap_residual={row['swap_residual']} "
            f"residual_map_s={row['residual_map_s']} residual_neg_s={row['residual_neg_s']} "
            f"residual_swap_s={row['residual_swap_s']} publish_s={row['publish_s']}"
        )


if __name__ == "__main__":
    main()
