#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-GPU Stateflow candidates against epoch runtime")
    parser.add_argument("config", help="Path to a training YAML config")
    parser.add_argument("--build-dir", help="Build directory containing gege_train and gege_stateflow_planner_analyzer")
    parser.add_argument("--train-bin", help="Path to gege_train")
    parser.add_argument("--analyzer-bin", help="Path to gege_stateflow_planner_analyzer")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs to run per candidate; last epoch is recorded as measured")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N candidates from the analyzer (0 = all)")
    parser.add_argument("--candidate", action="append", default=[], help="Exact candidate name to run; may be repeated")
    parser.add_argument("--run-root-base", help="Base directory for per-candidate model/checkpoint dirs")
    parser.add_argument("--log-dir", help="Directory for train logs")
    parser.add_argument("--output-csv", help="CSV output path")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=80.0,
        help="Minimum free space required on the run-root filesystem before starting the sweep",
    )
    parser.add_argument("--keep-run-roots", action="store_true", help="Do not delete per-candidate run roots after completion")
    parser.add_argument("--set-env", action="append", default=[], help="Extra environment variable assignment KEY=VALUE")
    return parser.parse_args()


def extract_required_value(text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Failed to find {label} in config")
    return match.group(1).strip()


def parse_config_metadata(config_path: Path) -> dict:
    text = config_path.read_text()
    metadata = {
        "dataset_dir": extract_required_value(text, r"(?m)^\s*dataset_dir:\s*(.+?)\s*$", "dataset_dir"),
        "num_partitions": int(extract_required_value(text, r"(?m)^\s*num_partitions:\s*(\d+)\s*$", "num_partitions")),
        "buffer_capacity": int(extract_required_value(text, r"(?m)^\s*buffer_capacity:\s*(\d+)\s*$", "buffer_capacity")),
        "randomly_assign_edge_buckets": extract_required_value(
            text, r"(?m)^\s*randomly_assign_edge_buckets:\s*(true|false)\s*$", "randomly_assign_edge_buckets"
        )
        == "true",
        "text": text,
    }
    return metadata


def ensure_binary(args: argparse.Namespace, explicit: str | None, fallback_name: str) -> str:
    if explicit:
        return explicit
    if args.build_dir:
        return str(Path(args.build_dir) / fallback_name)
    raise RuntimeError(f"Provide --{fallback_name.replace('_', '-')} or --build-dir")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def ensure_free_space(path: Path, min_free_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient free space at {path}: {free_gb:.1f} GiB available, need at least {min_free_gb:.1f} GiB"
        )


def rewrite_config(text: str, run_root: Path, epochs: int) -> str:
    updated = text
    replacements = [
        (r"(?m)^(\s*model_dir:\s*).*$", rf"\1{run_root}"),
        (r"(?m)^(\s*checkpoint_dir:\s*).*$", rf"\1{run_root}"),
        (r"(?m)^(\s*num_epochs:\s*)\d+\s*$", rf"\g<1>{epochs}"),
    ]
    for pattern, replacement in replacements:
        updated, count = re.subn(pattern, replacement, updated)
        if count == 0:
            raise RuntimeError(f"Failed to rewrite config field for pattern: {pattern}")
    return updated


def run_cmd(cmd: list[str], env: dict[str, str], log_path: Path | None = None) -> subprocess.CompletedProcess:
    if log_path is None:
        return subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    with log_path.open("w") as log_file:
        return subprocess.run(cmd, env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)


def parse_train_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="ignore")
    selected = re.findall(r"Stateflow planner selected family=([^\s]+)", text)
    epoch_runtimes = [int(value) for value in re.findall(r"Epoch Runtime: (\d+)ms", text)]
    init_match = re.search(r"Initialization Complete: ([0-9.]+)s", text)
    return {
        "selected_name": selected[-1] if selected else "",
        "epoch_runtimes_ms": epoch_runtimes,
        "measured_epoch_ms": epoch_runtimes[-1] if epoch_runtimes else None,
        "initialization_s": float(init_match.group(1)) if init_match else None,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    metadata = parse_config_metadata(config_path)

    train_bin = Path(ensure_binary(args, args.train_bin, "gege_train")).resolve()
    analyzer_bin = Path(ensure_binary(args, args.analyzer_bin, "gege_stateflow_planner_analyzer")).resolve()
    if not train_bin.exists():
        raise RuntimeError(f"Missing train binary: {train_bin}")
    if not analyzer_bin.exists():
        raise RuntimeError(f"Missing analyzer binary: {analyzer_bin}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    user = getpass.getuser()
    run_root_base = Path(
        args.run_root_base or f"/dev/shm/{user}/stateflow_candidate_sweep_{timestamp}"
    ).resolve()
    log_dir = Path(args.log_dir or f"/home/{user}/codex_runs/exp_logs").resolve()
    output_csv = Path(args.output_csv or (log_dir / f"stateflow_candidate_sweep_{timestamp}.csv")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    run_root_base.mkdir(parents=True, exist_ok=True)
    ensure_free_space(run_root_base, args.min_free_gb)

    base_env = os.environ.copy()
    for assignment in args.set_env:
        if "=" not in assignment:
            raise RuntimeError(f"Invalid --set-env assignment: {assignment}")
        key, value = assignment.split("=", 1)
        base_env[key] = value

    analyzer_cmd = [
        str(analyzer_bin),
        metadata["dataset_dir"],
        str(metadata["num_partitions"]),
        str(metadata["buffer_capacity"]),
        "--all-candidates",
        "--json",
    ]
    if metadata["randomly_assign_edge_buckets"]:
        analyzer_cmd.append("--random")

    print("Enumerating candidates:", " ".join(analyzer_cmd))
    analyzer_result = run_cmd(analyzer_cmd, base_env)
    if analyzer_result.returncode != 0:
        sys.stderr.write(analyzer_result.stdout)
        sys.stderr.write(analyzer_result.stderr)
        raise RuntimeError("Analyzer failed")
    candidates = json.loads(analyzer_result.stdout)

    if args.candidate:
        requested = set(args.candidate)
        candidates = [candidate for candidate in candidates if candidate["name"] in requested]
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise RuntimeError("No candidates selected for benchmark")

    fieldnames = [
        "candidate_name",
        "family",
        "variant",
        "estimated_cost",
        "admitted_partition_cost",
        "bucket_edge_cost",
        "boundary_cost",
        "lane_imbalance_cost",
        "weighted_admission_load",
        "boundary_count",
        "max_overlap",
        "total_partition_loads",
        "estimated_bucket_edges",
        "selected_name",
        "measured_epoch_ms",
        "initialization_s",
        "epoch_runtimes_ms",
        "returncode",
        "log_path",
    ]

    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with tempfile.TemporaryDirectory(prefix="stateflow_candidate_cfgs_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            for index, candidate in enumerate(candidates, start=1):
                candidate_name = candidate["name"]
                slug = sanitize_name(candidate_name.lower())
                run_root = run_root_base / slug
                run_root.mkdir(parents=True, exist_ok=True)
                ensure_free_space(run_root, args.min_free_gb)
                candidate_config = temp_dir_path / f"{slug}.yaml"
                candidate_config.write_text(rewrite_config(metadata["text"], run_root, args.epochs))
                log_path = log_dir / f"{config_path.stem}_{slug}_{timestamp}.log"

                run_env = base_env.copy()
                run_env["GEGE_STATEFLOW_PLANNER"] = "1"
                run_env["GEGE_HYBRID_COVER"] = "0"
                run_env["GEGE_STATEFLOW_FORCE_FAMILY"] = candidate["family"]
                if candidate["variant"] != "default":
                    run_env["GEGE_STATEFLOW_FORCE_VARIANT"] = candidate["variant"]
                else:
                    run_env.pop("GEGE_STATEFLOW_FORCE_VARIANT", None)

                print(f"[{index}/{len(candidates)}] running {candidate_name}")
                result = run_cmd([str(train_bin), str(candidate_config)], run_env, log_path)
                parsed = parse_train_log(log_path)
                row = {
                    "candidate_name": candidate_name,
                    "family": candidate["family"],
                    "variant": candidate["variant"],
                    "estimated_cost": candidate["estimated_cost"],
                    "admitted_partition_cost": candidate["cost_breakdown"]["admitted_partition_cost"],
                    "bucket_edge_cost": candidate["cost_breakdown"]["bucket_edge_cost"],
                    "boundary_cost": candidate["cost_breakdown"]["boundary_cost"],
                    "lane_imbalance_cost": candidate["cost_breakdown"]["lane_imbalance_cost"],
                    "weighted_admission_load": candidate["cost_breakdown"]["weighted_admission_load"],
                    "boundary_count": candidate["boundary_count"],
                    "max_overlap": candidate["max_overlap"],
                    "total_partition_loads": candidate["total_partition_loads"],
                    "estimated_bucket_edges": candidate["estimated_bucket_edges"],
                    "selected_name": parsed["selected_name"],
                    "measured_epoch_ms": parsed["measured_epoch_ms"],
                    "initialization_s": parsed["initialization_s"],
                    "epoch_runtimes_ms": ";".join(str(value) for value in parsed["epoch_runtimes_ms"]),
                    "returncode": result.returncode,
                    "log_path": str(log_path),
                }
                writer.writerow(row)
                csv_file.flush()

                if parsed["selected_name"] != candidate_name:
                    raise RuntimeError(
                        f"Forced candidate mismatch for {candidate_name}: selected '{parsed['selected_name']}'"
                    )
                if result.returncode != 0:
                    raise RuntimeError(f"Training run failed for {candidate_name}; see {log_path}")

                if not args.keep_run_roots:
                    shutil.rmtree(run_root, ignore_errors=True)

    print(f"Wrote results to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
