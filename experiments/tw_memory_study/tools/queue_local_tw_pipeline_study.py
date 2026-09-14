#!/usr/bin/env python3
"""Wait for an entire predecessor service, then run a frozen local TW study."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_local_tw_fixed_frames import cases, pipeline_cases, digest, gpu_processes, schedule_info, write_json


def predecessor_finished(properties, saved_status):
    if properties.get("LoadState") == "loaded":
        return properties.get("ActiveState") in ("inactive", "failed")
    return (properties.get("LoadState") == "not-found" and
            saved_status.get("status") in ("finished", "stopped"))


def study_rows(study):
    if study == "pipeline":
        return pipeline_cases(16, 4, 7)
    if study != "visible-hidden":
        raise ValueError("Unknown study")
    return [r for q in (7, 6, 5, 4, 3) for r in cases(q=q, min_frames=7, max_frames=7, partitions=16)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--after-unit", required=True)
    parser.add_argument("--after-status", type=Path, required=True)
    parser.add_argument("--study", choices=("visible-hidden", "pipeline"), required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--planner-python", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("At least two epochs are needed for steady timing")
    root, run, scratch = args.root.resolve(), args.run_dir.resolve(), args.scratch.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "queue_plan.json").exists():
        raise RuntimeError("Use a fresh run directory; preserve existing attempts")
    snapshot = run / "tools"
    snapshot.mkdir()
    for name in ("queue_local_tw_pipeline_study.py", "run_local_tw_fixed_frames.py",
                 "run_local_tw_fixed_frames_campaign.py", "prepare_ge2_partitioned_view.py", "plan_pipege_cover.py"):
        shutil.copy2(root / "tools" / name, snapshot / name)
    build = root / "src/ours_memory_ablation/ge2/dandelion-dev/gege/build_fixed_frames_sm86"
    inputs = [build / name for name in ("gege_train", "libge2.so", "gege_fixed_frame_buffer_test")]
    inputs += [root / "tools/prepare_ge2_partitioned_view.py"]
    reference = root / "runs/tw_p16_q4_20state_bs50k_auto10_20260824_v3"
    inputs += [reference / "ours_TW_Dot_1gpu.yaml", reference / "env_flags.sh"]
    hashes = {str(p): digest(p) for p in inputs}
    rows = study_rows(args.study)
    write_json(run / "queue_plan.json", dict(study=args.study, cases=rows, partitions=16, physical_frames=7,
               epochs=args.epochs, batch_size=50000, evaluation=False, whole_state_gpu_graph=True,
               graph_prefetch="varied" if args.study == "pipeline" else True,
               after_unit=args.after_unit, after_status=str(args.after_status), input_hashes=hashes,
               tool_hashes={p.name: digest(p) for p in snapshot.iterdir()}))

    def status(state, **extra):
        write_json(run / "status.json", dict(status=state, pid=os.getpid(), updated=time.time(), **extra))

    def check_inputs():
        for name, sha in hashes.items():
            if digest(name) != sha:
                raise RuntimeError(f"Queued input changed: {name}; not running a mixed-provenance experiment")

    results = []
    try:
        while True:
            query = subprocess.run(["systemctl", "--user", "show", args.after_unit,
                                    "--property=LoadState,ActiveState,SubState,Result"], text=True, capture_output=True)
            props = dict(line.split("=", 1) for line in query.stdout.splitlines() if "=" in line)
            try:
                previous = json.loads(args.after_status.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                previous = {}
            if predecessor_finished(props, previous):
                write_json(run / "predecessor.json", dict(unit=args.after_unit, service=props, saved_status=previous))
                break
            status("waiting_for_predecessor", after_unit=args.after_unit, service=props,
                   predecessor_status=previous.get("status"), error=query.stderr.strip() or None)
            time.sleep(30)
        while gpu_processes():
            status("waiting_for_idle_gpu")
            time.sleep(30)
        check_inputs()
        scratch.mkdir(parents=True, exist_ok=True)
        source = Path("/home/smansou2/newCode/ge2/dandelion-dev/datasets/twitter_16p_paper_10k_eval")
        data = scratch / "shared_p16"
        status("preparing_shared_p16")
        with (run / "prepare.log").open("w") as log:
            subprocess.run([sys.executable, str(snapshot / "prepare_ge2_partitioned_view.py"),
                            "--source-data-dir", str(source), "--output-dir", str(data),
                            "--num-partitions", "16", "--edge-columns", "2"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copy2(data / "partitioned_view_manifest.json", run / "data_manifest.json")
        shutil.copy2(data / "edges/train_partition_offsets.txt", run / "bucket_counts.txt")
        capacities = list(dict.fromkeys(r["q"] for r in rows))
        for q in capacities:
            check_inputs()
            schedule_root = run / "schedules" / f"q{q}"
            # Reuse certified witnesses for existing geometries, not just their counts.
            known = {3: root / "runs/tw_q3_schedules_20260913/p16",
                     4: root / "runs/maximum_overlap_validation_20260913/p16"}.get(q)
            if known is not None:
                schedule_info(known / "states.txt", known / "cover.json", q)
                target = schedule_root / "p16"
                target.mkdir(parents=True)
                for name in ("states.txt", "cover.json"):
                    shutil.copy2(known / name, target / name)
            status("running_capacity", q=q, detail_file=str(run / f"q{q}/campaign_status.json"))
            command = [sys.executable, str(snapshot / "run_local_tw_fixed_frames_campaign.py"),
                       "--root", str(root), "--run-dir", str(run / f"q{q}"),
                       "--scratch", str(scratch / f"q{q}"), "--schedule-root", str(schedule_root),
                       "--capacity", str(q), "--min-frames", "7", "--max-frames", "7",
                       "--partitions", "16", "--prepared-data-dir", str(data), "--epochs", str(args.epochs),
                       "--planner-python", str(args.planner_python), "--solver-seconds", "600"]
            if args.study == "pipeline":
                command.append("--pipeline-matrix")
            with (run / f"q{q}.log").open("w") as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            observed_path = run / f"q{q}/campaign_results.json"
            observed = {r["case"]: r for r in json.loads(observed_path.read_text())} if observed_path.exists() else {}
            for row in (r for r in rows if r["q"] == q):
                actual = observed.get(row["case"], {})
                results.append(dict(row, status=actual.get("status", "stage_failed" if child.returncode else "not_observed"),
                                    average_epoch_s=actual.get("average_epoch_s"), steady_epoch_s=actual.get("steady_epoch_s"),
                                    epochs_observed=actual.get("epochs_observed", 0),
                                    run_dir=actual.get("run_dir", str(run / f"q{q}"))))
            write_json(run / "results.json", results)
            with (run / "manifest.tsv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(results)
        shutil.rmtree(data)
        status("finished", total=len(results), valid=sum(r["status"] == "valid_timing" for r in results))
    except BaseException as error:
        status("stopped", error=str(error))
        raise


if __name__ == "__main__":
    main()
