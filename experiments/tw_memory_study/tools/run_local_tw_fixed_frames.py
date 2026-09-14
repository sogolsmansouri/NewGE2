#!/usr/bin/env python3
"""Serial, restartable fixed-frame TW timing sweep; evaluation is never run."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import yaml

NODES = 41652230
TRAIN = 1468345182
SOURCE_SHA = "d978663192467e9da277fadd5d374f28076a7a07f46a849ae907cff7902ccd95"
BASE = "57c73b5e0a895eb74db8fb22df300e34cae00b2f"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def cases(q=4, min_frames=None, max_frames=10, frame_budget_gib=20, baseline_p=None, partitions=None,
          frame_policy="fixed"):
    if min_frames is None:
        min_frames = q
    if q < 2 or min_frames < q or max_frames < min_frames or frame_budget_gib <= 0:
        raise ValueError("Require q>=2, max_frames>=min_frames>=q and positive frame budget")
    if partitions is not None and partitions < q:
        raise ValueError("Fixed partition count must be at least q")
    if frame_policy not in ("fixed", "shared"):
        raise ValueError("Unknown frame policy")
    rows = []
    for k in range(min_frames, max_frames + 1):
        if partitions is None:
            p = max(q, math.ceil(k * NODES * 800 / (frame_budget_gib * 2**30)))
            while k * math.ceil(NODES / p) * 800 > frame_budget_gib * 2**30:
                p += 1
        else:
            p = partitions
            if k * math.ceil(NODES / p) * 800 > frame_budget_gib * 2**30:
                raise ValueError("Fixed p,k exceed the parameter-frame budget")
        h = k - q
        if frame_policy == "shared" and h:
            rows.append(dict(case=f"k{k}_p{p}_shared{h}", k=k, p=p, q=q, hp=h, hs=h,
                             shared_hidden=True, parameter_pipeline=True, graph_prefetch=True,
                             frame_bytes=math.ceil(NODES / p) * 800))
            continue
        splits = [(h // 2, h - h // 2)]
        if h % 2:
            splits.append((h - h // 2, h // 2))
        for hp, hs in splits:
            rows.append(dict(case=f"k{k}_p{p}_hp{hp}_hs{hs}", k=k, p=p, q=q, hp=hp, hs=hs,
                             frame_bytes=math.ceil(NODES / p) * 100 * 4 * 2))
    if baseline_p is not None:
        if baseline_p < q or q * math.ceil(NODES / baseline_p) * 800 > frame_budget_gib * 2**30:
            raise ValueError("Baseline violates visible-frame budget")
        row = dict(case=f"k{q}_p{baseline_p}_hp0_hs0", k=q, p=baseline_p, q=q, hp=0, hs=0,
                   frame_bytes=math.ceil(NODES / baseline_p) * 800)
        if row["case"] not in {r["case"] for r in rows}:
            rows.append(row)
    return sorted(rows, key=lambda r: (r["p"], r["k"], r["hp"]))


def schedule_info(schedule, proof, q=4):
    states = [tuple(json.loads(line.removeprefix("state="))) for line in schedule.read_text().splitlines() if line.strip()]
    report = json.loads(proof.read_text())
    p = report["partitions"]
    assert report["capacity"] == q
    assert all(len(s) == q and len(set(s)) == q and all(0 <= x < p for x in s) for s in states)
    assert len(set(tuple(sorted(s)) for s in states)) == len(states)
    pairs = {tuple(sorted(pair)) for s in states for pair in itertools.combinations(s, 2)}
    assert len(pairs) == p * (p - 1) // 2
    overlap = sum(len(set(a) & set(b)) for a, b in zip(states, states[1:]))
    assert report["state_count"] == len(states) and report["total_overlap"] == overlap
    assert report["optimality"]["state_count_optimal"] is True
    assert report["optimality"]["overlap_optimality"] == "proven"
    assert digest(schedule) == report["schedule_sha256"]
    return dict(states=len(states), overlap=overlap, schedule_sha256=digest(schedule),
                max_admits=max((q - len(set(a) & set(b)) for a, b in zip(states, states[1:])), default=0))


def pipeline_cases(partitions=16, q=4, k=7):
    if not 2 <= q <= partitions or k <= q:
        raise ValueError("Pipeline controls require valid p,q and k>q")
    return [dict(case=f"param{int(param)}_graph{int(graph)}", p=partitions, q=q, k=k,
                 hp=k-q if param else 0, hs=k-q, shared_hidden=param,
                 parameter_pipeline=param, graph_prefetch=graph,
                 frame_bytes=math.ceil(NODES / partitions) * 800)
            for param, graph in ((False, False), (True, False), (False, True), (True, True))]


def environment(root, run, row, nodes):
    base = {key: value for key, value in os.environ.items() if not key.startswith(("GEGE_", "CONDA", "PYTHON"))}
    flags = run / "artifacts" / "reference_flags.sh"
    output = subprocess.check_output(["bash", "--noprofile", "--norc", "-c", 'source "$1"; env -0', "bash", str(flags)], env=base)
    env = dict(item.split("=", 1) for item in output.decode().split("\0") if "=" in item)
    prefix = Path(os.environ.get("TW_RUNTIME_ENV", root / "envs/ours_py39_cuda121"))
    binaries = run / "artifacts"
    env.update({
        "PATH": f"{prefix}/bin:/usr/bin:/bin", "LD_LIBRARY_PATH": f"{binaries}:{prefix}/lib:{prefix}/lib/python3.9/site-packages/torch/lib",
        "LD_PRELOAD": str(binaries / "libge2.so"), "CUDA_VISIBLE_DEVICES": os.environ.get("TW_PHYSICAL_GPU", "0"),
        "GEGE_FRAME_CACHE_FIXED_PRELOAD_FRAMES": "-1" if row.get("shared_hidden") else str(row["hp"]),
        "GEGE_FRAME_CACHE_HIDDEN_FRAMES": str(row["k"] - row["q"]),
        "GEGE_FRAME_CACHE_MAX_STALE_BACKLOG": str(row["hs"]),
        "GEGE_FRAME_CACHE_AUTO_PIPELINE_FRAMES": "0", "GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD": str(int(row["k"] > row["q"])),
        "GEGE_FRAME_CACHE_STRICT_FRAME_BUDGET": "1",
        "GEGE_SINGLE_GPU_ASYNC_EVICT_WRITEBACK": "0", "GEGE_MULTI_GPU_ASYNC_ADMIT_PRELOAD": "0",
        "GEGE_FRAME_CACHE_DEFER_STALE_WRITEBACK_UNTIL_PRELOAD": "0",
        "GEGE_PARTITION_BUFFER_PIPELINE_TIMING": "1", "GEGE_PARTITION_BUFFER_SWAP_TIMING": "1",
        "GEGE_PARTITION_BUFFER_REMAP_BREAKDOWN_TIMING": "1",
        "GEGE_UNIQUE_BITMAP_NUM_NODES": str(nodes), "GEGE_STATEFLOW_MAX_ADMITS": str(row["max_admits"]),
        "GEGE_BOUNDED_STATE_ORDER_FILE": str(run / "artifacts" / f"p{row['p']}.txt"),
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
    })
    if "parameter_pipeline" in row:
        env.update(GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=str(int(row["parameter_pipeline"])),
                   GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=str(int(row["parameter_pipeline"])),
                   GEGE_STARTUP_TIMING="1")
    if "TW_RUNTIME_PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["TW_RUNTIME_PYTHONPATH"]
    return env


def gpu_processes():
    selected = os.environ.get("TW_PHYSICAL_GPU")
    if selected is not None:
        uuid = subprocess.check_output(["nvidia-smi", "--id=" + selected, "--query-gpu=uuid", "--format=csv,noheader"],
                                       text=True, timeout=20).strip()
        if not uuid.startswith("GPU-") or "\n" in uuid:
            raise RuntimeError("A single physical GPU must be selected")
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                      text=True, timeout=20)
        return [int(row[1].strip()) for row in csv.reader(out.splitlines()) if len(row) == 2 and row[0].strip() == uuid]
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


def summarize(text, row, epochs, rc, foreign):
    q = row.get("q", 4)
    times = [int(x) / 1000 for x in re.findall(r"Epoch Runtime:\s*(\d+)ms", text)]
    allocations = [dict(zip(("visible", "preload", "stale", "physical", "partition_rows", "width", "bytes"), map(int, x)))
                   for x in re.findall(r"\[fixed-frame-budget\].*?visible=(\d+) preload=(\d+) stale=(\d+) physical=(\d+) partition_rows=(\d+) width=(\d+) bytes=(\d+)", text)]
    allocation_ok = len(allocations) == 2 and all((a["visible"], a["preload"], a["stale"], a["physical"]) ==
                                               (q, row["hp"], row["hs"], row["k"]) for a in allocations)
    startup = [dict(zip(("visible_rows", "physical_rows", "width", "hidden"), map(int, x))) for x in re.findall(
        r"\[startup-timing\]\[MemPartitionBuffer::ctor\] deferred backing allocation device=cuda:0 visible_rows=(\d+) physical_rows=(\d+) dim=(\d+) pinned=\w+ hidden_frames=(\d+)", text)]
    if row.get("shared_hidden"):
        allocation_ok = len(startup) == 2 and all(a["visible_rows"] % q == 0 and
                        a["physical_rows"] == a["visible_rows"] // q * row["k"] and
                        a["hidden"] == row["k"] - q for a in startup)
    orders = re.findall(r"Ordering states=(\d+) transitions=(\d+) total_buckets=(\d+) max_admits=(\d+) transition_admits=(\d+)", text, re.IGNORECASE)
    order_ok = bool(orders) and all((int(x[0]), int(x[2]), int(x[4])) ==
                                   (row["states"], row["p"]**2, q * (row["states"] - 1) - row["overlap"]) for x in orders)
    # Fallbacks are expected when hp or hs is below the peak transition width.
    stale_peak = max([int(x) for x in re.findall(r"stale_backlog_after_publish_max=(\d+)", text)] or [0])
    complete = rc == 0 and len(times) == epochs
    control_ok = True
    if "parameter_pipeline" in row:
        graph_modes = re.findall(r"\[outer-update[^\]]*\].*?prefetch=(true|false)", text)
        swap_preloads = re.findall(r"\bpreloaded_admit=(true|false)", text)
        hidden_publish = any(int(x) for x in re.findall(r"\bhidden_publish_parts=(\d+)", text))
        delayed = "deferred_stale_writeback=true" in text
        control_ok = bool(graph_modes) and all(x == str(row["graph_prefetch"]).lower() for x in graph_modes)
        if row["parameter_pipeline"]:
            control_ok = control_ok and "true" in swap_preloads and hidden_publish and delayed
        else:
            control_ok = control_ok and bool(swap_preloads) and "true" not in swap_preloads and not hidden_publish and not delayed
        # A partial admit may copy host data directly to an evicted visible slot.
        # Only the preload-consume range describes an extra CUDA admission stage.
        stage_rows = re.findall(r"\[partition-buffer-preload-consume\][^\n]*?\brows=(\d+)", text)
        if row.get("shared_hidden") and any(int(x) for x in stage_rows):
            control_ok = False
    valid = complete and allocation_ok and order_ok and control_ok and not foreign and stale_peak <= row["hs"]
    return dict(status="valid_timing" if valid else "failed_or_invalid", exit_code=rc, epochs_observed=len(times),
                epoch_times_s=times, average_epoch_s=statistics.mean(times) if times else None,
                steady_epoch_s=statistics.mean(times[1:]) if len(times) > 1 else None,
                evaluation="not_requested", accuracy_equivalence="not_tested", frame_allocations=allocations,
                allocation_ok=allocation_ok, order_ok=order_ok, control_ok=control_ok, startup_frame_allocations=startup,
                stale_backlog_peak=stale_peak,
                foreign_gpu_pids=sorted(foreign))


def graph_memory_preflight(run, row, data, total_gpu_bytes, fraction=1.0):
    """Payload lower bound only: never interpret a passing check as a peak-memory guarantee."""
    counts = np.loadtxt(data / "edges/train_partition_offsets.txt", dtype=np.int64).reshape(row["p"], row["p"])
    states = [json.loads(line.removeprefix("state=")) for line in
              (run / "artifacts" / f"p{row['p']}.txt").read_text().splitlines() if line.strip()]
    graph_bytes = [int(counts[np.ix_(s, s)].sum()) * 2 * 8 for s in states]
    frame_bytes = row["k"] * row["frame_bytes"]
    pairs = [a + b for a, b in zip(graph_bytes, graph_bytes[1:])]
    # Current runtime keeps current and next graphs alive during GPU graph prefetch.
    required = frame_bytes + max(graph_bytes + (pairs if row.get("graph_prefetch", True) else []))
    if not 0 < fraction <= 1:
        raise ValueError("Memory budget fraction must be in (0,1]")
    allowed_bytes = int(total_gpu_bytes * fraction)
    return dict(status="infeasible_payload" if required > allowed_bytes else "requires_runtime_memory_gate",
                total_gpu_bytes=total_gpu_bytes, frame_bytes=frame_bytes,
                allowed_bytes=allowed_bytes, memory_budget_fraction=fraction,
                state_graph_bytes=graph_bytes, minimum_peak_payload_bytes=required,
                excludes="CUDA context, batch/workspace, maps, graph/remapping temporaries, allocator reserve",
                graph_prefetch=row.get("graph_prefetch", True), whole_state_gpu_graph=True)


def validate_prepared_data(data, partitions):
    manifest = json.loads((data / "partitioned_view_manifest.json").read_text())
    metadata = yaml.safe_load((data / "dataset.yaml").read_text())
    if (manifest["num_partitions"] != partitions or manifest["num_nodes"] != NODES or
            manifest["edge_columns"] != 2 or metadata["num_train"] != TRAIN or metadata["num_nodes"] != NODES or
            manifest["splits"]["train"]["source_sha256"] != SOURCE_SHA):
        raise ValueError("Prepared data geometry/source does not match the experiment")
    for split in manifest["splits"].values():
        if not split["physically_bucket_ordered"]:
            raise ValueError("Prepared data is not bucket ordered")
        if digest(data / "edges" / split["output_path"]) != split["output_sha256"]:
            raise ValueError("Prepared edge data hash mismatch")
        if digest(data / "edges" / split["partition_offsets"]) != split["partition_offsets_sha256"]:
            raise ValueError("Prepared bucket-count hash mismatch")
    return manifest


def execute(root, run, row, data, model, epochs, tiny=False):
    directory = run / row["case"]
    directory.mkdir(exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists() and json.loads(result_path.read_text())["status"] == "valid_timing":
        return json.loads(result_path.read_text())
    if (directory / "train.log").exists():
        raise RuntimeError(f"Incomplete prior attempt at {directory}; use a new run directory, preserving evidence")
    if gpu_processes():
        raise RuntimeError("GPU occupied; refusing concurrent timing")
    cfg = yaml.safe_load((run / "artifacts/reference.yaml").read_text())
    ds = yaml.safe_load((data / "dataset.yaml").read_text())
    cfg["storage"]["dataset"] = {"dataset_dir": str(data) + "/", **{k: int(ds[k]) for k in ("num_nodes", "num_edges", "num_relations", "num_train", "num_valid", "num_test")}}
    cfg["storage"]["embeddings"]["options"].update(num_partitions=row["p"], buffer_capacity=row["q"], prefetching=False)
    cfg["storage"].update(model_dir=str(model) + "/", prefetch=row.get("graph_prefetch", True))
    cfg["training"].update(num_epochs=epochs, batch_size=50000, save_model=False)
    cfg["evaluation"].update(epochs_per_eval=epochs + 1, checkpoint_dir=str(model) + "/")
    if tiny:
        cfg["training"]["batch_size"] = 512
        cfg["training"]["negative_sampling"].update(num_chunks=4, negatives_per_positive=32, superbatch_negative_plan_batches=1)
    config = directory / "effective_config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = environment(root, run, row, ds["num_nodes"])
    if tiny:
        env["GEGE_BATCHED_NEGATIVE_PLAN_BATCHES"] = "1"
    write_json(directory / "invocation.json", {k: v for k, v in env.items() if k.startswith("GEGE_") or k in ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")})
    write_json(directory / "provenance.json", dict(row, config_sha256=digest(config), data_manifest=json.loads((data / "partitioned_view_manifest.json").read_text())))
    if not tiny:
        # Probe in a short-lived child; the supervisor must not retain a CUDA context.
        prefix = Path(os.environ.get("TW_RUNTIME_ENV", root / "envs/ours_py39_cuda121"))
        gpu_bytes = int(subprocess.check_output([str(prefix / "bin/python"), "-c",
                         "import torch; print(torch.cuda.get_device_properties(0).total_memory)"], env=env, text=True))
        memory = graph_memory_preflight(run, row, data, gpu_bytes,
                                        float(os.environ.get("TW_GPU_MEMORY_FRACTION", "1")))
        write_json(directory / "memory_preflight.json", memory)
        if memory["status"] == "infeasible_payload":
            result = dict(status="infeasible_payload", exit_code=None, epochs_observed=0, epoch_times_s=[],
                          average_epoch_s=None, steady_epoch_s=None, evaluation="not_requested",
                          accuracy_equivalence="not_tested", case=row["case"], tiny=False)
            write_json(result_path, result)
            print(row["case"], "infeasible_payload", memory["minimum_peak_payload_bytes"], flush=True)
            return result
    foreign = set()
    start = time.monotonic()
    model.mkdir(parents=True, exist_ok=False)
    command = [str(run / "artifacts/gege_train"), str(config)]
    with (directory / "train.log").open("w") as log, (directory / "gpu_samples.csv").open("w") as samples:
        samples.write("timestamp, memory_used_mib, utilization_percent, power_w\n")
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        write_json(run / "status.json", dict(status="training", case=row["case"], pid=process.pid, updated=time.time()))
        while process.poll() is None:
            try:
                selected = ["--id=" + os.environ["TW_PHYSICAL_GPU"]] if "TW_PHYSICAL_GPU" in os.environ else []
                samples.write(subprocess.check_output(["nvidia-smi", *selected, "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=20))
                samples.flush()
                foreign.update(pid for pid in gpu_processes() if pid != process.pid)
                if foreign or time.monotonic() - start > 7200 or time.time() >= float(os.environ.get("TW_DEADLINE_EPOCH", "inf")):
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            except (subprocess.SubprocessError, OSError):
                foreign.add(-1)  # Missing isolation evidence is not a clean timing.
                process.terminate()
            time.sleep(1 if tiny else 5)
        rc = process.wait()
    result = summarize((directory / "train.log").read_text(errors="replace"), row, epochs, rc, foreign)
    text = (directory / "train.log").read_text(errors="replace")
    completed_edge_counts = [int(a) for a, b in re.findall(r"Edges processed:\s*\[(\d+)/(\d+)\]", text) if a == b]
    result["completed_epoch_edge_counts"] = completed_edge_counts
    if completed_edge_counts != [ds["num_train"]] * epochs:
        result["status"] = "failed_or_invalid"
    result.update(wall_s=time.monotonic() - start, case=row["case"], tiny=tiny)
    if tiny:
        files = list(model.rglob("embeddings.bin"))
        if files:
            shutil.copy2(files[0], directory / "embeddings.bin")
    write_json(result_path, result)
    # These are this sweep's throwaway model files, never source/checkpoint data.
    shutil.rmtree(model)
    print(row["case"], result["status"], result["epoch_times_s"], flush=True)
    return result


def freeze(root, run, rows, schedule_root):
    artifacts = run / "artifacts"
    if artifacts.exists():
        pinned = json.loads((artifacts / "hashes.json").read_text())
        assert all(digest(artifacts / name) == sha for name, sha in pinned.items())
        return
    artifacts.mkdir()
    repo = root / "src/ours_memory_ablation"
    gege = repo / "ge2/dandelion-dev/gege"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", BASE, commit], check=True)
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Commit experiment changes before freezing a new build")
    build = gege / "build_fixed_frames_sm86"
    attestation = json.loads((build / "experiment_build.json").read_text())
    assert attestation["commit"] == commit
    assert all(digest(repo / name) == sha for name, sha in attestation["source_files"].items())
    assert all(digest(build / name) == sha for name, sha in attestation["binaries"].items())
    for name in ("gege_train", "libge2.so", "gege_fixed_frame_buffer_test"):
        shutil.copy2(gege / "build_fixed_frames_sm86" / name, artifacts / name)
    reference = root / "runs/tw_p16_q4_20state_bs50k_auto10_20260824_v3"
    shutil.copy2(reference / "ours_TW_Dot_1gpu.yaml", artifacts / "reference.yaml")
    shutil.copy2(reference / "env_flags.sh", artifacts / "reference_flags.sh")
    (artifacts / "source.patch").write_bytes(subprocess.check_output(["git", "-C", str(repo), "diff", "--", "ge2/dandelion-dev/gege"]))
    shutil.copy2(gege / "src/cpp/include/common/pipeline_nvtx.h", artifacts / "pipeline_nvtx.h")
    shutil.copy2(gege / "src/cpp/tests/fixed_frame_buffer_test.cpp", artifacts / "fixed_frame_buffer_test.cpp")
    shutil.copy2(gege / "build_fixed_frames_sm86/CMakeCache.txt", artifacts / "CMakeCache.txt")
    shutil.copy2(build / "experiment_build.json", artifacts / "experiment_build.json")
    shutil.copy2(Path(__file__), artifacts / "runner.py")
    shutil.copy2(root / "tools/prepare_ge2_partitioned_view.py", artifacts / "prepare_ge2_partitioned_view.py")
    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "ge2/dandelion-dev/gege"], text=True).splitlines()
    write_json(artifacts / "source_files.json", {name: digest(repo / name) for name in tracked if (repo / name).is_file()})
    (artifacts / "base_commit.txt").write_text(BASE + "\n")
    (artifacts / "source_commit.txt").write_text(commit + "\n")
    for p in sorted({r["p"] for r in rows}):
        source = schedule_root / f"p{p}"
        schedule_info(source / "states.txt", source / "cover.json", rows[0]["q"])
        shutil.copy2(source / "states.txt", artifacts / f"p{p}.txt")
        shutil.copy2(source / "cover.json", artifacts / f"p{p}.json")
    write_json(artifacts / "hashes.json", {p.name: digest(p) for p in artifacts.iterdir() if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate-run-dir", type=Path)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--min-frames", type=int)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--frame-budget-gib", type=float, default=20)
    parser.add_argument("--baseline-p", type=int)
    parser.add_argument("--schedule-root", type=Path)
    parser.add_argument("--only-partitions", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--partitions", type=int)
    parser.add_argument("--prepared-data-dir", type=Path)
    parser.add_argument("--pipeline-matrix", action="store_true")
    parser.add_argument("--frame-policy", choices=("fixed", "shared"), default="fixed")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.smoke and args.prepared_data_dir is not None:
        parser.error("The smoke gate must use its small fixture, not the full prepared data")
    epochs = 5 if args.smoke else args.epochs
    root, run, scratch = args.root.resolve(), args.run_dir.resolve(), args.scratch.resolve()
    run.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    lock = (run / "sweep.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = cases(args.capacity, args.min_frames, args.max_frames, args.frame_budget_gib, args.baseline_p, args.partitions,
                 args.frame_policy)
    if args.pipeline_matrix:
        if args.partitions is None or args.min_frames != args.max_frames:
            parser.error("--pipeline-matrix requires fixed --partitions and equal min/max frames")
        rows = pipeline_cases(args.partitions, args.capacity, args.max_frames)
    if args.only_partitions is not None:
        rows = [r for r in rows if r["p"] in args.only_partitions]
        if not rows:
            raise ValueError("No cases match --only-partitions")
    schedule_root = args.schedule_root or root / "runs/maximum_overlap_validation_20260913"
    freeze(root, run, rows, schedule_root)
    if not args.smoke:
        if args.gate_run_dir is None:
            raise RuntimeError("A completed runtime and value-test gate is required")
        gate = args.gate_run_dir.resolve()
        gate_status = json.loads((gate / "status.json").read_text())
        value_tests = json.loads((gate / "value_tests.json").read_text())
        assert gate_status.get("valid") == len(rows) and gate_status.get("total") == len(rows)
        assert len(value_tests) == len(rows) and all(r["exit_code"] == 0 for r in value_tests)
        assert {r["case"] for r in value_tests} == {r["case"] for r in rows}
        assert json.loads((gate / "plan.json").read_text())["capacity"] == args.capacity
        for name in ("gege_train", "libge2.so", "gege_fixed_frame_buffer_test", "runner.py", "prepare_ge2_partitioned_view.py"):
            assert digest(gate / "artifacts" / name) == digest(run / "artifacts" / name)
        write_json(run / "gate.json", dict(run_dir=str(gate), status=gate_status, value_tests=value_tests))
    source = Path("/home/smansou2/newCode/ge2/dandelion-dev/datasets/twitter_16p_paper_10k_eval")
    if args.smoke:
        source = scratch / "fixture"
        (source / "edges").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1987)
        for split, size in (("train", 32768), ("validation", 256), ("test", 256)):
            rng.integers(0, 1024, size=(size, 2), dtype=np.int32).tofile(source / "edges" / f"{split}_edges.bin")
        (source / "dataset.yaml").write_text(yaml.safe_dump(dict(num_nodes=1024, num_edges=32768, num_train=32768, num_valid=256, num_test=256, num_relations=1)))
    for row in rows:
        row.update(schedule_info(run / "artifacts" / f"p{row['p']}.txt", run / "artifacts" / f"p{row['p']}.json", args.capacity))
        row["preload_exceeds_single_transition"] = row["hp"] > row["max_admits"]
        row["stale_exceeds_single_transition"] = row["hs"] > row["max_admits"]
    write_json(run / "plan.json", dict(cases=rows, capacity=args.capacity, frame_budget_gib=args.frame_budget_gib, gpu_nominal_gib=24,
                                      epochs=epochs, batch_size=512 if args.smoke else 50000, evaluation=False))
    completed = []
    value_tests = []
    try:
        for p, group in itertools.groupby(rows, key=lambda r: r["p"]):
            if args.prepared_data_dir is not None:
                data = args.prepared_data_dir.resolve()
                write_json(run / "status.json", dict(status="verifying_shared_data", p=p, pid=os.getpid(), updated=time.time()))
                manifest = validate_prepared_data(data, p)
            else:
                data = scratch / f"data_p{p}"
                write_json(run / "status.json", dict(status="repartitioning", p=p, pid=os.getpid(), updated=time.time()))
                with (run / f"prepare_p{p}.log").open("a") as log:
                    subprocess.run([sys.executable, str(run / "artifacts/prepare_ge2_partitioned_view.py"), "--source-data-dir", str(source),
                                    "--output-dir", str(data), "--num-partitions", str(p), "--edge-columns", "2"], stdout=log, stderr=subprocess.STDOUT, check=True)
                manifest = json.loads((data / "partitioned_view_manifest.json").read_text())
            if not args.smoke:
                assert manifest["splits"]["train"]["source_sha256"] == SOURCE_SHA
                assert yaml.safe_load((data / "dataset.yaml").read_text())["num_train"] == TRAIN
            shutil.copy2(data / "partitioned_view_manifest.json", run / f"data_p{p}.json")
            shutil.copy2(data / "edges/train_partition_offsets.txt", run / f"data_p{p}_bucket_counts.txt")
            for row in group:
                if args.smoke:
                    directory = run / row["case"]
                    directory.mkdir(exist_ok=True)
                    with (directory / "value_test.log").open("w") as log:
                        value = subprocess.run([str(run / "artifacts/gege_fixed_frame_buffer_test"), str(p),
                                                str(run / "artifacts" / f"p{p}.txt"), str(scratch / "value_test.bin")],
                                               env=environment(root, run, row, 1024), stdout=log, stderr=subprocess.STDOUT,
                                               timeout=120)
                    value_tests.append(dict(case=row["case"], exit_code=value.returncode))
                    write_json(run / "value_tests.json", value_tests)
                    if value.returncode != 0:
                        raise RuntimeError(f"Frame value gate failed: {row['case']}")
                result = execute(root, run, row, data, scratch / row["case"], epochs, tiny=args.smoke)
                completed.append(dict(row, **{k: result[k] for k in ("status", "average_epoch_s", "steady_epoch_s", "epochs_observed")}))
                write_json(run / "results.json", completed)
                with (run / "manifest.tsv").open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(completed[0]), delimiter="\t")
                    writer.writeheader()
                    writer.writerows(completed)
                if args.smoke and result["status"] != "valid_timing":
                    raise RuntimeError(f"Runtime gate failed: {row['case']}")
            if args.prepared_data_dir is None:
                shutil.rmtree(data)
        write_json(run / "status.json", dict(status="finished", valid=sum(r["status"] == "valid_timing" for r in completed),
                                             total=len(completed), updated=time.time()))
    except BaseException as error:
        write_json(run / "status.json", dict(status="stopped", error=str(error), updated=time.time()))
        raise


if __name__ == "__main__":
    main()
