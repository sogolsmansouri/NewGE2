#!/usr/bin/env python3
"""Exercise the original GE2 library against independent per-batch equations."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "template", "work", "results", "env"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--observe-padding", action="store_true")
    args = parser.parse_args()
    allocation = subprocess.check_output(["scontrol", "show", "job", args.job, "-o"], text=True)
    if "JobState=RUNNING " not in allocation or f"UserId={os.environ['USER']}(" not in allocation:
        raise RuntimeError("A running owned allocation is required")
    if f"NodeList={os.uname().nodename.split('.')[0]} " not in allocation:
        raise RuntimeError("Allocation is on another node")
    apps = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-compute-apps=pid",
                                    "--format=csv,noheader"], text=True).strip()
    if apps:
        raise RuntimeError("Selected GPU is in use: " + apps)
    args.work.mkdir(parents=True, exist_ok=False)
    args.results.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GEGE_", "PYTHON", "CONDA"))}
    site = args.env / "lib/python3.9/site-packages"
    env.update(PATH=f"{args.env}/bin:/usr/bin:/bin", CUDA_VISIBLE_DEVICES=args.gpu,
               LD_LIBRARY_PATH=f"{site}/gege:{site}/torch/lib:{args.env}/lib",
               PYTHONPATH=str(site), GEGE_NO_BINDINGS="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    report = dict(allocation=allocation, binary_sha256=sha(args.binary),
                  library_sha256=sha(site / "gege/libge2.so"), cases=[])
    template = yaml.safe_load(args.template.read_text())
    for model in ("Dot", "DistMult", "ComplEx"):
        for nodes in (512, 513):
            case = f"{model.lower()}_{nodes}"
            data = args.work / case / "data"
            (data / "edges").mkdir(parents=True)
            (data / "nodes").mkdir()
            rng = np.random.default_rng(1930)
            size, relations, blocks = (nodes + 15) // 16, 1 if model == "Dot" else 7, []
            for src in range(16):
                for dst in range(16):
                    h = rng.integers(src * size, min((src + 1) * size, nodes), size=17)
                    t = rng.integers(dst * size, min((dst + 1) * size, nodes), size=17)
                    r = rng.integers(relations, size=17)
                    blocks.append(np.column_stack((h, t) if model == "Dot" else (h, r, t)))
            edges = np.concatenate(blocks).astype("<i4")
            edges.tofile(data / "edges/train_edges.bin")
            (data / "edges/train_partition_offsets.txt").write_text("17\n" * 256)
            (data / "nodes/node_mapping.txt").write_text("".join(f"{i},{i}\n" for i in range(nodes)))
            (data / "edges/relation_mapping.txt").write_text("".join(f"{i},{i}\n" for i in range(relations)))
            meta = dict(dataset_dir=str(data) + "/", num_nodes=nodes, num_relations=relations,
                        num_edges=len(edges), num_train=len(edges), num_valid=-1, num_test=-1,
                        node_feature_dim=-1, rel_feature_dim=-1, num_classes=-1, initialized=False)
            (data / "dataset.yaml").write_text(yaml.safe_dump(meta))
            config = copy.deepcopy(template)
            config["model"]["random_seed"] = 123
            config["model"]["encoder"]["layers"][0][0]["output_dim"] = 10
            config["model"]["decoder"]["options"]["input_dim"] = 10
            config["model"]["decoder"]["type"] = "COMPLEX" if model == "ComplEx" else "DISTMULT"
            config["storage"]["dataset"] = dict(dataset_dir=str(data) + "/")
            config["storage"]["model_dir"] = str(args.work / case / "model") + "/"
            config["storage"]["checkpoint_dir"] = config["storage"]["model_dir"]
            config["storage"]["save_model"] = False
            config["training"].update(num_epochs=2, batch_size=13, save_model=False)
            config["training"]["negative_sampling"].update(num_chunks=3, negatives_per_positive=12)
            path, result = args.results / f"{case}.yaml", args.results / f"{case}.json"
            path.write_text(yaml.safe_dump(config))
            with (args.results / f"{case}.log").open("w") as log:
                command = [str(args.binary), str(path), str(result)]
                if args.observe_padding:
                    command.append("--observe-padding")
                process = subprocess.run(command, env=env,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=120)
            entry = dict(case=case, exit_code=process.returncode, config_sha256=sha(path))
            if result.exists():
                entry.update(json.loads(result.read_text()))
            report["cases"].append(entry)
            report["passed"] = all(c.get("passed", False) for c in report["cases"])
            (args.results / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
    if not report["passed"]:
        raise RuntimeError("Native checks failed; inspect manifest and per-case logs")


if __name__ == "__main__":
    main()
