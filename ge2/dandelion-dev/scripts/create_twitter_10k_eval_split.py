#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic Twitter 16p paper-style eval split by holding out "
            "10k validation and 10k test edges from the already partitioned train file."
        )
    )
    parser.add_argument("input_dataset_dir", type=Path)
    parser.add_argument("output_dataset_dir", type=Path)
    parser.add_argument("--valid-count", type=int, default=10_000)
    parser.add_argument("--test-count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_dataset_yaml(output_dir: Path, dataset_stats: dict):
    yaml_path = output_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_stats, handle, sort_keys=False)


def main():
    args = parse_args()

    input_dir = args.input_dataset_dir.resolve()
    output_dir = args.output_dataset_dir.resolve()

    input_yaml = input_dir / "dataset.yaml"
    input_edges_dir = input_dir / "edges"
    input_nodes_dir = input_dir / "nodes"
    input_train = input_edges_dir / "train_edges.bin"
    input_offsets = input_edges_dir / "train_partition_offsets.txt"

    if not input_yaml.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {input_yaml}")
    if not input_train.exists():
        raise FileNotFoundError(f"Missing train edges: {input_train}")
    if not input_offsets.exists():
        raise FileNotFoundError(f"Missing train offsets: {input_offsets}")

    with input_yaml.open("r", encoding="utf-8") as handle:
        stats = yaml.safe_load(handle)

    num_train = int(stats["num_train"])
    num_nodes = int(stats["num_nodes"])
    num_relations = int(stats["num_relations"])
    total_holdout = args.valid_count + args.test_count
    if total_holdout >= num_train:
        raise ValueError("Holdout size must be smaller than num_train")

    num_cols = 2 if num_relations == 1 else 3
    train_edges = np.memmap(input_train, mode="r", dtype=np.int32, shape=(num_train, num_cols))
    offsets = np.loadtxt(input_offsets, dtype=np.int64)
    if offsets.ndim == 0:
        offsets = np.array([int(offsets)], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    sampled_positions = rng.choice(num_train, size=total_holdout, replace=False)
    valid_positions = sampled_positions[: args.valid_count]
    test_positions = sampled_positions[args.valid_count :]
    removed_positions = np.sort(sampled_positions.astype(np.int64))

    valid_edges = np.asarray(train_edges[valid_positions], dtype=np.int32)
    test_edges = np.asarray(train_edges[test_positions], dtype=np.int32)

    cumulative_offsets = np.cumsum(offsets)
    removed_bucket_ids = np.searchsorted(cumulative_offsets, removed_positions, side="right")
    removed_counts = np.bincount(removed_bucket_ids, minlength=len(offsets)).astype(np.int64)
    new_offsets = offsets - removed_counts

    if np.any(new_offsets < 0):
        raise RuntimeError("Computed negative bucket sizes while creating eval split")
    if int(new_offsets.sum()) != num_train - total_holdout:
        raise RuntimeError("New offset sum does not match expected train edge count")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir already exists: {output_dir}")
        shutil.rmtree(output_dir)

    (output_dir / "edges").mkdir(parents=True, exist_ok=True)
    (output_dir / "nodes").mkdir(parents=True, exist_ok=True)

    output_train = output_dir / "edges" / "train_edges.bin"
    with output_train.open("wb") as handle:
        start = 0
        for idx in removed_positions:
            if idx > start:
                train_edges[start:idx].tofile(handle)
            start = idx + 1
        if start < num_train:
            train_edges[start:num_train].tofile(handle)

    valid_edges.tofile(output_dir / "edges" / "validation_edges.bin")
    test_edges.tofile(output_dir / "edges" / "test_edges.bin")

    with (output_dir / "edges" / "train_partition_offsets.txt").open("w", encoding="utf-8") as handle:
        for value in new_offsets:
            handle.write(f"{int(value)}\n")

    shutil.copy2(input_nodes_dir / "node_mapping.txt", output_dir / "nodes" / "node_mapping.txt")

    dataset_stats = {
        "dataset_dir": str(output_dir) + "/",
        "num_edges": int(num_train - total_holdout),
        "num_nodes": num_nodes,
        "num_relations": num_relations,
        "num_train": int(num_train - total_holdout),
        "num_valid": int(args.valid_count),
        "num_test": int(args.test_count),
        "node_feature_dim": -1,
        "rel_feature_dim": -1,
        "num_classes": -1,
        "initialized": False,
    }
    write_dataset_yaml(output_dir, dataset_stats)

    print(f"Created eval split dataset: {output_dir}")
    print(f"train_edges={dataset_stats['num_train']} valid_edges={dataset_stats['num_valid']} test_edges={dataset_stats['num_test']}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
