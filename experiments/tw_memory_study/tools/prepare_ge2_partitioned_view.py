#!/usr/bin/env python3
"""Create a GE2-compatible bucket-ordered view of an existing edge split.

The logical source split is never modified. GE2's partition-buffer reader
expects each binary edge file to be physically grouped by (source partition,
destination partition), whereas other baseline conversions may reorder those
same records. This tool writes a separate physical view and records hashes for
both representations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


SPLITS = (
    ("train", "num_train"),
    ("validation", "num_valid"),
    ("test", "num_test"),
)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def edge_array(path: Path, columns: int) -> np.memmap:
    record_bytes = columns * np.dtype("<i4").itemsize
    size = path.stat().st_size
    if size % record_bytes:
        raise RuntimeError(
            f"{path} has {size} bytes, not a multiple of the "
            f"{record_bytes}-byte edge record"
        )
    return np.memmap(path, dtype="<i4", mode="r").reshape(-1, columns)


def bucket_ids(edges: np.ndarray, partition_size: int, partitions: int) -> np.ndarray:
    src = np.minimum(edges[:, 0] // partition_size, partitions - 1)
    dst = np.minimum(edges[:, -1] // partition_size, partitions - 1)
    return (src * partitions + dst).astype(np.int32, copy=False)


def count_buckets(
    edges: np.ndarray,
    partition_size: int,
    partitions: int,
    chunk_edges: int,
) -> np.ndarray:
    counts = np.zeros(partitions * partitions, dtype=np.int64)
    for start in range(0, len(edges), chunk_edges):
        ids = bucket_ids(
            np.asarray(edges[start : start + chunk_edges]),
            partition_size,
            partitions,
        )
        counts += np.bincount(ids, minlength=partitions * partitions)
    return counts


def verify_bucket_order(
    path: Path,
    columns: int,
    partition_size: int,
    partitions: int,
    expected_counts: np.ndarray,
    chunk_edges: int,
) -> None:
    edges = edge_array(path, columns)
    observed = np.zeros(partitions * partitions, dtype=np.int64)
    previous = -1
    for start in range(0, len(edges), chunk_edges):
        ids = bucket_ids(
            np.asarray(edges[start : start + chunk_edges]),
            partition_size,
            partitions,
        )
        if ids.size:
            if int(ids[0]) < previous or np.any(ids[1:] < ids[:-1]):
                raise RuntimeError(f"{path} is not physically ordered by edge bucket")
            previous = int(ids[-1])
            observed += np.bincount(ids, minlength=partitions * partitions)
    if not np.array_equal(observed, expected_counts):
        raise RuntimeError(f"{path} bucket counts changed while writing the view")


def write_partitioned_split(
    source_path: Path,
    output_path: Path,
    offsets_path: Path,
    columns: int,
    partition_size: int,
    partitions: int,
    expected_edges: int,
    chunk_edges: int,
) -> dict[str, object]:
    source_edges = edge_array(source_path, columns)
    if len(source_edges) != expected_edges:
        raise RuntimeError(
            f"{source_path} has {len(source_edges)} edges; expected {expected_edges}"
        )

    counts = count_buckets(source_edges, partition_size, partitions, chunk_edges)
    source_offsets = source_path.with_name(
        source_path.name.replace("_edges.bin", "_partition_offsets.txt")
    )
    if source_offsets.exists():
        recorded = np.atleast_1d(np.loadtxt(source_offsets, dtype=np.int64))
        if int(recorded.sum()) != expected_edges:
            raise RuntimeError(
                f"{source_offsets} sums to {int(recorded.sum())} edges; "
                f"expected {expected_edges} for {source_path}"
            )
        # Source offsets may describe a different partition cardinality. Their
        # bucket counts are directly comparable only when source and target
        # views have the same number of buckets.
        if recorded.shape == counts.shape and not np.array_equal(recorded, counts):
            raise RuntimeError(
                f"{source_offsets} does not describe the logical bucket counts "
                f"in {source_path}"
            )

    output_edges = np.memmap(
        output_path,
        dtype="<i4",
        mode="w+",
        shape=(expected_edges, columns),
    )
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
    )
    cursors = starts.copy()

    # Sorting one chunk at a time keeps memory bounded. Appending each chunk's
    # bucket groups also preserves source order within every bucket.
    for start in range(0, expected_edges, chunk_edges):
        chunk = np.asarray(source_edges[start : start + chunk_edges])
        ids = bucket_ids(chunk, partition_size, partitions)
        order = np.argsort(ids, kind="stable")
        ordered_ids = ids[order]
        boundaries = np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1
        group_starts = np.concatenate((np.zeros(1, dtype=np.int64), boundaries))
        group_ends = np.concatenate(
            (boundaries, np.array([len(order)], dtype=np.int64))
        )
        for group_start, group_end in zip(group_starts, group_ends):
            bucket = int(ordered_ids[group_start])
            size = int(group_end - group_start)
            destination = int(cursors[bucket])
            output_edges[destination : destination + size] = chunk[
                order[group_start:group_end]
            ]
            cursors[bucket] += size

    output_edges.flush()
    del output_edges
    del source_edges

    if not np.array_equal(cursors, starts + counts):
        raise RuntimeError(f"Failed to fill every edge bucket for {source_path}")

    offsets_path.write_text("".join(f"{int(value)}\n" for value in counts))
    verify_bucket_order(
        output_path,
        columns,
        partition_size,
        partitions,
        counts,
        chunk_edges,
    )

    return {
        "num_edges": expected_edges,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "output_path": str(output_path.name),
        "output_sha256": sha256_file(output_path),
        "partition_offsets": str(offsets_path.name),
        "partition_offsets_sha256": sha256_file(offsets_path),
        "physically_bucket_ordered": True,
    }


def existing_view_is_valid(
    source_dir: Path,
    output_dir: Path,
    partitions: int,
    columns: int,
) -> bool:
    manifest_path = output_dir / "partitioned_view_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["num_partitions"] != partitions or manifest["edge_columns"] != columns:
            return False
        for split, _ in SPLITS:
            entry = manifest["splits"][split]
            source_path = source_dir / "edges" / f"{split}_edges.bin"
            output_path = output_dir / "edges" / f"{split}_edges.bin"
            offsets_path = output_dir / "edges" / f"{split}_partition_offsets.txt"
            if not source_path.exists() or not output_path.exists() or not offsets_path.exists():
                return False
            if sha256_file(source_path) != entry["source_sha256"]:
                return False
            if sha256_file(output_path) != entry["output_sha256"]:
                return False
            if sha256_file(offsets_path) != entry["partition_offsets_sha256"]:
                return False
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), start=destination.parent.resolve())
    destination.symlink_to(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--edge-columns", type=int, default=2)
    parser.add_argument("--chunk-edges", type=int, default=2_000_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir:
        raise RuntimeError("The partitioned view must not overwrite the logical source split")
    if args.num_partitions < 1 or args.edge_columns < 2 or args.chunk_edges < 1:
        raise RuntimeError("Invalid partition, column, or chunk count")

    if output_dir.exists() and existing_view_is_valid(
        source_dir, output_dir, args.num_partitions, args.edge_columns
    ):
        print(
            json.dumps(
                {"status": "valid_existing", "output_dir": str(output_dir)},
                indent=2,
            )
        )
        return 0
    if output_dir.exists() and not args.force:
        raise RuntimeError(
            f"Existing GE2 view is incomplete or stale: {output_dir}; rerun with --force"
        )

    dataset_path = source_dir / "dataset.yaml"
    dataset = yaml.safe_load(dataset_path.read_text()) or {}
    num_nodes = int(dataset["num_nodes"])
    num_relations = int(dataset.get("num_relations", 1))
    partition_size = math.ceil(num_nodes / args.num_partitions)

    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "edges").mkdir(parents=True)
    (staging / "nodes").mkdir(parents=True)

    try:
        split_manifest = {}
        for split, count_key in SPLITS:
            source_path = source_dir / "edges" / f"{split}_edges.bin"
            split_manifest[split] = write_partitioned_split(
                source_path=source_path,
                output_path=staging / "edges" / f"{split}_edges.bin",
                offsets_path=staging / "edges" / f"{split}_partition_offsets.txt",
                columns=args.edge_columns,
                partition_size=partition_size,
                partitions=args.num_partitions,
                expected_edges=int(dataset[count_key]),
                chunk_edges=args.chunk_edges,
            )

        node_mapping = source_dir / "nodes" / "node_mapping.txt"
        if node_mapping.exists():
            relative_symlink(node_mapping, staging / "nodes" / node_mapping.name)
        relation_mapping = source_dir / "edges" / "relation_mapping.txt"
        if relation_mapping.exists():
            relative_symlink(relation_mapping, staging / "edges" / relation_mapping.name)

        output_dataset = dict(dataset)
        output_dataset["dataset_dir"] = str(output_dir) + "/"
        output_dataset["initialized"] = True
        (staging / "dataset.yaml").write_text(
            yaml.safe_dump(output_dataset, sort_keys=False)
        )

        manifest = {
            "format": "ge2-physical-bucket-view-v1",
            "source_data_dir": str(source_dir),
            "output_data_dir": str(output_dir),
            "num_nodes": num_nodes,
            "num_relations": num_relations,
            "num_partitions": args.num_partitions,
            "partition_size": partition_size,
            "edge_columns": args.edge_columns,
            "splits": split_manifest,
        }
        (staging / "partitioned_view_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print((output_dir / "partitioned_view_manifest.json").read_text(), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"prepare_ge2_partitioned_view.py: {error}", file=sys.stderr)
        raise SystemExit(1)
