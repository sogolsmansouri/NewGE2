#!/usr/bin/env python3
"""Create a Freebase86M p32 processed dataset from the existing p16 binary.

The raw text files and validation/test binaries do not depend on the training
partition count for the current FB repro path, so this script hardlinks them
into the new dataset directory. It rewrites only the training edge binary and
the train partition-offset file.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np


def parse_dataset_yaml(path: Path) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip()
        if re.fullmatch(r"-?\d+", value):
            values[key.strip()] = int(value)
        else:
            values[key.strip()] = value
    return values


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def detect_num_columns(edge_path: Path, num_edges: int) -> int:
    bytes_per_edge_value = np.dtype(np.int32).itemsize
    size = edge_path.stat().st_size
    if size % (num_edges * bytes_per_edge_value) != 0:
        raise ValueError(f"{edge_path} size is not divisible by num_edges * int32")
    columns = size // (num_edges * bytes_per_edge_value)
    if columns not in (3, 5):
        raise ValueError(f"unsupported edge column count {columns} for {edge_path}")
    return int(columns)


def write_dataset_yaml(src_yaml: Path, dst_yaml: Path, output_dir: Path) -> None:
    text = src_yaml.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^dataset_dir:.*$", f"dataset_dir: {output_dir.resolve()}/", text)
    dst_yaml.write_text(text, encoding="utf-8")


def link_static_dataset_files(input_dir: Path, output_dir: Path) -> None:
    for rel in (
        "entity2id.txt",
        "relation2id.txt",
        "train.txt",
        "valid.txt",
        "test.txt",
        "nodes/node_mapping.txt",
        "edges/relation_mapping.txt",
        "edges/validation_edges.bin",
        "edges/test_edges.bin",
    ):
        src = input_dir / rel
        if src.exists():
            hardlink_or_copy(src, output_dir / rel)


def append_bucket(path: Path, rows: np.ndarray) -> None:
    with path.open("ab") as handle:
        handle.write(np.ascontiguousarray(rows).tobytes())


def repartition_train_edges(
    input_dir: Path,
    output_dir: Path,
    num_partitions: int,
    chunk_edges: int,
    force: bool,
) -> None:
    meta = parse_dataset_yaml(input_dir / "dataset.yaml")
    num_nodes = int(meta["num_nodes"])
    num_train = int(meta["num_train"])
    input_train = input_dir / "edges" / "train_edges.bin"
    columns = detect_num_columns(input_train, num_train)
    dst_edges = output_dir / "edges"
    dst_train = dst_edges / "train_edges.bin"
    dst_offsets = dst_edges / "train_partition_offsets.txt"
    tmp_dir = output_dir / f".tmp_repartition_{num_partitions}p"

    if (dst_train.exists() or dst_offsets.exists()) and not force:
        raise FileExistsError(f"{dst_train} or {dst_offsets} already exists; pass --force to replace")
    if force:
        dst_train.unlink(missing_ok=True)
        dst_offsets.unlink(missing_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    dst_edges.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    partition_size = math.ceil(num_nodes / num_partitions)
    bucket_count = num_partitions * num_partitions
    offsets = np.zeros(bucket_count, dtype=np.int64)
    edges = np.memmap(input_train, dtype=np.int32, mode="r", shape=(num_train, columns))
    dst_col = 2 if columns == 5 else columns - 1

    for start in range(0, num_train, chunk_edges):
        stop = min(start + chunk_edges, num_train)
        chunk = np.asarray(edges[start:stop])
        src_part = chunk[:, 0].astype(np.int64) // partition_size
        dst_part = chunk[:, dst_col].astype(np.int64) // partition_size
        np.minimum(src_part, num_partitions - 1, out=src_part)
        np.minimum(dst_part, num_partitions - 1, out=dst_part)
        bucket_ids = src_part * num_partitions + dst_part
        order = np.argsort(bucket_ids, kind="stable")
        sorted_bucket_ids = bucket_ids[order]
        sorted_chunk = chunk[order]

        unique_ids, first_indices, counts = np.unique(
            sorted_bucket_ids, return_index=True, return_counts=True
        )
        for bucket_id, first, count in zip(unique_ids, first_indices, counts):
            bucket = int(bucket_id)
            append_bucket(tmp_dir / f"bucket_{bucket:04d}.bin", sorted_chunk[first : first + count])
            offsets[bucket] += int(count)

        done = stop / num_train * 100.0
        print(f"repartitioned {stop:,}/{num_train:,} edges ({done:.1f}%)", flush=True)

    with dst_train.open("wb") as out:
        for bucket in range(bucket_count):
            bucket_path = tmp_dir / f"bucket_{bucket:04d}.bin"
            if bucket_path.exists():
                with bucket_path.open("rb") as inp:
                    shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)

    with dst_offsets.open("w", encoding="utf-8") as handle:
        for count in offsets:
            handle.write(f"{int(count)}\n")

    if int(offsets.sum()) != num_train:
        raise RuntimeError(f"offset sum {int(offsets.sum())} did not match num_train {num_train}")

    shutil.rmtree(tmp_dir)
    print(f"wrote {dst_train}", flush=True)
    print(f"wrote {dst_offsets}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/smansou2/newCode/ge2/dandelion-dev/datasets/freebase86m_16p"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/smansou2/newCode/ge2/dandelion-dev/datasets/freebase86m_32p"),
    )
    parser.add_argument("--num-partitions", type=int, default=32)
    parser.add_argument("--chunk-edges", type=int, default=4_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "edges").mkdir(exist_ok=True)
    (args.output_dir / "nodes").mkdir(exist_ok=True)

    write_dataset_yaml(args.input_dir / "dataset.yaml", args.output_dir / "dataset.yaml", args.output_dir)
    link_static_dataset_files(args.input_dir, args.output_dir)
    repartition_train_edges(
        args.input_dir,
        args.output_dir,
        args.num_partitions,
        args.chunk_edges,
        args.force,
    )


if __name__ == "__main__":
    main()
