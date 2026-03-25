#!/usr/bin/env python3
"""
Preprocess a JF17K-4 dataset that has already been projected into GE2's 5-column
TUCKER4 format.

Input files (from convert_jf17k4.py):
    <input_dir>/{train,valid,test}_edges.txt
    <input_dir>/projection_metadata.json
    Format: src\trel\tdst\t__QUAL__\tqval  (5 tab-separated columns)

This script refuses to preprocess the projected dataset unless the caller
explicitly acknowledges the lossy conversion.
"""

import argparse
import json
import sys
import os

# Skip C++ bindings (not needed for preprocessing)
os.environ["GEGE_NO_BINDINGS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gege_pkg"))

from pathlib import Path
from gege.tools.preprocess.converters.torch_converter import TorchEdgeListConverter

METADATA_FILENAME = "projection_metadata.json"


def load_projection_metadata(input_dir: Path):
    metadata_path = input_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise RuntimeError(
            f"Missing {METADATA_FILENAME} in {input_dir}. "
            "Run convert_jf17k4.py first so the lossy projection is recorded explicitly."
        )
    with open(metadata_path, "r") as fin:
        return metadata_path, json.load(fin)


def main():
    parser = argparse.ArgumentParser(description="Preprocess projected JF17K-4 for GE2 TUCKER4")
    parser.add_argument("--input_dir", default="/home/smansou2/data/jf17k4_ge2",
                        help="Directory containing projected train/valid/test edge lists")
    parser.add_argument("--output_dir", default="/home/smansou2/data/jf17k4_ge2_processed",
                        help="Directory for the processed GE2 dataset")
    parser.add_argument(
        "--allow-lossy-projection",
        action="store_true",
        help="Acknowledge that the input was produced by a lossy JF17K-4 -> GE2 projection.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path, metadata = load_projection_metadata(input_dir)
    if metadata.get("lossy_projection", False) and not args.allow_lossy_projection:
        raise RuntimeError(
            f"{metadata_path} records a lossy JF17K-4 projection. "
            "Re-run with --allow-lossy-projection to preprocess this projected dataset intentionally."
        )

    print("Starting JF17K-4 preprocessing...")
    print(f"Projection metadata: {metadata_path}")

    converter = TorchEdgeListConverter(
        output_dir=output_dir,
        train_edges=input_dir / "train_edges.txt",
        valid_edges=input_dir / "valid_edges.txt",
        test_edges=input_dir / "test_edges.txt",
        num_partitions=1,
        remap_ids=True,
        partitioned_evaluation=False,
        columns=[0, 1, 2, 3, 4],  # src, rel, dst, qrel, qval
        delim="\t",
    )

    dataset_config = converter.convert()

    print(f"\nDataset config: {dataset_config}")
    print(f"\nOutput written to: {output_dir}")
    print("Files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
