#!/usr/bin/env python3
"""
Preprocess JF17K-4 (already converted to GE2 arity-4 format) using TorchEdgeListConverter.

Input files (from convert_jf17k4.py):
    /home/smansou2/data/jf17k4_ge2/{train,valid,test}_edges.txt
    Format: src\trel\tdst\t__QUAL__\tqval  (5 tab-separated columns)

Output:
    /home/smansou2/data/jf17k4_ge2_processed/
    - train_edges.bin, valid_edges.bin, test_edges.bin  (int32, shape N×5)
    - node_mapping.bin, rel_mapping.bin
    - dataset.yaml

Usage:
    cd /home/smansou2/newCode/ge2/dandelion-dev
    python scripts/preprocess_jf17k4.py
"""

import sys
import os

# Skip C++ bindings (not needed for preprocessing)
os.environ["GEGE_NO_BINDINGS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gege_pkg"))

from pathlib import Path
from gege.tools.preprocess.converters.torch_converter import TorchEdgeListConverter

INPUT_DIR = Path("/home/smansou2/data/jf17k4_ge2")
OUTPUT_DIR = Path("/home/smansou2/data/jf17k4_ge2_processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting JF17K-4 preprocessing...")

converter = TorchEdgeListConverter(
    output_dir=OUTPUT_DIR,
    train_edges=INPUT_DIR / "train_edges.txt",
    valid_edges=INPUT_DIR / "valid_edges.txt",
    test_edges=INPUT_DIR / "test_edges.txt",
    num_partitions=1,
    remap_ids=True,
    partitioned_evaluation=False,
    columns=[0, 1, 2, 3, 4],  # src, rel, dst, qrel, qval
    delim="\t",
)

dataset_config = converter.convert()

print(f"\nDataset config: {dataset_config}")
print(f"\nOutput written to: {OUTPUT_DIR}")
print("Files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size if f.is_file() else 0
    print(f"  {f.name}  ({size:,} bytes)")
