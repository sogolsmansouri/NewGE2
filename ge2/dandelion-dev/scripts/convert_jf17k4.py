#!/usr/bin/env python3
"""
Convert JF17K-4 dataset to GE2 arity-4 format.

JF17K-4 input format (tab-separated):
    rel  e1  e2  e3  e4

GE2 arity-4 output format (tab-separated, 5 columns):
    src  rel  dst  qrel  qval

Mapping:
    src  = e1   (col 1)
    rel  = rel  (col 0)
    dst  = e2   (col 2)
    qrel = __QUAL__  (dummy qualifier relation; e4 is dropped)
    qval = e3   (col 3)

Usage:
    python convert_jf17k4.py \
        --input_dir /home/smansou2/HySAE/data/JF17K \
        --output_dir /home/smansou2/data/jf17k4_ge2
"""

import argparse
import os

DUMMY_QREL = "__QUAL__"


def convert_file(input_path: str, output_path: str) -> int:
    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue  # skip malformed lines
            rel, e1, e2, e3 = parts[0], parts[1], parts[2], parts[3]
            # e4 = parts[4] if len(parts) > 4 else None  # dropped
            fout.write(f"{e1}\t{rel}\t{e2}\t{DUMMY_QREL}\t{e3}\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert JF17K-4 to GE2 arity-4 format")
    parser.add_argument("--input_dir", default="/home/smansou2/HySAE/data/JF17K",
                        help="Directory containing train_4.txt, valid_4.txt, test_4.txt")
    parser.add_argument("--output_dir", default="/home/smansou2/data/jf17k4_ge2",
                        help="Output directory for converted files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = [
        ("train_4.txt", "train_edges.txt"),
        ("valid_4.txt", "valid_edges.txt"),
        ("test_4.txt",  "test_edges.txt"),
    ]

    for in_name, out_name in splits:
        in_path = os.path.join(args.input_dir, in_name)
        out_path = os.path.join(args.output_dir, out_name)
        if not os.path.exists(in_path):
            print(f"  SKIP (not found): {in_path}")
            continue
        n = convert_file(in_path, out_path)
        print(f"  {in_name} -> {out_name}: {n} edges")

    print(f"\nOutput written to: {args.output_dir}")
    print(f"Format: src\\trel\\tdst\\t{DUMMY_QREL}\\tqval")


if __name__ == "__main__":
    main()
