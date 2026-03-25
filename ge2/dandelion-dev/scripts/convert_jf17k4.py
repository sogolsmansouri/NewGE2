#!/usr/bin/env python3
"""
Convert raw JF17K-4 facts into GE2's 5-column TUCKER4 input format.

Raw JF17K-4 facts are tab-separated:
    rel  e1  e2  e3  e4

GE2's current TUCKER4 path expects exactly one qualifier pair:
    src  rel  dst  qrel  qval

Those formats are not semantically equivalent. A raw 4-entity fact must be
projected into a base triple plus one qualifier pair, which necessarily drops
one original argument. This script therefore refuses conversion by default and
requires an explicit lossy-projection opt-in.
"""

import argparse
import json
import os

DUMMY_QREL = "__QUAL__"
METADATA_FILENAME = "projection_metadata.json"


def convert_file(input_path: str, output_path: str, qualifier_arg: str) -> int:
    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_no, line in enumerate(fin, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                raise RuntimeError(
                    f"{input_path}:{line_no}: expected 5 tab-separated columns "
                    f"'rel e1 e2 e3 e4', received {len(parts)}"
                )
            rel, e1, e2, e3, e4 = parts
            qval = e3 if qualifier_arg == "e3" else e4
            fout.write(f"{e1}\t{rel}\t{e2}\t{DUMMY_QREL}\t{qval}\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert JF17K-4 to GE2 arity-4 format")
    parser.add_argument("--input_dir", default="/home/smansou2/HySAE/data/JF17K",
                        help="Directory containing train_4.txt, valid_4.txt, test_4.txt")
    parser.add_argument("--output_dir", default="/home/smansou2/data/jf17k4_ge2",
                        help="Output directory for converted files")
    parser.add_argument(
        "--allow-lossy-projection",
        action="store_true",
        help="Explicitly allow projecting raw JF17K-4 facts into GE2's lossy 5-column TUCKER4 format.",
    )
    parser.add_argument(
        "--project-qualifier-arg",
        choices=["e3", "e4"],
        default="e3",
        help="Which raw argument to keep as qval in the lossy GE2 projection. The other extra argument is dropped.",
    )
    args = parser.parse_args()

    if not args.allow_lossy_projection:
        raise RuntimeError(
            "Raw JF17K-4 facts cannot be represented faithfully in GE2's current 5-column TUCKER4 format. "
            "Re-run with --allow-lossy-projection and choose --project-qualifier-arg explicitly."
        )

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
        n = convert_file(in_path, out_path, args.project_qualifier_arg)
        print(f"  {in_name} -> {out_name}: {n} edges")

    dropped_arg = "e4" if args.project_qualifier_arg == "e3" else "e3"
    metadata = {
        "source_format": "rel e1 e2 e3 e4",
        "target_format": "src rel dst qrel qval",
        "lossy_projection": True,
        "dummy_qrel": DUMMY_QREL,
        "projection_mode": "base_triple_plus_one_dummy_qualifier",
        "base_triple": ["e1", "rel", "e2"],
        "kept_qualifier_arg": args.project_qualifier_arg,
        "dropped_arg": dropped_arg,
    }
    metadata_path = os.path.join(args.output_dir, METADATA_FILENAME)
    with open(metadata_path, "w") as fout:
        json.dump(metadata, fout, indent=2, sort_keys=True)
        fout.write("\n")

    print(f"\nOutput written to: {args.output_dir}")
    print(f"Format: src\\trel\\tdst\\t{DUMMY_QREL}\\tqval")
    print(f"Lossy projection: kept {args.project_qualifier_arg} as qval, dropped {dropped_arg}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
