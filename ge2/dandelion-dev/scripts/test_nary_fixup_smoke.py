#!/usr/bin/env python3
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path("/home/smansou2/newCode/ge2/dandelion-dev")
GEGE_PY_ROOT = REPO_ROOT / "gege" / "src" / "python"


def load_local_gege_package():
    spec = importlib.util.spec_from_file_location(
        "gege",
        GEGE_PY_ROOT / "__init__.py",
        submodule_search_locations=[str(GEGE_PY_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gege"] = module
    spec.loader.exec_module(module)


def test_reader_accepts_four_columns():
    load_local_gege_package()
    from gege.tools.preprocess.converters.readers.pandas_readers import PandasDelimitedFileReader

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "train.tsv"
        path.write_text("a\tr\tb\tq\n")
        reader = PandasDelimitedFileReader(train_edges=path, columns=[0, 1, 2, 3], delim="\t")
        train_df, _, _ = reader.read()
        assert list(train_df.iloc[0]) == ["a", "r", "b", "q"]


def test_partitioner_uses_dst_column_for_nary():
    load_local_gege_package()
    from gege.tools.preprocess.converters.partitioners.torch_partitioner import partition_edges

    edges4 = torch.tensor(
        [
            [0, 10, 5, 99],
            [0, 10, 1, 5000],
        ],
        dtype=torch.int64,
    )
    partitioned4, offsets4 = partition_edges(edges4, num_nodes=8, num_partitions=2)
    assert partitioned4[:, 2].tolist() == [1, 5]
    assert offsets4 == [1, 1, 0, 0]

    edges5 = torch.tensor(
        [
            [0, 10, 5, 7, 99],
            [0, 10, 1, 8, 5000],
        ],
        dtype=torch.int64,
    )
    partitioned5, offsets5 = partition_edges(edges5, num_nodes=8, num_partitions=2)
    assert partitioned5[:, 2].tolist() == [1, 5]
    assert offsets5 == [1, 1, 0, 0]


def test_jf17k4_converter_requires_explicit_projection():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir()
        for split in ["train_4.txt", "valid_4.txt", "test_4.txt"]:
            (input_dir / split).write_text("rel\te1\te2\te3\te4\n")

        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "convert_jf17k4.py"), "--input_dir", str(input_dir), "--output_dir", str(output_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "--allow-lossy-projection" in proc.stderr

        cmd.extend(["--allow-lossy-projection", "--project-qualifier-arg", "e4"])
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads((output_dir / "projection_metadata.json").read_text())
        assert metadata["lossy_projection"] is True
        assert metadata["kept_qualifier_arg"] == "e4"
        assert metadata["dropped_arg"] == "e3"
        line = (output_dir / "train_edges.txt").read_text().strip()
        assert line == "e1\trel\te2\t__QUAL__\te4"


def main():
    test_reader_accepts_four_columns()
    test_partitioner_uses_dst_column_for_nary()
    test_jf17k4_converter_requires_explicit_projection()
    print("n-ary smoke checks passed")


if __name__ == "__main__":
    main()
