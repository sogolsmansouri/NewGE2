#!/usr/bin/env python3
"""Build standalone diagnostics against an existing original GE2 installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--programs', nargs='+', default=['audit_zenodo_native_training', 'train_zenodo_repartition_control'],
                        choices=['audit_zenodo_native_training', 'train_zenodo_repartition_control'])
    args = parser.parse_args()
    import torch
    prefix = Path(sys.prefix)
    site = Path(torch.__file__).parent.parent
    cpp = args.source_root / 'src/cpp'
    args.out.mkdir(parents=True, exist_ok=False)
    includes = [cpp / 'include', cpp / 'third_party/pybind11/include',
                cpp / 'third_party/spdlog/include', cpp / 'third_party/parallel-hashmap',
                prefix / 'include', prefix / ('include/python' + '.'.join(map(str, sys.version_info[:2]))),
                site / 'torch/include', site / 'torch/include/torch/csrc/api/include',
                prefix / 'targets/x86_64-linux/include']
    libraries = [site / 'gege', site / 'torch/lib', prefix / 'lib']
    common = ['g++', '-std=c++17', '-O0', '-g0',
              '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
              '-DGEGE_CUDA', '-DGEGE_OMP', '-fopenmp'] + ['-I' + str(p) for p in includes]
    link = [v for p in libraries for v in ('-L' + str(p), '-Wl,-rpath,' + str(p))]
    link += ['-lge2', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda',
             '-lpython' + '.'.join(map(str, sys.version_info[:2])), '-lcudart']
    report = dict(torch_version=torch.__version__, env=str(prefix), builds=[],
                  library_sha256=hashlib.sha256((site / 'gege/libge2.so').read_bytes()).hexdigest())
    for name in args.programs:
        source = Path(__file__).with_name(name + '.cpp')
        command = common + [str(source), '-o', str(args.out / name)] + link
        with (args.out / (name + '.build.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report['builds'].append(dict(command=command, exit_code=result.returncode,
                                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        (args.out / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError('Compilation failed: ' + str(args.out / (name + '.build.log')))
        print(name + ': built', flush=True)


if __name__ == '__main__':
    main()
