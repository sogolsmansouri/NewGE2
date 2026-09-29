#!/usr/bin/env python3
"""Compile a disclosed trainer-only correction against the released GE2 library."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from arc_job_support import write_json
from run_arc_paper_case import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.md5(args.archive.read_bytes()).hexdigest() != '6de3d9702241a0c822971939752d0834':
        raise ValueError('Not the frozen Zenodo archive')
    args.out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as z:
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archive path')
        z.extractall(args.out/'release')
    root = args.out/'release/dandelion-dev/gege'
    trainer = root/'src/cpp/src/engine/trainer.cpp'
    if digest(trainer) != '79a64f533193b74034708ad0d2c650cad474fe3d5330dadd855d1993afc33bc1':
        raise ValueError('Original trainer differs')
    scripts = Path(__file__).resolve().parent
    patch = scripts/'ge2_dense_barrier.patch'
    subprocess.run(['git', 'apply', '--recount', str(patch)], cwd=root, check=True)
    shutil.copy2(scripts/'ge2_dense_barrier.h', root/'src/cpp/include/ge2_dense_barrier.h')
    import torch
    prefix = Path(sys.prefix)
    site = Path(torch.__file__).parent.parent
    cpp = root/'src/cpp'
    includes = [cpp/'include', cpp/'third_party/pybind11/include', cpp/'third_party/spdlog/include',
                cpp/'third_party/parallel-hashmap', prefix/'include',
                prefix/'include/python3.9', site/'torch/include', site/'torch/include/torch/csrc/api/include',
                prefix/'targets/x86_64-linux/include']
    libraries = [site/'gege', site/'torch/lib', prefix/'lib']
    compiler = prefix/'bin/x86_64-conda-linux-gnu-c++'
    if not compiler.exists():
        compiler = Path(shutil.which('g++'))
    binary = args.out/'libge2_dense_repair.so'
    command = [str(compiler), '-std=c++17', '-O2', '-g1', '-fPIC', '-shared', '-fopenmp',
               '-DGEGE_CUDA', '-DGEGE_OMP',
               '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    command += ['-I'+str(p) for p in includes]
    command += [str(trainer), '-o', str(binary)]
    command += [v for p in libraries for v in ('-L'+str(p), '-Wl,-rpath,'+str(p))]
    command += ['-lge2', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lc10', '-lc10_cuda', '-lpython3.9', '-lcudart']
    report = dict(original_archive_sha256=digest(args.archive), original_library_sha256=digest(site/'gege/libge2.so'),
                  patch_sha256=digest(patch), patched_trainer_sha256=digest(trainer),
                  header_sha256=digest(scripts/'ge2_dense_barrier.h'), command=command,
                  scope='Trainer barrier only; all math/data/storage operators use the released GE2 library',
                  classification='patched released GE2, not unmodified release')
    write_json(args.out/'build.json', report)
    with (args.out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    test = args.out/'test_barrier'
    subprocess.run([str(compiler), '-std=c++17', '-O2', '-pthread', str(scripts/'test_ge2_dense_barrier.cpp'),
                    '-o', str(test)], check=True)
    subprocess.run([test], check=True, timeout=60)
    report.update(binary=str(binary), binary_sha256=digest(binary), status='compiled_and_cpu_tested')
    write_json(args.out/'build.json', report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
