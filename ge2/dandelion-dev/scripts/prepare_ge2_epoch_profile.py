#!/usr/bin/env python3
"""Apply the archived annotation patch and add an epoch-2 capture boundary."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one source anchor: '+old[:80])
    return text.replace(old,new,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--patch',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if hashlib.md5(a.archive.read_bytes()).hexdigest() != '6de3d9702241a0c822971939752d0834':
        raise ValueError('Not the pinned original GE2 archive')
    a.out.mkdir(parents=True,exist_ok=False)
    prefix='dandelion-dev/gege/'
    with zipfile.ZipFile(a.archive) as z:
        for name in z.namelist():
            if not name.startswith(prefix) or name.endswith('/'):
                continue
            rel=Path(name[len(prefix):])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive path')
            target=a.out/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(name))
    subprocess.run(['patch','--batch','--fuzz=0','-p3','-i',str(a.patch.resolve())],cwd=a.out,check=True)
    # Zenodo bundles the dependencies but omits their Git metadata.
    third=a.out/'src/cpp/third_party'
    cmake=third/'CMakeLists.txt'
    cmake_text=cmake.read_text()
    for dep,header in [('pybind11','include/pybind11/pybind11.h'),
                       ('spdlog','include/spdlog/spdlog.h'),
                       ('parallel-hashmap','parallel_hashmap/phmap.h')]:
        if not (third/dep/header).is_file():
            raise ValueError('Missing bundled dependency '+dep)
        cmake_text=replace_once(cmake_text,'initialize_submodule('+dep+')',
                               '# Bundled archive dependency: '+dep)
    cmake.write_text(cmake_text)
    path=a.out/'src/cpp/src/engine/trainer.cpp'
    text=path.read_text()
    text=replace_once(text,'#include "common/ge2_analysis.h"',
        '#include "common/ge2_analysis.h"\n#include <cuda_runtime_api.h>\n#include <stdexcept>')
    text=replace_once(text,'        ge2_analysis::reset_epoch(epoch + 1);', '''        const int64_t profile_epoch = dataloader_->getEpochsProcessed() + 1;
        const char *profile_setting = std::getenv("GE2_PROFILE_EPOCH");
        const bool capture = profile_setting && profile_epoch == std::strtoll(profile_setting, nullptr, 10);
        if (capture) {
            nvtxRangePushA("ge2.profile.epoch_cycle");
            nvtxRangePushA("ge2.profile.native_epoch");
        }
        ge2_analysis::reset_epoch(profile_epoch);''')
    text=replace_once(text,'        ge2_analysis::stop_epoch();', '''        ge2_analysis::stop_epoch();
        if (capture) {
            nvtxRangePop();
            nvtxRangePushA("ge2.profile.epoch_finalize");
        }''')
    text=replace_once(text,'        ge2_analysis::report_epoch(epoch_time);', '''        ge2_analysis::report_epoch(epoch_time);
        if (capture) {
            // This diagnostic-only drain is outside the native epoch timer.
            if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("profile drain failed");
            nvtxRangePop();
            nvtxRangePop();
        }''')
    path.write_text(text)
    print(a.out)


if __name__=='__main__':
    main()
