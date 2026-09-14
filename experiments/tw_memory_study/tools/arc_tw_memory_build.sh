#!/usr/bin/env bash
set -euo pipefail

WORK=${1:?node-local experiment directory}
ENV=${2:?validated node-local CUDA environment}
COMMIT=${3:?expected source commit}
cd "$WORK"
test "$(git -C "$WORK/repo" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$WORK/repo" status --porcelain --untracked-files=no)"
GEGE="$WORK/repo/ge2/dandelion-dev/gege"
BUILD="$WORK/build_git"
export PATH="$ENV/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ENV/lib:$ENV/lib/python3.9/site-packages/torch/lib:/usr/lib64"
export CUDA_HOME="$ENV"
export CUDAHOSTCXX="$ENV/bin/x86_64-conda-linux-gnu-c++"
export TORCH_CUDA_ARCH_LIST=8.6
unset LD_PRELOAD PYTHONPATH
"$ENV/bin/python" -c 'import torch, numpy, yaml; print(torch.__version__, torch.version.cuda, numpy.__version__)'
cmake -S "$GEGE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_OMP=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_C_COMPILER="$ENV/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_COMPILER="$ENV/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCUDA_TOOLKIT_ROOT_DIR="$ENV" -DCUDAToolkit_ROOT="$ENV" \
  -DPYTHON_EXECUTABLE="$ENV/bin/python" -DPython3_EXECUTABLE="$ENV/bin/python" \
  -DCUDA_CUDA_LIBRARY=/usr/lib64/libcuda.so -DCUDA_CUDA_LIB=/usr/lib64/libcuda.so \
  -DLIBNVTOOLSEXT="$ENV/lib/libnvToolsExt.so" \
  -DCMAKE_LIBRARY_PATH="$ENV/lib;$ENV/targets/x86_64-linux/lib;/usr/lib64" \
  -DCMAKE_BUILD_RPATH="$BUILD;$ENV/lib;$ENV/lib/python3.9/site-packages/torch/lib"
cmake --build "$BUILD" --target gege_train gege_fixed_frame_buffer_test -j 8
ldd "$BUILD/gege_train"
if ldd "$BUILD/gege_train" | grep -q 'not found'; then
  exit 1
fi
printf '%s\n' "$COMMIT" > "$WORK/build_git_completed_commit.txt"
