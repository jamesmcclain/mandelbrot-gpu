#!/usr/bin/env bash

$HOME/local/llvm-22.1.0-rc3/bin/clang \
  -target nvptx64-nvidia-cuda -march=sm_86 \
  -x cl -O3 -cl-std=CL2.0 \
  -Xclang -finclude-default-header \
  -Xclang -target-feature -Xclang +ptx71 \
  -Xclang -mlink-builtin-bitcode \
  -Xclang $HOME/local/llvm-22.1.0-rc3/share/clc/nvptx64-nvidia-cuda.bc \
  -S mandelbrot.cl -o mandelbrot.ptx
