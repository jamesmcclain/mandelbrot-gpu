#!/usr/bin/env bash

$HOME/local/llvm-22.1.0-rc3/bin/clang \
  -target amdgcn-amd-amdhsa -mcpu=gfx90c \
  -x cl -O3 -cl-std=CL2.0 \
  -Xclang -finclude-default-header \
  -nogpulib \
  -Xclang -mlink-builtin-bitcode \
  -Xclang $HOME/local/llvm-22.1.0-rc3/share/clc/amdgcn-amd-amdhsa.bc \
  -c mandelbrot.cl -o mandelbrot.hsaco
