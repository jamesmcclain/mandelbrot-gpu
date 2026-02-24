#!/usr/bin/env bash

docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v $(pwd):/workdir -w /workdir \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.12 \
  mandelbrot-rocm
