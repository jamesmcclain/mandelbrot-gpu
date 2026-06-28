# Pascal host renderer

This directory now contains a full Pascal-hosted Mandelbrot renderer for this repository's existing Pascal `DEVICE UNIT`.

What changed:

- `mandelbrot.pas` / `mandelbrot.inc` remain the unchanged device kernels
- `mandelbrot_host.pas` is a Pascal host program that allocates device memory, launches the kernel, copies the iteration buffer back, and writes a PNG
- `png_helper.c` is a tiny libpng helper used through Pascal C-FFI
- `Makefile` builds the whole thing against `../pascal-1981`

Prerequisites:

- a sibling checkout at `../pascal-1981`
- `clang`
- `libpng` development headers/library
- Python with whatever `pascal-1981` itself needs (`llvmlite`, etc.)

## Run on the CPU-device shim

```bash
cd pascal
make runtime          # one-time: builds ../pascal-1981/runtime/build/libpascalrt.a
make run              # DEVICE=cpu by default
```

This needs no GPU and no NVIDIA toolchain. The CPU runtime shim emulates the full launch geometry, so the unchanged `DEVICE` kernel runs end-to-end through the Pascal host path.

Successful output writes:

- `pascal/mandelbrot_pascal_f64.png`

## Optional CUDA build

If you have CUDA headers, `-lcuda`, and an NVIDIA device, the same host program can be built against the CUDA runtime shim:

```bash
cd pascal
make runtime-cuda
make DEVICE=cuda run
```

## Current defaults

The host program currently renders the Python project's built-in **overview** view at:

- `640 x 360`
- `max_iter = 512`
- `theme = classic`
- `precision = f64`

To switch to the single-precision kernel, change `use_f32` in `mandelbrot_host.pas`.
To switch theme, change the `theme` constant.
