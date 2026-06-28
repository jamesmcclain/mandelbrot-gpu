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

## Install the compiler into your venv

If you want to build with an installed toolchain instead of a source-checkout `PYTHONPATH`, install `pascal-1981` into the active virtual environment with `pip`.

That can be done either:

- from a local on-disk checkout, or
- directly from the GitHub repository.

Once installed, a manual rebuild looks like this:

```bash
cd pascal
make clean
mkdir -p build
python3 -m pascal1981 --dialect extended -f wide-integers mandelbrot_host.pas build/host.ll
python3 -m pascal1981 --dialect extended -f wide-integers mandelbrot.pas build/dev.ll
clang build/host.ll build/dev.ll png_helper.c /path/to/libpascalrt.a -lpng -lm -o mandelbrot_host
./mandelbrot_host pip_build.png 1 d 0
```

The only remaining non-`pip` dependency in that manual path is the Pascal runtime archive (`libpascalrt.a`), which must come from a built `pascal-1981` checkout.

This installed-toolchain flow was verified in this repository's venv: `python3 -m pascal1981` successfully rebuilt the example and wrote a valid PNG.

## Run on the CPU-device shim

```bash
cd pascal
make runtime          # one-time: builds ../pascal-1981/runtime/build/libpascalrt.a
make run              # DEVICE=cpu by default
```

The Makefile path still uses the sibling checkout directly. The installed-toolchain flow above is mainly useful as a packaging/smoke test.

This needs no GPU and no NVIDIA toolchain. The CPU runtime shim emulates the full launch geometry, so the unchanged `DEVICE` kernel runs end-to-end through the Pascal host path.

The host program now takes simple **positional** command-line parameters using the vintage `pascal-1981` program-parameter model:

```text
mandelbrot_host <outfile> <view> <prec> <theme>
```

Where:

- `outfile` — output PNG filename
- `view` — `1=overview`, `2=seahorse_valley`, `3=elephant_valley`, `4=double_spiral`
- `prec` — `s` for `f32`, anything else for `f64`
- `theme` — `0=classic`, `1=fire`, `2=ice`, `3=rainbow`, `4=emacs`, `5=grayscale`

Example:

```bash
./mandelbrot_host demo.png 2 s 1
```

If arguments are omitted, the Pascal runtime prompts for the missing trailing values.

## Optional CUDA build

If you have CUDA headers, `-lcuda`, and an NVIDIA device, the same host program can be built against the CUDA runtime shim:

```bash
cd pascal
make runtime-cuda
make DEVICE=cuda run
```

## Current defaults

The host program currently renders at:

- `640 x 360`
- `max_iter = 512`

The view/theme/precision are now selected at launch time through the positional arguments above.
