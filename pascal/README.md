# Pascal host renderer

This directory now contains a full Pascal-hosted Mandelbrot renderer for this repository's existing Pascal `DEVICE UNIT`.

What changed:

- `mandelbrot.pas` / `mandelbrot.inc` remain the unchanged device kernels
- `mandelbrot_host.pas` is the Pascal host source that allocates device memory, launches the kernel, copies the iteration buffer back, and writes a PNG
- `png_helper.c` is a tiny libpng helper used through Pascal C-FFI
- `Makefile` builds the whole thing against a local `pascal-1981` checkout plus its runtime archive

Prerequisites:

- a local `pascal-1981` checkout with its runtime built
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
clang build/host.ll build/dev.ll png_helper.c /path/to/libpascalrt.a -lpng -lm -o mandelbrot
./mandelbrot pip_build.png 1 d 0
```

The only remaining non-`pip` dependency in that manual path is the Pascal runtime archive (`libpascalrt.a`), which must come from a built `pascal-1981` checkout.

This installed-toolchain flow was verified in this repository's venv: `python3 -m pascal1981` successfully rebuilt the example and wrote a valid PNG.

## Run on the CPU-device shim

```bash
cd pascal
make runtime
make run              # DEVICE=cpu by default; writes mandelbrot.png
```

This needs no GPU and no NVIDIA toolchain. The CPU runtime shim emulates the full launch geometry, so the unchanged `DEVICE` kernel runs end-to-end through the Pascal host path.

## CLI

The host program uses the vintage `pascal-1981` program-parameter model, so its command line is **positional**:

```text
mandelbrot <outfile> <view> <prec> <theme>
```

Arguments:

- `outfile` — output PNG filename
- `view` — which built-in view to render:
  - `1` = overview
  - `2` = seahorse valley
  - `3` = elephant valley
  - `4` = double spiral
- `prec` — kernel precision:
  - `s` or `S` = `f32`
  - anything else = `f64`
- `theme` — color theme:
  - `0` = classic
  - `1` = fire
  - `2` = ice
  - `3` = rainbow
  - `4` = emacs
  - `5` = grayscale

Example:

```bash
./mandelbrot demo.png 2 s 1
```

The `make run` target supplies default positional arguments automatically:

- `mandelbrot.png 1 d 0`

You can override them if you want:

```bash
make run RUN_ARGS="demo.png 4 s 3"
```

That writes `demo.png` using:

- view `2` (seahorse valley)
- single-precision kernel (`f32`)
- theme `1` (fire)

If one or more trailing arguments are omitted, the Pascal runtime prompts for the missing values interactively.

## Optional CUDA build

The Makefile supports a CUDA path as well as the default CPU-device shim.

On a CUDA-capable machine, the intended flow is:

```bash
cd pascal
make runtime-cuda
make DEVICE=cuda run
```

Best-effort reconstruction of what that does, based on the `pascal-1981` examples:

- compiles `mandelbrot.pas` to PTX with `--target ptx --sm <arch>`
- packages the PTX text into a `__pas_device_ptx` blob object
- compiles the host with `--device-backend cuda`
- links the host, PTX blob, `png_helper.c`, the CUDA runtime shim archive, and `-lcuda`
- runs the resulting `./mandelbrot` binary

Expected prerequisites for that path are:

- a built CUDA runtime archive from the local `pascal-1981` checkout (`make runtime-cuda`)
- CUDA headers / driver library availability for `-lcuda`
- an NVIDIA GPU and working driver
- `llvmlite` with NVPTX support so `pascal1981 --target ptx` can emit PTX

This CUDA branch is wired in the Makefile, but it was **not executed in this VM** because this environment has no GPU and no full NVIDIA toolchain.

## Current defaults

The host program currently renders at:

- `640 x 360`
- `max_iter = 512`

The view/theme/precision are now selected at launch time through the positional arguments above.
