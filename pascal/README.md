# Pascal host renderer

This directory now contains a full Pascal-hosted Mandelbrot renderer for this repository's existing Pascal `DEVICE UNIT`.

What changed:

- `mandelbrot.pas` / `mandelbrot.inc` remain the unchanged device kernels
- `mandelbrot_host.pas` is the Pascal host source that allocates device memory, launches the kernel, copies the iteration buffer back, and writes a PNG
- `png_helper.c` is a tiny libpng helper used through Pascal C-FFI
- `Makefile` builds the whole thing with the pip-installed `pascal1981` compiler and runtime archives from the active Python environment

Prerequisites:

- `pascal-1981` installed into the active Python environment with `pip`
- the installed package's runtime archives (`libpascalrt.a` for CPU, and `libpascalrt_cuda.a` for CUDA builds)
- `clang`
- `libpng` development headers/library
- Python with whatever `pascal-1981` itself needs (`llvmlite`, etc.)

## Installed compiler/runtime flow

Install `pascal-1981` into the active virtual environment with `pip`. That can be done either:

- from a local on-disk checkout, or
- directly from the GitHub repository.

The Makefile discovers the runtime archive from the installed Python package and invokes the compiler as `python3 -m pascal1981`; it no longer assumes a sibling `../pascal-1981` source checkout or builds the runtime out-of-tree.

A manual rebuild equivalent to the Makefile looks like this:

```bash
cd pascal
make clean
mkdir -p build
python3 -m pascal1981 --dialect extended -f wide-integers mandelbrot_host.pas build/host.ll
python3 -m pascal1981 --dialect extended -f wide-integers mandelbrot.pas build/dev.ll
clang build/host.ll build/dev.ll png_helper.c \
  "$(python3 - <<'PY'
import pathlib, pascal1981
print(pathlib.Path(pascal1981.__file__).parent / 'libpascalrt.a')
PY
)" -lpng -lm -o mandelbrot
./mandelbrot pip_build.png 1 d 0
```

This installed-toolchain flow was verified in this repository's venv: `python3 -m pascal1981` successfully rebuilt the example and wrote a valid PNG.

## Run on the CPU-device shim

```bash
cd pascal
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

- view `4` (double spiral)
- single-precision kernel (`f32`)
- theme `3` (rainbow)

If one or more trailing arguments are omitted, the Pascal runtime prompts for the missing values interactively.

## Optional CUDA build

The Makefile supports a CUDA path as well as the default CPU-device shim.

On a CUDA-capable machine, the intended flow is:

```bash
cd pascal
make DEVICE=cuda run
```

The Makefile now uses the compiler driver's built-in PTX embedding path:

- compiles `mandelbrot.pas` to PTX with `--target ptx --sm <arch>`
- compiles the host with `--device-backend cuda --embed-device-ptx build/dev.ptx`
- links the host, `png_helper.c`, the CUDA runtime shim archive, and `-lcuda`
- runs the resulting `./mandelbrot` binary

That avoids the old hand-written `.incbin` assembly stub and the extra PTX-blob object file.

Expected prerequisites for that path are:

- `libpascalrt_cuda.a` included in the pip-installed `pascal1981` package
- CUDA headers / driver library availability for `-lcuda`
- an NVIDIA GPU and working driver
- `llvmlite` with NVPTX support so `pascal1981 --target ptx` can emit PTX

This CUDA branch is wired in the Makefile, but it was **not executed in this VM** because this environment has no GPU and no full NVIDIA toolchain.

## Current defaults

The host program currently renders at:

- `640 x 360`
- `max_iter = 512`

The view/theme/precision are now selected at launch time through the positional arguments above.
