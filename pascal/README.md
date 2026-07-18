# Pascal host renderer

This directory contains a full Pascal-hosted Mandelbrot renderer for this repository's existing Pascal `DEVICE UNIT` — with **no C shim**. Everything on the host side, including colorization and the libpng call that writes the PNG, is Pascal.

Contents:

- `mandelbrot.pas` / `mandelbrot.inc` are the unchanged device kernels
- `mandelbrot_host.pas` is the Pascal host source: it allocates the iteration buffer as a heap super array, launches the kernel, copies the iteration counts back, colorizes into a `WORD8` pixel buffer, and writes the PNG by calling libpng's simplified write API directly through the `pascal-1981` C foreign-function interface
- `Makefile` builds the whole thing with the pip-installed `pascal1981` gcc-style compiler driver and the runtime archives bundled with the active Python environment.  Three equivalent build variants are provided (see [Makefile variants](#makefile-variants) below); `Makefile` (Option A) is the default

The former `png_helper.c` shim is gone. The three jobs it did are now Pascal:

- **filename conversion** — the `LSTRING` program parameter is NUL-terminated into a `CHAR` array in `write_png`
- **colorization** — the classic/fire/ice/rainbow/emacs/grayscale palettes and the log-normalization of escape counts are ported into Pascal procedures, writing `WORD8` (uint8) channel values into a heap super-array pixel buffer
- **the libpng call** — `png_image` is declared as a Pascal `RECORD` (a field-for-field transcription of libpng 1.6's simplified-API struct, relying on the toolchain's C record-layout guarantee) and `png_image_write_to_file` is a `[C] EXTERN` function taking the record by `VAR` (= by pointer)

This depends on `pascal-1981` toolchain features documented in its `docs/c-abi-foreign-functions.md`: the `[C]` FFI, the C record-layout guarantee, the `WORD8`/`INTEGER8` extension types with the `WRD8` retype, and the heap super-array host-buffer pattern (`NEW(p, n)` buffers whose pointers coerce to `ADRMEM`/`void*` parameters, with `INTEGER32` indexing).

Prerequisites:

- `pascal-1981` installed into the active Python environment with `pip` (a version with `WORD8`/`INTEGER8` and the record-layout guarantee)
- the installed package's runtime archives (`libpascalrt.a` for CPU, and `libpascalrt_cuda.a` for CUDA builds)
- `clang`
- `libpng` (runtime library; no development headers are needed anymore, since nothing compiles against `png.h`)
- Python with whatever `pascal-1981` itself needs (`llvmlite`, etc.)

## Installed compiler/runtime flow

Install `pascal-1981` into the active virtual environment with `pip`. That can be done either:

- from a local on-disk checkout, or
- directly from the GitHub repository.

The Makefile invokes the compiler through the `pascal1981` console script (gcc-style stages: `-S` compiles to LLVM IR/PTX, `-c` assembles to an object, and a plain invocation links) and discovers the runtime archive gcc-style with `pascal1981 -print-file-name=libpascalrt.a`; it does not assume a sibling `../pascal-1981` source checkout or build the runtime out-of-tree.

A manual rebuild equivalent to the Makefile looks like this:

```bash
cd pascal
make clean
mkdir -p build
pascal1981 --dialect extended -c mandelbrot_host.pas -o build/host.o
pascal1981 --dialect extended -c mandelbrot.pas -o build/dev.o
clang build/host.o build/dev.o \
  "$(pascal1981 -print-file-name=libpascalrt.a)" -lpng -lm -o mandelbrot
./mandelbrot pip_build.png 1 d 0
```

The final link is an explicit `clang` invocation rather than the driver's own
link step because the driver compiles exactly one source file per invocation
and has no way to add the device unit's object to the link line.

## Makefile variants

Three equivalent Makefiles are provided; pick per taste.  All of them support
the same targets (`all`, `run`, `clean`) and the same `DEVICE`/`SM`/`RUN_ARGS`
variables, and all three produce byte-identical output on the CPU device.

- **`Makefile` — Option A (default): gcc-style per-unit objects + explicit
  final link.**  Each unit is compiled and assembled by the driver itself
  (`pascal1981 -c`), then `clang` links the objects with the runtime archive
  located via `pascal1981 -print-file-name=libpascalrt.a`.
- **`Makefile.driver-link` — Option B: the driver performs the final link.**
  `make -f Makefile.driver-link run`.  One driver invocation compiles the
  host unit and links the executable (the driver appends its bundled
  `libpascalrt.a` automatically); the device unit's object is passed through
  to the linker as a bare input file via `-Wl`.  CPU only — the CUDA branch
  falls back to Option A's explicit link, because the driver always appends
  its bundled CPU runtime, whose shim symbols would shadow the CUDA variants
  in `libpascalrt_cuda.a`.
- **`Makefile.ir` — Option C: keep inspectable `.ll` IR artifacts.**
  `make -f Makefile.ir run`.  The minimal diff against the pre-update
  Makefile: every unit stops at `-S` (LLVM IR, or PTX for the nvptx device
  unit) and `clang` is driven by hand.  Useful when you want the IR in
  `build/` to inspect, diff, or feed to other LLVM tools.

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

The Makefile uses the compiler driver's built-in PTX embedding path:

- compiles `mandelbrot.pas` to PTX with `-S --device-triple nvptx64-nvidia-cuda --sm <arch>`
- compiles the host to an object with `--device-backend cuda --embed-device-ptx build/dev.ptx -c`
- links the host object with the CUDA runtime shim archive (`libpascalrt_cuda.a`, found next to `$(pascal1981 -print-file-name=libpascalrt.a)`), `-lcuda`, and `-lpng -lm`
- runs the resulting `./mandelbrot` binary

Expected prerequisites for that path are:

- `libpascalrt_cuda.a` included in the pip-installed `pascal1981` package
- CUDA headers / driver library availability for `-lcuda`
- an NVIDIA GPU and working driver
- `llvmlite` with NVPTX support so `pascal1981 -S --device-triple nvptx64-nvidia-cuda` can emit PTX

This CUDA branch is wired in the Makefile, but it was **not executed in this VM** because this environment has no GPU and no full NVIDIA toolchain.

## Current defaults

The host program currently renders at:

- `3840 x 2160`
- `max_iter = 512`

The view/theme/precision are selected at launch time through the positional arguments above.

## Fidelity note

The colorization port was validated differentially against the old C shim at 640×360: the grayscale theme is byte-identical, and the color themes differ by at most 1 in a small fraction of channel values (well under 1%), because the shim computed its palette math in C `float` while the Pascal port uses `REAL` (f64). The kernels — and therefore the underlying iteration counts — are unchanged.
