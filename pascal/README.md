# Pascal host renderer

This directory contains a Pascal host renderer for the Mandelbrot `DEVICE UNIT`.
The renderer has no C shim.

## Files

- `mandelbrot.pas` and `mandelbrot.inc` contain the device kernels.
- `mandelbrot_host.pas` allocates buffers, launches kernels, reads results, colors pixels, and writes the PNG file.
- `Makefile` builds the CPU or CUDA program.

The host source uses the Pascal C foreign-function interface for libpng.
It uses `WORD8`, `INTEGER8`, and heap super arrays.

## Requirements

Install these tools and files:

- `pascal1981` on `PATH`
- `clang`
- `libpng`
- `libpascalrt.a` for a CPU build
- `libpascalrt_cuda.a` for a CUDA build

The compiler commands use `--dialect extended`.
Set `RUNTIME_DIR` to the directory that contains the runtime archives.
Set `RUNTIME_CPU` or `RUNTIME_CUDA` to override one archive path.

## Build the CPU program

1. Change to this directory.
2. Run the command with the runtime directory.

```bash
make DEVICE=cpu RUNTIME_DIR=/path/to/runtime
```

The CPU runtime emulates the device launch geometry.
It does not require an NVIDIA GPU or CUDA tools.

To build and run the default image, run:

```bash
make run RUNTIME_DIR=/path/to/runtime
```

The default arguments are `mandelbrot.png 1 d 0`.
Set `RUN_ARGS` to use different arguments.

```bash
make run RUNTIME_DIR=/path/to/runtime RUN_ARGS="demo.png 4 s 3 320 180"
```

## Program arguments

The program uses positional arguments.

```text
mandelbrot <outfile> <view> <prec> <theme> [<width> <height>]
```

- `outfile` is the PNG output file.
- `view` selects a view from `1` through `4`.
- `prec` selects `f32` for `s` or `S`. Other values select `f64`.
- `theme` selects a color theme from `0` through `5`.
- `width` and `height` are an optional pair of decimal dimensions from `1` through `8192`. Omit both to render the default `3840 x 2160` image.

The four required arguments use the Pascal runtime's normal missing-argument prompt. If either optional dimension is supplied, supply both.

## CPU regression test

Run the headless CPU regression test after a CPU build:

```bash
make DEVICE=cpu RUNTIME_DIR=/path/to/runtime test-cpu
```

The test renders the fixed `320 x 180` grayscale f64 case twice. It checks each process exit status, PNG existence and IHDR dimensions, byte determinism, the committed SHA-256 in `tests/cpu_grayscale_320x180.sha256`, and pixels against an independent scalar Python CPU implementation of the CUDA/Pascal kernel. It uses only the Python standard library.

## Build the CUDA program

A CUDA build needs these additional items:

- `libpascalrt_cuda.a`
- CUDA driver libraries and headers
- An NVIDIA GPU and driver to run the program
- LLVM NVPTX support in `pascal1981`

Run this command on a CUDA system:

```bash
make DEVICE=cuda RUNTIME_DIR=/path/to/runtime
```

The PTX blob can be checked without CUDA headers or a GPU:

```bash
make DEVICE=cuda RUNTIME_DIR=/path/to/runtime check-ptx-blob
```

This checks that `build/dev_ptx_blob.o` exports `__pas_device_ptx`.

A complete CUDA executable additionally requires `cuda.h` to build
`libpascalrt_cuda.a` and a CUDA driver stub (`libcuda.so`) to link. On systems
without those files, PTX generation and the blob-symbol check still work, but
the final CUDA link cannot be validated. The CPU regression test remains the
supported validation path there.

The Makefile generates PTX with `--emit-ptx`.
It passes `--device-triple nvptx64-nvidia-cuda` and `--ptx-cpu`.
`SM=sm_XX` sets the default value of `PTX_CPU`.
Set `PTX_CPU` to override that value.

The Makefile writes `build/dev.ptx`.
It creates `build/dev_ptx_blob.s` with `.incbin`.
The assembly defines `__pas_device_ptx` and adds a zero byte.
The build compiles the assembly into `build/dev_ptx_blob.o`.
The CUDA link includes that object and `libpascalrt_cuda.a`.

## Notes

Use `Makefile` as the build entry point.
`Makefile.driver-link` and `Makefile.ir` use an obsolete driver interface.

The default image size is `3840 x 2160`.
The default iteration limit is `512`.

The Pascal color output was compared with the former C shim at `640 x 360`.
The grayscale output was byte-identical.
The color output differed by no more than one channel value in fewer than one percent of pixels.
