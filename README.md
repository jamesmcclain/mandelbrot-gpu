# Mandelbrot GPU Renderer

<img width="768" height="432" alt="image" src="https://github.com/user-attachments/assets/0f95519b-3654-4268-9613-ee4768a97d18" />

A GPU-accelerated Mandelbrot set renderer that produces high-resolution PNG images. The compute kernel is written in OpenCL C (with an optional CUDA variant), and runs on both NVIDIA and AMD hardware via a pluggable Python backend system.

## Backends

| Backend | Flag | Kernel file | Runtime | Hardware |
|---|---|---|---|---|
| CUDA | `--backend cuda` | `mandelbrot.ptx` | PyCUDA | NVIDIA |
| OpenCL | `--backend opencl` | `mandelbrot.cl` | PyOpenCL | NVIDIA or AMD |
| AMDHSA | `--backend amdhsa` | `mandelbrot.hsaco` | PyOpenCL | AMD (experimental) |

The `cuda` backend is the default. The `opencl` backend is the recommended path for AMD hardware and works on any OpenCL-capable device. The `amdhsa` backend loads a pre-compiled `.hsaco` binary and is reserved for future work (see [docs/amd-opencl-porting.md](docs/amd-opencl-porting.md) for status).

## Rendering

```bash
python mandelbrot.py                                     # render all 10 built-in views at 4K (CUDA)
python mandelbrot.py --backend opencl                    # same, using OpenCL (AMD or NVIDIA)
python mandelbrot.py --views seahorse_valley             # single named view
python mandelbrot.py --width 1920 --height 1080 --theme fire
python mandelbrot.py --views "myspot:-0.76:-0.70:0.09:0.15:1024:ice"  # custom view
```

Pass `--list-views` to print all built-in views (Overview, Seahorse Valley, Elephant Valley, Double Spiral, etc.) with their default coordinates, iteration counts, and themes.

**Color themes:** `classic`, `fire`, `ice`, `rainbow`, `emacs`, `grayscale`

Output PNGs are written to the current directory by default; use `--output-dir` to change this.

## Compiling the Kernel

### CUDA backend (`mandelbrot.ptx`)

**Via nvcc (simplest):**
```bash
bash scripts/compile-cuda.sh    # requires nvcc; targets sm_86 by default
```

**Via Clang + OpenCL (no CUDA toolkit needed):**
```bash
bash scripts/compile-opencl-cuda.sh   # requires a libclc-enabled Clang build
```

See [docs/clang-opencl.md](docs/clang-opencl.md) for instructions on building Clang and libclc from source.

### AMDHSA backend (`mandelbrot.hsaco`)

```bash
bash scripts/compile-opencl-amdhsa.sh   # requires a libclc-enabled Clang build; targets gfx90c
```

Adjust `-mcpu=gfx90c` in the script for your GPU (run `amdgpu-arch` to find yours). The `opencl` backend compiles `.cl` source at runtime via the AMD driver and does not require a pre-compiled binary.

## Docker

Pre-built Docker environments are provided for both hardware families.

**NVIDIA:**
```bash
docker build -t mandelbrot-cuda -f docker/Dockerfile.cuda .
bash scripts/shell-cuda.sh   # launches the container with GPU access
```

**AMD (ROCm):**
```bash
docker build -t mandelbrot-rocm -f docker/Dockerfile.rocm .
bash scripts/shell-rocm.sh   # mounts /dev/kfd and /dev/dri, sets HSA_OVERRIDE_GFX_VERSION
```

> **Note for APU / integrated Vega users (e.g. Ryzen 5700G / `gfx90c`):** `gfx90c` is not on AMD's official ROCm supported GPU list. Set `HSA_OVERRIDE_GFX_VERSION=9.0.12` in the container (already done by `shell-rocm.sh`) to make the HSA runtime recognize the device.

## Dependencies

**NVIDIA path:**
- Python 3 with `numpy`, `Pillow`, and `pycuda`
- An NVIDIA GPU with a working CUDA driver
- `nvcc` **or** a libclc-enabled Clang build for kernel compilation

**AMD / OpenCL path:**
- Python 3 with `numpy`, `Pillow`, and `pyopencl`
- ROCm (easiest via the `rocm/dev-ubuntu-22.04:6.2-complete` Docker image)

## Documentation

- [docs/clang-opencl.md](docs/clang-opencl.md) — Building Clang and libclc from source; compiling OpenCL kernels to PTX the CUDA SDK.
- [docs/amd-opencl-porting.md](docs/amd-opencl-porting.md) — Porting notes for the NVIDIA → AMD migration: toolchain mapping, Docker setup, `gfx90c` quirks, binary-loading status, and the Python backend refactor.
