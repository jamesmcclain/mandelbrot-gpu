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

## Built-in Views

| Slug | Name | Theme | Max Iter | Description |
|---|---|---|---|---|
| `overview` | Overview | classic | 512 | The full Mandelbrot set — the iconic cardioid and primary bulb |
| `seahorse_valley` | Seahorse Valley | ice | 1024 | The classic 'Seahorse Valley' between the main cardioid and period-2 bulb |
| `elephant_valley` | Elephant Valley | fire | 1024 | The 'Elephant Valley' on the right side — elephants marching in a row |
| `double_spiral` | Double Spiral | emacs | 2048 | A stunning double-spiral inside Seahorse Valley |
| `starfish` | Starfish | rainbow | 1024 | The 'Starfish' — a five-armed spiral off the main body |
| `triple_spiral` | Triple Spiral | classic | 2048 | A triple spiral near the upper filaments |
| `mini_mandelbrot` | Mini Mandelbrot | emacs | 2048 | A self-similar 'mini-brot' on the real axis to the left of the main set |
| `feather` | Feather | ice | 2048 | A delicate feather-shaped region inside Seahorse Valley |
| `spiral_tendril` | Spiral Tendril | fire | 1024 | Spiralling tendrils on the tip of the left antenna |
| `quad_spiral` | Quad Spiral | rainbow | 2048 | A four-armed spiral galaxy in the upper filaments |

## Floating-Point Precision

The renderer supports both `float32` and `float64` kernels, selected via `--precision`:

```bash
python mandelbrot.py --precision single   # force float32 (fastest)
python mandelbrot.py --precision double   # force float64 (most accurate)
python mandelbrot.py --precision auto     # automatic selection (default)
```

In `auto` mode, `mandelbrot_precision.py` inspects the render parameters and selects the appropriate precision automatically. It upgrades to `float64` when the pixel spacing approaches the float32 ULP scale, or when sample orbit comparisons between float32 and float64 diverge — which typically occurs at deep zoom levels or with high iteration counts near the set boundary. For typical views at 4K, `float32` is usually sufficient.

## Custom Views

To render a region not in the built-in list, pass a custom view spec:

```
name:x_min:x_max:y_min:y_max[:max_iter[:theme]]
```

```bash
python mandelbrot.py --views "myview:-0.76:-0.70:0.09:0.15:1024:ice"
```

Fields after `y_max` are optional and default to `max_iter=1024` and `theme=classic`.

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

## Project Structure

```
mandelbrot.py               # Entry point: argument parsing, view resolution, image saving
mandelbrot_precision.py     # Auto precision selection: float32 vs float64 via orbit sampling
mandelbrot_cuda.py          # PyCUDA backend: loads .ptx, dispatches kernel
mandelbrot_opencl.py        # PyOpenCL backend: compiles .cl at runtime, dispatches kernel
mandelbrot_amdhsa.py        # AMD HSA backend: loads pre-compiled .hsaco binary
mandelbrot.cl               # OpenCL C kernel source (float32 + float64 entry points)
mandelbrot.cu               # CUDA C kernel source
scripts/
  compile-cuda.sh           # Compile .cu → .ptx via nvcc
  compile-opencl-cuda.sh    # Compile .cl → .ptx via Clang + libclc
  compile-opencl-amdhsa.sh  # Compile .cl → .hsaco via Clang + libclc
  shell-cuda.sh             # Launch CUDA Docker container
  shell-rocm.sh             # Launch ROCm Docker container
docker/
  Dockerfile.cuda           # NVIDIA image (CUDA + PyCUDA)
  Dockerfile.rocm           # AMD image (ROCm + PyOpenCL)
docs/
  clang-opencl.md           # Building Clang/libclc from source
  amd-opencl-porting.md     # NVIDIA → AMD porting notes
```

## Dependencies

**NVIDIA path:**
- Python 3 with `numpy`, `Pillow`, and `pycuda`
- An NVIDIA GPU with a working CUDA driver
- `nvcc` **or** a libclc-enabled Clang build for kernel compilation

**AMD / OpenCL path:**
- Python 3 with `numpy`, `Pillow`, and `pyopencl`
- ROCm (easiest via the `rocm/dev-ubuntu-22.04:6.2-complete` Docker image)

## Documentation

- [docs/clang-opencl.md](docs/clang-opencl.md) — Building Clang and libclc from source; compiling OpenCL kernels to PTX without the CUDA SDK.
- [docs/amd-opencl-porting.md](docs/amd-opencl-porting.md) — Porting notes for the NVIDIA → AMD migration: toolchain mapping, Docker setup, `gfx90c` quirks, binary-loading status, and the Python backend refactor.
