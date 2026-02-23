# Mandelbrot GPU Renderer

<img width="768" height="432" alt="image" src="https://github.com/user-attachments/assets/0f95519b-3654-4268-9613-ee4768a97d18" />

A GPU-accelerated Mandelbrot set renderer that produces high-resolution PNG images using NVIDIA hardware. The compute kernel is written in both OpenCL and CUDA, and a Python driver script handles kernel loading, argument parsing, colorization, and image output via PyCUDA.

## How It Works

The iteration kernel (`mandelbrot.cl` / `mandelbrot.cu`) runs entirely on the GPU — each thread computes the escape-time value for one pixel in parallel. The Python driver (`mandelbrot-nvidia.py`) loads a pre-compiled PTX binary, launches the kernel, copies results back to the host, applies a color theme, and saves the result as a PNG.

## Rendering

```bash
python mandelbrot-nvidia.py                        # render all 10 built-in views at 4K
python mandelbrot-nvidia.py --views overview       # render a single named view
python mandelbrot-nvidia.py --width 1920 --height 1080 --theme fire
python mandelbrot-nvidia.py --views "My Spot:-0.75:0.1:-0.1:0.1:1024:ice"  # custom view
```

Pass `--list-views` to print all available built-in views (Seahorse Valley, Elephant Valley, etc.) and their default settings. Output PNGs are written to the current directory by default; use `--output-dir` to change this.

**Color themes:** `classic`, `fire`, `ice`, `rainbow`, `emacs`, `grayscale`

## Compiling the Kernel

The Python driver expects a `mandelbrot.ptx` file in the working directory. Two compilation paths are provided, targeting `sm_86` (Ampere); adjust `-march` / `-arch` for other GPU generations.

**Via CUDA (simpler):**
```bash
bash scripts/compile-cuda.sh     # requires nvcc
```

**Via Clang + OpenCL (no CUDA toolkit needed):**
```bash
bash scripts/compile-opencl.sh   # requires a libclc-enabled Clang build
```

See `docs/clang-opencl.md` for details on building the Clang/OpenCL toolchain and working around common issues.

## Dependencies

- Python 3 with `numpy`, `Pillow`, and `pycuda`
- An NVIDIA GPU with a working CUDA driver
- `nvcc` (CUDA toolkit) **or** a libclc-enabled Clang build for kernel compilation
