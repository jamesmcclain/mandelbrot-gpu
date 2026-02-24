# Porting a Clang/OpenCL GPU Compute Pipeline from NVIDIA to AMD

## Background

The starting point was a working NVIDIA pipeline:

- **Kernel source:** `mandelbrot.cl` (OpenCL C)
- **Compilation:** `clang` targeting `nvptx64-nvidia-cuda`, producing `mandelbrot.ptx`
- **Host runtime:** Python + `pycuda`, loading the `.ptx` via `cuda.module_from_file()`

The goal was to port this to AMD hardware using the same `clang`-based offline
compilation approach, running the kernel via Python on the AMD side.

---

## Target Hardware

The AMD machine's GPU was identified with:

```bash
amdgpu-arch
# → gfx90c
```

`gfx90c` is the AMDGPU target for AMD Ryzen APUs with Vega integrated graphics
(e.g. the 5700G family). It is **not** on AMD's official ROCm supported GPU list
(which is datacenter-focused), but works in practice with the ROCm runtime.

---

## Mapping the NVIDIA Toolchain to AMD

| Concern | NVIDIA | AMD |
|---|---|---|
| Compiler target triple | `-target nvptx64-nvidia-cuda` | `-target amdgcn-amd-amdhsa` |
| Architecture | `-march=sm_86` | `-mcpu=gfx90c` |
| Built-in header | `-Xclang -finclude-default-header` | same, unchanged |
| Virtual ISA version pin | `-Xclang -target-feature -Xclang +ptx71` | **not needed** — `-mcpu` fully specifies the real ISA |
| libclc builtins | `-Xclang -mlink-builtin-bitcode -Xclang nvptx64-nvidia-cuda.bc` | `-Xclang -mlink-builtin-bitcode -Xclang amdgcn-amd-amdhsa.bc` |
| Output format | `-S` → `.ptx` (text, virtual ISA) | `-c` → `.hsaco` (ELF binary, real ISA) |
| Python runtime | `pycuda` | `pyopencl` |

**Key architectural difference:** PTX is a stable virtual ISA that NVIDIA's
driver JIT-compiles at load time, making a single binary forward-compatible
across GPU generations. AMDGPU `.hsaco` files are real ISA binaries tied to a
specific `gfx` target.

---

## Docker Setup

The AMD machine had ROCm working in Docker but no ROCm stack installed on the
host. The recommended image was:

```
rocm/dev-ubuntu-22.04:6.2-complete
```

The `-complete` tag includes the full ROCm stack including the OpenCL runtime
(`rocm-opencl-runtime`).

### Confirming the runtime sees the GPU

```bash
# Quickest check — ROCm's own tool
rocminfo

# OpenCL-specific check
clinfo

# Python check (after pip install pyopencl)
python3 -c "import pyopencl as cl; ctx = cl.create_some_context(interactive=False); print(ctx.devices)"
```

### gfx90c quirk: HSA_OVERRIDE_GFX_VERSION

Because `gfx90c` is an APU not officially listed in the ROCm supported GPU
list, the HSA runtime needs to be told to treat it as a supported GFX9 device.
Without this, `rocminfo` will list the CPU but not the GPU.

The correct value is `9.0.12` (`gfx90c` = major.minor.stepping 9.0.12):

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.12
```

This should be set inside the container (or passed via `-e` to `docker run`).

### docker run command

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.12 \
  rocm/dev-ubuntu-22.04:6.2-complete
```

---

## Compilation

Compilation is done on the build machine (not the AMD machine), so no ROCm
installation is needed there.

### Working compile command

```bash
$HOME/local/llvm-22.1.0-rc3/bin/clang \
  -target amdgcn-amd-amdhsa -mcpu=gfx90c \
  -x cl -O3 -cl-std=CL2.0 \
  -Xclang -finclude-default-header \
  -nogpulib \
  -Xclang -mlink-builtin-bitcode \
  -Xclang $HOME/local/llvm-22.1.0-rc3/share/clc/amdgcn-amd-amdhsa.bc \
  -c mandelbrot.cl -o mandelbrot.hsaco
```

### Errors encountered along the way

**Error 1:** Missing ROCm device libs

```
clang: error: cannot find ROCm device library; provide its path via
'--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to
build without ROCm device library
```

When targeting `amdgcn-amd-amdhsa`, clang automatically searches for ROCm
device libs. Since we are supplying libclc instead, `-nogpulib` suppresses
that search.

**Error 2:** Wrong libclc filename

Initial guess of `amdgcn--amdhsa.bc` (double-dash, no vendor) was wrong.
`find` revealed the correct filenames. The per-GPU file `gfx90c-amdgcn--.bc`
exists but has triple `amdgcn-unknown-unknown`, causing:

```
error: non-hsa intrinsic with hsa target
```

The correct file is the generic `amdgcn-amd-amdhsa.bc` at the bottom of the
`share/clc/` directory, which has the matching `amdgcn-amd-amdhsa` triple and
correct HSA ABI intrinsics for `get_global_id()` etc.

### libclc files available under `share/clc/`

The per-GPU files follow the pattern `<gpu>-amdgcn--.bc` (e.g.
`gfx90c-amdgcn--.bc`, `gfx1030-amdgcn--.bc`). These are built for
`amdgcn-unknown-unknown` and are **not** suitable for HSA targets. The correct
file for HSA targets is the single generic:

```
amdgcn-amd-amdhsa.bc
```

---

## Verifying the Binary

The AMD equivalent of running `ptxas` against a `.ptx` file to check for
unresolved symbols is:

```bash
llvm-nm mandelbrot.hsaco
```

### Expected clean output

```
0000000000000194 T __clang_ocl_kern_imp_mandelbrot
0000000000000000 T mandelbrot
0000000000000000 a mandelbrot.has_dyn_sized_stack
0000000000000000 a mandelbrot.has_recursion
0000000000000000 R mandelbrot.kd
0000000000000000 a mandelbrot.num_agpr
0000000000000009 a mandelbrot.num_vgpr
0000000000000010 a mandelbrot.numbered_sgpr
0000000000000000 a mandelbrot.private_seg_size
0000000000000000 a mandelbrot.uses_flat_scratch
0000000000000001 a mandelbrot.uses_vcc
```

What to look for:

- `T mandelbrot` — the kernel entry point is defined (not `U` for undefined)
- `T __clang_ocl_kern_imp_mandelbrot` — the OpenCL wrapper clang generates
- `R mandelbrot.kd` — the kernel descriptor (HSA metadata the runtime needs to dispatch the kernel)
- `a` symbols — assembler metadata: VGPR/SGPR register counts, stack usage etc.

No `U` symbols means no unresolved references. The binary is complete.

`.hsaco` is a standard ELF file, so `readelf` also works:

```bash
readelf -s mandelbrot.hsaco   # symbol table
readelf -h mandelbrot.hsaco   # ELF header / basic info
file mandelbrot.hsaco         # confirm ELF identity
```

---

## Python Host Code Structure

The host code was refactored to support multiple backends cleanly.

### Files

| File | Role |
|---|---|
| `mandelbrot.py` | Backend-agnostic host code: argument parsing, view definitions, colorization, image saving |
| `mandelbrot_cuda.py` | PyCUDA backend: loads `.ptx`, manages device memory, launches kernel |
| `mandelbrot_opencl.py` | PyOpenCL backend: compiles `.cl` at runtime, manages buffers, launches kernel |
| `mandelbrot_amdhsa.py` | Reserved for future binary-loading path: will load `.hsaco` directly when that is resolved |
| `mandelbrot.cl` | OpenCL C kernel source (shared by CUDA and OpenCL paths) |
| `mandelbrot.ptx` | Compiled NVIDIA PTX (produced by clang, consumed by CUDA backend) |
| `mandelbrot.hsaco` | Compiled AMD binary (produced by clang, reserved for amdhsa backend) |

### Backend interface

Each backend module exposes exactly two functions:

```python
def load_kernel(file):
    # Returns an opaque handle; type is backend-specific
    ...

def run_kernel(handle, WIDTH, HEIGHT, max_iter, x_min, x_max, y_min, y_max):
    # Returns a (HEIGHT, WIDTH) numpy int32 array of iteration counts
    ...
```

### --backend flag

```
python3 mandelbrot.py --backend cuda       # default
python3 mandelbrot.py --backend opencl     # AMD (or any OpenCL device)
python3 mandelbrot.py --backend amdhsa     # future: binary loading
```

---

## Why Binary Loading Doesn't Work (Yet) for AMD

Attempting to load `mandelbrot.hsaco` via `clCreateProgramWithBinary` /
`clBuildProgram` in PyOpenCL produced:

```
pyopencl._cl.RuntimeError: clBuildProgram failed: BUILD_PROGRAM_FAILURE
Error while Codegen phase: the binary is incomplete
```

This occurred even though `llvm-nm` confirmed the binary was complete.  The
problem is that AMD's OpenCL runtime treats `clBuildProgram` on a binary as a
further codegen/finalization step, and it is very sensitive to the exact
device-string match (including XNACK mode suffix, e.g. `gfx90c:xnack+`).
Compiling with `-mcpu=gfx90c:xnack+` did not resolve it.

The workaround — and the well-tested path for AMD OpenCL — is to pass the `.cl`
source directly and let the runtime compile it:

```python
program = cl.Program(ctx, source).build(options="-cl-std=CL2.0 -O3")
```

AMD's OpenCL runtime compiles and caches the result, so repeated runs do not
re-compile from scratch. Resolving the binary loading path is left for future
work in `mandelbrot_amdhsa.py`.
