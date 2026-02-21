# Building Clang and libclc for OpenCL → PTX → PyCUDA

This document describes how to build a self-contained Clang toolchain capable
of compiling OpenCL C kernels to PTX files that can be loaded directly by
PyCUDA — without a CUDA SDK, without a shim, and without NVVM intrinsics in
application code.

The two build steps are independent: Clang first, then libclc as a standalone
project against the installed Clang. No ROCm installation is required.

---

## 1. Building Clang

A standard upstream LLVM/Clang build is sufficient. The NVPTX and AMDGCN
backends are both part of mainline LLVM and require no special patches or
ROCm-specific configuration. RTTI must be enabled because libclc's CMake
will check for it.

```bash
cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm-22.1.0-rc3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind;compiler-rt" \
  -DCLANG_DEFAULT_CXX_STDLIB=libc++ \
  -DCLANG_DEFAULT_RTLIB=compiler-rt \
  -DCLANG_DEFAULT_LINKER=lld \
  -DLLVM_BUILD_LLVM_DYLIB=ON \
  -DLLVM_LINK_LLVM_DYLIB=ON \
  -DLLVM_ENABLE_RTTI=ON \
  ../llvm

make -j$(nproc)
make install
```

Note that `-DLLVM_TARGETS_TO_BUILD` is not specified; both the NVPTX and AMDGCN
backends are included in the default target set. After installation, confirm
that both are present:

```bash
$HOME/local/llvm-22.1.0-rc3/bin/llc --version | grep -E 'nvptx|amdgcn'
```

Expected output includes `nvptx`, `nvptx64`, and `amdgcn` target lines. The
presence of the AMDGCN backend in a build with no ROCm configuration is normal:
it has been part of upstream LLVM for years and is not a ROCm-specific patch.

---

## 2. Building libclc

libclc lives in the `libclc/` subdirectory of the LLVM monorepo and is **not**
built by the standard LLVM configuration above. It must be built as a separate
CMake project, pointed at the already-installed Clang.

```bash
mkdir build-libclc && cd build-libclc

cmake -G "Unix Makefiles" \
  -DCMAKE_INSTALL_PREFIX=$HOME/local/llvm-22.1.0-rc3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$HOME/local/llvm-22.1.0-rc3/bin/clang \
  -DCMAKE_CXX_COMPILER=$HOME/local/llvm-22.1.0-rc3/bin/clang++ \
  -DLLVM_CONFIG=$HOME/local/llvm-22.1.0-rc3/bin/llvm-config \
  -DLIBCLC_TARGETS_TO_BUILD="amdgcn--;amdgcn-amd-amdhsa;nvptx64--;nvptx64-nvidia-cuda" \
  ../libclc

make -j$(nproc)
make install
```

Two notes on configuration. First, the compiler variables must point at the
same Clang that will later use the libraries; using a different compiler risks
LLVM bitcode format version mismatches that produce silent link failures.
Second, `LIBCLC_TARGETS_TO_BUILD` uses exact triple names as enumerated by
libclc's CMake — `amdgcn-amd-amdhsa` (note the `amd` vendor field) rather than
the abbreviated `amdgcn--amdhsa`. Both AMD and NVIDIA targets are included here;
either can be omitted if not needed.

After installation, the relevant bitcode files will be at:

```
$HOME/local/llvm-22.1.0-rc3/share/clc/nvptx64--.bc
$HOME/local/llvm-22.1.0-rc3/share/clc/nvptx64-nvidia-cuda.bc
$HOME/local/llvm-22.1.0-rc3/share/clc/amdgcn-amd-amdhsa.bc
$HOME/local/llvm-22.1.0-rc3/share/clc/gfx1100-amdgcn--.bc
... (one file per GPU microarchitecture across the full GCN/RDNA range)
```

---

## 3. Writing the OpenCL Kernel

With libclc available, standard OpenCL built-ins like `get_global_id` work
directly. Before libclc, this function had no implementation for the NVPTX
target and had to be shimmed with raw NVVM PTX register intrinsics:

```c
// This shim is no longer needed when libclc is linked.
inline size_t get_global_id(uint dim) {
    if (dim == 0) return __nvvm_read_ptx_sreg_tid_x()
                       + __nvvm_read_ptx_sreg_ctaid_x()
                       * __nvvm_read_ptx_sreg_ntid_x();
    if (dim == 1) return __nvvm_read_ptx_sreg_tid_y()
                       + __nvvm_read_ptx_sreg_ctaid_y()
                       * __nvvm_read_ptx_sreg_ntid_y();
    return 0;
}
```

With libclc, the kernel source is clean standard OpenCL C:

```c
__kernel void my_kernel(__global float* output,
                        const int width,
                        const int height)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    if (x >= width || y >= height) return;

    output[y * width + x] = /* ... */;
}
```

For PyCUDA interop the PTX symbol name must be unmangled. Clang's OpenCL
front-end preserves the kernel name as-is, but it is worth verifying in the
generated PTX:

```bash
grep '\.visible .entry' kernel.ptx
```

---

## 4. Compiling an OpenCL Kernel to PTX

```bash
$HOME/local/llvm-22.1.0-rc3/bin/clang \
  -target nvptx64-nvidia-cuda -march=sm_86 \
  -x cl -O3 -cl-std=CL2.0 \
  -Xclang -finclude-default-header \
  -Xclang -target-feature -Xclang +ptx71 \
  -Xclang -mlink-builtin-bitcode \
  -Xclang $HOME/local/llvm-22.1.0-rc3/share/clc/nvptx64-nvidia-cuda.bc \
  -S mandelbrot.cl -o mandelbrot.ptx
```

The flags break down as follows.

**`-target nvptx64-nvidia-cuda`** must match the target triple embedded in the
libclc bitcode file. Using the bare `-target nvptx64` causes a triple mismatch
at link time. The two available NVIDIA bitcode files differ in their triple:
`nvptx64--.bc` (bare) and `nvptx64-nvidia-cuda.bc` (full CUDA triple); for
this use case the latter is correct.

**`-march=sm_86`** selects the GPU microarchitecture. Adjust to match your
hardware.

**`-Xclang -finclude-default-header`** provides the OpenCL built-in
*declarations* — type signatures and prototypes for functions like
`get_global_id`. This is distinct from libclc, which provides the
*implementations*. Both are required; they are complementary.

**`-Xclang -target-feature -Xclang +ptx71`** pins the PTX ISA version emitted.
Without it, Clang may default to an older PTX version incompatible with
`sm_86`. PTX 7.1 is the minimum for Ampere.

**`-Xclang -mlink-builtin-bitcode`** is the correct flag for linking libclc, as
opposed to the more blunt `-mlink-bitcode-file`. The "builtin" variant tells the
compiler that the bitcode contains library symbols, allowing it to internalize
them and dead-strip any routines the kernel does not call. This is also the
mechanism the HIP driver uses internally when processing `--hip-device-lib`.

---

## 5. Validating the PTX

Before loading through PyCUDA, validate the PTX with `ptxas`:

```bash
ptxas -arch=sm_86 -v -o mandelbrot.cubin mandelbrot.ptx
```

A clean run confirms the PTX is well-formed. The `-v` flag additionally reports
register count, shared memory consumption, and stack frame size — useful early
indicators of register spilling or unexpectedly large frames. The resulting
CUBIN can be disassembled to SASS if desired:

```bash
cuobjdump --dump-sass mandelbrot.cubin
```

---

## 6. Loading in PyCUDA

The compiled PTX is loaded in PyCUDA identically to one produced by `nvcc`:

```python
import pycuda.autoinit
import pycuda.driver as cuda

with open("mandelbrot.ptx", "r") as f:
    ptx = f.read()

mod = cuda.module_from_buffer(ptx.encode())
fn  = mod.get_function("mandelbrot")
```

No special handling is needed to distinguish OpenCL-derived PTX from
CUDA-derived PTX. From PyCUDA's perspective they are identical.

---

## Summary

| Step | Tool | Output |
|---|---|---|
| Build compiler | CMake + `make` against `../llvm` | `$HOME/local/llvm-22.1.0-rc3/bin/clang` |
| Build device library | CMake + `make` against `../libclc` | `share/clc/nvptx64-nvidia-cuda.bc` |
| Compile kernel | `clang -x cl ... -mlink-builtin-bitcode` | `mandelbrot.ptx` |
| Validate | `ptxas -arch=sm_86 -v` | `mandelbrot.cubin` |
| Load | `pycuda.driver.module_from_buffer` | callable GPU function |

The only external dependency at runtime is the CUDA driver itself (`libcuda.so`
via PyCUDA). No CUDA SDK, no ROCm installation, and no NVVM intrinsic shims are
required in application code.
