# Option A (default): gcc-style per-unit objects + explicit final link.
#
# Each Pascal unit is compiled and assembled by the driver itself
# (`pascal1981 -c`, which runs clang under the hood), and the final link is
# an explicit clang invocation.  The driver's own link step is not used
# because it compiles exactly one source file per invocation and has no way
# to add the device unit's object to the link line.
#
# The compiler is resolved from PATH. Set RUNTIME_DIR, RUNTIME_CPU, or
# RUNTIME_CUDA to select the matching runtime archives.
#
# Alternatives (see README.md):
#   Makefile.driver-link  - Option B: let the driver perform the final link
#   Makefile.ir           - Option C: keep .ll IR artifacts (minimal diff vs
#                           the pre-update Makefile)

DEVICE ?= cpu
# SM remains an input alias for existing invocations.  PTX_CPU is the current
# pascal1981 option value and uses LLVM names such as sm_70 or sm_86.
SM ?= sm_86
PTX_CPU ?= $(SM)
CUDA_HOME ?= /usr/local/cuda

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD    := $(THIS_DIR)/build
PAS          ?= pascal1981
RUNTIME_DIR  ?=
RUNTIME_CPU  ?= $(if $(RUNTIME_DIR),$(RUNTIME_DIR)/libpascalrt.a)
RUNTIME_CUDA ?= $(if $(RUNTIME_DIR),$(RUNTIME_DIR)/libpascalrt_cuda.a)
FEATURES     := --dialect extended
NVPTX_TRIPLE := nvptx64-nvidia-cuda

HOST_SRC    := mandelbrot_host.pas
DEVICE_UNIT := mandelbrot.pas
EXE         := mandelbrot
RUN_ARGS    ?= mandelbrot.png 1 d 0

PYTHON ?= python3

.PHONY: all run test-cpu check-ptx-blob clean check-runtime-cpu check-runtime-cuda

all: $(EXE)

run: $(EXE)
	./$(EXE) $(RUN_ARGS)

test-cpu: $(EXE)
	$(PYTHON) tests/cpu_render_test.py --binary ./$(EXE)

clean:
	rm -rf $(BUILD) $(EXE) *.png

check-runtime-cpu:
	@test -n "$(RUNTIME_CPU)" && test -f "$(RUNTIME_CPU)" || { echo "Missing Pascal CPU runtime; set RUNTIME_DIR or RUNTIME_CPU" >&2; exit 1; }

check-runtime-cuda:
	@test -n "$(RUNTIME_CUDA)" && test -f "$(RUNTIME_CUDA)" || { echo "Missing Pascal CUDA runtime; set RUNTIME_DIR or RUNTIME_CUDA" >&2; exit 1; }

$(BUILD):
	mkdir -p $(BUILD)

ifeq ($(DEVICE),cuda)
# Device unit -> PTX.  The native driver stops after code generation for
# --emit-ptx; its output is embedded below for the CUDA runtime shim.
$(BUILD)/dev.ptx: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) $(FEATURES) --emit-ptx --device-triple $(NVPTX_TRIPLE) --ptx-cpu $(PTX_CPU) -o $@ $<

# The CUDA runtime expects a NUL-terminated PTX string at __pas_device_ptx.
$(BUILD)/dev_ptx_blob.s: $(BUILD)/dev.ptx | $(BUILD)
	printf '\t.section .rodata\n\t.globl __pas_device_ptx\n__pas_device_ptx:\n\t.incbin "$<"\n\t.byte 0\n' > $@

$(BUILD)/dev_ptx_blob.o: $(BUILD)/dev_ptx_blob.s | $(BUILD)
	clang -c $< -o $@

check-ptx-blob: $(BUILD)/dev_ptx_blob.o
	@nm -g $< | grep -Eq '[[:space:]][A-Z][[:space:]]+__pas_device_ptx$$' || { echo 'missing __pas_device_ptx in $<' >&2; exit 1; }

# Host unit -> object for the CUDA backend.  PTX reaches the runtime through
# dev_ptx_blob.o rather than the removed --embed-device-ptx driver option.
$(BUILD)/host.o: $(HOST_SRC) | $(BUILD)
	$(PAS) $(FEATURES) --device-backend cuda -c -o $@ $<

# The CUDA link must use libpascalrt_cuda.a, never libpascalrt.a: both archives
# define the same shim symbols and the CPU variant would win a collision.
$(EXE): $(BUILD)/host.o $(BUILD)/dev_ptx_blob.o | check-runtime-cuda
	clang $^ $(RUNTIME_CUDA) -L$(CUDA_HOME)/lib64/stubs -lcuda -lpng -lm -o $@
else
$(BUILD)/dev.o: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) $(FEATURES) -c -o $@ $<

$(BUILD)/host.o: $(HOST_SRC) | $(BUILD)
	$(PAS) $(FEATURES) -c -o $@ $<

$(EXE): $(BUILD)/host.o $(BUILD)/dev.o | check-runtime-cpu
	clang $^ $(RUNTIME_CPU) -lpng -lm -o $@
endif
