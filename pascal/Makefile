# Option A (default): gcc-style per-unit objects + explicit final link.
#
# Each Pascal unit is compiled and assembled by the driver itself
# (`pascal1981 -c`, which runs clang under the hood), and the final link is
# an explicit clang invocation.  The driver's own link step is not used
# because it compiles exactly one source file per invocation and has no way
# to add the device unit's object to the link line.
#
# The runtime archive is located gcc-style, via
# `pascal1981 -print-file-name=libpascalrt.a`; the CUDA runtime lives in the
# same directory.
#
# Alternatives (see README.md):
#   Makefile.driver-link  - Option B: let the driver perform the final link
#   Makefile.ir           - Option C: keep .ll IR artifacts (minimal diff vs
#                           the pre-update Makefile)

DEVICE ?= cpu
SM ?= sm_86
CUDA_HOME ?= /usr/local/cuda

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD    := $(THIS_DIR)/build

PAS          := pascal1981
FEATURES     := --dialect extended
NVPTX_TRIPLE := nvptx64-nvidia-cuda

# gcc-style runtime discovery: ask the driver where its runtime archive lives.
RUNTIME_CPU  := $(shell $(PAS) -print-file-name=libpascalrt.a)
RUNTIME_CUDA := $(dir $(RUNTIME_CPU))libpascalrt_cuda.a

HOST_SRC    := mandelbrot_host.pas
DEVICE_UNIT := mandelbrot.pas
EXE         := mandelbrot
RUN_ARGS    ?= mandelbrot.png 1 d 0

.PHONY: all run clean check-runtime-cpu check-runtime-cuda

all: $(EXE)

run: $(EXE)
	./$(EXE) $(RUN_ARGS)

clean:
	rm -rf $(BUILD) $(EXE) *.png

check-runtime-cpu:
	@test -f "$(RUNTIME_CPU)" || { echo "Missing Pascal CPU runtime: $(RUNTIME_CPU)" >&2; exit 1; }

check-runtime-cuda:
	@test -f "$(RUNTIME_CUDA)" || { echo "Missing Pascal CUDA runtime: $(RUNTIME_CUDA)" >&2; exit 1; }

$(BUILD):
	mkdir -p $(BUILD)

ifeq ($(DEVICE),cuda)
# Device unit -> PTX assembly (nvptx device triple switches -S to PTX output).
$(BUILD)/dev.ptx: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) -S --device-triple $(NVPTX_TRIPLE) --sm $(SM) -o $@ $<

# Host unit -> object, with the device PTX embedded for the CUDA driver shim.
$(BUILD)/host.o: $(HOST_SRC) $(BUILD)/dev.ptx | $(BUILD)
	$(PAS) $(FEATURES) --device-backend cuda --embed-device-ptx $(BUILD)/dev.ptx -c -o $@ $<

# The CUDA link must use libpascalrt_cuda.a, never the driver's default
# libpascalrt.a: both archives define the same shim symbols, and the CPU
# variant would win any collision.  An explicit clang link keeps that
# ordering unambiguous.
$(EXE): $(BUILD)/host.o | check-runtime-cuda
	clang $< $(RUNTIME_CUDA) -L$(CUDA_HOME)/lib64/stubs -lcuda -lpng -lm -o $@
else
$(BUILD)/dev.o: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) $(FEATURES) -c -o $@ $<

$(BUILD)/host.o: $(HOST_SRC) | $(BUILD)
	$(PAS) $(FEATURES) -c -o $@ $<

$(EXE): $(BUILD)/host.o $(BUILD)/dev.o | check-runtime-cpu
	clang $^ $(RUNTIME_CPU) -lpng -lm -o $@
endif
