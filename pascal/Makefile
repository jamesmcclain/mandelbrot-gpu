DEVICE ?= cpu
SM ?= sm_86
CUDA_HOME ?= /usr/local/cuda

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD    := $(THIS_DIR)/build

PYTHON       ?= python3
PAS          := $(PYTHON) -m pascal1981
PASCALRT_DIR := $(shell $(PYTHON) -c 'import pathlib, pascal1981; print(pathlib.Path(pascal1981.__file__).parent)')
RUNTIME_CPU  := $(PASCALRT_DIR)/libpascalrt.a
RUNTIME_CUDA := $(PASCALRT_DIR)/libpascalrt_cuda.a
FEATURES     := --dialect extended

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
$(BUILD)/dev.ptx: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) --target ptx $< $@ --sm $(SM)

$(BUILD)/host.ll: $(HOST_SRC) $(BUILD)/dev.ptx | $(BUILD)
	$(PAS) $(FEATURES) --device-backend cuda --embed-device-ptx $(BUILD)/dev.ptx $< $@

$(EXE): $(BUILD)/host.ll | check-runtime-cuda
	clang $(BUILD)/host.ll $(RUNTIME_CUDA) \
	      -L$(CUDA_HOME)/lib64/stubs -lcuda -lpng -lm -o $@
else
$(BUILD)/dev.ll: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) $(FEATURES) $< $@

$(BUILD)/host.ll: $(HOST_SRC) | $(BUILD)
	$(PAS) $(FEATURES) $< $@

$(EXE): $(BUILD)/host.ll $(BUILD)/dev.ll | check-runtime-cpu
	clang $(BUILD)/host.ll $(BUILD)/dev.ll $(RUNTIME_CPU) -lpng -lm -o $@
endif
