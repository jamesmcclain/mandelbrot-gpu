DEVICE ?= cpu
SM ?= sm_86
CUDA_HOME ?= /usr/local/cuda

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO     := $(abspath $(THIS_DIR)/..)
PASCAL81 := $(abspath $(REPO)/../pascal-1981)
RUNTIME  := $(PASCAL81)/runtime
BUILD    := $(THIS_DIR)/build

PAS          := PYTHONPATH=$(PASCAL81)/src python3 -m pascal1981
RUNTIME_CPU  := $(RUNTIME)/build/libpascalrt.a
RUNTIME_CUDA := $(RUNTIME)/build/libpascalrt_cuda.a
FEATURES     := --dialect extended -f wide-integers

HOST_SRC    := mandelbrot_host.pas
DEVICE_UNIT := mandelbrot.pas
PNG_HELPER  := png_helper.c
EXE         := mandelbrot
RUN_ARGS    ?= mandelbrot.png 1 d 0

.PHONY: all run clean runtime runtime-cuda

all: $(EXE)

run: $(EXE)
	./$(EXE) $(RUN_ARGS)

clean:
	rm -rf $(BUILD) $(EXE) mandelbrot.png

runtime:
	$(MAKE) -C $(RUNTIME)

runtime-cuda:
	$(MAKE) -C $(RUNTIME) cuda

$(BUILD):
	mkdir -p $(BUILD)

ifeq ($(DEVICE),cuda)
$(BUILD)/dev.ptx: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) --target ptx $< $@ --sm $(SM) -f wide-integers

$(BUILD)/dev_ptx_blob.s: $(BUILD)/dev.ptx | $(BUILD)
	printf '\t.section .rodata\n\t.globl __pas_device_ptx\n__pas_device_ptx:\n\t.incbin "$(BUILD)/dev.ptx"\n\t.byte 0\n' > $@

$(BUILD)/dev_ptx_blob.o: $(BUILD)/dev_ptx_blob.s
	clang -c $< -o $@

$(BUILD)/host.ll: $(HOST_SRC) | $(BUILD)
	$(PAS) $(FEATURES) --device-backend cuda $< $@

$(EXE): $(BUILD)/host.ll $(BUILD)/dev_ptx_blob.o $(PNG_HELPER)
	clang $(BUILD)/host.ll $(BUILD)/dev_ptx_blob.o $(PNG_HELPER) $(RUNTIME_CUDA) \
	      -L$(CUDA_HOME)/lib64/stubs -lcuda -lpng -lm -o $@
else
$(BUILD)/dev.ll: $(DEVICE_UNIT) | $(BUILD)
	$(PAS) $(FEATURES) $< $@

$(BUILD)/host.ll: $(HOST_SRC) | $(BUILD)
	$(PAS) $(FEATURES) $< $@

$(EXE): $(BUILD)/host.ll $(BUILD)/dev.ll $(PNG_HELPER)
	clang $(BUILD)/host.ll $(BUILD)/dev.ll $(PNG_HELPER) $(RUNTIME_CPU) -lpng -lm -o $@
endif
