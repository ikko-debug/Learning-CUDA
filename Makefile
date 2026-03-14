CUDA_DEVICE ?= 0
PYTHON ?= python
DTYPE ?= bf16

# Usage:
#   make nf4                 # default bf16
#   make nf4 DTYPE=fp16      # explicit dtype
#   make nf4 fp16            # shorthand dtype selector

NF4_DIR := 03_nf4_dequant/ikko
NF4_CU := $(NF4_DIR)/mainla.cu
NF4_BIN := $(NF4_DIR)/mainla
NF4_VERIFY := $(NF4_DIR)/verify_mae.py

ifneq ($(filter fp16,$(MAKECMDGOALS)),)
DTYPE := fp16
endif
ifneq ($(filter bf16,$(MAKECMDGOALS)),)
DTYPE := bf16
endif

ifneq ($(filter $(DTYPE),bf16 fp16),$(DTYPE))
$(error Invalid DTYPE='$(DTYPE)'. Use bf16 or fp16)
endif

ifeq ($(DTYPE),fp16)
NF4_OUTPUT := $(NF4_DIR)/data/output_fp16.bin
else
NF4_OUTPUT := $(NF4_DIR)/data/output.bin
endif

.PHONY: nf4 bf16 fp16 help

help:
	@echo "Usage:"
	@echo "  make nf4"
	@echo "  make nf4 DTYPE=fp16"
	@echo "  make nf4 fp16"
	@echo "  make nf4 bf16"

# NF4 one-command pipeline: compile + run + verify
nf4:
	@echo "=== [NF4] Compiling $(NF4_CU) ==="
	@mkdir -p .tmp
	TMPDIR=$(CURDIR)/.tmp /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_80 $(NF4_CU) -o $(NF4_BIN)
	@echo "=== [NF4] Running $(NF4_BIN) ==="
	@echo "=== CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE), DTYPE=$(DTYPE) ==="
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ./$(NF4_BIN) $(DTYPE) $(NF4_OUTPUT)
	@echo "=== [NF4] Verifying MAE ==="
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(PYTHON) $(NF4_VERIFY)

bf16 fp16:
	@:
