CUDA_DEVICE ?= 0
PYTHON ?= python

NF4_DIR := 03_nf4_dequant/nf4/ikko
NF4_CU := $(NF4_DIR)/mainla.cu
NF4_BIN := $(NF4_DIR)/mainla
NF4_VERIFY := $(NF4_DIR)/verify_mae.py
NF4_OUTPUT := $(NF4_DIR)/data/output.bin

.PHONY: nf4

# NF4 one-command pipeline: compile + run + verify
nf4:
	@echo "=== [NF4] Compiling $(NF4_CU) ==="
	@mkdir -p .tmp
	TMPDIR=$(CURDIR)/.tmp /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_80 $(NF4_CU) -o $(NF4_BIN)
	@echo "=== [NF4] Running $(NF4_BIN) ==="
	@echo "=== CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ==="
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) ./$(NF4_BIN) bf16 $(NF4_OUTPUT)
	@echo "=== [NF4] Verifying MAE ==="
	CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) $(PYTHON) $(NF4_VERIFY)
