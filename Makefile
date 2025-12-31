# Compiler defaults (override with CXX=clang++ on command line)
ifneq ($(origin CXX),command line)
  override CXX := g++
endif
NVCC := nvcc

# Build options
VERSION ?= v0
DEBUG ?= 0
VERBOSE ?= 0
USE_SRUN ?= 0

# Base flags
CFLAGS := -O3 -march=native -ffast-math -flto
# CFLAGS += -mprefer-vector-width=512
OPENMP := -fopenmp-simd -fopenmp

# Warning flags per compiler
WARN_GCC   := -std=c++20 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas
WARN_CLANG := -Wall -Wextra -Wpedantic -Wconversion

# Debug/verbose flags
ifeq ($(DEBUG),1)
  DEBUG_FLAGS := -DDEBUG -g
  WARN_GCC   += -fopt-info-all
  WARN_CLANG += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
endif
VERBOSE_FLAGS := $(if $(filter 1,$(VERBOSE)),-DVERBOSE)

# Select warning set based on compiler
CXX_WARN := $(if $(filter clang++,$(CXX)),$(WARN_CLANG),$(WARN_GCC))

# Combined flags
CXXFLAGS  := $(CFLAGS) $(CXX_WARN) $(DEBUG_FLAGS) $(VERBOSE_FLAGS)
# CUDA_ARCH ?= sm_70
CUDA_ARCH ?= sm_86
NVCC_FLAGS := -O3 $(DEBUG_FLAGS) $(VERBOSE_FLAGS) -arch=$(CUDA_ARCH) -DUSE_CUDA

# Discovered kernel versions
SINGLE_VERSIONS := $(basename $(notdir $(wildcard kernels/single_thread/v*.cpp)))
MULTI_VERSIONS  := $(basename $(notdir $(wildcard kernels/multi_thread/v*.cpp)))
CUDA_VERSIONS   := $(filter-out v0,$(basename $(notdir $(wildcard kernels/cuda/v*.cu))))

# Output executable
EXEC ?= cmhsa.out

# =============================================================================
# Build targets
# =============================================================================

single:
	$(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" \
		-o $(EXEC) main.cpp kernels/single_thread/$(VERSION).cpp

multi:
	$(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"multi\" -DVERSION_STR=\"$(VERSION)\" \
		-o $(EXEC) main.cpp kernels/multi_thread/$(VERSION).cpp

cuda:
	$(NVCC) $(NVCC_FLAGS) -DBACKEND=\"cuda\" -DVERSION_STR=\"$(VERSION)\" \
		-o $(EXEC) main.cu kernels/cuda/$(VERSION).cu

# =============================================================================
# Benchmark
# =============================================================================

BENCH_BACKEND ?= single
BENCH_THREADS ?= 1
BENCH_OUTPUT_FILE ?=

# Benchmark parameters (overridable)
BENCH_BATCH ?= 2
BENCH_HEADS ?= 4
# BENCH_SEQLEN ?= 2048
BENCH_SEQLEN ?= 512
BENCH_HEADDIM ?= 128
BENCH_SEED ?= 1337
BENCH_WARMUP ?= 5
BENCH_ITERS ?= 20

# Common benchmark arguments
BENCH_COMMON_ARGS = --batch $(BENCH_BATCH) --n_heads $(BENCH_HEADS) \
    --seq_len $(BENCH_SEQLEN) --head_dim $(BENCH_HEADDIM) --seed $(BENCH_SEED) \
    --warmup $(BENCH_WARMUP) --iters $(BENCH_ITERS)

# Map backend to versions and source directory
ifeq ($(BENCH_BACKEND),cuda)
  BENCH_VERSIONS = $(CUDA_VERSIONS)
  BENCH_SRC_DIR  = kernels/cuda
  BENCH_COMPILER = $(NVCC) $(NVCC_FLAGS)
else
  BENCH_VERSIONS = $(if $(filter multi,$(BENCH_BACKEND)),$(MULTI_VERSIONS),$(SINGLE_VERSIONS))
  BENCH_SRC_DIR  = kernels/$(if $(filter multi,$(BENCH_BACKEND)),multi,single)_thread
  BENCH_COMPILER = $(CXX) $(CXXFLAGS) $(OPENMP)
endif
BENCH_BACKEND_ARG = $(if $(filter cuda,$(BENCH_BACKEND)),--backend cuda,$(if $(filter multi,$(BENCH_BACKEND)),--backend multi,--backend single))
BENCH_THREADS_ARG = $(if $(filter cuda,$(BENCH_BACKEND)),,--threads $(BENCH_THREADS))

benchmark:
	@bins=""; \
	for ver in $(BENCH_VERSIONS); do \
	  bin=cmhsa_$$ver.out; \
	  echo "Building $$bin (backend=$(BENCH_BACKEND), version=$$ver)"; \
	  $(BENCH_COMPILER) -DBACKEND=\"$(BENCH_BACKEND)\" -DVERSION_STR=\"$$ver\" \
	    -o $$bin $(if $(filter cuda,$(BENCH_BACKEND)),main.cu,main.cpp) \
	    $(BENCH_SRC_DIR)/$$ver$(if $(filter cuda,$(BENCH_BACKEND)),.cu,.cpp); \
	  bins="$$bins ./$$bin"; \
	done; \
	$(if $(filter 1,$(USE_SRUN)), \
	  python3 python_src/benchmark.py --bins $$bins $(BENCH_COMMON_ARGS) $(BENCH_BACKEND_ARG) $(BENCH_THREADS_ARG) --use-srun $(if $(BENCH_OUTPUT_FILE),--output $(BENCH_OUTPUT_FILE)), \
	  OMP_NUM_THREADS=$(BENCH_THREADS) MKL_NUM_THREADS=$(BENCH_THREADS) \
	  OPENBLAS_NUM_THREADS=$(BENCH_THREADS) NUMEXPR_NUM_THREADS=$(BENCH_THREADS) \
	  python3 python_src/benchmark.py --bins $$bins $(BENCH_COMMON_ARGS) $(BENCH_BACKEND_ARG) $(BENCH_THREADS_ARG) $(if $(BENCH_OUTPUT_FILE),--output $(BENCH_OUTPUT_FILE)))
	@$(MAKE) clean

# GPU detection (can be overridden with: make test HAS_GPU=false)
HAS_GPU ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 | grep -q . && echo true || echo false)

# =============================================================================
# Test
# =============================================================================

check-gpu:
	@if [ "$(HAS_GPU)" = "true" ]; then \
		echo "✓ GPU detected: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"; \
	else \
		echo "✗ No GPU detected - CUDA tests will be skipped"; \
	fi

test:
	@set -e; \
	echo "[check-gpu] Checking GPU availability..."; \
	if [ "$(HAS_GPU)" = "true" ]; then \
		echo "✓ GPU detected: $$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"; \
	else \
		echo "✗ No GPU detected - CUDA tests will be skipped"; \
	fi; \
	for backend in single multi; do \
	  case $$backend in \
	    single) versions="$(SINGLE_VERSIONS)" ;; \
	    multi)  versions="$(MULTI_VERSIONS)" ;; \
	  esac; \
	  for ver in $$versions; do \
	    echo "[test] $$backend $$ver"; \
	    $(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"$$backend\" -DVERSION_STR=\"$$ver\" \
	      -o $(EXEC) main.cpp kernels/$${backend}_thread/$$ver.cpp; \
	    python3 python_src/tests/validate_with_torch.py --bin ./$(EXEC) \
	      --batch 4 --n_heads 8 --seq_len 16 --head_dim 32 --seed 1337 \
	      $(if $(filter 1,$(USE_SRUN)),--use-srun); \
	  done; \
	done; \
	echo ""; \
	if [ "$(HAS_GPU)" = "true" ]; then \
		for ver in $(CUDA_VERSIONS); do \
			echo "[test] cuda $$ver"; \
			$(NVCC) $(NVCC_FLAGS) -DBACKEND=\"cuda\" -DVERSION_STR=\"$$ver\" \
				-o $(EXEC) main.cu kernels/cuda/$$ver.cu; \
			python3 python_src/tests/validate_with_torch.py --bin ./$(EXEC) \
				--batch 4 --n_heads 8 --seq_len 16 --head_dim 32 --seed 1337 \
				$(if $(filter 1,$(USE_SRUN)),--use-srun); \
		done; \
	else \
		echo "[skip] cuda tests - no GPU detected"; \
	fi

# =============================================================================
# Utilities
# =============================================================================

all: single

clean:
	rm -rf cmhsa* *.dSYM

help:
	@echo "Usage: make <target> [VARIABLES]"
	@echo ""
	@echo "Targets:"
	@echo "  single     Build single-thread backend (VERSION?=v0)"
	@echo "  multi      Build multi-core backend (OpenMP)"
	@echo "  cuda       Build CUDA backend"
	@echo "  test       Validate all kernel versions against PyTorch (skips CUDA if no GPU)"
	@echo "  benchmark  Benchmark all kernel versions"
	@echo "  clean      Remove build artifacts"
	@echo ""
	@echo "Build Variables:"
	@echo "  VERSION=v1         Kernel version (default: v0)"
	@echo "  DEBUG=1            Enable debug build"
	@echo "  VERBOSE=1          Enable verbose output"
	@echo "  CXX=clang++        Use clang instead of gcc"
	@echo "  CUDA_ARCH=sm_70    CUDA architecture (default: sm_70, options: sm_70, sm_80, sm_86, sm_89)"
	@echo "  USE_SRUN=1         Use srun for SLURM environments"
	@echo ""
	@echo "Benchmark Variables:"
	@echo "  BENCH_BACKEND      Backend: single (default), multi, or cuda"
	@echo "  BENCH_THREADS      Thread count (default: 1)"
	@echo "  BENCH_BATCH        Batch size (default: 2)"
	@echo "  BENCH_HEADS        Number of heads (default: 4)"
	@echo "  BENCH_SEQLEN       Sequence length (default: 2048)"
	@echo "  BENCH_HEADDIM      Head dimension (default: 128)"
	@echo "  BENCH_SEED         Random seed (default: 1337)"
	@echo "  BENCH_WARMUP       Warmup iterations (default: 5)"
	@echo "  BENCH_ITERS        Benchmark iterations (default: 20)"
	@echo "  BENCH_OUTPUT_FILE  Save results to CSV"

.PHONY: all single multi cuda check-gpu test benchmark clean help
