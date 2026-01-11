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

# Debug/verbose flags (must be defined before use)
ifeq ($(DEBUG),1)
  DEBUG_FLAGS := -DDEBUG -g
  DEBUG_GCC   := -fopt-info-all
  DEBUG_CLANG := -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
  DEBUG_NVCC  := -lineinfo
endif
VERBOSE_FLAGS := $(if $(filter 1,$(VERBOSE)),-DVERBOSE)

# Base flags
CFLAGS := -O3 -march=native -ffast-math -flto
# CFLAGS += -mprefer-vector-width=512

OPENMP := -fopenmp-simd -fopenmp

# Warning flags per compiler
WARN_GCC   := -std=c++20 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas $(DEBUG_GCC)
WARN_CLANG := -Wall -Wextra -Wpedantic -Wconversion $(DEBUG_CLANG)

# Select warning set based on compiler
CXX_WARN := $(if $(filter clang++,$(CXX)),$(WARN_CLANG),$(WARN_GCC))

# Combined flags
CXXFLAGS := $(CFLAGS) $(CXX_WARN) $(DEBUG_FLAGS) $(VERBOSE_FLAGS)

# CUDA_ARCH ?= sm_70
CUDA_ARCH ?= sm_86
NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -DUSE_CUDA --use_fast_math $(DEBUG_FLAGS) $(DEBUG_NVCC) $(VERBOSE_FLAGS)

# Discovered kernel versions
SINGLE_VERSIONS := $(basename $(notdir $(wildcard kernels/single_thread/v*.cpp)))
MULTI_VERSIONS  := $(basename $(notdir $(wildcard kernels/multi_thread/v*.cpp)))
CUDA_VERSIONS   := $(basename $(notdir $(wildcard kernels/cuda/v*.cu)))

# Output executable
EXEC ?= cmhsa.out

# =============================================================================
# Build targets
# =============================================================================

# Build recipe: $(1)=backend, $(2)=compiler+flags, $(3)=main, $(4)=src dir, $(5)=ext
define build_kernel
	$(2) -DBACKEND=\"$(1)\" -DVERSION_STR=\"$(VERSION)\" \
		-o $(EXEC) $(3) $(4)/$(VERSION)$(5)
endef

single:
	$(call build_kernel,single,$(CXX) $(CXXFLAGS) $(OPENMP),main.cpp,kernels/single_thread,.cpp)

multi:
	$(call build_kernel,multi,$(CXX) $(CXXFLAGS) $(OPENMP),main.cpp,kernels/multi_thread,.cpp)

cuda:
	$(NVCC) $(NVCC_FLAGS) -DBACKEND=\"cuda\" -DVERSION_STR=\"$(VERSION)\" \
		-o $(EXEC) main.cu kernels/cuda/$(VERSION).cu

# =============================================================================
# Test
# =============================================================================

test:
	@set -e; \
	for ver in $(SINGLE_VERSIONS); do \
	  echo "[test] single $$ver"; \
	  $(MAKE) single VERSION=$$ver; \
	  uv run pytest -v; \
	done; \
	for ver in $(MULTI_VERSIONS); do \
	  echo "[test] multi $$ver"; \
	  $(MAKE) multi VERSION=$$ver; \
	  uv run pytest -v; \
	done; \
	if command -v nvcc >/dev/null 2>&1; then \
	  for ver in $(CUDA_VERSIONS); do \
	    echo "[test] cuda $$ver"; \
	    $(MAKE) cuda VERSION=$$ver; \
	    uv run pytest -v; \
	  done; \
	else \
	  echo "[skip] cuda tests - nvcc not found"; \
	fi

# =============================================================================
# Benchmark
# =============================================================================

BENCH_THREADS ?= 1
BENCH_OUTPUT_FILE ?=

# Benchmark parameters (overridable)
BENCH_BATCH   ?= 2
BENCH_HEADS   ?= 4
BENCH_SEQLEN  ?= 2048
BENCH_HEADDIM ?= 128
BENCH_SEED    ?= 1337
BENCH_WARMUP  ?= 5
BENCH_ITERS   ?= 20

# Common benchmark arguments
BENCH_COMMON_ARGS := --batch $(BENCH_BATCH) --n_heads $(BENCH_HEADS) \
    --seq_len $(BENCH_SEQLEN) --head_dim $(BENCH_HEADDIM) --seed $(BENCH_SEED) \
    --warmup $(BENCH_WARMUP) --iters $(BENCH_ITERS)

# Thread environment for CPU benchmarks
THREAD_ENV := OMP_NUM_THREADS=$(BENCH_THREADS) MKL_NUM_THREADS=$(BENCH_THREADS) \
    OPENBLAS_NUM_THREADS=$(BENCH_THREADS) NUMEXPR_NUM_THREADS=$(BENCH_THREADS)

# SRUN prefix for cluster environments
SRUN_PREFIX := $(if $(filter 1,$(USE_SRUN)),srun)

# Benchmark recipe: $(1)=backend, $(2)=versions, $(3)=compiler+flags, $(4)=main, $(5)=src dir, $(6)=ext, $(7)=device
define run_benchmark
	@for ver in $(2); do \
	  echo "Building cmhsa_$$ver.out ($(1)/$$ver)"; \
	  $(3) -DBACKEND=\"$(1)\" -DVERSION_STR=\"$$ver\" \
	    -o cmhsa_$$ver.out $(4) $(5)/$$ver$(6); \
	done
	$(THREAD_ENV) $(SRUN_PREFIX) python3 python_src/benchmark.py \
	  --bins $(addprefix ./cmhsa_,$(addsuffix .out,$(2))) \
	  $(BENCH_COMMON_ARGS) --backend $(1) --device $(7) \
	  $(if $(filter-out cuda,$(1)),--threads $(BENCH_THREADS)) \
	  $(if $(BENCH_OUTPUT_FILE),--output $(BENCH_OUTPUT_FILE))
	@$(MAKE) clean
endef

benchmark-single:
	$(call run_benchmark,single,$(SINGLE_VERSIONS),$(CXX) $(CXXFLAGS) $(OPENMP),main.cpp,kernels/single_thread,.cpp,cpu)

benchmark-multi:
	$(call run_benchmark,multi,$(MULTI_VERSIONS),$(CXX) $(CXXFLAGS) $(OPENMP),main.cpp,kernels/multi_thread,.cpp,cpu)

benchmark-cuda:
	$(call run_benchmark,cuda,$(CUDA_VERSIONS),$(NVCC) $(NVCC_FLAGS),main.cu,kernels/cuda,.cu,cuda)

# Default benchmark alias
benchmark: benchmark-single

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
	@echo "  single           Build single-thread backend (VERSION?=v0)"
	@echo "  multi            Build multi-core backend (OpenMP)"
	@echo "  cuda             Build CUDA backend"
	@echo "  test             Build and test all kernel versions"
	@echo "  benchmark-single Benchmark all single-thread kernel versions"
	@echo "  benchmark-multi  Benchmark all multi-thread kernel versions"
	@echo "  benchmark-cuda   Benchmark all CUDA kernel versions"
	@echo "  clean            Remove build artifacts"
	@echo ""
	@echo "Build Variables:"
	@echo "  VERSION=v1       Kernel version (default: v0)"
	@echo "  DEBUG=1          Enable debug build"
	@echo "  VERBOSE=1        Enable verbose output"
	@echo "  CXX=clang++      Use clang instead of gcc"
	@echo "  CUDA_ARCH=sm_70  CUDA architecture (default: sm_86)"
	@echo "  USE_SRUN=1       Use srun for SLURM environments"
	@echo ""
	@echo "Benchmark Variables:"
	@echo "  BENCH_THREADS      Thread count for multi-thread (default: 1)"
	@echo "  BENCH_BATCH        Batch size (default: 2)"
	@echo "  BENCH_HEADS        Number of heads (default: 4)"
	@echo "  BENCH_SEQLEN       Sequence length (default: 1024)"
	@echo "  BENCH_HEADDIM      Head dimension (default: 128)"
	@echo "  BENCH_SEED         Random seed (default: 1337)"
	@echo "  BENCH_WARMUP       Warmup iterations (default: 5)"
	@echo "  BENCH_ITERS        Benchmark iterations (default: 20)"
	@echo "  BENCH_OUTPUT_FILE  Save results to CSV"

.PHONY: all single multi cuda test benchmark benchmark-single benchmark-multi benchmark-cuda clean help
