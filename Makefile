# NOTE: If user did NOT explicitly pass CXX=...
# set our own default and override the environment
ifneq ($(origin CXX),command line)
	# Compilers
  override CXX := g++
endif

NVCC = nvcc

CFLAGS = -O3 -march=native -ffast-math
# CFLAGS = -O3 -march=native -fassociative-math -fno-trapping-math -ffinite-math-only -fno-signed-zeros
# -mprefer-vector-width=512
# CFLAGS = -O3 -march=native -fassociative-math -fno-trapping-math -ffinite-math-only -fno-signed-zeros
# -fno-tree-loop-vectorize
# -flto
CFLAGS += -flto
# CFLAGS += -mveclibabi=svml
OPENMP = -fopenmp-simd -fopenmp

# Auto-select warning flags based on compiler
# Default warnings (GCC)
WARN_GCC = -std=c++20 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas
# Clang-specific recommended warnings
WARN_CLANG = -Wall -Wextra -Wpedantic -Wconversion

VERSION ?= v0
DEBUG ?= 0
VERBOSE ?= 0

ifeq ($(DEBUG),1)
DEBUG_FLAGS = -DDEBUG -g
WARN_GCC += -fopt-info-all
# WARN_CLANG += -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*
WARN_CLANG += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
else
DEBUG_FLAGS =
endif

ifeq ($(VERBOSE),1)
VERBOSE_FLAGS = -DVERBOSE
else
VERBOSE_FLAGS =
endif

# Choose warning set
ifeq ($(CXX),clang++)
  CXX_WARN = $(WARN_CLANG)
else
	# GCC alternative
  CXX_WARN = $(WARN_GCC)
endif

# C++/NVCC combined flags
CXXFLAGS = $(CFLAGS) $(CXX_WARN) $(DEBUG_FLAGS) $(VERBOSE_FLAGS)
NVCC_CFLAGS = -O3 $(DEBUG_FLAGS) $(VERBOSE_FLAGS) -DUSE_CUDA

# Discovered kernel versions
# SINGLE_VERSIONS := $(basename $(notdir $(wildcard kernels/single_thread/v*.cpp)))
# NOTE: For now let's exclude v0 since it takes a lot of time to compute
SINGLE_VERSIONS := $(filter-out v0,$(basename $(notdir $(wildcard kernels/single_thread/v*.cpp))))
# SINGLE_VERSIONS := $(basename $(notdir $(wildcard kernels/single_thread/v*.cpp)))
MULTI_VERSIONS := $(basename $(notdir $(wildcard kernels/multi_thread/v*.cpp)))
CUDA_VERSIONS := $(basename $(notdir $(wildcard kernels/cuda/v*.cu)))

# Targets (unified executable name 'cmhsa.out')
EXEC ?= cmhsa.out

single:
	$(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/single_thread/$(VERSION).cpp

multi:
	$(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"multi\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/multi_thread/$(VERSION).cpp

cuda:
	$(NVCC) $(NVCC_CFLAGS) -DBACKEND=\"cuda\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/cuda/$(VERSION).cu

# asm-single:
# 	$(CXX) $(CXXFLAGS) -fno-lto -S -fverbose-asm -masm=intel -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" main.cpp kernels/single_thread/$(VERSION).cpp

# Test target: build main (single-thread backend) and validate against PyTorch
# Runs validation for all discovered single-thread versions

# Benchmark: build and run single-thread binaries for all discovered versions
# Uses SINGLE_VERSIONS for discovery; fixed benchmark sizes below
# Resonable numbers to run it with
BENCH_BACKEND ?= single
BENCH_BACKEND := $(strip $(BENCH_BACKEND))
ifeq ($(BENCH_BACKEND),)
  BENCH_BACKEND := single
endif

BENCH_THREADS ?= 1
BENCH_THREADS := $(strip $(BENCH_THREADS))
ifeq ($(BENCH_THREADS),)
  BENCH_THREADS := 8
endif

# Use srun for SLURM environments (ensures proper CPU binding for child processes)
USE_SRUN ?= 0

# Select versions and source dir based on backend
ifeq ($(BENCH_BACKEND),single)
  BENCH_VERSIONS := $(SINGLE_VERSIONS)
  BENCH_SRC_DIR := kernels/single_thread
else ifeq ($(BENCH_BACKEND),multi)
  BENCH_VERSIONS := $(MULTI_VERSIONS)
  BENCH_SRC_DIR := kernels/multi_thread
else
  BENCH_VERSIONS := $(SINGLE_VERSIONS)
  BENCH_SRC_DIR := kernels/single_thread
endif

# Optional output file for CSV results
# @batch=8; heads=32; seqlen=4096; headdim=128; seed=1337; warmup=5; iters=20; threads=$(BENCH_THREADS); \
BENCH_OUTPUT_FILE ?=

benchmark:
	@batch=2; heads=4; seqlen=4096; headdim=128; seed=1337; warmup=5; iters=20; threads=$(BENCH_THREADS); \
	bins=""; \
	for ver in $(BENCH_VERSIONS); do \
	  bin=cmhsa_$$ver.out; \
	  src=$(BENCH_SRC_DIR)/$$ver.cpp; \
	  echo "Building $$bin (backend=$(BENCH_BACKEND), version=$$ver)"; \
	  $(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"$(BENCH_BACKEND)\" -DVERSION_STR=\"$$ver\" -o $$bin main.cpp $$src; \
	  bins="$$bins ./$$bin"; \
	done; \
	OMP_NUM_THREADS=$$threads MKL_NUM_THREADS=$$threads OPENBLAS_NUM_THREADS=$$threads NUMEXPR_NUM_THREADS=$$threads \
	  python3 python_src/benchmark.py \
	    --bins $$bins \
	    --batch $$batch --n_heads $$heads --seq_len $$seqlen --head_dim $$headdim \
	    --seed $$seed --warmup $$warmup --iters $$iters --threads $$threads \
	    $(if $(filter 1,$(USE_SRUN)),--use-srun) \
	    $(if $(BENCH_OUTPUT_FILE),--output $(BENCH_OUTPUT_FILE))
	@$(MAKE) clean


test:
	@set -e; \
	for backend in single multi; do \
	  case $$backend in \
	    single) versions="$(SINGLE_VERSIONS)" ;; \
	    multi)  versions="$(MULTI_VERSIONS)" ;; \
	  esac; \
	  for ver in $$versions; do \
	    echo "[test] $$backend $$ver"; \
	    $(CXX) $(CXXFLAGS) $(OPENMP) \
	      -DBACKEND=\"$$backend\" -DVERSION_STR=\"$$ver\" \
	      -o $(EXEC) main.cpp kernels/$$backend\_thread/$$ver.cpp; \
	    python3 python_src/tests/validate_with_torch.py --bin ./$(EXEC) \
	      --batch 4 --n_heads 8 --seq_len 16 --head_dim 32 --seed 1337 \
	      $(if $(filter 1,$(USE_SRUN)),--use-srun); \
	  done; \
	done
# cuda)   versions="$(CUDA_VERSIONS)" ;; \


# Convenience
all: single

# Help
help:
	@echo "Usage: make <target> [VARIABLES]"
	@echo ""
	@echo "Targets:"
	@echo "  single     Build single-thread backend (VERSION?=v0)"
	@echo "  multi      Build multi-core backend (OpenMP)"
	@echo "  cuda       Build CUDA backend (kernel stubbed)"
	@echo "  test       Validate all single-thread versions (float32)"
	@echo "  benchmark  Benchmark all kernel versions against PyTorch (runs PyTorch once)"
	@echo "  clean      Remove build/test artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  VERSION        Kernel version to use (e.g., v0, v1). Default v0"
	@echo "  DEBUG=1        Enable debug build and -DDEBUG"
	@echo "  VERBOSE=1      Enable verbose prints (-DVERBOSE)"
	@echo "  CXX=clang++    Use clang++ instead of g++"
	@echo "  BENCH_BACKEND  Backend for benchmark: single (default) or multi"
	@echo "  BENCH_THREADS  Threads used for both C++ and PyTorch benchmark"
	@echo "  USE_SRUN=1     Use srun for SLURM environments (proper CPU binding)"
	@echo "  BENCH_OUTPUT_FILE  Save benchmark results to CSV file"
	@echo ""
	@echo "Benchmark outputs a text summary by default. Use BENCH_OUTPUT_FILE to save CSV."

# Clean
clean:
	rm -rf cmhsa*
	rm -rf *.dSYM

.PHONY: all single multi cuda test benchmark bench clean help
