# NOTE: If user did NOT explicitly pass CXX=...
# set our own default and override the environment
ifneq ($(origin CXX),command line)
	# Compilers
  override CXX := g++
endif

NVCC = nvcc

CFLAGS = -O3
OPENMP = -fopenmp

# Auto-select warning flags based on compiler
# Default warnings (GCC)
WARN_GCC = -std=c++20 -Wall -Wextra -Wpedantic
# WARN_GCC = -std=c++20 -Wall -Wextra -Wpedantic -flto
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

# Targets (unified executable name 'cmhsa.out')
EXEC ?= cmhsa.out

# asm-single:
# 	$(CXX) $(CXXFLAGS) -fno-lto -S -fverbose-asm -masm=intel -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" main.cpp kernels/single_thread/$(VERSION).cpp

single:
	$(CXX) $(CXXFLAGS) -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/single_thread/$(VERSION).cpp

multi:
	$(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"multi\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/multi_core_cpu/$(VERSION).cpp

cuda:
	$(NVCC) $(NVCC_CFLAGS) -DBACKEND=\"cuda\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/cuda/$(VERSION).cu

# Test target: build main (single-thread backend) and validate against PyTorch
# Float32-only; main writes artifacts to python_test/ and Python cleans them afterward
#
# TODO: In the future this should test all of the available kernels instaed
test:
	$(CXX) $(CXXFLAGS) -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/single_thread/$(VERSION).cpp
	python python_tests/validate_with_torch.py --bin ./$(EXEC) --batch 1 --n_heads 1 --seq_len 32 --head_dim 64 --seed 1337

# Benchmark: build and run single-thread binaries for versions
# Configurable benchmark parameters
BENCH_VERSIONS ?= v0 v1
BENCH_BACKEND ?= single
BENCH_BATCH ?= 4
BENCH_HEADS ?= 8
BENCH_SEQ ?= 256
BENCH_DIM ?= 128
BENCH_SEED ?= 1337

benchmark:
	@for ver in $(BENCH_VERSIONS); do \
	  bin=cmhsa_$$ver.out; \
	  src=kernels/single_thread/$$ver.cpp; \
	  echo "Building $$bin (backend=$(BENCH_BACKEND), version=$$ver)"; \
	  $(CXX) $(CXXFLAGS) -DBACKEND=\"$(BENCH_BACKEND)\" -DVERSION_STR=\"$$ver\" -o $$bin main.cpp $$src; \
	  printf "$$ver: "; \
	  ./$$bin --batch $(BENCH_BATCH) --n_heads $(BENCH_HEADS) --seq_len $(BENCH_SEQ) --head_dim $(BENCH_DIM) --seed $(BENCH_SEED) | grep "CPU attention forward" || true; \
	done

# Convenience
all: single

# Help
help:
	@echo "Usage: make <target> [VARIABLES]"
	@echo ""
	@echo "Targets:"
	@echo "  single   Build single-thread backend (VERSION?=v0)"
	@echo "  multi    Build multi-core backend (OpenMP)"
	@echo "  cuda     Build CUDA backend (kernel stubbed)"
	@echo "  test     Build test and validate against PyTorch (float32)"
	@echo "  benchmark Compare single-thread v0 vs v1 timings"
	@echo "  clean    Remove build/test artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  VERSION   Kernel version to use (e.g., v0, v1). Default v0"
	@echo "  DEBUG=1   Enable debug build and -DDEBUG"
	@echo "  VERBOSE=1 Enable verbose prints (-DVERBOSE)"
	@echo "  CXX=clang++  Use clang++ instead of g++"

# Clean
clean:
	rm -rf cmhsa*
	rm -rf *.dSYM

.PHONY: all single multi cuda test benchmark clean help
