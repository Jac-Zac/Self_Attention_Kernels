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
WARN_GCC = -Wall -Wextra -Wpedantic
# Clang-specific recommended warnings
WARN_CLANG = -Wall -Wextra -Wpedantic -Wconversion

VERSION ?= v0
DEBUG ?= 0

ifeq ($(DEBUG),1)
DEBUG_FLAGS = -DDEBUG -g
WARN_GCC += -fopt-info-all
# WARN_CLANG += -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*
WARN_CLANG += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
else
DEBUG_FLAGS =
endif

# Choose warning set
ifeq ($(CXX),clang++)
  CXX_WARN = $(WARN_CLANG)
else
	# GCC alternative
  CXX_WARN = $(WARN_GCC)
endif

# C++/NVCC combined flags
CXXFLAGS = $(CFLAGS) $(CXX_WARN) $(DEBUG_FLAGS)
NVCC_CFLAGS = -O3 $(DEBUG_FLAGS) -DUSE_CUDA

# Targets (unified executable name 'cmhsa.out')
EXEC ?= cmhsa.out

single:
	$(CXX) $(CXXFLAGS) -DBACKEND=\"single\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/single_thread/$(VERSION).cpp

multi:
	$(CXX) $(CXXFLAGS) $(OPENMP) -DBACKEND=\"multi\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/multi_core_cpu/$(VERSION).cpp

cuda:
	$(NVCC) $(NVCC_CFLAGS) -DBACKEND=\"cuda\" -DVERSION_STR=\"$(VERSION)\" -o $(EXEC) main.cpp kernels/cuda/$(VERSION).cu

# Unified test target: builds 'cmhsa_test' and runs one at a time
# CUDA tests are not implemented yet
test:
	@for f in kernels/single_thread/*.cpp; do \
		v=$$(basename $$f .cpp); \
		echo "Testing single_thread $$v"; \
		$(CXX) $(CXXFLAGS) -o cmhsa_test tests/test.cpp kernels/single_thread/$$v.cpp && ./cmhsa_test || exit 1; \
	done
	@for f in kernels/multi_core_cpu/*.cpp; do \
		v=$$(basename $$f .cpp); \
		echo "Testing multi_core_cpu $$v"; \
		$(CXX) $(CXXFLAGS) $(OPENMP) -o cmhsa_test tests/test.cpp kernels/multi_core_cpu/$$v.cpp && ./cmhsa_test || exit 1; \
	done
	@echo "CUDA tests not implemented yet"

	# Clean everything after
	make clean


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
	@echo "  test     Run tests for single+multi across all versions"
	@echo "  clean    Remove build/test artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  VERSION  Kernel version to use (e.g., v0, v1). Default v0"
	@echo "  DEBUG=1  Enable debug build and -DDEBUG"
	@echo "  CXX=clang++  Use clang++ instead of g++"

# Clean
clean:
	rm -rf cmhsa*
	rm -rf *.dSYM

.PHONY: all single multi cuda test clean help
