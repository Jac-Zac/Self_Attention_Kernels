# Compilers
GCC = g++
NVCC = nvcc

# =================== COMPILATION FLAGS =================== 
# Common optimization flags
OFLAG = -O3
MARCH = -march=native

# Additional flags
# for math
# -lm

# Other flags
OPENMP = -fopenmp

# Version flag (set VERSION to select kernel version, e.g., v0)
VERSION ?= v0

# Debug flag (set DEBUG=1 to enable debug)
DEBUG ?= 0
ifeq ($(DEBUG),1)
DEBUG_FLAGS = -DDEBUG -g
GCC_DEBUG_EXTRA = -fopt-info-all
else
DEBUG_FLAGS =
GCC_DEBUG_EXTRA =
endif

# Combine flags for GCC
GCC_CFLAGS = $(OFLAG) $(MARCH) $(DEBUG_FLAGS) $(GCC_DEBUG_EXTRA)
GCC_LDFLAGS =

# Combine flags for NVCC
NVCC_CFLAGS = $(OFLAG) $(DEBUG_FLAGS) -DUSE_CUDA
NVCC_LDFLAGS =

# =================== COMPILATION OPTIONS =================== 

# =================== TARGETS =================== 
single:
	$(GCC) $(GCC_CFLAGS) $(GCC_LDFLAGS) -o cmhsa_single_$(VERSION) main.cpp kernels/single_thread/$(VERSION).cpp

multi:
	$(GCC) $(GCC_CFLAGS) $(GCC_LDFLAGS) $(OPENMP) -o cmhsa_multi_$(VERSION) main.cpp kernels/multi_core_cpu/$(VERSION).cpp

cuda:
	$(NVCC) $(NVCC_CFLAGS) $(NVCC_LDFLAGS) -o cmhsa_cuda_$(VERSION) main_cuda.cpp kernels/cuda/$(VERSION).cu

# =================== TEST TARGETS =================== 
test_single:
	$(GCC) $(GCC_CFLAGS) $(GCC_LDFLAGS) -o test_single tests/test.cpp kernels/single_thread/$(VERSION).cpp && ./test_single && echo "check: PASS" || (echo "check: FAIL" && exit 1)

test_multi:
	$(GCC) $(GCC_CFLAGS) $(GCC_LDFLAGS) $(OPENMP) -o test_multi tests/test.cpp kernels/multi_core_cpu/$(VERSION).cpp && ./test_multi && echo "check: PASS" || (echo "check: FAIL" && exit 1)

test_cuda:
	$(NVCC) $(NVCC_CFLAGS) $(NVCC_LDFLAGS) -o test_cuda tests/test_cuda.cpp kernels/cuda/$(VERSION).cu && ./test_cuda && echo "check: PASS" || (echo "check: FAIL" && exit 1)

# =================== ADDITION TARGETS ===================

test_all:
	@for f in kernels/single_thread/*.cpp; do \
		v=$$(basename $$f .cpp); \
		echo "Testing single_thread $$v"; \
		$(MAKE) VERSION=$$v test_single; \
	done
	@for f in kernels/multi_core_cpu/*.cpp; do \
		v=$$(basename $$f .cpp); \
		echo "Testing multi_core_cpu $$v"; \
		$(MAKE) VERSION=$$v test_multi; \
	done

  # TODO:
  # Do the same thing also for cuda when implemented
	make clean

test: test_all

# Default target (single)
all: single

# All versions target (build and test all versions)
VERSIONS = v0 v1 v2 final
all_versions:
	@for v in $(VERSIONS); do \
		if [ -d kernels/single_thread/$$v ] && [ -d kernels/multi_core_cpu/$$v ] && [ -d kernels/cuda/$$v ]; then \
			echo "Building and testing version $$v"; \
			$(MAKE) VERSION=$$v single multi cuda test_single test_multi; \
		fi; \
	done

# Clean target
clean:
	rm -rf cmhsa_* test_single test_multi test_cuda 
	rm -rf *.dSYM

.PHONY: all single multi cuda tests test_single test_multi test_cuda clean all_versions
