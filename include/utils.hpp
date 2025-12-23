#pragma once
#include <stddef.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vector_pragmas.h"

// Round up x to the next multiple of a (a must be power of 2)
static inline size_t round_up_pow2(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

// Shared outputs from attention run
struct Outputs {
  float *Q;
  float *K;
  float *V;
  float *out;
  size_t qkv_size;
  size_t stats_size;
  uint64_t elapsed_ns;
};

inline void free_outputs(struct Outputs *outputs) {
  if (!outputs)
    return;
  free(outputs->Q);
  free(outputs->K);
  free(outputs->V);
  free(outputs->out);
  outputs->Q = outputs->K = outputs->V = outputs->out = NULL;
  outputs->qkv_size = outputs->stats_size = 0;
  outputs->elapsed_ns = 0;
}

#ifdef VERBOSE
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
#else
#define VERBOSE_PRINT(...) ((void)0)
#endif

// RESTRICT macro
#ifndef RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#elif __STDC_VERSION__ >= 199901L
#define RESTRICT restrict
#else
#define RESTRICT
#endif
#endif

// Alignment configuration and helpers
#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

// Aligned allocation for float buffers using posix_memalign
#define ALIGNED_ALLOC_FLOAT(ptr, count)                                        \
  (posix_memalign((void **)&(ptr), ALIGNMENT, sizeof(float) * (count)))

// Compiler alignment assumption hint (generalized)
#define ASSUME_ALIGNED_FLOAT(ptr)                                              \
  (ptr = (float *)ASSUME_ALIGNED((ptr), ALIGNMENT))
