#pragma once
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

// ============================================================================
// Thread count resolution
// ============================================================================

#if defined(_OPENMP) && !defined(USE_CUDA)
#include <omp.h>
#endif

/**
 * Resolve thread count: if requested <= 0, use OMP max or env variable.
 * Also sets omp_set_num_threads() if OpenMP is available.
 */
inline int resolve_thread_count(int requested) {
  int threads = requested;
#ifdef _OPENMP
  if (threads <= 0) {
    threads = omp_get_max_threads();
  }
  if (threads < 1)
    threads = 1;
  omp_set_num_threads(threads);
#else
  if (threads <= 0) {
    const char *env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads && env_threads[0] != '\0') {
      int val = (int)strtol(env_threads, NULL, 10);
      if (val > 0)
        threads = val;
    }
  }
  if (threads < 1)
    threads = 1;
#endif
  return threads;
}

// ============================================================================
// Tensor allocation helpers
// ============================================================================

struct Tensors {
  float *RESTRICT Q;
  float *RESTRICT K;
  float *RESTRICT V;
  float *RESTRICT out;
  float *RESTRICT workspace;
};

/**
 * Allocate aligned Q, K, V, out tensors (each qkv_size floats)
 * and workspace (workspace_size floats). Returns 0 on success.
 */
inline int allocate_tensors(struct Tensors *t, size_t qkv_size,
                            size_t workspace_size) {
  t->Q = t->K = t->V = t->out = t->workspace = NULL;
  int err = 0;
  err |= (ALIGNED_ALLOC_FLOAT(t->Q, qkv_size) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->K, qkv_size) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->V, qkv_size) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->out, qkv_size) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->workspace, workspace_size) != 0);

  if (err) {
    fprintf(stderr, "Error: aligned memory allocation failed\n");
    free(t->Q);
    free(t->K);
    free(t->V);
    free(t->out);
    free(t->workspace);
    t->Q = t->K = t->V = t->out = t->workspace = NULL;
    return 1;
  }

  t->Q = (float *)ASSUME_ALIGNED(t->Q, ALIGNMENT);
  t->K = (float *)ASSUME_ALIGNED(t->K, ALIGNMENT);
  t->V = (float *)ASSUME_ALIGNED(t->V, ALIGNMENT);
  t->out = (float *)ASSUME_ALIGNED(t->out, ALIGNMENT);
  t->workspace = (float *)ASSUME_ALIGNED(t->workspace, ALIGNMENT);
  return 0;
}

inline void free_tensors(struct Tensors *t) {
  if (!t)
    return;
  free(t->Q);
  free(t->K);
  free(t->V);
  free(t->out);
  free(t->workspace);
  t->Q = t->K = t->V = t->out = t->workspace = NULL;
}

/**
 * Initialize Q, K, V, out with random values using NUMA-aware first-touch.
 * Parallelizes over batch x n_heads x seq_len to match kernel access pattern.
 */
inline void init_random_tensors(float *RESTRICT Q, float *RESTRICT K,
                                float *RESTRICT V, float *RESTRICT out,
                                size_t batch, size_t n_heads, size_t seq_len,
                                size_t head_dim_padded, unsigned seed) {
#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < batch; b++) {
    for (size_t h = 0; h < n_heads; h++) {
      for (size_t s = 0; s < seq_len; s++) {
        unsigned int thread_seed = seed + (unsigned int)(b * n_heads + h);
        size_t bh_offset = b * (n_heads * seq_len * head_dim_padded) +
                           h * (seq_len * head_dim_padded);
        size_t base = bh_offset + s * head_dim_padded;

        for (size_t d = 0; d < head_dim_padded; d++) {
          size_t idx = base + d;
          float rand_val = (float)rand_r(&thread_seed) / (float)RAND_MAX;
          Q[idx] = rand_val * 0.1f - 0.05f;
          K[idx] = rand_val * 0.1f - 0.05f;
          V[idx] = rand_val * 0.5f;
          out[idx] = 0.0f;
        }
      }
    }
  }
}
