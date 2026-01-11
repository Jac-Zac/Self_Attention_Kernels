#pragma once
// Tensor allocation, threading utilities, and debug macros.
// Memory alignment utilities are in memory.h.

#include "memory.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Debug Output
#ifdef VERBOSE
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
#else
#define VERBOSE_PRINT(...) ((void)0)
#endif

// CUDA error handling
#ifdef USE_CUDA
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#endif

#if defined(_OPENMP) && !defined(USE_CUDA)
#include <omp.h>
#endif

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

// Aligned tensor container for Q, K, V, out [B, H, S, D_pad] and workspace
struct Tensors {
  float *RESTRICT Q;
  float *RESTRICT K;
  float *RESTRICT V;
  float *RESTRICT out;
  float *RESTRICT workspace;
};

inline int allocate_tensors(struct Tensors *t, size_t qkv_size_floats,
                            size_t workspace_bytes) {
  t->Q = t->K = t->V = t->out = t->workspace = NULL;
  int err = 0;
  err |= (ALIGNED_ALLOC_FLOAT(t->Q, qkv_size_floats) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->K, qkv_size_floats) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->V, qkv_size_floats) != 0);
  err |= (ALIGNED_ALLOC_FLOAT(t->out, qkv_size_floats) != 0);
  size_t workspace_floats = 0;
  if (workspace_bytes != 0) {
    if (workspace_bytes % sizeof(float) != 0) {
      fprintf(stderr, "Error: workspace size must be float-multiple\n");
      return 1;
    }
    workspace_floats = workspace_bytes / sizeof(float);
  }
  if (workspace_floats > 0) {
    err |= (ALIGNED_ALLOC_FLOAT(t->workspace, workspace_floats) != 0);
  }

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
  if (t->workspace) {
    t->workspace = (float *)ASSUME_ALIGNED(t->workspace, ALIGNMENT);
  }
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

inline void init_random_tensors(float *RESTRICT Q, float *RESTRICT K,
                                float *RESTRICT V, float *RESTRICT out,
                                size_t batch, size_t n_heads, size_t seq_len,
                                size_t head_dim_padded, unsigned seed) {
#pragma omp parallel for collapse(2)
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
