#include <cuda_runtime.h>
#include <stdio.h>

#include "include/cmhsa_forward.h"
#include "include/io.hpp"
#include "include/parser.hpp"
#include "include/utils.hpp"

#ifndef BACKEND
#define BACKEND "cuda"
#endif
#ifndef VERSION_STR
#define VERSION_STR "v0"
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  RunConfig cfg;
  if (parse_args(argc, argv, &cfg) != 0) {
    return 1;
  }

  printf("backend=%s version=%s\n", BACKEND, VERSION_STR);
  printf("batch=%zu n_heads=%zu seq_len=%zu head_dim=%zu\n", cfg.batch,
         cfg.n_heads, cfg.seq_len, cfg.head_dim);

  // Setup dimensions and compute padded sizes
  AttentionDims dims = {cfg.batch, cfg.n_heads, cfg.seq_len, cfg.head_dim};
  const size_t head_dim_padded = round_up_pow2(cfg.head_dim, VEC_PADDING);
  const size_t qkv_size =
      cfg.batch * cfg.n_heads * cfg.seq_len * head_dim_padded;
  const size_t stats_size = cfg.batch * cfg.n_heads * cfg.seq_len;

  // Allocate CUDA managed memory
  float *Q, *K, *V, *out, *softmax_lse, *softmax_max;
  CUDA_CHECK(cudaMallocManaged(&Q, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&K, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&V, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&out, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&softmax_lse, stats_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&softmax_max, stats_size * sizeof(float)));

  // Initialize with random values (CPU, works on managed memory)
  init_random_tensors(Q, K, V, out, cfg.batch, cfg.n_heads, cfg.seq_len,
                      head_dim_padded, cfg.seed);

  // Initialize softmax buffers
  for (size_t i = 0; i < stats_size; i++) {
    softmax_lse[i] = 0.0f;
    softmax_max[i] = 0.0f;
  }

  VERBOSE_PRINT("Sample Q values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("Q[0][0][0][%zu] = %f\n", d, Q[d]);
  }

  printf("\nRunning attention forward pass...\n");

  // Warm-up runs
  for (int i = 0; i < cfg.warmup; i++) {
    cmhsa_forward_cuda(Q, K, V, out, softmax_lse, softmax_max, dims);
  }

  // Timed iterations using CUDA events
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  float total_ms = 0.0f;
  float checksum = 0.0f;
  for (int i = 0; i < cfg.iters; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    cmhsa_forward_cuda(Q, K, V, out, softmax_lse, softmax_max, dims);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    total_ms += ms;
    if (cfg.head_dim > 0)
      checksum += out[0];
  }

  printf("CUDA attention forward (total): %.3f ms\n", total_ms);
  printf("CUDA attention forward (per-iter): %.6f ms\n", total_ms / cfg.iters);
  VERBOSE_PRINT("Checksum (sum of out[0] over iters): %f\n", checksum);

  VERBOSE_PRINT("\nSample output values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("out[0][0][0][%zu] = %f\n", d, out[d]);
  }

  // Validation mode: write artifacts for Python
  if (cfg.validate) {
    struct Outputs outputs = {Q, K, V, out, qkv_size, stats_size, 0};
    write_validation_artifacts(cfg.validate_dir, &cfg, &outputs);
  }

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(out);
  cudaFree(softmax_lse);
  cudaFree(softmax_max);

  printf("\nCompleted successfully!\n");
  return 0;
}
