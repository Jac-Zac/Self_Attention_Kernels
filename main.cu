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

int main(int argc, char *argv[]) {
  RunConfig cfg;
  if (parse_args(argc, argv, &cfg) != 0) {
    return 1;
  }

  printf("backend=%s version=%s\n", BACKEND, VERSION_STR);
  printf("batch=%zu n_heads=%zu seq_len=%zu head_dim=%zu\n", cfg.batch,
         cfg.n_heads, cfg.seq_len, cfg.head_dim);

  // Display GPU device information
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaSuccess && device_count > 0) {
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) {
      printf("GPU Device: %s\n", prop.name);
#ifdef VERBOSE
      printf("GPU Memory: %.2f GB\n",
             (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
      printf("GPU SM Count: %d\n", prop.multiProcessorCount);
      printf("GPU Total Threads: %d\n",
             prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
      printf("GPU Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
      printf("GPU Max Threads Per Dimension: [%d, %d, %d] (x,y,z)\n",
             prop.maxThreadsDim[0], prop.maxThreadsDim[1],
             prop.maxThreadsDim[2]);
      printf("GPU Max Grid Size: [%d, %d, %d] (x,y,z)\n", prop.maxGridSize[0],
             prop.maxGridSize[1], prop.maxGridSize[2]);
#endif
    } else {
      printf("GPU Device: [Failed to get properties: %s]\n",
             cudaGetErrorString(err));
    }
  } else {
    printf("GPU Device: [No CUDA device available or error: %s]\n",
           err != cudaSuccess ? cudaGetErrorString(err) : "No devices found");
  }

  // Setup dimensions and compute padded sizes
  AttentionDims dims =
      make_attention_dims(cfg.batch, cfg.n_heads, cfg.seq_len, cfg.head_dim);
  const size_t head_dim_padded = dims.head_dim_padded;
  const size_t qkv_size =
      cfg.batch * cfg.n_heads * cfg.seq_len * head_dim_padded;

  // Allocate CUDA managed memory
  float *Q, *K, *V, *out;
  CUDA_CHECK(cudaMallocManaged(&Q, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&K, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&V, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&out, qkv_size * sizeof(float)));

  // Initialize with random values (CPU, works on managed memory)
  init_random_tensors(Q, K, V, out, cfg.batch, cfg.n_heads, cfg.seq_len,
                      head_dim_padded, cfg.seed);

  // Create cuda configuration
  // HACK: This created silent bugs in my code be very careful !
  // Swapped mapping: x=queries (up to 1024), y=heads, z=batch (up to 64)
  // dim3 threads_per_block(256, 1, 1);
  dim3 threads_per_block(512, 1, 1);
  CudaConfig cuda_conf = make_cuda_config(dims, threads_per_block);

  // Allocate workspace based on kernel version:
  // NOTE: I should probably do something like this
  // - v0, v1: Full rectangular workspace (total_threads * seq_len_padded)
  // - v2: Triangular workspace (batch * n_heads * seq_len * (seq_len + 1) / 2)
  // - v3: Uses shared memory, no global workspace needed (allocate minimal)
  const size_t seq_len_padded = round_up_pow2(dims.seq_len, VEC_PADDING);
  size_t workspace_size;

  // v0, v1 and others: Full rectangular workspace
  workspace_size = cuda_conf.total_threads * seq_len_padded * sizeof(float);
  VERBOSE_PRINT("Workspace: rectangular layout (%zu bytes)\n", workspace_size);

  float *workspace;
  CUDA_CHECK(cudaMallocManaged(&workspace, workspace_size));

  VERBOSE_PRINT("Sample Q values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("Q[0][0][0][%zu] = %f\n", d, Q[d]);
  }

  printf("\nRunning attention forward pass...\n");

  // Warm-up runs
  for (int i = 0; i < cfg.warmup; i++) {
    cmhsa_forward_cuda(Q, K, V, out, workspace, dims, cuda_conf);
  }

  // Timed iterations using CUDA events
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  float total_ms = 0.0f;
  float checksum = 0.0f;
  for (int i = 0; i < cfg.iters; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    cmhsa_forward_cuda(Q, K, V, out, workspace, dims, cuda_conf);
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
    struct Outputs outputs = {Q, K, V, out, qkv_size, 0, 0};
    write_validation_artifacts(cfg.validate_dir, &cfg, &outputs);
  }

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  cudaFree(workspace);
  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(out);

  printf("\nCompleted successfully!\n");
  return 0;
}
