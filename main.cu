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

  // Allocate CPU memory
  float *Q_host = NULL, *K_host = NULL, *V_host = NULL, *out_host = NULL;
  ALIGNED_ALLOC_FLOAT(Q_host, qkv_size);
  ALIGNED_ALLOC_FLOAT(K_host, qkv_size);
  ALIGNED_ALLOC_FLOAT(V_host, qkv_size);
  ALIGNED_ALLOC_FLOAT(out_host, qkv_size);
  Q_host = (float *)ASSUME_ALIGNED(Q_host, ALIGNMENT);
  K_host = (float *)ASSUME_ALIGNED(K_host, ALIGNMENT);
  V_host = (float *)ASSUME_ALIGNED(V_host, ALIGNMENT);
  out_host = (float *)ASSUME_ALIGNED(out_host, ALIGNMENT);

  // Allocate GPU memory
  float *Q_dev, *K_dev, *V_dev, *out_dev;
  CUDA_CHECK(cudaMalloc(&Q_dev, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K_dev, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V_dev, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out_dev, qkv_size * sizeof(float)));

  // Initialize with random values (CPU)
  init_random_tensors(Q_host, K_host, V_host, out_host, cfg.batch, cfg.n_heads,
                      cfg.seq_len, head_dim_padded, cfg.seed);

  // Allocate workspace using kernel's size query
  size_t workspace_size = cmhsa_get_workspace_size(dims);
  VERBOSE_PRINT("Workspace size: %zu bytes\n", workspace_size);

  float *workspace = NULL;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(Q_dev, Q_host, qkv_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(K_dev, K_host, qkv_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(V_dev, V_host, qkv_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  VERBOSE_PRINT("Sample Q values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("Q[0][0][0][%zu] = %f\n", d, Q_host[d]);
  }

  printf("\nRunning attention forward pass...\n");

  // Warm-up runs
  for (int i = 0; i < cfg.warmup; i++) {
    cmhsa_forward_cuda(Q_dev, K_dev, V_dev, out_dev, workspace, dims);
  }

  // Timed iterations using CUDA events
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  float total_ms = 0.0f;
  float checksum = 0.0f;
  for (int i = 0; i < cfg.iters; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    cmhsa_forward_cuda(Q_dev, K_dev, V_dev, out_dev, workspace, dims);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    // Copy output back to host for checksum
    CUDA_CHECK(cudaMemcpy(out_host, out_dev, qkv_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    total_ms += ms;
    if (cfg.head_dim > 0)
      checksum += out_host[0];
  }

  // Final copy of results to host
  CUDA_CHECK(cudaMemcpy(out_host, out_dev, qkv_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("CUDA attention forward (total): %.3f ms\n", total_ms);
  printf("CUDA attention forward (per-iter): %.6f ms\n", total_ms / cfg.iters);
  VERBOSE_PRINT("Checksum (sum of out[0] over iters): %f\n", checksum);

  VERBOSE_PRINT("\nSample output values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("out[0][0][0][%zu] = %f\n", d, out_host[d]);
  }

  // Validation mode: write artifacts for Python
  if (cfg.validate) {
    struct Outputs outputs = {Q_host, K_host, V_host, out_host, qkv_size, 0, 0};
    write_validation_artifacts(cfg.validate_dir, &cfg, &outputs);
  }

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  cudaFree(workspace);
  cudaFree(Q_dev);
  cudaFree(K_dev);
  cudaFree(V_dev);
  cudaFree(out_dev);
  free(Q_host);
  free(K_host);
  free(V_host);
  free(out_host);

  printf("\nCompleted successfully!\n");
  return 0;
}
