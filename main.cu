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

  // If input_dir specified, load dimensions from meta.json first
  if (cfg.input_dir) {
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.json", cfg.input_dir);
    if (read_meta(meta_path, &cfg) != 0) {
      return 1;
    }
    printf("Loaded dimensions from %s\n", meta_path);
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

  // Allocate CPU tensors (same as CPU version)
  struct Tensors t;
  if (allocate_tensors(&t, qkv_size, 0) != 0) {
    return 1;
  }

  // Initialize tensors: either load from files or generate random
  if (cfg.input_dir) {
    printf("Loading Q,K,V from %s\n", cfg.input_dir);
    if (load_input_qkv(cfg.input_dir, t.Q, t.K, t.V, &cfg) != 0) {
      free_tensors(&t);
      return 1;
    }
    // Zero-initialize output
    memset(t.out, 0, qkv_size * sizeof(float));
  } else {
    init_random_tensors(t.Q, t.K, t.V, t.out, cfg.batch, cfg.n_heads,
                        cfg.seq_len, head_dim_padded, cfg.seed);
  }

  // Allocate GPU memory
  float *Q_device, *K_device, *V_device, *out_device;

  // Note that cudaMalloc returns 256+ byte aligned pointers
  CUDA_CHECK(cudaMalloc(&Q_device, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K_device, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V_device, qkv_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out_device, qkv_size * sizeof(float)));

  // Allocate workspace using kernel's size query
  size_t workspace_size = cmhsa_get_workspace_size(dims);
  VERBOSE_PRINT("Workspace size: %zu bytes\n", workspace_size);

  float *workspace = NULL;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  }

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(Q_device, t.Q, qkv_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(K_device, t.K, qkv_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(V_device, t.V, qkv_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  VERBOSE_PRINT("Sample Q values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("Q[0][0][0][%zu] = %f\n", d, t.Q[d]);
  }

  printf("\nRunning attention forward pass...\n");

  // Warm-up runs
  for (int i = 0; i < cfg.warmup; i++) {
    cmhsa_forward_cuda(Q_device, K_device, V_device, out_device, workspace,
                       dims);
    CUDA_CHECK(cudaDeviceSynchronize()); // Add this for profiling
  }

  // ============================================================================
  // Timing Methodology (for fair comparison with PyTorch)
  // ============================================================================
  // We use batch timing: record start event, run all iterations, record end
  // event, then synchronize once. This matches how PyTorch's benchmark times
  // kernels (torch.cuda.Event around the entire loop, single sync at end).
  //
  // This reflects real-world usage where multiple kernel launches are pipelined
  // without synchronizing after each one. Per-iteration synchronization would
  // add artificial overhead not present in production code.
  // ============================================================================

  // Timed iterations using CUDA events (batch timing)
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure warmup is complete
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < cfg.iters; i++) {
    cmhsa_forward_cuda(Q_device, K_device, V_device, out_device, workspace,
                       dims);
    CUDA_CHECK(cudaDeviceSynchronize()); // Add this for profiling
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));

  // Copy final output to host (for validation)
  CUDA_CHECK(cudaMemcpy(t.out, out_device, qkv_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("CUDA attention forward (total): %.3f ms\n", total_ms);
  printf("CUDA attention forward (per-iter): %.6f ms\n", total_ms / cfg.iters);

  VERBOSE_PRINT("\nSample output values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("out[0][0][0][%zu] = %f\n", d, t.out[d]);
  }

  // Validation mode: write artifacts for Python
  if (cfg.validate) {
    size_t stats_size = cfg.batch * cfg.n_heads * cfg.seq_len;
    struct Outputs outputs = {t.Q, t.K, t.V, t.out, qkv_size, stats_size, 0};
    write_validation_artifacts(cfg.validate_dir, &cfg, &outputs);
  }

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  cudaFree(workspace);
  cudaFree(Q_device);
  cudaFree(K_device);
  cudaFree(V_device);
  cudaFree(out_device);
  free_tensors(&t);

  printf("\nCompleted successfully!\n");
  return 0;
}
