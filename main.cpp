#include "include/cmhsa_forward.h"
#include "include/io.hpp"
#include "include/macros.hpp"
#include "include/parser.hpp"
#include "include/timing.h"
#include <stdio.h>
#include <stdlib.h>

// Default values that are resolved when using make ...
#ifndef BACKEND
#define BACKEND "unknown"
#endif
#ifndef VERSION_STR
#define VERSION_STR "v0"
#endif

int main(int argc, char *argv[]) {
  RunConfig cfg;

  // Parse arguments into a struct
  if (parse_args(argc, argv, &cfg) != 0) {
    return 1; // invalid flags or usage
  }

  // Get the parsed arguments
  size_t batch = cfg.batch;
  size_t n_heads = cfg.n_heads;
  size_t seq_len = cfg.seq_len;
  size_t head_dim = cfg.head_dim;
  unsigned seed = cfg.seed;
  int validate = cfg.validate;
  const char *validate_dir = cfg.validate_dir;

  printf("backend=%s version=%s\n", BACKEND, VERSION_STR);
  printf("batch=%zu n_heads=%zu seq_len=%zu head_dim=%zu\n", batch, n_heads,
         seq_len, head_dim);

  // Setup dimensions
  AttentionDims dims = {batch, n_heads, seq_len, head_dim};

  // Calculate buffer sizes
  size_t qkv_size = batch * n_heads * seq_len * head_dim;
  size_t stats_size = batch * n_heads * seq_len;

  // Allocate input/output tensors
  // TODO: To allocate memory better and alligned
  float *Q = (float *)malloc(sizeof(float) * qkv_size);
  float *K = (float *)malloc(sizeof(float) * qkv_size);
  float *V = (float *)malloc(sizeof(float) * qkv_size);
  float *out = (float *)malloc(sizeof(float) * qkv_size);

  // Check allocations
  if (!Q || !K || !V || !out) {
    fprintf(stderr, "Error: memory allocation failed\n");
    free(Q);
    free(K);
    free(V);
    free(out);
    return 1;
  }

  // Initialize with random small values (typical for attention)
  srand(seed); // Fixed seed for reproducibility
  for (size_t i = 0; i < qkv_size; i++) {
    Q[i] = ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f; // [-0.05, 0.05]
    K[i] = ((float)rand() / (float)RAND_MAX) * 0.1f - 0.05f;
    V[i] = ((float)rand() / (float)RAND_MAX) * 0.5f; // [0, 0.5]
  }

  VERBOSE_PRINT("Sample Q values (first head, first token):\n");
  for (size_t d = 0; d < head_dim && d < 5; d++) {
    VERBOSE_PRINT("Q[0][0][0][%zu] = %f\n", d, Q[d]);
  }

  printf("\nRunning attention forward pass...\n");

  // Warm-up and timed iterations
  const int warmup = 5;
  const int iters = 25;

  // Warm-up runs (not timed)
  for (int i = 0; i < warmup; i++) {
    cmhsa_forward_cpu(Q, K, V, out, dims);
  }

  // Timed loop
  unsigned long long total_ns = 0ULL;
  float checksum = 0.0f;

  for (int i = 0; i < iters; i++) {
    struct timespec start, end;
    NOW(start);
    cmhsa_forward_cpu(Q, K, V, out, dims);
    NOW(end);
    total_ns += ns_diff(start, end);

    // Accumulate a small checksum to keep the compiler honest
    if (head_dim > 0) {
      checksum += out[0];
    }
  }

  print_timing("CPU attention forward (total)", total_ns);
  printf("CPU attention forward (per-iter): %.6f s\n",
         (double)total_ns / (double)iters / 1e9);
  VERBOSE_PRINT("Checksum (sum of out[0] over iters): %f\n", checksum);

  // Optional sample outputs
  VERBOSE_PRINT("\nSample output values (first head, first token):\n");
  for (size_t d = 0; d < head_dim && d < 5; d++) {
    VERBOSE_PRINT("out[0][0][0][%zu] = %f\n", d, out[d]);
  }

  // Validation mode: write artifacts for Python and exit
  if (validate) {
    struct Outputs outputs = {Q, K, V, out, qkv_size, stats_size, 0};

    write_validation_artifacts(validate_dir, &cfg, &outputs);
  }

  // Cleanup
  free(Q);
  free(K);
  free(V);
  free(out);

  printf("\nCompleted successfully!\n");
  return 0;
}
