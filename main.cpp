#include "include/cmhsa_forward.h"
#include "include/io.hpp"
#include "include/parser.hpp"
#include "include/timing.h"
#include "include/utils.hpp"
#include <stdio.h>

// Default values resolved at compile time via make
#ifndef BACKEND
#define BACKEND "unknown"
#endif
#ifndef VERSION_STR
#define VERSION_STR "v0"
#endif

int main(int argc, char *argv[]) {
  RunConfig cfg;
  if (parse_args(argc, argv, &cfg) != 0) {
    return 1;
  }

  int threads = resolve_thread_count(cfg.threads);
  printf("backend=%s version=%s\n", BACKEND, VERSION_STR);
  printf("batch=%zu n_heads=%zu seq_len=%zu head_dim=%zu threads=%d\n",
         cfg.batch, cfg.n_heads, cfg.seq_len, cfg.head_dim, threads);

  // Setup dimensions and compute padded sizes
  AttentionDims dims =
      make_attention_dims(cfg.batch, cfg.n_heads, cfg.seq_len, cfg.head_dim);
  const size_t head_dim_padded = dims.head_dim_padded;
  const size_t seq_len_padded = dims.seq_len_padded;
  const size_t qkv_size =
      cfg.batch * cfg.n_heads * cfg.seq_len * head_dim_padded;

  // NOTE: It is a bit wastfull but usefull for future implementations
  // Workspace size: always allocate threads * TILE_Q * seq_len_padded
  // Tiled kernels use the full workspace, non-tiled kernels use a subset
  const size_t workspace_size = (size_t)threads * TILE_Q * seq_len_padded;

  // Allocate tensors
  struct Tensors t;
  if (allocate_tensors(&t, qkv_size, workspace_size) != 0) {
    return 1;
  }

  // Initialize with random values (NUMA-aware)
  init_random_tensors(t.Q, t.K, t.V, t.out, cfg.batch, cfg.n_heads, cfg.seq_len,
                      head_dim_padded, cfg.seed);

  VERBOSE_PRINT("Sample Q values (first head, first token):\n");
  for (size_t d = 0; d < cfg.head_dim && d < 5; d++) {
    VERBOSE_PRINT("Q[0][0][0][%zu] = %f\n", d, t.Q[d]);
  }

  printf("\nRunning attention forward pass...\n");

  // Warm-up runs
  for (int i = 0; i < cfg.warmup; i++) {
    cmhsa_forward_cpu(t.Q, t.K, t.V, t.out, t.workspace, dims);
  }

  // ============================================================================
  // Timing Methodology (for fair comparison with PyTorch)
  // ============================================================================
  // We use batch timing: record start time, run all iterations, record end
  // time. This matches how PyTorch's benchmark times kernels (time.perf_counter
  // around the entire loop).
  //
  // For CPU, this is largely equivalent to per-iteration timing since CPU calls
  // are synchronous, but we use batch timing for consistency with the CUDA.
  // ============================================================================

  // Timed iterations (batch timing)
  struct timespec start, end;
  NOW(start);
  for (int i = 0; i < cfg.iters; i++) {
    cmhsa_forward_cpu(t.Q, t.K, t.V, t.out, t.workspace, dims);
  }
  NOW(end);
  unsigned long long total_ns = ns_diff(start, end);

  print_timing("CPU attention forward (total)", total_ns);
  printf("CPU attention forward (per-iter): %.6f s\n",
         (double)total_ns / (double)cfg.iters / 1e9);

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

  free_tensors(&t);
  printf("\nCompleted successfully!\n");
  return 0;
}
