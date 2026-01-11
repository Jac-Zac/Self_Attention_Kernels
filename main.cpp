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

  // If input_dir specified, load dimensions from meta.json first
  if (cfg.input_dir) {
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.json", cfg.input_dir);
    if (read_meta(meta_path, &cfg) != 0) {
      return 1;
    }
    printf("Loaded dimensions from %s\n", meta_path);
  }

  int threads = resolve_thread_count(cfg.threads);
  printf("backend=%s version=%s\n", BACKEND, VERSION_STR);
  printf("batch=%zu n_heads=%zu seq_len=%zu head_dim=%zu threads=%d\n",
         cfg.batch, cfg.n_heads, cfg.seq_len, cfg.head_dim, threads);

  // Setup dimensions and compute padded sizes
  AttentionDims dims =
      make_attention_dims(cfg.batch, cfg.n_heads, cfg.seq_len, cfg.head_dim);

  const size_t head_dim_padded = dims.head_dim_padded;
  const size_t qkv_size =
      cfg.batch * cfg.n_heads * cfg.seq_len * head_dim_padded;

  const size_t workspace_bytes = cmhsa_get_workspace_size_cpu(dims, threads);

  // Allocate tensors
  struct Tensors t;
  if (allocate_tensors(&t, qkv_size, workspace_bytes) != 0) {
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

  // Validation mode: write output artifact for Python
  if (cfg.validate) {
    write_output_artifact(cfg.validate_dir, t.out, &cfg);
  }

  free_tensors(&t);
  printf("\nCompleted successfully!\n");
  return 0;
}
