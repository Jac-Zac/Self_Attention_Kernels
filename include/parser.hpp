#pragma once
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple run configuration and parser (C-compatible)
typedef struct RunConfig {
  size_t batch;
  size_t n_heads;
  size_t seq_len;
  size_t head_dim;
  unsigned seed;
  int validate;
  const char *validate_dir;
  int warmup;
  int iters;
  int threads;           // 0 -> use OMP max or 1 if no OMP
  const char *input_dir; // NULL -> generate random Q,K,V; else load from dir
  int dims_explicit;     // Track if user explicitly set dimension flags
} RunConfig;

inline void print_usage() {
  fprintf(
      stderr,
      "Usage: program [--input-dir DIR] [--validate-outdir DIR] [--batch N] "
      "[--n_heads N] [--seq_len N] [--head_dim N] [--seed N] [--warmup N] "
      "[--iters N] [--threads N]\n\n"
      "  --input-dir DIR     Load Q,K,V from DIR (requires q.bin, k.bin, "
      "v.bin, meta.json)\n"
      "                      Dimensions are inferred from meta.json\n"
      "                      Cannot be combined with --batch/--n_heads/"
      "--seq_len/--head_dim\n");
}

// Helper to check if next argument exists
inline int check_next_arg(int i, int argc, const char *flag) {
  if (i + 1 >= argc) {
    fprintf(stderr, "Error: flag '%s' requires an argument\n", flag);
    print_usage();
    return 1;
  }
  return 0;
}

// Validate parsed config, returns 0 on success
inline int validate_config(const RunConfig *cfg) {
  // Check for conflict: --input-dir with explicit dimension flags
  if (cfg->input_dir && cfg->dims_explicit) {
    fprintf(stderr,
            "Error: cannot specify --batch/--n_heads/--seq_len/--head_dim "
            "when using --input-dir\n");
    return 1;
  }

  // Only validate dimensions if not loading from input_dir
  // (dimensions will be loaded from meta.json later)
  if (!cfg->input_dir) {
    if (cfg->batch == 0 || cfg->n_heads == 0 || cfg->seq_len == 0 ||
        cfg->head_dim == 0) {
      fprintf(stderr, "Error: all dimensions must be positive\n");
      return 1;
    }
  }

  if (cfg->warmup < 0 || cfg->iters <= 0) {
    fprintf(stderr,
            "Error: warmup must be non-negative and iters must be positive\n");
    return 1;
  }
  if (!cfg->input_dir && (cfg->batch > 1024 || cfg->n_heads > 128 ||
                          cfg->seq_len > 16384 || cfg->head_dim > 512)) {
    fprintf(stderr,
            "Warning: dimensions are very large and may cause memory issues\n");
  }
  return 0;
}

inline int parse_args(int argc, char **argv, RunConfig *cfg) {
  // defaults
  cfg->batch = 2;
  cfg->n_heads = 4;
  cfg->seq_len = 128;
  cfg->head_dim = 64;
  cfg->seed = 1337;
  cfg->validate = 0;
  cfg->validate_dir = "results/tmp";
  cfg->warmup = 5;
  cfg->iters = 25;
  cfg->threads = 0;      // resolved later in main
  cfg->input_dir = NULL; // NULL means generate random
  cfg->dims_explicit = 0;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--input-dir") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->input_dir = argv[++i];
    } else if (strcmp(arg, "--validate-outdir") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->validate = 1;
      cfg->validate_dir = argv[++i];
    } else if (strcmp(arg, "--batch") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->batch = (size_t)strtoull(argv[++i], NULL, 10);
      cfg->dims_explicit = 1;
    } else if (strcmp(arg, "--n_heads") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->n_heads = (size_t)strtoull(argv[++i], NULL, 10);
      cfg->dims_explicit = 1;
    } else if (strcmp(arg, "--seq_len") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->seq_len = (size_t)strtoull(argv[++i], NULL, 10);
      cfg->dims_explicit = 1;
    } else if (strcmp(arg, "--head_dim") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->head_dim = (size_t)strtoull(argv[++i], NULL, 10);
      cfg->dims_explicit = 1;
    } else if (strcmp(arg, "--seed") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->seed = (unsigned)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--warmup") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->warmup = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--iters") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->iters = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--threads") == 0) {
      if (check_next_arg(i, argc, arg))
        return 1;
      cfg->threads = (int)strtol(argv[++i], NULL, 10);
    } else {
      fprintf(stderr, "Error: unknown flag '%s'\n", arg);
      print_usage();
      return 1;
    }
  }

  return validate_config(cfg);
}
