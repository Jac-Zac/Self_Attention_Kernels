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
} RunConfig;

inline void print_usage() {
  fprintf(stderr,
          "Usage: program [--validate-outdir DIR] [--batch N] [--n_heads N] "
          "[--seq_len N] [--head_dim N] [--seed N] [--warmup N] [--iters N]\n");
}

inline int parse_args(int argc, char **argv, RunConfig *cfg) {
  // defaults
  cfg->batch = 2;
  cfg->n_heads = 4;
  cfg->seq_len = 128;
  cfg->head_dim = 64;
  cfg->seed = 1337;
  cfg->validate = 0;
  cfg->validate_dir = "python_test";
  cfg->warmup = 5;
  cfg->iters = 25;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--validate-outdir") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->validate = 1;
      cfg->validate_dir = argv[++i];
    } else if (strcmp(arg, "--batch") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->batch = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--n_heads") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->n_heads = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--seq_len") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->seq_len = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--head_dim") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->head_dim = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--seed") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->seed = (unsigned)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--warmup") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->warmup = (int)strtol(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--iters") == 0) {
      if (i + 1 >= argc) {
        print_usage();
        return 1;
      }
      cfg->iters = (int)strtol(argv[++i], NULL, 10);
    } else {
      fprintf(stderr, "Error: unknown flag '%s'\n", arg);
      print_usage();
      return 1;
    }
  }
  return 0;
}
