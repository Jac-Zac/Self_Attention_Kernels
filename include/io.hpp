#pragma once
#include "parser.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline void write_bin(const char *path, const float *data, size_t count) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Error: cannot open %s for writing\n", path);
    exit(1);
  }
  size_t n = fwrite(data, sizeof(float), count, f);
  if (n != count) {
    fprintf(stderr, "Error: short write to %s\n", path);
    fclose(f);
    exit(1);
  }
  fclose(f);
}

inline void write_meta(const char *path, const RunConfig *cfg) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Error: cannot open %s for writing\n", path);
    exit(1);
  }
  fprintf(f,
          "{\n  \"batch\": %zu,\n  \"n_heads\": %zu,\n  \"seq_len\": %zu,\n  "
          "\"head_dim\": %zu,\n  \"dtype\": \"float32\",\n  \"seed\": %u  }\n",
          cfg->batch, cfg->n_heads, cfg->seq_len, cfg->head_dim, cfg->seed);
  fclose(f);
}

inline void write_validation_artifacts(const char *dir, const RunConfig *cfg,
                                       const struct Outputs *out) {
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
  system(cmd);

  char path[256];
  snprintf(path, sizeof(path), "%s/q.bin", dir);
  write_bin(path, out->Q, out->qkv_size);
  snprintf(path, sizeof(path), "%s/k.bin", dir);
  write_bin(path, out->K, out->qkv_size);
  snprintf(path, sizeof(path), "%s/v.bin", dir);
  write_bin(path, out->V, out->qkv_size);
  snprintf(path, sizeof(path), "%s/out.bin", dir);
  write_bin(path, out->out, out->qkv_size);
  snprintf(path, sizeof(path), "%s/meta.json", dir);
  write_meta(path, cfg);
}
