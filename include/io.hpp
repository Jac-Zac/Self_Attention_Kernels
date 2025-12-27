#pragma once
#include "parser.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline void write_bin(const char *path, const float *data, size_t count) {
  if (!data) {
    fprintf(stderr, "Error: NULL data pointer for %s\n", path);
    exit(1);
  }
  if (count == 0) {
    fprintf(stderr, "Warning: writing 0 elements to %s\n", path);
  }
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Error: cannot open %s for writing\n", path);
    exit(1);
  }
  size_t n = fwrite(data, sizeof(float), count, f);
  if (n != count) {
    fprintf(stderr, "Error: short write to %s (wrote %zu of %zu elements)\n",
            path, n, count);
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
          "\"head_dim\": %zu,\n  \"dtype\": \"float32\",\n  \"seed\": %u\n}\n",
          cfg->batch, cfg->n_heads, cfg->seq_len, cfg->head_dim, cfg->seed);
  fclose(f);
}

// Write validation artifacts (Q, K, V, out tensors and metadata) to disk
// for comparison with reference implementations.
// Tensors are now contiguous (no padding), so we write them directly.
inline void write_validation_artifacts(const char *dir, const RunConfig *cfg,
                                       const struct Outputs *out) {
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
  int ret = system(cmd);
  if (ret != 0) {
    fprintf(stderr, "Warning: mkdir command returned %d\n", ret);
  }

  const size_t logical_elems =
      cfg->batch * cfg->n_heads * cfg->seq_len * cfg->head_dim;

  char path[256];
  snprintf(path, sizeof(path), "%s/q.bin", dir);
  write_bin(path, out->Q, logical_elems);
  snprintf(path, sizeof(path), "%s/k.bin", dir);
  write_bin(path, out->K, logical_elems);
  snprintf(path, sizeof(path), "%s/v.bin", dir);
  write_bin(path, out->V, logical_elems);
  snprintf(path, sizeof(path), "%s/out.bin", dir);
  write_bin(path, out->out, logical_elems);
  snprintf(path, sizeof(path), "%s/meta.json", dir);
  write_meta(path, cfg);
}
