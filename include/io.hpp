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

// Pack a logical [B,H,S,D] tensor from a padded-stride buffer into a
// contiguous temporary buffer, then write it to disk. The source layout is
// row-major with row stride = head_dim_stride.
inline void write_packed_qkv(const char *path, const float *src,
                             const RunConfig *cfg) {
  const size_t B = cfg->batch;
  const size_t H = cfg->n_heads;
  const size_t S = cfg->seq_len;
  const size_t D = cfg->head_dim;
  const size_t Dstride = round_up_pow2(D, VEC_PADDING);

  const size_t logical_elems = B * H * S * D;
  float *tmp = (float *)malloc(sizeof(float) * logical_elems);
  if (!tmp) {
    fprintf(stderr, "Error: OOM allocating pack buffer for %s\n", path);
    exit(1);
  }

  size_t dst_idx = 0;
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      const size_t bh_base = b * (H * S * Dstride) + h * (S * Dstride);
      for (size_t s = 0; s < S; ++s) {
        const size_t row_off = bh_base + s * Dstride;
        memcpy(&tmp[dst_idx], &src[row_off], sizeof(float) * D);
        dst_idx += D;
      }
    }
  }

  write_bin(path, tmp, logical_elems);
  free(tmp);
}

// Write validation artifacts (Q, K, V, out tensors and metadata) to disk
// for comparison with reference implementations
inline void write_validation_artifacts(const char *dir, const RunConfig *cfg,
                                       const struct Outputs *out) {
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", dir);
  int ret = system(cmd);
  if (ret != 0) {
    fprintf(stderr, "Warning: mkdir command returned %d\n", ret);
  }

  char path[256];
  snprintf(path, sizeof(path), "%s/q.bin", dir);
  write_packed_qkv(path, out->Q, cfg);
  snprintf(path, sizeof(path), "%s/k.bin", dir);
  write_packed_qkv(path, out->K, cfg);
  snprintf(path, sizeof(path), "%s/v.bin", dir);
  write_packed_qkv(path, out->V, cfg);
  snprintf(path, sizeof(path), "%s/out.bin", dir);
  write_packed_qkv(path, out->out, cfg);
  snprintf(path, sizeof(path), "%s/meta.json", dir);
  write_meta(path, cfg);
}
