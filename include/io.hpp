#pragma once
// Binary I/O utilities for reading/writing validation artifacts to disk.

#include "memory.h"
#include "parser.hpp"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ============================================================================
// Reading utilities
// ============================================================================

// Read binary float32 file into pre-allocated buffer.
// Returns 0 on success, -1 on error.
inline int read_bin(const char *path, float *dst, size_t expected_count) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error: cannot open %s for reading: %s\n", path,
            strerror(errno));
    return -1;
  }

  // Get file size
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  size_t file_count = (size_t)file_size / sizeof(float);
  if (file_count != expected_count) {
    fprintf(stderr, "Error: %s has %zu floats but expected %zu\n", path,
            file_count, expected_count);
    fclose(f);
    return -1;
  }

  size_t n = fread(dst, sizeof(float), expected_count, f);
  if (n != expected_count) {
    fprintf(stderr, "Error: short read from %s (read %zu of %zu elements)\n",
            path, n, expected_count);
    fclose(f);
    return -1;
  }

  fclose(f);
  return 0;
}

// Read contiguous [B,H,S,D] tensor from file and unpack into padded buffer.
// Source layout: contiguous [B,H,S,D]
// Destination layout: row-major with row stride = D_pad (padded to VEC_PADDING)
inline int read_packed_qkv(const char *path, float *dst, const RunConfig *cfg) {
  const size_t B = cfg->batch;
  const size_t H = cfg->n_heads;
  const size_t S = cfg->seq_len;
  const size_t D = cfg->head_dim;
  const size_t D_pad = round_up_pow2(D, VEC_PADDING);

  const size_t logical_elems = B * H * S * D;
  float *tmp = (float *)malloc(sizeof(float) * logical_elems);
  if (!tmp) {
    fprintf(stderr, "Error: OOM allocating unpack buffer for %s\n", path);
    return -1;
  }

  if (read_bin(path, tmp, logical_elems) != 0) {
    free(tmp);
    return -1;
  }

  // Unpack: contiguous -> padded layout
  size_t src_idx = 0;
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      const size_t bh_base = b * (H * S * D_pad) + h * (S * D_pad);
      for (size_t s = 0; s < S; ++s) {
        const size_t row_off = bh_base + s * D_pad;
        memcpy(&dst[row_off], &tmp[src_idx], sizeof(float) * D);
        // Zero-fill padding
        for (size_t d = D; d < D_pad; ++d) {
          dst[row_off + d] = 0.0f;
        }
        src_idx += D;
      }
    }
  }

  free(tmp);
  return 0;
}

// Simple JSON parser for meta.json - extracts batch, n_heads, seq_len, head_dim
// Returns 0 on success, -1 on error.
inline int read_meta(const char *path, RunConfig *cfg) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Error: cannot open %s for reading: %s\n", path,
            strerror(errno));
    return -1;
  }

  // Read entire file (meta.json is small)
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  if (file_size > 4096) {
    fprintf(stderr, "Error: meta.json too large (%ld bytes)\n", file_size);
    fclose(f);
    return -1;
  }

  char *buf = (char *)malloc((size_t)file_size + 1);
  if (!buf) {
    fprintf(stderr, "Error: OOM reading meta.json\n");
    fclose(f);
    return -1;
  }

  size_t n = fread(buf, 1, (size_t)file_size, f);
  buf[n] = '\0';
  fclose(f);

  // Simple parsing - look for "key": value patterns
  // This is intentionally simple and doesn't handle all JSON edge cases
  int found = 0;
  char *p;

  p = strstr(buf, "\"batch\"");
  if (p) {
    p = strchr(p, ':');
    if (p) {
      cfg->batch = (size_t)strtoull(p + 1, NULL, 10);
      found++;
    }
  }

  p = strstr(buf, "\"n_heads\"");
  if (p) {
    p = strchr(p, ':');
    if (p) {
      cfg->n_heads = (size_t)strtoull(p + 1, NULL, 10);
      found++;
    }
  }

  p = strstr(buf, "\"seq_len\"");
  if (p) {
    p = strchr(p, ':');
    if (p) {
      cfg->seq_len = (size_t)strtoull(p + 1, NULL, 10);
      found++;
    }
  }

  p = strstr(buf, "\"head_dim\"");
  if (p) {
    p = strchr(p, ':');
    if (p) {
      cfg->head_dim = (size_t)strtoull(p + 1, NULL, 10);
      found++;
    }
  }

  free(buf);

  if (found != 4) {
    fprintf(stderr,
            "Error: meta.json missing required fields (found %d of 4)\n",
            found);
    return -1;
  }

  return 0;
}

// Load Q, K, V from input directory. Reads meta.json first to get dimensions.
// Caller must allocate tensors AFTER this call since dimensions come from file.
inline int load_input_qkv(const char *dir, float *Q, float *K, float *V,
                          const RunConfig *cfg) {
  char path[256];

  snprintf(path, sizeof(path), "%s/q.bin", dir);
  if (read_packed_qkv(path, Q, cfg) != 0)
    return -1;

  snprintf(path, sizeof(path), "%s/k.bin", dir);
  if (read_packed_qkv(path, K, cfg) != 0)
    return -1;

  snprintf(path, sizeof(path), "%s/v.bin", dir);
  if (read_packed_qkv(path, V, cfg) != 0)
    return -1;

  return 0;
}

// ============================================================================
// Writing utilities
// ============================================================================

// Create directory if it doesn't exist (replaces system("mkdir -p"))
inline int ensure_directory(const char *path) {
  struct stat st;
  memset(&st, 0, sizeof(st));
  if (stat(path, &st) == -1) {
    if (mkdir(path, 0755) != 0 && errno != EEXIST) {
      fprintf(stderr, "Error: cannot create directory %s: %s\n", path,
              strerror(errno));
      return -1;
    }
  }
  return 0;
}

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

// Pack a logical [B,H,S,D] tensor from a padded buffer into a
// contiguous temporary buffer, then write it to disk. The source layout is
// row-major with row stride = head_dim_pad (padded to VEC_PADDING).
inline void write_packed_tensor(const char *path, const float *src,
                                const RunConfig *cfg) {
  const size_t B = cfg->batch;
  const size_t H = cfg->n_heads;
  const size_t S = cfg->seq_len;
  const size_t D = cfg->head_dim;
  const size_t D_pad = round_up_pow2(D, VEC_PADDING);

  const size_t logical_elems = B * H * S * D;
  float *tmp = (float *)malloc(sizeof(float) * logical_elems);
  if (!tmp) {
    fprintf(stderr, "Error: OOM allocating pack buffer for %s\n", path);
    exit(1);
  }

  size_t dst_idx = 0;
  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < H; ++h) {
      const size_t bh_base = b * (H * S * D_pad) + h * (S * D_pad);
      for (size_t s = 0; s < S; ++s) {
        const size_t row_off = bh_base + s * D_pad;
        memcpy(&tmp[dst_idx], &src[row_off], sizeof(float) * D);
        dst_idx += D;
      }
    }
  }

  write_bin(path, tmp, logical_elems);
  free(tmp);
}

// Write output tensor and metadata to disk for validation.
inline void write_output_artifact(const char *dir, const float *out,
                                  const RunConfig *cfg) {
  if (ensure_directory(dir) != 0) {
    fprintf(stderr, "Warning: could not create output directory %s\n", dir);
  }

  char path[256];
  snprintf(path, sizeof(path), "%s/out.bin", dir);
  write_packed_tensor(path, out, cfg);
  snprintf(path, sizeof(path), "%s/meta.json", dir);
  write_meta(path, cfg);
}
