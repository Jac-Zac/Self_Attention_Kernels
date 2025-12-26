#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Multi-threaded Flash Attention v4
// Optimizations over v3:
// 1. collapse(3) to parallelize over batch, heads, AND queries
// 2. schedule(dynamic) for better load balancing with causal mask
// 3. Unrolled weighted sum loop for better ILP

#ifndef TILE_K
#define TILE_K 256
#endif

static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  (void)attn_weights;

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);

  const float *q_aligned = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *k_aligned = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *v_aligned = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *out_aligned = (float *)ASSUME_ALIGNED(out, ALIGNMENT);

  // Parallelize over batch, heads, AND queries
  // Use dynamic scheduling because causal mask creates imbalanced work
  // (query 0 has 1 key, query N has N+1 keys)
#pragma omp parallel for collapse(3) schedule(dynamic, 16)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      for (size_t i = 0; i < seq_len; i++) {
        const size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                                 h * (seq_len * head_dim_stride);

        const float *q_row = (const float *)ASSUME_ALIGNED(
            &q_aligned[bh_offset + i * head_dim_stride], ALIGNMENT);
        float *out_row = (float *)ASSUME_ALIGNED(
            &out_aligned[bh_offset + i * head_dim_stride], ALIGNMENT);

        const size_t num_keys = i + 1;

        // Thread-local tile scores
        float tile_scores[TILE_K];

        // Online softmax running statistics
        float global_max = -FLT_MAX;
        float global_sum = 0.0f;

#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] = 0.0f;
        }

        // Process keys in tiles
        for (size_t k_start = 0; k_start < num_keys; k_start += TILE_K) {
          const size_t k_end = min_size(k_start + TILE_K, num_keys);
          const size_t tile_size = k_end - k_start;

          // Step 1: Compute scores for tile, find tile max
          float tile_max = -FLT_MAX;
          for (size_t t = 0; t < tile_size; t++) {
            const size_t k = k_start + t;
            const float *k_row = (const float *)ASSUME_ALIGNED(
                &k_aligned[bh_offset + k * head_dim_stride], ALIGNMENT);

            float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
            for (size_t d = 0; d < head_dim; d++) {
              dot += q_row[d] * k_row[d];
            }

            const float score = dot * scale;
            tile_scores[t] = score;
            tile_max = score > tile_max ? score : tile_max;
          }

          // Step 2: Update global max, compute correction
          const float new_global_max =
              tile_max > global_max ? tile_max : global_max;
          const float correction = expf(global_max - new_global_max);

          // Step 3: Rescale previous output and sum
          global_sum *= correction;
#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] *= correction;
          }

          // Step 4: Compute exp for tile, accumulate sum
          float tile_sum = 0.0f;
          for (size_t t = 0; t < tile_size; t++) {
            tile_scores[t] = expf(tile_scores[t] - new_global_max);
            tile_sum += tile_scores[t];
          }
          global_sum += tile_sum;

          // Step 5: Weighted sum - unrolled by 2 for better ILP
          size_t t = 0;
          for (; t + 1 < tile_size; t += 2) {
            const float *v_row0 = (const float *)ASSUME_ALIGNED(
                &v_aligned[bh_offset + (k_start + t) * head_dim_stride],
                ALIGNMENT);
            const float *v_row1 = (const float *)ASSUME_ALIGNED(
                &v_aligned[bh_offset + (k_start + t + 1) * head_dim_stride],
                ALIGNMENT);
            const float w0 = tile_scores[t];
            const float w1 = tile_scores[t + 1];

#pragma omp simd
            for (size_t d = 0; d < head_dim; d++) {
              out_row[d] += w0 * v_row0[d] + w1 * v_row1[d];
            }
          }
          // Handle remainder
          for (; t < tile_size; t++) {
            const float *v_row = (const float *)ASSUME_ALIGNED(
                &v_aligned[bh_offset + (k_start + t) * head_dim_stride],
                ALIGNMENT);
            const float weight = tile_scores[t];
#pragma omp simd
            for (size_t d = 0; d < head_dim; d++) {
              out_row[d] += weight * v_row[d];
            }
          }

          global_max = new_global_max;
        }

        // Final normalization
        const float inv_sum = 1.0f / global_sum;
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] *= inv_sum;
        }
      }
    }
  }
}
