#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Causal Multi-Head Self-Attention forward pass (CPU implementation) - v3
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * Key optimizations in v3:
 * - Uses #pragma omp simd for better auto-vectorization
 * - Fuses max-finding with score computation (like v2)
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      size_t bh_offset =
          b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim);

      float *aw = attn_weights;

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        const float *q_row = &Q[bh_offset + query_pos * head_dim];
        float max_score = -FLT_MAX;

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *k_row = &K[bh_offset + key_pos * head_dim];
          float dot_product = 0.0f;

#pragma omp simd reduction(+ : dot_product)
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += q_row[d] * k_row[d];
          }

          float score = dot_product * scale;
          max_score = score > max_score ? score : max_score;
          aw[key_pos] = score;
        }

        // Softmax Exponentiation
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        const float inv_sum_exp = 1.0f / sum_exp;
        float *out_row = &out[bh_offset + query_pos * head_dim];

#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] = 0.0f;
        }

        // Weighted Sum
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *v_row = &V[bh_offset + key_pos * head_dim];
          float weight = aw[key_pos] * inv_sum_exp;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] += weight * v_row[d];
          }
        }
      }
    }
  }
}
