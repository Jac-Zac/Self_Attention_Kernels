#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Causal Multi-Head Self-Attention forward pass (CPU implementation)
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * Uses padded head-dim stride for vectorization friendliness and a
 * per-(batch,head) scratch slice carved from the attn_weights workspace.
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  size_t batch_size = dims.batch;
  size_t num_heads = dims.n_heads;
  size_t seq_len = dims.seq_len;
  size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(seq_len, VEC_PADDING);

  // Tell the compiler these pointers are aligned
  const float *q_aligned = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *k_aligned = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *v_aligned = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *out_aligned = (float *)ASSUME_ALIGNED(out, ALIGNMENT);

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      float *aw = (float *)ASSUME_ALIGNED(
          attn_weights + (b * num_heads + h) * seq_len_padded, ALIGNMENT);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        const float *q_row = (const float *)ASSUME_ALIGNED(
            &q_aligned[bh_offset + query_pos * head_dim_stride], ALIGNMENT);
        float max_score = -FLT_MAX;

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *k_row = (const float *)ASSUME_ALIGNED(
              &k_aligned[bh_offset + key_pos * head_dim_stride], ALIGNMENT);
          float dot_product = 0.0f;

#pragma omp simd aligned(q_row, k_row : 64) reduction(+ : dot_product)
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
        float *out_row = (float *)ASSUME_ALIGNED(
            &out_aligned[bh_offset + query_pos * head_dim_stride], ALIGNMENT);

#pragma omp simd aligned(out_row : 64)
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] = 0.0f;
        }

        // Weighted Sum
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *v_row = (const float *)ASSUME_ALIGNED(
              &v_aligned[bh_offset + key_pos * head_dim_stride], ALIGNMENT);
          float weight = aw[key_pos] * inv_sum_exp;

#pragma omp simd aligned(out_row, v_row : 64)
          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] += weight * v_row[d];
          }
        }
      }
    }
  }
}
