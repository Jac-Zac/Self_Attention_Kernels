#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE: This version fuses max-finding with score computation to reduce passes
// over the data. However, this can actually limit instruction-level parallelism
// because the max comparison depends on each score computation, creating a
// data dependency chain. Compare with v1 which separates these operations.

/**
 * Causal Multi-Head Self-Attention forward pass (CPU implementation)
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_pad = round_up_pow2(head_dim, VEC_PADDING);

  // Process each batch and head independently
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      float *aw = (float *)ASSUME_ALIGNED(attn_weights, ALIGNMENT);

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_pad) +
                         h * (seq_len * head_dim_pad);

      // Process each query position
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim_pad;

        // Track max score while computing dot products
        float max_score = -FLT_MAX;

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim_pad;

#pragma omp simd reduction(+ : dot_product)
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          float score = dot_product * scale;
          max_score = score > max_score ? score : max_score;
          aw[key_pos] = score;
        }

        // Compute exp(score - max) and accumulate sum
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Weighted sum
        size_t output_offset = bh_offset + query_pos * head_dim_pad;
        const float inv_sum_exp = 1.0f / sum_exp;

        for (size_t d = 0; d < head_dim; d++) {
          out[output_offset + d] = 0.0f;
        }

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_pad;
          float normalized_weight = aw[key_pos] * inv_sum_exp;

          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += normalized_weight * V[value_offset + d];
          }
        }
      }
    }
  }
}
