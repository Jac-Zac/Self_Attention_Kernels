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
 * @param Q         Query tensor [B, H, S, D]
 * @param K         Key tensor [B, H, S, D]
 * @param V         Value tensor [B, H, S, D]
 * @param out       Output tensor [B, H, S, D]
 * @param attn_weights Output buffer [seq_len] for intermediate things
 * @param dims      Attention dimensions (batch, heads, seq_len, head_dim)
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT buffer, const AttentionDims dims) {

  (void)buffer;

  size_t batch_size = dims.batch;
  size_t num_heads = dims.n_heads;
  size_t seq_len = dims.seq_len;
  size_t head_dim = dims.head_dim;
  const float scale = 1 / sqrtf(head_dim);

#pragma omp parallel
  {
    float *attn_weights = (float *)alloca(seq_len * sizeof(float));

// Process each batch and head independently
#pragma omp for collapse(2)
    for (size_t b = 0; b < batch_size; b++) {
      for (size_t h = 0; h < num_heads; h++) {
        for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
          // Base offset for current batch and head: [b, h, :, :]
          size_t bh_offset =
              b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim);

          size_t query_offset = bh_offset + query_pos * head_dim;

          // NOTE: We can directly keep track of the max score inside when doing
          // the dot_product computation no need to do it later
          float max_score = -INFINITY;
          for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
            float dot_product = 0.0f;
            size_t key_offset = bh_offset + key_pos * head_dim;

            // Dot product across head dimension
#pragma omp simd reduction(+ : dot_product)
            for (size_t d = 0; d < head_dim; d++) {
              dot_product += Q[query_offset + d] * K[key_offset + d];
            }

            float score = dot_product * scale;
            max_score = score > max_score ? score : max_score;
            attn_weights[key_pos] = score;
          }

          // Compute exp(score - max) and accumulate sum
          float sum_exp = 0.0f;

#pragma omp simd
          for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
            float exp_val = expf(attn_weights[key_pos] - max_score);

            attn_weights[key_pos] = exp_val;
            sum_exp += exp_val;
          }

          // Step 3: Weighted sum of values
          size_t output_offset = bh_offset + query_pos * head_dim;
          // Pre-compute reciprocal
          const float inv_sum_exp = 1.0f / sum_exp;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            // Adding this initialization since now we are accumulating and want
            // to be sure that this memory is properly set to zero
            out[output_offset + d] = 0.0f;
          }

          // Step 3: Weighted Sum (Normalization merged here)
          for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
            size_t value_offset = bh_offset + key_pos * head_dim;
            // Merge normalization into the weight also no need for expensive
            // division every time we precompute it before
            float normalized_weight = attn_weights[key_pos] * inv_sum_exp;

#pragma omp simd
            for (size_t d = 0; d < head_dim; d++) {
              out[output_offset + d] += normalized_weight * V[value_offset + d];
            }
          }
        }
      }
    }
  }
}
