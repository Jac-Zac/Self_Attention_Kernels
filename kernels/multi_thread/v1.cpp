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
 * @param attn_base Workspace base [threads*seq_len_padded] (multi) or
 * [seq_len_padded] (single)
 * @param dims      Attention dimensions (batch, heads, seq_len, head_dim)
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_base, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // Padded strides for vectorization and alignment
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(seq_len, VEC_PADDING);

  // Assume base pointers are aligned
  const float *Qa = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *Ka = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *Va = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *Outa = (float *)ASSUME_ALIGNED(out, ALIGNMENT);

  // Process each batch and head independently
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      // Per-thread scratch slice
      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw = (float *)ASSUME_ALIGNED(
          attn_base + thread_id * seq_len_padded, ALIGNMENT);

      // Base offset for current (b,h)
      const size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                               h * (seq_len * head_dim_stride);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        const size_t query_offset = bh_offset + query_pos * head_dim_stride;

        // Step 1: Scaled dot-product attention scores (causal)
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          const size_t key_offset = bh_offset + key_pos * head_dim_stride;

#pragma omp simd reduction(+ : dot_product)
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Qa[query_offset + d] * Ka[key_offset + d];
          }
          aw[key_pos] = dot_product * scale;
        }

        // Step 2: Numerically stable softmax
        float max_score = -FLT_MAX;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          if (aw[key_pos] > max_score)
            max_score = aw[key_pos];
        }

        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Step 3: Weighted sum of values
        const size_t output_offset = bh_offset + query_pos * head_dim_stride;
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          Outa[output_offset + d] = 0.0f;
        }

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const size_t value_offset = bh_offset + key_pos * head_dim_stride;
          const float attn_weight = aw[key_pos] / sum_exp;
#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            Outa[output_offset + d] += attn_weight * Va[value_offset + d];
          }
        }
      }
    }
  }
}
