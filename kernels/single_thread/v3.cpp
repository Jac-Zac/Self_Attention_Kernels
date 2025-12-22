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
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  size_t batch_size = dims.batch;
  size_t num_heads = dims.n_heads;
  size_t seq_len = dims.seq_len;
  size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // Tell the compiler these pointers are aligned to the system's SIMD width
  // Assuming ALIGNMENT is 64 for AVX-512
  const float *q_aligned = (const float *)ASSUME_ALIGNED(Q, 64);
  const float *k_aligned = (const float *)ASSUME_ALIGNED(K, 64);
  const float *v_aligned = (const float *)ASSUME_ALIGNED(V, 64);
  float *out_aligned = (float *)ASSUME_ALIGNED(out, 64);

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      size_t bh_offset =
          b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        const float *q_row = &q_aligned[bh_offset + query_pos * head_dim];
        float max_score = -FLT_MAX;

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *k_row = &k_aligned[bh_offset + key_pos * head_dim];
          float dot_product = 0.0f;

// Help compiler recognize simple stride-1 access
#pragma omp simd reduction(+ : dot_product)
          LOOP_VECTORIZE
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += q_row[d] * k_row[d];
          }

          float score = dot_product * scale;
          max_score = score > max_score ? score : max_score;
          attn_weights[key_pos] = score;
        }

        // Softmax Exponentiation
        float sum_exp = 0.0f;
#pragma omp simd reduction(+ : sum_exp)
        LOOP_VECTORIZE
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(attn_weights[key_pos] - max_score);
          attn_weights[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        const float inv_sum_exp = 1.0f / sum_exp;
        float *out_row = &out_aligned[bh_offset + query_pos * head_dim];

// Zero out output row
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] = 0.0f;
        }

        // Weighted Sum (The most compute-intensive part)
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *v_row = &v_aligned[bh_offset + key_pos * head_dim];
          float weight = attn_weights[key_pos] * inv_sum_exp;

#pragma omp simd
          LOOP_VECTORIZE
          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] += weight * v_row[d];
          }
        }
      }
    }
  }
}
