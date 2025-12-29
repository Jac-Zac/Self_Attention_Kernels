#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// NOTE:
// Changes from v0:
//   - Added #pragma omp simd on all inner loops
//   - Added alignment hints for compiler optimization
//   - Padded exp loop to SIMD boundary (avoid scalar expf)

/**
 * Causal Multi-Head Self-Attention forward pass (multi-threaded CPU)
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * @param Q         Query tensor [B, H, S, D] (row-major, D padded)
 * @param K         Key tensor [B, H, S, D]
 * @param V         Value tensor [B, H, S, D]
 * @param out       Output tensor [B, H, S, D]
 * @param attn_base Workspace for attention weights [threads * seq_len_padded]
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
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(seq_len, VEC_PADDING);

  // Alignment hints for input pointers (helps compiler optimization)
  const float *RESTRICT Q_aligned = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *RESTRICT K_aligned = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *RESTRICT V_aligned = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *RESTRICT out_aligned = (float *)ASSUME_ALIGNED(out, ALIGNMENT);
  float *RESTRICT attn_aligned = (float *)ASSUME_ALIGNED(attn_base, ALIGNMENT);

// Parallelize over batch × heads × query_pos
#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {

        // Each thread gets its own scratch space for attention weights
        const size_t thread_id = (size_t)omp_get_thread_num();
        float *RESTRICT aw = attn_aligned + thread_id * seq_len_padded;
        aw = (float *)ASSUME_ALIGNED(aw, ALIGNMENT);

        // Base offset for current batch and head: [b, h, :, :]
        const size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                                 h * (seq_len * head_dim_stride);

        const size_t query_offset = bh_offset + query_pos * head_dim_stride;

        // Aligned pointer to query row
        const float *RESTRICT q_row = Q_aligned + query_offset;
        q_row = (const float *)ASSUME_ALIGNED(q_row, ALIGNMENT);

        // =====================================================================
        // Step 1: Compute scaled dot-product scores + track max (fused)
        // Only compute for key_pos <= query_pos (causal mask)
        // =====================================================================
        float max_score = -FLT_MAX;

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const size_t key_offset = bh_offset + key_pos * head_dim_stride;
          const float *RESTRICT k_row = K_aligned + key_offset;
          k_row = (const float *)ASSUME_ALIGNED(k_row, ALIGNMENT);

          // Vectorized dot product: Q[query_pos] . K[key_pos]
          float dot_product = 0.0f;
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += q_row[d] * k_row[d];
          }

          const float score = dot_product * scale;
          max_score = score > max_score ? score : max_score;
          aw[key_pos] = score;
        }

        // =====================================================================
        // Step 2: Numerically stable softmax with padded vectorization
        // Pad to SIMD boundary with -FLT_MAX so exp() gives 0
        // =====================================================================

        // Round up to next multiple of 16 for full AVX-512 vectorization
        const size_t valid_count = query_pos + 1;
        size_t exp_count = (valid_count + 15) & ~(size_t)15;
        if (exp_count > seq_len_padded) {
          exp_count = seq_len_padded;
        }

        // Pad remaining slots with -FLT_MAX (exp(-inf) = 0, won't affect sum)
        for (size_t k = valid_count; k < exp_count; k++) {
          aw[k] = -FLT_MAX;
        }

        // Vectorized exp over full SIMD-aligned range (no scalar cleanup)
        for (size_t k = 0; k < exp_count; k++) {
          aw[k] = expf(aw[k] - max_score);
        }

        // Sum only the valid entries
        float sum_exp = 0.0f;
        for (size_t k = 0; k < valid_count; k++) {
          sum_exp += aw[k];
        }

        // =====================================================================
        // Step 3: Weighted sum of values
        // =====================================================================
        const size_t output_offset = bh_offset + query_pos * head_dim_stride;
        const float inv_sum_exp = 1.0f / sum_exp;

        float *RESTRICT out_row = out_aligned + output_offset;
        out_row = (float *)ASSUME_ALIGNED(out_row, ALIGNMENT);

        // Initialize output to zero
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] = 0.0f;
        }

        // Accumulate weighted values
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const size_t value_offset = bh_offset + key_pos * head_dim_stride;
          const float *RESTRICT v_row = V_aligned + value_offset;
          v_row = (const float *)ASSUME_ALIGNED(v_row, ALIGNMENT);

          const float normalized_weight = aw[key_pos] * inv_sum_exp;

          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] += normalized_weight * v_row[d];
          }
        }
      }
    }
  }
}
