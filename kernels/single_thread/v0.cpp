#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

/**
 * Causal Multi-Head Self-Attention forward pass (CPU implementation) - v0
 *
 * This is the baseline/reference implementation with minimal optimizations.
 * It follows the standard attention algorithm directly:
 * 1. Compute all QK^T scores (including masked positions)
 * 2. Apply causal mask by setting future positions to -inf
 * 3. Compute softmax using numerically stable two-pass algorithm
 * 4. Explicitly zero out masked positions after softmax
 * 5. Compute weighted sum of values
 *
 * This version is intentionally simple and readable, serving as a reference
 * for correctness. Later versions optimize by avoiding masked computations.
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * @param Q           Query tensor [B, H, S, D]
 * @param K           Key tensor [B, H, S, D]
 * @param V           Value tensor [B, H, S, D]
 * @param out         Output tensor [B, H, S, D]
 * @param attn_weights Workspace base [threads*seq_len_padded]
 * @param dims        Attention dimensions (batch, heads, seq_len, head_dim)
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);

  // Process each batch and head independently
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      float *aw = attn_weights;

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      // Process each query position
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim_stride;

        // Step 1: Compute scaled dot-product attention scores for all positions
        for (size_t key_pos = 0; key_pos < seq_len; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim_stride;

          // Dot product across head dimension
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          aw[key_pos] = dot_product * scale;
        }

        // Apply causal mask: future positions get -inf (zeroed after softmax)
        for (size_t key_pos = query_pos + 1; key_pos < seq_len; key_pos++) {
          aw[key_pos] = -FLT_MAX;
        }

        // Step 2: Numerically stable softmax over full seq_len
        float max_score = -FLT_MAX;
        for (size_t key_pos = 0; key_pos < seq_len; key_pos++) {
          if (aw[key_pos] > max_score)
            max_score = aw[key_pos];
        }

        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos < seq_len; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Normalize to get probabilities
        for (size_t key_pos = 0; key_pos < seq_len; key_pos++) {
          aw[key_pos] /= sum_exp;
        }

        // Explicitly zero out masked positions (already -inf -> exp -> 0)
        for (size_t key_pos = query_pos + 1; key_pos < seq_len; key_pos++) {
          aw[key_pos] = 0.0f;
        }

        // Step 3: Weighted sum of values
        size_t output_offset = bh_offset + query_pos * head_dim_stride;

        for (size_t d = 0; d < head_dim; d++) {
          float weighted_sum = 0.0f;

          // Accumulate: sum over key positions of (attention_weight * value)
          for (size_t key_pos = 0; key_pos < seq_len; key_pos++) {
            size_t value_offset = bh_offset + key_pos * head_dim_stride;
            weighted_sum += aw[key_pos] * V[value_offset + d];
          }

          out[output_offset + d] = weighted_sum;
        }
      }
    }
  }
}
