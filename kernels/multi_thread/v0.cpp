#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// ============================================================================
// Multi-threaded Causal Multi-Head Self-Attention - v0
// ============================================================================
//
// Parallelization strategy:
// - Uses OpenMP collapse(2) to parallelize over batch × heads dimensions
// - With B=4, H=8 this gives 32 independent work units for good thread
//   utilization
// - Each thread gets its own scratch space for attention weights, indexed by
//   thread_id
//
// Algorithm:
// - Fuses max-finding with score computation (one fewer pass over attention
//   weights)
// - Uses multiplication by inverse instead of division for normalization
// - Respects causal mask: query at position i only attends to keys 0..i
//
// Memory layout:
// - All tensors: [batch, n_heads, seq_len, head_dim] (row-major)
// - head_dim_pad is padded to VEC_PADDING (16 floats) for vectorization
// - Workspace: threads * seq_len_padded floats (one row per thread)
//
// ============================================================================

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
  const size_t head_dim_pad = dims.head_dim_padded;
  const size_t seq_len_padded = dims.seq_len_padded;

// Parallelize over batch × heads (collapse gives more work units)
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      // Each thread gets its own scratch space for attention weights
      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw = attn_base + thread_id * seq_len_padded;

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_pad) +
                         h * (seq_len * head_dim_pad);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {

        size_t query_offset = bh_offset + query_pos * head_dim_pad;

        // Track max score while computing dot products (fused pass)
        float max_score = -FLT_MAX;

        // Step 1: Compute scaled dot-product scores + track max
        // Only compute for key_pos <= query_pos (causal mask)
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim_pad;

          // Dot product: Q[query_pos] . K[key_pos]
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          float score = dot_product * scale;
          max_score = score > max_score ? score : max_score;
          aw[key_pos] = score;
        }

        // Step 2: Numerically stable softmax exp(score - max) prevents overflow
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Step 3: Weighted sum of values
        // Normalize inline using multiplication by inverse
        size_t output_offset = bh_offset + query_pos * head_dim_pad;
        const float inv_sum_exp = 1.0f / sum_exp;

        // Initialize output for this query position
        for (size_t d = 0; d < head_dim; d++) {
          out[output_offset + d] = 0.0f;
        }

        // Accumulate: out[query_pos] = sum(softmax[key_pos] * V[key_pos])
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
