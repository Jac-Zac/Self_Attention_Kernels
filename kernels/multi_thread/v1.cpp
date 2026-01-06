#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// ============================================================================
// Multi-threaded Causal Multi-Head Self-Attention - v1
// ============================================================================
//
// Improvement over v0:
// - Query tiling: process queries in blocks of TILE_Q
// - Changes parallelization from collapse(3) to collapse(2)
// - Each thread processes all queries in a (batch, head) pair in tiles
// - Prepares structure for K/V cache reuse optimization in v2
//
// The algorithm per query remains identical to v0 (3-pass softmax).
// The only change is the loop structure for processing multiple queries.
//
// Parallelization: collapse(2) over batch x heads, sequential tiles over queries
// Workspace: uses shared attn_base (thread_id * TILE_Q * seq_len_padded)
// ============================================================================

// Number of queries processed per tile
// Workspace per thread: TILE_Q * seq_len_padded floats
#ifndef TILE_Q
#define TILE_Q 8
#endif

/**
 * Causal Multi-Head Self-Attention forward pass (multi-threaded CPU)
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * @param Q         Query tensor [B, H, S, D] (row-major, D padded)
 * @param K         Key tensor [B, H, S, D]
 * @param V         Value tensor [B, H, S, D]
 * @param out       Output tensor [B, H, S, D]
 * @param attn_base Workspace [threads * TILE_Q * seq_len_padded]
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

// Parallelize over batch x heads (each thread handles all queries for one head)
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Each thread gets its own scratch space for TILE_Q rows of attention weights
      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw_base = attn_base + thread_id * TILE_Q * seq_len_padded;

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_pad) +
                         h * (seq_len * head_dim_pad);

      // Process queries in tiles of TILE_Q
      for (size_t qi_base = 0; qi_base < seq_len; qi_base += TILE_Q) {
        size_t qi_end = (qi_base + TILE_Q < seq_len) ? qi_base + TILE_Q : seq_len;

        // Process each query in this tile (same algorithm as v0)
        for (size_t query_pos = qi_base; query_pos < qi_end; query_pos++) {
          size_t tile_idx = query_pos - qi_base;
          float *aw = aw_base + tile_idx * seq_len_padded;

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

          // Step 2: Numerically stable softmax
          // exp(score - max) prevents overflow
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
}
