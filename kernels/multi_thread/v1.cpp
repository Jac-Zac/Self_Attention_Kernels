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
// - K/V cache reuse: load each K/V row once per tile, use for all queries
// - For each key position, compute scores for ALL queries in the tile that
//   need it (due to causal masking: query at position i needs key j if j <= i)
// - For each value position, accumulate to ALL output rows in the tile
// - Improves L1/L2 cache hit rate for K and V tensors
//
// Parallelization: collapse(2) over batch x heads, sequential tiles over
// queries Workspace: uses shared attn_base (thread_id * TILE_Q *
// seq_len_padded)
// ============================================================================

#ifndef TILE_Q
#define TILE_Q 8
#endif

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_base, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const size_t seq_len_padded = dims.seq_len_padded;
  const float scale = 1.0f / sqrtf((float)head_dim);

  const float *RESTRICT Q_al = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *RESTRICT K_al = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *RESTRICT V_al = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *RESTRICT out_al = (float *)ASSUME_ALIGNED(out, ALIGNMENT);

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Each thread gets its own scratch space for TILE_Q rows of scores
      size_t thread_id = (size_t)omp_get_thread_num();
      float *RESTRICT scores = attn_base + thread_id * TILE_Q * seq_len_padded;
      scores = (float *)ASSUME_ALIGNED(scores, ALIGNMENT);

      // Per-query stats (small arrays, stay in registers/L1)
      float row_max[TILE_Q];
      float row_sum[TILE_Q];

      size_t bh_offset = (b * num_heads + h) * seq_len * head_dim_pad;

      // Process queries in tiles of TILE_Q
      for (size_t q_tile = 0; q_tile < seq_len; q_tile += TILE_Q) {
        size_t q_end = (q_tile + TILE_Q < seq_len) ? q_tile + TILE_Q : seq_len;
        size_t num_queries = q_end - q_tile;

        for (size_t i = 0; i < num_queries; i++) {
          row_max[i] = -FLT_MAX;
        }

        // =================================================================
        // Pass 1: Compute scores for all queries in this tile
        // Load each K row once, compute dot product with all relevant Q rows
        // =================================================================
        size_t max_key = q_end - 1;

        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          const float *RESTRICT k_row =
              K_al + bh_offset + key_pos * head_dim_pad;
          k_row = (const float *)ASSUME_ALIGNED(k_row, ALIGNMENT);

          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile;
            const float *RESTRICT q_row =
                Q_al + bh_offset + query_pos * head_dim_pad;
            q_row = (const float *)ASSUME_ALIGNED(q_row, ALIGNMENT);

            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
              dot += q_row[d] * k_row[d];
            }

            float score = dot * scale;
            scores[i * seq_len_padded + key_pos] = score;
            row_max[i] = score > row_max[i] ? score : row_max[i];
          }
        }

        // =================================================================
        // Pass 2: Softmax - compute exp and sum
        // =================================================================
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          size_t num_keys = query_pos + 1;
          float *RESTRICT row_scores = scores + i * seq_len_padded;
          float sum = 0.0f;

          for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
            float exp_val = expf(row_scores[key_pos] - row_max[i]);
            row_scores[key_pos] = exp_val;
            sum += exp_val;
          }
          row_sum[i] = 1.0f / sum;
        }

        // =================================================================
        // Pass 3: Weighted sum of V
        // Load each V row once, accumulate to all relevant output rows
        // =================================================================

        // Initialize outputs for this tile to zero
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          float *RESTRICT out_row =
              out_al + bh_offset + query_pos * head_dim_pad;
          out_row = (float *)ASSUME_ALIGNED(out_row, ALIGNMENT);

          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] = 0.0f;
          }
        }

        // Accumulate weighted V rows
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          const float *RESTRICT v_row =
              V_al + bh_offset + key_pos * head_dim_pad;
          v_row = (const float *)ASSUME_ALIGNED(v_row, ALIGNMENT);

          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile;
            float weight = scores[i * seq_len_padded + key_pos] * row_sum[i];

            float *RESTRICT out_row =
                out_al + bh_offset + query_pos * head_dim_pad;
            out_row = (float *)ASSUME_ALIGNED(out_row, ALIGNMENT);

            for (size_t d = 0; d < head_dim; d++) {
              out_row[d] += weight * v_row[d];
            }
          }
        }
      }
    }
  }
}
