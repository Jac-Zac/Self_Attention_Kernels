#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define ALIGNED_ROW_DECL(tensor, pos, bh_offset, dim_pad)                      \
  ((__typeof__((tensor)))ASSUME_ALIGNED(                                       \
      (tensor) + (bh_offset) + (pos) * (dim_pad), ALIGNMENT))

// ============================================================================
// Multi-threaded Causal Multi-Head Self-Attention - v1 (Q-Tiled)
// ============================================================================
//
// Algorithm overview:
//   For each batch/head (parallelized):
//     For each Q_tile of TILE_Q queries:
//       Pass 1: scores[TILE_Q][seq_len] = Q_tile @ K^T * scale  (+ track max)
//       Pass 2: scores = exp(scores - max), compute row sums
//       Pass 3: out = scores @ V (normalized by row sums)
//
//   - With tiling: each K row loaded once per TILE_Q queries = seq_len/TILE_Q
//   - Same benefit for V in Pass 3
//
// Causal masking:
//   - Query at position i only attends to keys 0..i
//   - q_start = max(key_pos, q_tile) ensures we skip masked positions
// ============================================================================

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

// Each (batch, head) pair is independent - parallelize across them
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Thread-local workspace for attention scores
      // Layout: scores[query_in_tile][key_pos] = scores[i * seq_len_padded + k]
      size_t thread_id = (size_t)omp_get_thread_num();
      float *RESTRICT scores = attn_base + thread_id * TILE_Q * seq_len_padded;
      scores = (float *)ASSUME_ALIGNED(scores, ALIGNMENT);

      // Per-query stats (small arrays, stay in registers/L1)
      float row_max[TILE_Q]; // max score per query
      float row_sum[TILE_Q]; // sum of exp(scores) per query -> becomes 1/sum

      // Base offset into Q/K/V/out for this (batch, head)
      size_t bh_offset = (b * num_heads + h) * seq_len * head_dim_pad;

      // =====================================================================
      // Process queries in tiles of TILE_Q
      // =====================================================================
      for (size_t q_tile = 0; q_tile < seq_len; q_tile += TILE_Q) {
        size_t q_end = (q_tile + TILE_Q < seq_len) ? q_tile + TILE_Q : seq_len;
        size_t num_queries = q_end - q_tile;

        // Initialize max trackers
        for (size_t i = 0; i < num_queries; i++) {
          row_max[i] = -FLT_MAX;
        }

        // ===================================================================
        // Compute Q @ K^T scores for this tile
        //
        // Key insight: iterate over keys in OUTER loop so each K row is
        // loaded once and reused for all queries in the tile that need it.
        //
        // Causal mask: query i can only see keys 0..i
        //   - max_key = q_end - 1 (last query in tile)
        //   - For each key, only compute for queries where key_pos <= query_pos
        //   - q_start = max(key_pos, q_tile) skips masked entries
        // ===================================================================
        size_t max_key = q_end - 1; // Last query position determines max key
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          const float *RESTRICT k_row =
              ALIGNED_ROW_DECL(K, key_pos, bh_offset, head_dim_pad);

          // First query in tile that can attend to this key
          // (due to causal mask: key_pos <= query_pos)
          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile; // Index within tile
            const float *RESTRICT q_row =
                ALIGNED_ROW_DECL(Q, query_pos, bh_offset, head_dim_pad);

            // Dot product: Q[query_pos] · K[key_pos]
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
              dot += q_row[d] * k_row[d];
            }

            float score = dot * scale;
            scores[i * seq_len_padded + key_pos] = score;
            row_max[i] = score > row_max[i] ? score : row_max[i];
          }
        }

        // ===================================================================
        // Pass 2: Softmax numerator - compute exp(score - max) and sum
        //
        // Subtracting max prevents overflow in exp() (numerical stability)
        // We store exp values back into scores buffer for use in Pass 3
        // ===================================================================
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          size_t num_keys = query_pos + 1; // Causal: can see keys 0..query_pos
          float *RESTRICT row_scores = scores + i * seq_len_padded;
          float sum = 0.0f;

          for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
            float exp_val = expf(row_scores[key_pos] - row_max[i]);
            row_scores[key_pos] = exp_val; // Overwrite score with exp
            sum += exp_val;
          }
          row_sum[i] = 1.0f / sum; // Store inverse for multiplication later
        }

        // ===================================================================
        // Pass 3: Compute output = softmax(scores) @ V
        //
        // Key insight: iterate over keys in OUTER loop so each V row is
        // loaded once and reused for all queries in the tile.
        //
        // For each V[key_pos], accumulate: out[query] += weight * V[key_pos]
        // where weight = exp_score[key_pos] / sum = exp_score * row_sum
        // ===================================================================

        // Initialize outputs for this tile to zero
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          float *RESTRICT out_row =
              ALIGNED_ROW_DECL(out, query_pos, bh_offset, head_dim_pad);

          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] = 0.0f;
          }
        }

        // Accumulate weighted V rows (V reuse across queries in tile)
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          const float *RESTRICT v_row =
              ALIGNED_ROW_DECL(V, key_pos, bh_offset, head_dim_pad);

          // Only accumulate for queries where this key is visible (causal)
          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile;
            // Normalized attention weight = exp(score - max) / sum
            float weight = scores[i * seq_len_padded + key_pos] * row_sum[i];

            float *RESTRICT out_row =
                ALIGNED_ROW_DECL(out, query_pos, bh_offset, head_dim_pad);

            for (size_t d = 0; d < head_dim; d++) {
              out_row[d] += weight * v_row[d];
            }
          }
        }
      }
    }
  }
}
