#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// ============================================================================
// Multi-threaded Causal Multi-Head Self-Attention - v1 (Q-Tiled)
// ============================================================================
//
// Key optimization over v0:
// - Q-tiling: process TILE_Q queries together for better K/V cache reuse
// - Each K row (Pass 1) and V row (Pass 3) loaded once per tile, not per query
//
//
// Causal masking:
// - Query at position i only attends to keys 0..i
// - q_start = max(key_pos, q_tile) ensures we skip masked positions
//
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

  // Alignment hints for compiler
  Q = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  K = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  V = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  out = (float *)ASSUME_ALIGNED(out, ALIGNMENT);
  attn_base = (float *)ASSUME_ALIGNED(attn_base, ALIGNMENT);

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Each thread gets its own scratch space for attention scores
      // (v0 uses 1 row per thread; v1 uses TILE_Q rows for tiling)
      size_t thread_id = (size_t)omp_get_thread_num();
      float *scores = attn_base + thread_id * TILE_Q * seq_len_padded;

      // Per-query stats (small arrays, stay in registers/L1)
      float row_max[TILE_Q]; // max score per query (same as v0's max_score)
      float row_sum[TILE_Q]; // sum of exp(scores) per query

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_pad) +
                         h * (seq_len * head_dim_pad);

      // =====================================================================
      // Process queries in tiles of TILE_Q
      // (v0 processes one query at a time; tiling improves K/V cache reuse)
      // =====================================================================
      for (size_t q_tile = 0; q_tile < seq_len; q_tile += TILE_Q) {
        // Track number of elmeents in the tile to avoid exceeding seq_len
        size_t q_end = (q_tile + TILE_Q < seq_len) ? q_tile + TILE_Q : seq_len;
        size_t num_queries = q_end - q_tile;

        // Initialize max trackers (same as v0's max_score = -FLT_MAX)
        for (size_t i = 0; i < num_queries; i++) {
          row_max[i] = -FLT_MAX;
        }

        // ===================================================================
        // Pass 1: Compute Q @ K^T scores for this tile (+ track max)
        //
        // Same as v0's Step 1 but processes TILE_Q queries together.
        // Key insight: iterate over keys in OUTER loop so each K row is
        // loaded once and reused for all queries in the tile.
        // ===================================================================
        size_t max_key = q_end - 1; // Last query position determines max key
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          size_t key_offset = bh_offset + key_pos * head_dim_pad;

          // First query in tile that can attend to this key (causal mask)
          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile; // Index within tile
            size_t query_offset = bh_offset + query_pos * head_dim_pad;

            // Dot product: Q[query_pos] · K[key_pos]
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
              dot += Q[query_offset + d] * K[key_offset + d];
            }

            float score = dot * scale;
            scores[i * seq_len_padded + key_pos] = score;
            row_max[i] = score > row_max[i] ? score : row_max[i];
          }
        }

        // ===================================================================
        // Pass 2: Softmax numerator - compute exp(score - max) and sum
        // ===================================================================
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          size_t num_keys = query_pos + 1;
          float *row_scores = scores + i * seq_len_padded;
          float sum = 0.0f;

          for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
            float exp_val = expf(row_scores[key_pos] - row_max[i]);
            row_scores[key_pos] = exp_val; // Overwrite score with exp
            sum += exp_val;
          }
          // Store inverse for multiplication later
          row_sum[i] = 1.0f / sum;
        }

        // ===================================================================
        // Pass 3: Compute output = softmax(scores) @ V
        // ===================================================================

        // Initialize outputs for this tile to zero
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          size_t output_offset = bh_offset + query_pos * head_dim_pad;

          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] = 0.0f;
          }
        }

        // Accumulate weighted V rows (V reuse across queries in tile)
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_pad;

          // Only accumulate for queries where this key is visible (causal)
          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile;
            size_t output_offset = bh_offset + query_pos * head_dim_pad;

            // Normalized attention weight = exp(score - max) / sum
            float weight = scores[i * seq_len_padded + key_pos] * row_sum[i];

            for (size_t d = 0; d < head_dim; d++) {
              out[output_offset + d] += weight * V[value_offset + d];
            }
          }
        }
      }
    }
  }
}
