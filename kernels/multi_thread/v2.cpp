#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// Default tile size (can be overridden via -DTILE_Q=N)
#ifndef TILE_Q
#define TILE_Q 32
#endif

// ============================================================================
// Multi-threaded Causal Multi-Head Self-Attention - v2 (Q-Tiled, padded
// scratch)
// ============================================================================
//
// Same algorithm as multi_thread/v1.cpp, but fixes potential false sharing in
// the shared scratch buffer by padding each thread's slice to a cache line.
//
// Scratch layout per thread:
//   scores[TILE_Q][seq_len_padded]
// Thread i uses a slice starting at:
//   attn_base + i * round_up_pow2(TILE_Q * seq_len_padded, CACHELINE_FLOATS)
//
// ============================================================================

static inline size_t cacheline_floats() { return ALIGNMENT / sizeof(float); }

size_t cmhsa_get_workspace_size_cpu(const AttentionDims dims, int threads) {
  const size_t seq_len_padded = dims.seq_len_padded;
  const size_t per_thread_raw = TILE_Q * seq_len_padded;
  const size_t per_thread_stride =
      round_up_pow2(per_thread_raw, cacheline_floats());
  return (size_t)threads * per_thread_stride * sizeof(float);
}

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

  const size_t per_thread_raw = TILE_Q * seq_len_padded;
  const size_t per_thread_stride =
      round_up_pow2(per_thread_raw, cacheline_floats());

// Each (batch, head) pair is independent - parallelize across them
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Thread-local workspace for attention scores
      // Layout: scores[query_in_tile][key_pos]
      size_t thread_id = (size_t)omp_get_thread_num();
      float *RESTRICT scores = attn_base + thread_id * per_thread_stride;
      scores = (float *)ASSUME_ALIGNED(scores, ALIGNMENT);

      // Per-query stats (small arrays, stay in registers/L1)
      float row_max[TILE_Q]; // max score per query
      float row_sum[TILE_Q]; // sum of exp(scores) per query -> becomes 1/sum

      // Base offset into Q/K/V/out for this (batch, head)
      size_t bh_offset = (b * num_heads + h) * seq_len * head_dim_pad;

      // Base pointers for this (batch, head)
      const float *RESTRICT Q_bh = ASSUME_ALIGNED_FLOAT(Q + bh_offset);
      const float *RESTRICT K_bh = ASSUME_ALIGNED_FLOAT(K + bh_offset);
      const float *RESTRICT V_bh = ASSUME_ALIGNED_FLOAT(V + bh_offset);
      float *RESTRICT out_bh = ASSUME_ALIGNED_FLOAT(out + bh_offset);

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
        // Pass 1: Compute Q @ K^T scores for this tile
        // ===================================================================
        size_t max_key = q_end - 1;
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          const float *RESTRICT k_row =
              ASSUME_ALIGNED_FLOAT(K_bh + key_pos * head_dim_pad);

          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile;
            const float *RESTRICT q_row =
                ASSUME_ALIGNED_FLOAT(Q_bh + query_pos * head_dim_pad);

            // Dot product: Q @ K^T
            float score = 0.0f;
            for (size_t d = 0; d < head_dim; d++) {
              score += q_row[d] * k_row[d];
            }
            score *= scale;

            scores[i * seq_len_padded + key_pos] = score;
            row_max[i] = score > row_max[i] ? score : row_max[i];
          }
        }

        // ===================================================================
        // Pass 2: Numerically stable softmax
        // ===================================================================
        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          size_t num_keys = query_pos + 1;
          float *RESTRICT row_scores =
              ASSUME_ALIGNED_FLOAT(scores + i * seq_len_padded);

          float sum = 0.0f;
          for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
            float exp_val = expf(row_scores[key_pos] - row_max[i]);
            row_scores[key_pos] = exp_val;
            sum += exp_val;
          }
          row_sum[i] = 1.0f / sum;
        }

        for (size_t i = 0; i < num_queries; i++) {
          size_t query_pos = q_tile + i;
          float *RESTRICT out_row =
              ASSUME_ALIGNED_FLOAT(out_bh + query_pos * head_dim_pad);

          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] = 0.0f;
          }
        }

        // ===================================================================
        // Pass 3: Weighted V accumulation
        // ===================================================================
        for (size_t key_pos = 0; key_pos <= max_key; key_pos++) {
          const float *RESTRICT v_row =
              ASSUME_ALIGNED_FLOAT(V_bh + key_pos * head_dim_pad);
          size_t q_start = (key_pos > q_tile) ? key_pos : q_tile;

          for (size_t query_pos = q_start; query_pos < q_end; query_pos++) {
            size_t i = query_pos - q_tile;
            float weight = scores[i * seq_len_padded + key_pos] * row_sum[i];

            float *RESTRICT out_row =
                ASSUME_ALIGNED_FLOAT(out_bh + query_pos * head_dim_pad);

            // Weighted accumulation: out += weight * V
            for (size_t d = 0; d < head_dim; d++) {
              out_row[d] += weight * v_row[d];
            }
          }
        }
      }
    }
  }
}
