#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// ============================================================================
// Multi-threaded Causal Multi-Head Self-Attention - v2
// ============================================================================
//
// Improvement over v1:
// - K/V cache reuse: load each K/V row once per tile, use for all queries
// - For each key position, compute scores for ALL queries in the tile that
//   need it (due to causal masking: query qi needs key kj if kj <= qi)
// - For each value position, accumulate to ALL output rows in the tile
// - Improves L1/L2 cache hit rate for K and V tensors
//
// Key insight from VTune profiling:
// - Memory Bound was 43.6% (L3 Bound 18.5%, FB Full 51.4%)
// - L1-dcache-load-misses: 27.44%
// - This optimization keeps K/V rows hot in cache across multiple queries
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

  const size_t B = dims.batch;
  const size_t H = dims.n_heads;
  const size_t S = dims.seq_len;
  const size_t D = dims.head_dim;
  const size_t D_pad = dims.head_dim_padded;
  const size_t S_pad = dims.seq_len_padded;
  const float scale = 1.0f / sqrtf((float)D);

  const float *RESTRICT Q_al = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *RESTRICT K_al = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *RESTRICT V_al = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *RESTRICT out_al = (float *)ASSUME_ALIGNED(out, ALIGNMENT);

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; b++) {
    for (size_t h = 0; h < H; h++) {

      // Each thread gets its own scratch space for TILE_Q rows of scores
      size_t thread_id = (size_t)omp_get_thread_num();
      float *RESTRICT scores = attn_base + thread_id * TILE_Q * S_pad;
      scores = (float *)ASSUME_ALIGNED(scores, ALIGNMENT);

      // Per-query stats (small arrays, stay in registers/L1)
      float row_max[TILE_Q];
      float row_sum[TILE_Q];

      const size_t bh_off = (b * H + h) * S * D_pad;

      // Process queries in tiles of TILE_Q
      for (size_t qi_base = 0; qi_base < S; qi_base += TILE_Q) {
        const size_t qi_end = (qi_base + TILE_Q < S) ? qi_base + TILE_Q : S;
        const size_t num_queries = qi_end - qi_base;

        // Initialize per-query stats
        for (size_t i = 0; i < num_queries; i++) {
          row_max[i] = -FLT_MAX;
        }

        // =================================================================
        // Pass 1: Compute scores for all queries in this tile
        // Load each K row once, compute dot product with all relevant Q rows
        // =================================================================
        const size_t max_key = qi_end - 1; // Last query needs keys 0..max_key

        for (size_t kj = 0; kj <= max_key; kj++) {
          const float *RESTRICT k_row = K_al + bh_off + kj * D_pad;
          k_row = (const float *)ASSUME_ALIGNED(k_row, ALIGNMENT);

          // Which queries in this tile need this key?
          // Causal: query qi needs key kj if kj <= qi
          const size_t qi_start = (kj > qi_base) ? kj : qi_base;

          for (size_t qi = qi_start; qi < qi_end; qi++) {
            const size_t i = qi - qi_base;
            const float *RESTRICT q_row = Q_al + bh_off + qi * D_pad;
            q_row = (const float *)ASSUME_ALIGNED(q_row, ALIGNMENT);

            float dot = 0.0f;
            for (size_t d = 0; d < D; d++) {
              dot += q_row[d] * k_row[d];
            }

            const float score = dot * scale;
            scores[i * S_pad + kj] = score;
            row_max[i] = score > row_max[i] ? score : row_max[i];
          }
        }

        // =================================================================
        // Pass 2: Softmax - compute exp and sum
        // =================================================================
        for (size_t i = 0; i < num_queries; i++) {
          const size_t qi = qi_base + i;
          const size_t num_keys = qi + 1; // Causal: keys 0..qi
          float *RESTRICT row_scores = scores + i * S_pad;
          float sum = 0.0f;

          for (size_t kj = 0; kj < num_keys; kj++) {
            const float exp_val = expf(row_scores[kj] - row_max[i]);
            row_scores[kj] = exp_val;
            sum += exp_val;
          }
          row_sum[i] = 1.0f / sum; // Store inverse for multiplication
        }

        // =================================================================
        // Pass 3: Weighted sum of V
        // Load each V row once, accumulate to all relevant output rows
        // =================================================================

        // Initialize outputs for this tile to zero
        for (size_t i = 0; i < num_queries; i++) {
          const size_t qi = qi_base + i;
          float *RESTRICT out_row = out_al + bh_off + qi * D_pad;
          out_row = (float *)ASSUME_ALIGNED(out_row, ALIGNMENT);

          for (size_t d = 0; d < D; d++) {
            out_row[d] = 0.0f;
          }
        }

        // Accumulate weighted V rows
        for (size_t kj = 0; kj <= max_key; kj++) {
          const float *RESTRICT v_row = V_al + bh_off + kj * D_pad;
          v_row = (const float *)ASSUME_ALIGNED(v_row, ALIGNMENT);

          // Which queries in this tile use this value?
          const size_t qi_start = (kj > qi_base) ? kj : qi_base;

          for (size_t qi = qi_start; qi < qi_end; qi++) {
            const size_t i = qi - qi_base;
            // Multiply by inv_sum here (fused normalization)
            const float weight = scores[i * S_pad + kj] * row_sum[i];

            float *RESTRICT out_row = out_al + bh_off + qi * D_pad;
            out_row = (float *)ASSUME_ALIGNED(out_row, ALIGNMENT);

            for (size_t d = 0; d < D; d++) {
              out_row[d] += weight * v_row[d];
            }
          }
        }
      }
    }
  }
}
