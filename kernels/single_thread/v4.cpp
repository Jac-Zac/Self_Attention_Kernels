#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// v4: Flash Attention Style with Online Softmax
// ============================================================================
//
// This version implements the "online softmax" algorithm from Flash Attention,
// which allows computing attention in a single pass over the keys without
// materializing the full attention matrix.
//
// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
//            by Dao et al. (2022)
//
// ============================================================================
// COMPARISON WITH v3 (Standard 3-Pass Algorithm)
// ============================================================================
//
// v3 uses the standard approach with 3 separate passes:
//
//   Pass 1: Compute ALL scores and find global max
//   Pass 2: Compute exp(score - max) for ALL keys, accumulate sum
//   Pass 3: Compute weighted sum of V
//
// This requires storing all scores in memory (O(seq_len) per query).
//
// ============================================================================
//
//     Step 1: Compute scores for this tile, find tile_max
//     Step 2: Update global max and compute correction factor
//     Step 3: Rescale previous output and sum by correction factor
//     Step 4: Compute exp(score - new_max) for tile, accumulate sum
//     Step 5: Accumulate weighted V for this tile
//     Update: global_max = new_global_max
//   Final: Normalize output

#ifndef TILE_K
#define TILE_K 64
#endif

static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

/**
 * Causal Multi-Head Self-Attention forward pass (CPU implementation)
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * This version uses Flash Attention's online softmax algorithm,
 * processing keys in tiles and incrementally updating the output.
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {
  // Note: attn_weights workspace is not used in this version.
  // We use a small local array (tile_scores) instead.
  (void)attn_weights;

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);

  // Tell the compiler these pointers are aligned for better vectorization
  const float *q_aligned = (const float *)ASSUME_ALIGNED(Q, ALIGNMENT);
  const float *k_aligned = (const float *)ASSUME_ALIGNED(K, ALIGNMENT);
  const float *v_aligned = (const float *)ASSUME_ALIGNED(V, ALIGNMENT);
  float *out_aligned = (float *)ASSUME_ALIGNED(out, ALIGNMENT);

  // Temporary storage for scores within a tile.
  // This is small enough to stay in L1 cache.
  float tile_scores[TILE_K];

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      // Offset to the start of this (batch, head) slice
      const size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                               h * (seq_len * head_dim_stride);

      // Process each query position
      for (size_t i = 0; i < seq_len; i++) {
        const float *q_row = (const float *)ASSUME_ALIGNED(
            &q_aligned[bh_offset + i * head_dim_stride], ALIGNMENT);
        float *out_row = (float *)ASSUME_ALIGNED(
            &out_aligned[bh_offset + i * head_dim_stride], ALIGNMENT);

        // Causal mask: query at position i can only attend to keys 0..i
        const size_t num_keys = i + 1;

        // Running statistics for online softmax across all tiles
        float global_max = -FLT_MAX;
        float global_sum = 0.0f;

        // Initialize output accumulator to zero
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] = 0.0f;
        }

        // Process keys in tiles
        for (size_t k_start = 0; k_start < num_keys; k_start += TILE_K) {
          const size_t k_end = min_size(k_start + TILE_K, num_keys);
          const size_t tile_size = k_end - k_start;

          // Step 1: Compute scores for this tile and find tile maximum
          // We track the tile maximum for the online softmax update.
          float tile_max = -FLT_MAX;

          for (size_t t = 0; t < tile_size; t++) {
            const size_t k = k_start + t;
            const float *k_row = (const float *)ASSUME_ALIGNED(
                &k_aligned[bh_offset + k * head_dim_stride], ALIGNMENT);

            float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
            for (size_t d = 0; d < head_dim; d++) {
              dot += q_row[d] * k_row[d];
            }

            const float score = dot * scale;
            tile_scores[t] = score;
            tile_max = score > tile_max ? score : tile_max;
          }

          // Step 2: Compute new global maximum and correction factor
          // If this tile has a larger max than what we've seen before,
          // we need to rescale all previous computations.
          const float new_global_max =
              tile_max > global_max ? tile_max : global_max;
          const float correction = expf(global_max - new_global_max);

          // Step 3: Rescale previous output and sum by correction factor
          global_sum *= correction;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out_row[d] *= correction;
          }

          // Step 4: Compute exp(score - max) for this tile, accumulate sum
          // Now compute softmax weights for this tile using the new global max
          float tile_sum = 0.0f;

          for (size_t t = 0; t < tile_size; t++) {
            tile_scores[t] = expf(tile_scores[t] - new_global_max);
            tile_sum += tile_scores[t];
          }
          global_sum += tile_sum;

          // Step 5: Accumulate weighted sum of V for this tile
          // Note: We accumulate UNNORMALIZED weights here.
          // Final normalization happens after all tiles are processed.
          for (size_t t = 0; t < tile_size; t++) {
            const size_t k = k_start + t;
            const float *v_row = (const float *)ASSUME_ALIGNED(
                &v_aligned[bh_offset + k * head_dim_stride], ALIGNMENT);
            const float weight = tile_scores[t];

#pragma omp simd
            for (size_t d = 0; d < head_dim; d++) {
              out_row[d] += weight * v_row[d];
            }
          }

          // Update global max for next tile
          global_max = new_global_max;
        }

        // Final normalization: divide by sum of all weights
        const float inv_sum = 1.0f / global_sum;

#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_row[d] *= inv_sum;
        }
      }
    }
  }
}
