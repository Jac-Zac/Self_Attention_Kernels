#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE:
// This version improves on the v0 since it doesn't waste time computing things
// that will be zero out by the mask moreover it changes the order of
// computation when computing attention weights times values

/**
 * Causal Multi-Head Self-Attention forward pass (CPU implementation)
 *
 * Computes: out = softmax(Q K^T / sqrt(d)) V with causal masking
 *
 * @param Q         Query tensor [B, H, S, D]
 * @param K         Key tensor [B, H, S, D]
 * @param V         Value tensor [B, H, S, D]
 * @param out       Output tensor [B, H, S, D]
 * @param attn_weights Output buffer [seq_len] for intermediate things
 * @param dims      Attention dimensions (batch, heads, seq_len, head_dim)
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  size_t batch_size = dims.batch;
  size_t num_heads = dims.n_heads;
  size_t seq_len = dims.seq_len;
  size_t head_dim = dims.head_dim;
  const float scale = 1 / sqrtf(head_dim);

  // Process each batch and head independently
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset =
          b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim);

      // Process each query position
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim;

        // NOTE: No need to compute it for key_pos > query_pos
        // Step 1: Compute scaled dot-product attention scores
        // QK^T for all valid (non-causal-masked) key positions
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim;

          // Dot product across head dimension
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          attn_weights[key_pos] = dot_product * scale;
        }

        // NOTE: Those part will not be used anymore
        // Apply causal mask: future positions get -inf (zeroed after softmax)
        // for (size_t key_pos = query_pos + 1; key_pos < seq_len; key_pos++) {
        //   attn_weights[key_pos] = -FLT_MAX;
        // }

        // Step 2: Numerically stable softmax
        // Find max for numerical stability (log-sum-exp trick)
        float max_score = -FLT_MAX;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          if (attn_weights[key_pos] > max_score)
            max_score = attn_weights[key_pos];
        }

        // Compute exp(score - max) and accumulate sum
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(attn_weights[key_pos] - max_score);
          // float exp_val = exp(attn_weights[key_pos] - max_score);

          attn_weights[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Normalize to get probabilities
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          attn_weights[key_pos] /= sum_exp;
        }

        // NOTE: No need for this anymore
        // Explicitly zero out masked positions (already -inf -> exp -> 0)
        // for (size_t key_pos = query_pos + 1; key_pos < seq_len; key_pos++) {
        //   attn_weights[key_pos] = 0.0f;
        // }

        // Step 3: Weighted sum of values
        size_t output_offset = bh_offset + query_pos * head_dim;

        // NOTE: Alternative version with better memory acess pattern
        // Initialize output to zero first since memory is just allocated with
        // malloc not set to zero also with calloc
        //
        for (size_t d = 0; d < head_dim; d++) {
          // Adding this initialization since now we are accumulating and want
          // to be sure that this memory is properly set to zero
          out[output_offset + d] = 0.0f;
        }

        // Accumulate: now d is inner loop
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim;
          float attn_weight = attn_weights[key_pos]; // Load once

          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += attn_weight * V[value_offset + d];
          }
        }
      }
    }
  }
}
