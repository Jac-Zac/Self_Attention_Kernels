#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Causal Multi-Head Self-Attention (Optimized Single-Threaded)
 * Focus: Reducing Port 6 pressure and Front-End index overhead.
 */
void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // --- Pointer Hoisting (Solves Index Arithmetic) ---
      // Pre-calculate the base of the current [batch, head] block.
      size_t head_offset =
          (b * num_heads * seq_len * head_dim) + (h * seq_len * head_dim);

      const float *RESTRICT head_Q = ASSUME_ALIGNED(&Q[head_offset], 64);
      const float *RESTRICT head_K = ASSUME_ALIGNED(&K[head_offset], 64);
      const float *RESTRICT head_V = ASSUME_ALIGNED(&V[head_offset], 64);
      float *RESTRICT head_out = ASSUME_ALIGNED(&out[head_offset], 64);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        // q_ptr and out_ptr are fixed for the duration of the key_pos loops.
        const float *RESTRICT q_ptr =
            ASSUME_ALIGNED(&head_Q[query_pos * head_dim], 64);
        float *RESTRICT out_ptr =
            ASSUME_ALIGNED(&head_out[query_pos * head_dim], 64);

        float max_score = -FLT_MAX;

        // --- Step 1: Dot Product (QK^T) ---
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *RESTRICT k_ptr =
              ASSUME_ALIGNED(&head_K[key_pos * head_dim], 64);
          float dp = 0.0f;

          // Inner loop: vectorized math
          LOOP_VECTORIZE
          for (size_t d = 0; d < head_dim; d++) {
            dp += q_ptr[d] * k_ptr[d];
          }

          float score = dp * scale;
          attn_weights[key_pos] = score;
          max_score = (score > max_score) ? score : max_score;
        }

        // --- Step 2: Softmax (Exp & Sum) ---
        float sum_exp = 0.0f;
        LOOP_VECTORIZE
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float ev = expf(attn_weights[key_pos] - max_score);
          attn_weights[key_pos] = ev;
          sum_exp += ev;
        }
        const float inv_sum_exp = 1.0f / sum_exp;

        // --- Step 3: Weighted Sum (Output) ---
        // Zero out the output row once.
        LOOP_VECTORIZE
        for (size_t d = 0; d < head_dim; d++) {
          out_ptr[d] = 0.0f;
        }

        // --- Reduction of Port 6 Pressure ---
        // Unroll the loop that manages v_ptr. This allows the CPU to issue
        // multiple FMA instructions from different rows of V simultaneously.
        LOOP_UNROLL_N(4)
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          const float *RESTRICT v_ptr =
              ASSUME_ALIGNED(&head_V[key_pos * head_dim], 64);
          const float weight = attn_weights[key_pos] * inv_sum_exp;

          LOOP_VECTORIZE
          VECTOR_ALIGNED
          for (size_t d = 0; d < head_dim; d++) {
            // Hot loop: vfmadd231ps
            out_ptr[d] += weight * v_ptr[d];
          }
        }
      }
    }
  }
}
