#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// NOTE:
// Improved multi-threaded version with collapse(2) to parallelize over
// batch × heads. This gives more work units for better thread utilization.
// With B=4, H=8 we get 32 parallel work units instead of just 4.

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_base, const AttentionDims dims) {

  size_t batch_size = dims.batch;
  size_t num_heads = dims.n_heads;
  size_t seq_len = dims.seq_len;
  size_t head_dim = dims.head_dim;
  const float scale = 1 / sqrtf(head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(seq_len, VEC_PADDING);

// Process each batch and head independently
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw = attn_base + thread_id * seq_len_padded;

      // float *aw = (float *)ASSUME_ALIGNED(attn_weights, ALIGNMENT);

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      // Process each query position
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim_stride;

        // NOTE: No need to compute it for key_pos > query_pos
        // Step 1: Compute scaled dot-product attention scores
        // QK^T for all valid (non-causal-masked) key positions
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim_stride;

          // Dot product across head dimension
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          aw[key_pos] = dot_product * scale;
        }

        // Step 2: Numerically stable softmax
        float max_score = -FLT_MAX;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          if (aw[key_pos] > max_score)
            max_score = aw[key_pos];
        }

        // Compute exp(score - max) and accumulate sum
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Normalize to get probabilities
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          aw[key_pos] /= sum_exp;
        }

        // Step 3: Weighted sum of values
        size_t output_offset = bh_offset + query_pos * head_dim_stride;

        // Initialize output
        for (size_t d = 0; d < head_dim; d++) {
          out[output_offset + d] = 0.0f;
        }

        // Accumulate: now d is inner loop
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float attn_weight = aw[key_pos];

          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += attn_weight * V[value_offset + d];
          }
        }
      }
    }
  }
}
