#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// NOTE:
// This version combines multi_thread/v1's parallelization (collapse(2) over
// batch × heads) with single_thread/v2's optimization of fusing max-finding
// with score computation. This reduces one pass over the attention weights.
//
// However, as noted in single_thread/v2, fusing max-finding with score
// computation can limit instruction-level parallelism because the max
// comparison depends on each score computation, creating a data dependency
// chain.

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_base, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(seq_len, VEC_PADDING);

// Process each batch and head independently
#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw = attn_base + thread_id * seq_len_padded;

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      // Process each query position
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim_stride;

        // Track max score while computing dot products (fused pass)
        float max_score = -FLT_MAX;

        // Step 1: Compute scaled dot-product attention scores + track max
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim_stride;

          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          float score = dot_product * scale;
          max_score = score > max_score ? score : max_score;
          aw[key_pos] = score;
        }

        // Step 2: Compute exp(score - max) and accumulate sum
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Step 3: Weighted sum of values (normalize inline)
        size_t output_offset = bh_offset + query_pos * head_dim_stride;
        const float inv_sum_exp = 1.0f / sum_exp;

        // Accumulate with normalized weights
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float normalized_weight = aw[key_pos] * inv_sum_exp;

          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += normalized_weight * V[value_offset + d];
          }
        }
      }
    }
  }
}
