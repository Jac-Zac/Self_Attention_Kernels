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

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(seq_len, VEC_PADDING);

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Each thread gets its own scratch space
      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw = attn_base + thread_id * seq_len_padded;

      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim_stride;

        // Step 1: Compute QK^T scores
        float max_score = -FLT_MAX;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t key_offset = bh_offset + key_pos * head_dim_stride;
          float dot_product = 0.0f;

#pragma omp simd reduction(+ : dot_product)
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }

          float score = dot_product * scale;
          aw[key_pos] = score;
          max_score = score > max_score ? score : max_score;
        }

        // Step 2: Softmax
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }
        float inv_sum = 1.0f / sum_exp;

        // Step 3: Weighted sum of values
        size_t output_offset = bh_offset + query_pos * head_dim_stride;

#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out[output_offset + d] = 0.0f;
        }

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float weight = aw[key_pos] * inv_sum;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += weight * V[value_offset + d];
          }
        }
      }
    }
  }
}
