#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

/**
 * Multi-threaded Causal Multi-Head Self-Attention - v0
 *
 * This is the baseline multi-threaded implementation:
 * - Uses OpenMP to parallelize over the batch dimension
 * - Each thread gets its own scratch space for attention weights
 * - Uses #pragma omp simd hints for inner loops to enable vectorization
 * - Otherwise follows the same algorithm as single-threaded v1
 *
 * Thread workspace: Each thread needs seq_len floats for attention
 * weights. Total workspace: threads * seq_len floats.
 */

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_base, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);

#pragma omp parallel for
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      // Each thread gets its own scratch space
      size_t thread_id = (size_t)omp_get_thread_num();
      float *aw = attn_base + thread_id * seq_len;

      size_t bh_offset =
          b * (num_heads * seq_len * head_dim) + h * (seq_len * head_dim);

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim;

        // Step 1: Compute QK^T scores
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t key_offset = bh_offset + key_pos * head_dim;
          float dot_product = 0.0f;

#pragma omp simd reduction(+ : dot_product)
          for (size_t d = 0; d < head_dim; d++) {
            dot_product += Q[query_offset + d] * K[key_offset + d];
          }
          aw[key_pos] = dot_product * scale;
        }

        // Step 2: Softmax
        float max_score = -FLT_MAX;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          if (aw[key_pos] > max_score)
            max_score = aw[key_pos];
        }

        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }

        // Step 3: Weighted sum of values
        size_t output_offset = bh_offset + query_pos * head_dim;

#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out[output_offset + d] = 0.0f;
        }

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim;
          float weight = aw[key_pos] / sum_exp;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += weight * V[value_offset + d];
          }
        }
      }
    }
  }
}
