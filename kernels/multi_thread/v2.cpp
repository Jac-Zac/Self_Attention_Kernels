#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>

// NOTE:
// This version combines collapse(2) parallelization with 2-query batching.
// Adjacent queries share most of their K rows due to causal masking:
// - Query q needs keys 0..q
// - Query q+1 needs keys 0..q+1
// By batching 2 queries, we load each K row once and compute 2 dot products.
// This is simpler than the 4-query batching in vx.cpp.

#define QUERY_BATCH 2

// Process a single query - used for remainder when seq_len is odd
static inline void
process_single_query(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT aw, size_t bh_offset, size_t query_pos,
                     size_t head_dim, size_t head_dim_stride, float scale) {

  size_t q_offset = bh_offset + query_pos * head_dim_stride;
  size_t num_keys = query_pos + 1;

  // Step 1: Dot products and find max
  float max_score = -FLT_MAX;
  for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
    size_t k_offset = bh_offset + key_pos * head_dim_stride;
    float dot_product = 0.0f;

#pragma omp simd reduction(+ : dot_product)
    for (size_t d = 0; d < head_dim; d++) {
      dot_product += Q[q_offset + d] * K[k_offset + d];
    }

    float score = dot_product * scale;
    aw[key_pos] = score;
    max_score = score > max_score ? score : max_score;
  }

  // Step 2: Softmax
  float sum_exp = 0.0f;
  for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
    float e = expf(aw[key_pos] - max_score);
    aw[key_pos] = e;
    sum_exp += e;
  }
  float inv_sum = 1.0f / sum_exp;

  // Step 3: Weighted sum
  size_t o_offset = bh_offset + query_pos * head_dim_stride;

#pragma omp simd
  for (size_t d = 0; d < head_dim; d++) {
    out[o_offset + d] = 0.0f;
  }

  for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
    size_t v_offset = bh_offset + key_pos * head_dim_stride;
    float weight = aw[key_pos] * inv_sum;

#pragma omp simd
    for (size_t d = 0; d < head_dim; d++) {
      out[o_offset + d] += weight * V[v_offset + d];
    }
  }
}

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

      // Each thread gets its own scratch space for 2 score buffers
      size_t tid = (size_t)omp_get_thread_num();
      float *aw0 = attn_base + tid * QUERY_BATCH * seq_len_padded;
      float *aw1 = aw0 + seq_len_padded;

      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      size_t query_pos = 0;

      // Process queries in pairs
      for (; query_pos + QUERY_BATCH <= seq_len; query_pos += QUERY_BATCH) {
        size_t q0_offset = bh_offset + query_pos * head_dim_stride;
        size_t q1_offset = bh_offset + (query_pos + 1) * head_dim_stride;

        // Both queries share keys 0..query_pos
        size_t common_keys = query_pos + 1;

        float max0 = -FLT_MAX, max1 = -FLT_MAX;

        // Phase 1a: Common keys - load K once, compute 2 dot products
        for (size_t key_pos = 0; key_pos < common_keys; key_pos++) {
          size_t k_offset = bh_offset + key_pos * head_dim_stride;

          float dot0 = 0.0f, dot1 = 0.0f;

#pragma omp simd reduction(+ : dot0, dot1)
          for (size_t d = 0; d < head_dim; d++) {
            float kd = K[k_offset + d];
            dot0 += Q[q0_offset + d] * kd;
            dot1 += Q[q1_offset + d] * kd;
          }

          float s0 = dot0 * scale;
          aw0[key_pos] = s0;
          max0 = s0 > max0 ? s0 : max0;

          float s1 = dot1 * scale;
          aw1[key_pos] = s1;
          max1 = s1 > max1 ? s1 : max1;
        }

        // Phase 1b: Extra key for query 1 (key at position query_pos+1)
        {
          size_t k_offset = bh_offset + (query_pos + 1) * head_dim_stride;
          float dot = 0.0f;

#pragma omp simd reduction(+ : dot)
          for (size_t d = 0; d < head_dim; d++) {
            dot += Q[q1_offset + d] * K[k_offset + d];
          }

          float s = dot * scale;
          aw1[query_pos + 1] = s;
          max1 = s > max1 ? s : max1;
        }

        // Phase 2: Softmax for both queries
        float sum0 = 0.0f, sum1 = 0.0f;

        for (size_t key_pos = 0; key_pos < query_pos + 1; key_pos++) {
          float e = expf(aw0[key_pos] - max0);
          aw0[key_pos] = e;
          sum0 += e;
        }
        float inv0 = 1.0f / sum0;

        for (size_t key_pos = 0; key_pos < query_pos + 2; key_pos++) {
          float e = expf(aw1[key_pos] - max1);
          aw1[key_pos] = e;
          sum1 += e;
        }
        float inv1 = 1.0f / sum1;

        // Phase 3: Weighted sum for both queries
        size_t o0_offset = bh_offset + query_pos * head_dim_stride;
        size_t o1_offset = bh_offset + (query_pos + 1) * head_dim_stride;

// Initialize outputs to zero
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out[o0_offset + d] = 0.0f;
          out[o1_offset + d] = 0.0f;
        }

        // Weighted sum for query 0
        for (size_t key_pos = 0; key_pos < query_pos + 1; key_pos++) {
          size_t v_offset = bh_offset + key_pos * head_dim_stride;
          float w = aw0[key_pos] * inv0;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out[o0_offset + d] += w * V[v_offset + d];
          }
        }

        // Weighted sum for query 1
        for (size_t key_pos = 0; key_pos < query_pos + 2; key_pos++) {
          size_t v_offset = bh_offset + key_pos * head_dim_stride;
          float w = aw1[key_pos] * inv1;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out[o1_offset + d] += w * V[v_offset + d];
          }
        }
      }

      // Handle remainder (if seq_len is odd)
      for (; query_pos < seq_len; query_pos++) {
        process_single_query(Q, K, V, out, aw0, bh_offset, query_pos, head_dim,
                             head_dim_stride, scale);
      }
    }
  }
}
