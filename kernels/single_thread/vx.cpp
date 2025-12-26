#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <stddef.h>

// NOTE:
// This version processes 4 queries at a time to reuse K loads.
// Adjacent queries share most of their K rows due to causal masking:
// - Query q needs keys 0..q
// - Query q+1 needs keys 0..q+1
// - Query q+2 needs keys 0..q+2
// - Query q+3 needs keys 0..q+3
// By batching, we load each K row once and compute 4 dot products.

#define QUERY_BATCH 4

// Process a single query - used for remainder
static inline void
process_single_query(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT aw, size_t bh_offset, size_t query_pos,
                     size_t head_dim, size_t head_dim_stride, float scale) {

  size_t query_offset = bh_offset + query_pos * head_dim_stride;
  size_t num_keys = query_pos + 1;

  // Step 1: Dot products and find max
  float max_score = -FLT_MAX;
  for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
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
  for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
    float e = expf(aw[key_pos] - max_score);
    aw[key_pos] = e;
    sum_exp += e;
  }
  float inv_sum = 1.0f / sum_exp;

  // Step 3: Weighted sum
  size_t output_offset = bh_offset + query_pos * head_dim_stride;
#pragma omp simd
  for (size_t d = 0; d < head_dim; d++) {
    out[output_offset + d] = 0.0f;
  }

  for (size_t key_pos = 0; key_pos < num_keys; key_pos++) {
    size_t value_offset = bh_offset + key_pos * head_dim_stride;
    float weight = aw[key_pos] * inv_sum;
#pragma omp simd
    for (size_t d = 0; d < head_dim; d++) {
      out[output_offset + d] += weight * V[value_offset + d];
    }
  }
}

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      size_t bh_offset = (b * num_heads + h) * seq_len * head_dim_stride;

      // Score buffers - one per query in batch
      float *aw0 = attn_weights;
      float *aw1 = attn_weights + seq_len;
      float *aw2 = attn_weights + 2 * seq_len;
      float *aw3 = attn_weights + 3 * seq_len;

      size_t query_pos = 0;

      // Process queries in batches of 4
      for (; query_pos + QUERY_BATCH <= seq_len; query_pos += QUERY_BATCH) {
        size_t q0_offset = bh_offset + query_pos * head_dim_stride;
        size_t q1_offset = bh_offset + (query_pos + 1) * head_dim_stride;
        size_t q2_offset = bh_offset + (query_pos + 2) * head_dim_stride;
        size_t q3_offset = bh_offset + (query_pos + 3) * head_dim_stride;

        // All 4 queries share keys 0..query_pos
        size_t common_keys = query_pos + 1;

        float max0 = -FLT_MAX, max1 = -FLT_MAX;
        float max2 = -FLT_MAX, max3 = -FLT_MAX;

        // Phase 1a: Common keys - load K once, compute 4 dot products
        for (size_t key_pos = 0; key_pos < common_keys; key_pos++) {
          size_t key_offset = bh_offset + key_pos * head_dim_stride;

          float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
#pragma omp simd reduction(+ : dot0, dot1, dot2, dot3)
          for (size_t d = 0; d < head_dim; d++) {
            float kd = K[key_offset + d];
            dot0 += Q[q0_offset + d] * kd;
            dot1 += Q[q1_offset + d] * kd;
            dot2 += Q[q2_offset + d] * kd;
            dot3 += Q[q3_offset + d] * kd;
          }

          float s0 = dot0 * scale;
          aw0[key_pos] = s0;
          max0 = s0 > max0 ? s0 : max0;
          float s1 = dot1 * scale;
          aw1[key_pos] = s1;
          max1 = s1 > max1 ? s1 : max1;
          float s2 = dot2 * scale;
          aw2[key_pos] = s2;
          max2 = s2 > max2 ? s2 : max2;
          float s3 = dot3 * scale;
          aw3[key_pos] = s3;
          max3 = s3 > max3 ? s3 : max3;
        }

        // Phase 1b: Extra keys for queries 1, 2, 3 (causal differences)
        // Query 1 needs key query_pos+1
        {
          size_t key_offset = bh_offset + (query_pos + 1) * head_dim_stride;
          float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
          for (size_t d = 0; d < head_dim; d++)
            dot += Q[q1_offset + d] * K[key_offset + d];
          float s = dot * scale;
          aw1[query_pos + 1] = s;
          max1 = s > max1 ? s : max1;
        }

        // Query 2 needs keys query_pos+1, query_pos+2
        {
          size_t k1_offset = bh_offset + (query_pos + 1) * head_dim_stride;
          size_t k2_offset = bh_offset + (query_pos + 2) * head_dim_stride;
          float dot1 = 0.0f, dot2 = 0.0f;
#pragma omp simd reduction(+ : dot1, dot2)
          for (size_t d = 0; d < head_dim; d++) {
            dot1 += Q[q2_offset + d] * K[k1_offset + d];
            dot2 += Q[q2_offset + d] * K[k2_offset + d];
          }
          float s1 = dot1 * scale;
          aw2[query_pos + 1] = s1;
          max2 = s1 > max2 ? s1 : max2;
          float s2 = dot2 * scale;
          aw2[query_pos + 2] = s2;
          max2 = s2 > max2 ? s2 : max2;
        }

        // Query 3 needs keys query_pos+1, query_pos+2, query_pos+3
        {
          size_t k1_offset = bh_offset + (query_pos + 1) * head_dim_stride;
          size_t k2_offset = bh_offset + (query_pos + 2) * head_dim_stride;
          size_t k3_offset = bh_offset + (query_pos + 3) * head_dim_stride;
          float dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
#pragma omp simd reduction(+ : dot1, dot2, dot3)
          for (size_t d = 0; d < head_dim; d++) {
            dot1 += Q[q3_offset + d] * K[k1_offset + d];
            dot2 += Q[q3_offset + d] * K[k2_offset + d];
            dot3 += Q[q3_offset + d] * K[k3_offset + d];
          }
          float s1 = dot1 * scale;
          aw3[query_pos + 1] = s1;
          max3 = s1 > max3 ? s1 : max3;
          float s2 = dot2 * scale;
          aw3[query_pos + 2] = s2;
          max3 = s2 > max3 ? s2 : max3;
          float s3 = dot3 * scale;
          aw3[query_pos + 3] = s3;
          max3 = s3 > max3 ? s3 : max3;
        }

        // Phase 2: Softmax for each query
        float inv0, inv1, inv2, inv3;

#define DO_SOFTMAX(aw, num_k, max_s, inv_out)                                  \
  {                                                                            \
    float sum = 0.0f;                                                          \
    for (size_t kk = 0; kk < num_k; kk++) {                                    \
      float e = expf(aw[kk] - max_s);                                          \
      aw[kk] = e;                                                              \
      sum += e;                                                                \
    }                                                                          \
    inv_out = 1.0f / sum;                                                      \
  }

        DO_SOFTMAX(aw0, query_pos + 1, max0, inv0);
        DO_SOFTMAX(aw1, query_pos + 2, max1, inv1);
        DO_SOFTMAX(aw2, query_pos + 3, max2, inv2);
        DO_SOFTMAX(aw3, query_pos + 4, max3, inv3);

#undef DO_SOFTMAX

        // Phase 3: Weighted sum for each query
        size_t o0_offset = bh_offset + query_pos * head_dim_stride;
        size_t o1_offset = bh_offset + (query_pos + 1) * head_dim_stride;
        size_t o2_offset = bh_offset + (query_pos + 2) * head_dim_stride;
        size_t o3_offset = bh_offset + (query_pos + 3) * head_dim_stride;

// Initialize outputs
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out[o0_offset + d] = 0.0f;
          out[o1_offset + d] = 0.0f;
          out[o2_offset + d] = 0.0f;
          out[o3_offset + d] = 0.0f;
        }

        // Weighted sum for query 0
        for (size_t key_pos = 0; key_pos < query_pos + 1; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float w = aw0[key_pos] * inv0;
#pragma omp simd
          for (size_t d = 0; d < head_dim; d++)
            out[o0_offset + d] += w * V[value_offset + d];
        }

        // Weighted sum for query 1
        for (size_t key_pos = 0; key_pos < query_pos + 2; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float w = aw1[key_pos] * inv1;
#pragma omp simd
          for (size_t d = 0; d < head_dim; d++)
            out[o1_offset + d] += w * V[value_offset + d];
        }

        // Weighted sum for query 2
        for (size_t key_pos = 0; key_pos < query_pos + 3; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float w = aw2[key_pos] * inv2;
#pragma omp simd
          for (size_t d = 0; d < head_dim; d++)
            out[o2_offset + d] += w * V[value_offset + d];
        }

        // Weighted sum for query 3
        for (size_t key_pos = 0; key_pos < query_pos + 4; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_stride;
          float w = aw3[key_pos] * inv3;
#pragma omp simd
          for (size_t d = 0; d < head_dim; d++)
            out[o3_offset + d] += w * V[value_offset + d];
        }
      }

      // Handle remaining queries
      for (; query_pos < seq_len; query_pos++) {
        process_single_query(Q, K, V, out, aw0, bh_offset, query_pos, head_dim,
                             head_dim_stride, scale);
      }
    }
  }
}
