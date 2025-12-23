#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);

#pragma omp parallel for collapse(2) schedule(static)
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      const size_t bh_offset =
          (b * num_heads + h) * (seq_len * head_dim_stride);

      for (size_t i = 0; i < seq_len; i++) {
        // Pointer to current query row and output row
        const float *RESTRICT q_ptr = (const float *)ASSUME_ALIGNED(
            &Q[bh_offset + i * head_dim_stride], ALIGNMENT);
        float *RESTRICT out_ptr = (float *)ASSUME_ALIGNED(
            &out[bh_offset + i * head_dim_stride], ALIGNMENT);

        // Online softmax running stats
        float m = -FLT_MAX; // Running max
        float l = 0.0f;     // Running sum

        // Initialize output row to zero
        for (size_t d = 0; d < head_dim; d++)
          out_ptr[d] = 0.0f;

        for (size_t j = 0; j <= i; j++) {
          const float *RESTRICT k_ptr = (const float *)ASSUME_ALIGNED(
              &K[bh_offset + j * head_dim_stride], ALIGNMENT);
          const float *RESTRICT v_ptr = (const float *)ASSUME_ALIGNED(
              &V[bh_offset + j * head_dim_stride], ALIGNMENT);

          // Step 1: Dot Product (QK^T)
          // GCC will use 512-bit vectors here due to -mprefer-vector-width=512
          float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
          for (size_t d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
          }
          dot *= scale;

          // Step 2: Online Softmax Update
          float m_prev = m;
          m = (dot > m) ? dot : m;

          // These exps are the main computational cost; -mveclibabi=svml helps
          // here
          float exp_scale = expf(m_prev - m);
          float exp_curr = expf(dot - m);

          // Step 3: Fused Weighted Sum
          // out = out * exp(m_old - m_new) + v * exp(dot - m_new)
          l = l * exp_scale + exp_curr;

#pragma omp simd
          for (size_t d = 0; d < head_dim; d++) {
            out_ptr[d] = out_ptr[d] * exp_scale + v_ptr[d] * exp_curr;
          }
        }

        // Final normalization for the query row
        float inv_l = 1.0f / l;
#pragma omp simd
        for (size_t d = 0; d < head_dim; d++) {
          out_ptr[d] *= inv_l;
        }
      }
    }
  }
}
