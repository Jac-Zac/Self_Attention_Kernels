#include "../../include/cmhsa_forward.h"
#include <float.h>
#include <math.h>

// NOTE:
// This version uses GCC vector extensions for explicit SIMD operations.
// The v4sf type maps to 128-bit vector registers (SSE on x86, NEON on ARM).
// This is portable across architectures without using intrinsics.

// GCC vector extension: 4 floats packed into 128 bits
typedef float v4sf __attribute__((vector_size(16)));

// Horizontal sum of 4-element float vector
static inline float hsum_v4sf(v4sf v) { return v[0] + v[1] + v[2] + v[3]; }

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_stride = round_up_pow2(head_dim, VEC_PADDING);

  // Vector width and iteration counts
  const size_t VEC_WIDTH = 4;
  const size_t head_dim_vecs = head_dim / VEC_WIDTH; // Full vector iterations

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {
      size_t bh_offset = b * (num_heads * seq_len * head_dim_stride) +
                         h * (seq_len * head_dim_stride);

      float *aw = attn_weights;

      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t q_offset = bh_offset + query_pos * head_dim_stride;
        float max_score = -FLT_MAX;

        // Phase 1: Compute Q @ K^T scores with vectorized dot products
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t k_offset = bh_offset + key_pos * head_dim_stride;

          // Vectorized dot product accumulator
          v4sf acc = {0.0f, 0.0f, 0.0f, 0.0f};

          // Process head_dim in chunks of 4
          for (size_t v = 0; v < head_dim_vecs; v++) {
            size_t idx = v * VEC_WIDTH;
            v4sf q_vec = *(const v4sf *)&Q[q_offset + idx];
            v4sf k_vec = *(const v4sf *)&K[k_offset + idx];
            acc += q_vec * k_vec;
          }

          float dot_product = hsum_v4sf(acc);

          // Handle remainder elements (if head_dim % 4 != 0)
          for (size_t d = head_dim_vecs * VEC_WIDTH; d < head_dim; d++) {
            dot_product += Q[q_offset + d] * K[k_offset + d];
          }

          float score = dot_product * scale;
          aw[key_pos] = score;
          max_score = score > max_score ? score : max_score;
        }

        // Phase 2: Softmax (scalar - no benefit from vectorization here)
        float sum_exp = 0.0f;
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float exp_val = expf(aw[key_pos] - max_score);
          aw[key_pos] = exp_val;
          sum_exp += exp_val;
        }
        const float inv_sum = 1.0f / sum_exp;

        // Phase 3: Weighted sum of values (vectorized)
        size_t o_offset = bh_offset + query_pos * head_dim_stride;

        // Initialize output to zero using vectors
        v4sf zero = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t v = 0; v < head_dim_vecs; v++) {
          *(v4sf *)&out[o_offset + v * VEC_WIDTH] = zero;
        }
        // Zero remainder elements
        for (size_t d = head_dim_vecs * VEC_WIDTH; d < head_dim; d++) {
          out[o_offset + d] = 0.0f;
        }

        // Accumulate weighted values
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t v_offset = bh_offset + key_pos * head_dim_stride;
          float weight = aw[key_pos] * inv_sum;

          // Broadcast weight to all vector lanes
          v4sf w_vec = {weight, weight, weight, weight};

          // Vectorized weighted accumulation
          for (size_t v = 0; v < head_dim_vecs; v++) {
            size_t idx = v * VEC_WIDTH;
            v4sf out_vec = *(v4sf *)&out[o_offset + idx];
            v4sf val_vec = *(const v4sf *)&V[v_offset + idx];
            *(v4sf *)&out[o_offset + idx] = out_vec + w_vec * val_vec;
          }

          // Handle remainder elements
          for (size_t d = head_dim_vecs * VEC_WIDTH; d < head_dim; d++) {
            out[o_offset + d] += weight * V[v_offset + d];
          }
        }
      }
    }
  }
}
