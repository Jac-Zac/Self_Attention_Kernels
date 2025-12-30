#include <cfloat>
#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // Process each batch and head independently
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t h = 0; h < num_heads; h++) {

      float *aw = attn_weights;

      // Base offset for current batch and head: [b, h, :, :]
      size_t bh_offset = b * (num_heads * seq_len * head_dim_pad) +
                         h * (seq_len * head_dim_pad);

      // Process each query position
      for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
        size_t query_offset = bh_offset + query_pos * head_dim_pad;

        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          float dot_product = 0.0f;
          size_t key_offset = bh_offset + key_pos * head_dim_pad;

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
        size_t output_offset = bh_offset + query_pos * head_dim_pad;

        // Initialize output
        for (size_t d = 0; d < head_dim; d++) {
          out[output_offset + d] = 0.0f;
        }

        // Accumulate: now d is inner loop
        for (size_t key_pos = 0; key_pos <= query_pos; key_pos++) {
          size_t value_offset = bh_offset + key_pos * head_dim_pad;
          float attn_weight = aw[key_pos];

          for (size_t d = 0; d < head_dim; d++) {
            out[output_offset + d] += attn_weight * V[value_offset + d];
          }
        }
      }
    }
  }
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 const AttentionDims dims) {

  const size_t num_threads = 1;

  // Load things correctly fot the correct dimensionality
  const size_t head_dim_padded = round_up_pow2(dims.head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(dims.seq_len, VEC_PADDING);

  // Set up workspace
  float *workspace;

  cudaMalloc(&workspace, num_threads * seq_len_padded);

  // WARNING: Not implemented yet - just launch empty kernel
  cmhsa_forward_kernel<<<1, num_threads>>>(Q, K, V, out, workspace, dims);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
