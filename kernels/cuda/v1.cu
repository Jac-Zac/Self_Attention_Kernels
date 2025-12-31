#include <cfloat>
#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // 2D parallelization: each thread handles one (batch, head) combination
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const size_t seq_len_padded = dims.seq_len_padded;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // Only proceed if this thread has valid work
  if (b < batch_size && h < num_heads) {
    // Each thread gets its own workspace for attention weights
    // Use a simple 1D thread ID for workspace indexing
    int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
    float *RESTRICT aw = attn_weights + thread_id * seq_len_padded;

    // Base offset for current batch and head: [b, h, :, :]
    size_t bh_offset =
        b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);

    // Sequential processing of all query positions within this thread
    // Process each query position sequentially within this thread
    for (size_t query_pos = 0; query_pos < seq_len; query_pos++) {
      size_t query_offset = bh_offset + query_pos * head_dim_pad;

      // Step 1: Compute scaled dot-product attention scores
      // QK^T for all valid (non-causal-masked) key positions
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

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 const AttentionDims dims) {

  // Load things correctly fot the correct dimensionality
  const size_t head_dim_padded = round_up_pow2(dims.head_dim, VEC_PADDING);
  const size_t seq_len_padded = round_up_pow2(dims.seq_len, VEC_PADDING);

  // Set up workspace using managed memory
  float *workspace;

  // 2D thread configuration: parallelize over (batch, head) only
  dim3 threads_per_block(16, 16); // 16x16 = 256 threads per block

  size_t blocks_x =
      (dims.batch + threads_per_block.x - 1) / threads_per_block.x;
  size_t blocks_y =
      (dims.n_heads + threads_per_block.y - 1) / threads_per_block.y;

  dim3 number_of_blocks(blocks_x, blocks_y);

  // Allocate workspace: one attention weight array per thread
  cudaMallocManaged(&workspace, threads_per_block.x * threads_per_block.y *
                                    seq_len_padded * sizeof(float));

  cmhsa_forward_kernel<<<number_of_blocks, threads_per_block>>>(
      Q, K, V, out, workspace, dims);

  cudaDeviceSynchronize();
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
