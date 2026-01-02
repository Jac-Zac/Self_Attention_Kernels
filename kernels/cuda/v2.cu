#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// NOTE:
// Changes from v1:
// - Uses triangular workspace with prefix sum indexing
// - Each thread handling query position q only needs (q+1) floats for attention
//   weights (due to causal masking), not the full seq_len_padded
// - Workspace offset for query q: q*(q+1)/2 within each (batch, head) pair
// - Total workspace per (batch, head): seq_len * (seq_len + 1) / 2 floats
// - This reduces memory usage by ~50% compared to v0/v1's rectangular
// allocation

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // 3D parallelization with swapped dimensions:
  // x dimension: queries/seq_len (supports up to 1024 threads)
  // y dimension: heads
  // z dimension: batch (typically small, so z's 64-thread limit is fine)
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z * blockDim.z + threadIdx.z;

  int q = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const size_t head_dim_pad = dims.head_dim_padded;

  // Only proceed if this thread has valid work
  if (b < batch_size && h < num_heads && q < seq_len) {
    // Triangular workspace indexing:
    // Each (batch, head) pair gets a triangular workspace of size T*(T+1)/2
    // Thread for query q starts at offset q*(q+1)/2 within its (b,h) workspace
    size_t triangular_size = seq_len * (seq_len + 1) / 2;
    size_t bh_workspace_offset = (b * num_heads + h) * triangular_size;
    size_t query_workspace_offset = (size_t)q * ((size_t)q + 1) / 2;

    float *RESTRICT aw =
        attn_weights + bh_workspace_offset + query_workspace_offset;

    // Base offset for current batch and head: [b, h, :, :]
    const size_t bh_offset =
        b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);

    const size_t query_offset = bh_offset + q * head_dim_pad;

    // =====================================================================
    // Step 1: Compute scaled dot-product scores + track max (fused)
    // Only compute for key_pos <= query_pos (causal mask)
    // =====================================================================
    float max_score = -FLT_MAX;
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      float dot_product = 0.0f;
      size_t key_offset = bh_offset + key_pos * head_dim_pad;

      // Dot product across head dimension
      for (size_t d = 0; d < head_dim; d++) {
        dot_product += Q[query_offset + d] * K[key_offset + d];
      }

      const float score = dot_product * scale;
      max_score = score > max_score ? score : max_score;
      aw[key_pos] = score;
    }

    // ===============================================
    // Step 2: Numerically stable softmax (plain expf)
    // ===============================================
    float sum_exp = 0.0f;
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      float exp_val = expf(aw[key_pos] - max_score);
      aw[key_pos] = exp_val;
      sum_exp += exp_val;
    }

    // Step 3: Weighted sum of values
    const size_t output_offset = bh_offset + q * head_dim_pad;
    const float inv_sum_exp = 1.0f / sum_exp;

    // Initialize output to zero
    for (size_t d = 0; d < head_dim; d++) {
      out[output_offset + d] = 0.0f;
    }

    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      size_t value_offset = bh_offset + key_pos * head_dim_pad;
      const float normalized_weight = aw[key_pos] * inv_sum_exp;

      for (size_t d = 0; d < head_dim; d++) {
        out[output_offset + d] += normalized_weight * V[value_offset + d];
      }
    }
  }
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims, CudaConfig config) {

  VERBOSE_PRINT("CUDA Debug: Thread block (%d,%d,%d), Grid (%d,%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.threads_per_block.z, config.number_of_blocks.x,
                config.number_of_blocks.y, config.number_of_blocks.z);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "Block dimensions: (%d,%d,%d)\n",
            config.threads_per_block.x, config.threads_per_block.y,
            config.threads_per_block.z);
  }
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
