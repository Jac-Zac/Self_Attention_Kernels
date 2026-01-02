#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// NOTE: Shared memory per block for attention weights
// Instead of allocating global workspace, each thread uses shared memory within
// its block to store attention weights. This eliminates global workspace
// entirely and provides faster memory access.
//
// Design:
// - Each block processes one (batch, head) pair
// - Threads within the block handle different query positions
// - Each thread gets a portion of shared memory for its attention weights
// - Shared memory is dynamically allocated based on seq_len and
// threads_per_block
//
// Limitation: seq_len must fit within available shared memory per thread.
// For very long sequences, you may need to reduce threads_per_block or use v1.

// Maximum shared memory per block (48KB on most GPUs, can be configured up to
// 96KB+) We'll use dynamic shared memory to be flexible

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const size_t smem_per_thread) {

  // Dynamic shared memory - each thread gets smem_per_thread floats
  extern __shared__ float shared_attn_weights[];

  // Block handles one (batch, head) pair
  // Thread handles one query position
  int b = blockIdx.z;
  int h = blockIdx.y;
  int q = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t batch_size = dims.batch;
  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = 1.0f / sqrtf((float)head_dim);

  if (b < batch_size && h < num_heads && q < (int)seq_len) {
    // Each thread's workspace starts at its thread index * smem_per_thread
    float *RESTRICT aw = shared_attn_weights + threadIdx.x * smem_per_thread;

    // Base offset for current batch and head in Q/K/V tensors: [b, h, :, :]
    size_t bh_offset =
        b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);

    size_t query_offset = bh_offset + q * head_dim_pad;

    // Step 1: Compute scaled dot-product attention scores
    // QK^T for all valid (non-causal-masked) key positions
    for (int key_pos = 0; key_pos <= q; key_pos++) {
      float dot_product = 0.0f;
      size_t key_offset = bh_offset + key_pos * head_dim_pad;

      for (size_t d = 0; d < head_dim; d++) {
        dot_product += Q[query_offset + d] * K[key_offset + d];
      }

      aw[key_pos] = dot_product * scale;
    }

    // Step 2: Numerically stable softmax
    float max_score = -FLT_MAX;
    for (int k = 0; k <= q; k++) {
      if (aw[k] > max_score)
        max_score = aw[k];
    }

    float sum_exp = 0.0f;
    for (int key_pos = 0; key_pos <= q; key_pos++) {
      float exp_val = expf(aw[key_pos] - max_score);
      aw[key_pos] = exp_val;
      sum_exp += exp_val;
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int key_pos = 0; key_pos <= q; key_pos++) {
      aw[key_pos] *= inv_sum;
    }

    // Step 3: Weighted sum of values
    size_t output_offset = bh_offset + q * head_dim_pad;

    for (size_t d = 0; d < head_dim; d++) {
      out[output_offset + d] = 0.0f;
    }

    for (int key_pos = 0; key_pos <= q; key_pos++) {
      size_t value_offset = bh_offset + key_pos * head_dim_pad;
      float attn_weight = aw[key_pos];

      for (size_t d = 0; d < head_dim; d++) {
        out[output_offset + d] += attn_weight * V[value_offset + d];
      }
    }
  }
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims, CudaConfig config) {
  // workspace parameter is unused in this version - we use shared memory
  // instead
  (void)workspace;

  // Calculate shared memory requirements
  // Each thread needs seq_len floats for its attention weights
  // But due to causal masking, we can use seq_len_padded to ensure alignment
  size_t smem_per_thread = dims.seq_len_padded;
  size_t threads_in_block = config.threads_per_block.x *
                            config.threads_per_block.y *
                            config.threads_per_block.z;
  size_t shared_mem_bytes = threads_in_block * smem_per_thread * sizeof(float);

  VERBOSE_PRINT("CUDA v3 Debug: Thread block (%d,%d,%d), Grid (%d,%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.threads_per_block.z, config.number_of_blocks.x,
                config.number_of_blocks.y, config.number_of_blocks.z);
  VERBOSE_PRINT("CUDA v3 Debug: Shared memory per block: %zu bytes\n",
                shared_mem_bytes);

  // Check if we have enough shared memory
  // Most GPUs have 48KB default, up to 96KB+ with opt-in
  if (shared_mem_bytes > 48 * 1024) {
    fprintf(stderr,
            "Warning: Requesting %zu bytes shared memory (> 48KB default). "
            "Consider reducing threads_per_block or seq_len.\n",
            shared_mem_bytes);
  }

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block,
                         shared_mem_bytes>>>(Q, K, V, out, dims,
                                             smem_per_thread);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "Block dimensions: (%d,%d,%d)\n",
            config.threads_per_block.x, config.threads_per_block.y,
            config.threads_per_block.z);
    fprintf(stderr, "Shared memory requested: %zu bytes\n", shared_mem_bytes);
  }
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
