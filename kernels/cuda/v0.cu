#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// NOTE: Baseline implementation

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
  const size_t head_dim_pad = dims.head_dim_padded;
  const size_t seq_len_padded = dims.seq_len_padded;
  const float scale = 1.0f / sqrtf((float)head_dim);

  // Only proceed if this thread has valid work
  if (b < batch_size && h < num_heads && q < seq_len) {
    // Each thread gets its own workspace for attention weights
    // Use GLOBAL thread ID for workspace indexing
    int block_id = blockIdx.z * (gridDim.x * gridDim.y) +
                   blockIdx.y * gridDim.x + blockIdx.x;

    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int thread_id_in_block = threadIdx.z * (blockDim.x * blockDim.y) +
                             threadIdx.y * blockDim.x + threadIdx.x;

    int global_thread_id = block_id * threads_per_block + thread_id_in_block;

    float *RESTRICT aw = attn_weights + global_thread_id * seq_len_padded;

    // Base offset for current batch and head: [b, h, :, :]
    size_t bh_offset =
        b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);

    // Sequential processing of all query positions within this thread
    // Process each query position sequentially within this thread
    size_t query_offset = bh_offset + q * head_dim_pad;

    // Step 1: Compute scaled dot-product attention scores
    // QK^T for all valid (non-causal-masked) key positions
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
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
    for (size_t k = 0; k <= q; k++) {
      if (aw[k] > max_score)
        max_score = aw[k];
    }

    // Compute exp(score - max) and accumulate sum
    float sum_exp = 0.0f;
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      float exp_val = expf(aw[key_pos] - max_score);
      aw[key_pos] = exp_val;
      sum_exp += exp_val;
    }

    // Normalize to get probabilities
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      aw[key_pos] /= sum_exp;
    }

    // Step 3: Weighted sum of values
    size_t output_offset = bh_offset + q * head_dim_pad;

    // Initialize output
    for (size_t d = 0; d < head_dim; d++) {
      out[output_offset + d] = 0.0f;
    }

    // Accumulate: now d is inner loop
    for (size_t key_pos = 0; key_pos <= q; key_pos++) {
      size_t value_offset = bh_offset + key_pos * head_dim_pad;
      float attn_weight = aw[key_pos];

      for (size_t d = 0; d < head_dim; d++) {
        out[output_offset + d] += attn_weight * V[value_offset + d];
      }
    }
  }
}

// ============================================================================
// Kernel Configuration (version-specific)
// ============================================================================
#define THREADS_PER_BLOCK_X 512
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 1

// Internal config struct - not exposed in header
typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y,
                         THREADS_PER_BLOCK_Z);

  // 3D mapping: x=queries/seq_len, y=heads, z=batch
  size_t blocks_x =
      (dims.seq_len + threads_per_block.x - 1) / threads_per_block.x;
  size_t blocks_y =
      (dims.n_heads + threads_per_block.y - 1) / threads_per_block.y;
  size_t blocks_z =
      (dims.batch + threads_per_block.z - 1) / threads_per_block.z;

  dim3 number_of_blocks(blocks_x, blocks_y, blocks_z);

  size_t total_threads =
      (blocks_x * blocks_y * blocks_z) *
      (threads_per_block.x * threads_per_block.y * threads_per_block.z);

  CudaConfig config;
  config.threads_per_block = threads_per_block;
  config.number_of_blocks = number_of_blocks;
  config.total_threads = total_threads;
  return config;
}

// ============================================================================
// Public API
// ============================================================================

size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);
  size_t seq_len_padded = round_up_pow2(dims.seq_len, VEC_PADDING);
  return config.total_threads * seq_len_padded * sizeof(float);
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);

  VERBOSE_PRINT("CUDA Debug: Thread block (%d,%d,%d), Grid (%d,%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.threads_per_block.z, config.number_of_blocks.x,
                config.number_of_blocks.y, config.number_of_blocks.z);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims);
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
