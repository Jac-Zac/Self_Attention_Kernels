#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// Changes from v0:
// 1. Loop fusion optimization - 4 loops instead of 6:
//    - v1: fused max computation with dot product, separate exp loop, then
//    fused output Reduces memory access overhead
//
// 2. Simplified early exit check:
//    - v1: if (q < seq_len)
//
// 3. Triangular workspace allocation (50% memory reduction):
//    - v1: Leverages causal mask triangular pattern
//          Thread q only needs q+1 attention weights (attends to positions
//          0..q) Total per (batch,head): sum_{i=1}^{seq_len} i = seq_len *
//          (seq_len+1) / 2 -> Memory savings: ~50%
//
// 4. Fusing batch and n_heads loops

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // 2D parallelization with swapped dimensions:
  // x dimension: query/seq_len
  // y dimension: heads * batch
  int bh = blockIdx.y * blockDim.y + threadIdx.y;
  int q = blockIdx.x * blockDim.x + threadIdx.x;

  if (q >= dims.seq_len || bh >= dims.batch * dims.n_heads)
    return;

  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;

  // Efficient 1 / sqrt function call
  const float scale = rsqrtf((float)head_dim);
  const size_t head_dim_pad = dims.head_dim_padded;

  // Map 1D bh back to batch and head for offset calculation
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Compute triangular workspace offset for this (batch, head, query)
  // position Each (batch, head) gets a triangular workspace
  // Query position q needs q+1 attention weights (positions 0..q)
  const size_t workspace_per_bh = seq_len * (seq_len + 1) / 2;

  // Base offset for this (batch, head) pair
  const size_t bh_workspace_offset = (b * num_heads + h) * workspace_per_bh;

  // Triangular offset: sum_{i=0}^{q-1} (i+1) = q * (q+1) / 2
  // This gives us the starting index for this thread's q+1 attention weights
  const size_t triangular_offset = q * (q + 1) / 2;

  float *RESTRICT aw = attn_weights + bh_workspace_offset + triangular_offset;

  // Base offset for current batch and head: [b, h, :, :]
  const size_t bh_offset =
      b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);

  const size_t query_offset = bh_offset + q * head_dim_pad;

  float max_score = -FLT_MAX;
  // Step 1: Compute scaled dot-product scores + track max (fused)
  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    float dot_product = 0.0f;
    size_t key_offset = bh_offset + key_pos * head_dim_pad;

    // Dot product across head dimension
    for (size_t d = 0; d < head_dim; d++) {
      dot_product += Q[query_offset + d] * K[key_offset + d];
    }

    const float score = dot_product * scale;
    max_score = fmaxf(max_score, score);
    aw[key_pos] = score;
  }
  //
  // Step 2: Numerically stable softmax (plain expf)
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

// ============================================================================
// Kernel Configuration (version-specific)
// ============================================================================
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 1

// Internal config struct - not exposed in header
typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

  // 3D mapping: x=queries/seq_len, y=head * batch
  size_t blocks_x = CEIL_DIV(dims.seq_len, threads_per_block.y);
  size_t blocks_y = CEIL_DIV(dims.batch * dims.n_heads, threads_per_block.z);

  dim3 number_of_blocks(blocks_x, blocks_y);

  size_t total_threads =
      (blocks_x * blocks_y) * (threads_per_block.x * threads_per_block.y);

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
  // Triangular workspace allocation leveraging causal mask pattern
  // Each (batch, head) needs: seq_len * (seq_len + 1) / 2 floats
  const size_t workspace_per_bh = dims.seq_len * (dims.seq_len + 1) / 2;
  return dims.batch * dims.n_heads * workspace_per_bh * sizeof(float);
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);

  VERBOSE_PRINT("CUDA Debug: Thread block (%d,%d), Grid (%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.number_of_blocks.x, config.number_of_blocks.y);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "Block dimensions: (%d,%d)\n", config.threads_per_block.x,
            config.threads_per_block.y);
  }
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
