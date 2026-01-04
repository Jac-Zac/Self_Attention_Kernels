#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// NOTE:
// Changes from v1:
// 3. Do head dim reduction
//
// 2. Simplified early exit check:
//    Removed the attn_weights by recomputing dot product trading of memory for
//    computation which currently is reasonable to avoid storing too much
//    useless memory I guess ?
//
// 4. Fixed block configuration to ensure correct warp reductions:
//    - Previously used fixed 32-thread blocks, requiring multiple blocks in
//    x-dim
//    - Warp reductions (__shfl_down_sync) only work within a single block
//    - Now uses power-of-2 block sizes (32, 64, 128, 256, 512, 1024) to fit
//    full head_dim
//    - Only 1 block in x-dimension, ensuring all threads for a query are in
//    same block

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // Thread mapping:
  // x: head_dim (32 threads = 1 warp)
  // y: query index (from seq_len)
  // z: batch * n_heads index
  int d = threadIdx.x;
  int q = blockIdx.y;
  int bh = blockIdx.z;

  if (d >= dims.head_dim || q >= dims.seq_len ||
      bh >= dims.batch * dims.n_heads)
    return;

  const int head_dim = dims.head_dim;
  const int head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  // Map 1D bh back to batch and head for offset calculation
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset =
      (size_t)b * dims.n_heads * dims.seq_len * head_dim_pad +
      (size_t)h * dims.seq_len * head_dim_pad;
  const size_t q_offset = bh_offset + (size_t)q * head_dim_pad;

  // Phase 1: Compute max score across all causal keys (0 to q)
  float global_max = -FLT_MAX;

  // We loop through all keys for this query to ensure numerical stability
  for (int k = 0; k <= q; k++) {
    const size_t k_offset = bh_offset + (size_t)k * head_dim_pad;

    // Each thread computes partial dot product for its dimension
    float partial_dot = Q[q_offset + d] * K[k_offset + d];

    // Warp reduction to sum across head_dim (assuming head_dim fits in warp)
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial_dot += __shfl_down_sync(0xffffffff, partial_dot, offset);
    }

    // Thread 0 has the full dot product, broadcast it back
    float score = __shfl_sync(0xffffffff, partial_dot, 0) * scale;
    global_max = fmaxf(global_max, score);
  }

  // Phase 2: Compute softmax denominator
  float global_sum_exp = 0.0f;

  for (int k = 0; k <= q; k++) {
    const size_t k_offset = bh_offset + (size_t)k * head_dim_pad;
    float partial_dot = Q[q_offset + d] * K[k_offset + d];

    for (int offset = 16; offset > 0; offset >>= 1) {
      partial_dot += __shfl_down_sync(0xffffffff, partial_dot, offset);
    }

    float score = __shfl_sync(0xffffffff, partial_dot, 0) * scale;
    global_sum_exp += expf(score - global_max);
  }

  float inv_sum = 1.0f / (global_sum_exp + 1e-9f);

  // Phase 3: Weighted value accumulation
  float out_val = 0.0f;

  for (int k = 0; k <= q; k++) {
    const size_t k_offset = bh_offset + (size_t)k * head_dim_pad;
    const size_t v_offset = bh_offset + (size_t)k * head_dim_pad;

    // Recompute dot product (trading compute for memory)
    float partial_dot = Q[q_offset + d] * K[k_offset + d];

    for (int offset = 16; offset > 0; offset >>= 1) {
      partial_dot += __shfl_down_sync(0xffffffff, partial_dot, offset);
    }

    float score = __shfl_sync(0xffffffff, partial_dot, 0) * scale;
    float weight = expf(score - global_max) * inv_sum;

    // Each thread accumulates its dimension of the output
    out_val += weight * V[v_offset + d];
  }

  out[q_offset + d] = out_val;
}

// ============================================================================
// Kernel Configuration (version-specific)
// ============================================================================

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  // Use power-of-2 block size to fit full head_dim in single block
  // This ensures warp reductions work correctly (they only work within a block)
  int block_size = 32;

  while (block_size < dims.head_dim && block_size < 1024) {
    block_size *= 2;
  }

  dim3 threads_per_block(block_size, 1, 1);

  // x: 1 block (all head_dim threads in same block for warp reduction)
  // y: sequence length (one block per query)
  // z: batch * heads
  size_t blocks_y = dims.seq_len;
  size_t blocks_z = dims.batch * dims.n_heads;

  dim3 number_of_blocks(1, blocks_y, blocks_z);

  CudaConfig config;
  config.threads_per_block = threads_per_block;
  config.number_of_blocks = number_of_blocks;
  return config;
}

// ============================================================================
// Public API
// ============================================================================
size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  (void)dims;
  return 0;
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
