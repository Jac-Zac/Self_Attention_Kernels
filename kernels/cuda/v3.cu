#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Changes from v2 to v3:
// ============================================================================
// 1. Eliminated workspace allocation for attention weights:
//    - v3: Recomputes Q·K dot products in each pass (two-pass algorithm)
//    - Memory savings: No external workspace needed (workspace size = 0)
//    - Tradeoff: Increased compute (2x dot products) for reduced memory
//
// 2. Query tiling with multiple queries per thread block:
//    - v3: TILE_Q=8 queries processed per block (y-dimension parallelism)
//    - Grid: blocks_y = ceil(seq_len / TILE_Q) instead of blocks_x = seq_len
//
// 3. Algorithm is now fully register-based for attention scores:
//    - Pass 1: Compute max score for numerical stability
//    - Pass 2: Compute exp sum and weighted output in one fused pass
//    - All intermediate values stay in registers, no global memory writes
// ============================================================================

#define TILE_Q 8
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(WARP_MASK, val, offset);
  }
  return val;
}

__inline__ __device__ float warp_broadcast(float val, int src_lane) {
  return __shfl_sync(WARP_MASK, val, src_lane);
}

__inline__ __device__ float warp_reduce_max(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(WARP_MASK, val, offset));
  }
  return val;
}

__inline__ __device__ float warp_qk_dot(const float *RESTRICT Q,
                                        const float *RESTRICT K,
                                        size_t q_offset, size_t k_offset,
                                        size_t head_dim, int lane_id) {
  float partial = 0.0f;
  // Strided over head_dim by warp
  for (size_t d = lane_id; d < head_dim; d += warpSize) {
    partial += Q[q_offset + d] * K[k_offset + d];
  }

  // Warp-wide reduction
  float dot = warp_reduce_sum(partial);

  // IMPORTANT:
  // - Only lane 0 receives the correct reduced value
  // - Other lanes get undefined data
  return dot;
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  // 2D parallelization:
  // x dimension: lane within warp (0-31)
  // y dimension: query position within tile
  //   - blockIdx.y * TILE_Q: tile start position
  //   - threadIdx.y: position within tile (0 to TILE_Q-1)
  // z dimension: (batch, head) pairs
  const int bh = blockIdx.z * blockDim.z + threadIdx.z;
  const int q = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane_id = threadIdx.x; // Warp lane ID (0-31)

  if (q >= dims.seq_len || bh >= dims.batch * dims.n_heads)
    return;

  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = rsqrtf((float)head_dim);
  const size_t head_dim_pad = dims.head_dim_padded;

  // Decompose linear (batch, head) index
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Tensor offsets: [batch, head, seq, head_dim]
  const size_t bh_offset =
      b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // PASS 1: Find maximum score for numerical stability
  // ===========================================================================
  // We need max(Q·K) before computing softmax to prevent overflow in exp()
  // This pass recomputes all dot products to find the maximum
  float max_score = -FLT_MAX;

  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;

    float score =
        warp_qk_dot(Q, K, query_offset, key_offset, head_dim, lane_id) * scale;

    // Track maximum (only lane 0 has the correct reduced value)
    if (lane_id == 0) {
      max_score = fmaxf(max_score, score);
    }
  }

  // Broadcast max to all lanes for use in Pass 2
  max_score = warp_broadcast(max_score, 0);

  // ===========================================================================
  // PASS 2: Compute softmax and weighted sum in one fused pass
  // ===========================================================================
  // Now we recompute Q·K and simultaneously:
  // 1. Compute exp(score - max_score) for numerical stability
  // 2. Accumulate sum of exponentials for normalization
  // 3. Accumulate weighted output: sum(softmax_weight * V)

  float sum_exp = 0.0f;

  // Initialize output accumulators in registers (one per dimension this lane
  // handles) Each lane accumulates for dimensions: lane_id, lane_id+32,
  // lane_id+64, ... We'll compute unnormalized weighted sum, then divide by
  // sum_exp at the end
  const size_t output_offset = bh_offset + q * head_dim_pad;

  // Initialize output to zero
  for (size_t d = lane_id; d < head_dim; d += warpSize) {
    out[output_offset + d] = 0.0f;
  }

  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;
    const size_t value_offset = bh_offset + key_pos * head_dim_pad;

    // Recompute Q·K dot product (same as Pass 1)
    float score =
        warp_qk_dot(Q, K, query_offset, key_offset, head_dim, lane_id) * scale;

    // Compute exp(score - max) - numerically stable
    // Broadcast score to all lanes (only lane 0 has correct value after reduce)
    score = warp_broadcast(score, 0);
    float exp_score = expf(score - max_score);

    // Accumulate sum for normalization (all lanes track this)
    sum_exp += exp_score;

    // Accumulate weighted values (unnormalized)
    // All lanes use same exp_score, each lane handles strided dimensions
    for (size_t d = lane_id; d < head_dim; d += warpSize) {
      out[output_offset + d] += exp_score * V[value_offset + d];
    }
  }

  // ===========================================================================
  // FINAL: Normalize output by sum of exponentials
  // ===========================================================================
  // out = out / sum_exp to convert from weighted sum to softmax-weighted sum
  float inv_sum_exp = 1.0f / sum_exp;

  for (size_t d = lane_id; d < head_dim; d += warpSize) {
    out[output_offset + d] *= inv_sum_exp;
  }
}

// ============================================================================
// Kernel Configuration
// ============================================================================
#define THREADS_PER_BLOCK_X 32     // One full warp per query
#define THREADS_PER_BLOCK_Y TILE_Q // TILE_Q queries per block
#define THREADS_PER_BLOCK_Z 1

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

  // Grid dimensions:
  // x = 1 (unused, warp parallelism is within-block)
  // y = ceil(seq_len / TILE_Q) query tiles
  // z = batch * n_heads
  size_t blocks_x = 1;
  size_t blocks_y = CEIL_DIV(dims.seq_len, THREADS_PER_BLOCK_Y);
  size_t blocks_z = CEIL_DIV(dims.batch * dims.n_heads, THREADS_PER_BLOCK_Z);

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

// v3 does not require workspace - all computation is register-based
size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  (void)dims; // Unused
  return 0;
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  (void)workspace; // Unused in v3 - kept for API compatibility

  CudaConfig config = make_cuda_config(dims);

  VERBOSE_PRINT("CUDA Debug: Thread block (%d,%d,%d), Grid (%d,%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.threads_per_block.z, config.number_of_blocks.x,
                config.number_of_blocks.y, config.number_of_blocks.z);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, dims);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    fprintf(stderr, "Block dimensions: (%d,%d,%d)\n",
            config.threads_per_block.x, config.threads_per_block.y,
            config.number_of_blocks.z);
  }
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
