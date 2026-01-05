#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// Changes from v1:
// 1. Warp-level parallelism for dot products:
//    - Each warp (32 threads) collaboratively computes Q·K dot products
//    - Threads handle strided dimensions: thread i processes {i, i+32, ...}
//    - Enables coalesced memory access across warp
//    - Uses warp shuffle intrinsics (__shfl_down_sync) for efficient reduction
//
// 2. Grid configuration optimization:
//    - blockIdx.x now maps directly to query positions (blocks_x = seq_len)
//
// 3. Warp reduction primitives:
//    - warp_reduce_sum: log2(32)=5 steps using shuffle-down
//    - warp_broadcast: distributes lane 0 result to all lanes
//
// Performance impact:
// - Dot product: parallelism within each query computation
// - Memory: Coalesced reads for Q, K, V across warp
// - Tradeoff: Requires warp synchronization overhead
//
// NOTE: o allow coalescing the threads within a warp have to access consecutive
// addresses, but the accesses don’t have to be consecutive within-warp.

// Full warp mask for synchronization
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum(float val) {
  // Parallel reduction: log2(32) = 5 steps
  // Step 1: lanes 0-15 add from lanes 16-31 (offset=16)
  // Step 2: lanes 0-7  add from lanes 8-15  (offset=8)
  // Step 3: lanes 0-3  add from lanes 4-7   (offset=4)
  // Step 4: lanes 0-1  add from lanes 2-3   (offset=2)
  // Step 5: lane  0    adds from lane  1    (offset=1)
  // Result: lane 0 holds complete sum

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(WARP_MASK, val, offset);
  }
  return val;
}

__inline__ __device__ float warp_broadcast(float val, int src_lane) {
  // Broadcast value from src_lane to all lanes in warp
  return __shfl_sync(WARP_MASK, val, src_lane);
}

__inline__ __device__ float warp_reduce_max(float val) {
  // Parallel reduction for maximum: log2(32) = 5 steps
  // Each step halves the active threads, comparing and keeping the max
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(WARP_MASK, val, offset));
  }
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // 2D parallelization:
  // x dimension: query position (one block per query)
  // y dimension: (batch, head) pairs
  const int bh = blockIdx.y * blockDim.y + threadIdx.y;
  const int q = blockIdx.x;        // Direct mapping: one block per query
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

  // Triangular workspace: each (batch, head) stores only causal attention
  // weights Query q needs weights for keys 0..q (total: q+1 weights)
  const size_t workspace_per_bh = seq_len * (seq_len + 1) / 2;
  const size_t bh_workspace_offset = (b * num_heads + h) * workspace_per_bh;
  const size_t triangular_offset = q * (q + 1) / 2;

  float *RESTRICT aw = attn_weights + bh_workspace_offset + triangular_offset;

  // Tensor offsets: [batch, head, seq, head_dim]
  const size_t bh_offset =
      b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // STEP 1: Compute Q·K scores using warp-parallel dot products
  // ===========================================================================
  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;

    // Warp-parallel dot product:
    // - Thread i processes dimensions: i, i+32, i+64, ...
    // - Ensures coalesced memory access across the warp
    float dot_partial = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += warpSize) {
      dot_partial += Q[query_offset + d] * K[key_offset + d];
    }

    // Warp reduction: All threads participate, result in lane 0
    // __shfl_down_sync provides implicit synchronization during reduction
    float dot_product = warp_reduce_sum(dot_partial);
    float score = dot_product * scale;

    // Only lane 0 writes the score
    if (lane_id == 0) {
      aw[key_pos] = score;
    }
  }

  // Why needed: Lane 0 wrote aw[], now ALL lanes will read it
  // Race condition: Write-After-Read (WAR) hazard without sync
  __syncwarp(WARP_MASK);

  // ===========================================================================
  // STEP 2: Parallel max-finding
  // ===========================================================================
  // Strategy: All threads participate in finding maximum
  // - Each thread scans strided subset: lane i reads {i, i+32, i+64, ...}
  // - Warp reduction combines local maxima into global maximum
  float local_max = -FLT_MAX;
  for (size_t key_pos = lane_id; key_pos <= q; key_pos += warpSize) {
    local_max = fmaxf(local_max, aw[key_pos]);
  }

  // Warp reduction: Find true maximum across all lanes
  // After this, lane 0 holds the global max
  float max_score = warp_reduce_max(local_max);

  // Broadcast max to all lanes (needed for exp computation)
  max_score = warp_broadcast(max_score, 0);

  // ===========================================================================
  // STEP 3: Parallel softmax computation
  // ===========================================================================
  // Strategy: Compute exp(score - max) and sum in parallel
  // - Numerically stable: subtracting max prevents overflow
  // - Each thread processes strided subset

  float local_sum_exp = 0.0f;
  for (size_t key_pos = lane_id; key_pos <= q; key_pos += warpSize) {
    float exp_val = expf(aw[key_pos] - max_score);
    aw[key_pos] = exp_val; // Overwrite scores with exp values
    local_sum_exp += exp_val;
  }

  // Why needed: All threads wrote to aw[], now lane 0 will reduce sum
  // Also: Ensures exp values are visible before normalization step
  __syncwarp(WARP_MASK);

  // Reduce partial sums to get total
  float sum_exp = warp_reduce_sum(local_sum_exp);

  // Broadcast sum to all lanes for normalization
  sum_exp = warp_broadcast(sum_exp, 0);
  const float inv_sum_exp = 1.0f / sum_exp;

  // ===========================================================================
  // STEP 4: Weighted sum of values (warp-parallel across head_dim)
  // ===========================================================================
  const size_t output_offset = bh_offset + q * head_dim_pad;

  // Initialize output: Each thread handles strided dimensions
  for (size_t d = lane_id; d < head_dim; d += warpSize) {
    out[output_offset + d] = 0.0f;
  }

  // Accumulate weighted values
  // Memory access pattern: All threads read same aw[key_pos] (broadcast)
  // then access strided dimensions of V (coalesced)
  for (size_t key_pos = 0; key_pos <= q; key_pos++) {
    const size_t value_offset = bh_offset + key_pos * head_dim_pad;
    float const normalized_weight = aw[key_pos] * inv_sum_exp;

    // Warp-parallel accumulation across head dimensions
    for (size_t d = lane_id; d < head_dim; d += warpSize) {
      out[output_offset + d] += normalized_weight * V[value_offset + d];
    }
  }
  // No final syncwarp needed: kernel exit provides implicit synchronization
}

// ============================================================================
// Kernel Configuration
// ============================================================================
#define THREADS_PER_BLOCK_X 32 // One full warp per block
#define THREADS_PER_BLOCK_Y 1

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

  // 2D grid: x = seq_len (one block per query), y = batch * n_heads
  size_t blocks_x = dims.seq_len;
  size_t blocks_y = CEIL_DIV(dims.batch * dims.n_heads, THREADS_PER_BLOCK_Y);

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
