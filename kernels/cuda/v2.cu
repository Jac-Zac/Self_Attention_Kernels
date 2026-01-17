#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v2: XOR Reduction + Multi-Warp Parallelism
// ============================================================================
// 1. XOR-based warp reduction instead of shuffle-down + broadcast
//    Why: All threads end up with the result directly
//    No need for separate warp_broadcast after reduction
//    Same number of shuffle ops, but simpler code flow
//
//    shuffle-down (v1):        XOR (v2):
//    lane 0 gets result   ->   all lanes get result
//    need broadcast       ->   no broadcast needed
//
// 2. Multiple warps per block (8 warps = 256 threads)
//    Grid shrinks from seq_len blocks to ceil(seq_len/8) blocks
//    Each warp handles one query independently
//
//    - Better GPU occupancy, more parallelism per SM
//    - Allows scheduler to interleave warps
//    - Hides memory latency

// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff

// XOR-based reduction: all threads end up with the sum (no broadcast needed)
// Compare to v2's warp_reduce_sum which only gives result to lane 0
__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  }
  return val;
}

// XOR-based max reduction: all threads end up with the max
__inline__ __device__ float warp_reduce_max_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(WARP_MASK, val, mask));
  }
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     float *RESTRICT attn_weights, const AttentionDims dims) {

  // Thread identification
  // - threadIdx.y: which warp in this block (0 to WARPS_PER_BLOCK -1)
  // - threadIdx.x: lane within warp (0 to WARP_SIZE -1)
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  // Query position: each warp handles one query
  const int query_group = blockIdx.x;
  const int q = query_group * WARPS_PER_BLOCK + warp_id;

  // Batch/head indices
  const int bh = blockIdx.y;

  if (q >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Triangular workspace: each (batch, head) stores only causal attention
  // weights. Query q needs weights for keys 0..q (total: q+1 weights)
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
  for (int key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;
    float dot_partial = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      dot_partial += Q[query_offset + d] * K[key_offset + d];
    }

    // XOR reduction: all threads get the result (no broadcast needed!)
    float score = warp_reduce_sum_xor(dot_partial) * scale;

    if (lane_id == 0) {
      aw[key_pos] = score;
    }
  }
  __syncwarp(WARP_MASK);

  // ===========================================================================
  // STEP 2: Parallel max-finding
  // ===========================================================================
  float local_max = -FLT_MAX;
  for (int key_pos = lane_id; key_pos <= q; key_pos += WARP_SIZE) {
    local_max = fmaxf(local_max, aw[key_pos]);
  }
  float max_score = warp_reduce_max_xor(local_max);

  // STEP 3: Parallel softmax computation
  float local_sum_exp = 0.0f;
  for (int key_pos = lane_id; key_pos <= q; key_pos += WARP_SIZE) {
    float exp_val = expf(aw[key_pos] - max_score);
    aw[key_pos] = exp_val;
    local_sum_exp += exp_val;
  }

  __syncwarp(WARP_MASK);

  // XOR reduction: all threads get the sum (no broadcast needed!)
  float sum_exp = warp_reduce_sum_xor(local_sum_exp);
  const float inv_sum_exp = 1.0f / sum_exp;

  // ===========================================================================
  // STEP 4: Weighted sum of values (warp-parallel across head_dim)
  // ===========================================================================
  const size_t output_offset = bh_offset + q * head_dim_pad;
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
    out[output_offset + d] = 0.0f;
  }

  // Accumulate weighted values (same as v2)
  for (int key_pos = 0; key_pos <= q; key_pos++) {
    const size_t value_offset = bh_offset + key_pos * head_dim_pad;
    float const normalized_weight = aw[key_pos] * inv_sum_exp;

    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      out[output_offset + d] += normalized_weight * V[value_offset + d];
    }
  }
}

// ============================================================================
// Kernel Configuration
// ============================================================================

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(WARP_SIZE, WARPS_PER_BLOCK);

  // Grid: each block handles more queires, y dimension is batch*heads
  size_t query_groups = CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK);
  dim3 number_of_blocks(query_groups, dims.batch * dims.n_heads);

  CudaConfig config;
  config.threads_per_block = threads_per_block;
  config.number_of_blocks = number_of_blocks;

  VERBOSE_PRINT("CUDA: %d warps/block, Block (%d,%d), Grid (%d,%d)\n",
                WARPS_PER_BLOCK, config.threads_per_block.x,
                config.threads_per_block.y, config.number_of_blocks.x,
                config.number_of_blocks.y);
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

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims);
}

#else
#error "This file requires USE_CUDA to be defined"
#endif
