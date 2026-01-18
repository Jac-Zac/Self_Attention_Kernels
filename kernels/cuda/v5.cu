#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5: FlashAttention-2 Style - Sliced-Q with K/V Shared Memory Tiling
// ============================================================================
// This implements the FlashAttention-2 algorithm with:
// - Q split across warps (sliced-Q scheme)
// - K/V loaded to shared memory and reused by all warps
// - Online softmax (flash-style) computed incrementally
//
// Key changes from v4:
// - Process TILE_Q queries per block instead of 1 query per warp
// - Each warp handles QUERIES_PER_WARP queries
// - K and V tiles loaded cooperatively to shared memory
// - Reuse K/V tiles across all warps in the block
// - Significant reduction in global memory bandwidth
//
// Supported head_dim: up to 128 (4 floats per lane * 32 lanes)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // Support up to head_dim=128
#define TILE_Q 64        // Queries per block
#define TILE_K 32        // Keys per tile

// Calculate queries per warp
#define QUERIES_PER_WARP (TILE_Q / WARPS_PER_BLOCK)

#define MAX_HEAD_DIM MAX_D_PER_LANE * WARP_SIZE

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__inline__ __device__ float warp_reduce_max_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = fmaxf(val, __shfl_xor_sync(WARP_MASK, val, mask));
  return val;
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int tid = warp_id * WARP_SIZE + lane_id;

  const int q_block = blockIdx.x;
  const int bh = blockIdx.y;

  if (bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Query range for this block
  const int q_start = q_block * TILE_Q;
  const int q_end_block = min(q_start + TILE_Q, (int)dims.seq_len);
  const int num_queries = q_end_block - q_start;

  // Query range for this warp
  const int q_warp_start = q_start + warp_id * QUERIES_PER_WARP;
  const int q_warp_end = min(q_warp_start + QUERIES_PER_WARP, q_end_block);
  const int num_warp_queries = q_warp_end - q_warp_start;

  // ====== Shared Memory ======
  // K and V tiles: [TILE_K, head_dim]
  __shared__ float K_smem[TILE_K * MAX_HEAD_DIM];
  __shared__ float V_smem[TILE_K * MAX_HEAD_DIM];

  // ====== Register Allocation ======
  // Q and O for each query this warp handles
  // Layout: [QUERIES_PER_WARP][MAX_D_PER_LANE]
  float Q_reg[QUERIES_PER_WARP][MAX_D_PER_LANE];
  float O_reg[QUERIES_PER_WARP][MAX_D_PER_LANE];

  // Online softmax state per query
  float m[QUERIES_PER_WARP];
  float l[QUERIES_PER_WARP];

  // ====== Initialize Registers ======
#pragma unroll
  for (int q_local = 0; q_local < QUERIES_PER_WARP; q_local++) {
    const int q = q_warp_start + q_local;

    // Load Q from global memory
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (q < q_end_block && d < head_dim) {
        const size_t q_offset = bh_offset + q * head_dim_pad + d;
        Q_reg[q_local][i] = Q[q_offset];
        O_reg[q_local][i] = 0.0f;
      } else {
        Q_reg[q_local][i] = 0.0f;
        O_reg[q_local][i] = 0.0f;
      }
    }

    // Initialize softmax state
    m[q_local] = -FLT_MAX;
    l[q_local] = 0.0f;
  }

  // ====== Loop over K blocks (causal: k <= q_end_block) ======
  // Process K blocks until we've seen all keys needed for queries in this block
  const int num_k_blocks = CEIL_DIV(q_end_block, TILE_K);

  for (int k_block = 0; k_block < num_k_blocks; k_block++) {
    const int k_start = k_block * TILE_K;
    const int k_end_block = min(k_start + TILE_K, (int)dims.seq_len);

    // ====== Cooperatively Load K, V to Shared Memory ======
    // All threads in block work together to load K/V tiles
    const int kv_elements = TILE_K * head_dim;
    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;

    for (int idx = tid; idx < kv_elements; idx += threads_per_block) {
      const int k_local = idx / head_dim;
      const int d = idx % head_dim;
      const int k = k_start + k_local;

      if (k < k_end_block && d < head_dim) {
        const size_t kv_offset = bh_offset + k * head_dim_pad + d;
        K_smem[k_local * MAX_HEAD_DIM + d] = K[kv_offset];
        V_smem[k_local * MAX_HEAD_DIM + d] = V[kv_offset];
      } else {
        K_smem[k_local * MAX_HEAD_DIM + d] = 0.0f;
        V_smem[k_local * MAX_HEAD_DIM + d] = 0.0f;
      }
    }

    __syncthreads();

    // ====== Compute for Each Query in Warp ======
    for (int q_local = 0; q_local < QUERIES_PER_WARP; q_local++) {
      const int q = q_warp_start + q_local;
      if (q >= q_end_block)
        break;

      // Causal: only process keys where k <= q
      const int k_end_in_tile = min(k_end_block, q + 1);
      if (k_start >= q + 1)
        continue;

      // Effective range of keys in this tile
      const int k_effective_start = max(k_start, 0);
      const int k_effective_end = k_end_in_tile;

      // ====== Pass: Compute QK Scores and Update Softmax ======
      for (int k_local = 0; k_local < (k_effective_end - k_effective_start); k_local++) {
        const int k = k_effective_start + k_local;
        const int k_idx = k - k_start;  // Index within the tile

        // Compute Q·K dot product (K from shared memory)
        float dot_partial = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_D_PER_LANE; i++) {
          const int d = lane_id + i * WARP_SIZE;
          if (d < head_dim) {
            dot_partial += Q_reg[q_local][i] * K_smem[k_idx * MAX_HEAD_DIM + d];
          }
        }

        float score = warp_reduce_sum_xor(dot_partial) * scale;

        // Online softmax update
        float new_max = fmaxf(m[q_local], score);
        float alpha = expf(m[q_local] - new_max);
        float weight = expf(score - new_max);

        l[q_local] = l[q_local] * alpha + weight;

        // Update output (V from shared memory)
#pragma unroll
        for (int i = 0; i < MAX_D_PER_LANE; i++) {
          const int d = lane_id + i * WARP_SIZE;
          if (d < head_dim) {
            O_reg[q_local][i] =
                O_reg[q_local][i] * alpha + weight * V_smem[k_idx * MAX_HEAD_DIM + d];
          }
        }

        m[q_local] = new_max;
      }
    }

    __syncthreads();
  }

  // ====== Normalize and Write Output ======
  for (int q_local = 0; q_local < QUERIES_PER_WARP; q_local++) {
    const int q = q_warp_start + q_local;
    if (q >= q_end_block)
      break;

    const float inv_l = 1.0f / l[q_local];

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        const size_t out_offset = bh_offset + q * head_dim_pad + d;
        out[out_offset] = O_reg[q_local][i] * inv_l;
      }
    }
  }
}

size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  (void)dims;
  return 0;
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  (void)workspace;

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid(CEIL_DIV(dims.seq_len, TILE_Q), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
