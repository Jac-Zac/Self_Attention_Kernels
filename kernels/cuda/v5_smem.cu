#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v6: K-Tiling with Shared Memory for Scores, K, and V
// ============================================================================
// Building on v5's score tiling, this version adds K/V shared memory tiling
// to reduce global memory bandwidth.
//
// Key changes from v5:
// - K and V tiles loaded into shared memory once per block
// - All WARPS_PER_BLOCK warps reuse the same K/V data from shared memory
// - Reduces global memory reads by ~8× (WARPS_PER_BLOCK factor)
// - Expected 20-40% speedup depending on memory bandwidth bottleneck
// ============================================================================

#define WARP_MASK 0xffffffff
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define TILE_K 32        // Keys processed per tile (tune for occupancy)
#define MAX_D_PER_LANE 4 // Support up to head_dim=128

#define MAX_HEAD_DIM MAX_D_PER_LANE *WARP_SIZE // Maximum head dimension -> 128

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

__inline__ __device__ float warp_broadcast(float val, int src_lane) {
  return __shfl_sync(WARP_MASK, val, src_lane);
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int tid = warp_id * WARP_SIZE + lane_id; // Thread ID within block

  const int bh = blockIdx.y;

  // Block-level early return (safe - all threads exit together)
  if (bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const bool valid_q = (q < (int)dims.seq_len);

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const size_t q_offset = bh_offset + q * head_dim_pad;
  const size_t out_offset = q_offset;

  // ====== Shared Memory Allocation ======
  // Scores: [WARPS_PER_BLOCK, TILE_K]
  __shared__ float scores[TILE_K * WARPS_PER_BLOCK];

  // K and V tiles: [TILE_K, MAX_HEAD_DIM]
  // These are shared across all warps in the block!
  __shared__ float K_tile[TILE_K * MAX_HEAD_DIM];
  __shared__ float V_tile[TILE_K * MAX_HEAD_DIM];

  // Max tiles for the block (all warps must agree for __syncthreads__)
  const int max_q_in_block =
      min((int)((blockIdx.x + 1) * WARPS_PER_BLOCK - 1), (int)dims.seq_len - 1);
  const int max_k_tiles = CEIL_DIV(max_q_in_block + 1, TILE_K);

  // Per-warp tile count (for conditional computation)
  const int num_keys = q + 1;
  const int num_k_tiles = valid_q ? CEIL_DIV(num_keys, TILE_K) : 0;

  // Online softmax running state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Register-based output accumulator
  float out_accum[MAX_D_PER_LANE];

  // Q loaded into registers once (avoids repeated global memory access)
  float q_reg[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out_accum[i] = 0.0f;
    q_reg[i] = (valid_q && d < head_dim) ? Q[q_offset + d] : 0.0f;
  }

  // Process keys in tiles (all warps iterate max_k_tiles for __syncthreads__)
  for (int tile = 0; tile < max_k_tiles; ++tile) {
    const int k_start = tile * TILE_K;
    // Block-wide k_end for cooperative loading
    const int k_end_block = min(k_start + TILE_K, max_q_in_block + 1);

    // ====== Cooperative Loading of K and V Tiles ======
    // All threads in the block work together to load K/V tiles
    // This is the KEY optimization: load once, reuse 8 times (across warps)
    const int total_elements = TILE_K * head_dim;
    const int threads_per_block = WARPS_PER_BLOCK * WARP_SIZE;

    for (int idx = tid; idx < total_elements; idx += threads_per_block) {
      const int k_local = idx / head_dim; // Which key in the tile
      const int d = idx % head_dim;       // Which dimension
      const int k = k_start + k_local;    // Global key index

      if (k < k_end_block && d < head_dim) {
        const size_t k_offset = bh_offset + k * head_dim_pad;
        K_tile[k_local * MAX_HEAD_DIM + d] = K[k_offset + d];
        V_tile[k_local * MAX_HEAD_DIM + d] = V[k_offset + d];
      } else {
        // Pad with zeros for out-of-bounds
        K_tile[k_local * MAX_HEAD_DIM + d] = 0.0f;
        V_tile[k_local * MAX_HEAD_DIM + d] = 0.0f;
      }
    }

    // Wait for all threads to finish loading K/V tiles
    __syncthreads();

    // ====== Only warps that need this tile compute ======
    if (valid_q && tile < num_k_tiles) {
      const int k_end = min(k_start + TILE_K, num_keys);
      const int tile_size = k_end - k_start;

      // ====== Pass 1: Compute Q·K Scores (using K_tile) ======
      for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
        // Q·K dot product parallelized across lanes
        // Now reading K from SHARED MEMORY instead of global!
        float dot_partial = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_D_PER_LANE; i++) {
          const int d = lane_id + i * WARP_SIZE;
          if (d < head_dim) {
            // KEY CHANGE: Read from K_tile (shared memory) not global K
            dot_partial += q_reg[i] * K_tile[k_idx * MAX_HEAD_DIM + d];
          }
        }

        // Reduce across warp lanes to get full dot product
        float score = warp_reduce_sum_xor(dot_partial) * scale;

        if (lane_id == 0) {
          scores[k_idx + (warp_id * TILE_K)] = score;
        }
      }

      // Each lane loads ONE score from shared memory
      float score =
          (lane_id < tile_size) ? scores[lane_id + warp_id * TILE_K] : -FLT_MAX;

      // Reduce across TILE_K to get tile max
      float tile_max = warp_reduce_max_xor(score);

      // ====== Online Softmax Update ======
      float new_max = fmaxf(running_max, tile_max);
      float alpha = expf(running_max - new_max);

      // Rescale previous accumulator
      running_sum *= alpha;
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        out_accum[i] *= alpha;
      }

      // ====== Pass 2: Accumulate Weighted V (using V_tile) ======
      for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
        // Read pre-computed score from shared memory and compute weight
        float weight = expf(scores[k_idx + (warp_id * TILE_K)] - new_max);

        running_sum += weight;

        // Accumulate weighted V
#pragma unroll
        for (int i = 0; i < MAX_D_PER_LANE; i++) {
          const int d = lane_id + i * WARP_SIZE;
          if (d < head_dim) {
            // KEY CHANGE: Read from V_tile (shared memory) not global V
            out_accum[i] += weight * V_tile[k_idx * MAX_HEAD_DIM + d];
          }
        }
      }

      running_max = new_max;
    }

    // Wait for all warps to finish reading K/V tiles before loading next tile
    __syncthreads();
  }

  // Only valid warps write output
  if (valid_q) {
    // Normalize and write to global memory (once!)
    float inv_sum = 1.0f / running_sum;
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        out[out_offset + d] = out_accum[i] * inv_sum;
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

  // Sanity check assertions
  static_assert(TILE_K <= WARP_SIZE,
                "TILE_K must be smaller or equal to warp size");
  assert(dims.head_dim <= MAX_HEAD_DIM);

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
