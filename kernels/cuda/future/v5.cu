#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5: K-Tiling with Shared Memory Score Buffer
// ============================================================================
// Building on v4's register-based output accumulator, this version tiles over
// the key sequence dimension to improve memory access patterns.
//
// Key changes from v4:
// - Process keys in tiles of TILE_K instead of one at a time
// - Store tile scores in shared memory for two-pass softmax within tile:
//   Pass 1: Compute all Q·K scores for the tile, find tile max
//   Pass 2: Compute weights and accumulate V contributions
// - Online softmax maintains running max/sum across tiles
// ============================================================================

#define WARP_MASK 0xffffffff
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define TILE_K 32        // Keys processed per tile (tune for occupancy)
#define MAX_D_PER_LANE 4 // Support up to head_dim=128

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

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const size_t q_offset = bh_offset + q * head_dim_pad;
  const size_t out_offset = q_offset;

  // Saved scores in shared memory
  __shared__ float scores[WARPS_PER_BLOCK][TILE_K];

  // Number of tiles needed to cover all keys (causal: 0 to q_idx inclusive)
  const int num_keys = q + 1;
  const int num_k_tiles = CEIL_DIV(num_keys, TILE_K);

  // Online softmax running state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Register-based output accumulator
  float out_accum[MAX_D_PER_LANE];

  // Q loaded into registers once (avoids repeated global memory access)
  float q_reg[MAX_D_PER_LANE];
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out_accum[i] = 0.0f;
    q_reg[i] = (d < head_dim) ? Q[q_offset + d] : 0.0f;
  }

  // Process keys in tiles
  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int k_start = tile * TILE_K;
    // Causal mask: don't go past q_idx
    const int k_end = min(k_start + TILE_K, num_keys);
    const int tile_size = k_end - k_start;

    for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
      const int k = k_start + k_idx;
      const size_t k_offset = bh_offset + k * head_dim_pad;

      // Q·K dot product parallelized across lanes
      // All 32 lanes compute partial dot products
      float dot_partial = 0.0f;
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          dot_partial += q_reg[i] * K[k_offset + d];
        }
      }

      // Reduce across warp lanes to get full dot product
      // After reduction, ALL lanes have the same score value
      // Only lane 0 writes to shared memory to avoid redundant writes
      float score = warp_reduce_sum_xor(dot_partial) * scale;
      if (lane_id == 0) {
        scores[warp_id][k_idx] = score;
      }

      // Note: No __syncthreads() needed here because:
      // 1. Each warp writes to disjoint memory region (scores[warp_id*TILE_K :
      // (warp_id+1)*TILE_K])
      // 2. Warp operations to compute the score implicitly synchronize
    }

    // Each lane loads ONE score from shared memory
    // This "transposes" the scores: after this, lane i holds score[k_i]
    // Lanes beyond tile_size get -FLT_MAX so they don't affect max
    float score = (lane_id < tile_size) ? scores[warp_id][lane_id] : -FLT_MAX;

    // Reduce across TILE_K to get tile max
    // And also implicitly broadcast via xor max
    // All lanes now have the tile_max value
    float tile_max = warp_reduce_max_xor(score);

    // Online softmax update for this tile
    // new_max = max(running_max, tile_max)
    // alpha = exp(running_max - new_max)  // rescale factor for old accumulator
    // Rescale running_sum and out_accum by alpha
    float new_max = fmaxf(running_max, tile_max);
    float alpha = expf(running_max - new_max);

    // Rescale previous accumulator
    running_sum *= alpha;
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      out_accum[i] *= alpha;
    }

    // Second pass: accumulate weighted V using pre-computed scores
    // Note: We do NOT recompute Q·K dot products here - we read from shared
    // memory
    for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
      const int k = k_start + k_idx;
      const size_t k_offset = bh_offset + k * head_dim_pad;

      // Read pre-computed score from shared memory and compute weight
      // All lanes read the same score, then compute same weight
      float weight = expf(scores[warp_id][k_idx] - new_max);

      // Update the running sum
      running_sum += weight;

      // Accumulate weighted V
      // Each lane handles different head_dim elements (parallel across d)
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] += weight * V[k_offset + d];
        }
      }
    }
    running_max = new_max;
  }

  // Normalize and write to global memory (once!)
  float inv_sum = 1.0f / running_sum;
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      out[out_offset + d] = out_accum[i] * inv_sum;
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
  static_assert(TILE_K <= WARP_SIZE,
                "TILE_K must be smaller or to equal warp size");

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
