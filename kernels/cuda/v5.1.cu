#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5.1: K-Tiling + 4-Key Processing (Strided Access)
// ============================================================================
// Building on v5's K-tiling and two-pass approach, this version adds 4-key
// per iteration processing for instruction-level parallelism (ILP).
//
// Key changes from v5:
// - Process 4 keys per inner loop iteration (from v4.1)
// - Compute 4 dot products in parallel before warp reductions
// - Better ILP: hide warp reduction latency with independent computations
// - Keep strided access pattern (lane_id + i * WARP_SIZE)
// - Keep two-pass per tile approach (compute scores, then accumulate)
//
// Performance improvements expected:
// - 2-3x speedup from 4-key ILP (hides warp reduction latency)
// - 10-15% speedup from loop unrolling
//
// Supported head_dim: up to 128
// ============================================================================

#define WARP_MASK 0xffffffff
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define TILE_K 32        // Keys processed per tile
#define MAX_D_PER_LANE 4 // Support up to head_dim=128
#define KEYS_PER_ITER 4  // Process 4 keys per iteration

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

  static_assert(TILE_K <= WARP_SIZE,
                "TILE_K must be smaller or to equal warp size");

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
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out_accum[i] = 0.0f;
    q_reg[i] = (d < head_dim) ? Q[q_offset + d] : 0.0f;
  }

  // Process keys in tiles
  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int k_start = tile * TILE_K;
    const int k_end = min(k_start + TILE_K, num_keys);
    const int tile_size = k_end - k_start;

    // ================================================================
    // Pass 1: Compute all Q·K scores for the tile, find tile max
    // ================================================================
    int k_idx = 0;
    for (; k_idx + KEYS_PER_ITER <= tile_size; k_idx += KEYS_PER_ITER) {
      const int k0 = k_start + k_idx;
      const int k1 = k0 + 1;
      const int k2 = k0 + 2;
      const int k3 = k0 + 3;

      const size_t k_offset0 = bh_offset + k0 * head_dim_pad;
      const size_t k_offset1 = bh_offset + k1 * head_dim_pad;
      const size_t k_offset2 = bh_offset + k2 * head_dim_pad;
      const size_t k_offset3 = bh_offset + k3 * head_dim_pad;

      // Compute 4 dot products in parallel (better ILP)
      float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;

#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          float q_val = q_reg[i];
          dot0 += q_val * K[k_offset0 + d];
          dot1 += q_val * K[k_offset1 + d];
          dot2 += q_val * K[k_offset2 + d];
          dot3 += q_val * K[k_offset3 + d];
        }
      }

      // Reduce all 4 across warp lanes
      float score0 = warp_reduce_sum_xor(dot0) * scale;
      float score1 = warp_reduce_sum_xor(dot1) * scale;
      float score2 = warp_reduce_sum_xor(dot2) * scale;
      float score3 = warp_reduce_sum_xor(dot3) * scale;

      // Only lane 0 writes to shared memory (avoids redundant writes)
      if (lane_id == 0) {
        scores[warp_id][k_idx] = score0;
        scores[warp_id][k_idx + 1] = score1;
        scores[warp_id][k_idx + 2] = score2;
        scores[warp_id][k_idx + 3] = score3;
      }
    }

    // Handle remaining keys (0-3)
    for (; k_idx < tile_size; ++k_idx) {
      const int k = k_start + k_idx;
      const size_t k_offset = bh_offset + k * head_dim_pad;

      float dot_partial = 0.0f;
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          dot_partial += q_reg[i] * K[k_offset + d];
        }
      }

      float score = warp_reduce_sum_xor(dot_partial) * scale;
      if (lane_id == 0) {
        scores[warp_id][k_idx] = score;
      }
    }

    // ================================================================
    // Pass 2: Compute tile max and accumulate weighted V
    // ================================================================

    // Each lane loads ONE score from shared memory
    float score = (lane_id < tile_size) ? scores[warp_id][lane_id] : -FLT_MAX;

    // Reduce across TILE_K to get tile max
    float tile_max = warp_reduce_max_xor(score);

    // Online softmax update for this tile
    float new_max = fmaxf(running_max, tile_max);
    float alpha = expf(running_max - new_max);

    // Rescale previous accumulator
    running_sum *= alpha;
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      out_accum[i] *= alpha;
    }

    // Second pass: accumulate weighted V using pre-computed scores
    k_idx = 0;
    for (; k_idx + KEYS_PER_ITER <= tile_size; k_idx += KEYS_PER_ITER) {
      const int k0 = k_start + k_idx;
      const int k1 = k0 + 1;
      const int k2 = k0 + 2;
      const int k3 = k0 + 3;

      const size_t k_offset0 = bh_offset + k0 * head_dim_pad;
      const size_t k_offset1 = bh_offset + k1 * head_dim_pad;
      const size_t k_offset2 = bh_offset + k2 * head_dim_pad;
      const size_t k_offset3 = bh_offset + k3 * head_dim_pad;

      // Read 4 scores from shared memory and compute weights
      float weight0 = expf(scores[warp_id][k_idx] - new_max);
      float weight1 = expf(scores[warp_id][k_idx + 1] - new_max);
      float weight2 = expf(scores[warp_id][k_idx + 2] - new_max);
      float weight3 = expf(scores[warp_id][k_idx + 3] - new_max);

      // Update the running sum (4 weights at once)
      running_sum += weight0 + weight1 + weight2 + weight3;

      // Accumulate 4 weighted V vectors
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] +=
              weight0 * V[k_offset0 + d] + weight1 * V[k_offset1 + d] +
              weight2 * V[k_offset2 + d] + weight3 * V[k_offset3 + d];
        }
      }
    }

    // Handle remaining keys (0-3)
    for (; k_idx < tile_size; ++k_idx) {
      const int k = k_start + k_idx;
      const size_t k_offset = bh_offset + k * head_dim_pad;

      float weight = expf(scores[warp_id][k_idx] - new_max);
      running_sum += weight;

#pragma unroll
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
#pragma unroll
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

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
