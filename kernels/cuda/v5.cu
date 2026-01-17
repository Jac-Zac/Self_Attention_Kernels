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
//
// Shared memory layout per warp: float scores[TILE_K]
// ============================================================================

#define WARP_MASK 0xffffffff
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define TILE_K 32        // Keys processed per tile (tune for occupancy)
#define MAX_D_PER_LANE 4 // Support up to head_dim=128

// Shared memory for per-warp score buffers
extern __shared__ float smem[];

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

  // Each warp gets its own score buffer in shared memory
  float *tile_scores = smem + warp_id * TILE_K;

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
    // Causal mask: don't go past q_idx
    const int k_end = min(k_start + TILE_K, num_keys);
    const int tile_size = k_end - k_start;

    // ========================================================================
    // Pass 1: Compute Q·K scores for all keys in this tile
    // ========================================================================
    // Process keys sequentially within tile, parallelize dot product across
    // lanes Store scores in shared memory for Pass 2

    float tile_max = -FLT_MAX;

    for (int k_local = 0; k_local < tile_size; ++k_local) {
      const int k = k_start + k_local;
      const size_t k_offset = bh_offset + k * head_dim_pad;

      // Q·K dot product parallelized across lanes
      float dot_partial = 0.0f;
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          dot_partial += q_reg[i] * K[k_offset + d];
        }
      }

      // Reduce across warp to get full dot product
      float score = warp_reduce_sum_xor(dot_partial) * scale;

      // Store score in shared memory (all lanes have same value after reduce)
      if (lane_id == 0) {
        tile_scores[k_local] = score;
      }

      // Track tile maximum
      tile_max = fmaxf(tile_max, score);
    }

    // Sync to ensure all scores are written
    __syncwarp();

    // ========================================================================
    // Online softmax update for this tile
    // ========================================================================
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

    // ========================================================================
    // Pass 2: Compute weights and accumulate V
    // ========================================================================
    for (int k_local = 0; k_local < tile_size; ++k_local) {
      const int k = k_start + k_local;
      const size_t k_offset = bh_offset + k * head_dim_pad;

      // Read score from shared memory and compute weight
      float score = tile_scores[k_local];
      float weight = expf(score - new_max);

      running_sum += weight;

      // Accumulate weighted V
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

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK), dims.batch * dims.n_heads);

  // Shared memory: each warp needs TILE_K floats for score buffer
  size_t smem_size = WARPS_PER_BLOCK * TILE_K * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims);
}
#endif
