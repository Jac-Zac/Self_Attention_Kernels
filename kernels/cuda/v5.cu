
#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// FlashAttention-v2 style causal MHSA (FP32)
// - One warp per query
// - K/V tiled into shared memory
// - Online softmax across tiles
// - Register-resident output accumulator
//
// Constraints:
//   head_dim <= 128
//   MAX_D_PER_LANE * 32 == 128
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // 4 * 32 = 128
#define TILE_K 8         // Tune: 8 or 16 on V100

// -----------------------------------------------------------------------------
// Warp reductions
// -----------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(WARP_MASK, v, offset);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(WARP_MASK, v, offset));
  return v;
}

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  const int lane_id = threadIdx.x; // [0,31]
  const int warp_id = threadIdx.y; // [0, WARPS_PER_BLOCK)

  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q_idx >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const int hd_pad = (int)dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = (size_t)b * (dims.n_heads * dims.seq_len * hd_pad) +
                           (size_t)h * (dims.seq_len * hd_pad);

  const size_t q_offset = bh_offset + (size_t)q_idx * hd_pad;
  const size_t out_offset = q_offset;

  // Saved scores in shared memory
  __shared__ float scores[WARPS_PER_BLOCK][TILE_K];

  // Warp-private registers
  float q_reg[MAX_D_PER_LANE];
  float out_acc[MAX_D_PER_LANE];

#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; ++i) {
    int d = lane_id + i * WARP_SIZE;
    q_reg[i] = (d < head_dim) ? Q[q_offset + d] : 0.f;
    out_acc[i] = 0.f;
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.f;

  // Loop over K/V tiles
  for (int k_tile = 0; k_tile <= q_idx; k_tile += TILE_K) {
    // -------------------------------------------------------------------------
    // load of K/V tiles in the future each warp can do it in a cooperative way
    // perhaps
    // -------------------------------------------------------------------------
    // First pass: tile max
    float tile_max = -FLT_MAX;
    for (int k_idx = 0; k_idx < TILE_K; ++k_idx) {
      float dot = 0.f;
      int k = k_tile + k_idx;

      const size_t k_offset = bh_offset + k * hd_pad;
      for (int i = 0; i < MAX_D_PER_LANE; ++i) {
        int d = i * WARP_SIZE + lane_id;
        if (d < head_dim) {
          dot += q_reg[i] * K[k_offset + d];
        }
      }

      float score = warp_reduce_sum(dot) * scale;
      tile_max = fmaxf(tile_max, score);

      if (lane_id == 0) {
        scores[warp_id][k_idx] = score;
      }
    }
    // Wait to make sure all of the warp have written to score
    __syncwarp(WARP_MASK);

    // Online softmax rescale
    float new_max = fmaxf(running_max, tile_max);
    float alpha = expf(running_max - new_max);

    running_sum *= alpha;
    for (int i = 0; i < MAX_D_PER_LANE; ++i)
      out_acc[i] *= alpha;

    // Do the multiplication up to tile size
    const int tile_size = min(TILE_K, q_idx + 1 - k_tile);

    // Second pass: accumulate V
    for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
      // for (int k_idx = 0; k_idx < TILE_K; ++k_idx) {

      float weight = expf(scores[warp_id][k_idx] - new_max);
      const int k = k_tile + k_idx;
      const size_t k_offset = bh_offset + k * hd_pad;

      running_sum += weight;
      for (int i = 0; i < MAX_D_PER_LANE; ++i) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_acc[i] += weight * V[k_offset + d];
        }
      }
    }

    running_max = new_max;
  }

  // Write output
  float inv_sum = 1.f / running_sum;
  for (int i = 0; i < MAX_D_PER_LANE; ++i) {
    int d = lane_id + i * WARP_SIZE;
    if (d < head_dim)
      out[out_offset + d] = out_acc[i] * inv_sum;
  }
}

// -----------------------------------------------------------------------------
// Host interface (unchanged ABI)
// -----------------------------------------------------------------------------
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
