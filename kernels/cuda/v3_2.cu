#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v3_2: Simple shared memory approach
//
// Strategy: Process one K position at a time, load K/V into smem per d_tile.
// All TILE_Q warps share the same K/V data (smem reuse across warps).
// Interleave K dot-product and V accumulation in the same d_tile loop
// to avoid reloading V.
// ============================================================================

#define WARP_SIZE 32
#define TILE_Q 8
#define TILE_D 32
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const size_t head_dim_pad) {

  // Block: (TILE_D, TILE_Q) = (32, 8) = 256 threads
  // - warp_id (threadIdx.y): which query position in tile
  // - lane_id (threadIdx.x): which dimension element
  //
  // All TILE_Q warps process the SAME K/V positions, so we load K/V once
  // into shared memory and all warps read from it.

  static_assert(TILE_D == WARP_SIZE, "TILE_D must equal WARP_SIZE");

  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;

  const int q = blockIdx.x * TILE_Q + warp_id;
  const int bh = blockIdx.y;

  const size_t head_dim = dims.head_dim;
  const size_t seq_len = dims.seq_len;

  // Shared memory: one K row and one V row per d_tile
  __shared__ float K_smem[TILE_D];
  __shared__ float V_smem[TILE_D];

  const bool valid_q = (q < seq_len) && (bh < dims.batch * dims.n_heads);

  const float scale = rsqrtf((float)head_dim);
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * seq_len * head_dim_pad) +
                           h * (seq_len * head_dim_pad);
  const size_t q_offset = bh_offset + q * head_dim_pad;

  // Initialize output
  if (valid_q) {
    for (size_t d = lane_id; d < head_dim; d += TILE_D)
      out[q_offset + d] = 0.0f;
  }

  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  const int num_d_tiles = CEIL_DIV(head_dim, TILE_D);

  // Max q in block - for early k loop termination
  const int max_q_in_block = min((int)((blockIdx.x + 1) * TILE_Q - 1), 
                                  (int)(seq_len - 1));

  // Loop over all key positions (up to max needed by any warp in block)
  for (int k = 0; k <= max_q_in_block && k < (int)seq_len; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    // === Phase 1: Compute full dot product Q[q] · K[k] ===
    float dot_partial = 0.0f;

    for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
      const int d_global = d_tile * TILE_D + lane_id;

      // Cooperative load: warp 0 loads K into smem
      if (warp_id == 0) {
        K_smem[lane_id] = (d_global < head_dim) ? K[k_offset + d_global] : 0.0f;
      }
      __syncthreads();

      // All warps compute partial dot
      if (valid_q && d_global < head_dim) {
        dot_partial += Q[q_offset + d_global] * K_smem[lane_id];
      }

      __syncthreads();
    }

    // Reduce across warp to get full dot product
    float score = warp_reduce_sum(dot_partial) * scale;

    // Causal mask: each warp has its own q
    if (!valid_q || k > q)
      continue;

    // Online softmax update
    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;

    // === Phase 2: Accumulate weighted V[k] into output ===
    for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
      const int d_global = d_tile * TILE_D + lane_id;

      // Cooperative load: warp 0 loads V into smem
      if (warp_id == 0) {
        V_smem[lane_id] = (d_global < head_dim) ? V[k_offset + d_global] : 0.0f;
      }
      __syncthreads();

      // All warps accumulate
      if (d_global < head_dim) {
        out[q_offset + d_global] =
            out[q_offset + d_global] * alpha + weight * V_smem[lane_id];
      }

      __syncthreads();
    }

    softmax_max = new_max;
  }

  // Final normalization
  if (valid_q) {
    float inv_sum = 1.0f / softmax_sum;
    for (size_t d = lane_id; d < head_dim; d += TILE_D)
      out[q_offset + d] *= inv_sum;
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

  dim3 block(TILE_D, TILE_Q);
  dim3 grid(CEIL_DIV(dims.seq_len, TILE_Q), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
