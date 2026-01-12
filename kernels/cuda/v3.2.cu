#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32
#define TILE_Q 8
#define TILE_K WARP_SIZE
#define TILE_D WARP_SIZE
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

  // Initial santity check assert
  static_assert(TILE_D == WARP_SIZE, "TILE_D must equal WARP_SIZE");
  assert(WARP_SIZE == warpSize);
  assert(blockDim.x == TILE_D);
  assert(blockDim.y == TILE_Q);

  const int lane_id = threadIdx.x; // 0..31
  const int warp_id = threadIdx.y; // 0..TILE_Q -1

  const int q = blockIdx.x * TILE_Q + warp_id;
  const int bh = blockIdx.y;

  const size_t head_dim = dims.head_dim;
  const size_t seq_len = dims.seq_len;

  if (q >= seq_len || bh >= (dims.batch * dims.n_heads))
    return;

  const float scale = rsqrtf((float)head_dim);
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * seq_len * head_dim_pad) +
                           h * (seq_len * head_dim_pad);
  const size_t q_offset = bh_offset + q * head_dim_pad;

  // Initialize output accumulator
  for (size_t d = lane_id; d < head_dim; d += TILE_D)
    out[q_offset + d] = 0.0f;

  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  const int num_k_tiles = CEIL_DIV(seq_len, TILE_K);
  const int num_d_tiles = CEIL_DIV(head_dim, TILE_D);

  // Loop over K tiles
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    for (int k_offset_in_tile = 0; k_offset_in_tile < TILE_K;
         ++k_offset_in_tile) {
      // Compute the global index for the key
      const int k = k_tile * TILE_K + k_offset_in_tile;

      // Skip -> for causal mask
      if (k > q)
        break;

      // Compute the offset inside K matrix
      const size_t k_offset = bh_offset + k * head_dim_pad;
      float dot_partial = 0.0f;

      // Loop over head-dim tiles to compute partial dot products
      for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
        const int d = d_tile * TILE_D + lane_id;
        // The entire warp will compute this
        if (d < head_dim) {
          dot_partial += Q[q_offset + d] * K[k_offset + d];
        }
      }

      // Reduce across the warp only after we computed each partial fully
      float score = warp_reduce_sum(dot_partial) * scale;

      float new_max = fmaxf(softmax_max, score);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score - new_max);

      softmax_sum = softmax_sum * alpha + weight;

      // Accumulate V
      for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
        const int d = d_tile * TILE_D + lane_id;
        if (d < head_dim) {
          out[q_offset + d] =
              out[q_offset + d] * alpha + weight * V[k_offset + d];
        }
      }
      softmax_max = new_max;
    }
  }

  // Final output normalization
  float inv_sum = 1.0f / softmax_sum;
  for (size_t d = lane_id; d < head_dim; d += TILE_D)
    out[q_offset + d] *= inv_sum;
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

  // dim3 block(WARP_SIZE, TILE_K);
  dim3 block(TILE_D, TILE_Q);
  size_t blocks_x = CEIL_DIV(dims.seq_len, TILE_Q); // Blocks Q
  dim3 grid(blocks_x, dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
