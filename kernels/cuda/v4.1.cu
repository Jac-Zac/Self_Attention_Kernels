#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4.1: Register-Based Output with Clean Array Storage
// ============================================================================
// Refactored v4 using float[4] arrays for cleaner, more compact code.
// Same algorithm as v4, just better organized.
//
// Key features (same as v4):
// - Online softmax (no workspace needed)
// - Output accumulator in registers
// - Single write to global memory at the end
//
// Code improvements:
// - Q, K, V, output stored in float[4] arrays
// - Cleaner loop structure with indexed access
// - Compiler keeps small arrays in registers (no spills)
//
// Supported head_dim: up to 128 (32 lanes * 4 floats)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  }
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const size_t head_dim_pad) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh_idx = blockIdx.y;

  if (q_idx >= (int)dims.seq_len ||
      bh_idx >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh_idx / dims.n_heads;
  const int h = bh_idx % dims.n_heads;
  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Each lane handles 4 elements at strided positions: lane_id, lane_id+32, ...
  // This matches v4's interleaved access pattern
  float r_Q[4] = {0}, r_K[4], r_V[4], r_out[4] = {0};

  // Load Q once (reused for all keys)
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      r_Q[i] = Q[bh_offset + q_idx * head_dim_pad + d];
    }
  }

  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Process one key at a time (causal: k <= q_idx)
  for (int k = 0; k <= q_idx; k++) {
    const size_t kv_offset = bh_offset + k * head_dim_pad;

    // Load K
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int d = lane_id + i * WARP_SIZE;
      r_K[i] = (d < head_dim) ? K[kv_offset + d] : 0.0f;
    }

    // Dot product Q . K
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      dot += r_Q[i] * r_K[i];
    }
    float score = warp_reduce_sum_xor(dot) * scale;

    // Online softmax update
    float new_max = fmaxf(softmax_max, score);
    float rescale = expf(softmax_max - new_max);
    float weight = expf(score - new_max);
    softmax_sum = softmax_sum * rescale + weight;
    softmax_max = new_max;

    // Load V and accumulate weighted output
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int d = lane_id + i * WARP_SIZE;
      r_V[i] = (d < head_dim) ? V[kv_offset + d] : 0.0f;
      r_out[i] = r_out[i] * rescale + weight * r_V[i];
    }
  }

  // Normalize and write to global memory
  const float inv_sum = 1.0f / softmax_sum;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      out[bh_offset + q_idx * head_dim_pad + d] = r_out[i] * inv_sum;
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

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
