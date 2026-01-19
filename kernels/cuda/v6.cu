#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v6: Pipelined K/V Loads with Explicit Prefetching
// ============================================================================
// Building on v5's vectorized approach, this version adds explicit prefetching
// to better overlap memory loads with computation.
//
// Key insight: The memory system can have multiple outstanding loads. By
// issuing loads early, we can hide memory latency behind computation.
//
// Pipeline structure (for 4 keys):
//   Iteration 0: Load K[0,1], Compute nothing yet
//   Iteration 1: Load K[2,3], Load V[0,1], Compute dot[0,1]
//   Iteration 2: Load K[4,5], Load V[2,3], Compute softmax[0,1], dot[2,3]
//   ...
//
// This creates a 2-stage pipeline:
//   Stage 1: K loads (2 keys ahead)
//   Stage 2: V loads + dot product + softmax update
//
// Additional optimizations:
// - Use __ldg for read-only data (texture cache path)
// - Minimize register pressure by reusing variables
// - Unroll inner loops for better scheduling
//
// Supported head_dim: up to 128 (32 lanes * 4 floats)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q_idx >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const int d_base = lane_id * 4;
  // Handle edge case: only mark active if we have at least 1 valid element
  const bool lane_active = (d_base < head_dim);

  // Load Q once
  float4 q_vec = make_float4(0.f, 0.f, 0.f, 0.f);
  if (lane_active) {
    q_vec = __ldg(reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]));
  }

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float4 out_acc = make_float4(0.f, 0.f, 0.f, 0.f);

  const int num_keys = q_idx + 1;

  // Prefetch first K
  float4 k_prefetch = make_float4(0.f, 0.f, 0.f, 0.f);
  if (num_keys > 0 && lane_active) {
    k_prefetch = __ldg(reinterpret_cast<const float4 *>(
        &K[bh_offset + d_base]));
  }

  // Main loop with prefetching
  for (int k = 0; k < num_keys; ++k) {
    // Use prefetched K
    float4 k_vec = k_prefetch;

    // Prefetch next K (if exists)
    if (k + 1 < num_keys && lane_active) {
      k_prefetch = __ldg(reinterpret_cast<const float4 *>(
          &K[bh_offset + (k + 1) * head_dim_pad + d_base]));
    }

    // Load V for current key
    float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
    if (lane_active) {
      v_vec = __ldg(reinterpret_cast<const float4 *>(
          &V[bh_offset + k * head_dim_pad + d_base]));
    }

    // Dot product Q · K
    float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                q_vec.z * k_vec.z + q_vec.w * k_vec.w;
    float score = warp_reduce_sum_xor(dot) * scale;

    // Online softmax update
    float new_max = fmaxf(running_max, score);
    float alpha = expf(running_max - new_max);
    float weight = expf(score - new_max);

    running_sum = running_sum * alpha + weight;
    out_acc.x = out_acc.x * alpha + weight * v_vec.x;
    out_acc.y = out_acc.y * alpha + weight * v_vec.y;
    out_acc.z = out_acc.z * alpha + weight * v_vec.z;
    out_acc.w = out_acc.w * alpha + weight * v_vec.w;
    running_max = new_max;
  }

  // Normalize and write output
  if (lane_active) {
    float inv_sum = 1.0f / running_sum;
    float4 result = make_float4(out_acc.x * inv_sum, out_acc.y * inv_sum,
                                out_acc.z * inv_sum, out_acc.w * inv_sum);
    *reinterpret_cast<float4 *>(
        &out[bh_offset + q_idx * head_dim_pad + d_base]) = result;
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
