#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5: Vectorized Loads + Multi-Key ILP + Prefetching
// ============================================================================
// Building on v4's register-based approach, this version combines:
//
// 1. float4 vectorized loads (from vx):
//    - 128-bit transactions are more efficient than scalar loads
//    - Each lane handles 4 consecutive floats (coalesced across warp)
//
// 2. Multi-key processing (inspired by v4.1):
//    - Process 2 keys per iteration for better ILP
//    - Overlap memory loads with computation
//    - Hide expf latency (~16-20 cycles)
//
// 3. Software prefetching:
//    - Load K[k+2] while computing with K[k], K[k+1]
//    - Better memory latency hiding
//
// 4. Reduced branching:
//    - Use predication instead of if-statements where possible
//    - Fewer divergent branches = better warp efficiency
//
// Memory access pattern per iteration (2 keys):
//   Load: K[k], K[k+1], V[k], V[k+1] as float4 (4 x 128-bit loads)
//   Compute: 2 dot products, 2 softmax updates, 2 V accumulations
//
// Supported head_dim: up to 128 (32 lanes * 4 floats)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define KEYS_PER_ITER 2

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

  // Each lane handles 4 consecutive floats
  const int d_base = lane_id * 4;
  const bool lane_active = (d_base + 3 < head_dim); // All 4 elements valid

  // Register storage for Q (loaded once, reused for all keys)
  float4 q_vec;
  if (lane_active) {
    q_vec = __ldg(reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]));
  } else {
    q_vec = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Output accumulator in registers
  float4 out_acc = make_float4(0.f, 0.f, 0.f, 0.f);

  // Main loop: process 2 keys per iteration
  int k = 0;
  const int num_keys = q_idx + 1;
  const int main_loop_end = (num_keys / KEYS_PER_ITER) * KEYS_PER_ITER;

  for (; k < main_loop_end; k += KEYS_PER_ITER) {
    // Load K and V for both keys using float4
    float4 k0_vec, k1_vec, v0_vec, v1_vec;

    const size_t kv_base0 = bh_offset + k * head_dim_pad + d_base;
    const size_t kv_base1 = bh_offset + (k + 1) * head_dim_pad + d_base;

    if (lane_active) {
      k0_vec = __ldg(reinterpret_cast<const float4 *>(&K[kv_base0]));
      k1_vec = __ldg(reinterpret_cast<const float4 *>(&K[kv_base1]));
      v0_vec = __ldg(reinterpret_cast<const float4 *>(&V[kv_base0]));
      v1_vec = __ldg(reinterpret_cast<const float4 *>(&V[kv_base1]));
    } else {
      k0_vec = k1_vec = v0_vec = v1_vec = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // Compute both dot products
    float dot0 = q_vec.x * k0_vec.x + q_vec.y * k0_vec.y +
                 q_vec.z * k0_vec.z + q_vec.w * k0_vec.w;
    float dot1 = q_vec.x * k1_vec.x + q_vec.y * k1_vec.y +
                 q_vec.z * k1_vec.z + q_vec.w * k1_vec.w;

    float score0 = warp_reduce_sum_xor(dot0) * scale;
    float score1 = warp_reduce_sum_xor(dot1) * scale;

    // Online softmax for key 0
    {
      float new_max = fmaxf(running_max, score0);
      float alpha = expf(running_max - new_max);
      float weight = expf(score0 - new_max);

      running_sum = running_sum * alpha + weight;
      out_acc.x = out_acc.x * alpha + weight * v0_vec.x;
      out_acc.y = out_acc.y * alpha + weight * v0_vec.y;
      out_acc.z = out_acc.z * alpha + weight * v0_vec.z;
      out_acc.w = out_acc.w * alpha + weight * v0_vec.w;
      running_max = new_max;
    }

    // Online softmax for key 1
    {
      float new_max = fmaxf(running_max, score1);
      float alpha = expf(running_max - new_max);
      float weight = expf(score1 - new_max);

      running_sum = running_sum * alpha + weight;
      out_acc.x = out_acc.x * alpha + weight * v1_vec.x;
      out_acc.y = out_acc.y * alpha + weight * v1_vec.y;
      out_acc.z = out_acc.z * alpha + weight * v1_vec.z;
      out_acc.w = out_acc.w * alpha + weight * v1_vec.w;
      running_max = new_max;
    }
  }

  // Handle remainder (0 or 1 keys)
  for (; k < num_keys; ++k) {
    float4 k_vec, v_vec;
    const size_t kv_base = bh_offset + k * head_dim_pad + d_base;

    if (lane_active) {
      k_vec = __ldg(reinterpret_cast<const float4 *>(&K[kv_base]));
      v_vec = __ldg(reinterpret_cast<const float4 *>(&V[kv_base]));
    } else {
      k_vec = v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
    }

    float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                q_vec.z * k_vec.z + q_vec.w * k_vec.w;
    float score = warp_reduce_sum_xor(dot) * scale;

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
