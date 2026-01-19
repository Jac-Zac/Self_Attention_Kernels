#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v7: Minimal Register Pressure + Fast Math
// ============================================================================
// This version focuses on reducing register pressure and maximizing occupancy.
//
// Key changes:
// 1. Use __fmaf_rn for fused multiply-add (faster, more accurate)
// 2. Use __expf for fast exponential (less accurate but faster)
// 3. Minimize live register count by reusing variables
// 4. Process K and V in same loop iteration to reduce register pressure
// 5. Use __fmul_rn and __fadd_rn for explicit fast rounding
//
// The idea is that higher occupancy (more warps per SM) can hide memory
// latency better than complex prefetching with lower occupancy.
//
// Register budget per thread (target: <64 for good occupancy):
// - q_vec: 4 floats (16 bytes)
// - out_acc: 4 floats (16 bytes)  
// - running_max, running_sum: 2 floats (8 bytes)
// - Loop variables: ~4-6 floats
// - Total: ~14-16 floats = 56-64 bytes
//
// Supported head_dim: up to 128
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

__global__ __launch_bounds__(256, 4) // 256 threads, aim for 4 blocks/SM
void cmhsa_forward_kernel(const float *RESTRICT Q,
                          const float *RESTRICT K,
                          const float *RESTRICT V,
                          float *RESTRICT out,
                          const AttentionDims dims) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  // Early exit for out-of-bounds
  if (q_idx >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);
  const size_t q_base = bh_offset + q_idx * head_dim_pad;

  const int d_base = lane_id * 4;
  const bool active = (d_base < head_dim);

  // Load Q - use float4 for coalesced access
  float qx = 0.f, qy = 0.f, qz = 0.f, qw = 0.f;
  if (active) {
    const float4 q_tmp = __ldg(reinterpret_cast<const float4 *>(&Q[q_base + d_base]));
    qx = q_tmp.x; qy = q_tmp.y; qz = q_tmp.z; qw = q_tmp.w;
  }

  // Accumulators
  float acc_x = 0.f, acc_y = 0.f, acc_z = 0.f, acc_w = 0.f;
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Main loop - process one key at a time, minimal register usage
  const int num_keys = q_idx + 1;
  
  for (int k = 0; k < num_keys; ++k) {
    const size_t kv_base = bh_offset + k * head_dim_pad + d_base;

    // Load K and V together
    float kx, ky, kz, kw, vx, vy, vz, vw;
    if (active) {
      const float4 k_tmp = __ldg(reinterpret_cast<const float4 *>(&K[kv_base]));
      const float4 v_tmp = __ldg(reinterpret_cast<const float4 *>(&V[kv_base]));
      kx = k_tmp.x; ky = k_tmp.y; kz = k_tmp.z; kw = k_tmp.w;
      vx = v_tmp.x; vy = v_tmp.y; vz = v_tmp.z; vw = v_tmp.w;
    } else {
      kx = ky = kz = kw = vx = vy = vz = vw = 0.f;
    }

    // Dot product using FMA
    float dot = __fmaf_rn(qx, kx, __fmaf_rn(qy, ky, __fmaf_rn(qz, kz, qw * kw)));
    float score = warp_reduce_sum_xor(dot) * scale;

    // Online softmax with fast exp
    float new_max = fmaxf(running_max, score);
    float alpha = __expf(running_max - new_max);
    float weight = __expf(score - new_max);

    running_sum = __fmaf_rn(running_sum, alpha, weight);
    
    // Update accumulators with FMA
    acc_x = __fmaf_rn(acc_x, alpha, weight * vx);
    acc_y = __fmaf_rn(acc_y, alpha, weight * vy);
    acc_z = __fmaf_rn(acc_z, alpha, weight * vz);
    acc_w = __fmaf_rn(acc_w, alpha, weight * vw);
    
    running_max = new_max;
  }

  // Normalize and write
  if (active) {
    float inv_sum = __frcp_rn(running_sum); // Fast reciprocal
    float4 result;
    result.x = acc_x * inv_sum;
    result.y = acc_y * inv_sum;
    result.z = acc_z * inv_sum;
    result.w = acc_w * inv_sum;
    *reinterpret_cast<float4 *>(&out[q_base + d_base]) = result;
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
