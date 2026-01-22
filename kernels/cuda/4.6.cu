#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5: Dual-Key Per Iteration (Reduced Warp Reductions, Lower Reg Pressure)
//
// Key changes:
// - Process 2 keys per loop iteration (k, k+1)
//   -> halves warp reductions and loop overhead
// - Scalar output accumulators instead of float4
//   -> shorter live ranges, lower register pressure
// - Explicit scoping of temporaries to help NVCC register reuse
// - No shared memory (kernel is register / reduction bound)
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

  const float4 *Q_ptr =
      reinterpret_cast<const float4 *>(Q + bh_offset + q * head_dim_pad);
  const float4 *K_base = reinterpret_cast<const float4 *>(K + bh_offset);
  const float4 *V_base = reinterpret_cast<const float4 *>(V + bh_offset);
  float4 *out_ptr =
      reinterpret_cast<float4 *>(out + bh_offset + q * head_dim_pad);

  // Load Q once (vectorized)
  float4 vec_q = {0.f, 0.f, 0.f, 0.f};
  if (lane_id * 4 < head_dim) {
    vec_q = Q_ptr[lane_id];
  }

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.f;

  // Scalar output accumulators (lower reg pressure than float4)
  float out_x = 0.f, out_y = 0.f, out_z = 0.f, out_w = 0.f;

  const int stride = head_dim_pad / 4;

  // Loop over keys (causal: k <= q), 2 keys per iteration
  int k = 0;
  for (; k + 1 <= q; k += 2) {

    const float4 *K0 = K_base + (k + 0) * stride;
    const float4 *K1 = K_base + (k + 1) * stride;
    const float4 *V0 = V_base + (k + 0) * stride;
    const float4 *V1 = V_base + (k + 1) * stride;

    // Dot(Q, K0), Dot(Q, K1)
    // -----------------------------
    float dot0 = 0.f;
    float dot1 = 0.f;

    if (lane_id * 4 < head_dim) {
      float4 k0 = K0[lane_id];
      float4 k1 = K1[lane_id];

      dot0 = vec_q.x * k0.x + vec_q.y * k0.y + vec_q.z * k0.z + vec_q.w * k0.w;

      dot1 = vec_q.x * k1.x + vec_q.y * k1.y + vec_q.z * k1.z + vec_q.w * k1.w;
    }

    dot0 = warp_reduce_sum_xor(dot0);
    dot1 = warp_reduce_sum_xor(dot1);

    // Score 0
    {
      float score = dot0 * scale;
      float new_max = fmaxf(running_max, score);
      float alpha = expf(running_max - new_max);
      float weight = expf(score - new_max);

      running_sum = running_sum * alpha + weight;

      if (lane_id * 4 < head_dim) {
        float4 v = V0[lane_id];
        out_x = out_x * alpha + weight * v.x;
        out_y = out_y * alpha + weight * v.y;
        out_z = out_z * alpha + weight * v.z;
        out_w = out_w * alpha + weight * v.w;
      }

      running_max = new_max;
    }

    // Score 1
    {
      float score = dot1 * scale;
      float new_max = fmaxf(running_max, score);
      float alpha = expf(running_max - new_max);
      float weight = expf(score - new_max);

      running_sum = running_sum * alpha + weight;

      if (lane_id * 4 < head_dim) {
        float4 v = V1[lane_id];
        out_x = out_x * alpha + weight * v.x;
        out_y = out_y * alpha + weight * v.y;
        out_z = out_z * alpha + weight * v.z;
        out_w = out_w * alpha + weight * v.w;
      }

      running_max = new_max;
    }
  }

  // Handle last key if q is odd
  if (k <= q) {
    const float4 *Kp = K_base + k * stride;
    const float4 *Vp = V_base + k * stride;

    float dot = 0.f;
    if (lane_id * 4 < head_dim) {
      float4 kv = Kp[lane_id];
      dot = vec_q.x * kv.x + vec_q.y * kv.y + vec_q.z * kv.z + vec_q.w * kv.w;
    }

    dot = warp_reduce_sum_xor(dot);
    float score = dot * scale;

    float new_max = fmaxf(running_max, score);
    float alpha = expf(running_max - new_max);
    float weight = expf(score - new_max);

    running_sum = running_sum * alpha + weight;

    if (lane_id * 4 < head_dim) {
      float4 v = Vp[lane_id];
      out_x = out_x * alpha + weight * v.x;
      out_y = out_y * alpha + weight * v.y;
      out_z = out_z * alpha + weight * v.z;
      out_w = out_w * alpha + weight * v.w;
    }

    running_max = new_max;
  }

  // Final normalization and store
  float inv_sum = 1.0f / running_sum;

  if (lane_id * 4 < head_dim) {
    float4 out_v;
    out_v.x = out_x * inv_sum;
    out_v.y = out_y * inv_sum;
    out_v.z = out_z * inv_sum;
    out_v.w = out_w * inv_sum;
    out_ptr[lane_id] = out_v;
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
