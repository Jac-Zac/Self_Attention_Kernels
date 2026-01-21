#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// Adding direct float4 operation decrease memory pressure

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

  // Precompute base float4 pointers
  const float4 *Q_ptr =
      reinterpret_cast<const float4 *>(Q + bh_offset + q * head_dim_pad);
  const float4 *K_base = reinterpret_cast<const float4 *>(K + bh_offset);
  const float4 *V_base = reinterpret_cast<const float4 *>(V + bh_offset);
  float4 *out_ptr =
      reinterpret_cast<float4 *>(out + bh_offset + q * head_dim_pad);

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Vectorized Load Q into registers (assuming head_dim <= 128)
  // Each thread in the warp loads one float4 (4 floats * 32 threads = 128 dims)
  float4 vec_q = {0.0f, 0.0f, 0.0f, 0.0f};
  if (lane_id * 4 < head_dim) {
    vec_q = Q_ptr[lane_id];
  }

  // Output Accumulator as float4
  float4 out_accum = {0.0f, 0.0f, 0.0f, 0.0f};

  // Loop over keys (causal: k <= q)
  for (int k = 0; k <= q; ++k) {
    const float4 *K_ptr = K_base + k * (head_dim_pad / 4);
    const float4 *V_ptr = V_base + k * (head_dim_pad / 4);

    // 1. Vectorized Load K
    float4 k_vec = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane_id * 4 < head_dim) {
      k_vec = K_ptr[lane_id];
    }

    // Q·K dot product (Q from registers) - single float4 per lane
    // 2. Dot Product (Local partial -> Warp reduce)
    float dot = (vec_q.x * k_vec.x) + (vec_q.y * k_vec.y) +
                (vec_q.z * k_vec.z) + (vec_q.w * k_vec.w);

    float score = warp_reduce_sum_xor(dot) * scale;

    // Online softmax update
    float new_max = fmaxf(running_max, score);
    float alpha = expf(running_max - new_max);
    float weight = expf(score - new_max);

    running_sum = running_sum * alpha + weight;

    // Load V (Vectorized) and Update Accumulator
    if (lane_id * 4 < head_dim) {
      float4 vec_v = V_ptr[lane_id];
      out_accum.x = out_accum.x * alpha + weight * vec_v.x;
      out_accum.y = out_accum.y * alpha + weight * vec_v.y;
      out_accum.z = out_accum.z * alpha + weight * vec_v.z;
      out_accum.w = out_accum.w * alpha + weight * vec_v.w;
    }

    running_max = new_max;
  }

  // Normalize and write to global memory
  // Final Normalization and Vectorized Store
  float inv_sum = 1.0f / running_sum;
  if (lane_id * 4 < head_dim) {
    out_accum.x *= inv_sum;
    out_accum.y *= inv_sum;
    out_accum.z *= inv_sum;
    out_accum.w *= inv_sum;

    out_ptr[lane_id] = out_accum;
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
