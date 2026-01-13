#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5.1: Vectorized float4 Loads + Multi-Key Processing
// ============================================================================
// Building on v4.2 (multi-key), this version combines float4 vectorized memory
// access with 4-keys-per-iteration ILP optimization.
//
// Key insight:
// - Each lane handles 4 consecutive floats (for head_dim up to 128)
// - Instead of 4 scalar loads, use 1 float4 load (128-bit transaction)
// - GPU memory controller handles 128-bit loads more efficiently
// - Processing 4 keys per iteration improves ILP
//
// Memory layout assumption:
// - head_dim_padded is aligned to 128 bytes (32 floats) for coalescing
// - Each lane loads float4 at offset: lane_id * 4
//
// Changes from v4.2:
// - Q, K, V loaded as float4 instead of scalar floats
// - Dot product computed from float4 components
// - Output written as float4
//
// Supported head_dim: up to 128 (32 lanes * 4 floats = 128)
// For smaller head_dim, unused components are masked out
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define KEYS_PER_ITER 4

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

// Compute dot product of two float4 vectors, respecting head_dim bounds
__inline__ __device__ float dot_float4(float4 a, float4 b, int lane_id,
                                       int head_dim) {
  float sum = 0.0f;
  int base_d = lane_id * 4;
  if (base_d + 0 < head_dim)
    sum += a.x * b.x;
  if (base_d + 1 < head_dim)
    sum += a.y * b.y;
  if (base_d + 2 < head_dim)
    sum += a.z * b.z;
  if (base_d + 3 < head_dim)
    sum += a.w * b.w;
  return sum;
}

// Fused multiply-add for float4: out = out * alpha + weight * v
__inline__ __device__ float4 fma_float4(float4 out, float alpha, float weight,
                                        float4 v) {
  return make_float4(out.x * alpha + weight * v.x, out.y * alpha + weight * v.y,
                     out.z * alpha + weight * v.z,
                     out.w * alpha + weight * v.w);
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const size_t head_dim_pad) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const size_t q_offset = bh_offset + q * head_dim_pad;
  const size_t out_offset = q_offset;

  // Load Q as float4 (once, reused for all keys)
  // Each lane loads 4 consecutive floats at lane_id * 4
  const float4 *Q4 = reinterpret_cast<const float4 *>(Q + q_offset);
  float4 q_vec =
      (lane_id * 4 < head_dim) ? Q4[lane_id] : make_float4(0, 0, 0, 0);

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Register-based output accumulator (as float4)
  float4 out_accum = make_float4(0, 0, 0, 0);

  // Main loop: process 4 keys per iteration
  int k = 0;
  for (; k + 3 <= q; k += KEYS_PER_ITER) {
    // Precompute offsets for 4 keys
    const float4 *K4_0 =
        reinterpret_cast<const float4 *>(K + bh_offset + k * head_dim_pad);
    const float4 *K4_1 = reinterpret_cast<const float4 *>(
        K + bh_offset + (k + 1) * head_dim_pad);
    const float4 *K4_2 = reinterpret_cast<const float4 *>(
        K + bh_offset + (k + 2) * head_dim_pad);
    const float4 *K4_3 = reinterpret_cast<const float4 *>(
        K + bh_offset + (k + 3) * head_dim_pad);

    // Load K vectors as float4
    float4 k_vec0 =
        (lane_id * 4 < head_dim) ? K4_0[lane_id] : make_float4(0, 0, 0, 0);
    float4 k_vec1 =
        (lane_id * 4 < head_dim) ? K4_1[lane_id] : make_float4(0, 0, 0, 0);
    float4 k_vec2 =
        (lane_id * 4 < head_dim) ? K4_2[lane_id] : make_float4(0, 0, 0, 0);
    float4 k_vec3 =
        (lane_id * 4 < head_dim) ? K4_3[lane_id] : make_float4(0, 0, 0, 0);

    // Compute all 4 dot products
    float dot0 = dot_float4(q_vec, k_vec0, lane_id, head_dim);
    float dot1 = dot_float4(q_vec, k_vec1, lane_id, head_dim);
    float dot2 = dot_float4(q_vec, k_vec2, lane_id, head_dim);
    float dot3 = dot_float4(q_vec, k_vec3, lane_id, head_dim);

    float score0 = warp_reduce_sum_xor(dot0) * scale;
    float score1 = warp_reduce_sum_xor(dot1) * scale;
    float score2 = warp_reduce_sum_xor(dot2) * scale;
    float score3 = warp_reduce_sum_xor(dot3) * scale;

    // V pointers for float4 loads
    const float4 *V4_0 =
        reinterpret_cast<const float4 *>(V + bh_offset + k * head_dim_pad);
    const float4 *V4_1 = reinterpret_cast<const float4 *>(
        V + bh_offset + (k + 1) * head_dim_pad);
    const float4 *V4_2 = reinterpret_cast<const float4 *>(
        V + bh_offset + (k + 2) * head_dim_pad);
    const float4 *V4_3 = reinterpret_cast<const float4 *>(
        V + bh_offset + (k + 3) * head_dim_pad);

    // Online softmax update for key 0
    {
      float new_max = fmaxf(softmax_max, score0);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score0 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
      float4 v_vec =
          (lane_id * 4 < head_dim) ? V4_0[lane_id] : make_float4(0, 0, 0, 0);
      out_accum = fma_float4(out_accum, alpha, weight, v_vec);
      softmax_max = new_max;
    }

    // Online softmax update for key 1
    {
      float new_max = fmaxf(softmax_max, score1);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score1 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
      float4 v_vec =
          (lane_id * 4 < head_dim) ? V4_1[lane_id] : make_float4(0, 0, 0, 0);
      out_accum = fma_float4(out_accum, alpha, weight, v_vec);
      softmax_max = new_max;
    }

    // Online softmax update for key 2
    {
      float new_max = fmaxf(softmax_max, score2);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score2 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
      float4 v_vec =
          (lane_id * 4 < head_dim) ? V4_2[lane_id] : make_float4(0, 0, 0, 0);
      out_accum = fma_float4(out_accum, alpha, weight, v_vec);
      softmax_max = new_max;
    }

    // Online softmax update for key 3
    {
      float new_max = fmaxf(softmax_max, score3);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score3 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
      float4 v_vec =
          (lane_id * 4 < head_dim) ? V4_3[lane_id] : make_float4(0, 0, 0, 0);
      out_accum = fma_float4(out_accum, alpha, weight, v_vec);
      softmax_max = new_max;
    }
  }

  // Handle remaining keys (0-3)
  for (; k <= q; ++k) {
    const float4 *K4 =
        reinterpret_cast<const float4 *>(K + bh_offset + k * head_dim_pad);
    const float4 *V4 =
        reinterpret_cast<const float4 *>(V + bh_offset + k * head_dim_pad);

    float4 k_vec =
        (lane_id * 4 < head_dim) ? K4[lane_id] : make_float4(0, 0, 0, 0);
    float dot = dot_float4(q_vec, k_vec, lane_id, head_dim);
    float score = warp_reduce_sum_xor(dot) * scale;

    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;
    float4 v_vec =
        (lane_id * 4 < head_dim) ? V4[lane_id] : make_float4(0, 0, 0, 0);
    out_accum = fma_float4(out_accum, alpha, weight, v_vec);
    softmax_max = new_max;
  }

  // Normalize and write to global memory as float4
  float inv_sum = 1.0f / softmax_sum;
  out_accum.x *= inv_sum;
  out_accum.y *= inv_sum;
  out_accum.z *= inv_sum;
  out_accum.w *= inv_sum;

  if (lane_id * 4 < head_dim) {
    float4 *out4 = reinterpret_cast<float4 *>(out + out_offset);
    out4[lane_id] = out_accum;
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
