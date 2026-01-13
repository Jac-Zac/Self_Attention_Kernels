#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5.1: Combined Optimizations with 4 Keys per Iteration
// ============================================================================
// Same as v5 but processing 4 keys per iteration instead of 2.
// This increases ILP further at the cost of more register pressure.
//
// Combines:
// - Register-based output accumulator (v4.1)
// - Q in registers (v4.4)
// - 4-key multi-processing (extended from v4.2)
//
// The 4-key approach:
// 1. Load K[k+0], K[k+1], K[k+2], K[k+3] and compute 4 dot products
// 2. Reduce all 4 scores
// 3. Apply online softmax updates sequentially (order matters!)
//
// Trade-off: More register pressure, but better memory access patterns
// and more opportunities to hide MUFU.EX2 latency.
//
// Supported head_dim: up to 128
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // Support up to head_dim=128
#define KEYS_PER_ITER 4  // Process 4 keys per iteration

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
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

  // Load Q into registers ONCE
  float q_reg[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    q_reg[i] = (d < head_dim) ? Q[q_offset + d] : 0.0f;
  }

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Register-based output accumulator
  float out_accum[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    out_accum[i] = 0.0f;
  }

  // Process 4 keys per iteration
  int k = 0;
  for (; k + 3 <= q; k += KEYS_PER_ITER) {
    const size_t k_offset0 = bh_offset + k * head_dim_pad;
    const size_t k_offset1 = bh_offset + (k + 1) * head_dim_pad;
    const size_t k_offset2 = bh_offset + (k + 2) * head_dim_pad;
    const size_t k_offset3 = bh_offset + (k + 3) * head_dim_pad;

    // Compute ALL 4 dot products before softmax (maximum ILP)
    float dot_partial0 = 0.0f;
    float dot_partial1 = 0.0f;
    float dot_partial2 = 0.0f;
    float dot_partial3 = 0.0f;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        float q_val = q_reg[i];
        dot_partial0 += q_val * K[k_offset0 + d];
        dot_partial1 += q_val * K[k_offset1 + d];
        dot_partial2 += q_val * K[k_offset2 + d];
        dot_partial3 += q_val * K[k_offset3 + d];
      }
    }

    float score0 = warp_reduce_sum_xor(dot_partial0) * scale;
    float score1 = warp_reduce_sum_xor(dot_partial1) * scale;
    float score2 = warp_reduce_sum_xor(dot_partial2) * scale;
    float score3 = warp_reduce_sum_xor(dot_partial3) * scale;

    // Online softmax updates (must be sequential - order matters!)
    // Key 0
    {
      float new_max = fmaxf(softmax_max, score0);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score0 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] = out_accum[i] * alpha + weight * V[k_offset0 + d];
        }
      }
      softmax_max = new_max;
    }

    // Key 1
    {
      float new_max = fmaxf(softmax_max, score1);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score1 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] = out_accum[i] * alpha + weight * V[k_offset1 + d];
        }
      }
      softmax_max = new_max;
    }

    // Key 2
    {
      float new_max = fmaxf(softmax_max, score2);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score2 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] = out_accum[i] * alpha + weight * V[k_offset2 + d];
        }
      }
      softmax_max = new_max;
    }

    // Key 3
    {
      float new_max = fmaxf(softmax_max, score3);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score3 - new_max);
      softmax_sum = softmax_sum * alpha + weight;
#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] = out_accum[i] * alpha + weight * V[k_offset3 + d];
        }
      }
      softmax_max = new_max;
    }
  }

  // Handle remaining keys (0-3 remaining)
  for (; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    float dot_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        dot_partial += q_reg[i] * K[k_offset + d];
      }
    }

    float score = warp_reduce_sum_xor(dot_partial) * scale;

    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        out_accum[i] = out_accum[i] * alpha + weight * V[k_offset + d];
      }
    }
    softmax_max = new_max;
  }

  // Normalize and write to global memory
  float inv_sum = 1.0f / softmax_sum;
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      out[out_offset + d] = out_accum[i] * inv_sum;
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
