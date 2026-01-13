#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4.2: Multi-Key Processing (2 keys per iteration)
// ============================================================================
// Building on v4.1's register-based output, this version processes 2 keys
// per iteration to improve instruction-level parallelism (ILP).
//
// Key insight from profiling:
// - MUFU.EX2 has ~16-20 cycle latency
// - After computing score, we immediately need exp results
// - By computing 2 dot products before the softmax update, we can overlap:
//   - While first exp is computing, we're loading data for second dot product
//   - Better pipeline utilization
//
// Changes from v4.1:
// 1. Process 2 keys per loop iteration
// 2. Compute both dot products before softmax updates
// 3. Still uses register-based output accumulator
//
// Supported head_dim: up to 128
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // Support up to head_dim=128
#define KEYS_PER_ITER 2  // Process 2 keys per iteration

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

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Register-based output accumulator
  float out_accum[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    out_accum[i] = 0.0f;
  }

  // Process keys 2 at a time for better ILP
  int k = 0;
  for (; k + 1 <= q; k += KEYS_PER_ITER) {
    const size_t k_offset0 = bh_offset + k * head_dim_pad;
    const size_t k_offset1 = bh_offset + (k + 1) * head_dim_pad;

    // Compute BOTH dot products before softmax (better ILP)
    float dot_partial0 = 0.0f;
    float dot_partial1 = 0.0f;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        float q_val = Q[q_offset + d];
        dot_partial0 += q_val * K[k_offset0 + d];
        dot_partial1 += q_val * K[k_offset1 + d];
      }
    }

    float score0 = warp_reduce_sum_xor(dot_partial0) * scale;
    float score1 = warp_reduce_sum_xor(dot_partial1) * scale;

    // Online softmax update for first key
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

    // Online softmax update for second key
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
  }

  // Handle remaining key (if q is even, we have one more key at k=q)
  if (k <= q) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    float dot_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        dot_partial += Q[q_offset + d] * K[k_offset + d];
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
