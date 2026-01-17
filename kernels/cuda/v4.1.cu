#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4.1: Multi-Key Processing (4 keys per iteration)
// ============================================================================
// Building on v4's register-based output accumulator and Q in registers, this
// version processes 4 keys per iteration to improve instruction-level
// parallelism (ILP).
//
// Key insight:
// - MUFU.EX2 (expf) has ~16-20 cycle latency
// - By computing 4 dot products before softmax updates, we overlap computation
// - While exp results compute, we're loading data for next operations
//
// Changes from v4:
// - Process 4 keys per loop iteration
// - Compute all 4 dot products before any softmax update
// - Handle remainder keys (0-3) after main loop
//
// Supported head_dim: up to 128
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // Support up to head_dim=128
#define KEYS_PER_ITER 4

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

  const size_t q_offset = bh_offset + q * head_dim_pad;
  const size_t out_offset = q_offset;

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Register-based output accumulator and Q in registers (inherited from v4)
  float out_accum[MAX_D_PER_LANE];
  float q_r[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out_accum[i] = 0.0f;
    q_r[i] = Q[q_offset + d];
  }

  // Main loop: process 4 keys per iteration
  int k = 0;
  for (; k + 3 <= q; k += KEYS_PER_ITER) {
    const size_t k_offset0 = bh_offset + k * head_dim_pad;
    const size_t k_offset1 = bh_offset + (k + 1) * head_dim_pad;
    const size_t k_offset2 = bh_offset + (k + 2) * head_dim_pad;
    const size_t k_offset3 = bh_offset + (k + 3) * head_dim_pad;

    // Compute all 4 dot products first (better ILP)
    float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;

    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      float q_val = q_r[i];
      dot0 += q_val * K[k_offset0 + d];
      dot1 += q_val * K[k_offset1 + d];
      dot2 += q_val * K[k_offset2 + d];
      dot3 += q_val * K[k_offset3 + d];
    }

    float score0 = warp_reduce_sum_xor(dot0) * scale;
    float score1 = warp_reduce_sum_xor(dot1) * scale;
    float score2 = warp_reduce_sum_xor(dot2) * scale;
    float score3 = warp_reduce_sum_xor(dot3) * scale;

    // Online softmax updates (sequential - order matters)
#pragma unroll
    for (int ki = 0; ki < KEYS_PER_ITER; ki++) {
      float score;
      switch (ki) {
      case 0:
        score = score0;
        break;
      case 1:
        score = score1;
        break;
      case 2:
        score = score2;
        break;
      default:
        score = score3;
        break;
      }

      size_t kv_offset = bh_offset + (k + ki) * head_dim_pad;

      float new_max = fmaxf(softmax_max, score);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score - new_max);

      softmax_sum = softmax_sum * alpha + weight;

#pragma unroll
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        const int d = lane_id + i * WARP_SIZE;
        out_accum[i] = out_accum[i] * alpha + weight * V[kv_offset + d];
      }
      softmax_max = new_max;
    }
  }

  // Handle remaining keys (0-3)
  for (; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    float dot_partial = 0.0f;
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      dot_partial += q_r[i] * K[k_offset + d];
    }

    float score = warp_reduce_sum_xor(dot_partial) * scale;

    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      out_accum[i] = out_accum[i] * alpha + weight * V[k_offset + d];
    }
    softmax_max = new_max;
  }

  // Normalize and write to global memory
  float inv_sum = 1.0f / softmax_sum;
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out[out_offset + d] = out_accum[i] * inv_sum;
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
