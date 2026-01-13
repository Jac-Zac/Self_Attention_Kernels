#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5: Combined Optimizations (v4.1 + v4.2 + v4.4)
// ============================================================================
// This version combines the most promising optimizations from the v4.x series:
//
// From v4.1: Register-based output accumulator
//   - Keeps output in registers instead of read-modify-write to global memory
//   - Eliminates ~90K V-load stalls + ~11K store stalls per iteration
//
// From v4.4: Q in registers
//   - Loads Q once at kernel start, reuses for all K iterations
//   - For seq_len=1024, eliminates up to 1024 redundant Q reads
//
// From v4.2: Multi-key processing (2 keys per iteration)
//   - Computes 2 dot products before softmax updates
//   - Better ILP to hide MUFU.EX2 latency (~16-20 cycles)
//   - While first exp computes, second dot product is in flight
//
// Inner loop now only reads K and V from global memory.
// Q, output accumulator, and softmax state all in registers.
//
// Supported head_dim: up to 128 (4 elements per lane * 32 lanes)
// Register usage per thread: ~20 floats (q_reg[4] + out_accum[4] + temps)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4  // Support up to head_dim=128
#define KEYS_PER_ITER 2   // Process 2 keys per iteration

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

  // ========================================
  // Load Q into registers ONCE (from v4.4)
  // ========================================
  float q_reg[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    q_reg[i] = (d < head_dim) ? Q[q_offset + d] : 0.0f;
  }

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // ========================================
  // Register-based output accumulator (from v4.1)
  // ========================================
  float out_accum[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    out_accum[i] = 0.0f;
  }

  // ========================================
  // Process 2 keys per iteration (from v4.2)
  // ========================================
  int k = 0;
  for (; k + 1 <= q; k += KEYS_PER_ITER) {
    const size_t k_offset0 = bh_offset + k * head_dim_pad;
    const size_t k_offset1 = bh_offset + (k + 1) * head_dim_pad;

    // Compute BOTH dot products before softmax (better ILP)
    // Uses Q from registers - no global memory read for Q
    float dot_partial0 = 0.0f;
    float dot_partial1 = 0.0f;

#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        float k0_val = K[k_offset0 + d];
        float k1_val = K[k_offset1 + d];
        dot_partial0 += q_reg[i] * k0_val;
        dot_partial1 += q_reg[i] * k1_val;
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
  }

  // ========================================
  // Normalize and write to global memory (once!)
  // ========================================
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
