#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// NOTE: Similarly to what is done for efficient kernel we will have the
// following kernels worok from a certain head dim downward

// ============================================================================
// v4.1: Register-based Output Accumulator
// ============================================================================
// Building on v3's online softmax, this version keeps output in registers
// instead of repeatedly reading/writing global memory.
//
// Key changes from v3:
// 1. Output accumulator kept in registers (up to 4 floats per lane for
// head_dim=128)
// 2. Only write to global memory once at the end (after normalization)
// 3. Eliminates load-modify-store pattern in inner loop
//
// Supported head_dim: up to 128 (4 elements per lane * 32 lanes)
// For head_dim=64: 2 registers per lane
// For head_dim=128: 4 registers per lane
//
// Expected improvement: Eliminates ~90K stalls from V-load waiting and
// ~11K stalls from output stores in the inner loop.
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // Support up to head_dim=128 (4 * 32 = 128)

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

  // Number of elements this lane handles
  const int d_per_lane = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Register-based output accumulator
  // Each lane handles up to MAX_D_PER_LANE elements
  float out_accum[MAX_D_PER_LANE];
#pragma unroll
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    out_accum[i] = 0.0f;
  }

  // Loop over keys (causal: k <= q)
  for (int k = 0; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    // Q·K dot product (warp-parallel across head_dim)
    float dot_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        dot_partial += Q[q_offset + d] * K[k_offset + d];
      }
    }

    float score = warp_reduce_sum_xor(dot_partial) * scale;

    // Online softmax update
    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;

    // Update output accumulator in registers (no global memory access!)
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        out_accum[i] = out_accum[i] * alpha + weight * V[k_offset + d];
      }
    }

    softmax_max = new_max;
  }

  // Normalize and write to global memory (once!)
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

  // This assert to avoid erorrs
  assert(dims.head_dim < -= MAX_D_PER_LANE * WARP_SIZE);
  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
