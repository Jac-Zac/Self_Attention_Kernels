#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4.3: Shuffle Down Reduction Test
// ============================================================================
// Same as v4.1, but using __shfl_down_sync instead of __shfl_xor_sync
// for the warp reduction.
//
// Rationale:
// - __shfl_xor_sync (butterfly pattern): all lanes participate equally
// - __shfl_down_sync: result accumulates in lane 0, then broadcast
//
// The profiler showed 21K stalls on the last shuffle (mask=1). This might
// be due to XOR pattern behavior. Down-shuffle might be better on sm_86.
//
// Note: down-shuffle naturally puts result in lane 0, so we need an extra
// broadcast. But modern GPUs can often combine these efficiently.
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4

// Down-shuffle reduction: result ends up in lane 0, then broadcast
__inline__ __device__ float warp_reduce_sum_down(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(WARP_MASK, val, offset);
  // Broadcast result from lane 0 to all lanes
  return __shfl_sync(WARP_MASK, val, 0);
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

  // Loop over keys
  for (int k = 0; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    // Q·K dot product
    float dot_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        dot_partial += Q[q_offset + d] * K[k_offset + d];
      }
    }

    // Use down-shuffle reduction
    float score = warp_reduce_sum_down(dot_partial) * scale;

    // Online softmax update
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
