#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4: Online Softmax (Flash Attention Core Concept)
// ============================================================================
// Changes from v3:
// - Online softmax: single pass over keys instead of storing all scores
// - Zero workspace required (v3 needed O(seq_len²))
//
// Online softmax algorithm for each key k:
//   new_max = max(running_max, score)
//   rescale = exp(running_max - new_max)
//   weight = exp(score - new_max)
//   output = output * rescale + weight * V[k]
//   sum = sum * rescale + weight
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  }
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

  const size_t head_dim = dims.head_dim;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;
  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;
  const size_t output_offset = query_offset;

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Initialize output
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
    out[output_offset + d] = 0.0f;
  }

  // Single pass over keys with online softmax
  for (int key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;

    // Q·K dot product
    float dot = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      dot += Q[query_offset + d] * K[key_offset + d];
    }
    float score = warp_reduce_sum_xor(dot) * scale;

    // Online softmax update
    float new_max = fmaxf(softmax_max, score);
    float rescale = expf(softmax_max - new_max);
    float weight = expf(score - new_max);
    softmax_sum = softmax_sum * rescale + weight;
    softmax_max = new_max;

    // Rescale and accumulate
    const size_t value_offset = bh_offset + key_pos * head_dim_pad;
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      out[output_offset + d] =
          out[output_offset + d] * rescale + weight * V[value_offset + d];
    }
  }

  // Final normalization
  const float inv_sum = 1.0f / softmax_sum;
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
    out[output_offset + d] *= inv_sum;
  }
}

// ============================================================================
// Public API
// ============================================================================

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

  const int query_groups = CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK);
  dim3 grid(query_groups, dims.batch * dims.n_heads);
  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}

#else
#error "This file requires USE_CUDA to be defined"
#endif
