#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff

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

  const size_t head_dim = dims.head_dim;
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

  // Initialize output accumulator
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE)
    out[out_offset + d] = 0.0f;

  // Loop over keys
  for (int k = 0; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    // Q·K
    float dot_partial = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE)
      dot_partial += Q[q_offset + d] * K[k_offset + d];

    float score = warp_reduce_sum_xor(dot_partial) * scale;

    // Online softmax update
    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;

    // Update output
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      out[out_offset + d] =
          out[out_offset + d] * alpha + weight * V[k_offset + d];
    }

    softmax_max = new_max;
  }

  // Normalize
  float inv_sum = 1.0f / softmax_sum;
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE)
    out[out_offset + d] *= inv_sum;
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
