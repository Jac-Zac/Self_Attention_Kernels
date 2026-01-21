#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v3: Online Softmax, No Workspace
// ============================================================================
// 1. Online/Flash-style softmax:
//    - Softmax normalization and value accumulation (sum_i Softmax(q,k_i) *
//    v_i) are
//      fused into a single loop using numerically stable running max/sum.
//    - No separate storage of attention matrix or workspace required.
//    - Avoids full (query, key) score matrix materialization.
//
// 2. Lower memory footprint, higher arithmetic intensity:
//    - Only Q, K, V, and output tensors allocated per batch/head.
//    - Intermediate attn_weights/softmax workspace is NOT needed.
//    - Peak memory cost drops significantly—enables larger batches or sequence
//    lengths.
//
// 3. Precompute pointeres for the tensors
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum(float val) {
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
  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q_idx >= dims.seq_len || bh >= dims.batch * dims.n_heads)
    return;

  const int head_dim = dims.head_dim;
  const int head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Calculate base pointers for this batch-head
  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Initialize pointers with offsets precomputed
  const float *Q_ptr = Q + bh_offset + q_idx * head_dim_pad;
  const float *K_base = K + bh_offset;
  const float *V_base = V + bh_offset;
  float *out_ptr = out + bh_offset + q_idx * head_dim_pad;

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Zero-initialize output
  for (int d = lane_id; d < head_dim; d += WARP_SIZE)
    out_ptr[d] = 0.0f;

  // Causal attention loop
  for (int k = 0; k <= q_idx; ++k) {
    const float *K_ptr = K_base + k * head_dim_pad;
    const float *V_ptr = V_base + k * head_dim_pad;

    // Compute Q·K (dot product)
    float dot_partial = 0.0f;

    for (int d = lane_id; d < head_dim; d += WARP_SIZE)
      dot_partial += Q_ptr[d] * K_ptr[d];

    const float score = warp_reduce_sum(dot_partial) * scale;

    // Online softmax update
    const float new_max = fmaxf(running_max, score);
    const float alpha = expf(running_max - new_max);
    const float weight = expf(score - new_max);

    running_sum = running_sum * alpha + weight;

    // Update output accumulator
    for (int d = lane_id; d < head_dim; d += WARP_SIZE)
      out_ptr[d] = out_ptr[d] * alpha + weight * V_ptr[d];

    running_max = new_max;
  }

  // Normalize by sum
  const float inv_sum = 1.0f / running_sum;
  for (int d = lane_id; d < head_dim; d += WARP_SIZE)
    out_ptr[d] *= inv_sum;
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
