#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

#define MAX_HEAD_DIM 128

// Reduction helpers for warp-level communication
__inline__ __device__ float warp_reduce_max(float val, unsigned mask) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_down_sync(mask, val, offset));
  return __shfl_sync(mask, val, 0); // Broadcast result to all threads in warp
}

__inline__ __device__ float warp_reduce_sum(float val, unsigned mask) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(mask, val, offset);
  return __shfl_sync(mask, val, 0); // Broadcast result to all threads in warp
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  // 1 Warp = 1 Query
  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q_idx >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;
  const size_t bh_offset =
      (size_t)b * (dims.n_heads * dims.seq_len * head_dim_pad) +
      (size_t)h * (dims.seq_len * head_dim_pad);

  const float *Q_ptr = Q + bh_offset + q_idx * head_dim_pad;
  const float *K_base = K + bh_offset;
  const float *V_base = V + bh_offset;
  float *out_ptr = out + bh_offset + q_idx * head_dim_pad;

  __shared__ float s_Q[WARPS_PER_BLOCK][MAX_HEAD_DIM];
  __shared__ float s_out[WARPS_PER_BLOCK][MAX_HEAD_DIM];

  // Load Q and initialize output
  for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
    s_Q[warp_id][d] = Q_ptr[d] * scale;
    s_out[warp_id][d] = 0.0f;
  }
  __syncwarp();

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // LOOP OVER K IN TILES OF 32
  // Each thread in the warp handles a different key 'k' in parallel
  for (int k_tile = 0; k_tile <= q_idx; k_tile += WARP_SIZE) {
    int k = k_tile + lane_id;
    bool active = (k <= q_idx);
    unsigned mask = __ballot_sync(0xffffffff, active);

    float dot = -FLT_MAX;
    if (active) {
      dot = 0.0f;
      const float *K_ptr = K_base + (size_t)k * head_dim_pad;
      for (int d = 0; d < head_dim; d++) {
        dot += s_Q[warp_id][d] * K_ptr[d];
      }
    }

    // Parallel Max Reduction over the 32 keys in this tile
    float max_tile = warp_reduce_max(active ? dot : -FLT_MAX, mask);

    // Compute weights for the tile
    float weight = active ? expf(dot - max_tile) : 0.0f;
    float sum_tile = warp_reduce_sum(weight, mask);

    // Online softmax update logic
    float new_max = fmaxf(running_max, max_tile);
    float alpha = expf(running_max - new_max);
    float beta = expf(max_tile - new_max);

    // Update global running sum
    running_sum = running_sum * alpha + sum_tile * beta;

    // Update the output vector cooperatively
    // Instead of each thread updating s_out, we loop through dimensions
    // and sum the contributions from all 32 keys in the warp
    for (int d = 0; d < head_dim; d++) {
      float v_val = active ? (weight * V_base[k * head_dim_pad + d]) : 0.0f;
      float v_sum = warp_reduce_sum(v_val, mask);

      // Only lane 0 needs to update shared memory, or we can distribute
      if (lane_id == (d % WARP_SIZE)) {
        s_out[warp_id][d] = s_out[warp_id][d] * alpha + v_sum * beta;
      }
    }

    running_max = new_max;
    __syncwarp();
  }

  // Final write-back
  float inv_sum = 1.0f / running_sum;
  for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
    out_ptr[d] = s_out[warp_id][d] * inv_sum;
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
  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid((dims.seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
            dims.batch * dims.n_heads);
  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
