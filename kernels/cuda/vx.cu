#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// vx_opt: Optimized vx with Better Block Configuration
// ============================================================================
// Changes from vx:
//   - Reduced WARPS_PER_BLOCK from 8 to 4
//   - Better occupancy: More blocks can run concurrently
//   - Less contention in shared memory (if any)
//   - Same register-blocking and vectorization as vx
//
// Why WARPS_PER_BLOCK=4?
//   - Higher block count → better scheduling
//   - Reduces pressure on L1/shared memory
//   - Better hides memory latency
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4 // Reduced from 8 for better occupancy
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
  const int q_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh_idx = blockIdx.y;

  if (q_idx >= (int)dims.seq_len || bh_idx >= (int)(dims.batch * dims.n_heads))
    return;

  const size_t head_dim = dims.head_dim;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh_idx / dims.n_heads;
  const int h = bh_idx % dims.n_heads;
  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const int d_base = lane_id * 4;
  const bool active = (d_base < (int)head_dim);

  float r_Q[4] = {0}, r_K[4], r_V[4], r_out[4] = {0};

  if (active) {
    float4 q_vec = *reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]);
    r_Q[0] = q_vec.x;
    r_Q[1] = q_vec.y;
    r_Q[2] = q_vec.z;
    r_Q[3] = q_vec.w;
  }

  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  for (int k_idx = 0; k_idx <= q_idx; k_idx++) {
    const size_t kv_base = bh_offset + k_idx * head_dim_pad + d_base;

    if (active) {
      float4 k_vec = __ldg(reinterpret_cast<const float4 *>(&K[kv_base]));
      r_K[0] = k_vec.x;
      r_K[1] = k_vec.y;
      r_K[2] = k_vec.z;
      r_K[3] = k_vec.w;
    } else {
      r_K[0] = r_K[1] = r_K[2] = r_K[3] = 0.0f;
    }

    float dot =
        r_Q[0] * r_K[0] + r_Q[1] * r_K[1] + r_Q[2] * r_K[2] + r_Q[3] * r_K[3];
    float score = warp_reduce_sum_xor(dot) * scale;

    float new_max = fmaxf(softmax_max, score);
    float rescale = expf(softmax_max - new_max);
    float weight = expf(score - new_max);
    softmax_sum = softmax_sum * rescale + weight;
    softmax_max = new_max;

    if (active) {
      float4 v_vec = __ldg(reinterpret_cast<const float4 *>(&V[kv_base]));
      r_V[0] = v_vec.x;
      r_V[1] = v_vec.y;
      r_V[2] = v_vec.z;
      r_V[3] = v_vec.w;
    } else {
      r_V[0] = r_V[1] = r_V[2] = r_V[3] = 0.0f;
    }

    r_out[0] = r_out[0] * rescale + weight * r_V[0];
    r_out[1] = r_out[1] * rescale + weight * r_V[1];
    r_out[2] = r_out[2] * rescale + weight * r_V[2];
    r_out[3] = r_out[3] * rescale + weight * r_V[3];
  }

  if (active) {
    const float inv_sum = 1.0f / softmax_sum;
    float4 out_vec = {r_out[0] * inv_sum, r_out[1] * inv_sum,
                      r_out[2] * inv_sum, r_out[3] * inv_sum};
    *reinterpret_cast<float4 *>(
        &out[bh_offset + q_idx * head_dim_pad + d_base]) = out_vec;
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
  const int query_groups = CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK);
  dim3 grid(query_groups, dims.batch * dims.n_heads);
  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
