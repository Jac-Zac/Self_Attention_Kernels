#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ==========================================================================
// v5.1: v5 + float4 vectorized loads from v4.1
// ==========================================================================
// - Keeps block-cooperative shared memory tiling of v5.
// - Uses float4 loads/stores for Q/K/V and shared memory to reduce
//   memory pressure and increase load throughput (borrowed from v4.1).
// - Since head_dim_padded is a multiple of 4 (project pads to VEC_PADDING).
// ==========================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff

#define TILE_K 8
#define MAX_HEAD_DIM 128
// Number of float4 elements in max head dim
#define MAX_HEAD_DIM_F4 (MAX_HEAD_DIM / 4)

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
  // Linear ID for cooperative loading (0 to 255)
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  // Constants
  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const int head_dim_f4 = (int)(head_dim_pad / 4);
  const float scale = rsqrtf((float)head_dim);

  // 2. Shared Memory Allocation (float4 tiles)
  // TILE_K timesteps, each has head_dim_padded/4 float4 elements.
  __shared__ float4 smem_K[TILE_K][MAX_HEAD_DIM_F4];
  __shared__ float4 smem_V[TILE_K][MAX_HEAD_DIM_F4];

  // 3. Early Exit (Block Level)
  if (bh >= (int)(dims.batch * dims.n_heads))
    return;

  // Base Pointers
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;
  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Float4 pointers for vectorized access
  const float4 *Q4_ptr =
      reinterpret_cast<const float4 *>(Q + bh_offset + q * head_dim_pad);
  const float4 *K4_base = reinterpret_cast<const float4 *>(K + bh_offset);
  const float4 *V4_base = reinterpret_cast<const float4 *>(V + bh_offset);
  float4 *out4_ptr =
      reinterpret_cast<float4 *>(out + bh_offset + q * head_dim_pad);

  // Load Q into registers (vectorized)
  bool valid_q = (q < dims.seq_len);
  float4 vec_q = {0.0f, 0.0f, 0.0f, 0.0f};
  if (valid_q && lane_id < head_dim_f4) {
    // Use read-only cache for Q loads to improve global memory throughput
    vec_q = __ldg(&Q4_ptr[lane_id]);
  }

  // Output accumulator (vectorized)
  float4 out_accum = {0.0f, 0.0f, 0.0f, 0.0f};

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // 5. Calculate Loop Bounds (same as v5)
  int max_q_block = (blockIdx.x + 1) * WARPS_PER_BLOCK;
  if (max_q_block > dims.seq_len)
    max_q_block = dims.seq_len;

  // MAIN LOOP: Tiling over K (cooperative loads fill smem as float4)
  for (int k_base = 0; k_base < max_q_block; k_base += TILE_K) {

    // Cooperative Load (All threads help)
    int total_elements_f4 = TILE_K * head_dim_f4; // float4 elements
    for (int i = tid; i < total_elements_f4; i += THREADS_PER_BLOCK) {
      int t = i / head_dim_f4;   // time offset
      int idx = i % head_dim_f4; // float4 index inside head
      int k_curr = k_base + t;

      if (k_curr < dims.seq_len) {
        // Use read-only cache for K/V global loads
        float4 k_tmp = __ldg(&K4_base[k_curr * head_dim_f4 + idx]);
        float4 v_tmp = __ldg(&V4_base[k_curr * head_dim_f4 + idx]);
        smem_K[t][idx] = k_tmp;
        smem_V[t][idx] = v_tmp;
      } else {
        // pad with zeros
        smem_K[t][idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        smem_V[t][idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    __syncthreads();

    // Compute (Per Warp) // Only process if this warp has a valid query
    if (valid_q) {
      for (int t = 0; t < TILE_K; ++t) {
        int k_curr = k_base + t;

        // CAUSAL MASK: Only attend if k <= q
        if (k_curr <= q) {

          // Load K from shared memory (per-lane)
          float4 k_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          if (lane_id < head_dim_f4) {
            k_vec = smem_K[t][lane_id];
          }

          // Dot Product: vec_q (reg) · k_vec (smem)
          float dot = vec_q.x * k_vec.x + vec_q.y * k_vec.y +
                      vec_q.z * k_vec.z + vec_q.w * k_vec.w;

          // Multiply by scale before the warp reduction to reduce
          // the amount of data participating in the shuffle reduction.
          dot *= scale;

          float score = warp_reduce_sum_xor(dot);

          // Softmax Update (online)
          float new_max = fmaxf(running_max, score);
          float alpha = expf(running_max - new_max);
          float weight = expf(score - new_max);

          running_sum = running_sum * alpha + weight;
          running_max = new_max;

          // Load V only when we need it (after weight computed) to
          // reduce register live-range and pressure (avoid spills)
          if (lane_id < head_dim_f4) {
            float4 v_vec = smem_V[t][lane_id];
            out_accum.x = out_accum.x * alpha + weight * v_vec.x;
            out_accum.y = out_accum.y * alpha + weight * v_vec.y;
            out_accum.z = out_accum.z * alpha + weight * v_vec.z;
            out_accum.w = out_accum.w * alpha + weight * v_vec.w;
          }
        }
      }
    }

    // Wait for compute to finish before overwriting SMEM
    __syncthreads();
  }

  // Final Write (normalize and store)
  if (valid_q) {
    float inv_sum = 1.0f / running_sum;
    if (lane_id < head_dim_f4) {
      out_accum.x *= inv_sum;
      out_accum.y *= inv_sum;
      out_accum.z *= inv_sum;
      out_accum.w *= inv_sum;
      out4_ptr[lane_id] = out_accum;
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

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
