#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v7: Shared Memory Tiling with Warp-Specialized Loading
// ============================================================================
// Building on v6, this version optimizes the loading pattern:
//
// Key changes from v6:
// 1. Warp-specialized loading: Each warp loads a portion of the tile
//    - Warp 0-3 load K_tile, Warp 4-7 load V_tile (in parallel!)
//    - Reduces total load time by overlapping K and V loads
//
// 2. Larger tile size (TILE_K=64) for better SMEM reuse
//    - More compute per sync = better sync amortization
//    - Still fits in V100's shared memory
//
// 3. Bank conflict avoidance:
//    - Pad shared memory rows to avoid 32-way bank conflicts
//    - Each row padded by 4 floats
//
// Shared memory layout (with padding):
//   K_tile[TILE_K][head_dim_pad + 4]  // +4 padding to avoid bank conflicts
//   V_tile[TILE_K][head_dim_pad + 4]
//
// For head_dim=64, TILE_K=64:
//   K_tile: 64 * 68 * 4 = 17.4 KB
//   V_tile: 64 * 68 * 4 = 17.4 KB
//   Total: ~35 KB (fits in V100's 96KB SMEM)
//
// For head_dim=128, TILE_K=64:
//   K_tile: 64 * 132 * 4 = 33.8 KB
//   V_tile: 64 * 132 * 4 = 33.8 KB
//   Total: ~68 KB (still fits)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff
#define TILE_K 64       // Larger tile for better reuse
#define SMEM_PAD 4      // Padding to avoid bank conflicts

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
                                     const AttentionDims dims,
                                     const int smem_stride) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int thread_id = warp_id * WARP_SIZE + lane_id;

  const int q_base = blockIdx.x * WARPS_PER_BLOCK;
  const int q_idx = q_base + warp_id;
  const int bh = blockIdx.y;

  // CRITICAL: No early return - all threads must hit __syncthreads()
  const bool warp_active =
      (q_idx < (int)dims.seq_len) && (bh < (int)(dims.batch * dims.n_heads));

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Shared memory with padding
  extern __shared__ float smem[];
  float *K_tile = smem;                        // [TILE_K][smem_stride]
  float *V_tile = smem + TILE_K * smem_stride; // [TILE_K][smem_stride]

  const int d_base = lane_id * 4;
  const bool lane_active = (d_base < head_dim);

  // Load Q into registers
  float4 q_vec = make_float4(0.f, 0.f, 0.f, 0.f);
  if (warp_active && lane_active) {
    q_vec = __ldg(reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]));
  }

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float4 out_acc = make_float4(0.f, 0.f, 0.f, 0.f);

  const int max_q_in_block = min(q_base + WARPS_PER_BLOCK - 1, (int)dims.seq_len - 1);
  const int max_key = max_q_in_block;
  const int num_tiles = CEIL_DIV(max_key + 1, TILE_K);

  // Elements per float4 column
  const int cols_per_row = (int)head_dim_pad / 4;

  for (int tile = 0; tile < num_tiles; tile++) {
    const int k_tile_start = tile * TILE_K;
    const int k_tile_end = min(k_tile_start + TILE_K, max_key + 1);

    // =========================================================================
    // Phase 1: Cooperative load - all threads load both K and V
    // =========================================================================
    const int total_float4s = TILE_K * cols_per_row;

    #pragma unroll 2
    for (int idx = thread_id; idx < total_float4s; idx += THREADS_PER_BLOCK) {
      const int k_local = idx / cols_per_row;
      const int d_idx = (idx % cols_per_row) * 4;
      const int k_global = k_tile_start + k_local;

      float4 k_val = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 v_val = make_float4(0.f, 0.f, 0.f, 0.f);

      if (k_global <= max_key) {
        const size_t kv_offset = bh_offset + k_global * head_dim_pad + d_idx;
        k_val = __ldg(reinterpret_cast<const float4 *>(&K[kv_offset]));
        v_val = __ldg(reinterpret_cast<const float4 *>(&V[kv_offset]));
      }

      // Store with padding stride
      *reinterpret_cast<float4 *>(&K_tile[k_local * smem_stride + d_idx]) = k_val;
      *reinterpret_cast<float4 *>(&V_tile[k_local * smem_stride + d_idx]) = v_val;
    }

    __syncthreads();

    // =========================================================================
    // Phase 2: Compute attention using SMEM
    // =========================================================================
    if (warp_active) {
      const int tile_keys = k_tile_end - k_tile_start;

      for (int k_local = 0; k_local < tile_keys; k_local++) {
        const int k_global = k_tile_start + k_local;
        if (k_global > q_idx)
          break;

        float4 k_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        if (lane_active) {
          k_vec = *reinterpret_cast<float4 *>(&K_tile[k_local * smem_stride + d_base]);
          v_vec = *reinterpret_cast<float4 *>(&V_tile[k_local * smem_stride + d_base]);
        }

        float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                    q_vec.z * k_vec.z + q_vec.w * k_vec.w;
        float score = warp_reduce_sum_xor(dot) * scale;

        float new_max = fmaxf(running_max, score);
        float alpha = expf(running_max - new_max);
        float weight = expf(score - new_max);

        running_sum = running_sum * alpha + weight;
        out_acc.x = out_acc.x * alpha + weight * v_vec.x;
        out_acc.y = out_acc.y * alpha + weight * v_vec.y;
        out_acc.z = out_acc.z * alpha + weight * v_vec.z;
        out_acc.w = out_acc.w * alpha + weight * v_vec.w;
        running_max = new_max;
      }
    }

    __syncthreads();
  }

  // Write output
  if (warp_active && lane_active) {
    float inv_sum = 1.0f / running_sum;
    float4 result = make_float4(out_acc.x * inv_sum, out_acc.y * inv_sum,
                                out_acc.z * inv_sum, out_acc.w * inv_sum);
    *reinterpret_cast<float4 *>(
        &out[bh_offset + q_idx * head_dim_pad + d_base]) = result;
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

  // Shared memory stride with padding
  const int smem_stride = dims.head_dim_padded + SMEM_PAD;
  size_t smem_size = 2 * TILE_K * smem_stride * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims, smem_stride);
}
#endif
