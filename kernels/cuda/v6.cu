#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v6: Shared Memory K/V Tiling with Cooperative Loading
// ============================================================================
// This version tiles K/V into shared memory for reuse across warps.
//
// Key insight: All 8 warps in a block process consecutive queries that need
// overlapping K/V ranges. By loading K/V tiles cooperatively into shared
// memory, we get 8x reuse per global memory load.
//
// Block structure:
//   - 8 warps × 32 threads = 256 threads
//   - Each warp handles one query
//   - All warps share K/V tiles in shared memory
//
// Tile sizes:
//   - TILE_K = 32 keys per tile (tuned for occupancy vs reuse tradeoff)
//   - K_tile[TILE_K][head_dim_pad] in shared memory
//   - V_tile[TILE_K][head_dim_pad] in shared memory
//
// Memory access pattern:
//   1. All 256 threads cooperatively load K_tile and V_tile (coalesced)
//   2. __syncthreads()
//   3. Each warp computes attention against tile using SMEM (fast broadcast)
//   4. __syncthreads() before next tile
//
// CRITICAL: To avoid deadlock from __syncthreads():
//   - NO early return for out-of-bounds warps
//   - Use warp_active flag to skip computation but still participate in sync
//   - All threads in block must hit every __syncthreads()
//
// Expected speedup: ~8x reduction in global memory traffic for K/V
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff
#define TILE_K 32 // Keys per tile - balance between SMEM usage and sync overhead

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
  const int thread_id = warp_id * WARP_SIZE + lane_id; // 0..255

  const int q_base = blockIdx.x * WARPS_PER_BLOCK;
  const int q_idx = q_base + warp_id;
  const int bh = blockIdx.y;

  // CRITICAL: DO NOT early return! All threads must reach __syncthreads()
  const bool warp_active =
      (q_idx < (int)dims.seq_len) && (bh < (int)(dims.batch * dims.n_heads));

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Shared memory for K and V tiles
  extern __shared__ float smem[];
  float *K_tile = smem;                         // [TILE_K][head_dim_pad]
  float *V_tile = smem + TILE_K * head_dim_pad; // [TILE_K][head_dim_pad]

  // Each lane handles 4 consecutive floats (float4 pattern)
  const int d_base = lane_id * 4;
  const bool lane_active = (d_base < head_dim);

  // Load Q into registers (only active warps)
  float4 q_vec = make_float4(0.f, 0.f, 0.f, 0.f);
  if (warp_active && lane_active) {
    q_vec = __ldg(reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]));
  }

  // Online softmax state
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float4 out_acc = make_float4(0.f, 0.f, 0.f, 0.f);

  // Maximum key index any warp in this block needs
  const int max_q_in_block = min(q_base + WARPS_PER_BLOCK - 1, (int)dims.seq_len - 1);
  const int max_key = max_q_in_block; // causal: need keys 0..max_q_in_block
  const int num_tiles = CEIL_DIV(max_key + 1, TILE_K);

  // Process K/V in tiles
  for (int tile = 0; tile < num_tiles; tile++) {
    const int k_tile_start = tile * TILE_K;
    const int k_tile_end = min(k_tile_start + TILE_K, max_key + 1);

    // =========================================================================
    // Phase 1: Cooperative K/V load into shared memory
    // All 256 threads participate - coalesced access pattern
    // =========================================================================
    // Each thread loads multiple float4s to cover TILE_K * head_dim_pad
    // Thread layout: thread_id loads positions [thread_id, thread_id+256, ...]
    #pragma unroll 2
    for (int idx = thread_id; idx < TILE_K * ((int)head_dim_pad / 4); idx += THREADS_PER_BLOCK) {
      const int k_local = idx / ((int)head_dim_pad / 4); // which key in tile
      const int d_idx = (idx % ((int)head_dim_pad / 4)) * 4; // which float4
      const int k_global = k_tile_start + k_local;

      float4 k_val = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 v_val = make_float4(0.f, 0.f, 0.f, 0.f);

      if (k_global <= max_key && d_idx < head_dim) {
        const size_t kv_offset = bh_offset + k_global * head_dim_pad + d_idx;
        k_val = __ldg(reinterpret_cast<const float4 *>(&K[kv_offset]));
        v_val = __ldg(reinterpret_cast<const float4 *>(&V[kv_offset]));
      }

      // Store to shared memory (AoS layout for easy access)
      *reinterpret_cast<float4 *>(&K_tile[k_local * head_dim_pad + d_idx]) = k_val;
      *reinterpret_cast<float4 *>(&V_tile[k_local * head_dim_pad + d_idx]) = v_val;
    }

    // CRITICAL: All threads must sync before using shared memory
    __syncthreads();

    // =========================================================================
    // Phase 2: Each warp computes attention for its query using SMEM
    // =========================================================================
    if (warp_active) {
      const int tile_keys = k_tile_end - k_tile_start;

      #pragma unroll 4
      for (int k_local = 0; k_local < tile_keys; k_local++) {
        const int k_global = k_tile_start + k_local;

        // Causal mask: skip keys beyond this query's position
        if (k_global > q_idx)
          break; // Safe to break here - within warp, all lanes break together

        // Load K from shared memory
        float4 k_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        if (lane_active) {
          k_vec = *reinterpret_cast<float4 *>(&K_tile[k_local * head_dim_pad + d_base]);
          v_vec = *reinterpret_cast<float4 *>(&V_tile[k_local * head_dim_pad + d_base]);
        }

        // Dot product Q · K
        float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                    q_vec.z * k_vec.z + q_vec.w * k_vec.w;
        float score = warp_reduce_sum_xor(dot) * scale;

        // Online softmax update
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

    // CRITICAL: All threads must sync before next tile overwrites SMEM
    __syncthreads();
  }

  // =========================================================================
  // Write output (only active warps)
  // =========================================================================
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

  // Shared memory: K_tile + V_tile = 2 * TILE_K * head_dim_padded * sizeof(float)
  size_t smem_size = 2 * TILE_K * dims.head_dim_padded * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims);
}
#endif
