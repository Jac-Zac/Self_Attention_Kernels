#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/ptx_utils.cuh"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v7: Async Copy with Double Buffering (sm_80+ / Ampere)
// ============================================================================
// This version uses cp.async for asynchronous GMEM->SMEM transfers with
// double buffering to overlap memory loads with computation.
//
// Key optimizations:
// 1. cp.async.cg: Async copy that bypasses L1 cache (direct GMEM->SMEM)
// 2. Double buffering: Load tile N+1 while computing on tile N
// 3. XOR swizzling: Bank-conflict-free SMEM access (from v6.2)
// 4. exp2f + FMA fusion: Optimized softmax (from v4.2)
//
// Double Buffering Pipeline:
//   Iteration 0:
//     - Async load tile 0 into buffer A
//     - Commit group 0
//     - Async load tile 1 into buffer B (if exists)
//     - Commit group 1
//     - Wait for group 0
//     - Compute on buffer A
//
//   Iteration 1+:
//     - Async load tile N+1 into buffer (N%2) [reusing old buffer]
//     - Commit group
//     - Wait for previous group
//     - Compute on buffer ((N+1)%2)
//
// Memory layout:
//   SMEM: [K_buf0][K_buf1][V_buf0][V_buf1]
//   Each buffer: TILE_K x head_dim_pad floats
//
// Requirements:
// - sm_80 or higher (Ampere, Ada Lovelace, Hopper)
// - Compile with: nvcc -arch=sm_80 (or higher)
//
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff
#define TILE_K 32
#define NUM_BUFFERS 2

// Use exp2f with fused scale (from v4.2)
#define LOG2E 1.4426950408889634f

// XOR swizzle helper (from v6.2)
__device__ __forceinline__ int swizzle_f4_idx(int row_idx, int f4_idx) {
  return (row_idx & 7) ^ f4_idx;
}

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

// Async load a tile of K and V into shared memory with XOR swizzling
// Uses cp.async for L1-bypassing async GMEM->SMEM transfer
__device__ __forceinline__ void async_load_tile(
    float *K_buf, float *V_buf, const float *K, const float *V,
    const int k_tile_start, const int max_k, const size_t bh_offset,
    const int head_dim, const int head_dim_pad, const int thread_id) {

  const int float4s_per_row = head_dim_pad / 4;
  const int float4s_per_tile = TILE_K * float4s_per_row;

  for (int idx = thread_id; idx < float4s_per_tile; idx += THREADS_PER_BLOCK) {
    int k_local = idx / float4s_per_row;
    int src_f4_idx = idx % float4s_per_row;
    int d = src_f4_idx * 4;
    int k_global = k_tile_start + k_local;

    // Calculate swizzled SMEM offset
    int swizzled_f4 = swizzle_f4_idx(k_local, src_f4_idx);
    int smem_offset = k_local * head_dim_pad + swizzled_f4 * 4;

    // Check bounds
    bool valid = (k_global <= max_k) && (d < head_dim);

    if (valid) {
      size_t gmem_offset = bh_offset + k_global * head_dim_pad + d;
      // Use async copy with zero-fill for out-of-bounds
      cp_async_16(&K_buf[smem_offset], &K[gmem_offset]);
      cp_async_16(&V_buf[smem_offset], &V[gmem_offset]);
    } else {
      // Zero-fill for out-of-bounds (write zeros directly)
      *reinterpret_cast<float4 *>(&K_buf[smem_offset]) =
          make_float4(0.f, 0.f, 0.f, 0.f);
      *reinterpret_cast<float4 *>(&V_buf[smem_offset]) =
          make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int thread_id = warp_id * WARP_SIZE + lane_id;

  const int q_base = blockIdx.x * WARPS_PER_BLOCK;
  const int q_idx = q_base + warp_id;
  const int bh = blockIdx.y;

  // CRITICAL: NO early return - all threads must participate in async loads
  const bool warp_active =
      (q_idx < (int)dims.seq_len) && (bh < (int)(dims.batch * dims.n_heads));

  const int head_dim = (int)dims.head_dim;
  const int head_dim_pad = (int)dims.head_dim_padded;
  const float scale_log2e = rsqrtf((float)head_dim) * LOG2E;

  const int b = bh / (int)dims.n_heads;
  const int h = bh % (int)dims.n_heads;

  const size_t bh_offset =
      (size_t)b * dims.n_heads * dims.seq_len * head_dim_pad +
      (size_t)h * dims.seq_len * head_dim_pad;

  // Double-buffered shared memory layout
  extern __shared__ float smem[];
  const int tile_size = TILE_K * head_dim_pad;
  float *K_buf[NUM_BUFFERS] = {smem, smem + tile_size};
  float *V_buf[NUM_BUFFERS] = {smem + 2 * tile_size, smem + 3 * tile_size};

  // For float4 access pattern
  const int d_base = lane_id * 4;
  const int f4_idx = lane_id;
  const bool lane_active = (d_base < head_dim);

  // Load Q (not tiled, fits in registers)
  float4 q_vec = make_float4(0.f, 0.f, 0.f, 0.f);
  if (warp_active && lane_active) {
    q_vec = __ldg(reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]));
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float4 out_acc = make_float4(0.f, 0.f, 0.f, 0.f);

  const int max_q_in_block =
      min(q_base + WARPS_PER_BLOCK - 1, (int)dims.seq_len - 1);
  const int num_tiles = (max_q_in_block + TILE_K) / TILE_K;

  // =========================================================================
  // Pipeline Prologue: Start loading first tile(s)
  // =========================================================================

  // Load tile 0 into buffer 0
  async_load_tile(K_buf[0], V_buf[0], K, V, 0, max_q_in_block, bh_offset,
                  head_dim, head_dim_pad, thread_id);
  cp_async_commit_group();

  // If more than one tile, start loading tile 1 into buffer 1
  if (num_tiles > 1) {
    async_load_tile(K_buf[1], V_buf[1], K, V, TILE_K, max_q_in_block, bh_offset,
                    head_dim, head_dim_pad, thread_id);
  }
  cp_async_commit_group();

  // =========================================================================
  // Main Loop: Process tiles with double buffering
  // =========================================================================
  for (int tile = 0; tile < num_tiles; tile++) {
    const int k_tile_start = tile * TILE_K;
    const int curr_buf = tile % NUM_BUFFERS;
    const int next_buf = (tile + 1) % NUM_BUFFERS;

    // Wait for current tile to finish loading
    // wait_group<1> means: wait until at most 1 group is still in flight
    // After this, the current buffer is ready
    cp_async_wait_group<1>();
    __syncthreads();

    // Start loading next tile (if exists) into the other buffer
    // This overlaps with computation below
    if (tile + 2 < num_tiles) {
      async_load_tile(K_buf[next_buf], V_buf[next_buf], K, V,
                      (tile + 2) * TILE_K, max_q_in_block, bh_offset, head_dim,
                      head_dim_pad, thread_id);
      cp_async_commit_group();
    }

    // =========================================================================
    // Compute attention for current tile
    // =========================================================================
    if (warp_active) {
      float *K_tile = K_buf[curr_buf];
      float *V_tile = V_buf[curr_buf];

      for (int k_local = 0; k_local < TILE_K; k_local++) {
        int k_global = k_tile_start + k_local;
        if (k_global > q_idx)
          break;
        if (k_global > max_q_in_block)
          break;

        // Load K and V with XOR swizzled access
        float4 k_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        if (lane_active) {
          int swizzled_f4 = swizzle_f4_idx(k_local, f4_idx);
          int smem_offset = k_local * head_dim_pad + swizzled_f4 * 4;

          k_vec = *reinterpret_cast<float4 *>(&K_tile[smem_offset]);
          v_vec = *reinterpret_cast<float4 *>(&V_tile[smem_offset]);
        }

        // Dot product
        float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z +
                    q_vec.w * k_vec.w;
        float score = warp_reduce_sum_xor(dot);

        // Online softmax with exp2f fusion
        float new_max = fmaxf(running_max, score);
        float alpha = exp2f((running_max - new_max) * scale_log2e);
        float weight = exp2f((score - new_max) * scale_log2e);

        running_sum = fmaf(running_sum, alpha, weight);
        out_acc.x = fmaf(out_acc.x, alpha, weight * v_vec.x);
        out_acc.y = fmaf(out_acc.y, alpha, weight * v_vec.y);
        out_acc.z = fmaf(out_acc.z, alpha, weight * v_vec.z);
        out_acc.w = fmaf(out_acc.w, alpha, weight * v_vec.w);
        running_max = new_max;
      }
    }

    __syncthreads();
  }

  // Wait for any remaining async operations
  cp_async_wait_all();

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

  // Double-buffered SMEM: 2x K buffers + 2x V buffers
  size_t smem_size = 4 * TILE_K * dims.head_dim_padded * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims);
}
#endif
