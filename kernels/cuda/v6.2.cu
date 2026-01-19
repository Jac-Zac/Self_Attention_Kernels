#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v6.2: Shared Memory Tiling with XOR Swizzling for Bank Conflict Elimination
// ============================================================================
// Building on v6.1's padding approach, this version uses XOR swizzling to
// completely eliminate bank conflicts with zero memory overhead.
//
// XOR Swizzling Theory:
// - GPU shared memory has 32 banks (4 bytes each)
// - With float4 access (16 bytes), we effectively have 8 banks
// - XOR swizzling permutes column indices based on row index
// - Formula: swizzled_col = row_index XOR col_index
// - This ensures threads in a warp access different banks
//
// How it works:
// - When storing: smem[row][row ^ col] = data
// - When loading: data = smem[row][row ^ col]
// - The XOR pattern guarantees that within any warp accessing the same
//   column across different rows, each thread hits a different bank
//
// Key insight from Flash Attention article:
// - Bank conflict occurs when threads access same bank, different addresses
// - XOR with row index creates a bijection for each row
// - Same logical column maps to different physical columns per row
// - Result: Perfect bank distribution across warp
//
// Memory layout (conceptual):
//   Logical view:  K_tile[row][col]
//   Physical:      K_tile[row][row ^ col]  (XOR applied to col index)
//
// Advantages over padding (v6.1):
// - Zero memory overhead (padding wastes 4 floats per row)
// - Complete bank conflict elimination (padding only reduces conflicts)
// - Same memory footprint as original v6
//
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff
#define TILE_K 32

// Use exp2f with fused scale (from v4.2)
#define LOG2E 1.4426950408889634f

// XOR swizzle helper for float4 access pattern
// With float4 (16 bytes), we have 8 effective banks (128 bytes / 16 bytes)
// row_idx: which row in the tile (0 to TILE_K-1)
// f4_idx: which float4 in the row (0 to head_dim_pad/4 - 1)
// Returns the swizzled float4 index
__device__ __forceinline__ int swizzle_f4_idx(int row_idx, int f4_idx) {
  // XOR the lower 3 bits of row with f4_idx
  // This creates a bijection for each row within 8 banks
  return (row_idx & 7) ^ f4_idx;
}

// Convert swizzled float4 index back to byte offset in SMEM row
__device__ __forceinline__ int swizzled_offset(int row_idx, int f4_idx,
                                               int row_stride) {
  int swizzled_f4 = swizzle_f4_idx(row_idx, f4_idx);
  return row_idx * row_stride + swizzled_f4 * 4;
}

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
  const int thread_id = warp_id * WARP_SIZE + lane_id;

  const int q_base = blockIdx.x * WARPS_PER_BLOCK;
  const int q_idx = q_base + warp_id;
  const int bh = blockIdx.y;

  // CRITICAL: NO early return - all threads must participate in SMEM loads
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

  extern __shared__ float smem[];
  // No padding needed with XOR swizzling
  float *K_tile = smem;
  float *V_tile = smem + TILE_K * head_dim_pad;

  // For float4 access pattern: each lane handles 4 consecutive floats
  const int d_base = lane_id * 4;
  const int f4_idx =
      lane_id; // float4 index = lane_id (since d_base = lane * 4)
  const bool lane_active = (d_base < head_dim);

  // Load Q as float4 (Q is not tiled, no swizzling needed)
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

  // Number of float4s per row and per tile
  const int float4s_per_row = head_dim_pad / 4;
  const int float4s_per_tile = TILE_K * float4s_per_row;

  for (int tile = 0; tile < num_tiles; tile++) {
    const int k_tile_start = tile * TILE_K;

    // =========================================================================
    // Phase 1: Load K and V tiles using float4 with XOR SWIZZLED SMEM layout
    // =========================================================================
    for (int idx = thread_id; idx < float4s_per_tile;
         idx += THREADS_PER_BLOCK) {
      int k_local = idx / float4s_per_row;
      int src_f4_idx = idx % float4s_per_row;
      int d = src_f4_idx * 4;
      int k_global = k_tile_start + k_local;

      float4 k_val = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 v_val = make_float4(0.f, 0.f, 0.f, 0.f);

      if (k_global <= max_q_in_block && d < head_dim) {
        size_t offset = bh_offset + k_global * head_dim_pad + d;
        k_val = __ldg(reinterpret_cast<const float4 *>(&K[offset]));
        v_val = __ldg(reinterpret_cast<const float4 *>(&V[offset]));
      }

      // Store to shared memory with XOR SWIZZLED column index
      // Key: swizzle the float4 index, not the byte offset
      int swizzled_f4 = swizzle_f4_idx(k_local, src_f4_idx);
      int smem_offset = k_local * head_dim_pad + swizzled_f4 * 4;

      *reinterpret_cast<float4 *>(&K_tile[smem_offset]) = k_val;
      *reinterpret_cast<float4 *>(&V_tile[smem_offset]) = v_val;
    }

    __syncthreads();

    // =========================================================================
    // Phase 2: Compute attention with XOR SWIZZLED SMEM reads
    // =========================================================================
    if (warp_active) {
      for (int k_local = 0; k_local < TILE_K; k_local++) {
        int k_global = k_tile_start + k_local;
        if (k_global > q_idx)
          break;
        if (k_global > max_q_in_block)
          break;

        // Load K and V from shared memory with XOR SWIZZLED column index
        // CRITICAL: Must apply same swizzle pattern as during store
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

        // Online softmax with exp2f fusion (from v4.2)
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

  // No padding needed - XOR swizzling has zero overhead
  size_t smem_size = 2 * TILE_K * dims.head_dim_padded * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims);
}
#endif
