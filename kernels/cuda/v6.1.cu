#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v6.1: Shared Memory Tiling with Bank Conflict Reduction (Padding)
// ============================================================================
// Building on v6's shared memory tiling approach, this version reduces bank
// conflicts by adding padding to each row in shared memory.
//
// Bank Conflict Problem:
// - GPU shared memory has 32 banks (4 bytes each = 128 bytes per bank cycle)
// - When threads in a warp access the same bank but different addresses,
//   accesses are serialized (bank conflict)
// - With our float4 access pattern (16 bytes per thread), we effectively
//   have 8 banks of 16 bytes each
// - Without padding, consecutive rows map to the same bank positions
//
// Solution: Row Padding
// - Add SMEM_PAD floats (4 floats = 16 bytes) to each row
// - This shifts each row's bank alignment, distributing accesses
// - Simple and effective, though uses slightly more shared memory
//
// Memory layout change:
//   Before: K_tile[row][head_dim_pad]        -> Same bank alignment per row
//   After:  K_tile[row][head_dim_pad + PAD]  -> Shifted bank alignment
//
// Expected improvements:
// - Reduced bank conflicts during SMEM reads
// - Lower mio_throttle and short_scoreboard stalls
// - ~10-30% performance improvement depending on conflict severity
//
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff
#define TILE_K 32

// Padding to reduce bank conflicts (4 floats = 16 bytes = 1 float4)
// This shifts each row by one 16-byte bank, breaking conflict patterns
#define SMEM_PAD 4

// Use exp2f with fused scale (from v4.2)
#define LOG2E 1.4426950408889634f

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const int smem_row_stride) {

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
  // Use padded stride for SMEM layout
  float *K_tile = smem;
  float *V_tile = smem + TILE_K * smem_row_stride;

  // For float4 access pattern: each lane handles 4 consecutive floats
  const int d_base = lane_id * 4;
  const bool lane_active = (d_base < head_dim);

  // Load Q as float4
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
    // Phase 1: Load K and V tiles using float4 with PADDED SMEM layout
    // =========================================================================
    for (int idx = thread_id; idx < float4s_per_tile;
         idx += THREADS_PER_BLOCK) {
      int k_local = idx / float4s_per_row;
      int f4_idx = idx % float4s_per_row;
      int d = f4_idx * 4;
      int k_global = k_tile_start + k_local;

      float4 k_val = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 v_val = make_float4(0.f, 0.f, 0.f, 0.f);

      if (k_global <= max_q_in_block && d < head_dim) {
        size_t offset = bh_offset + k_global * head_dim_pad + d;
        k_val = __ldg(reinterpret_cast<const float4 *>(&K[offset]));
        v_val = __ldg(reinterpret_cast<const float4 *>(&V[offset]));
      }

      // Store to shared memory with PADDED stride
      // This is the key change: smem_row_stride includes padding
      *reinterpret_cast<float4 *>(&K_tile[k_local * smem_row_stride + d]) =
          k_val;
      *reinterpret_cast<float4 *>(&V_tile[k_local * smem_row_stride + d]) =
          v_val;
    }

    __syncthreads();

    // =========================================================================
    // Phase 2: Compute attention with PADDED SMEM reads
    // =========================================================================
    if (warp_active) {
      for (int k_local = 0; k_local < TILE_K; k_local++) {
        int k_global = k_tile_start + k_local;
        if (k_global > q_idx)
          break;
        if (k_global > max_q_in_block)
          break;

        // Load K and V from shared memory with PADDED stride
        float4 k_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
        if (lane_active) {
          k_vec = *reinterpret_cast<float4 *>(
              &K_tile[k_local * smem_row_stride + d_base]);
          v_vec = *reinterpret_cast<float4 *>(
              &V_tile[k_local * smem_row_stride + d_base]);
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

  // Calculate padded SMEM row stride
  const int smem_row_stride = dims.head_dim_padded + SMEM_PAD;

  // Shared memory: 2 tiles (K and V) with padded rows
  size_t smem_size = 2 * TILE_K * smem_row_stride * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims,
                                                   smem_row_stride);
}
#endif
