#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5: v4 + Block-Cooperative Shared Memory Tiling
// ============================================================================
// - Keeps the simple register structure of v4.
// - Adds Shared Memory for K and V.
// - All 8 Warps (256 threads) cooperate to load the K/V tiles.
// - Eliminates repeated Global Memory hits for K/V.
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff

// v4 specific constants
#define MAX_D_PER_LANE 4

// Shared Memory Configuration
#define TILE_K 8
#define MAX_HEAD_DIM 128

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
  const float scale = rsqrtf((float)head_dim);

  // 2. Shared Memory Allocation
  // We need space for TILE_K timesteps, each having head_dim floats
  __shared__ float smem_K[TILE_K][MAX_HEAD_DIM];
  __shared__ float smem_V[TILE_K][MAX_HEAD_DIM];

  // 3. Early Exit (Block Level)
  // Note: We cannot return early for just *some* threads in the block
  // because of __syncthreads(). We only return if the whole BLOCK is invalid.
  // Individual validity checks happen inside.
  if (bh >= (int)(dims.batch * dims.n_heads))
    return;

  // Base Pointers
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;
  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const float *Q_ptr = Q + bh_offset + q * head_dim_pad;
  const float *K_base = K + bh_offset;
  const float *V_base = V + bh_offset;
  float *out_ptr = out + bh_offset + q * head_dim_pad;

  // Load Q into Registers
  // Only valid if this specific warp covers a valid q
  bool valid_q = (q < dims.seq_len);

  float q_r[MAX_D_PER_LANE];
  float out_accum[MAX_D_PER_LANE];

  // Initialize accumulators
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    out_accum[i] = 0.0f;
    int d = lane_id + i * WARP_SIZE;
    // Load Q if valid, else 0
    if (valid_q && d < head_dim) {
      q_r[i] = Q_ptr[d];
    } else {
      q_r[i] = 0.0f;
    }
  }

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // 5. Calculate Loop Bounds
  // All warps must loop together to keep __syncthreads in sync.
  // We loop until the furthest q in the block is satisfied.
  int max_q_block = (blockIdx.x + 1) * WARPS_PER_BLOCK;
  if (max_q_block > dims.seq_len)
    max_q_block = dims.seq_len;

  // ==========================================================================
  // MAIN LOOP: Tiling over K
  // ==========================================================================
  for (int k_base = 0; k_base < max_q_block; k_base += TILE_K) {

    // Cooperative Load (All threads help)
    // We treat the SMEM tile as a flat buffer of size TILE_K * head_dim.
    // Threads iterate with stride 256 to cover the area.
    int total_elements = TILE_K * head_dim; // e.g. 8 * 128 = 1024 floats
    for (int i = tid; i < total_elements; i += THREADS_PER_BLOCK) {
      int t = i / head_dim; // Time offset (0 to 7)
      int d = i % head_dim; // Dimension offset
      int k_curr = k_base + t;

      // Safety check: is k_curr within sequence bounds ?
      if (k_curr < dims.seq_len) {
        smem_K[t][d] = K_base[k_curr * head_dim_pad + d];
        smem_V[t][d] = V_base[k_curr * head_dim_pad + d];
      } else {
        smem_K[t][d] = 0.0f; // Padding
        smem_V[t][d] = 0.0f;
      }
    }

    // Wait for load to finish
    __syncthreads();

    // Compute (Per Warp) // Only process if this warp has a valid query
    if (valid_q) {
      for (int t = 0; t < TILE_K; ++t) {
        int k_curr = k_base + t;

        // CAUSAL MASK: Only attend if k <= q
        if (k_curr <= q) {

          // Dot Product: Q (Reg) * K (Shared)
          float dot = 0.0f;
          for (int i = 0; i < MAX_D_PER_LANE; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
              dot += q_r[i] * smem_K[t][d];
            }
          }

          float score = warp_reduce_sum_xor(dot) * scale;

          // Softmax Update
          float new_max = fmaxf(running_max, score);
          float alpha = expf(running_max - new_max);
          float weight = expf(score - new_max);

          running_sum = running_sum * alpha + weight;
          running_max = new_max;

          // Accumulate V: Output (Reg) += Weight * V (Shared)
          for (int i = 0; i < MAX_D_PER_LANE; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim) {
              out_accum[i] = out_accum[i] * alpha + weight * smem_V[t][d];
            }
          }
        }
      }
    }

    // Wait for compute to finish before overwriting SMEM
    __syncthreads();
  }

  // 6. Final Write
  if (valid_q) {
    float inv_sum = 1.0f / running_sum;
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      int d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        out_ptr[d] = out_accum[i] * inv_sum;
      }
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
  // Grid size covers all sequences
  dim3 grid(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
