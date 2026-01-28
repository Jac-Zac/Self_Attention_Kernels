#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v5.5: One warp computes TILE_Q queries (minimal extension of v5)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define FULL_MASK 0xffffffff

#define MAX_D_PER_LANE 4
#define TILE_K 8
#define TILE_Q 4 // keep small to control registers

__inline__ __device__ float warp_reduce_sum_xor(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1)
    v += __shfl_xor_sync(FULL_MASK, v, m);
  return v;
}

extern __shared__ float smem[];

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out, AttentionDims dims) {

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int tid = warp * WARP_SIZE + lane;

  const int bh = blockIdx.y;

  const int head_dim = dims.head_dim;
  const int head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset =
      ((size_t)b * dims.n_heads + h) * dims.seq_len * head_dim_pad;

  // --------------------------------------------------------------------------
  // Query tile this warp owns
  // --------------------------------------------------------------------------
  const int q_tile_base = (blockIdx.x * WARPS_PER_BLOCK + warp) * TILE_Q;

  // Warp active if *any* query in tile is valid
  const bool warp_active = (q_tile_base < (int)dims.seq_len);

  // --------------------------------------------------------------------------
  // Shared memory layout
  // [ Q_tile  ][ K_tile ][ V_tile ]
  // --------------------------------------------------------------------------
  float *out_tile = smem;
  float *Q_tile = out_tile + TILE_Q * head_dim_pad;
  float *K_tile = Q_tile + TILE_Q * head_dim_pad;
  float *V_tile = K_tile + TILE_K * head_dim_pad;

  // --------------------------------------------------------------------------
  // Load Q tile into shared memory
  // Each warp loads its own TILE_Q queries
  // Initialize also out
  // --------------------------------------------------------------------------
  if (warp_active) {
    for (int q = 0; q < TILE_Q; ++q) {
      int q_global = q_tile_base + q;
      for (int i = 0; i < MAX_D_PER_LANE; ++i) {
        int d = lane + i * WARP_SIZE;
        float v = 0.f;

        if (q_global < (int)dims.seq_len && d < head_dim) {
          v = Q[bh_offset + (size_t)q_global * head_dim_pad + d];
        }
        Q_tile[q * head_dim_pad + d] = v;
      }
    }
  }

  __syncthreads();

  // --------------------------------------------------------------------------
  // Per-query softmax state (registers)
  // --------------------------------------------------------------------------
  float running_max[TILE_Q];
  float running_sum[TILE_Q];

  for (int q = 0; q < TILE_Q; ++q) {
    running_max[q] = -FLT_MAX;
    running_sum[q] = 0.f;
  }

  // --------------------------------------------------------------------------
  // K/V tiling loop (unchanged logic)
  // --------------------------------------------------------------------------
  const int max_q = min(q_tile_base + TILE_Q - 1, (int)dims.seq_len - 1);
  const int num_tiles = (max_q + TILE_K) / TILE_K;

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int k_tile_start = tile * TILE_K;

    // Load K/V tile (block-uniform)
    for (int idx = tid; idx < TILE_K * head_dim_pad; idx += THREADS_PER_BLOCK) {
      int k_local = idx / head_dim_pad;
      int d = idx % head_dim_pad;
      int k_global = k_tile_start + k_local;

      float k_val = 0.f;
      float v_val = 0.f;
      if (k_global <= max_q && d < head_dim) {
        size_t off = bh_offset + (size_t)k_global * head_dim_pad + d;
        k_val = K[off];
        v_val = V[off];
      }

      K_tile[k_local * head_dim_pad + d] = k_val;
      V_tile[k_local * head_dim_pad + d] = v_val;
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Compute attention for each query in the tile
    // ------------------------------------------------------------------------
    for (int q = 0; q < TILE_Q; ++q) {
      int q_global = q_tile_base + q;

      if (!warp_active || q_global >= (int)dims.seq_len)
        continue;

      for (int k_local = 0; k_local < TILE_K; ++k_local) {
        int k_global = k_tile_start + k_local;

        if (k_global > q_global)
          continue;

        float dot = 0.f;
        for (int i = 0; i < MAX_D_PER_LANE; ++i) {
          int d = lane + i * WARP_SIZE;
          if (d < head_dim)
            dot += Q_tile[q * head_dim_pad + d] *
                   K_tile[k_local * head_dim_pad + d];
        }

        float score = warp_reduce_sum_xor(dot) * scale;

        float new_max = fmaxf(running_max[q], score);
        float alpha = expf(running_max[q] - new_max);
        float weight = expf(score - new_max);

        running_sum[q] = running_sum[q] * alpha + weight;

        for (int i = 0; i < MAX_D_PER_LANE; ++i) {
          int d = lane + i * WARP_SIZE;
          if (d < head_dim)
            // Accumulate output
            weight *V_tile[k_local * head_dim_pad + d];
        }

        running_max[q] = new_max;
      }
    }

    __syncthreads();
  }

  // --------------------------------------------------------------------------
  // Write outputs
  // --------------------------------------------------------------------------
  if (warp_active) {
    for (int q = 0; q < TILE_Q; ++q) {
      int q_global = q_tile_base + q;
      if (q_global >= (int)dims.seq_len)
        continue;

      float inv_sum = 1.f / running_sum[q];
      for (int i = 0; i < MAX_D_PER_LANE; ++i) {
        int d = lane + i * WARP_SIZE;
        if (d < head_dim) {
          out[bh_offset + (size_t)q_global * head_dim_pad + d] =
              out_tile... * inv_sum;
        }
      }
    }
  }
}

size_t cmhsa_get_workspace_size(const AttentionDims) { return 0; }

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {

  dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
  dim3 grid(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK * TILE_Q),
            dims.batch * dims.n_heads);

  // Correct this
  const size_t smem_bytes =
      (2 * TILE_Q + 2 * TILE_K) * dims.head_dim_padded * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_bytes>>>(Q, K, V, out, dims);
}
#endif
