#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v3_3: Batched shared memory approach
//
// Strategy: Load TILE_K keys worth of K/V into smem at once.
// Accumulate partial dot products for all TILE_K keys across d_tiles,
// then compute softmax and V accumulation for the entire tile.
//
// Shared memory: K_smem[TILE_K][TILE_D], V_smem[TILE_K][TILE_D]
// Registers: dot_partials[TILE_K] per thread to accumulate across d_tiles
//
// This amortizes smem load overhead over TILE_K keys and enables
// better memory coalescing.
// ============================================================================

#define WARP_SIZE 32
#define TILE_Q 8
#define TILE_K 8 // Number of K positions to batch
#define TILE_D 32
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const size_t head_dim_pad) {

  // Block: (TILE_D, TILE_Q) = (32, 8) = 256 threads
  // - warp_id (threadIdx.y): which query position in tile
  // - lane_id (threadIdx.x): which dimension element

  static_assert(TILE_D == WARP_SIZE, "TILE_D must equal WARP_SIZE");

  const int lane_id = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int tid = warp_id * TILE_D + lane_id; // Linear thread id 0..255

  const int q = blockIdx.x * TILE_Q + warp_id;
  const int bh = blockIdx.y;

  const size_t head_dim = dims.head_dim;
  const size_t seq_len = dims.seq_len;

  // Shared memory for TILE_K rows of K and V
  __shared__ float K_smem[TILE_K][TILE_D];
  __shared__ float V_smem[TILE_K][TILE_D];

  const bool valid_q = (q < seq_len) && (bh < dims.batch * dims.n_heads);

  const float scale = rsqrtf((float)head_dim);
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * seq_len * head_dim_pad) +
                           h * (seq_len * head_dim_pad);
  const size_t q_offset = bh_offset + q * head_dim_pad;

  // Initialize output
  if (valid_q) {
    for (size_t d = lane_id; d < head_dim; d += TILE_D)
      out[q_offset + d] = 0.0f;
  }

  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  const int num_k_tiles = CEIL_DIV(seq_len, TILE_K);
  const int num_d_tiles = CEIL_DIV(head_dim, TILE_D);

  // Max q in block - for early k_tile termination
  const int max_q_in_block =
      min((int)((blockIdx.x + 1) * TILE_Q - 1), (int)(seq_len - 1));

  // Loop over K tiles
  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    const int k_tile_start = k_tile * TILE_K;

    // Early exit if entire tile is beyond causal mask for all warps
    if (k_tile_start > max_q_in_block)
      break;

    // Partial dot products for each k in tile (per-thread registers)
    float dot_partials[TILE_K];
#pragma unroll
    for (int i = 0; i < TILE_K; ++i)
      dot_partials[i] = 0.0f;

    // === Phase 1: Accumulate dot products across all d_tiles ===
    for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
      const int d_global = d_tile * TILE_D + lane_id;

      // Cooperative load: all 256 threads load K_smem[TILE_K][TILE_D]
      // Total elements: TILE_K * TILE_D = 8 * 32 = 256 (perfect!)
      // Each thread loads exactly one element
      {
        const int k_local = tid / TILE_D; // 0..7
        const int d_local = tid % TILE_D; // 0..31
        const int k_global = k_tile_start + k_local;
        const int d_elem = d_tile * TILE_D + d_local;

        if (k_global < seq_len && d_elem < head_dim) {
          const size_t kv_offset = bh_offset + k_global * head_dim_pad + d_elem;
          K_smem[k_local][d_local] = K[kv_offset];
        } else {
          K_smem[k_local][d_local] = 0.0f;
        }
      }
      __syncthreads();

      // Load Q value for this thread's dimension
      float q_val = 0.0f;
      if (valid_q && d_global < head_dim) {
        q_val = Q[q_offset + d_global];
      }

// Accumulate partial dots for all TILE_K keys
#pragma unroll
      for (int k_local = 0; k_local < TILE_K; ++k_local) {
        dot_partials[k_local] += q_val * K_smem[k_local][lane_id];
      }

      __syncthreads();
    }

    // === Phase 2: Reduce dot products and compute softmax for each k ===
    // Then accumulate V contributions

    for (int k_local = 0; k_local < TILE_K; ++k_local) {
      const int k = k_tile_start + k_local;

      // Bounds check
      if (k >= seq_len)
        break;

      // Reduce partial dot across warp
      float score = warp_reduce_sum(dot_partials[k_local]) * scale;

      // Causal mask
      if (!valid_q || k > q)
        continue;

      // Online softmax update
      float new_max = fmaxf(softmax_max, score);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score - new_max);

      softmax_sum = softmax_sum * alpha + weight;

      // Accumulate V[k] into output
      // Need to load V for this k across all d_tiles
      const size_t k_offset = bh_offset + k * head_dim_pad;

      for (int d_tile = 0; d_tile < num_d_tiles; ++d_tile) {
        const int d_global = d_tile * TILE_D + lane_id;

        // Load V into smem (reuse K_smem row 0 as temp, or use V_smem[0])
        if (warp_id == 0) {
          V_smem[0][lane_id] =
              (d_global < head_dim) ? V[k_offset + d_global] : 0.0f;
        }
        __syncthreads();

        if (d_global < head_dim) {
          out[q_offset + d_global] =
              out[q_offset + d_global] * alpha + weight * V_smem[0][lane_id];
        }

        __syncthreads();
      }

      softmax_max = new_max;
    }
  }

  // Final normalization
  if (valid_q) {
    float inv_sum = 1.0f / softmax_sum;
    for (size_t d = lane_id; d < head_dim; d += TILE_D)
      out[q_offset + d] *= inv_sum;
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

  dim3 block(TILE_D, TILE_Q);
  dim3 grid(CEIL_DIV(dims.seq_len, TILE_Q), dims.batch * dims.n_heads);

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
