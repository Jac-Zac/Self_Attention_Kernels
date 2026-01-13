#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4: Cooperative K/V Tile Loading with Causal Skip
// ============================================================================
// Building on v3.1's online softmax approach, this version adds:
//
// 1. Shared memory tiling for K and V
//    - All 8 warps cooperatively load one K/V tile into shared memory
//    - Each warp then processes its query against the shared tile
//    - Reduces global memory bandwidth by reusing K/V across queries
//
// 2. Two-level causal skipping
//    - Tile-level: Skip entire tiles where k_tile_start > max_query_in_block
//    - Warp-level: Each warp only processes keys where k <= q
//
// 3. Outer loop over K-tiles, inner processing per warp
//    - Flash Attention v2 style: better for causal attention
//    - K/V tile loaded once, reused by all queries in block that need it
//
// Memory layout:
//   s_K[TILE_K][MAX_HEAD_DIM] - one tile of keys in shared memory
//   s_V[TILE_K][MAX_HEAD_DIM] - one tile of values in shared memory
//
// ============================================================================

#define MAX_HEAD_DIM 128
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define TILE_K 8 // Keys per tile - tune based on occupancy
#define WARP_MASK 0xffffffff

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__global__ void
cmhsa_forward_kernel(const float *RESTRICT Q, const float *RESTRICT K,
                     const float *RESTRICT V, float *RESTRICT out,
                     const AttentionDims dims, const size_t head_dim_pad) {

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  const size_t head_dim = dims.head_dim;
  const size_t seq_len = dims.seq_len;

  // Early exit for out-of-bounds batch/head
  if (bh >= (int)(dims.batch * dims.n_heads))
    return;

  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * seq_len * head_dim_pad) +
                           h * (seq_len * head_dim_pad);
  const size_t q_offset = bh_offset + q * head_dim_pad;
  const size_t out_offset = q_offset;

  // Shared memory for K/V tiles (static allocation)
  __shared__ float s_K[TILE_K][MAX_HEAD_DIM];
  __shared__ float s_V[TILE_K][MAX_HEAD_DIM];

  // Online softmax state (per-warp, in registers)
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Initialize output in global memory
  // (v4.1 will use register accumulator instead)
  if (q < (int)seq_len) {
    for (int d = lane_id; d < (int)head_dim; d += WARP_SIZE)
      out[out_offset + d] = 0.0f;
  }

  // Max query position in this block (for early tile termination)
  const int max_q_in_block =
      min((int)((blockIdx.x + 1) * WARPS_PER_BLOCK - 1), (int)(seq_len - 1));

  // Number of K-tiles we need to process for this block
  const int num_k_tiles = CEIL_DIV(max_q_in_block + 1, TILE_K);

  // Thread ID for cooperative loading (0-255)
  const int tid = warp_id * WARP_SIZE + lane_id;

  // === OUTER LOOP: Over K/V tiles ===
  for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    const int k_tile_start = k_tile * TILE_K;
    const int k_tile_end = min(k_tile_start + TILE_K, (int)seq_len);
    const int tile_height = k_tile_end - k_tile_start;

    // === COOPERATIVE LOAD: All 256 threads load K/V tile ===
    // Ensures previous tile processing is complete before overwriting
    __syncthreads();

    // Load pattern: thread tid loads elements tid, tid+256, tid+512, ...
    // For TILE_K=32, head_dim=128: 4096 elements, 16 per thread
    const int total_elements = tile_height * head_dim;

    for (int idx = tid; idx < total_elements;
         idx += WARPS_PER_BLOCK * WARP_SIZE) {
      const int k_local = idx / head_dim;
      const int d = idx % head_dim;
      const int k_global = k_tile_start + k_local;

      const size_t k_offset = bh_offset + k_global * head_dim_pad;
      s_K[k_local][d] = K[k_offset + d];
      s_V[k_local][d] = V[k_offset + d];
    }

    // Ensure all threads see the loaded tile before processing
    __syncthreads();

    // === PER-WARP PROCESSING ===
    // Skip if this warp's query is out of bounds or doesn't need this tile
    if (q >= (int)seq_len || k_tile_start > q)
      continue;

    // How many keys in this tile are valid for query q? (causal mask)
    const int valid_k_end = min(q, k_tile_end - 1);
    const int valid_k_count = valid_k_end - k_tile_start + 1;

    // Process each valid key in the tile
    for (int k_local = 0; k_local < valid_k_count; k_local++) {

      // Q·K dot product (warp-parallel across head_dim)
      float dot_partial = 0.0f;
      for (int d = lane_id; d < (int)head_dim; d += WARP_SIZE) {
        dot_partial += Q[q_offset + d] * s_K[k_local][d];
      }
      float score = warp_reduce_sum_xor(dot_partial) * scale;

      // Online softmax update (same as v3.1)
      float new_max = fmaxf(softmax_max, score);
      float alpha = expf(softmax_max - new_max);
      float weight = expf(score - new_max);

      softmax_sum = softmax_sum * alpha + weight;

      // Update output with rescaled accumulator + new weighted value
      for (int d = lane_id; d < (int)head_dim; d += WARP_SIZE) {
        out[out_offset + d] =
            out[out_offset + d] * alpha + weight * s_V[k_local][d];
      }

      softmax_max = new_max;
    }
  }

  // === NORMALIZE OUTPUT ===
  if (q < (int)seq_len) {
    float inv_sum = 1.0f / softmax_sum;
    for (int d = lane_id; d < (int)head_dim; d += WARP_SIZE)
      out[out_offset + d] *= inv_sum;
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

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims,
                                        dims.head_dim_padded);
}
#endif
