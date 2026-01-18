#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v6: Cooperative K/V Shared Memory Loading
// ============================================================================
// Building on v5's K-tiling, this version adds cooperative loading of K and V
// tiles into shared memory to dramatically reduce global memory access.
//
// Key changes from v5:
// - K and V tiles loaded cooperatively by all 256 threads (8 warps × 32 lanes)
// - Shared memory buffers for K and V, reused within each tile iteration
// - Dynamic shared memory allocation to support different head dimensions
// - Synchronization at end of tile loop to ensure safe shared memory updates
//
// Cooperative Load Distribution:
// - Each warp loads 4 keys (TILE_K/8 = 32/8)
// - Each lane loads 4 head_dim elements (128/32 = 4)
// - Total: 32 keys × 128 elements = 4096 floats per buffer
// - Two buffers (K and V) = 8192 floats = 32KB of shared memory
//
// Shared Memory Layout (dynamic allocation):
//   K_smem[TILE_K * head_dim_pad]  - Keys for current tile
//   V_smem[TILE_K * head_dim_pad]  - Values for current tile
//   scores[TILE_K * WARPS_PER_BLOCK] - Pre-computed Q·K scores
//
// Supported head_dim: up to 128
// ============================================================================

#define WARP_MASK 0xffffffff
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define TILE_K 32        // Keys processed per tile
#define MAX_D_PER_LANE 4 // Support up to head_dim=128

extern __shared__ float smem[];

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  return val;
}

__inline__ __device__ float warp_reduce_max_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = fmaxf(val, __shfl_xor_sync(WARP_MASK, val, mask));
  return val;
}

__inline__ __device__ float warp_broadcast(float val, int src_lane) {
  return __shfl_sync(WARP_MASK, val, src_lane);
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  static_assert(TILE_K <= WARP_SIZE,
                "TILE_K must be smaller or to equal warp size");

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;

  const int q = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const size_t q_offset = bh_offset + q * head_dim_pad;
  const size_t out_offset = q_offset;

  float *K_smem = smem;
  float *V_smem = K_smem + TILE_K * head_dim_pad;
  float *scores = V_smem + TILE_K * head_dim_pad;

  const int num_keys = q + 1;
  const int num_k_tiles = CEIL_DIV(num_keys, TILE_K);

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  float out_accum[MAX_D_PER_LANE];

  float q_reg[MAX_D_PER_LANE];
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out_accum[i] = 0.0f;
    q_reg[i] = (d < head_dim) ? Q[q_offset + d] : 0.0f;
  }

  const int keys_per_warp = TILE_K / WARPS_PER_BLOCK;

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int k_start = tile * TILE_K;
    const int k_end = min(k_start + TILE_K, num_keys);
    const int tile_size = k_end - k_start;

    // Load tiles in shared memory in shared memory
    for (int k_tile_idx = 0; k_tile_idx < keys_per_warp; ++k_tile_idx) {
      int k_global = k_start + warp_id * keys_per_warp + k_tile_idx;

      if (k_global < k_end) {
        size_t k_offset = bh_offset + k_global * head_dim_pad;

        for (int d_idx = 0; d_idx < MAX_D_PER_LANE; ++d_idx) {
          int d = lane_id + d_idx * WARP_SIZE;
          int smem_idx =
              (warp_id * keys_per_warp + k_tile_idx) * head_dim_pad + d;

          if (d < head_dim) {
            K_smem[smem_idx] = K[k_offset + d];
            V_smem[smem_idx] = V[k_offset + d];
          } else {
            K_smem[smem_idx] = 0.0f;
            V_smem[smem_idx] = 0.0f;
          }
        }
      }
    }

    // Wait for all of the threads to get to have loaded things in smem
    __syncthreads();

    for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
      float k_smem_base = k_idx * head_dim_pad;

      float dot_partial = 0.0f;
      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          dot_partial += q_reg[i] * K_smem[(int)k_smem_base + d];
        }
      }

      float score = warp_reduce_sum_xor(dot_partial) * scale;
      if (lane_id == 0) {
        scores[k_idx + (warp_id * TILE_K)] = score;
      }
    }

    // NOTE: This is actually coaleasced memory access
    // Get the score for each lane in the warp from the smem
    float score =
        (lane_id < tile_size) ? scores[lane_id + warp_id * TILE_K] : -FLT_MAX;

    float tile_max = warp_reduce_max_xor(score);

    float new_max = fmaxf(running_max, tile_max);
    float alpha = expf(running_max - new_max);

    running_sum *= alpha;
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      out_accum[i] *= alpha;
    }

    for (int k_idx = 0; k_idx < tile_size; ++k_idx) {
      float v_smem_base = k_idx * head_dim_pad;
      float weight = expf(scores[k_idx + (warp_id * TILE_K)] - new_max);

      running_sum += weight;

      for (int i = 0; i < MAX_D_PER_LANE; i++) {
        int d = lane_id + i * WARP_SIZE;
        if (d < head_dim) {
          out_accum[i] += weight * V_smem[(int)v_smem_base + d];
        }
      }
    }
    running_max = new_max;

    // Wait to avoid overwriting shared memory
    __syncthreads();
  }

  // Final normalization
  float inv_sum = 1.0f / running_sum;
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      out[out_offset + d] = out_accum[i] * inv_sum;
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

  size_t smem_size = 2 * TILE_K * dims.head_dim_padded * sizeof(float) +
                     TILE_K * WARPS_PER_BLOCK * sizeof(float);

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims);
}
#endif
