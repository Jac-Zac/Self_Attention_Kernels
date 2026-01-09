#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v3.0: Online Softmax with K/V Tiling (Flash Attention-style)
// ============================================================================
// Building on v2.5, this version eliminates the O(seq_len²) memory bottleneck:
//
// 1. K/V Tiling (NEW):
//    - Process K and V in chunks of TILE_SIZE instead of full sequence
//    - Load K_tile[TILE_SIZE × head_dim] and V_tile[TILE_SIZE × head_dim] to
//    SMEM
//    - Dramatically reduces memory footprint from O(seq_len²) to O(seq_len)
//    - For seq_len=2048, TILE_SIZE=64: 32× memory reduction
//
// 2. Online Softmax Algorithm (NEW):
//    - Maintains running statistics (max and denominator) across tiles
//    - Each tile: compute scores → update max → rescale previous output →
//    accumulate
//    - Numerically stable: uses "rescaling trick" to avoid overflow
//    - Algorithm from "Online normalizer calculation for softmax" (Milakov &
//    Gimelshein, 2018)
//
//    Key invariant: After processing tile i,
//      m_i = max of all scores seen so far
//      d_i = sum of exp(score - m_i) for all scores seen so far
//      O_i = (1/d_i) * sum of exp(score - m_i) * V for all seen scores
//
// 3. Shared Memory Layout (UPDATED):
//    - Q vector: [head_dim] (kept from v2.5)
//    - K tile: [TILE_SIZE × head_dim] (NEW)
//    - V tile: [TILE_SIZE × head_dim] (NEW)
//    - Total per warp: head_dim + 2*TILE_SIZE*head_dim floats
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff

// Tile size for K/V: tuned for head_dim=128, seq_len~2048-4096
// SMEM usage per warp: (128 + 2*64*128)*4 = 66KB (within 48-96KB limits)
#define TILE_SIZE 32

__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  }
  return val;
}

__inline__ __device__ float warp_reduce_max_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(WARP_MASK, val, mask));
  }
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

  if (q >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // ===========================================================================
  // Shared Memory Layout (per warp):
  //   - q_shared: [head_dim]
  //   - k_tile:   [TILE_SIZE × head_dim]
  //   - v_tile:   [TILE_SIZE × head_dim]
  // ===========================================================================
  extern __shared__ float smem[];
  const size_t smem_per_warp = head_dim + 2 * TILE_SIZE * head_dim;
  float *RESTRICT warp_smem = smem + (warp_id * smem_per_warp);

  float *RESTRICT q_shared = warp_smem;
  float *RESTRICT k_tile = q_shared + head_dim;
  float *RESTRICT v_tile = k_tile + TILE_SIZE * head_dim;

  // Global Memory Offsets
  const size_t bh_offset = b * (dims.n_heads * seq_len * head_dim_pad) +
                           h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // STEP 0: Cache Q in Shared Memory (Same as v2.5)
  // ===========================================================================
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
    q_shared[d] = Q[query_offset + d];
  }
  __syncwarp(WARP_MASK);

  // ===========================================================================
  // Online Softmax State Initialization
  // ===========================================================================
  float m_state = -FLT_MAX; // Running max across all tiles
  float d_state = 0.0f;     // Running denominator: sum(exp(score - m_state))

  // Output accumulators: one float4 per thread handles 4 dimensions
  // Each thread processes dimensions [lane_id*4, lane_id*4+3]
  const int n_float4_per_thread =
      (head_dim_pad + WARP_SIZE * 4 - 1) / (WARP_SIZE * 4);
  float4
      output_acc[8]; // Max 8 float4s per thread (supports head_dim up to 1024)

  // #pragma unroll
  for (int i = 0; i < n_float4_per_thread; ++i) {
    output_acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }

  // ===========================================================================
  // TILE LOOP: Process K/V in chunks of TILE_SIZE
  // ===========================================================================
  const int num_valid_keys = q + 1; // Causal: only keys 0..q are valid

  for (int tile_start = 0; tile_start < num_valid_keys;
       tile_start += TILE_SIZE) {
    const int tile_end = min(tile_start + TILE_SIZE, num_valid_keys);
    const int tile_size = tile_end - tile_start;

    // =========================================================================
    // STEP 1: Cooperatively load K and V tiles to shared memory
    // =========================================================================
    // Each thread loads multiple elements to fill the tile
    // Pattern: coalesced loads, strided by warp size
    for (int i = lane_id; i < tile_size * head_dim; i += WARP_SIZE) {
      const int local_k = i / head_dim; // Which key in tile (0..tile_size-1)
      const int d = i % head_dim;       // Which dimension
      const int global_k = tile_start + local_k;

      const size_t k_offset = bh_offset + global_k * head_dim_pad + d;
      k_tile[local_k * head_dim + d] = K[k_offset];
      v_tile[local_k * head_dim + d] = V[k_offset];
    }
    __syncwarp(WARP_MASK);

    // =========================================================================
    // STEP 2: Compute Q·K scores for this tile (from SMEM)
    // =========================================================================
    float scores[TILE_SIZE]; // Register array: one score per key in tile

    // #pragma unroll
    for (int local_k = 0; local_k < TILE_SIZE; ++local_k) {
      if (local_k < tile_size) {
        float dot_partial = 0.0f;

        // Dot product: sum over head_dim, distributed across warp
        // #pragma unroll 4
        for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
          dot_partial += q_shared[d] * k_tile[local_k * head_dim + d];
        }

        // Reduce across warp and scale
        float score = warp_reduce_sum_xor(dot_partial) * scale;
        scores[local_k] = score;
      } else {
        scores[local_k] = -FLT_MAX; // Pad unused slots
      }
    }

    // =========================================================================
    // STEP 3: Online Softmax Update - Part A (Statistics)
    // =========================================================================
    // Find max score in this tile
    float tile_max = -FLT_MAX;
    // #pragma unroll
    for (int local_k = 0; local_k < tile_size; ++local_k) {
      tile_max = fmaxf(tile_max, scores[local_k]);
    }

    // Update global max
    float m_prev = m_state;
    float m_new = fmaxf(m_prev, tile_max);
    m_state = m_new;

    // Correction factor for previous tiles: exp(m_old - m_new)
    // This rescales previous output to account for new max
    float correction = expf(m_prev - m_new);

    // =========================================================================
    // STEP 4: Rescale previous output accumulator
    // =========================================================================
    // Critical numerical stability step: adjust for change in max
    // #pragma unroll
    for (int i = 0; i < n_float4_per_thread; ++i) {
      output_acc[i].x *= correction;
      output_acc[i].y *= correction;
      output_acc[i].z *= correction;
      output_acc[i].w *= correction;
    }

    // =========================================================================
    // STEP 5: Online Softmax Update - Part B (Denominator)
    // =========================================================================
    // Compute exp(score - m_new) for each score in tile
    // and accumulate to denominator
    float tile_sum_exp = 0.0f;

    // #pragma unroll
    for (int local_k = 0; local_k < tile_size; ++local_k) {
      float exp_score = expf(scores[local_k] - m_new);
      scores[local_k] = exp_score; // Overwrite with exp for reuse
      tile_sum_exp += exp_score;
    }

    // Update running denominator (with correction for previous tiles)
    d_state = d_state * correction + tile_sum_exp;

    // =========================================================================
    // STEP 6: Accumulate weighted V values for this tile
    // =========================================================================
    // Now scores[local_k] = exp(original_score - m_new)
    // We compute: output += sum_k scores[k] * V[k]
    // Final normalization by d_state happens after all tiles

    for (int i = 0; i < n_float4_per_thread; ++i) {
      const size_t d = (lane_id + i * WARP_SIZE) * 4;
      if (d < head_dim_pad) {
        float4 tile_contribution = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // #pragma unroll
        for (int local_k = 0; local_k < tile_size; ++local_k) {
          float weight = scores[local_k];

          // Load V values from shared memory (already in SMEM)
          const size_t v_idx = local_k * head_dim + d;

          // Manually load 4 consecutive floats (simulating float4)
          // Note: d is already aligned to 4
          if (d + 3 < head_dim) {
            float v0 = v_tile[v_idx + 0];
            float v1 = v_tile[v_idx + 1];
            float v2 = v_tile[v_idx + 2];
            float v3 = v_tile[v_idx + 3];

            tile_contribution.x += weight * v0;
            tile_contribution.y += weight * v1;
            tile_contribution.z += weight * v2;
            tile_contribution.w += weight * v3;
          } else {
            // Handle boundary (shouldn't happen if head_dim is multiple of 4)
            for (int j = 0; j < 4 && d + j < head_dim; ++j) {
              (&tile_contribution.x)[j] += weight * v_tile[v_idx + j];
            }
          }
        }

        // Accumulate to output
        output_acc[i].x += tile_contribution.x;
        output_acc[i].y += tile_contribution.y;
        output_acc[i].z += tile_contribution.z;
        output_acc[i].w += tile_contribution.w;
      }
    }

    __syncwarp(WARP_MASK); // Sync before loading next tile
  }

  // ===========================================================================
  // STEP 7: Final normalization and write output
  // ===========================================================================
  const float inv_d_state = 1.0f / (d_state + 1e-6f);
  const size_t output_offset = bh_offset + q * head_dim_pad;

  for (int i = 0; i < n_float4_per_thread; ++i) {
    const size_t d = (lane_id + i * WARP_SIZE) * 4;
    if (d < head_dim_pad) {
      output_acc[i].x *= inv_d_state;
      output_acc[i].y *= inv_d_state;
      output_acc[i].z *= inv_d_state;
      output_acc[i].w *= inv_d_state;

      reinterpret_cast<float4 *>(&out[output_offset + d])[0] = output_acc[i];
    }
  }
}

// ============================================================================
// Configuration & Launch
// ============================================================================

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t shared_mem_size;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  CudaConfig config;
  config.threads_per_block = dim3(WARP_SIZE, WARPS_PER_BLOCK);
  config.number_of_blocks =
      dim3(CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK), dims.batch * dims.n_heads);

  // SMEM per warp: Q + K_tile + V_tile
  // const size_t smem_per_warp =
  //     dims.head_dim * sizeof(float) +             // Q
  //     TILE_SIZE * dims.head_dim * sizeof(float) + // K tile
  //     TILE_SIZE * dims.head_dim * sizeof(float);  // V tile

  const size_t smem_per_warp =
      dims.head_dim * sizeof(float) +                // Q
      2 * TILE_SIZE * dims.head_dim * sizeof(float); // K, V tile

  config.shared_mem_size = WARPS_PER_BLOCK * smem_per_warp;
  return config;
}

size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  return 0; // Online softmax eliminates the need for storing attention weights
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);

  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock,
                         0);

  if (config.shared_mem_size > (size_t)max_shared_mem) {
    fprintf(stderr, "Error: SMEM requested (%zu) > limit (%d)\n",
            config.shared_mem_size, max_shared_mem);
    fprintf(stderr, "Try reducing TILE_SIZE (current: %d)\n", TILE_SIZE);
    return;
  }

  // Note: workspace parameter is now unused, kept for API compatibility
  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block,
                         config.shared_mem_size>>>(Q, K, V, out, dims,
                                                   dims.head_dim_padded);
}
#endif
