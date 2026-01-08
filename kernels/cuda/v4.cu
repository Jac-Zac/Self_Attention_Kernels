#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v3: Shared Memory Workspace for Coalesced Softmax
// ============================================================================
// This version keeps the XOR reduction and Multi-Warp structure of v2 but
// moves the attention weights into Shared Memory (SRAM) for Steps 2 & 3.
//
// Why: Global memory triangular indexing (q*(q+1)/2) is rarely 128-byte
// aligned. By using SRAM, we eliminate the 31.5/32 byte uncoalesced penalty and
// reduce the "Mem Pipes Busy" congestion.
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff

// Limit based on shared memory capacity (e.g., 1024 floats * 8 warps * 4 bytes
// = 32KB)
#define MAX_SRAM_SEQ 1024

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
                     float *RESTRICT attn_weights, const AttentionDims dims,
                     const size_t head_dim_pad) {

  // Each warp gets a dedicated row in shared memory
  __shared__ float s_aw[WARPS_PER_BLOCK][MAX_SRAM_SEQ];

  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  const int query_group = blockIdx.x;
  const int q = query_group * WARPS_PER_BLOCK + warp_id;
  const int bh = blockIdx.y;

  if (q >= (int)dims.seq_len || bh >= (int)(dims.batch * dims.n_heads))
    return;

  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Global Workspace Offsets (Used only for final storage if needed)
  const size_t workspace_per_bh = seq_len * (seq_len + 1) / 2;
  const size_t bh_workspace_offset = (b * num_heads + h) * workspace_per_bh;
  const size_t triangular_offset = q * (q + 1) / 2;
  float *RESTRICT aw_global =
      attn_weights + bh_workspace_offset + triangular_offset;

  const size_t bh_offset =
      b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // STEP 1: Compute Q·K scores
  // ===========================================================================
  for (int key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;
    float dot_partial = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      dot_partial += Q[query_offset + d] * K[key_offset + d];
    }

    float score = warp_reduce_sum_xor(dot_partial) * scale;

    if (lane_id == 0) {
      // Store in Shared Memory instead of Global to fix alignment issues later
      if (key_pos < MAX_SRAM_SEQ)
        s_aw[warp_id][key_pos] = score;
      // Optional: still write to global if the workspace is needed elsewhere
      aw_global[key_pos] = score;
    }
  }
  __syncwarp(WARP_MASK);

  // ===========================================================================
  // STEP 2: Parallel max-finding (Now in SRAM)
  // ===========================================================================
  float local_max = -FLT_MAX;
  for (int key_pos = lane_id; key_pos <= q && key_pos < MAX_SRAM_SEQ;
       key_pos += WARP_SIZE) {
    local_max = fmaxf(local_max, s_aw[warp_id][key_pos]);
  }
  float max_score = warp_reduce_max_xor(local_max);

  // ===========================================================================
  // STEP 3: Parallel softmax computation (Now in SRAM)
  // ===========================================================================
  float local_sum_exp = 0.0f;
  for (int key_pos = lane_id; key_pos <= q && key_pos < MAX_SRAM_SEQ;
       key_pos += WARP_SIZE) {
    float exp_val = expf(s_aw[warp_id][key_pos] - max_score);
    s_aw[warp_id][key_pos] = exp_val;
    local_sum_exp += exp_val;
  }
  __syncwarp(WARP_MASK);

  float sum_exp = warp_reduce_sum_xor(local_sum_exp);
  const float inv_sum_exp = 1.0f / sum_exp;

  // ===========================================================================
  // STEP 4: Weighted sum of values
  // ===========================================================================
  const size_t output_offset = bh_offset + q * head_dim_pad;
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
    out[output_offset + d] = 0.0f;
  }

  for (int key_pos = 0; key_pos <= q; key_pos++) {
    const size_t value_offset = bh_offset + key_pos * head_dim_pad;
    // Load weight from Shared Memory (Fast, aligned)
    float const normalized_weight = s_aw[warp_id][key_pos] * inv_sum_exp;

    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      out[output_offset + d] += normalized_weight * V[value_offset + d];
    }
  }
}

// ============================================================================
// Kernel Configuration
// ============================================================================

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(WARP_SIZE, WARPS_PER_BLOCK);
  size_t query_groups = CEIL_DIV(dims.seq_len, WARPS_PER_BLOCK);
  dim3 number_of_blocks(query_groups, dims.batch * dims.n_heads);

  // Check SRAM limits as you requested
  int max_sram_per_block;
  cudaDeviceGetAttribute(&max_sram_per_block,
                         cudaDevAttrMaxSharedMemoryPerBlock, 0);
  size_t requested_sram = WARPS_PER_BLOCK * MAX_SRAM_SEQ * sizeof(float);

  if (requested_sram > (size_t)max_sram_per_block) {
    // In a PhD project, you'd handle this with dynamic allocation or tiling
    printf("Warning: Requested SRAM %zu exceeds device max %d\n",
           requested_sram, max_sram_per_block);
  }

  CudaConfig config;
  config.threads_per_block = threads_per_block;
  config.number_of_blocks = number_of_blocks;
  return config;
}

size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  const size_t workspace_per_bh = dims.seq_len * (dims.seq_len + 1) / 2;
  return dims.batch * dims.n_heads * workspace_per_bh * sizeof(float);
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);
  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, workspace, dims, dims.head_dim_padded);
}

#endif
