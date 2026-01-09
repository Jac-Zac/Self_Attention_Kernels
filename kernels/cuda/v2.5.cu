#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v2.5: Shared Memory + Loop Unrolling
// ============================================================================
// Combines optimizations from v2_smem and v2_unroll:
//
// 1. Shared Memory (from v2_smem):
//    - Cache Q vector in shared memory for efficient access
//    - Coalesced global loads, then broadcast from smem
//
// 2. XOR Reduction (from v2):
//    - All threads get reduction result directly
//    - No separate warp_broadcast needed
//
// 3. Multi-Warp (from v2):
//    - 8 warps per block (256 threads)
//    - Better GPU occupancy
//    - Allows scheduler to interleave warps
//    - Hides memory latency
//
// 4. 4x Loop Unrolling (from v2_unroll):
//    - In weighted sum step (Step 4)
//    - Each thread maintains 4 accumulators
//    - Better instruction-level parallelism
//
//    Moreover we load directly 4 floats -> iprovement over v2_unroll
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WARP_MASK 0xffffffff

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

  // Shared Memory: Each warp gets a slice of size 'head_dim' to store its Query
  extern __shared__ float smem[];
  float *RESTRICT q_shared = smem + (warp_id * head_dim);

  // Global Memory Offsets
  const size_t bh_offset = b * (dims.n_heads * seq_len * head_dim_pad) +
                           h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // STEP 0: Cache Q in Shared Memory (Coalesced Load)
  // ===========================================================================
  for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
    q_shared[d] = Q[query_offset + d];
  }
  __syncwarp(WARP_MASK); // Ensure Q is fully loaded

  // ===========================================================================
  // STEP 1: Compute Q·K scores (Q is now read from SMEM)
  // ===========================================================================
  const size_t workspace_per_bh = seq_len * (seq_len + 1) / 2;
  const size_t triangular_offset = q * (q + 1) / 2;
  float *RESTRICT aw =
      attn_weights + (bh * workspace_per_bh) + triangular_offset;

  for (int key_pos = 0; key_pos <= q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;
    float dot_partial = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += WARP_SIZE) {
      // Broadcast read from shared memory
      dot_partial += q_shared[d] * K[key_offset + d];
    }

    float score = warp_reduce_sum_xor(dot_partial) * scale;
    if (lane_id == 0)
      aw[key_pos] = score;
  }
  __syncwarp(WARP_MASK);

  // ===========================================================================
  // STEP 2 & 3: Softmax (Max + Exp Sum) (Updated to pre-multiply inv_sum_exp)
  // ===========================================================================
  float local_max = -FLT_MAX;
  for (int key_pos = lane_id; key_pos <= q; key_pos += WARP_SIZE) {
    local_max = fmaxf(local_max, aw[key_pos]);
  }
  float max_score = warp_reduce_max_xor(local_max);

  float local_sum_exp = 0.0f;
  for (int key_pos = lane_id; key_pos <= q; key_pos += WARP_SIZE) {
    float exp_val = expf(aw[key_pos] - max_score);
    aw[key_pos] = exp_val;
    local_sum_exp += exp_val;
  }
  float sum_exp = warp_reduce_sum_xor(local_sum_exp);
  const float inv_sum_exp = 1.0f / (sum_exp + 1e-6f); // Safety epsilon

  // ===========================================================================
  // STEP 4: Weighted sum of values (4x unrolled)
  // ===========================================================================
  const size_t output_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // STEP 4: Weighted sum (Vectorized + LSU Optimized)
  // ===========================================================================
  // Process 4 floats at a time per thread using float4
  for (size_t d = lane_id * 4; d < head_dim; d += WARP_SIZE * 4) {
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k <= q; ++k) {
      float w = aw[k] * inv_sum_exp;

      // Use __ldg() to hint to the compiler to use the Read-Only/Data Cache
      // This is highly effective for the V matrix in attention
      const float4 v_val = __ldg(reinterpret_cast<const float4 *>(
          &V[bh_offset + k * head_dim_pad + d]));

      acc.x += w * v_val.x;
      acc.y += w * v_val.y;
      acc.z += w * v_val.z;
      acc.w += w * v_val.w;
    }

    // Write back using a single 128-bit store
    reinterpret_cast<float4 *>(&out[output_offset + d])[0] = acc;
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

  config.shared_mem_size = WARPS_PER_BLOCK * dims.head_dim * sizeof(float);
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

  int max_shared_mem;
  cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock,
                         0);

  if (config.shared_mem_size > (size_t)max_shared_mem) {
    fprintf(stderr, "Error: SMEM requested (%zu) > limit (%d)\n",
            config.shared_mem_size, max_shared_mem);
    return;
  }

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block,
                         config.shared_mem_size>>>(Q, K, V, out, workspace,
                                                   dims, dims.head_dim_padded);
}
#endif
