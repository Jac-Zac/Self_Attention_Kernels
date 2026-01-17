#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4: Register-Based Output Accumulator + Query in Registers
// ============================================================================
// Building on v3's online softmax, this version eliminates the
// read-modify-write pattern in the inner loop by keeping the output accumulator
// in registers.
//
// Key changes from v3:
// - Output stored in registers (out_accum[4]) instead of global memory
// - Q loaded into registers once before the loop (avoids repeated global loads)
// - Only write to global memory once at the end after normalization
//
// Supported head_dim: up to 128 (4 floats per lane * 32 lanes)
//
// IMPORTANT: No bounds checks (if d < head_dim) needed because:
// 1. VEC_PADDING=32 when USE_CUDA is defined (see vector_pragmas.h)
// 2. head_dim_padded is always a multiple of WARP_SIZE (32)
// 3. Padding elements are zero-initialized, so 0*0=0 contributes nothing
// 4. This enables fully branchless, unrollable loops
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff
#define MAX_D_PER_LANE 4 // Support up to head_dim=128

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

  // Online softmax state
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f;

  // Register-based output accumulator (key optimization!)
  float out_accum[MAX_D_PER_LANE];
  // Q loaded into registers once (avoids repeated global memory access)
  float q_r[MAX_D_PER_LANE];
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out_accum[i] = 0.0f;
    q_r[i] = (d < head_dim) ? Q[q_offset + d] : 0.f;
  }

  // Loop over keys (causal: k <= q)
  for (int k = 0; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    // Q·K dot product (Q from registers)
    float dot_partial = 0.0f;
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      dot_partial += q_r[i] * K[k_offset + d];
    }

    float score = warp_reduce_sum_xor(dot_partial) * scale;

    // Online softmax update
    float new_max = fmaxf(softmax_max, score);
    float alpha = expf(softmax_max - new_max);
    float weight = expf(score - new_max);

    softmax_sum = softmax_sum * alpha + weight;

    // Update output in registers (no global memory access!)
    for (int i = 0; i < MAX_D_PER_LANE; i++) {
      const int d = lane_id + i * WARP_SIZE;
      out_accum[i] = out_accum[i] * alpha + weight * V[k_offset + d];
    }

    softmax_max = new_max;
  }

  // Normalize and write to global memory (once!)
  float inv_sum = 1.0f / softmax_sum;
  for (int i = 0; i < MAX_D_PER_LANE; i++) {
    const int d = lane_id + i * WARP_SIZE;
    out[out_offset + d] = out_accum[i] * inv_sum;
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

  cmhsa_forward_kernel<<<grid, block>>>(Q, K, V, out, dims);
}
#endif
