#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v4.2: Instruction Fusion for Softmax (exp2f + FMA)
// ============================================================================
// Building on v4.1's vectorized approach, this version reduces instruction
// count and scoreboard stalls through:
//
// 1. exp2f instead of expf:
//    - Uses identity: exp(x) = exp2(x * log2(e))
//    - exp2f is faster as it maps directly to hardware (single MUFU)
//    - expf internally uses exp2f anyway, but with extra overhead
//
// 2. Fused scale into exp argument:
//    - Instead of: score = dot * scale; weight = exp(score - max)
//    - We use: weight = exp2((score - max) * scale_log2e)
//    - This eliminates a separate multiply instruction
//
// 3. Explicit fmaf() for FMA fusion:
//    - Guarantees fused multiply-add instead of separate MUL + ADD
//    - Reduces instruction count and improves numerical accuracy
//
// Expected improvements:
//    - ~50% reduction in FP32 instructions for softmax
//    - Reduced short_scoreboard stalls (fewer instruction dependencies)
//    - Better instruction-level parallelism
//
// Supported head_dim: up to 128 (32 lanes * 4 floats)
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define WARP_MASK 0xffffffff

// log2(e) for converting exp(x) to exp2(x * log2e)
#define LOG2E 1.4426950408889634f

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

  // Pre-compute scale with log2(e) for fused exp2f argument
  // This combines: rsqrt(head_dim) * log2(e)
  const float scale_log2e = rsqrtf((float)head_dim) * LOG2E;

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  const size_t q_offset = bh_offset + q * head_dim_pad;

  // Online softmax state
  // Note: running_max is NOT pre-scaled; we scale in the exp2f argument
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  // Vectorized Load Q into registers (assuming head_dim <= 128)
  float4 vec_q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (lane_id * 4 < head_dim) {
    vec_q = reinterpret_cast<const float4 *>(Q + q_offset)[lane_id];
  }

  // Output Accumulator as float4
  float4 out_accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  // Loop over keys (causal: k <= q)
  for (int k = 0; k <= q; ++k) {
    const size_t k_offset = bh_offset + k * head_dim_pad;

    // 1. Vectorized Load K
    float4 k_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (lane_id * 4 < head_dim) {
      k_vec = __ldg(reinterpret_cast<const float4 *>(K + k_offset) + lane_id);
    }

    // 2. Q·K dot product - DON'T scale here, defer to exp2f argument
    float dot = (vec_q.x * k_vec.x) + (vec_q.y * k_vec.y) +
                (vec_q.z * k_vec.z) + (vec_q.w * k_vec.w);
    float score = warp_reduce_sum_xor(dot);

    // 3. Online softmax with exp2f fusion
    // Instead of: alpha = exp((old_max - new_max) * scale)
    // We use:     alpha = exp2((old_max - new_max) * scale * log2e)
    //           = exp2((old_max - new_max) * scale_log2e)
    float new_max = fmaxf(running_max, score);

    // Compute scaling factors using exp2f with fused scale
    // This saves instructions by combining scale into the exp argument
    float alpha = exp2f((running_max - new_max) * scale_log2e);
    float weight = exp2f((score - new_max) * scale_log2e);

    // Update running sum using fmaf for guaranteed FMA
    running_sum = fmaf(running_sum, alpha, weight);

    // 4. Load V and update accumulator using fmaf
    if (lane_id * 4 < head_dim) {
      float4 vec_v =
          __ldg(reinterpret_cast<const float4 *>(V + k_offset) + lane_id);

      // Use fmaf for fused multiply-add: acc = acc * alpha + weight * v
      // This is equivalent to: acc * alpha + weight * v
      // Using fmaf(acc, alpha, weight * v) ensures the multiply-add is fused
      out_accum.x = fmaf(out_accum.x, alpha, weight * vec_v.x);
      out_accum.y = fmaf(out_accum.y, alpha, weight * vec_v.y);
      out_accum.z = fmaf(out_accum.z, alpha, weight * vec_v.z);
      out_accum.w = fmaf(out_accum.w, alpha, weight * vec_v.w);
    }

    running_max = new_max;
  }

  // Final Normalization and Vectorized Store
  float inv_sum = 1.0f / running_sum;
  if (lane_id * 4 < head_dim) {
    float4 result = make_float4(out_accum.x * inv_sum, out_accum.y * inv_sum,
                                out_accum.z * inv_sum, out_accum.w * inv_sum);
    reinterpret_cast<float4 *>(out + q_offset)[lane_id] = result;
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
