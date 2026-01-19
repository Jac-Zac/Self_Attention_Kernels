#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v8: Split-K Parallelism (Two Warps per Query)
// ============================================================================
// This version uses two warps to process each query in parallel, splitting
// the key range between them and merging results.
//
// Key insight: For long sequences, a single warp processing all keys serially
// is slow. By splitting the key range across two warps, we can:
// 1. Double the compute throughput per query
// 2. Better utilize memory bandwidth (two warps loading different K/V)
//
// Block structure:
//   - 8 warps = 4 query pairs
//   - Warp 0,1 handle query 0 (warp 0: even keys, warp 1: odd keys)
//   - Warp 2,3 handle query 1
//   - etc.
//
// Algorithm:
//   1. Each warp computes partial attention for its subset of keys
//   2. Merge partial results using online softmax formula:
//      combined_max = max(max_a, max_b)
//      combined_sum = sum_a * exp(max_a - combined_max) + sum_b * exp(max_b - combined_max)
//      combined_out = (out_a * sum_a * exp(max_a - combined_max) + 
//                      out_b * sum_b * exp(max_b - combined_max)) / combined_sum
//
// Shared memory for merge:
//   - partial_max[4]   - max per query pair
//   - partial_sum[4]   - sum per query pair
//   - partial_out[4][head_dim] - output per query pair
//
// ============================================================================

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define QUERIES_PER_BLOCK 4  // 2 warps per query
#define WARPS_PER_QUERY 2
#define WARP_MASK 0xffffffff

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

  // Which query this warp contributes to
  const int local_query = warp_id / WARPS_PER_QUERY;  // 0-3
  const int warp_in_query = warp_id % WARPS_PER_QUERY; // 0 or 1

  const int q_base = blockIdx.x * QUERIES_PER_BLOCK;
  const int q_idx = q_base + local_query;
  const int bh = blockIdx.y;

  // No early return - all threads must hit __syncthreads()
  const bool warp_active =
      (q_idx < (int)dims.seq_len) && (bh < (int)(dims.batch * dims.n_heads));

  const int head_dim = (int)dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const float scale = rsqrtf((float)head_dim);

  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  const size_t bh_offset = b * (dims.n_heads * dims.seq_len * head_dim_pad) +
                           h * (dims.seq_len * head_dim_pad);

  // Shared memory for merge
  // Layout: [query][warp_in_query] for max/sum, [query][warp_in_query][head_dim] for output
  extern __shared__ float smem[];
  float *partial_max = smem;                                    // [4][2]
  float *partial_sum = smem + QUERIES_PER_BLOCK * 2;            // [4][2]
  float *partial_out = smem + QUERIES_PER_BLOCK * 4;            // [4][2][head_dim_pad]

  const int d_base = lane_id * 4;
  const bool lane_active = (d_base < head_dim);

  // Load Q
  float4 q_vec = make_float4(0.f, 0.f, 0.f, 0.f);
  if (warp_active && lane_active) {
    q_vec = __ldg(reinterpret_cast<const float4 *>(
        &Q[bh_offset + q_idx * head_dim_pad + d_base]));
  }

  // Online softmax state for this warp's partial computation
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float4 out_acc = make_float4(0.f, 0.f, 0.f, 0.f);

  if (warp_active) {
    const int num_keys = q_idx + 1;

    // Split keys: warp 0 takes even indices, warp 1 takes odd
    for (int k = warp_in_query; k < num_keys; k += WARPS_PER_QUERY) {
      const size_t kv_base = bh_offset + k * head_dim_pad + d_base;

      float4 k_vec = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 v_vec = make_float4(0.f, 0.f, 0.f, 0.f);
      if (lane_active) {
        k_vec = __ldg(reinterpret_cast<const float4 *>(&K[kv_base]));
        v_vec = __ldg(reinterpret_cast<const float4 *>(&V[kv_base]));
      }

      float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                  q_vec.z * k_vec.z + q_vec.w * k_vec.w;
      float score = warp_reduce_sum_xor(dot) * scale;

      float new_max = fmaxf(running_max, score);
      float alpha = expf(running_max - new_max);
      float weight = expf(score - new_max);

      running_sum = running_sum * alpha + weight;
      out_acc.x = out_acc.x * alpha + weight * v_vec.x;
      out_acc.y = out_acc.y * alpha + weight * v_vec.y;
      out_acc.z = out_acc.z * alpha + weight * v_vec.z;
      out_acc.w = out_acc.w * alpha + weight * v_vec.w;
      running_max = new_max;
    }
  }

  // Store partial results to shared memory for merge
  if (lane_id == 0) {
    partial_max[local_query * 2 + warp_in_query] = running_max;
    partial_sum[local_query * 2 + warp_in_query] = running_sum;
  }
  if (lane_active) {
    float *out_ptr = &partial_out[(local_query * 2 + warp_in_query) * head_dim_pad + d_base];
    *reinterpret_cast<float4 *>(out_ptr) = out_acc;
  }

  __syncthreads();

  // =========================================================================
  // Merge phase: warp 0 of each query pair merges results
  // =========================================================================
  if (warp_in_query == 0 && warp_active) {
    // Load partial results from both warps
    float max_a = partial_max[local_query * 2 + 0];
    float max_b = partial_max[local_query * 2 + 1];
    float sum_a = partial_sum[local_query * 2 + 0];
    float sum_b = partial_sum[local_query * 2 + 1];

    float4 out_a = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 out_b = make_float4(0.f, 0.f, 0.f, 0.f);
    if (lane_active) {
      out_a = *reinterpret_cast<float4 *>(&partial_out[(local_query * 2 + 0) * head_dim_pad + d_base]);
      out_b = *reinterpret_cast<float4 *>(&partial_out[(local_query * 2 + 1) * head_dim_pad + d_base]);
    }

    // Merge using online softmax formula
    float combined_max = fmaxf(max_a, max_b);
    float scale_a = expf(max_a - combined_max);
    float scale_b = expf(max_b - combined_max);
    float combined_sum = sum_a * scale_a + sum_b * scale_b;

    float inv_sum = 1.0f / combined_sum;
    float w_a = sum_a * scale_a * inv_sum;
    float w_b = sum_b * scale_b * inv_sum;

    float4 result;
    result.x = out_a.x * w_a + out_b.x * w_b;
    result.y = out_a.y * w_a + out_b.y * w_b;
    result.z = out_a.z * w_a + out_b.z * w_b;
    result.w = out_a.w * w_a + out_b.w * w_b;

    // Write final output
    if (lane_active) {
      *reinterpret_cast<float4 *>(
          &out[bh_offset + q_idx * head_dim_pad + d_base]) = result;
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
  dim3 grid(CEIL_DIV(dims.seq_len, QUERIES_PER_BLOCK), dims.batch * dims.n_heads);

  // Shared memory: partial_max[4][2] + partial_sum[4][2] + partial_out[4][2][head_dim_pad]
  size_t smem_size = QUERIES_PER_BLOCK * 2 * sizeof(float) +  // max
                     QUERIES_PER_BLOCK * 2 * sizeof(float) +  // sum
                     QUERIES_PER_BLOCK * 2 * dims.head_dim_padded * sizeof(float); // out

  cmhsa_forward_kernel<<<grid, block, smem_size>>>(Q, K, V, out, dims);
}
#endif
