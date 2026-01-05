#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// v3: Online Softmax (single-pass attention)
// ============================================================================
//
// Changes from v2:
// 1. Online softmax - compute softmax incrementally as we iterate over keys
//    Why: No need to store attention weights, zero workspace required
//    Single pass over keys instead of two passes (score, max, normalize)
//
// 2. XOR-based warp reduction - all threads get the result directly
//    Why: No need for separate warp_broadcast after reduction
//
// Online softmax algorithm:
//   For each new score, we:
//   1. Update the running max: new_max = max(old_max, score)
//   2. Compute rescale factor: rescale = exp(old_max - new_max)
//   3. Rescale previous accumulations: output *= rescale, sum *= rescale
//   4. Add new contribution: weight = exp(score - new_max)
//   This is numerically stable and requires only O(1) extra state.
//
// ============================================================================

// Full warp mask for synchronization
#define WARP_MASK 0xffffffff

// XOR-based reduction: all threads end up with the sum (no broadcast needed)
__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(WARP_MASK, val, mask);
  }
  return val;
}

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  // 2D parallelization (same as v2):
  // x dimension: query position (one block per query)
  // y dimension: (batch, head) pairs
  const int bh = blockIdx.y * blockDim.y + threadIdx.y;
  const int q = blockIdx.x;        // Direct mapping: one block per query
  const int lane_id = threadIdx.x; // Warp lane ID (0-31)

  if (q >= dims.seq_len || bh >= dims.batch * dims.n_heads)
    return;

  const size_t num_heads = dims.n_heads;
  const size_t seq_len = dims.seq_len;
  const size_t head_dim = dims.head_dim;
  const float scale = rsqrtf((float)head_dim);
  const size_t head_dim_pad = dims.head_dim_padded;

  // Decompose linear (batch, head) index
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Tensor offsets: [batch, head, seq, head_dim]
  const size_t bh_offset =
      b * (num_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;
  const size_t output_offset = bh_offset + q * head_dim_pad;

  // ===========================================================================
  // Initialize output to zero
  // ===========================================================================
  // Each thread handles strided dimensions
  for (size_t d = lane_id; d < head_dim; d += warpSize) {
    out[output_offset + d] = 0.0f;
  }

  // Online softmax state (replaces workspace from v2)
  // Running maximum score (for numerical stability)
  float softmax_max = -FLT_MAX;
  float softmax_sum = 0.0f; // Running sum of exp(score - max)

  // ===========================================================================
  // Single-pass attention loop (online softmax)
  // ===========================================================================
  // Here we fuse everything into a single pass.
  for (size_t key_pos = 0; key_pos <= (size_t)q; key_pos++) {
    const size_t key_offset = bh_offset + key_pos * head_dim_pad;
    float dot_partial = 0.0f;
    for (size_t d = lane_id; d < head_dim; d += warpSize) {
      dot_partial += Q[query_offset + d] * K[key_offset + d];
    }

    // XOR reduction: all threads get the final score (no broadcast needed)
    float score = warp_reduce_sum_xor(dot_partial) * scale;

    // Online softmax update
    // When we see a new score, we need to:
    // 1. Update the running max
    // 2. Rescale all previous accumulations by exp(old_max - new_max)
    // 3. Add the new contribution with weight exp(score - new_max)
    float new_max = fmaxf(softmax_max, score);

    // Rescale factor: if new score is larger, we shrink previous values
    // If new_max == softmax_max, rescale = 1.0 (no change)
    // If new_max > softmax_max, rescale < 1.0 (shrink previous)
    float rescale = expf(softmax_max - new_max);

    // This key's attention weight (unnormalized)
    float attn_weight = expf(score - new_max);

    // Update running sum: rescale old sum and add new weight
    softmax_sum = softmax_sum * rescale + attn_weight;

    // Update max for next iteration
    softmax_max = new_max;

    // Rescale previous output and accumulate new V contribution
    // Each thread handles its own strided subset of dimensions.
    // No syncwarp needed: thread i only touches dims {i, i+32, i+64, ...}
    const size_t value_offset = bh_offset + key_pos * head_dim_pad;
    for (size_t d = lane_id; d < head_dim; d += warpSize) {
      out[output_offset + d] =
          out[output_offset + d] * rescale + attn_weight * V[value_offset + d];
    }
  }

  // ===========================================================================
  // Final normalization
  // ===========================================================================
  // Divide by softmax sum to get proper attention weights
  const float inv_sum = 1.0f / softmax_sum;
  for (size_t d = lane_id; d < head_dim; d += warpSize) {
    out[output_offset + d] *= inv_sum;
  }
}

// ============================================================================
// Kernel Configuration
// ============================================================================
#define THREADS_PER_BLOCK_X 32 // One full warp per block
#define THREADS_PER_BLOCK_Y 1

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

  // 2D grid: x = seq_len (one block per query), y = batch * n_heads
  size_t blocks_x = dims.seq_len;
  size_t blocks_y = CEIL_DIV(dims.batch * dims.n_heads, THREADS_PER_BLOCK_Y);

  dim3 number_of_blocks(blocks_x, blocks_y);

  size_t total_threads =
      (blocks_x * blocks_y) * (threads_per_block.x * threads_per_block.y);

  CudaConfig config;
  config.threads_per_block = threads_per_block;
  config.number_of_blocks = number_of_blocks;
  config.total_threads = total_threads;
  return config;
}

// ============================================================================
// Public API
// ============================================================================

// Online softmax doesn't need workspace
size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  (void)dims;
  return 0;
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  (void)workspace; // Unused - online softmax needs no workspace

  CudaConfig config = make_cuda_config(dims);

  VERBOSE_PRINT("CUDA v3: Thread block (%d,%d), Grid (%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.number_of_blocks.x, config.number_of_blocks.y);

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, dims);
}

#else
#error "This file requires USE_CUDA to be defined"
#endif
