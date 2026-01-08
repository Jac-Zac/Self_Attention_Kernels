#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"
#include "../../include/utils.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <math.h>

// =============================================================================
// FlashAttention-2 Style Forward Pass (Simplified)
// =============================================================================
//
// CHANGES FROM v2 TO v3 (aligned with FlashAttention-2 paper arXiv:2307.08691):
//
// v2: 1 thread = 1 query position
// - Each thread independently processes its assigned query
// - Threads in a warp handle DIFFERENT queries
// - No collaboration within warp
//
// v3: 1 warp (32 threads) = 1 query position
// - All 32 threads COLLABORATE on the SAME query
// - Threads parallelize over head_dim (thread i handles dims i, i+32,...)
// - Uses warp shuffle for reductions
//
//
//  v2 (NOT coalesced):
//      Thread 0 reads: Q[query=0][d=0], Q[query=0][d=1], ... (sequential)
//      Thread 1 reads: Q[query=1][d=0], Q[query=1][d=1], ... (sequential)
//      -> Different threads access addresses separated by head_dim_padded
//      -> Multiple memory transactions per warp
//
//  v3 (COALESCED):
//      All threads work on SAME query q:
//      Thread 0 reads: Q[q][d=0]   → address: base + 0*sizeof(float)
//      Thread 1 reads: Q[q][d=1]   → address: base + 1*sizeof(float)
//      Thread 31 reads: Q[q][d=31] → address: base + 31*sizeof(float)
//      -> All 32 threads access CONSECUTIVE addresses
//      -> Single 128-byte memory transaction for the warp
//
//  Softmax Computation (Online softmax)
//  v3 (Online softmax, Algorithm 1):
//      Single pass over keys, maintaining running statistics:
//      - running_max: running maximum score (for numerical stability)
//      - running_sum running sum of exp(scores - m)
//      - O_tilde: running weighted sum (un-normalized)
//
//      For each key k:
//        score = Q · K[k]
//        max_new = max(running_max, score)
//        running_sum = exp(running_max - max_new) * running_sum + exp(score -
//        max_new) O_tilde = exp(running_max - max_new) * O_tilde + exp(score -
//        max_new) * V[k] running_max = max_new
//      Final: O = O_tilde / running_sum
//
//      -> Never materializes N×N attention matrix
//      -> Memory: O(1) per query (just registers)
//
// References:
// - FlashAttention-2: arXiv:2307.08691, Sections 3.1.1, 3.2, Algorithm 1
// - Online softmax: arXiv:1805.02867, arXiv:2112.05682

// =============================================================================
// Warp-Level Primitives
// =============================================================================

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffff

// ============================================================================
// v3: XOR Reduction + Multi-Warp Parallelism
// ============================================================================
// 1. XOR-based warp reduction instead of shuffle-down + broadcast
//    Why: All threads end up with the result directly
//    No need for separate warp_broadcast after reduction
//    Same number of shuffle ops, but simpler code flow
//
//    shuffle-down (v2):        XOR (v3):
//    lane 0 gets result   ->   all lanes get result
//    need broadcast       ->   no broadcast needed
// NOTE: Pherpas use this
//
// XOR-based reduction: all threads end up with the sum (no broadcast needed)
// Compare to v2's warp_reduce_sum which only gives result to lane 0
//
__inline__ __device__ float warp_reduce_sum_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FULL_WARP_MASK, val, mask);
  }
  return val;
}

// XOR-based max reduction: all threads end up with the max
__inline__ __device__ float warp_reduce_max_xor(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(FULL_WARP_MASK, val, mask));
  }
  return val;
}

// Warp reduction for sum: All threads contribute, result in lane 0
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
  }
  return val; // Only lane 0 has correct result
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  // Each step halves the active threads, comparing and keeping the max
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
  }
  return val;
}

// Broadcast from lane 0 to all lanes
__device__ __forceinline__ float warp_broadcast(float val, int src_lane = 0) {
  // Broadcast value from src_lane to all lanes in warp
  return __shfl_sync(FULL_WARP_MASK, val, src_lane);
}

// =============================================================================
// Main Kernel: FlashAttention-2 Forward Pass
// =============================================================================

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {

  // Thread/Block Mapping
  // Grid: (seq_len, batch * n_heads)
  // Block: (32, 1) = one warp per query position
  //
  // Each block handles one query position for one (batch, head) pair
  // All 32 threads in the warp collaborate on the same query

  const int q = blockIdx.x;        // Query position [0, seq_len)
  const int bh = blockIdx.y;       // Flattened (batch, head) index
  const int lane_id = threadIdx.x; // Lane within warp [0, 31]

  // Early exit for out-of-bounds
  if (q >= dims.seq_len || bh >= dims.batch * dims.n_heads)
    return;

  // Decompose batch-head index
  const int b = bh / dims.n_heads;
  const int h = bh % dims.n_heads;

  // Dimension constants
  const size_t head_dim = dims.head_dim;
  const size_t head_dim_pad = dims.head_dim_padded;
  const size_t seq_len = dims.seq_len;
  const size_t n_heads = dims.n_heads;
  const float scale = rsqrtf((float)head_dim);

  // ---------------------------------------------------------------------------
  // Compute Base Offsets
  // ---------------------------------------------------------------------------
  // Memory layout: [batch, heads, seq_len, head_dim_padded]
  const size_t bh_offset =
      b * (n_heads * seq_len * head_dim_pad) + h * (seq_len * head_dim_pad);
  const size_t query_offset = bh_offset + q * head_dim_pad;

  // ---------------------------------------------------------------------------
  // Load Query Vector into Registers (Coalesced)
  // ---------------------------------------------------------------------------
  // Each thread loads one or more elements of Q[q]
  // Pattern: Thread i loads Q[q][i], Q[q][i+32], Q[q][i+64], ...

  // FIX: I want to change this to support head dim bigger then this !!!!!
  // We support head_dim up to 128 with 4 registers per thread
  float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const size_t d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      q_reg[i] = Q[query_offset + d];
    }
  }

  // ---------------------------------------------------------------------------
  // Online Softmax Accumulators (FlashAttention-2 Algorithm 1)
  // ---------------------------------------------------------------------------
  // Per FlashAttention-2 Section 3.1.1:
  // - m: running maximum (for numerical stability)
  // - ℓ: running sum of exp(score - m)
  // - O_tilde: un-normalized output accumulator
  //
  // Key insight: We DON'T store attention weights to memory!
  // Instead, we incrementally update O_tilde as we process each key.

  float running_max = -FLT_MAX; // Running max score
  float running_exp_sum = 0.0f; // Running sum of exp(score - m)

  // Output accumulator: each thread holds part of the head_dim
  float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // ---------------------------------------------------------------------------
  // Main Loop: Iterate Over Keys (Causal: k ∈ [0, q])
  // ---------------------------------------------------------------------------
  // This is the inner loop from FlashAttention-2 Algorithm 1
  // For simplicity, we process one key at a time (tile size = 1)
  // A more optimized version would tile over multiple keys

  for (int k = 0; k <= q; k++) {
    const size_t key_offset = bh_offset + k * head_dim_pad;

    // -------------------------------------------------------------------------
    // Step 1: Compute Q·K[k] dot product (warp-parallel)
    // -------------------------------------------------------------------------
    // Each thread computes partial dot product for its dimensions
    // Then warp reduction combines partials
    //
    // COALESCED ACCESS: Same pattern as query loading
    //   Thread i reads K[key_offset + i], K[key_offset + i+32], ...

    float dot_partial = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const size_t d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        float k_val = K[key_offset + d]; // Coalesced read
        dot_partial += q_reg[i] * k_val;
      }
    }

    // Warp reduction: sum partial products across all lanes
    float score = warp_reduce_sum(dot_partial) * scale;

    // Broadcast score to all lanes (needed for online softmax update)
    score = warp_broadcast(score, 0);

    // -------------------------------------------------------------------------
    // Step 2: Online Softmax Update (FlashAttention-2 Eq. in Section 3.1.1)
    // -------------------------------------------------------------------------
    // m_new = max(m, score)
    // ℓ_new = exp(m - m_new) * ℓ + exp(score - m_new)
    // O_tilde_new = exp(m - m_new) * O_tilde + exp(score - m_new) * V[k]
    //
    // Note: We keep O_tilde UN-normalized during iteration.
    // Final normalization: O = O_tilde / ℓ  (done at the end)

    float m_new = fmaxf(running_max, score);
    float exp_diff = expf(running_max - m_new); // Rescale factor for old values
    float exp_score = expf(score - m_new);      // Weight for current V[k]

    // Update running sum of exponentials
    running_exp_sum = exp_diff * running_exp_sum + exp_score;

    // Load V[k] and update output accumulator (coalesced)
    const size_t value_offset = bh_offset + k * head_dim_pad;

#pragma unroll
    for (int i = 0; i < 4; i++) {
      const size_t d = lane_id + i * WARP_SIZE;
      if (d < head_dim) {
        float v_val = V[value_offset + d]; // Coalesced read
        // Rescale old accumulator and add new contribution
        o_reg[i] = exp_diff * o_reg[i] + exp_score * v_val;
      }
    }

    // Update running max
    running_max = m_new;
  }

  // ---------------------------------------------------------------------------
  // Final Normalization: O = O_tilde / ℓ
  // ---------------------------------------------------------------------------
  // Per FlashAttention-2 Section 3.1.1:
  // "Only at the very end of the loop do we scale the final O_tilde
  //  by diag(ℓ)^{-1} to get the right output"

  const float inv_ell = 1.0f / running_exp_sum;
  const size_t output_offset = bh_offset + q * head_dim_pad;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    const size_t d = lane_id + i * WARP_SIZE;
    if (d < head_dim) {
      out[output_offset + d] = o_reg[i] * inv_ell; // Coalesced write
    }
  }
}

// =============================================================================
// Kernel Configuration
// =============================================================================

typedef struct {
  dim3 threads_per_block;
  dim3 number_of_blocks;
  size_t total_threads;
} CudaConfig;

static CudaConfig make_cuda_config(const AttentionDims dims) {
  // One warp (32 threads) per query position
  // All threads in warp collaborate on same query
  dim3 threads_per_block(WARP_SIZE, 1);

  // Grid: one block per (query_position, batch*head) pair
  // This enables full parallelism over sequence length
  // (FlashAttention-2 Section 3.2)
  dim3 number_of_blocks(dims.seq_len, dims.batch * dims.n_heads);

  size_t total_threads = dims.seq_len * dims.batch * dims.n_heads * WARP_SIZE;

  CudaConfig config;
  config.threads_per_block = threads_per_block;
  config.number_of_blocks = number_of_blocks;
  config.total_threads = total_threads;
  return config;
}

// =============================================================================
// Public API
// =============================================================================

size_t cmhsa_get_workspace_size(const AttentionDims dims) {
  // FlashAttention-2 forward pass needs no workspace!
  // (Only backward pass needs O(N) space for logsumexp L)
  //
  // We never materialize the N×N attention matrix.
  // All intermediate values fit in registers.
  return 0;
}

__host__ void cmhsa_forward_cuda(const float *RESTRICT Q,
                                 const float *RESTRICT K,
                                 const float *RESTRICT V, float *RESTRICT out,
                                 float *RESTRICT workspace,
                                 const AttentionDims dims) {
  CudaConfig config = make_cuda_config(dims);

  VERBOSE_PRINT("CUDA Debug: Thread block (%d,%d), Grid (%d,%d)\n",
                config.threads_per_block.x, config.threads_per_block.y,
                config.number_of_blocks.x, config.number_of_blocks.y);
  VERBOSE_PRINT(
      "CUDA Debug: FlashAttention-2 style - coalesced memory access\n");

  cmhsa_forward_kernel<<<config.number_of_blocks, config.threads_per_block>>>(
      Q, K, V, out, dims);
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
