#pragma once
#include "utils.hpp"
#include <stddef.h>
#include <stdint.h>

// Logical-to-padded dimension helpers
// These functions compute padded dimensions
static inline size_t pad_head_dim(size_t head_dim) {
  return round_up_pow2(head_dim, VEC_PADDING);
}
static inline size_t pad_seq_len(size_t seq_len) {
  return round_up_pow2(seq_len, VEC_PADDING);
}

typedef struct {
  size_t batch;           // B: batch size
  size_t n_heads;         // H: number of attention heads
  size_t seq_len;         // T: sequence length (logical, unpadded)
  size_t head_dim;        // C: dimension per head (logical, unpadded)
  size_t seq_len_padded;  // Padded sequence length (for alignment/SIMD)
  size_t head_dim_padded; // Padded head dimension (for alignment/SIMD)
} AttentionDims;

// Helper to construct an AttentionDims instance with padding calculated
static inline AttentionDims make_attention_dims(size_t batch, size_t n_heads,
                                                size_t seq_len,
                                                size_t head_dim) {
  AttentionDims dims;
  dims.batch = batch;
  dims.n_heads = n_heads;
  dims.seq_len = seq_len;
  dims.head_dim = head_dim;
  dims.seq_len_padded = pad_seq_len(seq_len);
  dims.head_dim_padded = pad_head_dim(head_dim);
  return dims;
}

// ============================================================================
// Multi-Head Self-Attention Forward Pass
// ============================================================================
//
// This implements the standard self-attention mechanism:
//   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// All input tensors (Q, K, V) are outputs from linear projection layers
// applied to the same input sequence, hence "self-attention".
//
// Dimensions:
//   B = batch size
//   H = number of attention heads
//   T = sequence length (same for Q, K, V in self-attention)
//   C = head dimension (embedding_dim / num_heads)
//   D = embedding dimension (H * C)
//
// Memory layout: All tensors are row-major, contiguous in memory.
// ============================================================================

// ============================================================================
// CPU Implementation of Multi-Head Self-Attention
// ============================================================================
//
// Computes: out = MultiHeadAttention(Q, K, V)
//
// Algorithm:
//   For each head h:
//     1. scores = (Q_h @ K_h^T) * scale
//     2. probs = softmax(scores)  [numerically stable via softmax_max/lse]
//     3. out_h = probs @ V_h
//
// Parameters:
//   Q              - Query tensor [batch, n_heads, seq_len, head_dim]
//   K              - Key tensor [batch, n_heads, seq_len, head_dim]
//   V              - Value tensor [batch, n_heads, seq_len, head_dim]
//   out            - Output tensor [batch, n_heads, seq_len, head_dim]
//   attn_weights   - Workspace base, sized at least
//                     threads*pad_seq_len(seq_len) floats (multi-thread)
//                     or 1*pad_seq_len(seq_len) (single-thread)
//   dims           - Dimension specification
//
// Note: Internal row strides use pad_head_dim(head_dim). Logical math loops
//       still iterate over head_dim and seq_len (or head_dim_pad for full
//       SIMD).
// ============================================================================
void cmhsa_forward_cpu(
    const float *RESTRICT Q,      // [batch, n_heads, seq_len, head_dim]
    const float *RESTRICT K,      // [batch, n_heads, seq_len, head_dim]
    const float *RESTRICT V,      // [batch, n_heads, seq_len, head_dim]
    float *RESTRICT out,          // [batch, n_heads, seq_len, head_dim]
    float *RESTRICT attn_weights, // workspace base
    const AttentionDims dims);

#ifdef USE_CUDA

// ============================================================================
// CUDA Implementation of Multi-Head Self-Attention
// ============================================================================

// Returns the workspace size in bytes required for the CUDA kernel.
// Caller should allocate this much memory and pass it to cmhsa_forward_cuda.
size_t cmhsa_get_workspace_size(const AttentionDims dims);

// CUDA forward pass - computes thread/block config internally
void cmhsa_forward_cuda(const float *RESTRICT Q, const float *RESTRICT K,
                        const float *RESTRICT V, float *RESTRICT out,
                        float *RESTRICT workspace, const AttentionDims dims);

#endif
