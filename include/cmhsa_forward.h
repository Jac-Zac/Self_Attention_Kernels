#pragma once
#include "utils.hpp"
#include <stddef.h>
#include <stdint.h>

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

typedef struct {
  size_t batch;    // B: batch size
  size_t n_heads;  // H: number of attention heads
  size_t seq_len;  // T: sequence length
  size_t head_dim; // C: dimension per head
} AttentionDims;

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
//   attn_weights   - Tmp scratch space [seq_len]
//   dims           - Dimension specification
//
// ============================================================================
void cmhsa_forward_cpu(
    const float *RESTRICT Q,      // [batch, n_heads, seq_len, head_dim]
    const float *RESTRICT K,      // [batch, n_heads, seq_len, head_dim]
    const float *RESTRICT V,      // [batch, n_heads, seq_len, head_dim]
    float *RESTRICT out,          // [batch, n_heads, seq_len, head_dim]
    float *RESTRICT attn_weights, // [seq_len]
    const AttentionDims dims);

#ifdef USE_CUDA

// ============================================================================
//
// CUDA Implementation of Multi-Head Self-Attention
// ============================================================================
// TODO: Update this signature to match the CPU version (use AttentionDims,
//       add softmax_lse/softmax_max buffers, use RESTRICT macro)
// ============================================================================
void cmhsa_forward_cuda(
    const float *RESTRICT Q,     // [batch, n_heads, seq_len, head_dim]
    const float *RESTRICT K,     // [batch, n_heads, seq_len, head_dim]
    const float *RESTRICT V,     // [batch, n_heads, seq_len, head_dim]
    float *RESTRICT out,         // [batch, n_heads, seq_len, head_dim]
    float *RESTRICT softmax_lse, // [batch, n_heads, seq_len]
    float *RESTRICT softmax_max, // [batch, n_heads, seq_len]
    const AttentionDims dims);

#endif
