#pragma once

// Vector padding width for AVX-512 (floats)
#ifndef VEC_PADDING
#define VEC_PADDING 16
#endif

// ============================================================================
// Alignment Invariant
// ============================================================================
//
// Memory layout: tensors are [batch, n_heads, seq_len, head_dim_pad] where
// head_dim_pad = round_up_pow2(head_dim, VEC_PADDING).
//
// Row alignment guarantee:
//   - Base pointers are 64-byte aligned (via posix_memalign with ALIGNMENT=64)
//   - Row stride = head_dim_pad * sizeof(float) bytes
//   - Since VEC_PADDING=16 and sizeof(float)=4: min stride = 16*4 = 64 bytes
//   - Therefore: every row start is 64-byte aligned
//
// This enables ASSUME_ALIGNED(row_ptr, 64) on any row pointer derived as:
//   row_ptr = base + (b*H*S + h*S + s) * head_dim_pad
//
// CRITICAL: If VEC_PADDING is reduced below 16, this invariant breaks and
// ASSUME_ALIGNED hints on row pointers become undefined behavior.
// ============================================================================

// the following macro is useful only to
// produce pragma strings in the source files
//
#define _DO_PRAGMA(x) _Pragma(#x)
//

/* =========================================================================
 * ==                          D I C T I O N A R Y                         ==
 * =========================================================================
 * VECTORIZATION
 *   - IVDEP
 *   - LOOP_VECTORIZE              defined for clang and icx, empty for gcc
 *   - LOOP_VECTOR_LENGTH(N)       defined for clang, empty for others
 *   - VECTOR_ALWAYS / VECTOR_ALIGNED / VECTOR_UNALIGNED (Intel only)
 *
 * LOOPS
 *   - LOOP_UNROLL
 *   - LOOP_UNROLL_N(n)
 *
 * ALIGNMENT
 *   - ASSUME_ALIGNED
 *   - ATTRIBUTE_ALIGNED
 * =========================================================================
 */

/* ----------------------- INTEL COMPILER -------------------------------- */
#if defined(__INTEL_LLVM_COMPILER)
#pragma message "using Intel LLVM Compiler"

#define IVDEP _Pragma("ivdep")
#define LOOP_VECTORIZE _Pragma("vector")
#define LOOP_VECTOR_LENGTH(N)
#define VECTOR_ALWAYS _Pragma("vector always")
#define VECTOR_ALIGNED _Pragma("vector aligned")
#define VECTOR_UNALIGNED _Pragma("vector unaligned")

#define LOOP_UNROLL _Pragma("unroll")
#define LOOP_UNROLL_N(N) _DO_PRAGMA(unroll N)

#define ASSUME_ALIGNED(V, A) ((__typeof__(V))__builtin_assume_aligned((V), (A)))
#define ATTRIBUTE_ALIGNED(A) __attribute__((aligned((A))))

/* ----------------------------- CLANG ----------------------------------- */
#elif defined(__clang__)

#define IVDEP _Pragma("clang ivdep")
#define LOOP_VECTORIZE _DO_PRAGMA(clang loop vectorize(enable))
#define LOOP_VECTOR_LENGTH(N) _DO_PRAGMA(clang vectorize_width(N))

#define LOOP_UNROLL _DO_PRAGMA(clang loop interleave(enable))
#define LOOP_UNROLL_N(N) _DO_PRAGMA(clang loop interleave_count(N))

#define ASSUME_ALIGNED(V, A) ((__typeof__(V))__builtin_assume_aligned((V), (A)))
#define ATTRIBUTE_ALIGNED(A) __attribute__((__aligned__((A))))

/* ----------------------------- CUDA ------------------------------------ */
#elif defined(__CUDACC__)

#define IVDEP
#define LOOP_VECTORIZE
#define LOOP_VECTOR_LENGTH(N)

#define LOOP_UNROLL _Pragma("unroll")
#define LOOP_UNROLL_N(N) _DO_PRAGMA(unroll N)

#define ASSUME_ALIGNED(V, A) (V)
#define ATTRIBUTE_ALIGNED(A) __attribute__((aligned((A))))

/* ------------------------------ GCC ------------------------------------ */
#elif defined(__GNUC__)

#define IVDEP
#define LOOP_VECTORIZE
#define LOOP_VECTOR_LENGTH(N)

#define LOOP_UNROLL
#define LOOP_UNROLL_N(N) _DO_PRAGMA(GCC unroll N)

#define ASSUME_ALIGNED(V, A) ((__typeof__(V))__builtin_assume_aligned((V), (A)))
#define ATTRIBUTE_ALIGNED(A) __attribute__((aligned((A))))

/* --------------------------- OTHER ------------------------------------- */
#elif defined(__CC_ARM)

#error "ARM compilers are not supported yet"

#else

#error "UNKNOWN COMPILER USED"

#endif

#if !defined(__INTEL_LLVM_COMPILER)
#define VECTOR_ALWAYS
#define VECTOR_ALIGNED
#define VECTOR_UNALIGNED
#endif

#define FORCE_VECTORIZE IVDEP
