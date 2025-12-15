#pragma once

#ifdef VERBOSE
#define VERBOSE_PRINT(...) printf(__VA_ARGS__)
#else
#define VERBOSE_PRINT(...) ((void)0)
#endif

// Vectorization macros
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define FORCE_VECTORIZE _Pragma("ivdep")
#elif defined(__clang__)
#define FORCE_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define FORCE_VECTORIZE _Pragma("GCC ivdep")
#else
#define FORCE_VECTORIZE /* nothing */
#endif

#ifndef RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#elif __STDC_VERSION__ >= 199901L
#define RESTRICT restrict
#else
#define RESTRICT
#endif
#endif

// Alignment configuration and helpers
#ifndef ALIGNMENT
#define ALIGNMENT 64
#endif

// Aligned allocation for float buffers using posix_memalign
#define ALIGNED_ALLOC_FLOAT(ptr, count)                                        \
  (posix_memalign((void **)&(ptr), ALIGNMENT, sizeof(float) * (count)))

// Compiler alignment assumption hint
#if defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED_FLOAT(ptr)                                              \
  (ptr = (float *)__builtin_assume_aligned((ptr), ALIGNMENT))
#else
#define ASSUME_ALIGNED_FLOAT(ptr) ((void)0)
#endif
