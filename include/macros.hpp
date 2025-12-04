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
