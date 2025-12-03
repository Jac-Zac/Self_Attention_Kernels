#pragma once

#ifdef DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
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
