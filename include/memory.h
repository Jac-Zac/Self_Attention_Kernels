#pragma once
// Memory alignment utilities: RESTRICT, ALIGNMENT, and aligned allocation
// macros.

#include "vector_pragmas.h"
#include <stddef.h>
#include <stdlib.h>

// Round up x to the next multiple of a (a must be power of 2).
static inline size_t round_up_pow2(size_t x, size_t a) {
  return (x + a - 1) & ~(a - 1);
}

// RESTRICT macro for pointer aliasing hints
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

#ifndef ALIGNMENT
#define ALIGNMENT 64 // AVX-512 alignment
#endif

#define ALIGNED_ALLOC_FLOAT(ptr, count)                                        \
  (posix_memalign((void **)&(ptr), ALIGNMENT, sizeof(float) * (count)))

#define ASSUME_ALIGNED_FLOAT(ptr)                                              \
  (ptr = (float *)ASSUME_ALIGNED((ptr), ALIGNMENT))
