#pragma once

#include <stddef.h>

inline void cmhsa_reference(const float *A, const float *B, float *Out,
                            size_t N) {
  for (size_t i = 0; i < N; i++) {
    Out[i] = A[i] + B[i];
  }
}
