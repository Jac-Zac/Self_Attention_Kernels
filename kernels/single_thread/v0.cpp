#include "../../include/cmhsa_forward.h"

void cmhsa_forward_cpu(const float *__restrict__ A, const float *__restrict__ B,
                       float *__restrict__ Out, size_t N) {

// Tell the compiler to not vectorize
#pragma clang loop vectorize(disable)
#pragma GCC loop vectorize(disable)
  for (size_t i = 0; i < N; i++) {
    Out[i] = A[i] + B[i];
  }
}
