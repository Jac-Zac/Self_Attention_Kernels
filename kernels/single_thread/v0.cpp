#include "../../include/cmhsa_forward.h"

void cmhsa_forward_cpu(const float *__restrict__ A, const float *__restrict__ B,
                       float *__restrict__ Out, size_t N) {

  for (size_t i = 0; i < N; i++) {
    Out[i] = A[i] + B[i];
  }
}
