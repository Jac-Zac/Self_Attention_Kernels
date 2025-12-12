#include "../../include/cmhsa_forward.h"
#include <omp.h>

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       const AttentionDims dims, const float scale) {
  // TODO: Implement OpenMP parallelized version
  (void)Q;
  (void)K;
  (void)V;
  (void)out;
  (void)dims;
  (void)scale;
}
