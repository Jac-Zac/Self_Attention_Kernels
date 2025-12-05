#include "../../include/cmhsa_forward.h"
#include <omp.h>

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT softmax_lse, float *RESTRICT softmax_max,
                       const AttentionDims dims, const float scale) {
  // TODO: Implement OpenMP parallelized version
  (void)Q;
  (void)K;
  (void)V;
  (void)out;
  (void)softmax_lse;
  (void)softmax_max;
  (void)dims;
  (void)scale;
}
