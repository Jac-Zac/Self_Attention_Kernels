#include "../../include/cmhsa_forward.h"

void cmhsa_forward_cpu(const float *RESTRICT Q, const float *RESTRICT K,
                       const float *RESTRICT V, float *RESTRICT out,
                       float *RESTRICT softmax_lse, float *RESTRICT softmax_max,
                       const AttentionDims dims, const float scale) {

  // Tell the compiler to not vectorize
  // TODO: Implmeent this
}
