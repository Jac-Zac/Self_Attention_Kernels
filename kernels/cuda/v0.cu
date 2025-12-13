#include "../../include/cmhsa_forward.h"

__global__ void cmhsa_forward_kernel(const float *Q, const float *K,
                                     const float *V, float *out,
                                     float *softmax_lse, float *softmax_max,
                                     size_t batch, size_t n_heads,
                                     size_t seq_len, size_t head_dim) {
  // WARNING: Not implemented yet
  (void)Q;
  (void)K;
  (void)V;
  (void)out;
  (void)softmax_lse;
  (void)softmax_max;
  (void)batch;
  (void)n_heads;
  (void)seq_len;
  (void)head_dim;
}

void cmhsa_forward_cuda(const float *RESTRICT Q, const float *RESTRICT K,
                        const float *RESTRICT V, float *RESTRICT out,
                        float *RESTRICT softmax_lse,
                        float *RESTRICT softmax_max, const AttentionDims dims) {
  // WARNING: Not implemented yet
  (void)Q;
  (void)K;
  (void)V;
  (void)out;
  (void)softmax_lse;
  (void)softmax_max;
  (void)dims;
}
