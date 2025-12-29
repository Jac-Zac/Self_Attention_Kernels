#ifdef USE_CUDA
#include "../../include/cmhsa_forward.h"

__global__ void cmhsa_forward_kernel(const float *RESTRICT Q,
                                     const float *RESTRICT K,
                                     const float *RESTRICT V,
                                     float *RESTRICT out,
                                     const AttentionDims dims) {
  // WARNING: Not implemented yet
  (void)Q;
  (void)K;
  (void)V;
  (void)out;
  (void)dims;
}

void cmhsa_forward_cuda(const float *RESTRICT Q, const float *RESTRICT K,
                        const float *RESTRICT V, float *RESTRICT out,
                        const AttentionDims dims) {
  // WARNING: Not implemented yet - just launch empty kernel
  cmhsa_forward_kernel<<<1, 1>>>(Q, K, V, out, dims);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}
#else
#error "This file requires USE_CUDA to be defined"
#endif
