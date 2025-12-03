#pragma once

#include "macros.hpp"
#include <stddef.h>

void cmhsa_forward_cpu(const float *__restrict__ A, const float *__restrict__ B,
                       float *__restrict__ Out, size_t N);

#ifdef USE_CUDA
void cmhsa_forward_cuda(const float *__restrict__ A,
                        const float *__restrict__ B, float *__restrict__ Out,
                        size_t N);
#endif
