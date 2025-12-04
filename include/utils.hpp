#pragma once
#include <stddef.h>
#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Shared outputs from attention run
struct Outputs {
  float *Q;
  float *K;
  float *V;
  float *out;
  float *softmax_lse;
  float *softmax_max;
  size_t qkv_size;
  size_t stats_size;
  uint64_t elapsed_ns;
};

inline void free_outputs(struct Outputs *outputs) {
  if (!outputs)
    return;
  free(outputs->Q);
  free(outputs->K);
  free(outputs->V);
  free(outputs->out);
  free(outputs->softmax_lse);
  free(outputs->softmax_max);
  outputs->Q = outputs->K = outputs->V = outputs->out = outputs->softmax_lse =
      outputs->softmax_max = NULL;
  outputs->qkv_size = outputs->stats_size = 0;
  outputs->elapsed_ns = 0;
}
