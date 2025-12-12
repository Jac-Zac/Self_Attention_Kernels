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
  outputs->Q = outputs->K = outputs->V = outputs->out = NULL;
  outputs->qkv_size = outputs->stats_size = 0;
  outputs->elapsed_ns = 0;
}
