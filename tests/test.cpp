#include "../include/cmhsa_forward.h"
#include "reference.hpp"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  const int n = 5;
  float *a = (float *)malloc(sizeof(float) * n);
  float *b = (float *)malloc(sizeof(float) * n);
  float *c = (float *)malloc(sizeof(float) * n);
  float *expected = (float *)malloc(sizeof(float) * n);

  // Initialize
  for (int i = 0; i < n; i++) {
    a[i] = i * 1.0f;
    b[i] = i * 2.0f;
  }

  // Call the function
  cmhsa_reference(a, b, expected, n);

  // Call the function to test
  cmhsa_forward_cpu(a, b, c, n);

  // Check correctness
  for (int i = 0; i < n; i++) {
    assert(c[i] == expected[i]);
  }

  printf("check: PASS\n");

  free(a);
  free(b);
  free(c);
  free(expected);

  return 0;
}
