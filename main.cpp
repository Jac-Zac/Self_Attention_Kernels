#include "include/cmhsa_forward.h"
#include "include/macros.hpp"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  const int n = 5;
  float *a = (float *)malloc(sizeof(float) * n);
  float *b = (float *)malloc(sizeof(float) * n);
  float *c = (float *)malloc(sizeof(float) * n);

  // Initialize with simple initialization strategy
  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i;
  }

  for (int i = 0; i < n; i++) {
    DEBUG_PRINT("Value of a[%d] = %f\n", i, a[i]);
    DEBUG_PRINT("Value of b[%d] = %f\n", i, b[i]);
  }

  // Call the cmhsa kernel for CPU
  cmhsa_forward_cpu(a, b, c, n);

  printf("Sum results:\n");
  for (int i = 0; i < n; i++) {
    printf("c[%d] = %f\n", i, c[i]);
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
