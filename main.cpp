#include "include/cmhsa_forward.h"
#include "include/macros.hpp"
#include "include/timing.h"
#include <stdio.h>
#include <stdlib.h>

// Default values that are resolved when using make ...
#ifndef BACKEND
#define BACKEND "unknown"
#endif
#ifndef VERSION_STR
#define VERSION_STR "v0"
#endif

int main(int argc, char *argv[]) {
  // Simple CLI: optional positional N, default 1024
  size_t n = 1024ULL;
  if (argc > 1) {
    char *end = NULL;
    unsigned long long v = strtoull(argv[1], &end, 10);
    if (!end || *end != '\0' || v == 0ULL) {
      fprintf(stderr, "Error: invalid N; pass a positive integer\n");
      return 1;
    }
    n = (size_t)v;
  }

  printf("backend=%s version=%s n=%zu\n", BACKEND, VERSION_STR, n);

  float *a = (float *)malloc(sizeof(float) * n);
  float *b = (float *)malloc(sizeof(float) * n);
  float *c = (float *)malloc(sizeof(float) * n);

  if (!a || !b || !c) {
    fprintf(stderr, "Error: memory allocation failed\n");
    free(a);
    free(b);
    free(c);
    return 1;
  }

  // Initialize with simple initialization strategy
  for (size_t i = 0; i < n; i++) {
    a[i] = (float)i;
    b[i] = (float)i;
  }

  for (size_t i = 0; i < n; i++) {
    DEBUG_PRINT("Value of a[%zu] = %f\n", i, a[i]);
    DEBUG_PRINT("Value of b[%zu] = %f\n", i, b[i]);
  }

  // Call the cmhsa kernel for CPU with wall-clock timing
  struct timespec start, end;
  NOW(start);
  cmhsa_forward_cpu(a, b, c, n);
  NOW(end);

  print_timing("CPU", ns_diff(start, end));

  DEBUG_PRINT("Sum results:\n");
  for (size_t i = 0; i < n; i++) {
    DEBUG_PRINT("c[%zu] = %f\n", i, c[i]);
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
