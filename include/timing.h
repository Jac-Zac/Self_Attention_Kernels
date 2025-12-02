#pragma once
#include <stdio.h>
#include <time.h>

// --- Basic time getters (double seconds) ---
#define TIME_S()                                                               \
  ({                                                                           \
    struct timespec ts;                                                        \
    clock_gettime(CLOCK_REALTIME, &ts);                                        \
    (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;                             \
  })

#define TIME_THREAD_S()                                                        \
  ({                                                                           \
    struct timespec ts;                                                        \
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);                               \
    (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;                             \
  })

// --- Wall clock timestamp (ns‐resolution) ---
#define NOW(tsvar) clock_gettime(CLOCK_MONOTONIC, &(tsvar))

// --- ns difference ---
static inline unsigned long long ns_diff(struct timespec a, struct timespec b) {
  long long s = b.tv_sec - a.tv_sec;
  long long ns = b.tv_nsec - a.tv_nsec;
  if (ns < 0) {
    s -= 1;
    ns += 1000000000LL;
  }
  return (unsigned long long)s * 1000000000ULL + (unsigned long long)ns;
}

// --- print helper ---
static inline void print_timing(const char *label, unsigned long long ns) {
  printf("%s: %llu ns (%.6f s)\n", label, ns, (double)ns / 1e9);
}
