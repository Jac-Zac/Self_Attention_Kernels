#pragma once
#include <time.h>

// ·········································································
//  CPU TIME for process
// ·········································································

// return process cpu time
#define PCPU_TIME                                                              \
  ({                                                                           \
    struct timespec ts;                                                        \
    (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts),                             \
     (long double)ts.tv_sec + (long double)ts.tv_nsec * 1e-9);                 \
  })

// ·········································································
//  CPU TIME for thread
// ·········································································

// return thread cpu time
#define TCPU_TIME                                                              \
  ({                                                                           \
    struct timespec ts;                                                        \
    (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts),                              \
     (long double)ts.tv_sec + (long double)ts.tv_nsec * 1e-9);                 \
  })

/* ·········································································
 *  return the number of nanosecond between two different point,
 *  processing two timespec structures
 *  returns a single unsigned long long
 * ·········································································
 */

// both TSTART and TSTOP are struct timespec
// for instance returned by clock_gettime

#define GET_DELTAT(TSTART, TSTOP)                                              \
  ({                                                                           \
    long long sec_diff = (TSTOP).tv_sec - (TSTART).tv_sec;                     \
    long long nsec_diff = (TSTOP).tv_nsec - (TSTART).tv_nsec;                  \
    if (nsec_diff < 0) {                                                       \
      nsec_diff += 1000000000LL;                                               \
      sec_diff -= 1;                                                           \
    }                                                                          \
    (unsigned long long)sec_diff * 1000000000ULL +                             \
        (unsigned long long)nsec_diff;                                         \
  })
