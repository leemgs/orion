#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <time.h>

static volatile uint64_t sink;

double chase_latency_ns(const uint64_t *next, size_t hops) {
    struct timespec start, end;
    uint64_t index = 0;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (size_t i = 0; i < hops; ++i) {
        index = next[index];
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    sink = index;
    const uint64_t elapsed =
        (uint64_t)(end.tv_sec - start.tv_sec) * UINT64_C(1000000000) +
        (uint64_t)(end.tv_nsec - start.tv_nsec);
    return (double)elapsed / (double)hops;
}
