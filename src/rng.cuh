#pragma once

// Random numbers for the tracer: cuRAND's Philox4x32-10 generator.
//
// Philox is "counter based": the same (seed, subsequence) pair always gives
// the same stream. We use the ray index as the subsequence, so every ray has
// its own independent stream and results do not depend on how the GPU
// schedules threads.

#include <curand_kernel.h>

using Rng = curandStatePhilox4_32_10_t;

// Start the random stream for one ray.
__device__ inline Rng make_rng(unsigned long long seed, unsigned long long ray) {
    Rng rng;
    curand_init(seed, /*subsequence=*/ray, /*offset=*/0, &rng);
    return rng;
}

// Uniform random double in (0, 1]  (0 is never returned, so log(u) is safe).
__device__ inline double uniform01(Rng& rng) {
    return curand_uniform_double(&rng);
}
