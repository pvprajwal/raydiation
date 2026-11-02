#pragma once

// cuRAND Philox4x32-10

#include <curand_kernel.h>

using Rng = curandStatePhilox4_32_10_t;

__device__ inline Rng make_rng(unsigned long long seed, unsigned long long ray)
{
    Rng rng;
    curand_init(seed, ray, 0, &rng);
    return rng;
}

// Uniform random double in (0, 1] 
__device__ inline double uniform01(Rng &rng)
{
    return curand_uniform_double(&rng);
}
