#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <memory>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385f;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__device__ inline float random_float(curandState *state) {
    // Returns a random real in [0,1).
    // curand_uniform gives (0,1]. Map the single value 1.0 to 0.0 instead of
    // computing 1 - x, which in float can round up to exactly 1.0.
    float r = curand_uniform(state);
    return (r == 1.0f) ? 0.0f : r;
}

__device__ inline float random_float(curandState *state, float min, float max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_float(state);
}

__device__ inline int random_int(curandState *state, int min, int max) {
    // Returns a random integer in [min,max].
    int i = int(random_float(state, float(min), float(max+1)));
    return (i > max) ? max : i;   // guard against float rounding up to max+1
}

// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif