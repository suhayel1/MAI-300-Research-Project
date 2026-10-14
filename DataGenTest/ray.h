#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    __host__ __device__ ray() {}
    
    __host__ __device__ ray(const point3& origin, const vec3& direction, float time)
      : orig(origin), dir(direction), tm(time) {}

    __host__ __device__ ray(const point3& origin, const vec3& direction)
      : ray(origin, direction, 0) {}

    __host__ __device__ const point3& origin() const  { return orig; }
    __host__ __device__ const vec3& direction() const { return dir; }

    __host__ __device__ float time() const { return tm; }

    __host__ __device__ point3 at(float t) const {
        return orig + t*dir;
    }

  private:
    point3 orig;
    vec3 dir;
    float tm;
};

#endif