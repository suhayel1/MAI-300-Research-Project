#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"

class material;

class hit_record {
  public:
    point3 p;
    vec3 normal;
    material* mat;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
  public:
    __device__ virtual ~hittable() = default;

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const = 0;

    __device__ virtual aabb bounding_box() const = 0;

    __device__ virtual float pdf_value(const point3& origin, const vec3& direction) const {
        return 0.0f;
    }

    __device__ virtual vec3 random(const point3& origin, curandState* state) const {
        return vec3(1,0,0);
    }
};

#endif