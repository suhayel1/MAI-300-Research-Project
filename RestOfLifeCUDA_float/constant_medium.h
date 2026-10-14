#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hittable.h"
#include "material.h"
#include "texture.h"

class constant_medium : public hittable {
  public:
    // owns_boundary: true  -> this medium deletes the boundary (boundary is NOT added to any list)
    //                false -> the boundary is also in a list, which deletes it instead
    __device__ constant_medium(hittable* boundary, float density, texture* tex, bool owns_boundary = true)
      : boundary(boundary), neg_inv_density(-1/density),
        phase_function(new isotropic(tex)), owns_boundary(owns_boundary)
    {}

    __device__ constant_medium(hittable* boundary, float density, const color& albedo, bool owns_boundary = true)
      : boundary(boundary), neg_inv_density(-1/density),
        phase_function(new isotropic(albedo)), owns_boundary(owns_boundary)
    {}

    __device__ ~constant_medium() override {
        delete phase_function;           // always created by this medium
        if (owns_boundary) delete boundary;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const override {
        hit_record rec1, rec2;

        if (!boundary->hit(r, universe_interval(), rec1, state))
            return false;

        if (!boundary->hit(r, interval(rec1.t+0.0001f, infinity), rec2, state))
            return false;

        if (rec1.t < ray_t.min) rec1.t = ray_t.min;
        if (rec2.t > ray_t.max) rec2.t = ray_t.max;

        if (rec1.t >= rec2.t)
            return false;

        if (rec1.t < 0)
            rec1.t = 0;

        auto ray_length = r.direction().length();
        auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
        auto hit_distance = neg_inv_density * std::log(random_float(state));

        if (hit_distance > distance_inside_boundary)
            return false;

        rec.t = rec1.t + hit_distance / ray_length;
        rec.p = r.at(rec.t);

        rec.normal = vec3(1,0,0);  // arbitrary
        rec.front_face = true;     // also arbitrary
        rec.mat = phase_function;

        return true;
    }

    __device__ aabb bounding_box() const override { return boundary->bounding_box(); }

  private:
    hittable* boundary;
    float neg_inv_density;
    material* phase_function;
    bool owns_boundary;
};

#endif