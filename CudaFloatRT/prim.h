#ifndef PRIM_H
#define PRIM_H

#include "hittable.h"
#include "mat3.h"

// A flat shape in the plane through Q spanned by u and v (the book's quad maths).
// Each shape says which points of the plane belong to it (is_interior),
// how big it is (area) and how to pick a random point on it (random).

class prim : public hittable {
  public:
    __device__ prim(const point3& Q, const vec3& u, const vec3& v, material* mat)
      : Q(Q), u(u), v(v), mat(mat)
    {
        auto n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n,n);
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const override {
        auto denom = dot(normal, r.direction());

        // No hit if the ray is parallel to the plane.
        if (std::fabs(denom) < 1e-8f)
            return false;

        // Return false if the hit point parameter t is outside the ray interval.
        auto t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.contains(t))
            return false;

        // Determine if the hit point lies within the planar shape using its plane coordinates.
        auto intersection = r.at(t);
        vec3 planar_hitpt_vector = intersection - Q;
        auto alpha = dot(w, cross(planar_hitpt_vector, v));
        auto beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec))
            return false;

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t;
        rec.p = intersection;
        rec.mat = mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    __device__ virtual bool is_interior(float a, float b, hit_record& rec) const = 0;

    __device__ float pdf_value(const point3& origin, const vec3& direction) const override {
        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001f, infinity), rec, nullptr))
            return 0;

        auto distance_squared = rec.t * rec.t * direction.length_squared();
        auto cosine = std::fabs(dot(direction, rec.normal) / direction.length());

        return distance_squared / (cosine * area);
    }

  protected:
    point3 Q;
    vec3 u, v;
    vec3 w;
    material* mat;
    aabb bbox;
    vec3 normal;
    float D;
    float area;
};

#endif