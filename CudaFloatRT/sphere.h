#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "onb.h"

class sphere : public hittable {
  public:
    // Stationary Sphere
    __device__ sphere(const point3& static_center, float radius, material* mat)
      : center(static_center, vec3(0,0,0)), radius(std::fmax(0.0f,radius)), mat(mat)
    {
        auto rvec = vec3(radius, radius, radius);
        bbox = aabb(static_center - rvec, static_center + rvec);
    }

    // Moving Sphere
    __device__ sphere(const point3& center1, const point3& center2, float radius, material* mat)
      : center(center1, center2 - center1), radius(std::fmax(0.0f,radius)), mat(mat)
    {
        auto rvec = vec3(radius, radius, radius);
        aabb box1(center.at(0) - rvec, center.at(0) + rvec);
        aabb box2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = aabb(box1, box2);
    }

    // No destructor: materials are shared between objects and freed once in free_world.

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const override {
        point3 current_center = center.at(r.time());
        vec3 oc = current_center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        // Float-friendly version of the book's maths (same result, less rounding error).
        // h*h - a*c is rewritten as a*(r^2 - |l|^2), where l is oc minus its part along the ray.
        vec3 l = oc - (h / a) * r.direction();
        auto discriminant = a * (radius*radius - l.length_squared());
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Compute the root that doesn't suffer cancellation first, then get the other
        // one from root_a * root_b = c / a.
        auto q = h + std::copysign(sqrtd, h);
        auto root_a = q / a;
        auto root_b = c / q;
        auto root_near = std::fmin(root_a, root_b);
        auto root_far  = std::fmax(root_a, root_b);

        // Find the nearest root that lies in the acceptable range.
        auto root = root_near;
        if (!ray_t.surrounds(root)) {
            root = root_far;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ float pdf_value(const point3& origin, const vec3& direction) const override {
        // This method only works for stationary spheres.

        hit_record rec;
        if (!this->hit(ray(origin, direction), interval(0.001f, infinity), rec, nullptr))
            return 0;

        auto dist_squared = (center.at(0) - origin).length_squared();
        auto cos_theta_max = std::sqrt(std::fmax(0.0f, 1 - radius*radius/dist_squared));
        auto solid_angle = 2*pi*(1-cos_theta_max);

        return  1.0f / solid_angle;
    }

    __device__ vec3 random(const point3& origin, curandState* state) const override {
        vec3 direction = center.at(0) - origin;
        auto distance_squared = direction.length_squared();
        onb uvw(direction);
        return uvw.transform(random_to_sphere(radius, distance_squared, state));
    }

  private:
    ray center;
    float radius;
    material* mat;
    aabb bbox;

    __host__ __device__ static void get_sphere_uv(const point3& p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = std::acos(-p.y());
        auto phi = std::atan2(-p.z(), p.x()) + pi;

        u = phi / (2*pi);
        v = theta / pi;
    }

    __device__ static vec3 random_to_sphere(float radius, float distance_squared, curandState* state) {
        auto r1 = random_float(state);
        auto r2 = random_float(state);
        auto z = 1 + r2*(std::sqrt(std::fmax(0.0f, 1 - radius*radius/distance_squared)) - 1);

        auto phi = 2*pi*r1;
        auto x = std::cos(phi) * std::sqrt(std::fmax(0.0f, 1-z*z));
        auto y = std::sin(phi) * std::sqrt(std::fmax(0.0f, 1-z*z));

        return vec3(x, y, z);
    }
};

// Unit sphere: diameter 1, centred at the origin (spans -0.5 to +0.5, like unit_box).
// Place it with ONE transform:
//     new transform(unit_sphere(mat), translation(p) * rotation_local(...) * scaling(size))
// Don't put a transformed sphere in the lights list; use sphere(center, radius, ...) there.
__device__ inline sphere* unit_sphere(material* mat) {
    return new sphere(point3(0, 0, 0), 0.5f, mat);
}

#endif