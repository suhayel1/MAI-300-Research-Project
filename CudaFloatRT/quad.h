#ifndef QUAD_H
#define QUAD_H

#include "prim.h"
#include "hittable_list.h"

class quad : public prim {
  public:
    __device__ quad(const point3& Q, const vec3& u, const vec3& v, material* mat)
      : prim(Q, u, v, mat)
    {
        area = cross(u, v).length();
        set_bounding_box();
    }

    // Unit square centred at the origin in the xy plane, facing +z.
    // Place it with transform(...); lights should use the constructor above instead.
    __device__ quad(material* mat)
      : quad(point3(-0.5f, -0.5f, 0), vec3(1, 0, 0), vec3(0, 1, 0), mat) {}

    // Unit square placed by `placement`, applied to Q, u and v right here, so the
    // result is an ordinary quad (no transform wrapper) and works as a light.
    __device__ quad(const affine& placement, material* mat)
      : quad(placement.apply_point(point3(-0.5f, -0.5f, 0)),
             placement.apply_vector(vec3(1, 0, 0)),
             placement.apply_vector(vec3(0, 1, 0)), mat) {}

    // No destructor: materials are shared and freed once in free_world.

    __device__ virtual void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        auto bbox_diagonal1 = aabb(Q, Q + u + v);
        auto bbox_diagonal2 = aabb(Q + u, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    __device__ virtual bool is_interior(float a, float b, hit_record& rec) const {
        interval unit_interval = interval(0, 1);
        // Given the hit point in plane coordinates, return false if it is outside the
        // primitive, otherwise set the hit record UV coordinates and return true.

        if (!unit_interval.contains(a) || !unit_interval.contains(b))
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

    __device__ vec3 random(const point3& origin, curandState* state) const override {
        auto p = Q + (random_float(state) * u) + (random_float(state) * v);
        return p - origin;
    }
};

// Returns the 3D box (six sides) that contains the two opposite vertices a & b.
// The caller owns the returned list (deleting it deletes the six quads).
__device__ inline hittable_list* transf_box(const point3& a, const point3& b, material* mat)
{
    auto sides = new hittable_list();

    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = point3(std::fmin(a.x(),b.x()), std::fmin(a.y(),b.y()), std::fmin(a.z(),b.z()));
    auto max = point3(std::fmax(a.x(),b.x()), std::fmax(a.y(),b.y()), std::fmax(a.z(),b.z()));

    auto dx = vec3(max.x() - min.x(), 0, 0);
    auto dy = vec3(0, max.y() - min.y(), 0);
    auto dz = vec3(0, 0, max.z() - min.z());

    sides->add(new quad(point3(min.x(), min.y(), max.z()),  dx,  dy, mat)); // front
    sides->add(new quad(point3(max.x(), min.y(), max.z()), -dz,  dy, mat)); // right
    sides->add(new quad(point3(max.x(), min.y(), min.z()), -dx,  dy, mat)); // back
    sides->add(new quad(point3(min.x(), min.y(), min.z()),  dz,  dy, mat)); // left
    sides->add(new quad(point3(min.x(), max.y(), max.z()),  dx, -dz, mat)); // top
    sides->add(new quad(point3(min.x(), min.y(), min.z()),  dx,  dz, mat)); // bottom

    return sides;
}

// Unit cube centred at the origin (corners at -0.5 and +0.5), faces built directly
// with Q, u, v, so there is no per-face transform. Place the whole cube with ONE
// transform:  new transform(unit_box(mat), translation(p) * rotation_local(...) * scaling(size))
__device__ inline hittable_list* box(material* mat)
{
    return transf_box(point3(-0.5f, -0.5f, -0.5f), point3(0.5f, 0.5f, 0.5f), mat);
}

#endif