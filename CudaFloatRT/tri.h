#ifndef TRI_H
#define TRI_H

#include "prim.h"
#include "quad.h"   // pyramid() uses a quad for its base
#include "hittable_list.h"

// Triangle with corners Q, Q+u, Q+v.
class tri : public prim {
  public:
    __device__ tri(const point3& Q, const vec3& u, const vec3& v, material* mat)
      : prim(Q, u, v, mat)
    {
        area = 0.5f * cross(u, v).length();     // half the parallelogram
        set_bounding_box();                     // the three corners
    }

    // Equilateral triangle with sides of length 1, centred (centroid) at the origin in
    // the xy plane, facing +z, one corner pointing up (+y).
    // Place it with transform(...); lights should use the constructor above instead.
    __device__ tri(material* mat)
      : tri(point3(-0.5f, -0.28867513f, 0), vec3(1, 0, 0), vec3(0.5f, 0.8660254f, 0), mat) {}

    // Unit triangle placed by `placement`, applied to Q, u and v right here, so the
    // result is an ordinary tri (no transform wrapper) and works as a light.
    __device__ tri(const affine& placement, material* mat)
      : tri(placement.apply_point(point3(-0.5f, -0.28867513f, 0)),
            placement.apply_vector(vec3(1, 0, 0)),
            placement.apply_vector(vec3(0.5f, 0.8660254f, 0)), mat) {}

    __device__ virtual void set_bounding_box() {
        // Compute the bounding box of all four vertices.
        auto bbox_diagonal1 = aabb(Q, Q + u);
        auto bbox_diagonal2 = aabb(Q, Q + v);
        bbox = aabb(bbox_diagonal1, bbox_diagonal2);
    }

    __device__ bool is_interior(float a, float b, hit_record& rec) const override {
        if (a < 0 || b < 0 || a + b > 1)
            return false;

        rec.u = a;
        rec.v = b;
        return true;
    }

    __device__ vec3 random(const point3& origin, curandState* state) const override {
        // Random point in the parallelogram; if it lands in the far half, mirror it
        // back into the triangle. Every point of the triangle is then equally likely,
        // which is what pdf_value's 1/area assumes.
        auto a = random_float(state);
        auto b = random_float(state);
        if (a + b > 1) { a = 1 - a; b = 1 - b; }
        auto p = Q + (a * u) + (b * v);
        return p - origin;
    }
};

// Square pyramid: base centred at base_center, base side 2*half_width,
// apex `height` above the centre. Every face's normal points outward
// (needed for glass). The caller owns the returned list
// (deleting it deletes the five faces).
__device__ inline hittable_list* transf_square_pyramid(const point3& base_center, float half_width,
                                         float height, material* mat) {
    auto faces = new hittable_list();
    float h = half_width;
    point3 c = base_center;

    point3 p0(c.x() - h, c.y(), c.z() - h);
    point3 p1(c.x() + h, c.y(), c.z() - h);
    point3 p2(c.x() + h, c.y(), c.z() + h);
    point3 p3(c.x() - h, c.y(), c.z() + h);
    point3 apex(c.x(), c.y() + height, c.z());

    faces->add(new quad(p0, p1 - p0, p3 - p0, mat));    // base  (faces down)
    faces->add(new tri(p0, apex - p0, p1 - p0, mat));   // front (-z)
    faces->add(new tri(p1, apex - p1, p2 - p1, mat));   // right (+x)
    faces->add(new tri(p2, apex - p2, p3 - p2, mat));   // back  (+z)
    faces->add(new tri(p3, apex - p3, p0 - p3, mat));   // left  (-x)

    return faces;
}

// Triangular pyramid: equilateral triangle base centred at base_center, with each
// base corner `radius` away from the centre, and the apex `height` above the centre.
// Every face's normal points outward (needed for glass). The caller owns the returned list.
__device__ inline hittable_list* transf_triangle_pyramid(const point3& base_center, float radius,
                                                  float height, material* mat) {
    auto faces = new hittable_list();
    point3 c = base_center;
    const float s = 0.8660254f;   // sqrt(3)/2 = cos(30 degrees)

    // Base corners at 210, 330 and 90 degrees around the centre (in the xz plane).
    point3 p0(c.x() - s * radius, c.y(), c.z() - 0.5f * radius);
    point3 p1(c.x() + s * radius, c.y(), c.z() - 0.5f * radius);
    point3 p2(c.x(),              c.y(), c.z() + radius);
    point3 apex(c.x(), c.y() + height, c.z());

    faces->add(new tri(p0, p1 - p0, p2 - p0, mat));     // base  (faces down)
    faces->add(new tri(p0, apex - p0, p1 - p0, mat));   // front (-z)
    faces->add(new tri(p1, apex - p1, p2 - p1, mat));   // right-back
    faces->add(new tri(p2, apex - p2, p0 - p2, mat));   // left-back

    return faces;
}

// Unit square pyramid: base 1 x 1 centred on the y axis at y = -0.5, apex at y = +0.5.
// Place it with ONE transform, like unit_box.
__device__ inline hittable_list* square_pyramid(material* mat) {
    return transf_square_pyramid(point3(0, -0.5f, 0), 0.5f, 1.0f, mat);
}

// Unit triangular pyramid: equilateral base with sides of length 1, centred on the
// y axis at y = -0.5, apex at y = +0.5. Place it with ONE transform, like unit_box.
__device__ inline hittable_list* triangle_pyramid(material* mat) {
    return transf_triangle_pyramid(point3(0, -0.5f, 0), 0.57735027f, 1.0f, mat);   // radius = 1/sqrt(3) gives side 1
}

#endif