#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "hittable.h"
#include "mat3.h"

// Places an object in the world with one chain of moves, rotations, scales and shears:
//
//     obj = new transform(obj, translation(position) * rotation_local(tilt, turn, roll)
//                              * scaling(size));
//
// Read the chain right to left: scale, then rotate, then move. Build the object around
// the origin so it rotates about its own centre. A chain without translation(...)
// also works, e.g. transform(obj, rotation_y(30)) turns it without moving it.
//
// Like the book's translate and rotate_y, it does not pass pdf_value / random through,
// so don't use a transformed object as a light. No zero scale factors.
class transform : public hittable {
  public:
    __device__ transform(hittable* object, const affine& placement)
      : object(object), to_world(placement)
    {
        to_object = placement.inverse();
        normal_matrix = to_object.M.transpose();

        // Bounding box: place the 8 corners of the object's box and wrap them.
        aabb b = object->bounding_box();
        point3 min( infinity,  infinity,  infinity);
        point3 max(-infinity, -infinity, -infinity);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    point3 corner(i ? b.x.max : b.x.min,
                                  j ? b.y.max : b.y.min,
                                  k ? b.z.max : b.z.min);
                    point3 p = to_world.apply_point(corner);
                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fmin(min[c], p[c]);
                        max[c] = std::fmax(max[c], p[c]);
                    }
                }
            }
        }
        bbox = aabb(min, max);
    }

    __device__ ~transform() override { delete object; }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const override {
        // Move the ray into object space (undo the move first, then the matrix: this
        // order loses less float precision than applying to_object in one step).
        // The direction is NOT normalized afterwards, so the object-space t equals the
        // world-space t and ray_t needs no change.
        point3 local_origin = to_object.M * (r.origin() - to_world.t);
        ray local_r(local_origin, to_object.apply_vector(r.direction()), r.time());

        if (!object->hit(local_r, ray_t, rec, state))
            return false;

        // Hit point back to world space.
        rec.p = to_world.apply_point(rec.p);

        // Normals use the inverse transpose of the matrix (for a pure rotation that is the
        // rotation itself, for a scale it divides by the scale factors), then need
        // normalizing again. front_face stays correct: dot(direction, normal) keeps its sign.
        rec.normal = unit_vector(normal_matrix * rec.normal);

        return true;
    }

    __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable* object;
    affine to_world;       // object space -> world space (the placement you gave)
    affine to_object;      // world space -> object space (its inverse)
    mat3 normal_matrix;    // turns object-space normals into world-space normals
    aabb bbox;
};

#endif
