#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.h"
#include "hittable.h"

class hittable_list : public hittable {
  public:
    hittable** objects = nullptr;   // was std::vector<shared_ptr<hittable>>
    int size = 0;

    __device__ hittable_list() {}
    __device__ hittable_list(hittable* object) { add(object); }

    // The list OWNS its objects: deleting the list deletes everything in it.
    __device__ ~hittable_list() override {
        for (int i = 0; i < size; i++)
            delete objects[i];
        delete[] objects;
    }

    // No copying: a copy would share the same pointers, and both copies would
    // delete them (crash). Pass lists by reference or pointer instead.
    hittable_list(const hittable_list&) = delete;
    hittable_list& operator=(const hittable_list&) = delete;

    __device__ void clear() {
        for (int i = 0; i < size; i++)
            delete objects[i];
        delete[] objects;
        objects = nullptr;
        size = capacity = 0;
        bbox = aabb();
    }

    __device__ void add(hittable* object) {
        // Grow the array when full (what std::vector::push_back does for you).
        if (size == capacity) {
            int new_capacity = (capacity == 0) ? 8 : capacity * 2;
            hittable** bigger = new hittable*[new_capacity];
            for (int i = 0; i < size; i++)
                bigger[i] = objects[i];
            delete[] objects;
            objects = bigger;
            capacity = new_capacity;
        }
        objects[size++] = object;
        bbox = aabb(bbox, object->bounding_box());
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < size; i++) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec, state)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    __device__ aabb bounding_box() const override { return bbox; }

    __device__ float pdf_value(const point3& origin, const vec3& direction) const override {
        auto weight = 1.0f / size;
        auto sum = 0.0f;

        for (int i = 0; i < size; i++)
            sum += weight * objects[i]->pdf_value(origin, direction);

        return sum;
    }

    __device__ vec3 random(const point3& origin, curandState* state) const override {
        return objects[random_int(state, 0, size-1)]->random(origin, state);
    }

  private:
    int capacity = 0;
    aabb bbox;
};

#endif