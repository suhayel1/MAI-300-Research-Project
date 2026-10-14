#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

class bvh_node : public hittable {
  public:
    // Builds the BVH over a COPY of the list's pointer array, so the list itself isn't reordered.
    // (The book got this copy for free by passing the std::vector by value.)
    // The list still owns the objects: keep it alive while the BVH is in use.
    __device__ bvh_node(const hittable_list& list) {
        int n = list.size;
        hittable** objects = new hittable*[n];
        for (int i = 0; i < n; i++)
            objects[i] = list.objects[i];

        build(objects, 0, n);

        delete[] objects;   // only the temporary array; the objects themselves stay
    }

    __device__ bvh_node(hittable** objects, int start, int end) {
        build(objects, start, end);
    }

    // Only delete child nodes this node created. Leaves are scene objects owned by the list.
    __device__ ~bvh_node() override {
        if (owns_left)  delete left;
        if (owns_right) delete right;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, curandState* state) const override {
        if (!bbox.hit(r, ray_t))
            return false;

        bool hit_left = left->hit(r, ray_t, rec, state);
        bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, state);

        return hit_left || hit_right;
    }

    __device__ aabb bounding_box() const override { return bbox; }

  private:
    hittable* left;
    hittable* right;
    bool owns_left = false;
    bool owns_right = false;
    aabb bbox;

    __device__ void build(hittable** objects, int start, int end) {
        // Build the bounding box of the span of source objects.
        bbox = aabb();   // empty (was aabb::empty)
        for (int i = start; i < end; i++)
            bbox = aabb(bbox, objects[i]->bounding_box());

        int axis = bbox.longest_axis();
        int object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start+1];
        } else {
            sort_by_axis(objects, start, end, axis);

            int mid = start + object_span/2;
            left  = new bvh_node(objects, start, mid);  owns_left  = true;
            right = new bvh_node(objects, mid, end);    owns_right = true;
        }
    }

    __device__ static bool box_compare(const hittable* a, const hittable* b, int axis_index) {
        auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
        auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    // std::sort isn't available on the GPU, so: insertion sort.
    // Plenty fast for scenes up to a few thousand objects (it only runs once, at setup).
    __device__ static void sort_by_axis(hittable** objects, int start, int end, int axis) {
        for (int i = start + 1; i < end; i++) {
            hittable* key = objects[i];
            int j = i - 1;
            while (j >= start && box_compare(key, objects[j], axis)) {
                objects[j+1] = objects[j];
                j--;
            }
            objects[j+1] = key;
        }
    }
};

#endif