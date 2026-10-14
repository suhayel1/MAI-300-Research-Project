#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
  public:
    float min, max;

    __host__ __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __host__ __device__ interval(float min, float max) : min(min), max(max) {}

    __host__ __device__ interval(const interval& a, const interval& b) {
        // Create the interval tightly enclosing the two input intervals.
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    __host__ __device__ float size() const {
        return max - min;
    }

    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__ interval expand(float delta) const {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }

    //static const interval empty, universe;
};

// const interval interval::empty    = interval(+infinity, -infinity);
// const interval interval::universe = interval(-infinity, +infinity);
__host__ __device__ static interval empty_interval()    { return interval(+infinity, -infinity); }
__host__ __device__ static interval universe_interval() { return interval(-infinity, +infinity); }

__host__ __device__ inline interval operator+(const interval& ival, float displacement) {
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__ inline interval operator+(float displacement, const interval& ival) {
    return ival + displacement;
}

#endif