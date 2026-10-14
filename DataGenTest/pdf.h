#ifndef PDF_H
#define PDF_H

#include "hittable_list.h"
#include "onb.h"


class pdf {
  public:
    __device__ virtual ~pdf() {}

    __device__ virtual float value(const vec3& direction) const = 0;
    __device__ virtual vec3 generate(curandState* state) const = 0;
};

class sphere_pdf : public pdf {
  public:
    __device__ sphere_pdf() {}

    __device__ float value(const vec3& direction) const override {
        return 1.0f / (4.0f * pi);
    }

    __device__ vec3 generate(curandState* state) const override {
        return random_unit_vector(state);
    }
};

class cosine_pdf : public pdf {
  public:
    __device__ cosine_pdf(const vec3& w) : uvw(w) {}

    __device__ float value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return std::fmax(0.0f, cosine_theta/pi);
    }

    __device__ vec3 generate(curandState* state) const override {
        return uvw.transform(random_cosine_direction(state));
    }

  private:
    onb uvw;
};

class hittable_pdf : public pdf {
  public:
    __device__ hittable_pdf(const hittable& objects, const point3& origin)
      : objects(objects), origin(origin)
    {}

    __device__ float value(const vec3& direction) const override {
        return objects.pdf_value(origin, direction);
    }

    __device__ vec3 generate(curandState* state) const override {
        return objects.random(origin, state);
    }

  private:
    const hittable& objects;
    point3 origin;
};

class mixture_pdf : public pdf {
  public:
    __device__ mixture_pdf(pdf* p0, pdf* p1) {
        p[0] = p0;
        p[1] = p1;
    }

    __device__ float value(const vec3& direction) const override {
        return 0.5f * p[0]->value(direction) + 0.5f * p[1]->value(direction);
    }

    __device__ vec3 generate(curandState* state) const override {
        if (random_float(state) < 0.5f)
            return p[0]->generate(state);
        else
            return p[1]->generate(state);
    }

  private:
    pdf* p[2];
};

#endif