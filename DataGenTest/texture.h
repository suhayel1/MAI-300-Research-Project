#ifndef TEXTURE_H
#define TEXTURE_H

#include "perlin.h"
#include "rtw_stb_image.h"

class texture {
  public:
    __device__ virtual ~texture() = default;

    __device__ virtual color value(float u, float v, const point3& p) const = 0;
};

class solid_color : public texture {
  public:
    __device__ solid_color(const color& albedo) : albedo(albedo) {}

    __device__ solid_color(float red, float green, float blue) : solid_color(color(red,green,blue)) {}

    __device__ color value(float u, float v, const point3& p) const override {
        return albedo;
    }

  private:
    color albedo;
};

class checker_texture : public texture {
  public:
    __device__ checker_texture(float scale, texture* even, texture* odd)
      : inv_scale(1.0f / scale), even(even), odd(odd) {}

    __device__ checker_texture(float scale, const color& c1, const color& c2)
      : checker_texture(scale, new solid_color(c1), new solid_color(c2)) { owns = true; }

    __device__ ~checker_texture() override {
        if (owns) { delete even; delete odd; }
    }

    __device__ color value(float u, float v, const point3& p) const override {
        auto xInteger = int(std::floor(inv_scale * p.x()));
        auto yInteger = int(std::floor(inv_scale * p.y()));
        auto zInteger = int(std::floor(inv_scale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

  private:
    float inv_scale;
    texture* even;
    texture* odd;
    bool owns = false;
};

class image_texture : public texture {
  public:
    // data must point to GPU memory: width*height pixels, 3 bytes each (RGB)
    __device__ image_texture(const unsigned char* data, int width, int height)
      : data(data), width(width), height(height) {}

    __device__ color value(float u, float v, const point3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (data == nullptr || height <= 0) return color(0,1,1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = interval(0,1).clamp(u);
        v = 1.0f - interval(0,1).clamp(v);  // Flip V to image coordinates

        int i = int(u * width);
        int j = int(v * height);
        if (i >= width)  i = width - 1;
        if (j >= height) j = height - 1;

        const unsigned char* pixel = data + 3 * (j * width + i);

        auto color_scale = 1.0f / 255.0f;
        return color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

  private:
    const unsigned char* data;   // not owned: main() cudaFree's it
    int width, height;
};

class noise_texture : public texture {
  public:
    __device__ noise_texture(float scale, curandState* state) : noise(state), scale(scale) {}

    __device__ color value(float u, float v, const point3& p) const override {
        return color(.5f, .5f, .5f) * (1 + std::sin(scale * p.z() + 10 * noise.turb(p, 7)));
    }

  private:
    perlin noise;
    float scale;
};

#endif