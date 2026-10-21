#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "pdf.h"
#include "material.h"
#include "npy.h"

#include <fstream>
#include <iostream>
#include <vector>

// Stops with a readable message if a CUDA call fails.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::cerr << "CUDA error " << cudaGetErrorName(err_) << " at "        \
                      << __FILE__ << ":" << __LINE__ << ": "                      \
                      << cudaGetErrorString(err_) << "\n";                        \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

class camera {
  public:
    float aspect_ratio      = 1.0f;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 10;   // Count of random samples for each pixel
    int    max_depth         = 10;   // Maximum number of ray bounces into scene
    color  background;               // Scene background color

    float vfov = 90;                  // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    float defocus_angle = 0;  // Variation angle of rays through each pixel
    float focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    bool save_gbuffer_previews = true;  // also write small PPM pictures of the G-buffer (for checking)

    // CPU: sets up the camera, launches the GPU kernels, and writes the image.
    // world and lights point to GPU memory (filled in by create_world).
    void render(hittable** world, hittable** lights);

    // ---- Derived values: set by initialize() on the CPU, read by the GPU kernel ----
    // (Public so the render kernel, which can't be a member function, can read them.)
    int    image_height;         // Rendered image height
    float pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    int    sqrt_spp;             // Square root of number of samples per pixel
    float recip_sqrt_spp;       // 1 / sqrt_spp
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    void initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        sqrt_spp = int(std::sqrt(samples_per_pixel));
        pixel_samples_scale = 1.0f / (sqrt_spp * sqrt_spp);
        recip_sqrt_spp = 1.0f / sqrt_spp;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (float(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ ray get_ray(int i, int j, int s_i, int s_j, curandState* state) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j for stratified sample square s_i, s_j.

        auto offset = sample_square_stratified(s_i, s_j, state);
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(state);
        auto ray_direction = pixel_sample - ray_origin;
        auto ray_time = random_float(state);

        return ray(ray_origin, ray_direction, ray_time);
    }

    __device__ vec3 sample_square_stratified(int s_i, int s_j, curandState* state) const {
        // Returns the vector to a random point in the square sub-pixel specified by grid
        // indices s_i and s_j, for an idealized unit square pixel [-.5,-.5] to [+.5,+.5].

        auto px = ((s_i + random_float(state)) * recip_sqrt_spp) - 0.5f;
        auto py = ((s_j + random_float(state)) * recip_sqrt_spp) - 0.5f;

        return vec3(px, py, 0);
    }

    __device__ point3 defocus_disk_sample(curandState* state) const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk(state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    // The book's recursive ray_color, rewritten as a loop.
    //   recursive:  return emitted + attenuation * ray_color(next)
    //   loop:       radiance += throughput * emitted;  throughput *= attenuation
    __device__ color ray_color(const ray& r_in, const hittable& world, const hittable& lights,
                               curandState* state) const {
        ray r = r_in;
        color radiance(0,0,0);     // light collected so far
        color throughput(1,1,1);   // product of all attenuations so far

        for (int depth = 0; depth < max_depth; depth++) {
            hit_record rec;

            // If the ray hits nothing, add the background color and stop.
            if (!world.hit(r, interval(0.001f, infinity), rec, state)) {
                radiance += throughput * background;
                break;
            }

            scatter_record srec;
            color color_from_emission = rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);
            radiance += throughput * color_from_emission;

            if (!rec.mat->scatter(state, r, rec, srec))
                break;

            // Metal / glass: follow the single scattered ray.
            if (srec.skip_pdf) {
                throughput = throughput * srec.attenuation;
                r = srec.skip_pdf_ray;
                continue;
            }

            // Local PDF objects (no new/delete): mixture of light sampling and the material's PDF.
            hittable_pdf light_pdf(lights, rec.p);
            cosine_pdf   cos_pdf(rec.normal);
            sphere_pdf   sph_pdf;
            pdf* mat_pdf = (srec.pdf_type == PDF_SPHERE) ? (pdf*)&sph_pdf : (pdf*)&cos_pdf;
            mixture_pdf p(&light_pdf, mat_pdf);

            ray scattered = ray(rec.p, p.generate(state), r.time());
            auto pdf_value = p.value(scattered.direction());

            float scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);

            if (!(pdf_value > 0.0f))   // also catches NaN
                break;

            throughput = throughput * srec.attenuation * scattering_pdf / pdf_value;
            r = scattered;
        }

        return radiance;
    }
};

// ---- G-buffer ------------------------------------------------------------------
// For one unjittered ray through each pixel centre, the first SURFACE the camera sees
// (fog / constant_medium is looked through).
// 18 floats per pixel, in this order:
//    0      depth         distance in front of the camera, along the viewing axis
//    1- 3   position      world-space hit point
//    4- 6   normal        unit normal, facing the camera
//    7- 9   albedo        surface colour (glass: 1,1,1)
//   10-12   view_dir      unit vector from the surface toward the camera (filled even on a miss)
//   13-15   material      one-hot: diffuse, metal, dielectric
//   16      roughness     metal fuzz (0 otherwise)
//   17      ior           glass index of refraction (0 otherwise)
// Pixels where the ray hits nothing are 0 except view_dir. Lights are hit (depth > 0)
// but have an all-zero material one-hot.
constexpr int GBUFFER_CHANNELS = 18;

__device__ inline void gbuffer_pixel(const camera& cam, const hittable& world, int i, int j,
                                     curandState* state, float* g) {
    for (int k = 0; k < GBUFFER_CHANNELS; k++) g[k] = 0.0f;

    point3 pixel_center = cam.pixel00_loc + (float(i) * cam.pixel_delta_u) + (float(j) * cam.pixel_delta_v);
    ray r(cam.center, pixel_center - cam.center, 0.0f);

    vec3 view_dir = -unit_vector(r.direction());
    g[10] = view_dir.x(); g[11] = view_dir.y(); g[12] = view_dir.z();

    // nullptr instead of a random state: fog (constant_medium) is skipped, so the
    // G-buffer describes the surface behind it. Nothing else uses the state in hit().
    hit_record rec;
    if (!world.hit(r, interval(0.001f, infinity), rec, nullptr))
        return;

    gbuffer_info m = rec.mat->gbuffer(rec);

    g[0] = dot(rec.p - cam.center, -cam.w);
    g[1] = rec.p.x();      g[2] = rec.p.y();      g[3] = rec.p.z();
    g[4] = rec.normal.x(); g[5] = rec.normal.y(); g[6] = rec.normal.z();
    g[7] = m.albedo.x();   g[8] = m.albedo.y();   g[9] = m.albedo.z();
    if (m.type != GMAT_NONE) g[13 + (m.type - 1)] = 1.0f;
    g[16] = m.roughness;
    g[17] = m.ior;
}

// ---- GPU kernels ------------------------------------------------------------

// One thread per pixel: give each pixel its own random-number stream.
__global__ void render_init(int width, int height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int pixel_index = j * width + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// One thread per pixel: fill the G-buffer. Uses a COPY of the pixel's random state and
// doesn't save it back, so the render's random numbers are exactly the same as without it.
__global__ void gbuffer_kernel(camera cam, hittable** world, curandState* rand_state, float* gbuf) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= cam.image_width || j >= cam.image_height) return;
    int pixel_index = j * cam.image_width + i;

    curandState local_state = rand_state[pixel_index];
    gbuffer_pixel(cam, **world, i, j, &local_state, gbuf + size_t(pixel_index) * GBUFFER_CHANNELS);
}

// One thread per pixel, ONE sample per launch (stratum s_i, s_j), added into accum.
// Short launches avoid Windows killing kernels that run longer than ~2 seconds.
__global__ void render_kernel(camera cam, hittable** world, hittable** lights,
                              curandState* rand_state, color* accum, int s_i, int s_j) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= cam.image_width || j >= cam.image_height) return;
    int pixel_index = j * cam.image_width + i;

    curandState local_state = rand_state[pixel_index];   // work on a local copy (faster)

    ray r = cam.get_ray(i, j, s_i, s_j, &local_state);
    color c = cam.ray_color(r, **world, **lights, &local_state);
    if (c.x() == c.x() && c.y() == c.y() && c.z() == c.z())   // NaN != NaN
        accum[pixel_index] += c;

    rand_state[pixel_index] = local_state;                // save it for the next launch
}

// ---- G-buffer previews (CPU) ------------------------------------------------------
// Small binary PPM pictures for checking the G-buffer by eye. Not used for training.

inline unsigned char preview_byte(float x) {
    if (!(x > 0)) return 0;          // also catches NaN
    if (x > 0.999f) x = 0.999f;
    return (unsigned char)(int(256 * x));
}

inline void write_preview_ppm(const char* path, const std::vector<unsigned char>& rgb, int width, int height) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
}

inline void write_gbuffer_previews(const std::vector<float>& gbuf, int width, int height) {
    int n = width * height;
    std::vector<unsigned char> normal(3 * n), albedo(3 * n), depth(3 * n), mat(3 * n);

    float max_depth = 0;
    for (int i = 0; i < n; i++) max_depth = std::fmax(max_depth, gbuf[size_t(i) * GBUFFER_CHANNELS]);

    for (int i = 0; i < n; i++) {
        const float* g = &gbuf[size_t(i) * GBUFFER_CHANNELS];
        bool hit = g[0] > 0;
        for (int c = 0; c < 3; c++) {
            normal[3*i + c] = hit ? preview_byte(0.5f * g[4 + c] + 0.5f) : 0;  // -1..1 -> 0..1
            albedo[3*i + c] = preview_byte(std::sqrt(std::fmax(0.0f, g[7 + c])));  // gamma, like write_color
            depth[3*i + c]  = hit ? preview_byte(1.0f - 0.8f * g[0] / max_depth) : 0;  // near = bright
            // material: diffuse = red, metal = green, glass = blue, light = white
            bool is_light = hit && g[13] == 0 && g[14] == 0 && g[15] == 0;
            mat[3*i + c] = is_light ? 255 : preview_byte(g[13 + c]);
        }
    }
    write_preview_ppm("gbuffer_normal.ppm", normal, width, height);
    write_preview_ppm("gbuffer_albedo.ppm", albedo, width, height);
    write_preview_ppm("gbuffer_depth.ppm", depth, width, height);
    write_preview_ppm("gbuffer_material.ppm", mat, width, height);
}

// ---- camera::render (CPU) -----------------------------------------------------

inline void camera::render(hittable** world, hittable** lights) {
    initialize();

    int num_pixels = image_width * image_height;

    curandState* d_rand_state;
    color* d_accum;
    CUDA_CHECK(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_accum, num_pixels * sizeof(color)));
    CUDA_CHECK(cudaMemset(d_accum, 0, num_pixels * sizeof(color)));   // all zeros = color(0,0,0)

    dim3 threads(8, 8);
    dim3 blocks((image_width  + threads.x - 1) / threads.x,
                (image_height + threads.y - 1) / threads.y);

    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // G-buffer: one ray per pixel, takes about a millisecond.
    float* d_gbuf;
    CUDA_CHECK(cudaMalloc(&d_gbuf, size_t(num_pixels) * GBUFFER_CHANNELS * sizeof(float)));
    gbuffer_kernel<<<blocks, threads>>>(*this, world, d_rand_state, d_gbuf);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gbuf(size_t(num_pixels) * GBUFFER_CHANNELS);
    CUDA_CHECK(cudaMemcpy(gbuf.data(), d_gbuf, gbuf.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_gbuf));

    if (!write_npy("gbuffer.npy", gbuf.data(), image_height, image_width, GBUFFER_CHANNELS))
        std::cerr << "Could not write gbuffer.npy\n";
    if (save_gbuffer_previews)
        write_gbuffer_previews(gbuf, image_width, image_height);

    for (int s_j = 0; s_j < sqrt_spp; s_j++) {
        std::clog << "\rStrata rows remaining: " << (sqrt_spp - s_j) << ' ' << std::flush;
        for (int s_i = 0; s_i < sqrt_spp; s_i++) {
            render_kernel<<<blocks, threads>>>(*this, world, lights, d_rand_state, d_accum, s_i, s_j);
            CUDA_CHECK(cudaGetLastError());       // bad launch configuration, etc.
            CUDA_CHECK(cudaDeviceSynchronize());  // errors while the kernel ran
        }
    }

    // Copy the summed colors back from the GPU.
    std::vector<color> accum(num_pixels);
    CUDA_CHECK(cudaMemcpy(accum.data(), d_accum, num_pixels * sizeof(color), cudaMemcpyDeviceToHost));

    // Linear radiance: average over samples, nothing clipped or gamma-corrected.
    std::vector<color> radiance(num_pixels);
    for (int i = 0; i < num_pixels; i++)
        radiance[i] = pixel_samples_scale * accum[i];
    if (!write_npy("image.npy", reinterpret_cast<const float*>(radiance.data()), image_height, image_width, 3))
        std::cerr << "Could not write image.npy\n";

    // 8-bit preview image, exactly like the book.
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int i = 0; i < num_pixels; i++)
        write_color(std::cout, radiance[i]);

    std::clog << "\rDone.                 \n";

    CUDA_CHECK(cudaFree(d_accum));
    CUDA_CHECK(cudaFree(d_rand_state));
}

#endif