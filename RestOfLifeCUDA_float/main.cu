#include "rtweekend.h"

#include "bvh.h"
#include "camera.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"

#include <chrono>

// Builds the whole scene ON THE GPU (one thread), so every object's vtable is a GPU address.
// The resulting pointers are written into GPU memory that main() allocated:
//   d_world[0]  = the world list
//   d_lights[0] = the lights list
//   d_mats[0..4] = the materials (kept so free_world can delete them once)
__global__ void create_world(hittable** d_world, hittable** d_lights, material** d_mats) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;   // only one thread builds the scene

    auto world = new hittable_list();

    auto red   = new lambertian(color(.65, .05, .05));
    auto white = new lambertian(color(.73, .73, .73));
    auto green = new lambertian(color(.12, .45, .15));
    auto blue = new lambertian(color(.12f, .15f, .45f));
    auto light = new diffuse_light(color(15, 15, 15));

    // Cornell box sides
    world->add(new quad(point3(555,0,0), vec3(0,0,555), vec3(0,555,0), green));
    world->add(new quad(point3(0,0,555), vec3(0,0,-555), vec3(0,555,0), red));
    world->add(new quad(point3(0,555,0), vec3(555,0,0), vec3(0,0,555), white));
    world->add(new quad(point3(0,0,555), vec3(555,0,0), vec3(0,0,-555), white));
    world->add(new quad(point3(555,0,555), vec3(-555,0,0), vec3(0,555,0), white));

    // Light
    world->add(new quad(point3(213,554,227), vec3(130,0,0), vec3(0,0,105), light));

    // Box
    hittable* box1 = box(point3(0,0,0), point3(165,330,165), white);
    box1 = new rotate_y(box1, 15);
    box1 = new translate(box1, vec3(265,0,295));
    world->add(box1);

    // Glass Sphere
    auto glass = new dielectric(1.5);
    world->add(new sphere(point3(190,90,190), 90, glass));

    // Light Sources (separate objects from the world's, so nothing is deleted twice)
    material* empty_material = nullptr;
    auto lights = new hittable_list();
    lights->add(new quad(point3(343,554,332), vec3(-130,0,0), vec3(0,0,-105), empty_material));
    lights->add(new sphere(point3(190, 90, 190), 90, empty_material));

    *d_world  = world;
    *d_lights = lights;
    d_mats[0] = red;
    d_mats[1] = white;
    d_mats[2] = green;
    d_mats[3] = light;
    d_mats[4] = glass;
}

// Frees everything create_world made. Deleting a list deletes the objects in it
// (and translate/rotate_y delete what they wrap), so only lists + materials are listed here.
__global__ void free_world(hittable** d_world, hittable** d_lights, material** d_mats, int num_mats) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    delete *d_world;
    delete *d_lights;
    for (int i = 0; i < num_mats; i++)
        delete d_mats[i];
}

int main() {
    // Virtual calls stop the compiler from working out how much stack each GPU thread needs,
    // so give each thread a bit more than the 1 KB default (hit() calls nest several levels deep).
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    // GPU memory to hold the pointers create_world produces.
    const int num_mats = 5;
    hittable** d_world;
    hittable** d_lights;
    material** d_mats;
    CUDA_CHECK(cudaMalloc(&d_world,  sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc(&d_lights, sizeof(hittable*)));
    CUDA_CHECK(cudaMalloc(&d_mats,   num_mats * sizeof(material*)));

    create_world<<<1, 1>>>(d_world, d_lights, d_mats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    camera cam;

    cam.aspect_ratio      = 1.0;
    cam.image_width       = 256;
    cam.samples_per_pixel = 1000;
    cam.max_depth         = 50;
    cam.background        = color(0,0,0);

    cam.vfov     = 40;
    cam.lookfrom = point3(278, 278, -800);
    cam.lookat   = point3(278, 278, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0;

    auto start = std::chrono::steady_clock::now();
    cam.render(d_world, d_lights);
    auto stop = std::chrono::steady_clock::now();
    std::clog << "Render time: " << std::chrono::duration<double>(stop - start).count() << " s\n";

    free_world<<<1, 1>>>(d_world, d_lights, d_mats, num_mats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_world));
    CUDA_CHECK(cudaFree(d_lights));
    CUDA_CHECK(cudaFree(d_mats));

    cudaDeviceReset();
}