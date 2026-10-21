#include "rtweekend.h"

#include "bvh.h"
#include "camera.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"
#include "transform.h"
#include "tri.h"

#include <chrono>

// Builds the whole scene ON THE GPU (one thread), so every object's vtable is a GPU address.
// The resulting pointers are written into GPU memory that main() allocated:
//   d_world[0]  = the world list
//   d_lights[0] = the lights list
//   d_mats[0..5] = the materials (kept so free_world can delete them once)
__global__ void create_world(hittable** d_world, hittable** d_lights, material** d_mats) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;   // only one thread builds the scene

    auto world = new hittable_list();

    auto red   = new lambertian(color(.65f, .05f, .05f));
    auto white = new lambertian(color(.73f, .73f, .73f));
    auto green = new lambertian(color(.12f, .45f, .15f));
    auto blue  = new lambertian(color(.12f, .15f, .45f));
    auto light = new diffuse_light(color(15, 15, 15));

    // Cornell box sides: unit quads (facing +z) scaled to 555 x 555, turned to face
    // into the room, and moved to the middle of each wall.
    vec3 wall(555, 555, 1);
    world->add(new quad(translation(vec3(555,    277.5f, 277.5f)) * rotation_y(-90) * scaling(wall), green)); // right wall, faces -x
    world->add(new quad(translation(vec3(0,      277.5f, 277.5f)) * rotation_y(90)  * scaling(wall), red));   // left wall,  faces +x
    world->add(new quad(translation(vec3(277.5f, 555,    277.5f)) * rotation_x(90)  * scaling(wall), white)); // ceiling,    faces -y
    world->add(new quad(translation(vec3(277.5f, 0,      277.5f)) * rotation_x(-90) * scaling(wall), white)); // floor,      faces +y
    world->add(new quad(translation(vec3(277.5f, 277.5f, 555   )) * rotation_y(180) * scaling(wall), white)); // back wall,  faces -z

    // Light: a 130 x 105 panel just below the ceiling, facing down. The same placement is
    // used for its copy in the lights list below, so the two always match.
    affine light_panel = translation(vec3(278, 554, 279.5f)) * rotation_x(90) * scaling(vec3(130, 105, 1));
    world->add(new quad(light_panel, light));

    // Box, 165 x 330 x 165: the unit cube is scaled, lifted so its corner sits at the
    // origin, turned 15 degrees about that corner, then moved (exactly the book's box).
    world->add(new transform(box(white),
        translation(vec3(265, 0, 295)) * rotation_y(15) * translation(vec3(82.5f, 165, 82.5f)) * scaling(vec3(165, 330, 165))));

    // Glass sphere, radius 90 (the unit sphere has diameter 1, so scale by 180)
    auto glass = new dielectric(1.5f);
    world->add(new transform(unit_sphere(glass), translation(vec3(190, 90, 190)) * scaling(vec3(180, 180, 180))));

    // Light Sources (separate objects from the world's, so nothing is deleted twice).
    // Only plain or placed quad / tri and plain spheres may go here, never transform(...).
    material* empty_material = nullptr;
    auto lights = new hittable_list();
    lights->add(new quad(light_panel, empty_material));
    lights->add(new sphere(point3(190, 90, 190), 90, empty_material));

    *d_world  = world;
    *d_lights = lights;
    d_mats[0] = red;
    d_mats[1] = white;
    d_mats[2] = green;
    d_mats[3] = light;
    d_mats[4] = glass;
    d_mats[5] = blue;
}

// Frees everything create_world made. Deleting a list deletes the objects in it
// (and a transform deletes what it wraps), so only lists + materials are listed here.
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
    const int num_mats = 6;   // must match the d_mats[...] entries in create_world
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

    cam.aspect_ratio      = 1.0f;
    cam.image_width       = 256;
    cam.samples_per_pixel = 5000;
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