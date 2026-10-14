// ============================================================================
// main.cu - random scene generator for the CUDA port of "The Rest of Your Life"
//
// Renders any number of random scenes and saves each one as
//     <out>/scene_00000.ppm   the image
//     <out>/scene_00000.txt   exactly what is in the scene (camera, materials,
//                             shapes, lights), so every image can be explained
//                             and re-created later
//
// Every scene comes from (seed, index). Scene 37 with seed 1 is always the
// same scene, so you can stop the program at any time and run it again: images
// that already exist are skipped.
//
// Build (x64 Native Tools Command Prompt):
//   nvcc -O3 -arch=native -std=c++17 -diag-suppress 550 -o rt_gen.exe main.cu
//
// Run:
//   rt_gen.exe --count 20
//   rt_gen.exe --count 500 --start 0 --size 256 --spp 1024 --out renders --seed 1
// ============================================================================

#include "rtweekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// 1. SETTINGS
// ============================================================================

// Pyramids need the `tri` class and pyramid() helper in quad.h.
// Set this to 0 if you have not added them (pyramids then become boxes).
#define USE_PYRAMIDS 1

// Number of objects standing in each scene (lights not counted).
constexpr int   MIN_OBJECTS = 3;
constexpr int   MAX_OBJECTS = 10;

// Shape mix for those objects. The rest are pyramids.
constexpr float P_SPHERE = 0.60f;
constexpr float P_BOX    = 0.20f;

// Material mix (same as the ARP: 40% diffuse, 30% metal, 30% glass).
constexpr float P_DIFFUSE        = 0.40f;
constexpr float P_METAL          = 0.30f;   // the rest are glass
constexpr float P_CHECKER_OBJECT = 0.15f;   // share of diffuse objects that get a checker pattern
constexpr float P_GLOWING_SPHERE = 0.05f;   // chance a sphere is a light instead
constexpr float FUZZ_MAX         = 0.30f;   // metal roughness 0 - 0.3
constexpr float IOR_MIN          = 1.3f;    // glass index of refraction 1.3 - 1.8
constexpr float IOR_MAX          = 1.8f;

// Camera field of view, degrees.
constexpr float FOV_MIN = 30.0f;
constexpr float FOV_MAX = 60.0f;

// Share of scenes that are an open-fronted room (Cornell-box style).
// The rest stand on an endless floor under a sky colour.
constexpr float P_ROOM = 0.5f;

// Path depth (the book's max_depth).
constexpr int MAX_DEPTH = 50;

// Scale each image so its average brightness lands near a target (see
// auto_exposure() below). This is the same as making every light in the scene
// brighter or dimmer, so the image stays physically correct; the factor used
// is written to the .txt file. Set to false to save the raw brightness.
constexpr bool AUTO_EXPOSURE = true;

// checker_texture objects are deleted by free_world_kernel, assuming
// lambertian(texture*) does NOT delete the texture it is given. If your
// lambertian(texture*) constructor sets owns_tex = true, set this to false,
// otherwise the texture is deleted twice.
constexpr bool FREE_CHECKER_TEXTURES = true;

// Fixed-size arrays for one scene description.
constexpr int MAX_MATS   = 32;
constexpr int MAX_SHAPES = 32;

// ============================================================================
// 2. SCENE DESCRIPTION (built on the CPU)
//
// A scene is first described with plain numbers in a SceneDesc. That struct is
// copied to the GPU, where build_scene() turns it into real objects with new.
// Keeping the description separate means the random choices happen on the CPU
// with a fixed, repeatable random generator, and the same description is
// written to the .txt file.
// ============================================================================

enum MatKind   { MAT_DIFFUSE, MAT_CHECKER, MAT_METAL, MAT_GLASS, MAT_LIGHT };
enum ShapeKind { SHAPE_SPHERE, SHAPE_QUAD, SHAPE_BOX, SHAPE_PYRAMID };

struct MatDesc {
    int   kind;   // MatKind
    color c1;     // albedo (diffuse, metal), first colour (checker), emitted light (light)
    color c2;     // second colour (checker)
    float param;  // fuzz (metal), index of refraction (glass), square size (checker)
};

struct ShapeDesc {
    int    kind;      // ShapeKind
    int    mat;       // index into SceneDesc::mats
    bool   is_light;  // also added to the lights list, so it is sampled directly
    point3 pos;       // sphere: centre | quad: corner Q | box, pyramid: centre of the base
    vec3   a;         // sphere: (radius,0,0) | quad: u | box: size | pyramid: (half width, height, 0)
    vec3   b;         // quad: v
    float  angle;     // box, pyramid: rotation about the y axis, degrees
};

struct SceneDesc {
    int       num_mats = 0;
    int       num_shapes = 0;
    MatDesc   mats[MAX_MATS];
    ShapeDesc shapes[MAX_SHAPES];

    // Only used on the CPU.
    bool   room = false;
    float  room_w = 0, room_h = 0, room_d = 0;
    point3 lookfrom, lookat;
    float  vfov = 40;
    color  background;
    float  exposure_target = 0.18f;
    int    num_objects = 0;
    int    num_lights = 0;
};

// Random numbers for scene building. std::mt19937 and std::seed_seq give the
// same numbers with every compiler; the std::uniform_*_distribution classes do
// not, so the conversions to float and int are written out here.
struct SceneRng {
    std::mt19937 gen;

    SceneRng(uint32_t seed, uint32_t index) {
        std::seed_seq seq{seed, index};
        gen.seed(seq);
    }
    float uniform() { return float(gen() >> 8) * (1.0f / 16777216.0f); }  // [0,1)
    float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
    int   range_int(int lo, int hi) { return lo + int(gen() % uint32_t(hi - lo + 1)); }
    bool  chance(float p) { return uniform() < p; }
};

// ---- Colours ----------------------------------------------------------------
// Colours are picked in HSV, like a colour picker, then squared to turn the
// displayed value into the linear albedo the renderer uses (write_color does
// the reverse with sqrt). Value stays at or below about 0.95, so albedo
// stays below about 0.9.

color hsv_to_linear(float h, float s, float v) {
    h = (h - std::floor(h)) * 6.0f;
    int   i = int(h) % 6;
    float f = h - std::floor(h);
    float p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (i) {
        case 0:  r = v; g = t; b = p; break;
        case 1:  r = q; g = v; b = p; break;
        case 2:  r = p; g = v; b = t; break;
        case 3:  r = p; g = q; b = v; break;
        case 4:  r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return color(r * r, g * g, b * b);
}

float luminance(const color& c) { return 0.2126f * c.x() + 0.7152f * c.y() + 0.0722f * c.z(); }

color random_albedo(SceneRng& rng)       { return hsv_to_linear(rng.uniform(), rng.range(0.25f, 0.85f), rng.range(0.45f, 0.92f)); }
color random_metal_color(SceneRng& rng)  { return hsv_to_linear(rng.uniform(), rng.range(0.00f, 0.35f), rng.range(0.80f, 0.95f)); }
color random_wall_color(SceneRng& rng)   { return hsv_to_linear(rng.uniform(), rng.range(0.00f, 0.50f), rng.range(0.60f, 0.90f)); }
color random_neutral_color(SceneRng& rng){ return hsv_to_linear(rng.uniform(), rng.range(0.00f, 0.12f), rng.range(0.70f, 0.90f)); }

// Light colour with luminance 1, so brightness is set separately.
color random_light_tint(SceneRng& rng) {
    color c;
    float pick = rng.uniform();
    if (pick < 0.45f)      c = hsv_to_linear(rng.range(0.06f, 0.12f), rng.range(0.0f, 0.30f), 1.0f);  // warm white
    else if (pick < 0.85f) c = hsv_to_linear(rng.range(0.55f, 0.62f), rng.range(0.0f, 0.20f), 1.0f);  // cool white
    else                   c = hsv_to_linear(rng.uniform(),           rng.range(0.4f, 0.75f), 1.0f);  // coloured
    return c / luminance(c);
}

// ---- Adding to the description ---------------------------------------------

int add_mat(SceneDesc& s, int kind, const color& c1, const color& c2 = color(0, 0, 0), float param = 0) {
    if (s.num_mats >= MAX_MATS) return s.num_mats - 1;  // never happens with the settings above
    MatDesc& m = s.mats[s.num_mats];
    m.kind = kind; m.c1 = c1; m.c2 = c2; m.param = param;
    return s.num_mats++;
}

void add_shape(SceneDesc& s, int kind, int mat, const point3& pos, const vec3& a,
               const vec3& b = vec3(0, 0, 0), float angle = 0) {
    if (s.num_shapes >= MAX_SHAPES) return;
    ShapeDesc& sh = s.shapes[s.num_shapes++];
    sh.kind = kind; sh.mat = mat; sh.pos = pos; sh.a = a; sh.b = b; sh.angle = angle;
    sh.is_light = (s.mats[mat].kind == MAT_LIGHT);
    if (sh.is_light) s.num_lights++;
}

// Light panel facing straight down: cross(u, v) = cross(+x, +z) = -y.
void add_ceiling_light(SceneDesc& s, float cx, float y, float cz, float sx, float sz, const color& emit) {
    int m = add_mat(s, MAT_LIGHT, emit);
    add_shape(s, SHAPE_QUAD, m, point3(cx - sx / 2, y, cz - sz / 2), vec3(sx, 0, 0), vec3(0, 0, sz));
}

// ---- Placing objects without overlaps ----------------------------------------

struct Footprint { float x, z, r; };

struct Region {
    bool  disk;                    // true: circle of `radius` around the origin
    float radius;
    float xmin, xmax, zmin, zmax;  // otherwise: rectangle
};

// Finds a spot for a footprint of radius r that stays inside the region and
// keeps a small gap to everything already placed. Returns false if the scene
// is too full.
bool find_spot(SceneRng& rng, const Region& reg, const std::vector<Footprint>& used,
               float r, float& x, float& z) {
    for (int attempt = 0; attempt < 200; attempt++) {
        if (reg.disk) {
            float ang = rng.range(0, 2 * pi);
            float rad = (reg.radius - r) * std::sqrt(rng.uniform());
            if (rad < 0) return false;
            x = rad * std::cos(ang);
            z = rad * std::sin(ang);
        } else {
            if (reg.xmax - reg.xmin < 2 * r || reg.zmax - reg.zmin < 2 * r) return false;
            x = rng.range(reg.xmin + r, reg.xmax - r);
            z = rng.range(reg.zmin + r, reg.zmax - r);
        }
        bool clear = true;
        for (const Footprint& f : used) {
            float dx = x - f.x, dz = z - f.z;
            float min_d = r + f.r + 0.15f;
            if (dx * dx + dz * dz < min_d * min_d) { clear = false; break; }
        }
        if (clear) return true;
    }
    return false;
}

// ---- Objects -----------------------------------------------------------------

int random_object_material(SceneDesc& s, SceneRng& rng, bool may_glow) {
    if (may_glow && rng.chance(P_GLOWING_SPHERE))
        return add_mat(s, MAT_LIGHT, random_light_tint(rng) * rng.range(3.0f, 8.0f));

    float p = rng.uniform();
    if (p < P_DIFFUSE) {
        if (rng.chance(P_CHECKER_OBJECT)) {
            color c = random_albedo(rng);
            return add_mat(s, MAT_CHECKER, c, c * rng.range(0.1f, 0.4f), rng.range(0.15f, 0.4f));
        }
        return add_mat(s, MAT_DIFFUSE, random_albedo(rng));
    }
    if (p < P_DIFFUSE + P_METAL)
        return add_mat(s, MAT_METAL, random_metal_color(rng), color(0, 0, 0), rng.range(0.0f, FUZZ_MAX));
    return add_mat(s, MAT_GLASS, color(1, 1, 1), color(0, 0, 0), rng.range(IOR_MIN, IOR_MAX));
}

// Everything stays below MAX_OBJECT_TOP, and lights hang above MIN_LIGHT_BOTTOM,
// so objects and lights never touch.
constexpr float MAX_OBJECT_TOP   = 3.0f;
constexpr float MIN_LIGHT_BOTTOM = 3.5f;

void add_objects(SceneDesc& s, SceneRng& rng, const Region& reg, std::vector<Footprint>& used) {
    int wanted = rng.range_int(MIN_OBJECTS, MAX_OBJECTS);
    for (int n = 0; n < wanted; n++) {
        float pick = rng.uniform();
        int kind = (pick < P_SPHERE) ? SHAPE_SPHERE : (pick < P_SPHERE + P_BOX) ? SHAPE_BOX : SHAPE_PYRAMID;
#if !USE_PYRAMIDS
        if (kind == SHAPE_PYRAMID) kind = SHAPE_BOX;
#endif
        float x, z;
        if (kind == SHAPE_SPHERE) {
            int m = random_object_material(s, rng, true);
            bool glowing = (s.mats[m].kind == MAT_LIGHT);
            float r = glowing ? rng.range(0.25f, 0.5f) : rng.range(0.4f, 1.3f);  // small, so it doesn't fill the frame
            if (!find_spot(rng, reg, used, r, x, z)) { s.num_mats--; continue; }  // no room: drop the unused material
            float y = r;                                       // resting on the floor
            if (rng.chance(0.2f)) {                            // or floating
                float lift_max = MAX_OBJECT_TOP - 2 * r;
                if (lift_max > 0.3f) y = r + rng.range(0.3f, lift_max);
            }
            add_shape(s, SHAPE_SPHERE, m, point3(x, y, z), vec3(r, 0, 0));
            used.push_back({x, z, r});
        } else if (kind == SHAPE_BOX) {
            vec3 size(rng.range(0.6f, 1.8f), rng.range(0.6f, 2.4f), rng.range(0.6f, 1.8f));
            float r = 0.5f * std::sqrt(size.x() * size.x() + size.z() * size.z());
            if (!find_spot(rng, reg, used, r, x, z)) continue;
            int m = random_object_material(s, rng, false);
            // Lifted 0.002 so the bottom face is not exactly on the floor plane.
            add_shape(s, SHAPE_BOX, m, point3(x, 0.002f, z), size, vec3(0, 0, 0), rng.range(0, 90));
            used.push_back({x, z, r});
        } else {
            float hw = rng.range(0.4f, 1.0f), h = rng.range(0.8f, 2.2f);
            float r = hw * 1.4143f;
            if (!find_spot(rng, reg, used, r, x, z)) continue;
            int m = random_object_material(s, rng, false);
            add_shape(s, SHAPE_PYRAMID, m, point3(x, 0.002f, z), vec3(hw, h, 0), vec3(0, 0, 0), rng.range(0, 90));
            used.push_back({x, z, r});
        }
        s.num_objects++;
    }
}

// ---- Layout 1: open-fronted room, camera looking in through the open side ----

void make_room(SceneDesc& s, SceneRng& rng) {
    float W = rng.range(6.5f, 9.0f), H = rng.range(5.5f, 7.5f), D = rng.range(6.5f, 9.0f);
    s.room = true; s.room_w = W; s.room_h = H; s.room_d = D;
    s.background = color(0, 0, 0);

    // Walls: half the rooms get Cornell-style coloured side walls.
    color left_c, right_c;
    if (rng.chance(0.5f)) {
        float hue = rng.uniform();
        left_c  = hsv_to_linear(hue,                          rng.range(0.5f, 0.8f), rng.range(0.55f, 0.8f));
        right_c = hsv_to_linear(hue + rng.range(0.25f, 0.6f), rng.range(0.5f, 0.8f), rng.range(0.55f, 0.8f));
    } else {
        left_c = random_wall_color(rng);
        right_c = random_wall_color(rng);
    }
    int m_left  = add_mat(s, MAT_DIFFUSE, left_c);
    int m_right = add_mat(s, MAT_DIFFUSE, right_c);
    int m_back  = add_mat(s, MAT_DIFFUSE, rng.chance(0.6f) ? random_neutral_color(rng) : random_wall_color(rng));
    int m_ceil  = add_mat(s, MAT_DIFFUSE, random_neutral_color(rng));
    int m_floor;
    if (rng.chance(0.4f)) {
        color c = random_wall_color(rng);
        m_floor = add_mat(s, MAT_CHECKER, c, c * rng.range(0.1f, 0.4f), rng.range(0.5f, 1.5f));
    } else {
        m_floor = add_mat(s, MAT_DIFFUSE, random_wall_color(rng));
    }

    float x0 = -W / 2, z0 = -D / 2;
    add_shape(s, SHAPE_QUAD, m_floor, point3(x0, 0, z0), vec3(W, 0, 0), vec3(0, 0, D));
    add_shape(s, SHAPE_QUAD, m_ceil,  point3(x0, H, z0), vec3(W, 0, 0), vec3(0, 0, D));
    add_shape(s, SHAPE_QUAD, m_back,  point3(x0, 0, z0), vec3(W, 0, 0), vec3(0, H, 0));
    add_shape(s, SHAPE_QUAD, m_left,  point3(x0, 0, z0), vec3(0, 0, D), vec3(0, H, 0));
    add_shape(s, SHAPE_QUAD, m_right, point3(-x0, 0, z0), vec3(0, 0, D), vec3(0, H, 0));

    // Ceiling light, just below the ceiling (like the Cornell box's 554 vs 555).
    // Brightness is set from the share of the ceiling it covers, so a small
    // panel is brighter than a big one and the room gets similar total light.
    // Panels are fairly large on purpose: light that reaches a wall through
    // glass or metal (a caustic) can't be aimed at by the light sampling, so it
    // shows up as white speckles ("fireflies"); a bigger, dimmer panel makes
    // those much weaker for the same amount of light.
    float lw = W * rng.range(0.3f, 0.5f), ld = D * rng.range(0.3f, 0.5f);
    float lx = rng.range(-0.15f, 0.15f) * W, lz = rng.range(-0.15f, 0.15f) * D;
    float panel_strength = rng.range(1.5f, 2.7f) / (lw * ld / (W * D));
    add_ceiling_light(s, lx, H - 0.01f, lz, lw, ld, random_light_tint(rng) * panel_strength);

    // Sometimes a second light: a hanging bulb with 30-80% of the panel's power.
    if (rng.chance(0.3f)) {
        float r = rng.range(0.2f, 0.4f);
        float y = rng.range(MIN_LIGHT_BOTTOM + r + 0.2f, H - 0.5f - r);  // H >= 5.5, so this range is never empty
        float x = rng.range(-W / 2 + 1, W / 2 - 1), z = rng.range(-D / 2 + 1, D / 2 - 1);
        float strength = rng.range(0.3f, 0.8f) * panel_strength * (lw * ld) / (4 * pi * r * r);
        int m = add_mat(s, MAT_LIGHT, random_light_tint(rng) * strength);
        add_shape(s, SHAPE_SPHERE, m, point3(x, y, z), vec3(r, 0, 0));
    }

    // Objects stand on the floor, away from the walls and the open front.
    Region reg = {false, 0, -W / 2 + 0.4f, W / 2 - 0.4f, -D / 2 + 0.4f, D / 2 - 1.2f};
    std::vector<Footprint> used;
    add_objects(s, rng, reg, used);

    // Camera outside the open front (+z side), framing the opening like the
    // Cornell box camera. zoom < 1 crops in a little, > 1 shows some outside.
    s.vfov = rng.range(FOV_MIN, FOV_MAX);
    float half_open = 0.5f * std::max(W, H);
    float zoom = rng.range(0.85f, 1.05f);
    float dist_to_opening = half_open / std::tan(degrees_to_radians(s.vfov / 2)) * zoom;
    float az = degrees_to_radians(rng.range(-15.0f, 15.0f));
    float el = degrees_to_radians(rng.range(-3.0f, 10.0f));
    s.lookat = point3(0, H * rng.range(0.4f, 0.5f), 0);
    vec3 dir(std::sin(az) * std::cos(el), std::sin(el), std::cos(az) * std::cos(el));
    s.lookfrom = s.lookat + (dist_to_opening + D / 2) * dir;
}

// ---- Layout 2: objects on an endless floor under a sky colour -----------------

void make_open(SceneDesc& s, SceneRng& rng) {
    s.room = false;

    // Background: usually a sky colour (it lights the scene too), sometimes black.
    if (rng.chance(0.7f)) {
        color sky = rng.chance(0.6f)
                  ? hsv_to_linear(rng.range(0.52f, 0.65f), rng.range(0.1f, 0.5f), 1.0f)   // blue-ish
                  : hsv_to_linear(rng.uniform(),           rng.range(0.0f, 0.3f), 1.0f);  // any tint
        s.background = (sky / luminance(sky)) * rng.range(0.15f, 1.0f);
    } else {
        s.background = color(0, 0, 0);
    }

    // Ground: a 200 x 200 square, which looks endless from the camera.
    int m_floor;
    if (rng.chance(0.5f)) {
        color c = random_wall_color(rng);
        m_floor = add_mat(s, MAT_CHECKER, c, c * rng.range(0.1f, 0.4f), rng.range(0.6f, 2.0f));
    } else {
        m_floor = add_mat(s, MAT_DIFFUSE, random_wall_color(rng));
    }
    add_shape(s, SHAPE_QUAD, m_floor, point3(-100, 0, -100), vec3(200, 0, 0), vec3(0, 0, 200));

    // 1 or 2 lights above the objects: a panel facing down, or a glowing ball.
    // "power" ~ brightness x area, so lights of different sizes look alike.
    float place_r = rng.range(3.5f, 5.5f);
    int num_lights = rng.chance(0.4f) ? 2 : 1;
    bool black_sky = (s.background.x() == 0 && s.background.y() == 0 && s.background.z() == 0);
    Region light_reg = {false, 0, -place_r - 1, place_r + 1, -place_r - 1, place_r + 1};
    std::vector<Footprint> light_spots;  // keeps two lights from overlapping
    for (int i = 0; i < num_lights; i++) {
        float power = rng.range(50.0f, 110.0f) / num_lights;
        if (black_sky) power *= 2;  // the lights are the only light
        color tint = random_light_tint(rng);
        bool panel = rng.chance(0.5f);
        float sx = rng.range(1.5f, 4.0f), sz = rng.range(1.5f, 4.0f);  // panel size
        float r  = rng.range(0.35f, 0.7f);                              // ball radius
        float footprint = panel ? 0.5f * std::sqrt(sx * sx + sz * sz) : r;
        float x, z;
        if (!find_spot(rng, light_reg, light_spots, footprint, x, z)) continue;
        light_spots.push_back({x, z, footprint});
        if (panel) {
            add_ceiling_light(s, x, rng.range(5.0f, 8.0f), z, sx, sz, tint * (power / (sx * sz)));
        } else {
            float y = rng.range(MIN_LIGHT_BOTTOM + r, 6.5f);
            int m = add_mat(s, MAT_LIGHT, tint * (power / (pi * r * r)));
            add_shape(s, SHAPE_SPHERE, m, point3(x, y, z), vec3(r, 0, 0));
        }
    }

    Region reg = {true, place_r, 0, 0, 0, 0};
    std::vector<Footprint> used;
    add_objects(s, rng, reg, used);

    // Camera on a circle around the objects, looking slightly down at them.
    s.vfov = rng.range(FOV_MIN, FOV_MAX);
    float dist = place_r / std::tan(degrees_to_radians(s.vfov / 2)) * rng.range(0.95f, 1.3f);
    float az = degrees_to_radians(rng.range(0.0f, 360.0f));
    float el = degrees_to_radians(rng.range(8.0f, 35.0f));
    s.lookat = point3(0, rng.range(0.5f, 1.2f), 0);
    vec3 dir(std::sin(az) * std::cos(el), std::sin(el), std::cos(az) * std::cos(el));
    s.lookfrom = s.lookat + dist * dir;
}

void generate_scene(SceneDesc& s, SceneRng& rng) {
    if (rng.chance(P_ROOM)) make_room(s, rng);
    else                    make_open(s, rng);
    s.exposure_target = 0.18f * rng.range(0.7f, 1.4f);  // a little brightness variety
}

// ============================================================================
// 3. BUILDING THE SCENE ON THE GPU
//
// build_scene() runs in a <<<1,1>>> kernel and creates the objects with new,
// exactly like create_world in your Cornell box main.cu. GpuScene remembers
// every pointer that free_world_kernel has to delete.
// ============================================================================

struct GpuScene {
    hittable*      world;    // what rays are traced against (a BVH over `objects`)
    hittable_list* objects;  // owns every object
    hittable_list* lights;   // owns separate copies of the light shapes
    material*      mats[MAX_MATS];
    texture*       texs[MAX_MATS];
    int            num_mats;
    int            num_texs;
};

__device__ void build_scene(const SceneDesc& d, GpuScene& g) {
    // Materials
    g.num_mats = d.num_mats;
    g.num_texs = 0;
    for (int i = 0; i < d.num_mats; i++) {
        const MatDesc& m = d.mats[i];
        switch (m.kind) {
            case MAT_CHECKER: {
                texture* tex = new checker_texture(m.param, m.c1, m.c2);
                g.texs[g.num_texs++] = tex;
                g.mats[i] = new lambertian(tex);
                break;
            }
            case MAT_METAL: g.mats[i] = new metal(m.c1, m.param);      break;
            case MAT_GLASS: g.mats[i] = new dielectric(m.param);       break;
            case MAT_LIGHT: g.mats[i] = new diffuse_light(m.c1);       break;
            default:        g.mats[i] = new lambertian(m.c1);          break;
        }
    }

    // Shapes
    g.objects = new hittable_list();
    g.lights  = new hittable_list();
    for (int i = 0; i < d.num_shapes; i++) {
        const ShapeDesc& s = d.shapes[i];
        material* mat = g.mats[s.mat];

        switch (s.kind) {
            case SHAPE_SPHERE: {
                float r = s.a.x();
                g.objects->add(new sphere(s.pos, r, mat));
                if (s.is_light) g.lights->add(new sphere(s.pos, r, nullptr));
                break;
            }
            case SHAPE_QUAD: {
                g.objects->add(new quad(s.pos, s.a, s.b, mat));
                if (s.is_light) g.lights->add(new quad(s.pos, s.a, s.b, nullptr));
                break;
            }
            case SHAPE_BOX: {
                // Built around the origin with its base on y = 0, then rotated and moved.
                float hx = s.a.x() / 2, hz = s.a.z() / 2;
                hittable* obj = box(point3(-hx, 0, -hz), point3(hx, s.a.y(), hz), mat);
                obj = new rotate_y(obj, s.angle);
                obj = new translate(obj, s.pos);
                g.objects->add(obj);
                break;
            }
            case SHAPE_PYRAMID: {
#if USE_PYRAMIDS
                hittable* obj = pyramid(point3(0, 0, 0), s.a.x(), s.a.y(), mat);
                obj = new rotate_y(obj, s.angle);
                obj = new translate(obj, s.pos);
                g.objects->add(obj);
#endif
                break;
            }
        }
    }

    // The BVH only points at the objects; `objects` still owns them.
    g.world = new bvh_node(*g.objects);
}

__global__ void build_world_kernel(const SceneDesc* desc, GpuScene* g,
                                   hittable** d_world, hittable** d_lights) {
    build_scene(*desc, *g);
    *d_world = g->world;
    *d_lights = g->lights;
}

__global__ void free_world_kernel(GpuScene* g) {
    delete static_cast<bvh_node*>(g->world);  // BVH nodes only
    delete g->objects;                        // every object (and through box/pyramid lists, their faces)
    delete g->lights;                         // the light copies
    for (int i = 0; i < g->num_mats; i++) delete g->mats[i];
    if (FREE_CHECKER_TEXTURES)
        for (int i = 0; i < g->num_texs; i++) delete g->texs[i];
}

// ============================================================================
// 4. RENDERING AND SAVING
// ============================================================================

#define GEN_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {          \
    std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n", cudaGetErrorString(e_),       \
                 __FILE__, __LINE__); std::exit(1); } } while (0)

// Like render_init, but with a seed per image so each image gets its own noise.
__global__ void seed_rng_kernel(int width, int height, unsigned long long seed, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int pixel_index = j * width + i;
    curand_init(seed, pixel_index, 0, &states[pixel_index]);
}

// GPU buffers, allocated once and reused for every image.
struct RenderBuffers {
    curandState* d_rand  = nullptr;
    color*       d_accum = nullptr;
    std::vector<color> image;  // linear colour, averaged over samples
};

// Same loop as camera::render, but keeps the result in memory instead of
// writing it to std::cout.
void render_image(const camera& cam, hittable** d_world, hittable** d_lights,
                  RenderBuffers& buf, unsigned long long gpu_seed) {
    int w = cam.image_width, h = cam.image_height;
    dim3 threads(8, 8);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    seed_rng_kernel<<<blocks, threads>>>(w, h, gpu_seed, buf.d_rand);
    GEN_CHECK(cudaGetLastError());
    GEN_CHECK(cudaMemset(buf.d_accum, 0, sizeof(color) * w * h));

    // One launch per stratum, so no single launch runs long enough to trigger
    // the Windows driver timeout.
    for (int s_j = 0; s_j < cam.sqrt_spp; s_j++) {
        for (int s_i = 0; s_i < cam.sqrt_spp; s_i++) {
            render_kernel<<<blocks, threads>>>(cam, d_world, d_lights, buf.d_rand, buf.d_accum, s_i, s_j);
            GEN_CHECK(cudaGetLastError());
        }
    }
    GEN_CHECK(cudaDeviceSynchronize());

    buf.image.resize(size_t(w) * h);
    GEN_CHECK(cudaMemcpy(buf.image.data(), buf.d_accum, sizeof(color) * w * h, cudaMemcpyDeviceToHost));
    for (color& c : buf.image) c = cam.pixel_samples_scale * c;
}

// Scale factor that brings the image's typical brightness to `target`.
// "Typical" is the geometric mean of pixel luminance (the usual "key" used in
// photographic tone mapping), skipping pure-black pixels such as a black
// background, so a few very bright pixels (visible lights) don't dominate.
// Multiplying the image by a constant is the same as multiplying every light
// and the background by it, so the result is still a correct render.
float auto_exposure(const std::vector<color>& img, float target) {
    double log_sum = 0;
    size_t n = 0;
    for (const color& c : img) {
        float y = luminance(c);
        if (y > 1e-3f) { log_sum += std::log(double(y)); n++; }
    }
    if (n < img.size() / 20) return 1.0f;  // (almost) all black: leave it alone
    float key = float(std::exp(log_sum / double(n)));
    return std::min(8.0f, std::max(0.125f, target / key));
}

// Binary PPM (P6): same format family as the book's text PPM (P3) and opens
// in the same viewers, but about 5x smaller. The byte conversion is the one
// write_color uses (gamma 2, clamp to 0.999).
unsigned char to_byte(float x) {
    if (!(x > 0)) return 0;  // also catches NaN
    x = std::sqrt(x);
    if (x > 0.999f) x = 0.999f;
    return (unsigned char)(int(256 * x));
}

bool write_ppm(const std::filesystem::path& path, const std::vector<color>& img, int w, int h, float scale) {
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) return false;
        out << "P6\n" << w << ' ' << h << "\n255\n";
        std::vector<unsigned char> bytes(size_t(w) * h * 3);
        for (size_t i = 0; i < img.size(); i++) {
            color c = scale * img[i];
            bytes[3 * i + 0] = to_byte(c.x());
            bytes[3 * i + 1] = to_byte(c.y());
            bytes[3 * i + 2] = to_byte(c.z());
        }
        out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        if (!out) return false;
    }
    // Written under a temporary name first, so a crash never leaves a
    // half-written image that would be skipped on the next run.
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

const char* mat_name(int kind) {
    switch (kind) {
        case MAT_DIFFUSE: return "diffuse";
        case MAT_CHECKER: return "checker";
        case MAT_METAL:   return "metal";
        case MAT_GLASS:   return "glass";
        default:          return "light";
    }
}

void write_description(const std::filesystem::path& path, const SceneDesc& s, int index,
                       unsigned seed, int width, int spp, float exposure) {
    std::ofstream out(path);
    out << "index " << index << "\nseed " << seed << "\n";
    out << "size " << width << " x " << width << "\nsamples_per_pixel " << spp << "\n";
    if (s.room) out << "layout room  width " << s.room_w << "  height " << s.room_h << "  depth " << s.room_d << "\n";
    else        out << "layout open\n";
    out << "camera lookfrom " << s.lookfrom << "  lookat " << s.lookat << "  vfov " << s.vfov << "\n";
    out << "background " << s.background << "\n";
    out << "exposure_scale " << exposure << "   (every light and the background were multiplied by this)\n";
    out << "objects " << s.num_objects << "  lights " << s.num_lights << "\n\n";

    out << "materials\n";
    for (int i = 0; i < s.num_mats; i++) {
        const MatDesc& m = s.mats[i];
        out << "  m" << i << " " << mat_name(m.kind);
        switch (m.kind) {
            case MAT_DIFFUSE: out << "  albedo " << m.c1; break;
            case MAT_CHECKER: out << "  colors " << m.c1 << " / " << m.c2 << "  square " << m.param; break;
            case MAT_METAL:   out << "  albedo " << m.c1 << "  fuzz " << m.param; break;
            case MAT_GLASS:   out << "  ior " << m.param; break;
            default:          out << "  emit " << m.c1; break;
        }
        out << "\n";
    }

    out << "\nshapes\n";
    for (int i = 0; i < s.num_shapes; i++) {
        const ShapeDesc& sh = s.shapes[i];
        out << "  ";
        switch (sh.kind) {
            case SHAPE_SPHERE:  out << "sphere   center " << sh.pos << "  radius " << sh.a.x(); break;
            case SHAPE_QUAD:    out << "quad     Q " << sh.pos << "  u " << sh.a << "  v " << sh.b; break;
            case SHAPE_BOX:     out << "box      base_center " << sh.pos << "  size " << sh.a << "  rotate_y " << sh.angle; break;
            case SHAPE_PYRAMID: out << "pyramid  base_center " << sh.pos << "  half_width " << sh.a.x()
                                    << "  height " << sh.a.y() << "  rotate_y " << sh.angle; break;
        }
        out << "  material m" << sh.mat << (sh.is_light ? "  (light)" : "") << "\n";
    }
}

// ============================================================================
// 5. MAIN
// ============================================================================

struct Options {
    int         count = 10;
    int         start = 0;
    int         size  = 256;
    int         spp   = 1024;
    unsigned    seed  = 1;
    std::string out   = "renders";
};

void usage() {
    std::fprintf(stderr,
        "usage: rt_gen [--count N] [--start N] [--size N] [--spp N] [--seed N] [--out DIR]\n"
        "  --count  number of images to make            (default 10)\n"
        "  --start  index of the first image            (default 0)\n"
        "  --size   image width and height in pixels    (default 256)\n"
        "  --spp    samples per pixel, best a square    (default 1024)\n"
        "  --seed   changes every scene                 (default 1)\n"
        "  --out    output folder                       (default renders)\n");
}

bool parse_options(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (i + 1 >= argc) { usage(); return false; }
        std::string v = argv[++i];
        if      (a == "--count") o.count = std::atoi(v.c_str());
        else if (a == "--start") o.start = std::atoi(v.c_str());
        else if (a == "--size")  o.size  = std::atoi(v.c_str());
        else if (a == "--spp")   o.spp   = std::atoi(v.c_str());
        else if (a == "--seed")  o.seed  = unsigned(std::strtoul(v.c_str(), nullptr, 10));
        else if (a == "--out")   o.out   = v;
        else { usage(); return false; }
    }
    if (o.count < 1 || o.start < 0 || o.size < 8 || o.spp < 1) { usage(); return false; }
    return true;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_options(argc, argv, opt)) return 1;

    namespace fs = std::filesystem;
    fs::create_directories(opt.out);

    // Same stack limit as before; a bigger heap for the objects made with new.
    GEN_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    GEN_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t(64) << 20));

    // Camera settings shared by every image.
    camera cam;
    cam.aspect_ratio      = 1.0f;
    cam.image_width       = opt.size;
    cam.samples_per_pixel = opt.spp;
    cam.max_depth         = MAX_DEPTH;
    cam.vup               = vec3(0, 1, 0);
    cam.defocus_angle     = 0;
    cam.initialize();
    int w = cam.image_width, h = cam.image_height;
    int real_spp = cam.sqrt_spp * cam.sqrt_spp;

    // GPU memory, allocated once.
    RenderBuffers buf;
    SceneDesc* d_desc;
    GpuScene*  d_scene;
    hittable** d_world;
    hittable** d_lights;
    GEN_CHECK(cudaMalloc(&buf.d_rand, sizeof(curandState) * w * h));
    GEN_CHECK(cudaMalloc(&buf.d_accum, sizeof(color) * w * h));
    GEN_CHECK(cudaMalloc(&d_desc, sizeof(SceneDesc)));
    GEN_CHECK(cudaMalloc(&d_scene, sizeof(GpuScene)));
    GEN_CHECK(cudaMalloc(&d_world, sizeof(hittable*)));
    GEN_CHECK(cudaMalloc(&d_lights, sizeof(hittable*)));

    std::printf("Making %d images (%d x %d, %d samples per pixel) in \"%s\"\n",
                opt.count, w, h, real_spp, opt.out.c_str());
    if (real_spp != opt.spp)
        std::printf("Note: %d is not a square number, so %d samples are used.\n", opt.spp, real_spp);

    auto t_all = std::chrono::steady_clock::now();
    int done = 0;
    double render_seconds = 0;

    for (int k = 0; k < opt.count; k++) {
        int index = opt.start + k;
        char name[32];
        std::snprintf(name, sizeof(name), "scene_%05d", index);
        fs::path ppm_path = fs::path(opt.out) / (std::string(name) + ".ppm");
        fs::path txt_path = fs::path(opt.out) / (std::string(name) + ".txt");

        if (fs::exists(ppm_path)) {
            std::printf("[%d/%d] %s already exists, skipped\n", k + 1, opt.count, name);
            continue;
        }

        auto t0 = std::chrono::steady_clock::now();

        // 1. Describe the scene on the CPU.
        SceneRng rng(opt.seed, uint32_t(index));
        SceneDesc desc;
        generate_scene(desc, rng);

        // 2. Build it on the GPU.
        GEN_CHECK(cudaMemcpy(d_desc, &desc, sizeof(SceneDesc), cudaMemcpyHostToDevice));
        build_world_kernel<<<1, 1>>>(d_desc, d_scene, d_world, d_lights);
        GEN_CHECK(cudaGetLastError());
        GEN_CHECK(cudaDeviceSynchronize());

        // 3. Render it.
        cam.lookfrom   = desc.lookfrom;
        cam.lookat     = desc.lookat;
        cam.vfov       = desc.vfov;
        cam.background = desc.background;
        cam.focus_dist = (desc.lookfrom - desc.lookat).length();
        cam.initialize();
        unsigned long long gpu_seed = (unsigned long long)opt.seed * 1000003ULL + (unsigned long long)index;
        render_image(cam, d_world, d_lights, buf, gpu_seed);

        // 4. Delete the scene's objects.
        free_world_kernel<<<1, 1>>>(d_scene);
        GEN_CHECK(cudaGetLastError());
        GEN_CHECK(cudaDeviceSynchronize());

        // 5. Save.
        float exposure = AUTO_EXPOSURE ? auto_exposure(buf.image, desc.exposure_target) : 1.0f;
        if (!write_ppm(ppm_path, buf.image, w, h, exposure)) {
            std::fprintf(stderr, "Could not write %s\n", ppm_path.string().c_str());
            return 1;
        }
        write_description(txt_path, desc, index, opt.seed, w, real_spp, exposure);

        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        render_seconds += secs;
        done++;
        double eta_min = render_seconds / done * (opt.count - k - 1) / 60.0;
        std::printf("[%d/%d] %s  %-4s  %2d objects  %d light%s  %5.1f s   (about %.0f min left)\n",
                    k + 1, opt.count, name, desc.room ? "room" : "open", desc.num_objects,
                    desc.num_lights, desc.num_lights == 1 ? " " : "s", secs, eta_min);
    }

    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    std::printf("Done: %d new images in %.1f s\n", done, total);

    GEN_CHECK(cudaFree(buf.d_rand));
    GEN_CHECK(cudaFree(buf.d_accum));
    GEN_CHECK(cudaFree(d_desc));
    GEN_CHECK(cudaFree(d_scene));
    GEN_CHECK(cudaFree(d_world));
    GEN_CHECK(cudaFree(d_lights));
    return 0;
}
