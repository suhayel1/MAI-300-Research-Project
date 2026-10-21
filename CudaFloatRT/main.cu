// ============================================================================
// main.cu - dataset generator: a "complexity ladder" of scenes
//
// Level 1 is the simplest scene (one diffuse sphere in the middle of a closed
// Cornell-style room, camera just inside the front wall). Each level after it switches on one more factor: more shapes,
// more materials, random placement, full transforms, more objects, a moving
// camera, a varying light and room. The levels are the rows of LEVELS below;
// edit that table to change, reorder or add levels.
//
// Every scene gets its own folder:
//     <out>/level_04_one_object_placed/scene_00012/
//         image.npy     linear HDR radiance, (height, width, 3) float32
//         gbuffer.npy   18-channel G-buffer, (height, width, 18) float32
//         scene.json    everything about the scene (camera, light, room, objects,
//                       materials, placements, train/val/test split)
//         image.ppm     8-bit preview (skip with --no-preview)
// and a line is added to <out>/index.csv.
//
// A scene is fixed by (seed, level, index), so runs can be stopped and restarted:
// scenes whose folder is already complete are skipped.
//
// Build (x64 Native Tools Command Prompt):
//   nvcc -O3 -arch=native -std=c++17 -diag-suppress 550 -o rt_dataset.exe main.cu
// Run:
//   rt_dataset.exe --list
//   rt_dataset.exe --level 1 --count 50
//   rt_dataset.exe --level 7 --count 1000 --start 0 --size 256 --spp 1024 --out dataset
//
// Needs from your headers: unit_box (quad.h), unit_sphere (sphere.h),
// unit_square_pyramid and unit_triangle_pyramid (tri.h), quad(const affine&, material*),
// transform (transform.h), gbuffer_kernel / render_kernel / GBUFFER_CHANNELS (camera.h),
// write_npy (npy.h).
// ============================================================================

#include "rtweekend.h"

#include "bvh.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "npy.h"
#include "quad.h"
#include "sphere.h"
#include "texture.h"
#include "transform.h"
#include "tri.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// 1. THE LEVELS
// ============================================================================

enum ShapeKind    { SHAPE_SPHERE = 0, SHAPE_BOX = 1, SHAPE_SQUARE_PYRAMID = 2, SHAPE_TRIANGLE_PYRAMID = 3 };
enum RotationMode { ROT_NONE = 0, ROT_Y = 1, ROT_ANY = 2 };

struct LevelSpec {
    const char* name;
    int   min_objects, max_objects;
    bool  any_shape;          // false: spheres only
    bool  any_material;       // false: diffuse only
    bool  random_position;    // false: one object in the middle of the floor
    float min_size, max_size; // overall size of each object, before stretching (world units)
    bool  stretch;            // different scale along each axis
    int   rotation;           // RotationMode: none, about y only, any orientation
    float max_shear;          // 0 = no shear
    bool  random_camera;      // false: fixed Cornell-box view
    bool  random_light;       // false: fixed ceiling light
    bool  random_room;        // false: fixed room size and classic red/green/white walls
};

const LevelSpec LEVELS[] = {
//   name                         objects  shapes materials position  size          stretch rotation  shear  camera light  room
    {"one_diffuse_sphere",         1, 1,   false, false,    false,    2.5f, 3.5f,   false,  ROT_NONE, 0.0f,  false, false, false},
    {"one_object_any_shape",       1, 1,   true,  false,    false,    2.5f, 3.5f,   false,  ROT_NONE, 0.0f,  false, false, false},
    {"one_object_any_material",    1, 1,   true,  true,     false,    2.5f, 3.5f,   false,  ROT_NONE, 0.0f,  false, false, false},
    {"one_object_placed",          1, 1,   true,  true,     true,     2.0f, 3.5f,   false,  ROT_Y,    0.0f,  false, false, false},
    {"one_object_transformed",     1, 1,   true,  true,     true,     2.0f, 3.5f,   true,   ROT_ANY,  0.35f, false, false, false},
    {"two_to_three_objects",       2, 3,   true,  true,     true,     1.4f, 2.8f,   true,   ROT_ANY,  0.35f, false, false, false},
    {"three_to_ten_objects",       3, 10,  true,  true,     true,     1.0f, 2.2f,   true,   ROT_ANY,  0.35f, false, false, false},
    {"random_camera",              3, 10,  true,  true,     true,     1.0f, 2.2f,   true,   ROT_ANY,  0.35f, true,  false, false},
    {"random_light_and_room",      3, 10,  true,  true,     true,     1.0f, 2.2f,   true,   ROT_ANY,  0.35f, true,  true,  true },
};
constexpr int NUM_LEVELS = int(sizeof(LEVELS) / sizeof(LEVELS[0]));

// Other settings shared by all levels.
constexpr float P_DIFFUSE        = 0.40f;  // material mix when any_material is on (ARP: 0.4 / 0.3 / 0.3)
constexpr float P_METAL          = 0.30f;  // the rest are glass
constexpr float P_CHECKER_OBJECT = 0.15f;  // share of diffuse objects with a checker pattern
constexpr float FUZZ_MAX         = 0.30f;  // metal roughness 0 - 0.3
constexpr float IOR_MIN          = 1.3f;   // glass index of refraction 1.3 - 1.8
constexpr float IOR_MAX          = 1.8f;
constexpr float FOV_MIN          = 30.0f;  // random camera field of view, degrees
constexpr float FOV_MAX          = 60.0f;
constexpr float FIXED_FOV        = 55.0f;  // the fixed camera's field of view
constexpr int   MAX_DEPTH        = 50;

// Train / validation / test shares, chosen per scene from (seed, level, index).
constexpr int SPLIT_TRAIN_PERCENT = 80;
constexpr int SPLIT_VAL_PERCENT   = 10;   // the rest is test

// The fixed room (levels without random_room) and its fixed light and camera.
constexpr float ROOM_W = 8.0f, ROOM_H = 6.5f, ROOM_D = 12.0f;   // deep, so the camera sees the whole room

// Fixed-size arrays for one scene.
constexpr int MAX_OBJECTS = 16;
constexpr int MAX_MATS    = 32;
constexpr int MAX_QUADS   = 8;   // 6 walls + 1 light

// ============================================================================
// 2. SCENE DESCRIPTION (built on the CPU)
//
// The random choices are made on the CPU and stored as plain numbers in a
// SceneDesc. It is copied to the GPU, where build_scene() creates the objects,
// and the same description is written to scene.json.
// ============================================================================

enum MatKind { MAT_DIFFUSE = 0, MAT_CHECKER = 1, MAT_METAL = 2, MAT_GLASS = 3, MAT_LIGHT = 4 };

struct MatDesc {
    int   kind;   // MatKind
    color c1;     // albedo (diffuse, metal), first colour (checker), emitted light (light)
    color c2;     // second colour (checker)
    float param;  // fuzz (metal), index of refraction (glass), square size (checker)
};

struct ObjectDesc {
    int    shape;        // ShapeKind
    int    mat;          // index into SceneDesc::mats
    affine placement;    // translation(position) * rotation * shear * scaling(scale)
    // The parts of the placement, kept for scene.json:
    vec3   scale;
    mat3   rotation;
    float  shear[6];     // xy, xz, yx, yz, zx, zy (see shearing() in mat3.h)
    vec3   position;
    point3 bbox_min, bbox_max;   // world-space bounding box
};

struct QuadDesc {
    affine placement;    // applied to the unit square (xy plane, facing +z)
    int    mat;
    const char* name;    // CPU only, for scene.json
};

struct SceneDesc {
    int        num_mats = 0, num_objects = 0, num_quads = 0;
    MatDesc    mats[MAX_MATS];
    ObjectDesc objects[MAX_OBJECTS];
    QuadDesc   quads[MAX_QUADS];   // walls and the light
    int        light_quad = -1;    // which quad is the light

    // CPU only
    int    level = 0;
    float  room_w = ROOM_W, room_h = ROOM_H, room_d = ROOM_D;
    point3 lookfrom, lookat;
    float  vfov = 40;
    color  background;
    point3 light_center;
    float  light_w = 0, light_d = 0;
    color  light_emission;
};

// Random numbers for scene building. std::mt19937 and std::seed_seq give the
// same numbers with every compiler; the std::uniform_*_distribution classes do
// not, so the conversions to float and int are written out here.
struct SceneRng {
    std::mt19937 gen;

    SceneRng(uint32_t seed, uint32_t level, uint32_t index) {
        std::seed_seq seq{seed, level, index};
        gen.seed(seq);
    }
    float uniform() { return float(gen() >> 8) * (1.0f / 16777216.0f); }  // [0,1)
    float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
    int   range_int(int lo, int hi) { return lo + int(gen() % uint32_t(hi - lo + 1)); }
    bool  chance(float p) { return uniform() < p; }
};

// ---- Colours ----------------------------------------------------------------
// Picked in HSV like a colour picker, then squared to turn the displayed value
// into the linear albedo the renderer uses (write_color does the reverse).

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

color random_albedo(SceneRng& rng)        { return hsv_to_linear(rng.uniform(), rng.range(0.25f, 0.85f), rng.range(0.45f, 0.92f)); }
color random_metal_color(SceneRng& rng)   { return hsv_to_linear(rng.uniform(), rng.range(0.00f, 0.35f), rng.range(0.80f, 0.95f)); }
color random_wall_color(SceneRng& rng)    { return hsv_to_linear(rng.uniform(), rng.range(0.00f, 0.50f), rng.range(0.60f, 0.90f)); }
color random_neutral_color(SceneRng& rng) { return hsv_to_linear(rng.uniform(), rng.range(0.00f, 0.12f), rng.range(0.70f, 0.90f)); }

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

void add_quad(SceneDesc& s, const affine& placement, int mat, const char* name) {
    if (s.num_quads >= MAX_QUADS) return;
    QuadDesc& q = s.quads[s.num_quads++];
    q.placement = placement; q.mat = mat; q.name = name;
}

// ---- The room, its light and the camera ----------------------------------------
// Closed room: floor y = 0, ceiling y = H, back wall at z = -D/2, front wall at z = +D/2,
// side walls at x = -W/2 (left, red in the fixed room) and +W/2 (right, green).
// The camera is inside, near the front wall. (With an open front, mirrors facing the
// camera would reflect the black outside; closed, they reflect the room.)

void make_room(SceneDesc& s, SceneRng& rng, const LevelSpec& L) {
    color left_c, right_c, back_c, ceil_c, front_c;
    int m_floor;
    if (L.random_room) {
        s.room_w = rng.range(7.0f, 10.0f);
        s.room_h = rng.range(5.5f, 7.5f);
        s.room_d = rng.range(9.0f, 13.0f);
        if (rng.chance(0.5f)) {   // Cornell-style coloured side walls
            float hue = rng.uniform();
            left_c  = hsv_to_linear(hue,                          rng.range(0.5f, 0.8f), rng.range(0.55f, 0.8f));
            right_c = hsv_to_linear(hue + rng.range(0.25f, 0.6f), rng.range(0.5f, 0.8f), rng.range(0.55f, 0.8f));
        } else {
            left_c  = random_wall_color(rng);
            right_c = random_wall_color(rng);
        }
        back_c  = rng.chance(0.6f) ? random_neutral_color(rng) : random_wall_color(rng);
        front_c = rng.chance(0.6f) ? random_neutral_color(rng) : random_wall_color(rng);
        ceil_c  = random_neutral_color(rng);
        if (rng.chance(0.4f)) {   // checker floor makes refraction easy to see
            color c = random_wall_color(rng);
            m_floor = add_mat(s, MAT_CHECKER, c, c * rng.range(0.1f, 0.4f), rng.range(0.5f, 1.5f));
        } else {
            m_floor = add_mat(s, MAT_DIFFUSE, random_wall_color(rng));
        }
    } else {
        s.room_w = ROOM_W; s.room_h = ROOM_H; s.room_d = ROOM_D;
        left_c  = color(.65f, .05f, .05f);   // red, like the Cornell box
        right_c = color(.12f, .45f, .15f);   // green
        back_c  = ceil_c = front_c = color(.73f, .73f, .73f);
        m_floor = add_mat(s, MAT_DIFFUSE, color(.73f, .73f, .73f));
    }
    float W = s.room_w, H = s.room_h, D = s.room_d;
    s.background = color(0, 0, 0);

    int m_left  = add_mat(s, MAT_DIFFUSE, left_c);
    int m_right = add_mat(s, MAT_DIFFUSE, right_c);
    int m_back  = add_mat(s, MAT_DIFFUSE, back_c);
    int m_ceil  = add_mat(s, MAT_DIFFUSE, ceil_c);
    int m_front = add_mat(s, MAT_DIFFUSE, front_c);

    // Unit squares (xy plane, facing +z) scaled to each wall, turned to face into the room.
    add_quad(s, translation(vec3(0, 0, 0))          * rotation_x(-90) * scaling(vec3(W, D, 1)), m_floor, "floor");
    add_quad(s, translation(vec3(0, H, 0))          * rotation_x(90)  * scaling(vec3(W, D, 1)), m_ceil,  "ceiling");
    add_quad(s, translation(vec3(0, H / 2, -D / 2))                   * scaling(vec3(W, H, 1)), m_back,  "back_wall");
    add_quad(s, translation(vec3(-W / 2, H / 2, 0)) * rotation_y(90)  * scaling(vec3(D, H, 1)), m_left,  "left_wall");
    add_quad(s, translation(vec3(W / 2, H / 2, 0))  * rotation_y(-90) * scaling(vec3(D, H, 1)), m_right, "right_wall");
    add_quad(s, translation(vec3(0, H / 2, D / 2))  * rotation_y(180) * scaling(vec3(W, H, 1)), m_front, "front_wall");

    // One rectangular ceiling light, facing down, just below the ceiling.
    // Brightness is set from the share of the ceiling it covers, so a small panel is
    // brighter than a big one and the room gets similar total light.
    float lw, ld, lx, lz, strength;
    color tint(1, 1, 1);
    if (L.random_light) {
        lw = W * rng.range(0.2f, 0.45f);
        ld = D * rng.range(0.2f, 0.45f);
        lx = rng.range(-0.2f, 0.2f) * W;
        lz = rng.range(-0.2f, 0.2f) * D;
        strength = rng.range(0.35f, 0.7f);
        tint = random_light_tint(rng);
    } else {
        lw = 0.3f * W; ld = 0.3f * D; lx = 0; lz = 0;
        strength = 0.45f;
    }
    color emission = tint * (strength / (lw * ld / (W * D)));
    int m_light = add_mat(s, MAT_LIGHT, emission);
    s.light_quad = s.num_quads;
    add_quad(s, translation(vec3(lx, H - 0.01f, lz)) * rotation_x(90) * scaling(vec3(lw, ld, 1)), m_light, "light");
    s.light_center = point3(lx, H - 0.01f, lz);
    s.light_w = lw; s.light_d = ld;
    s.light_emission = emission;

    // Camera inside the room, near the front wall, looking toward the back.
    if (L.random_camera) {
        s.vfov = rng.range(FOV_MIN, FOV_MAX);
        s.lookfrom = point3(rng.range(-0.25f, 0.25f) * W, rng.range(0.3f, 0.75f) * H, D / 2 - rng.range(0.25f, 1.2f));
        s.lookat   = point3(rng.range(-0.15f, 0.15f) * W, rng.range(0.15f, 0.45f) * H, rng.range(-D / 2 + 1.0f, 0.0f));
    } else {
        s.vfov = FIXED_FOV;
        s.lookfrom = point3(0, 0.5f * H, D / 2 - 0.25f);
        s.lookat   = point3(0, 0.42f * H, -D / 2);
    }
}

// ---- Objects -------------------------------------------------------------------

// Corner points of each unit shape (all fit in -0.5..0.5). Spheres have no corners;
// their extent is worked out separately.
int unit_shape_points(int shape, point3* pts) {
    switch (shape) {
        case SHAPE_BOX: {
            int n = 0;
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) for (int k = 0; k < 2; k++)
                pts[n++] = point3(i ? 0.5f : -0.5f, j ? 0.5f : -0.5f, k ? 0.5f : -0.5f);
            return 8;
        }
        case SHAPE_SQUARE_PYRAMID:
            pts[0] = point3(-0.5f, -0.5f, -0.5f); pts[1] = point3(0.5f, -0.5f, -0.5f);
            pts[2] = point3( 0.5f, -0.5f,  0.5f); pts[3] = point3(-0.5f, -0.5f, 0.5f);
            pts[4] = point3(0, 0.5f, 0);
            return 5;
        case SHAPE_TRIANGLE_PYRAMID:
            pts[0] = point3(-0.5f, -0.5f, -0.28867513f); pts[1] = point3(0.5f, -0.5f, -0.28867513f);
            pts[2] = point3( 0.0f, -0.5f,  0.57735027f); pts[3] = point3(0, 0.5f, 0);
            return 4;
        default:
            return 0;
    }
}

// Bounding box of a unit shape after the matrix M (no translation).
void shape_extent(int shape, const mat3& M, point3& lo, point3& hi) {
    if (shape == SHAPE_SPHERE) {
        // A sphere of radius 0.5 under M: half-extent along axis k is 0.5 * |row k of M|.
        vec3 e;
        for (int k = 0; k < 3; k++)
            e[k] = 0.5f * std::sqrt(M.m[k][0] * M.m[k][0] + M.m[k][1] * M.m[k][1] + M.m[k][2] * M.m[k][2]);
        lo = -e; hi = e;
        return;
    }
    point3 pts[8];
    int n = unit_shape_points(shape, pts);
    lo = point3( infinity,  infinity,  infinity);
    hi = point3(-infinity, -infinity, -infinity);
    for (int i = 0; i < n; i++) {
        point3 p = M * pts[i];
        for (int k = 0; k < 3; k++) { lo[k] = std::fmin(lo[k], p[k]); hi[k] = std::fmax(hi[k], p[k]); }
    }
}

// Radius of a circle (centred at cx, cz) around the shape seen from above.
float shape_footprint(int shape, const mat3& M, float& cx, float& cz) {
    point3 lo, hi;
    shape_extent(shape, M, lo, hi);
    cx = 0.5f * (lo.x() + hi.x());
    cz = 0.5f * (lo.z() + hi.z());
    if (shape == SHAPE_SPHERE) {
        float ex = 0.5f * (hi.x() - lo.x()), ez = 0.5f * (hi.z() - lo.z());
        return std::sqrt(ex * ex + ez * ez);   // safe circle around the ellipse
    }
    point3 pts[8];
    int n = unit_shape_points(shape, pts);
    float r2 = 0;
    for (int i = 0; i < n; i++) {
        point3 p = M * pts[i];
        float dx = p.x() - cx, dz = p.z() - cz;
        r2 = std::fmax(r2, dx * dx + dz * dz);
    }
    return std::sqrt(r2);
}

int random_object_material(SceneDesc& s, SceneRng& rng, const LevelSpec& L) {
    float p = L.any_material ? rng.uniform() : 0.0f;
    if (p < P_DIFFUSE) {
        if (L.any_material && rng.chance(P_CHECKER_OBJECT)) {
            color c = random_albedo(rng);
            return add_mat(s, MAT_CHECKER, c, c * rng.range(0.1f, 0.4f), rng.range(0.15f, 0.4f));
        }
        return add_mat(s, MAT_DIFFUSE, random_albedo(rng));
    }
    if (p < P_DIFFUSE + P_METAL)
        return add_mat(s, MAT_METAL, random_metal_color(rng), color(0, 0, 0), rng.range(0.0f, FUZZ_MAX));
    return add_mat(s, MAT_GLASS, color(1, 1, 1), color(0, 0, 0), rng.range(IOR_MIN, IOR_MAX));
}

struct Footprint { float x, z, r; };

void add_objects(SceneDesc& s, SceneRng& rng, const LevelSpec& L) {
    float W = s.room_w, H = s.room_h, D = s.room_d;
    float max_height = 0.55f * H;   // stays well below the light
    // Floor area objects may use: away from the walls and clear of the camera.
    float xmin = -W / 2 + 0.3f, xmax = W / 2 - 0.3f, zmin = -D / 2 + 0.3f, zmax = D / 2 - 3.5f;

    std::vector<Footprint> used;
    int wanted = rng.range_int(L.min_objects, L.max_objects);

    for (int n = 0; n < wanted && s.num_objects < MAX_OBJECTS; n++) {
        ObjectDesc o;
        o.shape = L.any_shape ? rng.range_int(0, 3) : SHAPE_SPHERE;

        // Size, then stretch, shear and rotation (applied right to left: scale first).
        float size = rng.range(L.min_size, L.max_size);
        o.scale = vec3(size, size, size);
        if (L.stretch)
            o.scale = vec3(size * rng.range(0.6f, 1.4f), size * rng.range(0.6f, 1.4f), size * rng.range(0.6f, 1.4f));
        for (int k = 0; k < 6; k++) o.shear[k] = (L.max_shear > 0) ? rng.range(-L.max_shear, L.max_shear) : 0.0f;
        if (L.rotation == ROT_Y)        o.rotation = rotation_y(rng.range(0.0f, 360.0f));
        else if (L.rotation == ROT_ANY) o.rotation = random_rotation(rng.uniform(), rng.uniform(), rng.uniform());
        else                            o.rotation = mat3();

        mat3 M = o.rotation
               * shearing(o.shear[0], o.shear[1], o.shear[2], o.shear[3], o.shear[4], o.shear[5])
               * scaling(o.scale);

        // Too tall? Shrink it evenly.
        point3 lo, hi;
        shape_extent(o.shape, M, lo, hi);
        if (hi.y() - lo.y() > max_height) {
            float f = max_height / (hi.y() - lo.y());
            o.scale = f * o.scale;
            M = o.rotation
              * shearing(o.shear[0], o.shear[1], o.shear[2], o.shear[3], o.shear[4], o.shear[5])
              * scaling(o.scale);
            shape_extent(o.shape, M, lo, hi);
        }

        // Where it stands: the middle of the floor, or a free spot. If the floor is too
        // crowded, shrink the object a little and try again (up to 4 times) rather than
        // leaving it out, so scenes get the number of objects they asked for.
        float cx, cz;
        float r = shape_footprint(o.shape, M, cx, cz);
        float x = 0, z = 0;
        if (L.random_position) {
            bool found = false;
            for (int shrink = 0; shrink < 5 && !found; shrink++) {
                if (shrink > 0) {
                    o.scale = 0.85f * o.scale;
                    M = o.rotation
                      * shearing(o.shear[0], o.shear[1], o.shear[2], o.shear[3], o.shear[4], o.shear[5])
                      * scaling(o.scale);
                    shape_extent(o.shape, M, lo, hi);
                    r = shape_footprint(o.shape, M, cx, cz);
                }
                for (int attempt = 0; attempt < 200 && !found; attempt++) {
                    if (xmax - xmin < 2 * r || zmax - zmin < 2 * r) break;
                    x = rng.range(xmin + r, xmax - r);
                    z = rng.range(zmin + r, zmax - r);
                    found = true;
                    for (const Footprint& f : used) {
                        float dx = x - f.x, dz = z - f.z, min_d = r + f.r + 0.15f;
                        if (dx * dx + dz * dz < min_d * min_d) { found = false; break; }
                    }
                }
            }
            if (!found) continue;   // still no room: fewer objects in this scene
        }
        used.push_back({x, z, r});

        // Lift it so its lowest point rests on the floor (0.002 above, so a flat
        // bottom isn't exactly on the floor plane), and centre it on (x, z).
        o.position = vec3(x - cx, -lo.y() + 0.002f, z - cz);
        o.placement = translation(o.position) * M;
        o.bbox_min = lo + o.position;
        o.bbox_max = hi + o.position;
        o.mat = random_object_material(s, rng, L);
        s.objects[s.num_objects++] = o;
    }
}

void generate_scene(SceneDesc& s, SceneRng& rng, int level) {
    const LevelSpec& L = LEVELS[level - 1];
    s.level = level;
    make_room(s, rng, L);
    add_objects(s, rng, L);
}

// ============================================================================
// 3. BUILDING THE SCENE ON THE GPU
// ============================================================================

struct GpuScene {
    hittable*      world;    // BVH over `objects`
    hittable_list* objects;  // owns every object
    hittable_list* lights;   // owns the copy of the light quad
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
                g.texs[g.num_texs++] = tex;        // lambertian(texture*) doesn't delete it
                g.mats[i] = new lambertian(tex);
                break;
            }
            case MAT_METAL: g.mats[i] = new metal(m.c1, m.param);   break;
            case MAT_GLASS: g.mats[i] = new dielectric(m.param);    break;
            case MAT_LIGHT: g.mats[i] = new diffuse_light(m.c1);    break;
            default:        g.mats[i] = new lambertian(m.c1);       break;
        }
    }

    g.objects = new hittable_list();
    g.lights  = new hittable_list();

    // Walls and light: placed unit quads (no transform wrapper, so the light can be sampled).
    for (int i = 0; i < d.num_quads; i++)
        g.objects->add(new quad(d.quads[i].placement, g.mats[d.quads[i].mat]));
    g.lights->add(new quad(d.quads[d.light_quad].placement, nullptr));

    // Objects: unit shape + one transform.
    for (int i = 0; i < d.num_objects; i++) {
        const ObjectDesc& o = d.objects[i];
        material* mat = g.mats[o.mat];
        hittable* shape;
        switch (o.shape) {
            case SHAPE_BOX:              shape = box(mat);                  break;
            case SHAPE_SQUARE_PYRAMID:   shape = square_pyramid(mat);       break;
            case SHAPE_TRIANGLE_PYRAMID: shape = triangle_pyramid(mat);     break;
            default:                     shape = unit_sphere(mat);          break;
        }
        g.objects->add(new transform(shape, o.placement));
    }

    // The BVH only points at the objects; `objects` still owns them.
    g.world = new bvh_node(*g.objects);
}

__global__ void build_world_kernel(const SceneDesc* desc, GpuScene* g, hittable** d_world, hittable** d_lights) {
    build_scene(*desc, *g);
    *d_world  = g->world;
    *d_lights = g->lights;
}

__global__ void free_world_kernel(GpuScene* g) {
    delete static_cast<bvh_node*>(g->world);   // BVH nodes only
    delete g->objects;                         // every object
    delete g->lights;                          // the light copy
    for (int i = 0; i < g->num_mats; i++) delete g->mats[i];
    for (int i = 0; i < g->num_texs; i++) delete g->texs[i];
}

// ============================================================================
// 4. RENDERING AND SAVING
// ============================================================================

#define GEN_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {          \
    std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n", cudaGetErrorString(e_),       \
                 __FILE__, __LINE__); std::exit(1); } } while (0)

// Like render_init, but with a seed per scene so each scene gets its own noise.
__global__ void seed_rng_kernel(int width, int height, unsigned long long seed, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int pixel_index = j * width + i;
    curand_init(seed, pixel_index, 0, &states[pixel_index]);
}

// GPU and CPU buffers, allocated once and reused for every scene.
struct Buffers {
    curandState* d_rand  = nullptr;
    color*       d_accum = nullptr;
    float*       d_gbuf  = nullptr;
    std::vector<color> image;   // linear radiance, averaged over samples
    std::vector<float> gbuf;    // GBUFFER_CHANNELS floats per pixel
};

void render_scene(const camera& cam, hittable** d_world, hittable** d_lights, Buffers& buf,
                  unsigned long long gpu_seed) {
    int w = cam.image_width, h = cam.image_height, n = w * h;
    dim3 threads(8, 8);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);

    seed_rng_kernel<<<blocks, threads>>>(w, h, gpu_seed, buf.d_rand);
    GEN_CHECK(cudaGetLastError());

    // G-buffer first: it uses a copy of each pixel's random state, so the render
    // below gets exactly the same random numbers as without it.
    gbuffer_kernel<<<blocks, threads>>>(cam, d_world, buf.d_rand, buf.d_gbuf);
    GEN_CHECK(cudaGetLastError());

    GEN_CHECK(cudaMemset(buf.d_accum, 0, sizeof(color) * n));
    for (int s_j = 0; s_j < cam.sqrt_spp; s_j++) {
        for (int s_i = 0; s_i < cam.sqrt_spp; s_i++) {
            render_kernel<<<blocks, threads>>>(cam, d_world, d_lights, buf.d_rand, buf.d_accum, s_i, s_j);
            GEN_CHECK(cudaGetLastError());
        }
    }
    GEN_CHECK(cudaDeviceSynchronize());

    buf.image.resize(n);
    buf.gbuf.resize(size_t(n) * GBUFFER_CHANNELS);
    GEN_CHECK(cudaMemcpy(buf.image.data(), buf.d_accum, sizeof(color) * n, cudaMemcpyDeviceToHost));
    GEN_CHECK(cudaMemcpy(buf.gbuf.data(), buf.d_gbuf, sizeof(float) * buf.gbuf.size(), cudaMemcpyDeviceToHost));
    for (color& c : buf.image) c = cam.pixel_samples_scale * c;
}

// ---- Previews (8-bit, for looking at; not training data) ---------------------------

unsigned char to_byte(float x) {          // same conversion as write_color (gamma 2, clamp)
    if (!(x > 0)) return 0;
    x = std::sqrt(x);
    if (x > 0.999f) x = 0.999f;
    return (unsigned char)(int(256 * x));
}

unsigned char unit_to_byte(float x) {     // 0..1 straight to 0..255
    if (!(x > 0)) return 0;
    if (x > 0.999f) x = 0.999f;
    return (unsigned char)(int(256 * x));
}

void write_ppm(const std::filesystem::path& path, const std::vector<unsigned char>& rgb, int w, int h) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << ' ' << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
}

void write_image_preview(const std::filesystem::path& path, const std::vector<color>& img, int w, int h) {
    std::vector<unsigned char> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < img.size(); i++)
        for (int c = 0; c < 3; c++) rgb[3 * i + c] = to_byte(img[i][c]);
    write_ppm(path, rgb, w, h);
}

void write_gbuffer_preview_files(const std::filesystem::path& dir, const std::vector<float>& g, int w, int h) {
    int n = w * h;
    std::vector<unsigned char> normal(3 * n), albedo(3 * n), depth(3 * n), mat(3 * n);
    float max_depth = 0;
    for (int i = 0; i < n; i++) max_depth = std::fmax(max_depth, g[size_t(i) * GBUFFER_CHANNELS]);
    for (int i = 0; i < n; i++) {
        const float* p = &g[size_t(i) * GBUFFER_CHANNELS];
        bool hit = p[0] > 0;
        bool is_light = hit && p[13] == 0 && p[14] == 0 && p[15] == 0;
        for (int c = 0; c < 3; c++) {
            normal[3*i + c] = hit ? unit_to_byte(0.5f * p[4 + c] + 0.5f) : 0;
            albedo[3*i + c] = to_byte(p[7 + c]);
            depth[3*i + c]  = hit ? unit_to_byte(1.0f - 0.8f * p[0] / max_depth) : 0;
            mat[3*i + c]    = is_light ? 255 : unit_to_byte(p[13 + c]);   // diffuse red, metal green, glass blue
        }
    }
    write_ppm(dir / "gbuffer_normal.ppm",   normal, w, h);
    write_ppm(dir / "gbuffer_albedo.ppm",   albedo, w, h);
    write_ppm(dir / "gbuffer_depth.ppm",    depth,  w, h);
    write_ppm(dir / "gbuffer_material.ppm", mat,    w, h);
}

// ---- scene.json ------------------------------------------------------------------

std::string num(float x) {
    if (!(x == x)) return "0";
    char b[32];
    std::snprintf(b, sizeof(b), "%.6g", double(x));
    return b;
}
std::string vec_json(const vec3& v) { return "[" + num(v.x()) + ", " + num(v.y()) + ", " + num(v.z()) + "]"; }
std::string mat_json(const mat3& M) {
    std::string s = "[";
    for (int i = 0; i < 3; i++) {
        s += "[" + num(M.m[i][0]) + ", " + num(M.m[i][1]) + ", " + num(M.m[i][2]) + "]";
        if (i < 2) s += ", ";
    }
    return s + "]";
}

const char* shape_name(int s) {
    switch (s) {
        case SHAPE_BOX:              return "box";
        case SHAPE_SQUARE_PYRAMID:   return "square_pyramid";
        case SHAPE_TRIANGLE_PYRAMID: return "triangle_pyramid";
        default:                     return "sphere";
    }
}

std::string material_json(const MatDesc& m) {
    switch (m.kind) {
        case MAT_CHECKER: return "{\"type\": \"diffuse\", \"pattern\": \"checker\", \"albedo\": " + vec_json(m.c1)
                               + ", \"albedo2\": " + vec_json(m.c2) + ", \"square_size\": " + num(m.param) + "}";
        case MAT_METAL:   return "{\"type\": \"metal\", \"albedo\": " + vec_json(m.c1) + ", \"roughness\": " + num(m.param) + "}";
        case MAT_GLASS:   return "{\"type\": \"glass\", \"albedo\": [1, 1, 1], \"ior\": " + num(m.param) + "}";
        case MAT_LIGHT:   return "{\"type\": \"light\", \"emission\": " + vec_json(m.c1) + "}";
        default:          return "{\"type\": \"diffuse\", \"albedo\": " + vec_json(m.c1) + "}";
    }
}

void write_scene_json(const std::filesystem::path& path, const SceneDesc& s, int index, unsigned seed,
                      const char* split, int width, int height, int spp) {
    const LevelSpec& L = LEVELS[s.level - 1];
    std::ofstream o(path);
    o << "{\n";
    o << "  \"level\": " << s.level << ",\n";
    o << "  \"level_name\": \"" << L.name << "\",\n";
    o << "  \"index\": " << index << ",\n";
    o << "  \"seed\": " << seed << ",\n";
    o << "  \"split\": \"" << split << "\",\n";
    o << "  \"render\": {\"width\": " << width << ", \"height\": " << height
      << ", \"samples_per_pixel\": " << spp << ", \"max_depth\": " << MAX_DEPTH << "},\n";
    o << "  \"files\": {\"image\": \"image.npy\", \"gbuffer\": \"gbuffer.npy\", \"gbuffer_channels\": "
         "[\"depth\", \"pos_x\", \"pos_y\", \"pos_z\", \"normal_x\", \"normal_y\", \"normal_z\", "
         "\"albedo_r\", \"albedo_g\", \"albedo_b\", \"view_x\", \"view_y\", \"view_z\", "
         "\"is_diffuse\", \"is_metal\", \"is_glass\", \"roughness\", \"ior\"]},\n";
    o << "  \"camera\": {\"lookfrom\": " << vec_json(s.lookfrom) << ", \"lookat\": " << vec_json(s.lookat)
      << ", \"vup\": [0, 1, 0], \"vfov\": " << num(s.vfov) << "},\n";
    o << "  \"background\": " << vec_json(s.background) << ",\n";
    o << "  \"room\": {\"width\": " << num(s.room_w) << ", \"height\": " << num(s.room_h)
      << ", \"depth\": " << num(s.room_d) << "},\n";
    o << "  \"light\": {\"center\": " << vec_json(s.light_center) << ", \"size\": [" << num(s.light_w) << ", "
      << num(s.light_d) << "], \"emission\": " << vec_json(s.light_emission) << ", \"facing\": [0, -1, 0]},\n";
    o << "  \"light_descriptor\": [" << num(s.light_center.x()) << ", " << num(s.light_center.y()) << ", "
      << num(s.light_center.z()) << ", " << num(s.light_w) << ", " << num(s.light_d) << ", "
      << num(s.light_emission.x()) << ", " << num(s.light_emission.y()) << ", " << num(s.light_emission.z()) << "],\n";

    o << "  \"walls\": [\n";
    bool first = true;
    for (int i = 0; i < s.num_quads; i++) {
        if (i == s.light_quad) continue;
        if (!first) o << ",\n";
        first = false;
        o << "    {\"name\": \"" << s.quads[i].name << "\", \"material\": " << material_json(s.mats[s.quads[i].mat]) << "}";
    }
    o << "\n  ],\n";

    o << "  \"objects\": [\n";
    for (int i = 0; i < s.num_objects; i++) {
        const ObjectDesc& ob = s.objects[i];
        o << "    {\"shape\": \"" << shape_name(ob.shape) << "\",\n";
        o << "     \"material\": " << material_json(s.mats[ob.mat]) << ",\n";
        o << "     \"scale\": " << vec_json(ob.scale) << ",\n";
        o << "     \"shear\": [" << num(ob.shear[0]) << ", " << num(ob.shear[1]) << ", " << num(ob.shear[2]) << ", "
          << num(ob.shear[3]) << ", " << num(ob.shear[4]) << ", " << num(ob.shear[5]) << "],\n";
        o << "     \"rotation\": " << mat_json(ob.rotation) << ",\n";
        o << "     \"position\": " << vec_json(ob.position) << ",\n";
        o << "     \"matrix\": " << mat_json(ob.placement.M) << ",\n";
        o << "     \"bbox_min\": " << vec_json(ob.bbox_min) << ", \"bbox_max\": " << vec_json(ob.bbox_max) << "}";
        o << (i + 1 < s.num_objects ? ",\n" : "\n");
    }
    o << "  ]\n";
    o << "}\n";
}

// Train / val / test from (seed, level, index), independent of the scene's contents.
const char* choose_split(unsigned seed, int level, int index) {
    uint64_t x = (uint64_t(seed) << 40) ^ (uint64_t(level) << 32) ^ uint64_t(uint32_t(index));
    x += 0x9E3779B97F4A7C15ULL;                          // splitmix64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    int p = int(x % 100);
    if (p < SPLIT_TRAIN_PERCENT) return "train";
    if (p < SPLIT_TRAIN_PERCENT + SPLIT_VAL_PERCENT) return "val";
    return "test";
}

std::string level_folder(int level) {
    char b[128];
    std::snprintf(b, sizeof(b), "level_%02d_%s", level, LEVELS[level - 1].name);
    return b;
}

// ============================================================================
// 5. MAIN
// ============================================================================

struct Options {
    int         level = 0;
    int         count = 10;
    int         start = 0;
    int         size  = 256;
    int         spp   = 3000;
    unsigned    seed  = 1;
    std::string out   = "dataset";
    bool        preview = true;
    bool        gbuffer_previews = false;
    bool        list = false;
};

void usage() {
    std::fprintf(stderr,
        "usage: rt_dataset --level N [--count N] [--start N] [--size N] [--spp N] [--seed N] [--out DIR]\n"
        "                  [--no-preview] [--gbuffer-previews]\n"
        "       rt_dataset --list\n"
        "  --level             which level to make (see --list)\n"
        "  --count             number of scenes                   (default 10)\n"
        "  --start             index of the first scene           (default 0)\n"
        "  --size              image width and height in pixels   (default 256)\n"
        "  --spp               samples per pixel, best a square   (default 1024)\n"
        "  --seed              changes every scene                (default 1)\n"
        "  --out               dataset folder                     (default dataset)\n"
        "  --no-preview        don't write image.ppm\n"
        "  --gbuffer-previews  also write 4 small G-buffer pictures per scene\n");
}

bool parse_options(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--list")             { o.list = true; continue; }
        if (a == "--no-preview")       { o.preview = false; continue; }
        if (a == "--gbuffer-previews") { o.gbuffer_previews = true; continue; }
        if (i + 1 >= argc) { usage(); return false; }
        std::string v = argv[++i];
        if      (a == "--level") o.level = std::atoi(v.c_str());
        else if (a == "--count") o.count = std::atoi(v.c_str());
        else if (a == "--start") o.start = std::atoi(v.c_str());
        else if (a == "--size")  o.size  = std::atoi(v.c_str());
        else if (a == "--spp")   o.spp   = std::atoi(v.c_str());
        else if (a == "--seed")  o.seed  = unsigned(std::strtoul(v.c_str(), nullptr, 10));
        else if (a == "--out")   o.out   = v;
        else { usage(); return false; }
    }
    if (o.list) return true;
    if (o.level < 1 || o.level > NUM_LEVELS) {
        std::fprintf(stderr, "--level must be between 1 and %d (see --list)\n", NUM_LEVELS);
        return false;
    }
    if (o.count < 1 || o.start < 0 || o.size < 8 || o.spp < 1) { usage(); return false; }
    return true;
}

void list_levels() {
    std::printf("level  name                         objects  shapes  materials  position  stretch  rotation  shear  camera  light  room\n");
    const char* rot[] = {"none", "y", "any"};
    for (int i = 0; i < NUM_LEVELS; i++) {
        const LevelSpec& L = LEVELS[i];
        std::printf("%5d  %-27s  %2d-%-2d    %-6s  %-9s  %-8s  %-7s  %-8s  %-5s  %-6s  %-5s  %s\n",
                    i + 1, L.name, L.min_objects, L.max_objects,
                    L.any_shape ? "all" : "sphere", L.any_material ? "all" : "diffuse",
                    L.random_position ? "random" : "centre", L.stretch ? "yes" : "no", rot[L.rotation],
                    L.max_shear > 0 ? "yes" : "no", L.random_camera ? "random" : "fixed",
                    L.random_light ? "random" : "fixed", L.random_room ? "random" : "fixed");
    }
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_options(argc, argv, opt)) return 1;
    if (opt.list) { list_levels(); return 0; }

    namespace fs = std::filesystem;
    fs::path level_dir = fs::path(opt.out) / level_folder(opt.level);
    fs::create_directories(level_dir);

    GEN_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
    GEN_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t(64) << 20));

    // Camera settings shared by every scene.
    camera cam;
    cam.aspect_ratio      = 1.0f;
    cam.image_width       = opt.size;
    cam.samples_per_pixel = opt.spp;
    cam.max_depth         = MAX_DEPTH;
    cam.vup               = vec3(0, 1, 0);
    cam.defocus_angle     = 0;
    cam.initialize();
    int w = cam.image_width, h = cam.image_height, npix = w * h;
    int real_spp = cam.sqrt_spp * cam.sqrt_spp;

    // GPU memory, allocated once.
    Buffers buf;
    SceneDesc* d_desc;
    GpuScene*  d_scene;
    hittable** d_world;
    hittable** d_lights;
    GEN_CHECK(cudaMalloc(&buf.d_rand,  sizeof(curandState) * npix));
    GEN_CHECK(cudaMalloc(&buf.d_accum, sizeof(color) * npix));
    GEN_CHECK(cudaMalloc(&buf.d_gbuf,  sizeof(float) * size_t(npix) * GBUFFER_CHANNELS));
    GEN_CHECK(cudaMalloc(&d_desc,  sizeof(SceneDesc)));
    GEN_CHECK(cudaMalloc(&d_scene, sizeof(GpuScene)));
    GEN_CHECK(cudaMalloc(&d_world,  sizeof(hittable*)));
    GEN_CHECK(cudaMalloc(&d_lights, sizeof(hittable*)));

    // index.csv: one line per finished scene, for the whole dataset.
    fs::path index_path = fs::path(opt.out) / "index.csv";
    bool new_index = !fs::exists(index_path);
    std::ofstream index_csv(index_path, std::ios::app);
    if (new_index) index_csv << "level,level_name,index,split,folder,objects,seconds\n";

    std::printf("Level %d (%s): %d scenes, %d x %d, %d samples per pixel, into \"%s\"\n",
                opt.level, LEVELS[opt.level - 1].name, opt.count, w, h, real_spp, level_dir.string().c_str());
    if (real_spp != opt.spp)
        std::printf("Note: %d is not a square number, so %d samples are used.\n", opt.spp, real_spp);

    auto t_all = std::chrono::steady_clock::now();
    int done = 0;
    double seconds_sum = 0;

    for (int k = 0; k < opt.count; k++) {
        int index = opt.start + k;
        char name[32];
        std::snprintf(name, sizeof(name), "scene_%05d", index);
        fs::path dir = level_dir / name;

        if (fs::exists(dir / "scene.json") && fs::exists(dir / "image.npy") && fs::exists(dir / "gbuffer.npy")) {
            std::printf("[%d/%d] %s already done, skipped\n", k + 1, opt.count, name);
            continue;
        }
        fs::create_directories(dir);
        auto t0 = std::chrono::steady_clock::now();

        // 1. Describe the scene on the CPU.
        SceneRng rng(opt.seed, uint32_t(opt.level), uint32_t(index));
        SceneDesc desc;
        generate_scene(desc, rng, opt.level);
        const char* split = choose_split(opt.seed, opt.level, index);

        // 2. Build it on the GPU.
        GEN_CHECK(cudaMemcpy(d_desc, &desc, sizeof(SceneDesc), cudaMemcpyHostToDevice));
        build_world_kernel<<<1, 1>>>(d_desc, d_scene, d_world, d_lights);
        GEN_CHECK(cudaGetLastError());
        GEN_CHECK(cudaDeviceSynchronize());

        // 3. Render it (G-buffer + image).
        cam.lookfrom   = desc.lookfrom;
        cam.lookat     = desc.lookat;
        cam.vfov       = desc.vfov;
        cam.background = desc.background;
        cam.focus_dist = (desc.lookfrom - desc.lookat).length();
        cam.initialize();
        unsigned long long gpu_seed = (unsigned long long)opt.seed * 1000003ULL
                                    + (unsigned long long)opt.level * 100003ULL + (unsigned long long)index;
        render_scene(cam, d_world, d_lights, buf, gpu_seed);

        // 4. Delete the scene's objects.
        free_world_kernel<<<1, 1>>>(d_scene);
        GEN_CHECK(cudaGetLastError());
        GEN_CHECK(cudaDeviceSynchronize());

        // 5. Save. scene.json last, so a folder with scene.json is always complete.
        bool ok = write_npy((dir / "image.npy").string(), reinterpret_cast<const float*>(buf.image.data()), h, w, 3)
               && write_npy((dir / "gbuffer.npy").string(), buf.gbuf.data(), h, w, GBUFFER_CHANNELS);
        if (!ok) { std::fprintf(stderr, "Could not write files in %s\n", dir.string().c_str()); return 1; }
        if (opt.preview)          write_image_preview(dir / "image.ppm", buf.image, w, h);
        if (opt.gbuffer_previews) write_gbuffer_preview_files(dir, buf.gbuf, w, h);
        write_scene_json(dir / "scene.json", desc, index, opt.seed, split, w, h, real_spp);

        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        index_csv << opt.level << ',' << LEVELS[opt.level - 1].name << ',' << index << ',' << split << ','
                  << (fs::path(level_folder(opt.level)) / name).generic_string() << ','
                  << desc.num_objects << ',' << secs << '\n';
        index_csv.flush();

        seconds_sum += secs;
        done++;
        double eta_min = seconds_sum / done * (opt.count - k - 1) / 60.0;
        std::printf("[%d/%d] %s  %-5s  %2d object%s  %5.1f s   (about %.0f min left)\n",
                    k + 1, opt.count, name, split, desc.num_objects, desc.num_objects == 1 ? " " : "s",
                    secs, eta_min);
    }

    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all).count();
    std::printf("Done: %d new scenes in %.1f s\n", done, total);

    GEN_CHECK(cudaFree(buf.d_rand));
    GEN_CHECK(cudaFree(buf.d_accum));
    GEN_CHECK(cudaFree(buf.d_gbuf));
    GEN_CHECK(cudaFree(d_desc));
    GEN_CHECK(cudaFree(d_scene));
    GEN_CHECK(cudaFree(d_world));
    GEN_CHECK(cudaFree(d_lights));
    return 0;
}
