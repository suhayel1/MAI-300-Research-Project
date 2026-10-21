#ifndef MAT3_H
#define MAT3_H

// 3x3 matrix for rotating, scaling and shearing points and directions.
// Include after rtweekend.h (it uses vec3 and degrees_to_radians).
//
// Matrices combine by multiplying, and the RIGHTMOST one is applied first:
//     rotation_y(30) * scaling(vec3(2, 1, 1))   scales first, then rotates.
//
// Everything is __host__ __device__, so matrices can be built on the CPU (for
// example in a scene description) and copied to the GPU as plain data.
class mat3 {
  public:
    float m[3][3];   // m[row][column]

    __host__ __device__ mat3() : m{{1,0,0},{0,1,0},{0,0,1}} {}   // identity

    __host__ __device__ mat3(float a00, float a01, float a02,
                             float a10, float a11, float a12,
                             float a20, float a21, float a22)
      : m{{a00,a01,a02},{a10,a11,a12},{a20,a21,a22}} {}

    __host__ __device__ vec3 operator*(const vec3& v) const {
        return vec3(m[0][0]*v.x() + m[0][1]*v.y() + m[0][2]*v.z(),
                    m[1][0]*v.x() + m[1][1]*v.y() + m[1][2]*v.z(),
                    m[2][0]*v.x() + m[2][1]*v.y() + m[2][2]*v.z());
    }

    __host__ __device__ mat3 operator*(const mat3& b) const {
        mat3 r;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r.m[i][j] = m[i][0]*b.m[0][j] + m[i][1]*b.m[1][j] + m[i][2]*b.m[2][j];
        return r;
    }

    __host__ __device__ mat3 transpose() const {
        return mat3(m[0][0], m[1][0], m[2][0],
                    m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2]);
    }

    __host__ __device__ float determinant() const {
        return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
             - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
             + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    }

    // Only valid if determinant() != 0 (no zero scale factors).
    __host__ __device__ mat3 inverse() const {
        float inv_det = 1.0f / determinant();
        return mat3(
            (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * inv_det,
            (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv_det,
            (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv_det,
            (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * inv_det,
            (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv_det,
            (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv_det,
            (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * inv_det,
            (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv_det,
            (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv_det);
    }
};

// ---- Building blocks ----------------------------------------------------------
// Angles in degrees. Rotations use the same direction as the rotate_x / rotate_y /
// rotate_z classes (right-hand rule): positive x turns +y toward +z, positive y turns
// +z toward +x, positive z turns +x toward +y.

__host__ __device__ inline mat3 scaling(const vec3& s) {
    return mat3(s.x(), 0,     0,
                0,     s.y(), 0,
                0,     0,     s.z());
}

__host__ __device__ inline mat3 rotation_x(float angle) {
    float r = degrees_to_radians(angle), c = std::cos(r), s = std::sin(r);
    return mat3(1, 0,  0,
                0, c, -s,
                0, s,  c);
}

__host__ __device__ inline mat3 rotation_y(float angle) {
    float r = degrees_to_radians(angle), c = std::cos(r), s = std::sin(r);
    return mat3( c, 0, s,
                 0, 1, 0,
                -s, 0, c);
}

__host__ __device__ inline mat3 rotation_z(float angle) {
    float r = degrees_to_radians(angle), c = std::cos(r), s = std::sin(r);
    return mat3(c, -s, 0,
                s,  c, 0,
                0,  0, 1);
}

// All three rotations at once: about x first, then y, then z (degrees).
// The order matters; this is the same as rotation_z(z) * rotation_y(y) * rotation_x(x).
__host__ __device__ inline mat3 rotation(float x_angle, float y_angle, float z_angle) {
    return rotation_z(z_angle) * rotation_y(y_angle) * rotation_x(x_angle);
}

// Turns about the OBJECT's own axes, in the order given: about its x axis first, then
// its (already turned) y axis, then its (twice turned) z axis. Degrees.
// Same as rotation_x(x) * rotation_y(y) * rotation_z(z).
__host__ __device__ inline mat3 rotation_local(float x_angle, float y_angle, float z_angle) {
    return rotation_x(x_angle) * rotation_y(y_angle) * rotation_z(z_angle);
}

// Rotation by `angle` degrees about any axis through the origin (right-hand rule).
// The axis doesn't need to be unit length. rotation_axis(vec3(0,1,0), a) == rotation_y(a).
__host__ __device__ inline mat3 rotation_axis(const vec3& axis, float angle) {
    vec3 k = unit_vector(axis);
    float r = degrees_to_radians(angle), c = std::cos(r), s = std::sin(r), t = 1 - c;
    float x = k.x(), y = k.y(), z = k.z();
    return mat3(c + x*x*t,    x*y*t - z*s,  x*z*t + y*s,
                y*x*t + z*s,  c + y*y*t,    y*z*t - x*s,
                z*x*t - y*s,  z*y*t + x*s,  c + z*z*t);
}

// A random orientation, every orientation equally likely, from three random numbers
// in [0,1) (from any random generator, CPU or GPU). Note that picking the three
// angles of rotation(x, y, z) uniformly at random does NOT do this: it favours
// some orientations over others.
__host__ __device__ inline mat3 random_rotation(float u1, float u2, float u3) {
    // Random unit quaternion (Shoemake, "Uniform random rotations", Graphics Gems III, 1992).
    float a = std::sqrt(1 - u1), b = std::sqrt(u1);
    float t2 = 2 * pi * u2, t3 = 2 * pi * u3;
    float x = a * std::sin(t2), y = a * std::cos(t2), z = b * std::sin(t3), w = b * std::cos(t3);
    return mat3(1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
                2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
                2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y));
}

// Shear: each coordinate is pushed along by the others.
//   x' = x + xy*y + xz*z
//   y' = y + yx*x + yz*z
//   z' = z + zx*x + zy*y
// e.g. shearing(0.5f, 0, 0, 0, 0, 0) leans an object over in +x as y increases.
__host__ __device__ inline mat3 shearing(float xy, float xz, float yx, float yz, float zx, float zy) {
    return mat3(1,  xy, xz,
                yx, 1,  yz,
                zx, zy, 1);
}

// ---- Placements that include a move ---------------------------------------------
// affine = a full placement: a matrix (rotate / scale / shear) followed by a move,
//     p  ->  M * p + t
// This is what transform takes. Combine with *, rightmost applied first, exactly like
// mat3. A plain mat3 mixes in automatically, so translations can go anywhere in the chain:
//     translation(vec3(3,0,1)) * rotation_y(30) * scaling(vec3(2,1,1))
// Rotating about a point c instead of the origin:
//     translation(c) * rotation_z(30) * translation(-c)
class affine {
  public:
    mat3 M;   // rotate / scale / shear
    vec3 t;   // then move

    __host__ __device__ affine() {}
    __host__ __device__ affine(const mat3& M, const vec3& t = vec3(0, 0, 0)) : M(M), t(t) {}

    // Move a point: rotate / scale / shear it, then move it.
    __host__ __device__ point3 apply_point(const point3& p) const { return M * p + t; }

    // Turn a direction: rotate / scale / shear only. Directions have no position,
    // so moving doesn't apply to them.
    __host__ __device__ vec3 apply_vector(const vec3& v) const { return M * v; }

    // The placement that undoes this one. Only valid if M has no zero scale factors.
    __host__ __device__ affine inverse() const {
        mat3 M_inv = M.inverse();
        return affine(M_inv, -(M_inv * t));
    }

    // (this * b)(p) = this(b(p))
    __host__ __device__ affine operator*(const affine& b) const { return affine(M * b.M, M * b.t + t); }
};

// mat3 * affine (affine * mat3 already works through the conversion above).
__host__ __device__ inline affine operator*(const mat3& a, const affine& b) { return affine(a) * b; }

__host__ __device__ inline affine translation(const vec3& offset) { return affine(mat3(), offset); }

#endif
