#ifndef NPY_H
#define NPY_H

#include <fstream>
#include <string>

// Saves a height x width x channels array of floats in NumPy's .npy format,
// so Python can read it with  img = np.load("file.npy")  (shape (height, width, channels)).
// Values are stored exactly as given: linear, unclamped, full float precision.
inline bool write_npy(const std::string& path, const float* data, int height, int width, int channels) {
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
                       + std::to_string(height) + ", " + std::to_string(width) + ", "
                       + std::to_string(channels) + "), }";
    // The fixed start (10 bytes) + header + '\n' must add up to a multiple of 64.
    size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header += '\n';

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write("\x93NUMPY", 6);                           // magic string
    out.put(1); out.put(0);                              // format version 1.0
    unsigned short len = (unsigned short)header.size();  // header length, little-endian
    out.put(char(len & 0xff)); out.put(char(len >> 8));
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char*>(data),
              std::streamsize(sizeof(float) * size_t(height) * width * channels));
    return bool(out);
}

#endif