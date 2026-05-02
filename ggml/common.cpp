/**
 * Common utilities for LocalVQE GGML inference and block tests.
 *
 * Implements:
 * - Minimal NumPy .npy reader/writer for f32 C-contiguous arrays
 * - Comparison helpers
 */

#include "common.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

// ── NpyArray ────────────────────────────────────────────────────────────────

int64_t NpyArray::numel() const {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

// ── .npy format ─────────────────────────────────────────────────────────────
//
// NumPy .npy format (v1.0):
//   6 bytes: magic "\x93NUMPY"
//   1 byte:  major version (1)
//   1 byte:  minor version (0)
//   2 bytes: header length (little-endian uint16)
//   N bytes: ASCII header dict, padded with spaces + \n to 64-byte alignment
//   rest:    raw data
//
// Header dict example: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 257, 20, 2), }\n"

static std::vector<int64_t> parse_npy_shape(const std::string& header) {
    // Find 'shape': (...)
    auto pos = header.find("'shape'");
    if (pos == std::string::npos)
        throw std::runtime_error("npy: no 'shape' in header");

    auto paren_start = header.find('(', pos);
    auto paren_end = header.find(')', paren_start);
    if (paren_start == std::string::npos || paren_end == std::string::npos)
        throw std::runtime_error("npy: malformed shape");

    std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

    std::vector<int64_t> shape;
    size_t i = 0;
    while (i < shape_str.size()) {
        // Skip whitespace and commas
        while (i < shape_str.size() && (shape_str[i] == ' ' || shape_str[i] == ','))
            i++;
        if (i >= shape_str.size()) break;

        // Parse integer
        int64_t val = 0;
        while (i < shape_str.size() && shape_str[i] >= '0' && shape_str[i] <= '9') {
            val = val * 10 + (shape_str[i] - '0');
            i++;
        }
        shape.push_back(val);
    }
    return shape;
}

NpyArray npy_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("npy_load: cannot open " + path);

    // Read magic
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_load: bad magic in " + path);

    // Read version
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else if (major == 2) {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("npy_load: unsupported version " +
                                 std::to_string(major) + "." + std::to_string(minor));
    }

    // Read header string
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Verify dtype is float32 little-endian
    if (header.find("'<f4'") == std::string::npos &&
        header.find("'float32'") == std::string::npos) {
        throw std::runtime_error("npy_load: expected float32, got header: " + header);
    }

    // Verify C-contiguous (not Fortran order)
    if (header.find("True") != std::string::npos) {
        throw std::runtime_error("npy_load: Fortran order not supported");
    }

    // Parse shape
    auto shape = parse_npy_shape(header);

    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    // Read data
    NpyArray arr;
    arr.shape = shape;
    arr.data.resize(numel);
    f.read(reinterpret_cast<char*>(arr.data.data()), numel * sizeof(float));

    if (!f)
        throw std::runtime_error("npy_load: short read in " + path);

    return arr;
}

void npy_save(const std::string& path, const float* data,
              const std::vector<int64_t>& shape) {
    // Build header dict
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size()) shape_str += ", ";
        else if (shape.size() == 1) shape_str += ",";  // trailing comma for 1-d
    }
    shape_str += ")";

    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                       shape_str + ", }";

    // Pad header to 64-byte alignment (magic=6 + version=2 + header_len=2 + header)
    size_t preamble = 6 + 1 + 1 + 2;
    size_t total = preamble + dict.size() + 1;  // +1 for \n
    size_t padding = (64 - (total % 64)) % 64;
    dict += std::string(padding, ' ') + "\n";

    uint16_t header_len = (uint16_t)dict.size();

    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("npy_save: cannot open " + path);

    // Write magic + version
    f.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    f.write(reinterpret_cast<char*>(&major), 1);
    f.write(reinterpret_cast<char*>(&minor), 1);
    f.write(reinterpret_cast<char*>(&header_len), 2);
    f.write(dict.data(), dict.size());
    f.write(reinterpret_cast<const char*>(data), numel * sizeof(float));
}

// ── GGUF tensor loading ────────────────────────────────────────────────────

NpyArray load_tensor_from_ggml(struct ggml_context* ctx,
                               const std::string& name,
                               struct gguf_context* gctx,
                               bool verbose) {
    struct ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "Missing tensor: %s\n", name.c_str());
        return {};
    }

    NpyArray arr;
    int nd = ggml_n_dims(t);
    for (int d = nd - 1; d >= 0; d--)
        arr.shape.push_back(t->ne[d]);

    int64_t n = ggml_nelements(t);
    arr.data.resize(n);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(arr.data.data(), t->data, n * sizeof(float));
    } else {
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, arr.data.data(), n);
            if (verbose)
                printf("  Dequantized %s (%s -> F32, %lld elements)\n",
                       name.c_str(), ggml_type_name(t->type), (long long)n);
        } else {
            fprintf(stderr, "Unsupported tensor type for %s: %s\n",
                    name.c_str(), ggml_type_name(t->type));
            return {};
        }
    }

    // Quantized tensors may have been flattened to 1D for block-size
    // alignment.  Restore original shape from GGUF metadata if present.
    if (gctx) {
        char key[256];
        snprintf(key, sizeof(key), "localvqe.shape.%s.ndim", name.c_str());
        int ndim_idx = gguf_find_key(gctx, key);
        if (ndim_idx >= 0) {
            int ndim = (int)gguf_get_val_u32(gctx, ndim_idx);
            arr.shape.clear();
            for (int d = 0; d < ndim; d++) {
                snprintf(key, sizeof(key), "localvqe.shape.%s.%d",
                         name.c_str(), d);
                int d_idx = gguf_find_key(gctx, key);
                if (d_idx >= 0)
                    arr.shape.push_back((int64_t)gguf_get_val_u32(gctx, d_idx));
            }
        }
    }

    return arr;
}

// ── Comparison ──────────────────────────────────────────────────────────────

float max_abs_diff(const float* a, const float* b, int64_t n) {
    float max_err = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float err = std::fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float mean_abs_diff(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sum += std::fabs(a[i] - b[i]);
    }
    return (float)(sum / n);
}

bool print_result(const std::string& name, float max_err, float mean_err,
                  float ok_threshold, float fail_threshold) {
    const char* status;
    if (max_err < ok_threshold)
        status = "OK";
    else if (max_err < fail_threshold)
        status = "WARN";
    else
        status = "FAIL";

    printf("  [%s] %s: max=%.2e mean=%.2e\n", status, name.c_str(), max_err, mean_err);
    return max_err < fail_threshold;
}

