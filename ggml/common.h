#pragma once

/**
 * Common utilities for LocalVQE GGML inference and block tests.
 *
 * - NumPy .npy file I/O (f32, C-contiguous only)
 * - Comparison helpers (max/mean absolute error)
 * - Result reporting
 */

#include <cstdint>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── NumPy .npy I/O ──────────────────────────────────────────────────────────

struct NpyArray {
    std::vector<float> data;
    std::vector<int64_t> shape;

    int64_t numel() const;
    int64_t dim(int i) const { return shape[i]; }
    int ndim() const { return (int)shape.size(); }
};

/// Load a .npy file (float32, C-contiguous). Throws on error.
NpyArray npy_load(const std::string& path);

/// Save a .npy file (float32, C-contiguous).
void npy_save(const std::string& path, const float* data,
              const std::vector<int64_t>& shape);

inline void npy_save(const std::string& path, const NpyArray& arr) {
    npy_save(path, arr.data.data(), arr.shape);
}

// Audio file I/O is in audio_io.h.

// ── GGUF tensor loading ────────────────────────────────────────────────────

struct ggml_context;  // forward decl — avoid pulling in ggml.h
struct gguf_context;

/// Load tensor from ggml context, dequantizing quantized types to float32.
/// If gctx is non-null, restores original shape from GGUF metadata
/// (quantized conv weights are stored flattened to 1D).
/// Returns empty NpyArray on failure.
NpyArray load_tensor_from_ggml(struct ggml_context* ctx,
                               const std::string& name,
                               struct gguf_context* gctx = nullptr,
                               bool verbose = false);

// ── Comparison ──────────────────────────────────────────────────────────────

/// Maximum absolute difference between two arrays.
float max_abs_diff(const float* a, const float* b, int64_t n);

/// Mean absolute difference between two arrays.
float mean_abs_diff(const float* a, const float* b, int64_t n);

/// Print comparison result with OK/WARN/FAIL classification.
/// Returns true if max_err < fail_threshold (1e-2).
bool print_result(const std::string& name, float max_err, float mean_err,
                  float ok_threshold = 1e-4f, float fail_threshold = 1e-2f);

