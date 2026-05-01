#pragma once

/**
 * LocalVQE model inference using GGML computational graphs.
 *
 * All inference uses GGML graph ops dispatched to CPU (with SIMD) or GPU.
 */

#include "localvqe_model.h"  // localvqe_hparams, NpyArray

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <map>
#include <string>
#include <vector>

struct dvqe_graph_model {
    localvqe_hparams hparams;

    // Backend (CPU or CUDA)
    ggml_backend_t backend = nullptr;

    // Weight storage — kept as ggml tensors (may be quantized)
    struct ggml_context* weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    // Named weight lookup
    std::map<std::string, struct ggml_tensor*> weights;

    struct ggml_tensor* w(const std::string& name) const {
        auto it = weights.find(name);
        if (it == weights.end()) return nullptr;
        return it->second;
    }
};

/// Load model into GGML graph format. Weights stay as ggml_tensor*
/// (potentially quantized). n_threads=0 means auto (nproc - 1, min 1).
/// Defaults to the CPU backend, device 0.
bool load_graph_model(const char* path, dvqe_graph_model& model,
                      bool verbose = true, int n_threads = 0);

/// Same as load_graph_model() but with an explicit backend + device. Useful
/// when ggml_backend_load_all() has registered multiple GPU backends and
/// the caller wants to pick a specific one. backend_name matches
/// ggml_backend_reg_name() (e.g. "CPU", "Vulkan", "CUDA").
bool load_graph_model_ex(const char* path, dvqe_graph_model& model,
                         const char* backend_name, int device_index,
                         bool verbose, int n_threads);

/// Print every registered backend + device to `out`. Indices passed to
/// load_graph_model_ex() match the per-backend ordering printed here.
/// Calls ggml_backend_load_all() lazily.
void dvqe_list_devices(FILE* out);

/// Free all resources.
void free_graph_model(dvqe_graph_model& model);

/// Per-block timing result.
struct block_timing {
    std::string name;
    double us;  // microseconds
};

// ── Streaming graph (T=1 per frame, with history buffers) ───────────────────

/// Pre-built GGML graph for frame-by-frame streaming inference.
/// Built once, reused for every frame. History persists between frames.
struct dvqe_stream_graph {
    struct ggml_context* ctx = nullptr;
    struct ggml_cgraph* graph = nullptr;
    ggml_gallocr_t galloc = nullptr;

    // Top-level inputs: 512-sample PCM windows (fed to DCT-II analysis head).
    struct ggml_tensor* mic_pcm_in = nullptr;   // (512,)
    struct ggml_tensor* ref_pcm_in = nullptr;   // (512,)

    // Post-DCT views of the analysis basis output, shape (2, 1, F). Used
    // internally (fed to feature extraction and CCM); not set by callers.
    struct ggml_tensor* mic_in = nullptr;   // (2, 1, F)
    struct ggml_tensor* ref_in = nullptr;   // (2, 1, F)

    // Per-conv-layer history: input (F_i, kh-1, C_i), output (F_i, kh-1, C_i)
    std::vector<struct ggml_tensor*> conv_hist_in;
    std::vector<struct ggml_tensor*> conv_hist_out;

    // S4D bottleneck hidden state (complex diagonal)
    struct ggml_tensor* s4d_h_real_in = nullptr;
    struct ggml_tensor* s4d_h_real_out = nullptr;
    struct ggml_tensor* s4d_h_imag_in = nullptr;
    struct ggml_tensor* s4d_h_imag_out = nullptr;

    // AlignBlock: K and ref histories (dmax-1 frames each)
    struct ggml_tensor* align_K_hist_in = nullptr;
    struct ggml_tensor* align_K_hist_out = nullptr;
    struct ggml_tensor* align_ref_hist_in = nullptr;
    struct ggml_tensor* align_ref_hist_out = nullptr;

    // AlignBlock: smooth conv history (kh time frames)
    struct ggml_tensor* align_smooth_hist_in = nullptr;
    struct ggml_tensor* align_smooth_hist_out = nullptr;

    // CCM: mic encoded history (2 time frames)
    struct ggml_tensor* ccm_hist_in = nullptr;
    struct ggml_tensor* ccm_hist_out = nullptr;

    // Pre-DCT-inverse tensor (2, 1, F) — kept for profile visibility.
    struct ggml_tensor* enhanced_out = nullptr;

    // Top-level output: 512-sample PCM window (DCT-II synthesis of enhanced_out).
    struct ggml_tensor* enhanced_pcm_out = nullptr;

    // Persistent scratch for history copies (avoids per-frame heap allocation)
    std::vector<uint8_t> hist_scratch;
};

/// Build the streaming graph (T=1). Call once, then process_frame_graph() per frame.
bool build_stream_graph(dvqe_graph_model& m, dvqe_stream_graph& sg);

/// Process one frame. mic/ref are 512-sample PCM windows (50% overlap:
/// last hop is 256 new samples, first half is previous 256). out is a
/// 512-sample PCM frame to be overlap-added by the caller.
void process_frame_graph(dvqe_stream_graph& sg, dvqe_graph_model& m,
                         const float* mic_pcm_window,
                         const float* ref_pcm_window,
                         float* enhanced_pcm_window);

/// Zero all history buffers (call before processing a new utterance).
void reset_stream_graph(dvqe_stream_graph& sg, dvqe_graph_model& m);

/// Free streaming graph resources.
void free_stream_graph(dvqe_stream_graph& sg);

/// Print weight-buffer + activation-buffer + history-scratch sizes.
void print_memory_budget(const dvqe_graph_model& m, const dvqe_stream_graph& sg);

/// Print the op-type histogram of a built graph (count per GGML op kind).
void print_op_histogram(const struct ggml_cgraph* graph);
