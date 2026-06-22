#pragma once

/**
 * GTCRN-AEC — standalone C++ inference for the 48,965-param AEC-aware GTCRN
 * backend (ERB sub-band + SFE + grouped-conv encoder + 2x grouped dual-path
 * RNN + grouped-deconv decoder + complex-ratio mask, own 512/256 STFT).
 *
 * This arch shares nothing with the LocalVQE engine (S4D/AlignBlock/CCM/DCT),
 * so it is a separate implementation. Weights come from a GGUF produced by
 * train/export_gtcrn_ggml.py, which folds every BatchNorm into its conv and
 * rewrites all ConvTranspose2d weights into equivalent regular-conv weights,
 * leaving one conv code path. The forward here is a 1:1 translation of that
 * exporter's validated numpy/torch reference (<1e-4 vs PyTorch).
 *
 * Layout: PyTorch-style, batch=1. Activations are flat (C, T, F) row-major.
 * Spectra are flat (F=257, T, 2) row-major (matching the exporter fixtures).
 */

#include "common.h"  // NpyArray

#include <map>
#include <string>
#include <vector>

// 3D activation tensor (C, T, F), batch=1, row-major: idx = (c*T + t)*F + f.
struct GTensor {
    std::vector<float> d;
    int C = 0, T = 0, F = 0;
    float& at(int c, int t, int f) { return d[((size_t)c * T + t) * F + f]; }
    float at(int c, int t, int f) const { return d[((size_t)c * T + t) * F + f]; }
    void resize(int c, int t, int f) { C = c; T = t; F = f; d.assign((size_t)c * t * f, 0.0f); }
    NpyArray to_npy() const {
        NpyArray a; a.shape = {1, C, T, F}; a.data = d; return a;
    }
};

struct GtcrnModel {
    std::map<std::string, NpyArray> W;  // normalized weights (numpy order/data)

    /// Load weights from the GGUF written by export_gtcrn_ggml.py.
    bool load(const char* gguf_path, bool verbose = false);

    /// Scalar REFERENCE forward (no ggml ops) — kept as the parity oracle for
    /// the real ggml-graph path below. spec_e/spec_y are (257,T,2) row-major;
    /// returns (257,T,2). `cap` stores per-stage activations if non-null.
    std::vector<float> forward(const float* spec_e, const float* spec_y, int T,
                               std::map<std::string, NpyArray>* cap = nullptr) const;

    /// STFT/ISTFT matching torch (n_fft=512, hop=256, hann^0.5, center=True),
    /// via the windowed-DFT matrices shipped in the GGUF (stft.*). These are the
    /// audio<->spectrogram framing layer around the network.
    int n_frames(int L) const { return 1 + L / 256; }
    std::vector<float> stft(const float* sig, int L) const;          // -> (257, n_frames, 2)
    std::vector<float> istft(const float* spec, int T, int L) const; // -> (L,)

    /// Single-frame primitives for streaming OLA (no padding, no envelope
    /// normalization — the caller owns the 512-sample analysis buffer and the
    /// overlap-add). stft_frame: 512 windowed samples -> 257 (re,im).
    /// istft_frame: 257 (re,im) -> 512 samples (pre-OLA). win2() is the
    /// synthesis window energy (512) for the OLA divisor.
    void stft_frame(const float* fr512, float* re257, float* im257) const;
    void istft_frame(const float* re257, const float* im257, float* ft512) const;
    const std::vector<float>& win2() const { return W.at("stft.win2").data; }
};

// Forward decls (avoid pulling ggml headers into this header).
struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;

/// Real GGML inference: the whole GTCRN-AEC forward built as a ggml_cgraph of
/// ggml ops and dispatched through a backend (CPU SIMD variants by default).
/// Feed-forward layers are batched ggml ops over all T; the GRU recurrences
/// (intra over freq, inter/TRA over time) are unrolled in the graph builder.
struct GtcrnGraph {
    ggml_backend_t backend = nullptr;
    ggml_context* wctx = nullptr;             // weight tensors (own backend buffer)
    ggml_backend_buffer_t wbuf = nullptr;
    std::map<std::string, ggml_tensor*> wt;   // name -> weight tensor

    bool load(const char* gguf_path, int n_threads = 1, bool verbose = false);
    ~GtcrnGraph();

    /// Build + run the ggml graph for one clip (whole-T, recurrences unrolled).
    /// spec_e/spec_y (257,T,2) row-major, returns (257,T,2). If `cap` non-null,
    /// captures per-stage activations. If `compute_ms` non-null, stores the pure
    /// ggml_backend_graph_compute time.
    std::vector<float> forward(const float* spec_e, const float* spec_y, int T,
                               std::map<std::string, NpyArray>* cap = nullptr,
                               double* compute_ms = nullptr) const;

    /// STREAMING form: a T=1 graph built ONCE, run per frame with carried state
    /// (depthwise-conv history, TRA + inter-GRU hidden). This is the deployment
    /// path and the honest per-frame cost. Same I/O as forward(); cap captures
    /// out_spec only. compute_ms = summed per-frame ggml compute time.
    std::vector<float> forward_stream(const float* spec_e, const float* spec_y, int T,
                                      std::map<std::string, NpyArray>* cap = nullptr,
                                      double* compute_ms = nullptr) const;
};

/// Persistent streaming session: builds the T=1 graph + allocator ONCE, then
/// step() runs one frame, carrying recurrent state across calls (the real-time
/// per-hop path, e.g. the C API frame interface). Mirrors the graph in
/// GtcrnGraph::forward_stream — keep the two in sync. Not thread-safe; one
/// session per stream.
struct GtcrnStream {
    ggml_backend_t backend = nullptr;
    ggml_context* sctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_tensor *re_e = nullptr, *im_e = nullptr, *re_y = nullptr, *im_y = nullptr;
    ggml_tensor *zeros = nullptr, *out = nullptr;
    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> states;  // (in, out) per recurrence
    std::vector<std::vector<float>> sbuf;                       // carried host state
    std::vector<float> zbuf;

    bool begin(const GtcrnGraph& g);   // build + allocate + zero state
    void reset();                      // zero carried state (new utterance)
    /// One frame. spec_e/spec_y: 257*2 interleaved (re,im); out_spec: 257*2.
    void step(const float* spec_e, const float* spec_y, float* out_spec);
    ~GtcrnStream();
};
