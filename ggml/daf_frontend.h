#pragma once

/**
 * DAF front-end: gated GCC-PHAT prealignment + PBFDKF (M=128, N=128, k=2)
 * + the 2,742-param v2xp controller. Subtractive: e = mic - y_hat; the
 * backend mask consumes (e, yhat = mic - e).
 *
 * Hand-rolled C++ (sequential DSP — wrong shape for a ggml graph). Weights
 * and configuration come from the gguf "localvqe.daf.*" KVs and "daf.*"
 * tensors; the filter/controller constants below mirror the PyTorch
 * reference the weights were trained with, and the implementation is
 * verified against it per-stage (regression fixtures pin it end-to-end).
 */

#include "localvqe_graph.h"
#include <vector>

struct daf_frontend {
    bool loaded = false;
    bool enable_prealign = true;  // off => cur_shift stays 0 (parity tests)

    // ---- filter constants ----
    static constexpr int M = 128;          // block (8 ms)
    static constexpr int N = 128;          // partitions (1.024 s span)
    static constexpr int NFFT = 256;
    static constexpr int NB = 129;         // bins
    static constexpr int K_ITER = 2;       // data-reuse updates per block
    static constexpr float A_DECAY = 0.999f;
    static constexpr float P_INIT = 1e3f;

    // ---- GCC constants ----
    static constexpr int G_WIN = 16384;
    static constexpr int G_HOP = 8000;
    static constexpr int G_NFFT = 32768;
    static constexpr int G_MAXLAG = 16384;
    static constexpr float G_GATE_DB = 26.0f;
    static constexpr float G_CONF_THR = 8.0f;

    // controller weights (flattened, row-major as in pytorch)
    std::vector<float> g_ln_w, g_ln_b, g_wih, g_whh, g_bih, g_bhh;       // glob
    std::vector<float> b_ln_w, b_ln_b, b_wih, b_whh, b_bih, b_bhh;       // bins
    std::vector<float> p_ln_w, p_ln_b, p_wih, p_whh, p_bih, p_bhh;       // part
    std::vector<float> head_w, head_b, p_head_w, p_head_b;

    // filter state
    std::vector<float> Hr, Hi;             // (N, NB)
    std::vector<float> Xr_r, Xr_i;         // (N, NB) ring, newest at p=0
    std::vector<float> P;                  // (N, NB)
    std::vector<float> x_old;              // (M)
    int p_idx = 0;

    // controller state
    std::vector<float> hg;                 // (8)
    std::vector<float> hb;                 // (NB, 16)
    std::vector<float> hp;                 // (N, 8)

    // GCC state
    std::vector<float> gcc_Sr, gcc_Si;     // (G_NFFT/2+1)
    std::vector<float> mic_ring, ref_ring; // G_WIN history for windows
    long long n_seen = 0;
    float gcc_maxrms = 1e-12f;
    float gcc_conf = 0.0f;                 // confidence of the last gcc_update
    int cur_shift = 0;                     // applied ref delay (samples)
    bool gcc_locked = false;               // shift frozen after first confident
                                           // estimate past LOCK_AFTER (stable
                                           // ref stream; re-shifts force filter
                                           // re-convergence)
    static constexpr long long LOCK_AFTER = 56000;  // 3.5 s                     // applied ref delay (samples)

    // ref delay line for shift application
    std::vector<float> ref_dline;          // G_MAXLAG + M
    size_t ref_dpos = 0;

    // scratch
    std::vector<float> scratch;
};

/// Load daf.* tensors from the model. Returns false (and leaves
/// fe.loaded=false) when the gguf carries no front-end.
bool daf_init(daf_frontend& fe, const dvqe_graph_model& m);

/// Same, from the NpyArray-based loader (native engine path).
#include "localvqe_model.h"
bool daf_init_npy(daf_frontend& fe, const localvqe_model& m);

/// Init directly from a raw NpyArray tensor map — used when a GTCRN gguf embeds
/// the DAF front-end under daf.* (a self-contained compact-line model). Returns
/// false if no daf.* tensors are present.
bool daf_init_tensors(daf_frontend& fe, const std::map<std::string, NpyArray>& tensors);

/// Reset all online state (new utterance).
void daf_reset(daf_frontend& fe);

/// One-shot global bulk-delay estimate over the whole (already-available)
/// mic/ref signal, for non-real-time callers (file/demo). Seeds cur_shift and
/// freezes it (gcc_locked) so the standalone filter is aligned from frame 0 —
/// no GCC acquisition window, no mid-stream re-shift. Leaves the filter state
/// reset, ready for daf_process. Streaming/real-time callers skip this and get
/// the normal online acquisition.
void daf_prime_delay(daf_frontend& fe, const float* mic, const float* ref, int n);

/// Process one hop of `hop` samples (must be a multiple of M=128).
/// e_out receives the echo-cancelled mic; yhat_out (optional, may be null)
/// receives mic - e.
void daf_process(daf_frontend& fe, const float* mic, const float* ref,
                 int hop, float* e_out, float* yhat_out);
