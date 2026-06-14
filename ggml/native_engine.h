#pragma once

/**
 * Hand-rolled streaming engine for the compact LocalVQE backend —
 * a development/reference implementation tuned for one specific CPU
 * (built -march=znver4; NOT part of release builds, which use the
 * portable GGML graph). Direct convs on persistent pre-padded windows
 * (no im2col, no concat, no graph dispatch), fused norm/act, FFT codec
 * via radix-2 rfft (not basis matmul).
 *
 * Semantics replicated 1:1 from the graph builders in
 * localvqe_graph.cpp (CausalGroupNorm + SiLU blocks, S4D, subpixel,
 * CCM) and validated per-layer against the graph's debug taps; the
 * regression fixtures pin both implementations end-to-end.
 *
 * Layouts: per-frame tensors are (C, F) channel-major, each channel's F
 * contiguous. Conv windows are (C, kh, Fp) pre-padded in freq; stride-2
 * conv windows store rows de-interleaved as [even | odd] halves so the
 * inner loops stay unit-stride.
 */

#include "localvqe_model.h"
#include <vector>
#include <cstdint>

struct ne_conv {
    int C_in = 0, C_out = 0, kh = 0, kw = 0, sF = 0;
    int F_in = 0, F_out = 0, Fp = 0;        // Fp = F_in + kw - 1
    std::vector<float> w;                   // (C_out, C_in, kh, kw) as stored
    std::vector<float> b;                   // (C_out)
    std::vector<float> win;                 // (C_in, kh, Fp); stride2: [ev|od]
    std::vector<uint32_t> wbf;              // bf16 r-pair weights (see .cpp)
};

struct ne_norm { std::vector<float> g, b; };

struct ne_block {                            // encoder or residual-ish block
    ne_norm n1, n2;
    ne_conv c1, c2;
};

struct ne_dec {
    ne_norm skip_n, res_n, dec_n;
    std::vector<float> skip_w, skip_b;       // 1x1: (C_out, C_in)
    ne_conv res, deconv;
};

struct native_engine {
    bool loaded = false;
    int F = 256, dmax = 16, Hal = 16, bn_h = 80;
    float power_c = 0.3f;

    // codec
    std::vector<float> win512;               // sqrt-Hann

    ne_block enc[7];                         // mic1,2 far1,2 mic3,4,5
    ne_dec dec[5];                           // dec5..dec1

    // align
    std::vector<float> pmw, pmb, prw, prb;   // 1x1 H x C
    std::vector<float> sw, sb;               // smooth (1,H,5,3) + (1)
    std::vector<float> K_win;                // (H, dmax, F2)
    std::vector<float> ref_win;              // (C2, dmax, F2)
    std::vector<float> S_win;                // (H, 5, dmax+2)

    // s4d
    std::vector<float> s4d_inw, s4d_inb, s4d_outw, s4d_outb;
    std::vector<float> s4d_ar, s4d_ai, s4d_Br, s4d_Bi, s4d_Cr, s4d_Ci, s4d_D;
    std::vector<float> s4d_hr, s4d_hi;

    // ccm
    std::vector<float> ccm_win;              // (2, 3, F+2) raw spectrum window

    // scratch
    std::vector<float> t0, t1, t2, t3, t4, skipsave[6];
};

bool ne_init(native_engine& ne, const localvqe_model& m);
void ne_reset(native_engine& ne);
/// Drop-in for process_frame_graph: 512-sample PCM windows in, synthesis
/// window out (caller OLAs).
void ne_process_frame(native_engine& ne, const float* mic_win,
                      const float* ref_win, float* out_win);
/// Debug taps: after a frame, these hold the latest per-block outputs
/// (C, F) — populated only when ne_debug_taps(true) was called.
void ne_debug_taps(bool on);
const std::vector<std::pair<std::string, std::vector<float>>>& ne_taps();
