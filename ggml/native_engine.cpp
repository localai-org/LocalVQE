#include "native_engine.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <string>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

// ── radix-2 FFT (same as daf_frontend; small sizes, instruction-light) ─────
static void fft_ip(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = (inv ? 2.0 : -2.0) * M_PI / (double)len;
        const double wr = cos(ang), wi = sin(ang);
        for (size_t i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (size_t k = 0; k < len / 2; k++) {
                const float ur = re[i+k], ui = im[i+k];
                const float vr = (float)(re[i+k+len/2]*cr - im[i+k+len/2]*ci);
                const float vi = (float)(re[i+k+len/2]*ci + im[i+k+len/2]*cr);
                re[i+k] = ur + vr;       im[i+k] = ui + vi;
                re[i+k+len/2] = ur - vr; im[i+k+len/2] = ui - vi;
                const double nc = cr*wr - ci*wi;
                ci = cr*wi + ci*wr; cr = nc;
            }
        }
    }
    if (inv) {
        const float s = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
    }
}

static bool g_taps_on = false;
static std::vector<std::pair<std::string, std::vector<float>>> g_taps;
void ne_debug_taps(bool on) { g_taps_on = on; }
const std::vector<std::pair<std::string, std::vector<float>>>& ne_taps() { return g_taps; }
static void tap(const char* nm, const float* d, size_t n) {
    if (g_taps_on) g_taps.emplace_back(nm, std::vector<float>(d, d + n));
}

// ── helpers ────────────────────────────────────────────────────────────────
static const std::vector<float>& W(const localvqe_model& m, const std::string& n) {
    auto it = m.tensors.find(n);
    if (it == m.tensors.end()) {
        fprintf(stderr, "native: missing tensor %s\n", n.c_str());
        static std::vector<float> empty;
        return empty;
    }
    return it->second.data;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// CausalGroupNorm over (C, F): x -> (x - mean)/sqrt(var+eps) * g[c] + b[c]
static void norm_cf(const float* x, float* y, int C, int F,
                    const float* g, const float* b) {
    double s = 0.0, s2 = 0.0;
    const int n = C * F;
    for (int i = 0; i < n; i++) { s += x[i]; s2 += (double)x[i] * x[i]; }
    const float mean = (float)(s / n);
    const float var = (float)(s2 / n) - mean * mean;
    const float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int c = 0; c < C; c++) {
        const float a = g[c] * inv, d = b[c] - mean * a;
        const float* xr = x + (size_t)c * F;
        float* yr = y + (size_t)c * F;
        for (int f = 0; f < F; f++) yr[f] = a * xr[f] + d;
    }
}

static inline uint16_t f32_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t r = ((u >> 16) & 1) + 0x7FFF;
    return (uint16_t)((u + r) >> 16);
}

static void conv_setup(ne_conv& cv, const std::vector<float>& w,
                       const std::vector<float>& b, int C_out, int C_in,
                       int kh, int kw, int F_in, int sF) {
    cv.C_in = C_in; cv.C_out = C_out; cv.kh = kh; cv.kw = kw; cv.sF = sF;
    cv.F_in = F_in; cv.Fp = F_in + kw - 1;
    cv.F_out = (cv.Fp - kw) / sF + 1;
    cv.w = w; cv.b = b;
    cv.win.assign((size_t)C_in * kh * cv.Fp + 64, 0.0f);
#ifdef __AVX512BF16__
    // r-pair-interleaved bf16 weights: for tap s, one uint32 packs
    // (w[2p][s], w[2p+1][s]) so a single vdpbf16ps accumulates both rows.
    const int np = kh / 2;
    cv.wbf.assign((size_t)C_out * C_in * np * kw, 0);
    for (int o = 0; o < C_out; o++)
        for (int ci = 0; ci < C_in; ci++)
            for (int p = 0; p < np; p++)
                for (int s = 0; s < kw; s++) {
                    const float w0 = w[(((size_t)o * C_in + ci) * kh + 2*p) * kw + s];
                    const float w1 = w[(((size_t)o * C_in + ci) * kh + 2*p + 1) * kw + s];
                    cv.wbf[(((size_t)o * C_in + ci) * np + p) * kw + s] =
                        (uint32_t)f32_bf16(w0) | ((uint32_t)f32_bf16(w1) << 16);
                }
#endif
}

// Write the current frame (C_in, F_in) into the window's last row at the
// causal-pad column offset, run the direct conv, producing (C_out, F_out).
static void conv_run(ne_conv& cv, const float* x, float* y) {
    const int pad_l = (cv.kw - 1) / 2;
    for (int c = 0; c < cv.C_in; c++)
        std::memcpy(&cv.win[((size_t)c * cv.kh + (cv.kh - 1)) * cv.Fp + pad_l],
                    x + (size_t)c * cv.F_in, cv.F_in * sizeof(float));
    const int FO = cv.F_out, sF = cv.sF;
    // De-interleave stride-2 rows once per (c, r) per frame so the FMA inner
    // loops are unit-stride for both stride cases.
    static thread_local std::vector<float> dei;   // (C_in*kh, 2, half) + slack
    const int half = cv.Fp / 2 + 2;
    if (sF == 2) {
        dei.resize((size_t)cv.C_in * cv.kh * 2 * half + 64);
        for (int c = 0; c < cv.C_in; c++)
            for (int r = 0; r < cv.kh; r++) {
                const float* row = &cv.win[((size_t)c * cv.kh + r) * cv.Fp];
                float* ev = &dei[(((size_t)c * cv.kh + r) * 2 + 0) * half];
                float* od = &dei[(((size_t)c * cv.kh + r) * 2 + 1) * half];
                const int nh = cv.Fp / 2;
                for (int i = 0; i < nh; i++) { ev[i] = row[2*i]; od[i] = row[2*i+1]; }
                ev[nh] = (2*nh < cv.Fp) ? row[2*nh] : 0.0f;
                od[nh] = 0.0f; ev[nh+1] = 0.0f; od[nh+1] = 0.0f;
            }
    }
#ifdef __AVX512BF16__
    // BF16 path (vdpbf16ps): one uint32 lane carries an r-pair of bf16
    // window samples; each dot instruction accumulates BOTH rows of the
    // pair into f32 lanes — 2x MAC throughput vs f32 FMA. Accumulation
    // stays f32; only the conv inputs/weights are rounded to bf16.
    {
        static thread_local std::vector<uint32_t> rp;
        const int np = cv.kh / 2;
        const int L = (sF == 1) ? cv.Fp : half;
        rp.resize((size_t)cv.C_in * np * (sF == 1 ? 1 : 2) * L + 64);
        for (int ci = 0; ci < cv.C_in; ci++)
            for (int p = 0; p < np; p++) {
                if (sF == 1) {
                    const float* r0 = &cv.win[((size_t)ci * cv.kh + 2*p) * cv.Fp];
                    const float* r1 = &cv.win[((size_t)ci * cv.kh + 2*p + 1) * cv.Fp];
                    uint32_t* d = &rp[((size_t)ci * np + p) * L];
                    for (int i = 0; i < cv.Fp; i++)
                        d[i] = (uint32_t)f32_bf16(r0[i]) | ((uint32_t)f32_bf16(r1[i]) << 16);
                } else {
                    for (int par = 0; par < 2; par++) {
                        const float* r0 = &dei[(((size_t)ci * cv.kh + 2*p) * 2 + par) * half];
                        const float* r1 = &dei[(((size_t)ci * cv.kh + 2*p + 1) * 2 + par) * half];
                        uint32_t* d = &rp[(((size_t)ci * np + p) * 2 + par) * L];
                        for (int i = 0; i < half; i++)
                            d[i] = (uint32_t)f32_bf16(r0[i]) | ((uint32_t)f32_bf16(r1[i]) << 16);
                    }
                }
            }
        for (int o = 0; o < cv.C_out; o++) {
            float* yr = y + (size_t)o * FO;
            const float bo = cv.b[o];
            const uint32_t* wo = &cv.wbf[((size_t)o * cv.C_in) * np * cv.kw];
            for (int ft = 0; ft < FO; ft += 32) {
                const int tn = std::min(32, FO - ft);
                const bool wide = tn > 16;
                __m512 a0[4], a1[4];
                const __m512 z = _mm512_setzero_ps();
                for (int s = 0; s < 4; s++) { a0[s] = z; a1[s] = z; }
                for (int ci = 0; ci < cv.C_in; ci++) {
                    for (int p = 0; p < np; p++) {
                        const uint32_t* wr = wo + ((size_t)ci * np + p) * cv.kw;
                        if (sF == 1) {
                            const uint32_t* row = &rp[((size_t)ci * np + p) * L] + ft;
                            for (int s = 0; s < cv.kw; s++) {
                                const __m512bh wv = (__m512bh)_mm512_set1_epi32((int)wr[s]);
                                a0[s] = _mm512_dpbf16_ps(a0[s], wv,
                                    (__m512bh)_mm512_loadu_si512(row + s));
                                if (wide)
                                    a1[s] = _mm512_dpbf16_ps(a1[s], wv,
                                        (__m512bh)_mm512_loadu_si512(row + s + 16));
                            }
                        } else {
                            const uint32_t* pe = &rp[(((size_t)ci * np + p) * 2 + 0) * L] + ft;
                            const uint32_t* po = &rp[(((size_t)ci * np + p) * 2 + 1) * L] + ft;
                            for (int s = 0; s < cv.kw; s++) {
                                const __m512bh wv = (__m512bh)_mm512_set1_epi32((int)wr[s]);
                                const uint32_t* rs = ((s & 1) ? po : pe) + (s >> 1);
                                a0[s] = _mm512_dpbf16_ps(a0[s], wv,
                                    (__m512bh)_mm512_loadu_si512(rs));
                                if (wide)
                                    a1[s] = _mm512_dpbf16_ps(a1[s], wv,
                                        (__m512bh)_mm512_loadu_si512(rs + 16));
                            }
                        }
                    }
                }
                const __m512 bv = _mm512_set1_ps(bo);
                __m512 s0 = _mm512_add_ps(_mm512_add_ps(a0[0], a0[1]),
                                          _mm512_add_ps(a0[2], a0[3]));
                s0 = _mm512_add_ps(s0, bv);
                __m512 s1 = _mm512_add_ps(_mm512_add_ps(a1[0], a1[1]),
                                          _mm512_add_ps(a1[2], a1[3]));
                s1 = _mm512_add_ps(s1, bv);
                if (tn >= 32) {
                    _mm512_storeu_ps(yr + ft, s0);
                    _mm512_storeu_ps(yr + ft + 16, s1);
                } else if (tn >= 16) {
                    _mm512_storeu_ps(yr + ft, s0);
                    if (tn > 16)
                        _mm512_mask_storeu_ps(yr + ft + 16,
                                              (__mmask16)((1u << (tn - 16)) - 1), s1);
                } else {
                    _mm512_mask_storeu_ps(yr + ft, (__mmask16)((1u << tn) - 1), s0);
                }
            }
        }
        return;
    }
#endif
#ifdef __AVX512F__
    // Hand AVX-512: 32-wide accumulator tile pinned in two zmm across the
    // full (c, r, s) reduction; masked tail stores; buffers carry slack for
    // lane over-reads (discarded by the mask).
    for (int o = 0; o < cv.C_out; o++) {
        float* yr = y + (size_t)o * FO;
        const float bo = cv.b[o];
        const float* wo = &cv.w[((size_t)o * cv.C_in) * cv.kh * cv.kw];
        for (int ft = 0; ft < FO; ft += 32) {
            const int tn = std::min(32, FO - ft);
            const bool wide = tn > 16;
            // 4 independent accumulator chains per 16-lane half (one per
            // kernel tap s) — breaks the FMA latency chain; summed at end.
            __m512 a0[4], a1[4];
            const __m512 z = _mm512_setzero_ps();
            for (int s = 0; s < 4; s++) { a0[s] = z; a1[s] = z; }
            for (int c = 0; c < cv.C_in; c++) {
                for (int r = 0; r < cv.kh; r++) {
                    const float* wr = wo + ((size_t)c * cv.kh + r) * cv.kw;
                    if (sF == 1) {
                        const float* p0 = &cv.win[((size_t)c * cv.kh + r) * cv.Fp] + ft;
                        if (wide) {
                            for (int s = 0; s < cv.kw; s++) {
                                const __m512 wv = _mm512_set1_ps(wr[s]);
                                a0[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(p0 + s), a0[s]);
                                a1[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(p0 + s + 16), a1[s]);
                            }
                        } else {
                            for (int s = 0; s < cv.kw; s++) {
                                const __m512 wv = _mm512_set1_ps(wr[s]);
                                a0[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(p0 + s), a0[s]);
                            }
                        }
                    } else {
                        const float* p0 = &dei[(((size_t)c * cv.kh + r) * 2 + 0) * half] + ft;
                        const float* p1 = &dei[(((size_t)c * cv.kh + r) * 2 + 1) * half] + ft;
                        if (wide) {
                            for (int s = 0; s < cv.kw; s++) {
                                const __m512 wv = _mm512_set1_ps(wr[s]);
                                const float* rs = ((s & 1) ? p1 : p0) + (s >> 1);
                                a0[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(rs), a0[s]);
                                a1[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(rs + 16), a1[s]);
                            }
                        } else {
                            for (int s = 0; s < cv.kw; s++) {
                                const __m512 wv = _mm512_set1_ps(wr[s]);
                                const float* rs = ((s & 1) ? p1 : p0) + (s >> 1);
                                a0[s] = _mm512_fmadd_ps(wv, _mm512_loadu_ps(rs), a0[s]);
                            }
                        }
                    }
                }
            }
            const __m512 bv = _mm512_set1_ps(bo);
            __m512 s0 = _mm512_add_ps(_mm512_add_ps(a0[0], a0[1]),
                                      _mm512_add_ps(a0[2], a0[3]));
            s0 = _mm512_add_ps(s0, bv);
            __m512 s1 = _mm512_add_ps(_mm512_add_ps(a1[0], a1[1]),
                                      _mm512_add_ps(a1[2], a1[3]));
            s1 = _mm512_add_ps(s1, bv);
            if (tn >= 32) {
                _mm512_storeu_ps(yr + ft, s0);
                _mm512_storeu_ps(yr + ft + 16, s1);
            } else if (tn >= 16) {
                _mm512_storeu_ps(yr + ft, s0);
                if (tn > 16)
                    _mm512_mask_storeu_ps(yr + ft + 16,
                                          (__mmask16)((1u << (tn - 16)) - 1), s1);
            } else {
                _mm512_mask_storeu_ps(yr + ft, (__mmask16)((1u << tn) - 1), s0);
            }
        }
    }
#else
    constexpr int TF = 32;
    for (int o = 0; o < cv.C_out; o++) {
        float* __restrict yr = y + (size_t)o * FO;
        const float bo = cv.b[o];
        const float* wo = &cv.w[((size_t)o * cv.C_in) * cv.kh * cv.kw];
        for (int ft = 0; ft < FO; ft += TF) {
            const int tn = std::min(TF, FO - ft);
            float acc[TF];
            for (int j = 0; j < TF; j++) acc[j] = bo;
            for (int c = 0; c < cv.C_in; c++)
                for (int r = 0; r < cv.kh; r++) {
                    const float* wr = wo + ((size_t)c * cv.kh + r) * cv.kw;
                    const float* base = (sF == 1)
                        ? &cv.win[((size_t)c * cv.kh + r) * cv.Fp] + ft : nullptr;
                    for (int s = 0; s < cv.kw; s++) {
                        const float wv = wr[s];
                        const float* rs = (sF == 1) ? base + s
                            : &dei[(((size_t)c * cv.kh + r) * 2 + (s & 1)) * half] + ft + (s >> 1);
                        for (int j = 0; j < TF; j++) acc[j] += wv * rs[j];
                    }
                }
            for (int j = 0; j < tn; j++) yr[ft + j] = acc[j];
        }
    }
#endif
}

static void conv_shift(ne_conv& cv) {
    for (int c = 0; c < cv.C_in; c++) {
        float* base = &cv.win[(size_t)c * cv.kh * cv.Fp];
        std::memmove(base, base + cv.Fp, (size_t)(cv.kh - 1) * cv.Fp * sizeof(float));
    }
}

// generic (C, rows, L) window: write cur (C, L0) into last row at col_off
static void win_write(std::vector<float>& w, int C, int rows, int L,
                      const float* cur, int L0, int col_off) {
    for (int c = 0; c < C; c++)
        std::memcpy(&w[((size_t)c * rows + (rows - 1)) * L + col_off],
                    cur + (size_t)c * L0, L0 * sizeof(float));
}
static void win_shift(std::vector<float>& w, int C, int rows, int L) {
    for (int c = 0; c < C; c++) {
        float* base = &w[(size_t)c * rows * L];
        std::memmove(base, base + L, (size_t)(rows - 1) * L * sizeof(float));
    }
}

// ── init ───────────────────────────────────────────────────────────────────
static void load_block(ne_block& bl, const localvqe_model& m, const std::string& p,
                       int C_in, int C_out, int kh, int kw, int F_in, int sF) {
    bl.n1.g = W(m, p + ".norm.weight"); bl.n1.b = W(m, p + ".norm.bias");
    conv_setup(bl.c1, W(m, p + ".conv.weight"), W(m, p + ".conv.bias"),
               C_out, C_in, kh, kw, F_in, sF);
    bl.n2.g = W(m, p + ".resblock.norm.weight");
    bl.n2.b = W(m, p + ".resblock.norm.bias");
    conv_setup(bl.c2, W(m, p + ".resblock.conv.weight"),
               W(m, p + ".resblock.conv.bias"),
               C_out, C_out, kh, kw, bl.c1.F_out, 1);
}

bool ne_init(native_engine& ne, const localvqe_model& m) {
    const auto& hp = m.hparams;
    ne.F = hp.n_freq_bins; ne.dmax = hp.dmax; ne.power_c = hp.power_law_c;
    const auto& mc = hp.mic_channels;   // [2,16,20,20,20,20]
    const auto& fc = hp.far_channels;   // [2,16,20]
    const int kh = hp.kernel_size_h, kw = hp.kernel_size_w;
    int F = ne.F;
    ne.win512.resize(512);
    for (int n = 0; n < 512; n++)
        ne.win512[n] = sqrtf(0.5f - 0.5f * cosf(2.0f * (float)M_PI * (n + 0.5f) / 512.0f));

    // encoders: mic1(F), mic2(F/2), far1(F), far2(F/2), mic3(F/4, C=2*mc2),
    // mic4(F/8), mic5(F/16)
    load_block(ne.enc[0], m, "mic_enc1", mc[0], mc[1], kh, kw, F,     2);
    load_block(ne.enc[1], m, "mic_enc2", mc[1], mc[2], kh, kw, F/2,   2);
    load_block(ne.enc[2], m, "far_enc1", fc[0], fc[1], kh, kw, F,     2);
    load_block(ne.enc[3], m, "far_enc2", fc[1], fc[2], kh, kw, F/2,   2);
    load_block(ne.enc[4], m, "mic_enc3", mc[2]+fc[2], mc[3], kh, kw, F/4, 2);
    load_block(ne.enc[5], m, "mic_enc4", mc[3], mc[4], kh, kw, F/8,   2);
    load_block(ne.enc[6], m, "mic_enc5", mc[4], mc[5], kh, kw, F/16,  2);

    // align (on mic_e2/far_e2 at F/4)
    ne.Hal = hp.align_hidden;
    ne.pmw = W(m, "align.pconv_mic.weight"); ne.pmb = W(m, "align.pconv_mic.bias");
    ne.prw = W(m, "align.pconv_ref.weight"); ne.prb = W(m, "align.pconv_ref.bias");
    ne.sw  = W(m, "align.conv.1.weight");    ne.sb  = W(m, "align.conv.1.bias");
    ne.K_win.assign((size_t)ne.Hal * ne.dmax * (F/4), 0.0f);
    ne.ref_win.assign((size_t)fc[2] * ne.dmax * (F/4), 0.0f);
    ne.S_win.assign((size_t)ne.Hal * 5 * (ne.dmax + 2), 0.0f);

    // s4d
    ne.s4d_inw = W(m, "bottleneck.input_proj.weight");
    ne.s4d_inb = W(m, "bottleneck.input_proj.bias");
    ne.s4d_outw = W(m, "bottleneck.output_proj.weight");
    ne.s4d_outb = W(m, "bottleneck.output_proj.bias");
    ne.s4d_ar = W(m, "bottleneck.a_real"); ne.s4d_ai = W(m, "bottleneck.a_imag");
    ne.s4d_Br = W(m, "bottleneck.B_real"); ne.s4d_Bi = W(m, "bottleneck.B_imag");
    ne.s4d_Cr = W(m, "bottleneck.C_real"); ne.s4d_Ci = W(m, "bottleneck.C_imag");
    ne.s4d_D  = W(m, "bottleneck.D");
    ne.bn_h = (int)ne.s4d_ar.size();

    // decoders dec5..dec1 (input F/32 -> ... -> F)
    const char* dn[5] = {"dec5", "dec4", "dec3", "dec2", "dec1"};
    int Fd = F / 32;
    int Cd = mc[5];
    for (int i = 0; i < 5; i++) {
        ne_dec& d = ne.dec[i];
        std::string p = dn[i];
        d.skip_n.g = W(m, p + ".skip_norm.weight");
        d.skip_n.b = W(m, p + ".skip_norm.bias");
        d.skip_w = W(m, p + ".skip_conv.weight");
        d.skip_b = W(m, p + ".skip_conv.bias");
        d.res_n.g = W(m, p + ".resblock.norm.weight");
        d.res_n.b = W(m, p + ".resblock.norm.bias");
        conv_setup(d.res, W(m, p + ".resblock.conv.weight"),
                   W(m, p + ".resblock.conv.bias"), Cd, Cd, kh, kw, Fd, 1);
        d.dec_n.g = W(m, p + ".deconv.norm.weight");
        d.dec_n.b = W(m, p + ".deconv.norm.bias");
        int C_next = (int)(W(m, p + ".deconv.conv.bias").size() / 2);
        conv_setup(d.deconv, W(m, p + ".deconv.conv.weight"),
                   W(m, p + ".deconv.conv.bias"), C_next * 2, Cd, kh, kw, Fd, 1);
        Cd = C_next; Fd *= 2;
    }

    ne.ccm_win.assign((size_t)2 * 3 * (F + 2), 0.0f);
    ne.s4d_hr.assign(ne.bn_h, 0.0f); ne.s4d_hi.assign(ne.bn_h, 0.0f);
    size_t mx = (size_t)64 * 512;
    ne.t0.resize(mx); ne.t1.resize(mx); ne.t2.resize(mx); ne.t3.resize(mx); ne.t4.resize(mx);
    for (auto& s : ne.skipsave) s.resize(mx);
    ne.loaded = !ne.s4d_ar.empty() && !ne.enc[0].c1.w.empty();
    ne_reset(ne);
    return ne.loaded;
}

void ne_reset(native_engine& ne) {
    for (auto& b : ne.enc) {
        std::fill(b.c1.win.begin(), b.c1.win.end(), 0.0f);
        std::fill(b.c2.win.begin(), b.c2.win.end(), 0.0f);
    }
    for (auto& d : ne.dec) {
        std::fill(d.res.win.begin(), d.res.win.end(), 0.0f);
        std::fill(d.deconv.win.begin(), d.deconv.win.end(), 0.0f);
    }
    std::fill(ne.K_win.begin(), ne.K_win.end(), 0.0f);
    std::fill(ne.ref_win.begin(), ne.ref_win.end(), 0.0f);
    std::fill(ne.S_win.begin(), ne.S_win.end(), 0.0f);
    std::fill(ne.ccm_win.begin(), ne.ccm_win.end(), 0.0f);
    std::fill(ne.s4d_hr.begin(), ne.s4d_hr.end(), 0.0f);
    std::fill(ne.s4d_hi.begin(), ne.s4d_hi.end(), 0.0f);
}

// run one encoder-style block: y = silu(conv1(norm1(x))); out = silu(conv2(norm2(y))) + y
static void run_block(ne_block& bl, const float* x, float* tmp, float* y, float* out) {
    norm_cf(x, tmp, bl.c1.C_in, bl.c1.F_in, bl.n1.g.data(), bl.n1.b.data());
    conv_run(bl.c1, tmp, y);
    const int n1 = bl.c1.C_out * bl.c1.F_out;
    for (int i = 0; i < n1; i++) y[i] = silu(y[i]);
    norm_cf(y, tmp, bl.c2.C_in, bl.c2.F_in, bl.n2.g.data(), bl.n2.b.data());
    conv_run(bl.c2, tmp, out);
    for (int i = 0; i < n1; i++) out[i] = silu(out[i]) + y[i];
}

void ne_process_frame(native_engine& ne, const float* mic_win,
                      const float* ref_win, float* out_win) {
    g_taps.clear();
    const int F = ne.F;
    static thread_local std::vector<float> re(512), im(512);
    static thread_local std::vector<float> micS(2 * 256), refS(2 * 256);

    auto analysis = [&](const float* w, float* spec) {  // spec (2, F): [re|im]
        for (int n = 0; n < 512; n++) { re[n] = w[n] * ne.win512[n]; im[n] = 0.0f; }
        fft_ip(re, im, false);
        for (int k = 0; k < F; k++) { spec[k] = re[k + 1]; spec[F + k] = im[k + 1]; }
    };
    analysis(mic_win, micS.data());
    analysis(ref_win, refS.data());

    // FE power-law: x / (mag^(1-c)) with mag = sqrt(re^2+im^2+1e-12)
    static thread_local std::vector<float> micF(2 * 256), refF(2 * 256);
    auto fe = [&](const float* s, float* d) {
        for (int f = 0; f < F; f++) {
            const float m2 = s[f]*s[f] + s[F+f]*s[F+f] + 1e-12f;
            const float sc = 1.0f / (powf(sqrtf(m2), 1.0f - ne.power_c) + 1e-12f);
            d[f] = s[f] * sc; d[F + f] = s[F + f] * sc;
        }
    };
    fe(micS.data(), micF.data());
    fe(refS.data(), refF.data());
    tap("fe_mic", micF.data(), 2 * F);
    tap("fe_far", refF.data(), 2 * F);

    float* t0 = ne.t0.data(); float* t1 = ne.t1.data(); float* t2 = ne.t2.data();
    float* t3 = ne.t3.data(); float* t4 = ne.t4.data();

    // mic enc1/enc2
    run_block(ne.enc[0], micF.data(), t0, t1, ne.skipsave[1].data());   // mic_e1
    tap("mic_e1", ne.skipsave[1].data(), ne.enc[0].c1.C_out * ne.enc[0].c1.F_out);
    run_block(ne.enc[1], ne.skipsave[1].data(), t0, t1, ne.skipsave[2].data()); // mic_e2
    tap("mic_e2", ne.skipsave[2].data(), ne.enc[1].c1.C_out * ne.enc[1].c1.F_out);
    // far enc1/enc2
    run_block(ne.enc[2], refF.data(), t0, t1, t2);                      // far_e1
    tap("far_e1", t2, ne.enc[2].c1.C_out * ne.enc[2].c1.F_out);
    run_block(ne.enc[3], t2, t0, t1, t3);                               // far_e2
    tap("far_e2", t3, ne.enc[3].c1.C_out * ne.enc[3].c1.F_out);

    // ── align: Q/K 1x1 projections, lag attention, smooth, weighted sum ──
    const int F2 = F / 4, H = ne.Hal, C2 = ne.enc[1].c1.C_out, dmax = ne.dmax;
    const float* mic_e2 = ne.skipsave[2].data();
    const float* far_e2 = t3;
    static thread_local std::vector<float> Q, Kc, sim, attn, aligned;
    Q.resize((size_t)H * F2); Kc.resize((size_t)H * F2);
    sim.resize((size_t)dmax * H); attn.resize(dmax);
    aligned.resize((size_t)C2 * F2);
    auto p1x1 = [&](const float* x, const std::vector<float>& w,
                    const std::vector<float>& b, float* y) {
        for (int h = 0; h < H; h++) {
            float* yr = y + (size_t)h * F2;
            for (int f = 0; f < F2; f++) yr[f] = b[h];
            for (int c = 0; c < C2; c++) {
                const float wv = w[(size_t)h * C2 + c];
                const float* xr = x + (size_t)c * F2;
                for (int f = 0; f < F2; f++) yr[f] += wv * xr[f];
            }
        }
    };
    p1x1(mic_e2, ne.pmw, ne.pmb, Q.data());
    p1x1(far_e2, ne.prw, ne.prb, Kc.data());
    win_write(ne.K_win, H, dmax, F2, Kc.data(), F2, 0);
    win_write(ne.ref_win, C2, dmax, F2, far_e2, F2, 0);
    const float scl = 1.0f / sqrtf((float)F2);
    for (int d = 0; d < dmax; d++)
        for (int h = 0; h < H; h++) {
            const float* kr = &ne.K_win[((size_t)h * dmax + d) * F2];
            const float* qr = &Q[(size_t)h * F2];
            float s = 0;
            for (int f = 0; f < F2; f++) s += kr[f] * qr[f];
            sim[(size_t)d * H + h] = s * scl;   // V layout (H, 1, dmax) below
        }
    // smooth window: (H, 5, dmax+2); write V (per h, dmax) at row 4, col 1
    for (int h = 0; h < H; h++) {
        float* dst = &ne.S_win[((size_t)h * 5 + 4) * (dmax + 2) + 1];
        for (int d = 0; d < dmax; d++) dst[d] = sim[(size_t)d * H + h];
    }
    // smooth conv: kernel (1, H, 5, 3) -> out (dmax)
    for (int d = 0; d < dmax; d++) {
        float acc = ne.sb[0];
        for (int h = 0; h < H; h++)
            for (int r = 0; r < 5; r++) {
                const float* row = &ne.S_win[((size_t)h * 5 + r) * (dmax + 2)];
                const float* wr = &ne.sw[((size_t)h * 5 + r) * 3];
                acc += wr[0]*row[d] + wr[1]*row[d+1] + wr[2]*row[d+2];
            }
        attn[d] = acc;
    }
    float mx = attn[0];
    for (int d = 1; d < dmax; d++) mx = std::max(mx, attn[d]);
    float den = 0;
    for (int d = 0; d < dmax; d++) { attn[d] = expf(attn[d] - mx); den += attn[d]; }
    for (int d = 0; d < dmax; d++) attn[d] /= den;
    for (int c = 0; c < C2; c++) {
        float* ar = &aligned[(size_t)c * F2];
        std::memset(ar, 0, F2 * sizeof(float));
        for (int d = 0; d < dmax; d++) {
            const float a = attn[d];
            const float* rr = &ne.ref_win[((size_t)c * dmax + d) * F2];
            for (int f = 0; f < F2; f++) ar[f] += a * rr[f];
        }
    }
    tap("aligned", aligned.data(), (size_t)C2 * F2);

    // concat (mic_e2, aligned) -> enc3/4/5
    static thread_local std::vector<float> cat;
    cat.resize((size_t)(C2 * 2) * F2);
    std::memcpy(cat.data(), mic_e2, (size_t)C2 * F2 * sizeof(float));
    std::memcpy(cat.data() + (size_t)C2 * F2, aligned.data(), (size_t)C2 * F2 * sizeof(float));
    run_block(ne.enc[4], cat.data(), t0, t1, ne.skipsave[3].data());    // mic_e3
    tap("mic_e3", ne.skipsave[3].data(), ne.enc[4].c1.C_out * ne.enc[4].c1.F_out);
    run_block(ne.enc[5], ne.skipsave[3].data(), t0, t1, ne.skipsave[4].data()); // mic_e4
    tap("mic_e4", ne.skipsave[4].data(), ne.enc[5].c1.C_out * ne.enc[5].c1.F_out);
    run_block(ne.enc[6], ne.skipsave[4].data(), t0, t1, ne.skipsave[5].data()); // mic_e5
    tap("mic_e5", ne.skipsave[5].data(), ne.enc[6].c1.C_out * ne.enc[6].c1.F_out);

    // ── S4D bottleneck ──
    const int C5 = ne.enc[6].c1.C_out, F5 = ne.enc[6].c1.F_out;
    const int IS = C5 * F5, Hb = ne.bn_h;
    static thread_local std::vector<float> v, ybn, bnout;
    v.resize(Hb); ybn.resize(Hb); bnout.resize(IS);
    const float* u = ne.skipsave[5].data();   // (C5, F5) == (c f) flatten
    for (int h = 0; h < Hb; h++) {
        float s = ne.s4d_inb[h];
        const float* wr = &ne.s4d_inw[(size_t)h * IS];
        for (int i = 0; i < IS; i++) s += wr[i] * u[i];
        v[h] = s;
    }
    for (int h = 0; h < Hb; h++) {
        const float hr = ne.s4d_ar[h]*ne.s4d_hr[h] - ne.s4d_ai[h]*ne.s4d_hi[h] + ne.s4d_Br[h]*v[h];
        const float hi = ne.s4d_ar[h]*ne.s4d_hi[h] + ne.s4d_ai[h]*ne.s4d_hr[h] + ne.s4d_Bi[h]*v[h];
        ne.s4d_hr[h] = hr; ne.s4d_hi[h] = hi;
        ybn[h] = ne.s4d_Cr[h]*hr - ne.s4d_Ci[h]*hi;
    }
    for (int i = 0; i < IS; i++) {
        float s = ne.s4d_outb[i];
        const float* wr = &ne.s4d_outw[(size_t)i * Hb];
        for (int h = 0; h < Hb; h++) s += wr[h] * ybn[h];
        bnout[i] = s + ne.s4d_D[i] * u[i];
    }
    tap("bottleneck", bnout.data(), IS);

    // ── decoders ──
    // x = bnout; skips: dec5<-mic_e5, dec4<-mic_e4, dec3<-mic_e3,
    // dec2<-mic_e2, dec1<-mic_e1
    const float* skips[5] = { ne.skipsave[5].data(), ne.skipsave[4].data(),
                              ne.skipsave[3].data(), ne.skipsave[2].data(),
                              ne.skipsave[1].data() };
    const char* dtap[5] = {"d5", "d4", "d3", "d2", "d1"};
    float* x = bnout.data();
    static thread_local std::vector<float> dbuf1, dbuf2;
    dbuf1.resize((size_t)64 * 512); dbuf2.resize((size_t)64 * 512);
    for (int i = 0; i < 5; i++) {
        ne_dec& d = ne.dec[i];
        const int C = d.res.C_in, Fd = d.res.F_in;
        // skip = 1x1(norm(x_en)); y = x + skip
        norm_cf(skips[i], t0, C, Fd, d.skip_n.g.data(), d.skip_n.b.data());
        for (int o = 0; o < C; o++) {
            float* yr = t1 + (size_t)o * Fd;
            for (int f = 0; f < Fd; f++) yr[f] = d.skip_b[o];
            for (int c = 0; c < C; c++) {
                const float wv = d.skip_w[(size_t)o * C + c];
                const float* xr = t0 + (size_t)c * Fd;
                for (int f = 0; f < Fd; f++) yr[f] += wv * xr[f];
            }
        }
        const int n = C * Fd;
        for (int j = 0; j < n; j++) t1[j] += x[j];       // y
        // res
        norm_cf(t1, t0, C, Fd, d.res_n.g.data(), d.res_n.b.data());
        conv_run(d.res, t0, t2);
        for (int j = 0; j < n; j++) t2[j] = silu(t2[j]) + t1[j];
        // deconv + shuffle
        norm_cf(t2, t0, C, Fd, d.dec_n.g.data(), d.dec_n.b.data());
        conv_run(d.deconv, t0, t3);                       // (2*C_next, Fd)
        const int Cn = d.deconv.C_out / 2;
        float* dst = (i % 2 == 0) ? dbuf1.data() : dbuf2.data();
        for (int co = 0; co < Cn; co++) {
            std::memcpy(dst + (size_t)co * (2 * Fd),
                        t3 + (size_t)co * Fd, Fd * sizeof(float));
            std::memcpy(dst + (size_t)co * (2 * Fd) + Fd,
                        t3 + (size_t)(co + Cn) * Fd, Fd * sizeof(float));
        }
        const int n2 = Cn * 2 * Fd;
        if (i != 4)
            for (int j = 0; j < n2; j++) dst[j] = silu(dst[j]);
        x = dst;
        tap(dtap[i], x, n2);
    }

    // ── CCM: mask = x (27, F); apply to RAW mic spectrum window ──
    static const float VR[3] = {1.0f, -0.5f, -0.5f};
    static const float VI[3] = {0.0f, 0.86602540378f, -0.86602540378f};
    win_write(ne.ccm_win, 2, 3, F + 2, micS.data(), F, 1);
    static thread_local std::vector<float> er, ei;
    er.assign(F, 0.0f); ei.assign(F, 0.0f);
    for (int mrow = 0; mrow < 3; mrow++) {
        const float* xr = &ne.ccm_win[((size_t)0 * 3 + mrow) * (F + 2)];
        const float* xi = &ne.ccm_win[((size_t)1 * 3 + mrow) * (F + 2)];
        for (int nn = 0; nn < 3; nn++) {
            const int ki = mrow * 3 + nn;
            // H[ki] from mask: Hr = sum_r mask[r*9+ki]*VR[r], Hi likewise
            for (int f = 0; f < F; f++) {
                float hr = 0, hi = 0;
                for (int r = 0; r < 3; r++) {
                    const float mv = x[(size_t)(r * 9 + ki) * F + f];
                    hr += mv * VR[r]; hi += mv * VI[r];
                }
                const float xrv = xr[f + nn], xiv = xi[f + nn];
                er[f] += hr * xrv - hi * xiv;
                ei[f] += hr * xiv + hi * xrv;
            }
        }
    }
    tap("enh", er.data(), F);  // re half; test compares both via taps order
    tap("enh_i", ei.data(), F);

    // ── synthesis: full[k]=E_k (1..255), full[256]=2*E_256; frame = w*irfft ──
    std::fill(re.begin(), re.end(), 0.0f);
    std::fill(im.begin(), im.end(), 0.0f);
    for (int k = 0; k < F - 1; k++) { re[k + 1] = er[k]; im[k + 1] = ei[k]; }
    re[256] = 2.0f * er[F - 1];
    // hermitian extension for the c2c inverse
    for (int k = 1; k < 256; k++) { re[512 - k] = re[k]; im[512 - k] = -im[k]; }
    fft_ip(re, im, true);
    for (int n = 0; n < 512; n++) out_win[n] = re[n] * ne.win512[n];

    // ── end-of-frame: shift all windows ──
    for (auto& b : ne.enc) { conv_shift(b.c1); conv_shift(b.c2); }
    for (auto& d : ne.dec) { conv_shift(d.res); conv_shift(d.deconv); }
    win_shift(ne.K_win, H, dmax, F2);
    win_shift(ne.ref_win, C2, dmax, F2);
    win_shift(ne.S_win, H, 5, dmax + 2);
    win_shift(ne.ccm_win, 2, 3, F + 2);
}
