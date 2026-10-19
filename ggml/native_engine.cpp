#include "native_engine.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <cmath>
#include <cstring>
#include <cstdlib>
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

[[maybe_unused]] static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// 4-accumulator dot product for the reductions clang can't auto-vectorize
// under strict FP (align sim, S4D input/output projections): 4 independent
// NEON FMA chains hide the 4-cycle FMA latency, then one horizontal add.
// This changes summation order vs. the sequential scalar loop (expect
// ~1e-7..1e-6 max abs diff, not bit-identical).
#if defined(__ARM_NEON)
static inline float dot4(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#else
static inline float dot4(const float* a, const float* b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i]; s1 += a[i+1] * b[i+1];
        s2 += a[i+2] * b[i+2]; s3 += a[i+3] * b[i+3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
// y[i] = silu(y[i]) (+ add[i]) over n, via vForce's vectorized expf
// (correctly rounded to ~1 ulp, so numerics track the scalar path).
static void silu_inplace(float* y, int n, const float* add) {
    static thread_local std::vector<float> e;
    if (e.size() < (size_t)n) e.resize(n);
    float* ep = e.data();
    for (int i = 0; i < n; i++) ep[i] = -y[i];
    vvexpf(ep, ep, &n);
    if (add) for (int i = 0; i < n; i++) y[i] = y[i] / (1.0f + ep[i]) + add[i];
    else     for (int i = 0; i < n; i++) y[i] = y[i] / (1.0f + ep[i]);
}
#else
static void silu_inplace(float* y, int n, const float* add) {
    if (add) for (int i = 0; i < n; i++) y[i] = silu(y[i]) + add[i];
    else     for (int i = 0; i < n; i++) y[i] = silu(y[i]);
}
#endif

// CausalGroupNorm over (C, F): x -> (x - mean)/sqrt(var+eps) * g[c] + b[c]
static void norm_cf(const float* x, float* y, int C, int F,
                    const float* g, const float* b) {
    // 4 independent double accumulators (sum and sum-of-squares each) so the
    // dependent add chain pipelines instead of serializing one add/cycle;
    // combined pairwise at the end. Changes summation order vs. the single
    // running sum.
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;
    const int n = C * F;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += x[i];     q0 += (double)x[i]     * x[i];
        s1 += x[i + 1]; q1 += (double)x[i + 1] * x[i + 1];
        s2 += x[i + 2]; q2 += (double)x[i + 2] * x[i + 2];
        s3 += x[i + 3]; q3 += (double)x[i + 3] * x[i + 3];
    }
    double s = (s0 + s1) + (s2 + s3), ssq = (q0 + q1) + (q2 + q3);
    for (; i < n; i++) { s += x[i]; ssq += (double)x[i] * x[i]; }
    const float mean = (float)(s / n);
    const float var = (float)(ssq / n) - mean * mean;
    const float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int c = 0; c < C; c++) {
        const float a = g[c] * inv, d = b[c] - mean * a;
        const float* xr = x + (size_t)c * F;
        float* yr = y + (size_t)c * F;
        for (int f = 0; f < F; f++) yr[f] = a * xr[f] + d;
    }
}

[[maybe_unused]] static inline uint16_t f32_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t r = ((u >> 16) & 1) + 0x7FFF;
    return (uint16_t)((u + r) >> 16);
}

static inline void conv_set_head(ne_conv& cv, int head) {
    cv.head = head;
    for (int r = 0; r < cv.kh; r++) cv.prow[r] = (r + head) % cv.kh;
}

// Returns false when the tensors do not have the shape the kernels index
// ((C_out, C_in, kh, kw) weights, (C_out) bias) or the geometry is empty —
// ne_init then fails and the caller falls back to the graph.
static bool conv_setup(ne_conv& cv, const std::vector<float>& w,
                       const std::vector<float>& b, int C_out, int C_in,
                       int kh, int kw, int F_in, int sF) {
    if (C_out <= 0 || C_in <= 0 || F_in <= 0 || (sF != 1 && sF != 2)) return false;
    if (w.size() != (size_t)C_out * C_in * kh * kw || b.size() != (size_t)C_out) return false;
    cv.C_in = C_in; cv.C_out = C_out; cv.kh = kh; cv.kw = kw; cv.sF = sF;
    cv.F_in = F_in; cv.Fp = F_in + kw - 1;
    cv.F_out = (cv.Fp - kw) / sF + 1;
    if (cv.F_out <= 0) return false;
    cv.w = w; cv.b = b;
    cv.win.assign((size_t)C_in * kh * cv.Fp + 64, 0.0f);
    conv_set_head(cv, 0);
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
    return true;
}

#if defined(__ARM_NEON) && !defined(__AVX512F__)
// NEON direct conv tile: NO output channels x (4*NV) output bins, all
// (c, r, s) taps accumulated into 16 independent q-register chains
// (NO*NV == 16 for the main tile), so the 4-cycle FMA latency is hidden
// on 4 pipes. One weight vector per (o, c, r) holds the kw==4 taps and is
// consumed by lane (no broadcasts); each window load feeds NO FMAs.
// Requires kw == 4 (asserted by the caller).
template <int NO, int NV>
static inline void conv_tile_neon(const ne_conv& cv, const float* dei, int half,
                                  int o0, int ft, int tn, float* y, int FO) {
    float32x4_t acc[NO][NV];
    for (int i = 0; i < NO; i++) {
        const float32x4_t bv = vdupq_n_f32(cv.b[o0 + i]);
        for (int v = 0; v < NV; v++) acc[i][v] = bv;
    }
    const int kh = cv.kh, C_in = cv.C_in;
    const float* wbase = &cv.w[(size_t)o0 * C_in * kh * 4];
    const size_t ostride = (size_t)C_in * kh * 4;
    for (int c = 0; c < C_in; c++) {
        for (int r = 0; r < kh; r++) {
            float32x4_t w[NO];
            for (int i = 0; i < NO; i++)
                w[i] = vld1q_f32(wbase + (size_t)i * ostride + ((size_t)c * kh + r) * 4);
            const float* p0; const float* p1; const float* p2; const float* p3;
            if (cv.sF == 1) {
                // Row pointer only: physical row for logical r under the
                // ring buffer (see ne_conv::head); kernel FMA loops below
                // (LVQE_TAP) are untouched.
                const float* p = &cv.win[((size_t)c * kh + cv.prow[r]) * cv.Fp] + ft;
                p0 = p; p1 = p + 1; p2 = p + 2; p3 = p + 3;
            } else {
                const float* pe = dei + (((size_t)c * kh + r) * 2 + 0) * half + ft;
                const float* po = dei + (((size_t)c * kh + r) * 2 + 1) * half + ft;
                p0 = pe; p1 = po; p2 = pe + 1; p3 = po + 1;
            }
#define LVQE_TAP(S, P)                                                        \
            for (int v = 0; v < NV; v++) {                                    \
                const float32x4_t xv = vld1q_f32((P) + 4 * v);                \
                for (int i = 0; i < NO; i++)                                  \
                    acc[i][v] = vfmaq_laneq_f32(acc[i][v], xv, w[i], S);      \
            }
            LVQE_TAP(0, p0)
            LVQE_TAP(1, p1)
            LVQE_TAP(2, p2)
            LVQE_TAP(3, p3)
#undef LVQE_TAP
        }
    }
    for (int i = 0; i < NO; i++) {
        float* yr = y + (size_t)(o0 + i) * FO + ft;
        if (tn == 4 * NV) {
            for (int v = 0; v < NV; v++) vst1q_f32(yr + 4 * v, acc[i][v]);
        } else {
            float tmp[4 * NV];
            for (int v = 0; v < NV; v++) vst1q_f32(tmp + 4 * v, acc[i][v]);
            for (int j = 0; j < tn; j++) yr[j] = tmp[j];
        }
    }
}

template <int NO>
static inline void conv_rows_neon(const ne_conv& cv, const float* dei, int half,
                                  int o0, float* y, int FO) {
    int ft = 0;
    for (; ft + 16 <= FO; ft += 16) conv_tile_neon<NO, 4>(cv, dei, half, o0, ft, 16, y, FO);
    for (; ft + 8 <= FO; ft += 8)   conv_tile_neon<NO, 2>(cv, dei, half, o0, ft, 8, y, FO);
    if (ft < FO)                    conv_tile_neon<NO, 2>(cv, dei, half, o0, ft, FO - ft, y, FO);
}

// Decoder skip 1x1 conv (C_out, C_in) x (C_in, Fd) -> (C_out, Fd): same
// reuse pattern as conv_tile_neon — one x vector, loaded once per (c,
// bin-tile), feeds NO independent output-channel FMA accumulators, instead
// of the naive per-output-channel loop that reloads x from memory C_out
// times. Summation order per (o, f) is unchanged (c = 0..C_in-1 in order);
// only the FMA vs separate-multiply-add rounding can differ from scalar.
template <int NO, int NV>
static inline void skip1x1_tile_neon(const float* w, const float* bias,
                                     const float* x, int C_in, int Fd,
                                     int o0, int f0, int tn, float* y) {
    float32x4_t acc[NO][NV];
    for (int i = 0; i < NO; i++) {
        const float32x4_t bv = vdupq_n_f32(bias[o0 + i]);
        for (int v = 0; v < NV; v++) acc[i][v] = bv;
    }
    for (int c = 0; c < C_in; c++) {
        const float* xr = x + (size_t)c * Fd + f0;
        float32x4_t xv[NV];
        for (int v = 0; v < NV; v++) xv[v] = vld1q_f32(xr + 4 * v);
        for (int i = 0; i < NO; i++) {
            const float32x4_t wv = vdupq_n_f32(w[(size_t)(o0 + i) * C_in + c]);
            for (int v = 0; v < NV; v++) acc[i][v] = vfmaq_f32(acc[i][v], wv, xv[v]);
        }
    }
    for (int i = 0; i < NO; i++) {
        float* yr = y + (size_t)(o0 + i) * Fd + f0;
        if (tn == 4 * NV) {
            for (int v = 0; v < NV; v++) vst1q_f32(yr + 4 * v, acc[i][v]);
        } else {
            float tmp[4 * NV];
            for (int v = 0; v < NV; v++) vst1q_f32(tmp + 4 * v, acc[i][v]);
            for (int j = 0; j < tn; j++) yr[j] = tmp[j];
        }
    }
}

template <int NO>
static inline void skip1x1_rows_neon(const float* w, const float* bias,
                                     const float* x, int C_in, int Fd,
                                     int o0, float* y) {
    int f = 0;
    for (; f + 16 <= Fd; f += 16) skip1x1_tile_neon<NO, 4>(w, bias, x, C_in, Fd, o0, f, 16, y);
    for (; f + 8 <= Fd; f += 8)   skip1x1_tile_neon<NO, 2>(w, bias, x, C_in, Fd, o0, f, 8, y);
    if (f < Fd)                    skip1x1_tile_neon<NO, 2>(w, bias, x, C_in, Fd, o0, f, Fd - f, y);
}

static void skip1x1_run(const float* w, const float* bias, const float* x,
                        int C_in, int C_out, int Fd, float* y) {
    int o = 0;
    for (; o + 4 <= C_out; o += 4) skip1x1_rows_neon<4>(w, bias, x, C_in, Fd, o, y);
    switch (C_out - o) {
        case 3: skip1x1_rows_neon<3>(w, bias, x, C_in, Fd, o, y); break;
        case 2: skip1x1_rows_neon<2>(w, bias, x, C_in, Fd, o, y); break;
        case 1: skip1x1_rows_neon<1>(w, bias, x, C_in, Fd, o, y); break;
        default: break;
    }
}
#else
// Portable fallback (also used on x86 builds): c-major outer loop so x is
// loaded once per input channel instead of once per (output, input) pair.
static void skip1x1_run(const float* w, const float* bias, const float* x,
                        int C_in, int C_out, int Fd, float* y) {
    for (int o = 0; o < C_out; o++) {
        float* yr = y + (size_t)o * Fd;
        for (int f = 0; f < Fd; f++) yr[f] = bias[o];
    }
    for (int c = 0; c < C_in; c++) {
        const float* xr = x + (size_t)c * Fd;
        for (int o = 0; o < C_out; o++) {
            const float wv = w[(size_t)o * C_in + c];
            float* yr = y + (size_t)o * Fd;
            for (int f = 0; f < Fd; f++) yr[f] += wv * xr[f];
        }
    }
}
#endif

// Write the current frame (C_in, F_in) into the window's last row at the
// causal-pad column offset, run the direct conv, producing (C_out, F_out).
static void conv_run(ne_conv& cv, const float* x, float* y) {
    const int pad_l = (cv.kw - 1) / 2;
    // Ring buffer: logical row r (0=oldest .. kh-1=newest) lives at physical
    // row (r + cv.head) % kh. The newest row's physical slot is therefore
    // (kh-1+head)%kh, which is exactly the slot the previous end-of-frame
    // "shift" (now just cv.head++) vacated by dropping the old oldest row.
    const int kh = cv.kh;
    const int wrow = cv.prow[kh - 1];
    for (int c = 0; c < cv.C_in; c++)
        std::memcpy(&cv.win[((size_t)c * kh + wrow) * cv.Fp + pad_l],
                    x + (size_t)c * cv.F_in, cv.F_in * sizeof(float));
    const int FO = cv.F_out, sF = cv.sF;
    // De-interleave stride-2 rows once per (c, r) per frame so the FMA inner
    // loops are unit-stride for both stride cases.
    static thread_local std::vector<float> dei;   // (C_in*kh, 2, half) + slack
    const int half = cv.Fp / 2 + 2;
    if (sF == 2) {
        { const size_t need = (size_t)cv.C_in * kh * 2 * half + 64;
          if (dei.size() < need) dei.resize(need); }
        for (int c = 0; c < cv.C_in; c++)
            for (int r = 0; r < kh; r++) {
                const float* row = &cv.win[((size_t)c * kh + cv.prow[r]) * cv.Fp];
                float* ev = &dei[(((size_t)c * kh + r) * 2 + 0) * half];
                float* od = &dei[(((size_t)c * kh + r) * 2 + 1) * half];
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
                    const float* r0 = &cv.win[((size_t)ci * cv.kh + cv.prow[2*p]) * cv.Fp];
                    const float* r1 = &cv.win[((size_t)ci * cv.kh + cv.prow[2*p + 1]) * cv.Fp];
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
                        const float* p0 = &cv.win[((size_t)c * cv.kh + cv.prow[r]) * cv.Fp] + ft;
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
#elif defined(__ARM_NEON)
    {
        const float* dp = dei.data();
        int o = 0;
        for (; o + 4 <= cv.C_out; o += 4) conv_rows_neon<4>(cv, dp, half, o, y, FO);
        switch (cv.C_out - o) {
            case 3: conv_rows_neon<3>(cv, dp, half, o, y, FO); break;
            case 2: conv_rows_neon<2>(cv, dp, half, o, y, FO); break;
            case 1: conv_rows_neon<1>(cv, dp, half, o, y, FO); break;
            default: break;
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
                        ? &cv.win[((size_t)c * cv.kh + cv.prow[r]) * cv.Fp] + ft : nullptr;
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

// Ring-buffer end-of-frame step: logical row r (0=oldest..kh-1=newest) lives
// at physical row (r + head) % kh; advancing head by one is equivalent to
// the old memmove-shift (drops logical row 0, everything else ages by one)
// because the vacated physical slot is exactly where the next frame's
// newest-row write will land (see conv_run).
static inline void conv_advance(ne_conv& cv) { conv_set_head(cv, (cv.head + 1) % cv.kh); }

// generic (C, rows, L) ring window: write cur (C, L0) into the newest
// logical row (rows-1), i.e. physical row (rows-1+head)%rows, at col_off.
static void win_write(std::vector<float>& w, int C, int rows, int L,
                      const float* cur, int L0, int col_off, int head) {
    const int pr = (rows - 1 + head) % rows;
    for (int c = 0; c < C; c++)
        std::memcpy(&w[((size_t)c * rows + pr) * L + col_off],
                    cur + (size_t)c * L0, L0 * sizeof(float));
}

// ── init ───────────────────────────────────────────────────────────────────
static bool norm_ok(const ne_norm& n, int C) {
    return n.g.size() == (size_t)C && n.b.size() == (size_t)C;
}

static bool load_block(ne_block& bl, const localvqe_model& m, const std::string& p,
                       int C_in, int C_out, int kh, int kw, int F_in, int sF) {
    bl.n1.g = W(m, p + ".norm.weight"); bl.n1.b = W(m, p + ".norm.bias");
    if (!norm_ok(bl.n1, C_in)) return false;
    if (!conv_setup(bl.c1, W(m, p + ".conv.weight"), W(m, p + ".conv.bias"),
                    C_out, C_in, kh, kw, F_in, sF)) return false;
    bl.n2.g = W(m, p + ".resblock.norm.weight");
    bl.n2.b = W(m, p + ".resblock.norm.bias");
    if (!norm_ok(bl.n2, C_out)) return false;
    return conv_setup(bl.c2, W(m, p + ".resblock.conv.weight"),
                      W(m, p + ".resblock.conv.bias"),
                      C_out, C_out, kh, kw, bl.c1.F_out, 1);
}

// Largest (C, F) activation any scratch buffer must hold for this model.
static size_t conv_extent(const ne_conv& cv) {
    return std::max((size_t)cv.C_in * cv.F_in, (size_t)cv.C_out * cv.F_out);
}

bool ne_init(native_engine& ne, const localvqe_model& m) {
    const auto& hp = m.hparams;
    ne.F = hp.n_freq_bins; ne.dmax = hp.dmax; ne.power_c = hp.power_law_c;
    const auto& mc = hp.mic_channels;   // [2,16,20,20,20,20]
    const auto& fc = hp.far_channels;   // [2,16,20]
    const int kh = hp.kernel_size_h, kw = hp.kernel_size_w;
    // Every conv kernel path here (NEON lane-indexed taps, AVX-512 4-chain
    // tiles, the bf16 row pairs) is written for kw == 4 and kh <= 8, and the
    // bf16 path pairs rows (kh / 2 pairs), so kh must be even or the newest
    // row would silently drop out of the conv. Refuse anything else so the
    // caller falls back to the graph instead of computing garbage.
    if (kw != 4 || kh < 2 || kh > 8 || kh % 2 != 0) {
        fprintf(stderr, "native: unsupported kernel %dx%d (need kw==4, even kh<=8)\n", kh, kw);
        return false;
    }
    int F = ne.F;
    ne.loaded = false;
    // Geometry the frame code hardcodes: STFT-256 codec (512-pt FFT, F=256
    // bins), 2-channel (re, im) FE output into both encoder stacks, a shared
    // channel count for the mic/far align pair (the ref history and the
    // concat use mic_e2's width), five encoder halvings down to F/32, and a
    // 27-channel (3 x 9) complex-mask output for the CCM. Anything else must
    // fail here, not index out of range at runtime.
    if (F != 256 || hp.n_fft != 512 || hp.hop_length != 256 || hp.dmax <= 0 || hp.align_hidden <= 0 ||
        mc.size() != 6 || fc.size() != 3 || mc[0] != 2 || fc[0] != 2 ||
        mc[2] != fc[2] || F % 32 != 0) {
        fprintf(stderr, "native: unsupported model geometry\n");
        return false;
    }
    for (int c : mc) if (c <= 0) return false;
    for (int c : fc) if (c <= 0) return false;
    ne.win512.resize(512);
    for (int n = 0; n < 512; n++)
        ne.win512[n] = sqrtf(0.5f - 0.5f * cosf(2.0f * (float)M_PI * (n + 0.5f) / 512.0f));

    // encoders: mic1(F), mic2(F/2), far1(F), far2(F/2), mic3(F/4, C=2*mc2),
    // mic4(F/8), mic5(F/16)
    bool ok = true;
    ok = ok && load_block(ne.enc[0], m, "mic_enc1", mc[0], mc[1], kh, kw, F,     2);
    ok = ok && load_block(ne.enc[1], m, "mic_enc2", mc[1], mc[2], kh, kw, F/2,   2);
    ok = ok && load_block(ne.enc[2], m, "far_enc1", fc[0], fc[1], kh, kw, F,     2);
    ok = ok && load_block(ne.enc[3], m, "far_enc2", fc[1], fc[2], kh, kw, F/2,   2);
    ok = ok && load_block(ne.enc[4], m, "mic_enc3", mc[2]+fc[2], mc[3], kh, kw, F/4, 2);
    ok = ok && load_block(ne.enc[5], m, "mic_enc4", mc[3], mc[4], kh, kw, F/8,   2);
    ok = ok && load_block(ne.enc[6], m, "mic_enc5", mc[4], mc[5], kh, kw, F/16,  2);
    if (!ok) { fprintf(stderr, "native: encoder tensor shapes do not match hparams\n"); return false; }
    // The frame code assumes each stride-2 encoder exactly halves F.
    if (ne.enc[1].c1.F_in != ne.enc[0].c1.F_out || ne.enc[4].c1.F_in != ne.enc[1].c1.F_out ||
        ne.enc[4].c1.F_in != ne.enc[3].c1.F_out || ne.enc[5].c1.F_in != ne.enc[4].c1.F_out ||
        ne.enc[6].c1.F_in != ne.enc[5].c1.F_out || ne.enc[6].c1.F_out != F / 32) {
        fprintf(stderr, "native: encoder F chain does not halve to F/32\n"); return false;
    }

    // align (on mic_e2/far_e2 at F/4)
    ne.Hal = hp.align_hidden;
    ne.pmw = W(m, "align.pconv_mic.weight"); ne.pmb = W(m, "align.pconv_mic.bias");
    ne.prw = W(m, "align.pconv_ref.weight"); ne.prb = W(m, "align.pconv_ref.bias");
    ne.sw  = W(m, "align.conv.1.weight");    ne.sb  = W(m, "align.conv.1.bias");
    {
        const int H = ne.Hal, C2 = mc[2];
        if (ne.pmw.size() != (size_t)H * C2 || ne.pmb.size() != (size_t)H ||
            ne.prw.size() != (size_t)H * C2 || ne.prb.size() != (size_t)H ||
            ne.sw.size() != (size_t)H * 5 * 3 || ne.sb.size() != 1) {
            fprintf(stderr, "native: align tensor shapes do not match hparams\n"); return false;
        }
    }
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
    {
        const size_t IS = (size_t)ne.enc[6].c1.C_out * ne.enc[6].c1.F_out, Hb = ne.bn_h;
        if (Hb == 0 || ne.s4d_inw.size() != Hb * IS || ne.s4d_inb.size() != Hb ||
            ne.s4d_outw.size() != IS * Hb || ne.s4d_outb.size() != IS ||
            ne.s4d_ai.size() != Hb || ne.s4d_Br.size() != Hb || ne.s4d_Bi.size() != Hb ||
            ne.s4d_Cr.size() != Hb || ne.s4d_Ci.size() != Hb || ne.s4d_D.size() != IS) {
            fprintf(stderr, "native: bottleneck tensor shapes do not match\n"); return false;
        }
    }

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
        d.dec_n.g = W(m, p + ".deconv.norm.weight");
        d.dec_n.b = W(m, p + ".deconv.norm.bias");
        const size_t nb = W(m, p + ".deconv.conv.bias").size();
        const int C_next = (int)(nb / 2);
        // skip: dec i consumes mic encoder output mic_e(5-i), which must have Cd
        // channels. enc[] interleaves the far stack (mic1,2 far1,2 mic3,4,5), so
        // mic_e2/mic_e1 are enc[1]/enc[0], not enc[3]/enc[2].
        static const int skip_enc[5] = {6, 5, 4, 1, 0};
        const ne_conv& skip_src = ne.enc[skip_enc[i]].c1;
        if (skip_src.C_out != Cd || skip_src.F_out != Fd ||
            !norm_ok(d.skip_n, Cd) || !norm_ok(d.res_n, Cd) || !norm_ok(d.dec_n, Cd) ||
            d.skip_w.size() != (size_t)Cd * Cd || d.skip_b.size() != (size_t)Cd ||
            nb == 0 || nb % 2 != 0 ||
            !conv_setup(d.res, W(m, p + ".resblock.conv.weight"),
                        W(m, p + ".resblock.conv.bias"), Cd, Cd, kh, kw, Fd, 1) ||
            !conv_setup(d.deconv, W(m, p + ".deconv.conv.weight"),
                        W(m, p + ".deconv.conv.bias"), C_next * 2, Cd, kh, kw, Fd, 1)) {
            fprintf(stderr, "native: %s tensor shapes do not match\n", dn[i]); return false;
        }
        Cd = C_next; Fd *= 2;
    }
    // CCM: the mask is (3 x 9, F) complex-mixing taps, applied at F.
    if (Cd != 27 || Fd != F) {
        fprintf(stderr, "native: decoder output %d x %d, need 27 x %d for the CCM\n", Cd, Fd, F);
        return false;
    }

    ne.ccm_win.assign((size_t)2 * 3 * (F + 2), 0.0f);
    ne.s4d_hr.assign(ne.bn_h, 0.0f); ne.s4d_hi.assign(ne.bn_h, 0.0f);
    // Scratch: sized from this model's largest activation (plus the concat
    // and the pre-shuffle deconv output), never a fixed constant.
    size_t mx = 2 * (size_t)mc[2] * (F / 4);
    for (auto& b : ne.enc) mx = std::max({mx, conv_extent(b.c1), conv_extent(b.c2)});
    for (auto& d : ne.dec) mx = std::max({mx, conv_extent(d.res), conv_extent(d.deconv)});
    mx = std::max(mx, (size_t)2 * F);
    ne.scratch = mx;
    ne.t0.resize(mx); ne.t1.resize(mx); ne.t2.resize(mx); ne.t3.resize(mx); ne.t4.resize(mx);
    for (auto& s : ne.skipsave) s.resize(mx);
    ne.loaded = true;
    ne_reset(ne);
    return ne.loaded;
}

void ne_reset(native_engine& ne) {
    for (auto& b : ne.enc) {
        std::fill(b.c1.win.begin(), b.c1.win.end(), 0.0f); conv_set_head(b.c1, 0);
        std::fill(b.c2.win.begin(), b.c2.win.end(), 0.0f); conv_set_head(b.c2, 0);
    }
    for (auto& d : ne.dec) {
        std::fill(d.res.win.begin(), d.res.win.end(), 0.0f); conv_set_head(d.res, 0);
        std::fill(d.deconv.win.begin(), d.deconv.win.end(), 0.0f); conv_set_head(d.deconv, 0);
    }
    std::fill(ne.K_win.begin(), ne.K_win.end(), 0.0f);
    std::fill(ne.ref_win.begin(), ne.ref_win.end(), 0.0f);
    std::fill(ne.S_win.begin(), ne.S_win.end(), 0.0f);
    std::fill(ne.ccm_win.begin(), ne.ccm_win.end(), 0.0f);
    ne.K_head = 0; ne.ref_head = 0; ne.S_head = 0; ne.ccm_head = 0;
    std::fill(ne.s4d_hr.begin(), ne.s4d_hr.end(), 0.0f);
    std::fill(ne.s4d_hi.begin(), ne.s4d_hi.end(), 0.0f);
}

// run one encoder-style block: y = silu(conv1(norm1(x))); out = silu(conv2(norm2(y))) + y
static void run_block(ne_block& bl, const float* x, float* tmp, float* y, float* out) {
    norm_cf(x, tmp, bl.c1.C_in, bl.c1.F_in, bl.n1.g.data(), bl.n1.b.data());
    conv_run(bl.c1, tmp, y);
    const int n1 = bl.c1.C_out * bl.c1.F_out;
    silu_inplace(y, n1, nullptr);
    norm_cf(y, tmp, bl.c2.C_in, bl.c2.F_in, bl.n2.g.data(), bl.n2.b.data());
    conv_run(bl.c2, tmp, out);
    silu_inplace(out, n1, y);
}

void ne_process_frame(native_engine& ne, const float* mic_win,
                      const float* ref_win, float* out_win) {
    if (g_taps_on) g_taps.clear();
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
    win_write(ne.K_win, H, dmax, F2, Kc.data(), F2, 0, ne.K_head);
    win_write(ne.ref_win, C2, dmax, F2, far_e2, F2, 0, ne.ref_head);
    const float scl = 1.0f / sqrtf((float)F2);
    for (int d = 0; d < dmax; d++) {
        const int pd = (d + ne.K_head) % dmax;
        for (int h = 0; h < H; h++) {
            const float* kr = &ne.K_win[((size_t)h * dmax + pd) * F2];
            const float* qr = &Q[(size_t)h * F2];
            sim[(size_t)d * H + h] = dot4(kr, qr, F2) * scl;   // V layout (H, 1, dmax) below
        }
    }
    // smooth window: (H, 5, dmax+2); write V (per h, dmax) at row 4, col 1
    {
        const int swr = (5 - 1 + ne.S_head) % 5;
        for (int h = 0; h < H; h++) {
            float* dst = &ne.S_win[((size_t)h * 5 + swr) * (dmax + 2) + 1];
            for (int d = 0; d < dmax; d++) dst[d] = sim[(size_t)d * H + h];
        }
    }
    // smooth conv: kernel (1, H, 5, 3) -> out (dmax)
    int srow[5];
    for (int r = 0; r < 5; r++) srow[r] = (r + ne.S_head) % 5;
    for (int d = 0; d < dmax; d++) {
        float acc = ne.sb[0];
        for (int h = 0; h < H; h++)
            for (int r = 0; r < 5; r++) {
                const float* row = &ne.S_win[((size_t)h * 5 + srow[r]) * (dmax + 2)];
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
    static thread_local std::vector<int> refrow;
    refrow.resize(dmax);
    for (int d = 0; d < dmax; d++) refrow[d] = (d + ne.ref_head) % dmax;
    for (int c = 0; c < C2; c++) {
        float* ar = &aligned[(size_t)c * F2];
        std::memset(ar, 0, F2 * sizeof(float));
        for (int d = 0; d < dmax; d++) {
            const float a = attn[d];
            const float* rr = &ne.ref_win[((size_t)c * dmax + refrow[d]) * F2];
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
        const float* wr = &ne.s4d_inw[(size_t)h * IS];
        v[h] = ne.s4d_inb[h] + dot4(wr, u, IS);
    }
    for (int h = 0; h < Hb; h++) {
        const float hr = ne.s4d_ar[h]*ne.s4d_hr[h] - ne.s4d_ai[h]*ne.s4d_hi[h] + ne.s4d_Br[h]*v[h];
        const float hi = ne.s4d_ar[h]*ne.s4d_hi[h] + ne.s4d_ai[h]*ne.s4d_hr[h] + ne.s4d_Bi[h]*v[h];
        ne.s4d_hr[h] = hr; ne.s4d_hi[h] = hi;
        ybn[h] = ne.s4d_Cr[h]*hr - ne.s4d_Ci[h]*hi;
    }
    for (int i = 0; i < IS; i++) {
        const float* wr = &ne.s4d_outw[(size_t)i * Hb];
        bnout[i] = ne.s4d_outb[i] + dot4(wr, ybn.data(), Hb) + ne.s4d_D[i] * u[i];
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
    if (dbuf1.size() < ne.scratch) dbuf1.resize(ne.scratch);
    if (dbuf2.size() < ne.scratch) dbuf2.resize(ne.scratch);
    for (int i = 0; i < 5; i++) {
        ne_dec& d = ne.dec[i];
        const int C = d.res.C_in, Fd = d.res.F_in;
        // skip = 1x1(norm(x_en)); y = x + skip
        norm_cf(skips[i], t0, C, Fd, d.skip_n.g.data(), d.skip_n.b.data());
        skip1x1_run(d.skip_w.data(), d.skip_b.data(), t0, C, C, Fd, t1);
        const int n = C * Fd;
        for (int j = 0; j < n; j++) t1[j] += x[j];       // y
        // res
        norm_cf(t1, t0, C, Fd, d.res_n.g.data(), d.res_n.b.data());
        conv_run(d.res, t0, t2);
        silu_inplace(t2, n, t1);
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
            silu_inplace(dst, n2, nullptr);
        x = dst;
        tap(dtap[i], x, n2);
    }

    // ── CCM: mask = x (27, F); apply to RAW mic spectrum window ──
    static const float VR[3] = {1.0f, -0.5f, -0.5f};
    static const float VI[3] = {0.0f, 0.86602540378f, -0.86602540378f};
    win_write(ne.ccm_win, 2, 3, F + 2, micS.data(), F, 1, ne.ccm_head);
    static thread_local std::vector<float> er, ei;
    er.assign(F, 0.0f); ei.assign(F, 0.0f);
    for (int mrow = 0; mrow < 3; mrow++) {
        const int pmrow = (mrow + ne.ccm_head) % 3;
        const float* xr = &ne.ccm_win[((size_t)0 * 3 + pmrow) * (F + 2)];
        const float* xi = &ne.ccm_win[((size_t)1 * 3 + pmrow) * (F + 2)];
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

    // ── end-of-frame: advance all ring windows (was memmove-shift) ──
    for (auto& b : ne.enc) { conv_advance(b.c1); conv_advance(b.c2); }
    for (auto& d : ne.dec) { conv_advance(d.res); conv_advance(d.deconv); }
    ne.K_head = (ne.K_head + 1) % dmax;
    ne.ref_head = (ne.ref_head + 1) % dmax;
    ne.S_head = (ne.S_head + 1) % 5;
    ne.ccm_head = (ne.ccm_head + 1) % 3;
}
