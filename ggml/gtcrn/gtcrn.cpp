/**
 * GTCRN-AEC forward — 1:1 C++ translation of train/export_gtcrn_ggml.py's
 * validated reference (which matches PyTorch to <1e-4). See gtcrn.h.
 *
 * All ConvTranspose2d weights are pre-rewritten by the exporter into regular
 * conv weights (1x1 deconv transposed; depthwise stride-1 deconv spatially
 * flipped; stride-2 deconv = zero-insert upsample + flipped conv), so there is
 * a single conv code path here. BatchNorm is pre-folded into every conv.
 */

#include "gtcrn.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// ── small ops ───────────────────────────────────────────────────────────────

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// General 2D conv. Weight (OC, IC/groups, KT, KF). Time stride 1; freq stride
// stride_f. Asymmetric time padding (top/bottom) lets the depthwise blocks be
// causal (top-pad only). Out-of-range taps are treated as zero.
static GTensor conv2d(const GTensor& x, const NpyArray& w, const NpyArray& b,
                      int groups, int stride_f, int pad_f,
                      int pad_t_top, int pad_t_bot, int dil_t, int dil_f) {
    const int OC = (int)w.dim(0), ING = (int)w.dim(1), KT = (int)w.dim(2), KF = (int)w.dim(3);
    const int IC = x.C, Tin = x.T, Fin = x.F;
    const int Tout = Tin + pad_t_top + pad_t_bot - dil_t * (KT - 1);
    const int Fout = (Fin + 2 * pad_f - dil_f * (KF - 1) - 1) / stride_f + 1;
    const int opg = OC / groups, ipg = IC / groups;
    GTensor y; y.resize(OC, Tout, Fout);
    for (int g = 0; g < groups; ++g) {
        for (int oc = 0; oc < opg; ++oc) {
            const int co = g * opg + oc;
            const float bias = b.data.empty() ? 0.0f : b.data[co];
            for (int t = 0; t < Tout; ++t) {
                for (int f = 0; f < Fout; ++f) {
                    float s = bias;
                    for (int ci = 0; ci < ING; ++ci) {
                        const int cin = g * ipg + ci;
                        for (int kt = 0; kt < KT; ++kt) {
                            const int tin = t + kt * dil_t - pad_t_top;
                            if (tin < 0 || tin >= Tin) continue;
                            for (int kf = 0; kf < KF; ++kf) {
                                const int fin = f * stride_f + kf * dil_f - pad_f;
                                if (fin < 0 || fin >= Fin) continue;
                                s += w.data[(((size_t)co * ING + ci) * KT + kt) * KF + kf] *
                                     x.at(cin, tin, fin);
                            }
                        }
                    }
                    y.at(co, t, f) = s;
                }
            }
        }
    }
    return y;
}

static void prelu_(GTensor& x, const NpyArray& slope) {
    const bool scalar = slope.data.size() == 1;
    for (int c = 0; c < x.C; ++c) {
        const float a = scalar ? slope.data[0] : slope.data[c];
        for (int t = 0; t < x.T; ++t)
            for (int f = 0; f < x.F; ++f) {
                float& v = x.at(c, t, f);
                if (v < 0) v *= a;
            }
    }
}

static void tanh_(GTensor& x) {
    for (auto& v : x.d) v = std::tanh(v);
}

// SFE: 3-tap freq unfold. out channel c*3+k, k in {0:f-1, 1:f, 2:f+1} (zero pad).
static GTensor sfe(const GTensor& x) {
    GTensor y; y.resize(x.C * 3, x.T, x.F);
    for (int c = 0; c < x.C; ++c)
        for (int t = 0; t < x.T; ++t)
            for (int f = 0; f < x.F; ++f) {
                float fm = (f - 1 >= 0) ? x.at(c, t, f - 1) : 0.0f;
                float f0 = x.at(c, t, f);
                float fp = (f + 1 < x.F) ? x.at(c, t, f + 1) : 0.0f;
                y.at(c * 3 + 0, t, f) = fm;
                y.at(c * 3 + 1, t, f) = f0;
                y.at(c * 3 + 2, t, f) = fp;
            }
    return y;
}

// ERB band-merge: (C,T,257) -> (C,T,129). low 65 pass; high 192 -> 64 via Wbm(64,192).
static GTensor erb_bm(const GTensor& x, const NpyArray& Wbm) {
    GTensor y; y.resize(x.C, x.T, 129);
    for (int c = 0; c < x.C; ++c)
        for (int t = 0; t < x.T; ++t) {
            for (int f = 0; f < 65; ++f) y.at(c, t, f) = x.at(c, t, f);
            for (int j = 0; j < 64; ++j) {
                float s = 0.0f;
                for (int i = 0; i < 192; ++i)
                    s += x.at(c, t, 65 + i) * Wbm.data[(size_t)j * 192 + i];
                y.at(c, t, 65 + j) = s;
            }
        }
    return y;
}

// ERB band-split (synthesis): (C,T,129) -> (C,T,257). low 65 pass; high 64 -> 192 via Wbs(192,64).
static GTensor erb_bs(const GTensor& x, const NpyArray& Wbs) {
    GTensor y; y.resize(x.C, x.T, 257);
    for (int c = 0; c < x.C; ++c)
        for (int t = 0; t < x.T; ++t) {
            for (int f = 0; f < 65; ++f) y.at(c, t, f) = x.at(c, t, f);
            for (int j = 0; j < 192; ++j) {
                float s = 0.0f;
                for (int i = 0; i < 64; ++i)
                    s += x.at(c, t, 65 + i) * Wbs.data[(size_t)j * 64 + i];
                y.at(c, t, 65 + j) = s;
            }
        }
    return y;
}

// GRU over a length-L sequence. x: L*I row-major. weights PyTorch layout
// (3H,I)/(3H,H), gate order [r,z,n], n = tanh(gi_n + r*gh_n). out: L*H.
static std::vector<float> gru_seq(const float* x, int L, int I,
                                  const NpyArray& wih, const NpyArray& whh,
                                  const NpyArray& bih, const NpyArray& bhh,
                                  bool reverse) {
    const int H = (int)whh.dim(1);
    std::vector<float> out((size_t)L * H, 0.0f);
    std::vector<float> h(H, 0.0f), gi(3 * H), gh(3 * H);
    for (int s = 0; s < L; ++s) {
        const int ti = reverse ? (L - 1 - s) : s;
        const float* xt = x + (size_t)ti * I;
        for (int gidx = 0; gidx < 3 * H; ++gidx) {
            float a = bih.data[gidx];
            const float* wr = &wih.data[(size_t)gidx * I];
            for (int i = 0; i < I; ++i) a += wr[i] * xt[i];
            gi[gidx] = a;
            float c = bhh.data[gidx];
            const float* hr = &whh.data[(size_t)gidx * H];
            for (int j = 0; j < H; ++j) c += hr[j] * h[j];
            gh[gidx] = c;
        }
        for (int k = 0; k < H; ++k) {
            float r = sigmoidf(gi[k] + gh[k]);
            float z = sigmoidf(gi[H + k] + gh[H + k]);
            float n = std::tanh(gi[2 * H + k] + r * gh[2 * H + k]);
            h[k] = (1.0f - z) * n + z * h[k];
        }
        for (int k = 0; k < H; ++k) out[(size_t)ti * H + k] = h[k];
    }
    return out;
}

// Grouped RNN. x: (N, seq, I) row-major. Split I in half -> rnn1/rnn2; each
// optionally bidirectional (forward||reverse concat). Returns (N, seq, Iout)
// where Iout = (rnn1 out) + (rnn2 out). For the GTCRN configs Iout == 16.
static std::vector<float> grnn(const std::vector<float>& x, int N, int seq, int I,
                               const GtcrnModel& M, const std::string& prefix, bool bidir) {
    const int half = I / 2;
    // determine per-rnn output width from whh
    const NpyArray& whh1 = M.W.at(prefix + ".rnn1.weight_hh_l0");
    const int H1 = (int)whh1.dim(1);
    const int out1 = bidir ? 2 * H1 : H1;
    const NpyArray& whh2 = M.W.at(prefix + ".rnn2.weight_hh_l0");
    const int H2 = (int)whh2.dim(1);
    const int out2 = bidir ? 2 * H2 : H2;
    const int Iout = out1 + out2;
    std::vector<float> y((size_t)N * seq * Iout, 0.0f);

    std::vector<float> sub((size_t)seq * half);
    for (int n = 0; n < N; ++n) {
        for (int rnn = 0; rnn < 2; ++rnn) {
            const std::string p = prefix + (rnn == 0 ? ".rnn1" : ".rnn2");
            const int feat_off = rnn * half;
            const int out_off = rnn == 0 ? 0 : out1;
            const int Hh = rnn == 0 ? H1 : H2;
            const int outw = rnn == 0 ? out1 : out2;
            // extract this half-sequence
            for (int s = 0; s < seq; ++s)
                for (int i = 0; i < half; ++i)
                    sub[(size_t)s * half + i] = x[((size_t)n * seq + s) * I + feat_off + i];
            auto yf = gru_seq(sub.data(), seq, half,
                              M.W.at(p + ".weight_ih_l0"), M.W.at(p + ".weight_hh_l0"),
                              M.W.at(p + ".bias_ih_l0"), M.W.at(p + ".bias_hh_l0"), false);
            std::vector<float> yr;
            if (bidir)
                yr = gru_seq(sub.data(), seq, half,
                             M.W.at(p + ".weight_ih_l0_reverse"), M.W.at(p + ".weight_hh_l0_reverse"),
                             M.W.at(p + ".bias_ih_l0_reverse"), M.W.at(p + ".bias_hh_l0_reverse"), true);
            for (int s = 0; s < seq; ++s) {
                float* dst = &y[((size_t)n * seq + s) * Iout + out_off];
                for (int k = 0; k < Hh; ++k) dst[k] = yf[(size_t)s * Hh + k];
                if (bidir)
                    for (int k = 0; k < Hh; ++k) dst[Hh + k] = yr[(size_t)s * Hh + k];
                (void)outw;
            }
        }
    }
    return y;
}

// linear over last dim: x (M, I) -> (M, O), W (O,I), b (O).
static std::vector<float> linear(const std::vector<float>& x, int Mrows, int I,
                                 const NpyArray& w, const NpyArray& b) {
    const int O = (int)w.dim(0);
    std::vector<float> y((size_t)Mrows * O);
    for (int m = 0; m < Mrows; ++m)
        for (int o = 0; o < O; ++o) {
            float s = b.data[o];
            const float* wr = &w.data[(size_t)o * I];
            const float* xr = &x[(size_t)m * I];
            for (int i = 0; i < I; ++i) s += wr[i] * xr[i];
            y[(size_t)m * O + o] = s;
        }
    return y;
}

// LayerNorm over the joint (33,16) tail per time frame. x: (T,33,16) flat.
static void layernorm_last2_(std::vector<float>& x, int T, int Fd, int C,
                             const NpyArray& w, const NpyArray& b, float eps) {
    const int n = Fd * C;
    for (int t = 0; t < T; ++t) {
        float* xt = &x[(size_t)t * n];
        float mu = 0.0f;
        for (int i = 0; i < n; ++i) mu += xt[i];
        mu /= n;
        float var = 0.0f;
        for (int i = 0; i < n; ++i) { float d = xt[i] - mu; var += d * d; }
        var /= n;
        float inv = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < n; ++i) xt[i] = (xt[i] - mu) * inv * w.data[i] + b.data[i];
    }
}

// Temporal Recurrent Attention: gate x (C,T,F) by sigmoid(fc(gru(mean_f x^2))).
static GTensor tra(const GTensor& x, const GtcrnModel& M, const std::string& prefix) {
    const int C = x.C, T = x.T, Fd = x.F;
    std::vector<float> seq((size_t)T * C);  // (T, C)
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) {
            float s = 0.0f;
            for (int f = 0; f < Fd; ++f) { float v = x.at(c, t, f); s += v * v; }
            seq[(size_t)t * C + c] = s / Fd;
        }
    auto y = gru_seq(seq.data(), T, C, M.W.at(prefix + ".tra.gru.weight_ih_l0"),
                     M.W.at(prefix + ".tra.gru.weight_hh_l0"),
                     M.W.at(prefix + ".tra.gru.bias_ih_l0"),
                     M.W.at(prefix + ".tra.gru.bias_hh_l0"), false);  // (T, 2C)
    auto at = linear(y, T, 2 * C, M.W.at(prefix + ".tra.fc.w"), M.W.at(prefix + ".tra.fc.b"));  // (T,C)
    GTensor out = x;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) {
            float g = sigmoidf(at[(size_t)t * C + c]);
            for (int f = 0; f < Fd; ++f) out.at(c, t, f) *= g;
        }
    return out;
}

// channel shuffle: interleave processed h and bypass x2 -> 2C, out[2c]=h, out[2c+1]=x2.
static GTensor shuffle2(const GTensor& h, const GTensor& x2) {
    GTensor y; y.resize(h.C * 2, h.T, h.F);
    for (int c = 0; c < h.C; ++c)
        for (int t = 0; t < h.T; ++t)
            for (int f = 0; f < h.F; ++f) {
                y.at(2 * c + 0, t, f) = h.at(c, t, f);
                y.at(2 * c + 1, t, f) = x2.at(c, t, f);
            }
    return y;
}

static GTensor chunk_first(const GTensor& x) {
    GTensor y; y.resize(x.C / 2, x.T, x.F);
    for (int c = 0; c < y.C; ++c)
        for (int t = 0; t < x.T; ++t)
            for (int f = 0; f < x.F; ++f) y.at(c, t, f) = x.at(c, t, f);
    return y;
}
static GTensor chunk_second(const GTensor& x) {
    GTensor y; y.resize(x.C / 2, x.T, x.F);
    const int off = x.C / 2;
    for (int c = 0; c < y.C; ++c)
        for (int t = 0; t < x.T; ++t)
            for (int f = 0; f < x.F; ++f) y.at(c, t, f) = x.at(off + c, t, f);
    return y;
}

// zero-insertion upsample along freq: F -> (F-1)*factor + 1.
static GTensor upsample_zero_f(const GTensor& x, int factor) {
    const int Fout = (x.F - 1) * factor + 1;
    GTensor y; y.resize(x.C, x.T, Fout);
    for (int c = 0; c < x.C; ++c)
        for (int t = 0; t < x.T; ++t)
            for (int f = 0; f < x.F; ++f) y.at(c, t, f * factor) = x.at(c, t, f);
    return y;
}

static GTensor add_(const GTensor& a, const GTensor& b) {
    GTensor y = a;
    for (size_t i = 0; i < y.d.size(); ++i) y.d[i] += b.d[i];
    return y;
}

// ── block forwards ──────────────────────────────────────────────────────────

// GTConvBlock (encoder or decoder; deconv difference is baked into weights).
static GTensor gt_block(const GTensor& x, const GtcrnModel& M, const std::string& p, int dil) {
    GTensor x1 = chunk_first(x), x2 = chunk_second(x);
    GTensor s = sfe(x1);
    GTensor h = conv2d(s, M.W.at(p + ".pc1.w"), M.W.at(p + ".pc1.b"), 1, 1, 0, 0, 0, 1, 1);
    prelu_(h, M.W.at(p + ".pc1.prelu"));
    const int hidden = h.C;
    h = conv2d(h, M.W.at(p + ".dw.w"), M.W.at(p + ".dw.b"), hidden, 1, /*pad_f*/1,
               /*pad_t_top*/2 * dil, /*pad_t_bot*/0, /*dil_t*/dil, /*dil_f*/1);
    prelu_(h, M.W.at(p + ".dw.prelu"));
    h = conv2d(h, M.W.at(p + ".pc2.w"), M.W.at(p + ".pc2.b"), 1, 1, 0, 0, 0, 1, 1);
    h = tra(h, M, p);
    return shuffle2(h, x2);
}

// ConvBlock: conv(folded BN) -> PReLU (or tanh if last). deconv -> upsample+conv.
static GTensor conv_block(const GTensor& x, const GtcrnModel& M, const std::string& p,
                          int groups, int stride_f, bool deconv, bool is_last) {
    GTensor y;
    if (!deconv) {
        y = conv2d(x, M.W.at(p + ".w"), M.W.at(p + ".b"), groups, stride_f, 2, 0, 0, 1, 1);
    } else {
        GTensor xu = upsample_zero_f(x, stride_f);
        y = conv2d(xu, M.W.at(p + ".w"), M.W.at(p + ".b"), groups, 1, 2, 0, 0, 1, 1);
    }
    if (is_last) tanh_(y);
    else prelu_(y, M.W.at(p + ".prelu"));
    return y;
}

// DPGRNN: intra (bidir, over freq) + inter (unidir, over time), each with FC,
// LayerNorm and a residual.
static GTensor dpgrnn(const GTensor& x, const GtcrnModel& M, const std::string& d) {
    const int C = x.C, T = x.T, Fd = x.F;  // (16,T,33)
    // xp = permute to (T,F,C)
    std::vector<float> xp((size_t)T * Fd * C);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < Fd; ++f)
            for (int c = 0; c < C; ++c) xp[((size_t)t * Fd + f) * C + c] = x.at(c, t, f);

    // intra: grnn over freq sequences (N=T, seq=Fd, I=C)
    auto intra = grnn(xp, T, Fd, C, M, d + ".intra_rnn", true);
    intra = linear(intra, T * Fd, C, M.W.at(d + ".intra_fc.w"), M.W.at(d + ".intra_fc.b"));
    layernorm_last2_(intra, T, Fd, C, M.W.at(d + ".intra_ln.w"), M.W.at(d + ".intra_ln.b"), 1e-8f);
    for (size_t i = 0; i < intra.size(); ++i) intra[i] += xp[i];  // intra_out (T,F,C)

    // inter: permute to (F,T,C); grnn over time (N=Fd, seq=T, I=C)
    std::vector<float> xq((size_t)Fd * T * C);
    for (int f = 0; f < Fd; ++f)
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) xq[((size_t)f * T + t) * C + c] = intra[((size_t)t * Fd + f) * C + c];
    auto inter = grnn(xq, Fd, T, C, M, d + ".inter_rnn", false);
    inter = linear(inter, Fd * T, C, M.W.at(d + ".inter_fc.w"), M.W.at(d + ".inter_fc.b"));
    // permute back to (T,F,C)
    std::vector<float> interp((size_t)T * Fd * C);
    for (int f = 0; f < Fd; ++f)
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) interp[((size_t)t * Fd + f) * C + c] = inter[((size_t)f * T + t) * C + c];
    layernorm_last2_(interp, T, Fd, C, M.W.at(d + ".inter_ln.w"), M.W.at(d + ".inter_ln.b"), 1e-8f);
    for (size_t i = 0; i < interp.size(); ++i) interp[i] += intra[i];  // inter_out

    // back to (C,T,F)
    GTensor out; out.resize(C, T, Fd);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < Fd; ++f)
            for (int c = 0; c < C; ++c) out.at(c, t, f) = interp[((size_t)t * Fd + f) * C + c];
    return out;
}

// feature extraction for one spectrum (257,T,2) -> (9,T,129).
static GTensor feat(const float* spec, int T, const GtcrnModel& M) {
    GTensor f3; f3.resize(3, T, 257);  // [mag, real, imag]
    for (int t = 0; t < T; ++t)
        for (int fb = 0; fb < 257; ++fb) {
            float re = spec[((size_t)fb * T + t) * 2 + 0];
            float im = spec[((size_t)fb * T + t) * 2 + 1];
            f3.at(1, t, fb) = re;
            f3.at(2, t, fb) = im;
            f3.at(0, t, fb) = std::sqrt(re * re + im * im + 1e-12f);
        }
    GTensor banded = erb_bm(f3, M.W.at("erb.bm"));  // (3,T,129)
    return sfe(banded);                              // (9,T,129)
}

// ── public API ──────────────────────────────────────────────────────────────

bool GtcrnModel::load(const char* path, bool verbose) {
    struct ggml_context* ctx = nullptr;
    struct gguf_init_params gp;
    gp.no_alloc = false;
    gp.ctx = &ctx;
    struct gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "gtcrn: failed to open GGUF %s\n", path); return false; }
    const int n = (int)gguf_get_n_tensors(g);
    for (int i = 0; i < n; ++i) {
        std::string name = gguf_get_tensor_name(g, i);
        W[name] = load_tensor_from_ggml(ctx, name, g, verbose);
    }
    gguf_free(g);
    if (ctx) ggml_free(ctx);
    if (verbose) printf("gtcrn: loaded %d tensors\n", n);
    return !W.empty();
}

std::vector<float> GtcrnModel::forward(const float* spec_e, const float* spec_y, int T,
                                       std::map<std::string, NpyArray>* cap) const {
    auto put = [&](const char* name, const GTensor& g) {
        if (cap) (*cap)[name] = g.to_npy();
    };

    GTensor fe = feat(spec_e, T, *this);
    GTensor fy = feat(spec_y, T, *this);
    // concat e||y along channels -> (18,T,129)
    GTensor ft; ft.resize(18, T, 129);
    for (int c = 0; c < 9; ++c)
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < 129; ++f) {
                ft.at(c, t, f) = fe.at(c, t, f);
                ft.at(9 + c, t, f) = fy.at(c, t, f);
            }
    put("feat", ft);

    // encoder
    GTensor en0 = conv_block(ft, *this, "encoder.en_convs.0", 1, 2, false, false); put("enc0", en0);
    GTensor en1 = conv_block(en0, *this, "encoder.en_convs.1", 2, 2, false, false); put("enc1", en1);
    GTensor en2 = gt_block(en1, *this, "encoder.en_convs.2", 1); put("enc2", en2);
    GTensor en3 = gt_block(en2, *this, "encoder.en_convs.3", 2); put("enc3", en3);
    GTensor en4 = gt_block(en3, *this, "encoder.en_convs.4", 5); put("enc4", en4);

    // dual-path RNN
    GTensor d1 = dpgrnn(en4, *this, "dpgrnn1"); put("dpgrnn1", d1);
    GTensor d2 = dpgrnn(d1, *this, "dpgrnn2"); put("dpgrnn2", d2);

    // decoder (skip-add encoder outs, reversed)
    GTensor x = gt_block(add_(d2, en4), *this, "decoder.de_convs.0", 5); put("dec0", x);
    x = gt_block(add_(x, en3), *this, "decoder.de_convs.1", 2); put("dec1", x);
    x = gt_block(add_(x, en2), *this, "decoder.de_convs.2", 1); put("dec2", x);
    x = conv_block(add_(x, en1), *this, "decoder.de_convs.3", 2, 2, true, false); put("dec3", x);
    x = conv_block(add_(x, en0), *this, "decoder.de_convs.4", 1, 2, true, true); put("dec4", x);

    // ERB synthesis -> complex ratio mask on spec_e
    GTensor m = erb_bs(x, W.at("erb.bs"));  // (2,T,257)
    put("mask", m);

    std::vector<float> out((size_t)257 * T * 2);
    for (int fb = 0; fb < 257; ++fb)
        for (int t = 0; t < T; ++t) {
            float er = spec_e[((size_t)fb * T + t) * 2 + 0];
            float ei = spec_e[((size_t)fb * T + t) * 2 + 1];
            float mr = m.at(0, t, fb);
            float mi = m.at(1, t, fb);
            out[((size_t)fb * T + t) * 2 + 0] = er * mr - ei * mi;
            out[((size_t)fb * T + t) * 2 + 1] = ei * mr + er * mi;
        }
    if (cap) {
        NpyArray o; o.shape = {1, 257, T, 2}; o.data = out; (*cap)["out_spec"] = o;
    }
    return out;
}

// ── STFT / ISTFT (torch-matching, via shipped windowed-DFT matrices) ─────────

std::vector<float> GtcrnModel::stft(const float* sig, int L) const {
    const int N = 512, H = 256, Fb = 257, T = n_frames(L);
    const NpyArray& wcos = W.at("stft.wcos");  // (257,512)
    const NpyArray& wsin = W.at("stft.wsin");
    std::vector<float> pad((size_t)L + N, 0.0f);  // reflect-pad H each side
    for (int i = 0; i < L; ++i) pad[H + i] = sig[i];
    for (int j = 0; j < H; ++j) { pad[j] = sig[H - j]; pad[H + L + j] = sig[L - 2 - j]; }
    std::vector<float> spec((size_t)Fb * T * 2, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float* fr = &pad[(size_t)t * H];
        for (int f = 0; f < Fb; ++f) {
            const float* wc = &wcos.data[(size_t)f * N];
            const float* ws = &wsin.data[(size_t)f * N];
            float re = 0, im = 0;
            for (int n = 0; n < N; ++n) { re += wc[n] * fr[n]; im += ws[n] * fr[n]; }
            spec[((size_t)f * T + t) * 2 + 0] = re;
            spec[((size_t)f * T + t) * 2 + 1] = im;
        }
    }
    return spec;
}

std::vector<float> GtcrnModel::istft(const float* spec, int T, int L) const {
    const int N = 512, H = 256, Fb = 257;
    const NpyArray& icos = W.at("stft.icos");  // (512,257)
    const NpyArray& isin = W.at("stft.isin");
    const NpyArray& win2 = W.at("stft.win2");  // (512,)
    std::vector<float> ytmp((size_t)L + N, 0.0f), wenv((size_t)L + N, 0.0f);
    std::vector<float> ft(N);
    for (int t = 0; t < T; ++t) {
        std::fill(ft.begin(), ft.end(), 0.0f);
        for (int f = 0; f < Fb; ++f) {
            float re = spec[((size_t)f * T + t) * 2 + 0];
            float im = spec[((size_t)f * T + t) * 2 + 1];
            for (int n = 0; n < N; ++n)
                ft[n] += icos.data[(size_t)n * Fb + f] * re + isin.data[(size_t)n * Fb + f] * im;
        }
        for (int n = 0; n < N; ++n) { ytmp[(size_t)t * H + n] += ft[n]; wenv[(size_t)t * H + n] += win2.data[n]; }
    }
    std::vector<float> y((size_t)L, 0.0f);
    for (int i = 0; i < L; ++i) {
        float w = wenv[(size_t)H + i];
        y[i] = (w > 1e-11f) ? ytmp[(size_t)H + i] / w : 0.0f;
    }
    return y;
}
