#include "daf_frontend.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

// ── tiny iterative radix-2 complex FFT ─────────────────────────────────────
static void fft_inplace(std::vector<float>& re, std::vector<float>& im, bool inv) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; i++) {            // bit reversal
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
                const double ncr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr; cr = ncr;
            }
        }
    }
    if (inv) {
        const float s = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
    }
}

// rfft of x (n real) -> nb = n/2+1 bins
static void rfft(const float* x, int n, float* outr, float* outi) {
    static thread_local std::vector<float> re, im;
    re.assign(x, x + n); im.assign(n, 0.0f);
    fft_inplace(re, im, false);
    for (int k = 0; k <= n / 2; k++) { outr[k] = re[k]; outi[k] = im[k]; }
}

// irfft: nb = n/2+1 bins (hermitian) -> n real samples
static void irfft(const float* br, const float* bi, int n, float* out) {
    static thread_local std::vector<float> re, im;
    re.assign(n, 0.0f); im.assign(n, 0.0f);
    for (int k = 0; k <= n / 2; k++) { re[k] = br[k]; im[k] = bi[k]; }
    for (int k = 1; k < n / 2; k++) { re[n-k] = br[k]; im[n-k] = -bi[k]; }
    fft_inplace(re, im, true);
    for (int i = 0; i < n; i++) out[i] = re[i];
}

// ── weight loading ─────────────────────────────────────────────────────────
static bool fetchw(const dvqe_graph_model& m, const char* name, std::vector<float>& v) {
    auto* t = m.w(name);
    if (!t) { fprintf(stderr, "daf: missing tensor %s\n", name); return false; }
    v.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return true;
}

static bool fetchw_npy(const localvqe_model& m, const char* name,
                       std::vector<float>& v) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) { fprintf(stderr, "daf: missing %s\n", name); return false; }
    v = it->second.data;
    return true;
}

bool daf_init_npy(daf_frontend& fe, const localvqe_model& m) {
    fe.loaded = false;
    if (m.tensors.find("daf.glob.gru.weight_ih") == m.tensors.end()) return false;
    bool ok = true;
    ok &= fetchw_npy(m, "daf.glob.norm.ln.weight", fe.g_ln_w);
    ok &= fetchw_npy(m, "daf.glob.norm.ln.bias",   fe.g_ln_b);
    ok &= fetchw_npy(m, "daf.glob.gru.weight_ih",  fe.g_wih);
    ok &= fetchw_npy(m, "daf.glob.gru.weight_hh",  fe.g_whh);
    ok &= fetchw_npy(m, "daf.glob.gru.bias_ih",    fe.g_bih);
    ok &= fetchw_npy(m, "daf.glob.gru.bias_hh",    fe.g_bhh);
    ok &= fetchw_npy(m, "daf.bins.norm.ln.weight", fe.b_ln_w);
    ok &= fetchw_npy(m, "daf.bins.norm.ln.bias",   fe.b_ln_b);
    ok &= fetchw_npy(m, "daf.bins.gru.weight_ih",  fe.b_wih);
    ok &= fetchw_npy(m, "daf.bins.gru.weight_hh",  fe.b_whh);
    ok &= fetchw_npy(m, "daf.bins.gru.bias_ih",    fe.b_bih);
    ok &= fetchw_npy(m, "daf.bins.gru.bias_hh",    fe.b_bhh);
    ok &= fetchw_npy(m, "daf.part.ln.weight",      fe.p_ln_w);
    ok &= fetchw_npy(m, "daf.part.ln.bias",        fe.p_ln_b);
    ok &= fetchw_npy(m, "daf.part.gru.weight_ih",  fe.p_wih);
    ok &= fetchw_npy(m, "daf.part.gru.weight_hh",  fe.p_whh);
    ok &= fetchw_npy(m, "daf.part.gru.bias_ih",    fe.p_bih);
    ok &= fetchw_npy(m, "daf.part.gru.bias_hh",    fe.p_bhh);
    ok &= fetchw_npy(m, "daf.head.weight",         fe.head_w);
    ok &= fetchw_npy(m, "daf.head.bias",           fe.head_b);
    ok &= fetchw_npy(m, "daf.part.head.weight",    fe.p_head_w);
    ok &= fetchw_npy(m, "daf.part.head.bias",      fe.p_head_b);
    if (!ok) return false;
    daf_reset(fe);
    fe.loaded = true;
    return true;
}

bool daf_init(daf_frontend& fe, const dvqe_graph_model& m) {
    fe.loaded = false;
    if (!m.w("daf.glob.gru.weight_ih")) return false;   // no front-end embedded
    bool ok = true;
    ok &= fetchw(m, "daf.glob.norm.ln.weight", fe.g_ln_w);
    ok &= fetchw(m, "daf.glob.norm.ln.bias",   fe.g_ln_b);
    ok &= fetchw(m, "daf.glob.gru.weight_ih",  fe.g_wih);
    ok &= fetchw(m, "daf.glob.gru.weight_hh",  fe.g_whh);
    ok &= fetchw(m, "daf.glob.gru.bias_ih",    fe.g_bih);
    ok &= fetchw(m, "daf.glob.gru.bias_hh",    fe.g_bhh);
    ok &= fetchw(m, "daf.bins.norm.ln.weight", fe.b_ln_w);
    ok &= fetchw(m, "daf.bins.norm.ln.bias",   fe.b_ln_b);
    ok &= fetchw(m, "daf.bins.gru.weight_ih",  fe.b_wih);
    ok &= fetchw(m, "daf.bins.gru.weight_hh",  fe.b_whh);
    ok &= fetchw(m, "daf.bins.gru.bias_ih",    fe.b_bih);
    ok &= fetchw(m, "daf.bins.gru.bias_hh",    fe.b_bhh);
    ok &= fetchw(m, "daf.part.ln.weight",      fe.p_ln_w);
    ok &= fetchw(m, "daf.part.ln.bias",        fe.p_ln_b);
    ok &= fetchw(m, "daf.part.gru.weight_ih",  fe.p_wih);
    ok &= fetchw(m, "daf.part.gru.weight_hh",  fe.p_whh);
    ok &= fetchw(m, "daf.part.gru.bias_ih",    fe.p_bih);
    ok &= fetchw(m, "daf.part.gru.bias_hh",    fe.p_bhh);
    ok &= fetchw(m, "daf.head.weight",         fe.head_w);
    ok &= fetchw(m, "daf.head.bias",           fe.head_b);
    ok &= fetchw(m, "daf.part.head.weight",    fe.p_head_w);
    ok &= fetchw(m, "daf.part.head.bias",      fe.p_head_b);
    if (!ok) return false;
    daf_reset(fe);
    fe.loaded = true;
    return true;
}

void daf_reset(daf_frontend& fe) {
    const int N = daf_frontend::N, NB = daf_frontend::NB, M = daf_frontend::M;
    fe.Hr.assign((size_t)N*NB + 16, 0); fe.Hi.assign((size_t)N*NB + 16, 0);
    fe.Xr_r.assign((size_t)N*NB + 16, 0); fe.Xr_i.assign((size_t)N*NB + 16, 0);
    fe.P.assign((size_t)N*NB + 16, daf_frontend::P_INIT);
    fe.x_old.assign(M, 0);
    fe.p_idx = 0;
    fe.hg.assign(8, 0);
    fe.hb.assign((size_t)NB*16, 0);
    fe.hp.assign((size_t)N*8, 0);
    fe.gcc_Sr.assign(daf_frontend::G_NFFT/2 + 1, 0);
    fe.gcc_Si.assign(daf_frontend::G_NFFT/2 + 1, 0);
    fe.mic_ring.assign(daf_frontend::G_WIN, 0);
    fe.ref_ring.assign(daf_frontend::G_WIN, 0);
    fe.n_seen = 0;
    fe.gcc_maxrms = 1e-12f;
    fe.gcc_conf = 0.0f;
    fe.cur_shift = 0;
    fe.gcc_locked = false;
    fe.ref_dline.assign(daf_frontend::G_MAXLAG + daf_frontend::M, 0);
    fe.ref_dpos = 0;
}

// torch GRUCell: gates [r,z,n]; h' = (1-z)*n + z*h
static void gru_cell(const float* x, int nin, float* h, int nh,
                     const float* wih, const float* whh,
                     const float* bih, const float* bhh,
                     float* gi /*3nh*/, float* gh /*3nh*/) {
    for (int g = 0; g < 3 * nh; g++) {
        float s = bih[g];
        const float* w = wih + (size_t)g * nin;
        for (int i = 0; i < nin; i++) s += w[i] * x[i];
        gi[g] = s;
        float t = bhh[g];
        const float* v = whh + (size_t)g * nh;
        for (int i = 0; i < nh; i++) t += v[i] * h[i];
        gh[g] = t;
    }
    for (int i = 0; i < nh; i++) {
        const float r = 1.0f / (1.0f + expf(-(gi[i] + gh[i])));
        const float z = 1.0f / (1.0f + expf(-(gi[nh+i] + gh[nh+i])));
        const float n = tanhf(gi[2*nh+i] + r * gh[2*nh+i]);
        h[i] = (1.0f - z) * n + z * h[i];
    }
}

// LayerNorm over the first 6 features (cos features pass through raw)
static void cos_ln(const float* in, float* out, const float* w, const float* b, int k) {
    float mean = 0;
    for (int i = 0; i < 6; i++) mean += in[i];
    mean /= 6.0f;
    float var = 0;
    for (int i = 0; i < 6; i++) { const float d = in[i] - mean; var += d * d; }
    var /= 6.0f;
    const float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < 6; i++) out[i] = (in[i] - mean) * inv * w[i] + b[i];
    for (int i = 6; i < k; i++) out[i] = in[i];
}

static inline float log10e(float x) { return log10f(x + 1e-10f); }

// ── GCC-PHAT (gated growing sum); updates fe.cur_shift every hop ───────────
static void gcc_update(daf_frontend& fe) {
    if (fe.gcc_locked) return;
    const int W = daf_frontend::G_WIN, NF = daf_frontend::G_NFFT;
    static thread_local std::vector<float> Xr(NF/2+1), Xi(NF/2+1), Dr(NF/2+1), Di(NF/2+1);
    static thread_local std::vector<float> pad(NF), c(NF);
    float rms = 0;
    for (int i = 0; i < W; i++) rms += fe.ref_ring[i] * fe.ref_ring[i];
    rms = sqrtf(rms / W);
    fe.gcc_maxrms = std::max(fe.gcc_maxrms, rms);
    const bool keep = rms > fe.gcc_maxrms * powf(10.0f, -daf_frontend::G_GATE_DB / 20.0f);
    if (keep) {
        std::memcpy(pad.data(), fe.ref_ring.data(), W * sizeof(float));
        std::memset(pad.data() + W, 0, (NF - W) * sizeof(float));
        rfft(pad.data(), NF, Xr.data(), Xi.data());
        std::memcpy(pad.data(), fe.mic_ring.data(), W * sizeof(float));
        std::memset(pad.data() + W, 0, (NF - W) * sizeof(float));
        rfft(pad.data(), NF, Dr.data(), Di.data());
        for (int k = 0; k <= NF/2; k++) {
            const float sr = Dr[k]*Xr[k] + Di[k]*Xi[k];     // D * conj(X)
            const float si = Di[k]*Xr[k] - Dr[k]*Xi[k];
            const float mag = sqrtf(sr*sr + si*si) + 1e-9f;
            fe.gcc_Sr[k] += sr / mag;
            fe.gcc_Si[k] += si / mag;
        }
    }
    irfft(fe.gcc_Sr.data(), fe.gcc_Si.data(), NF, c.data());
    int best = 0; float peak = c[0], asum = 0;
    for (int l = 0; l < daf_frontend::G_MAXLAG; l++) {
        if (c[l] > peak) { peak = c[l]; best = l; }
        asum += fabsf(c[l]);
    }
    const float conf = peak / (asum / daf_frontend::G_MAXLAG + 1e-12f);
    fe.gcc_conf = conf;
    fe.cur_shift = (conf > daf_frontend::G_CONF_THR)
        ? std::max(0, best - daf_frontend::M) / daf_frontend::M * daf_frontend::M
        : 0;
    if (conf > daf_frontend::G_CONF_THR && fe.n_seen >= daf_frontend::LOCK_AFTER)
        fe.gcc_locked = true;
}

// ── one M-sample filter block ──────────────────────────────────────────────
static void daf_block(daf_frontend& fe, const float* d_cur, const float* x_cur,
                      float* e_out) {
    const int M = daf_frontend::M, N = daf_frontend::N, NB = daf_frontend::NB,
              NF = daf_frontend::NFFT;
    static thread_local std::vector<float> xn(NF), Xr_(NB), Xi_(NB),
        Yr(NB), Yi(NB), buf(NF), Er(NB), Ei(NB), Dr(NB), Di(NB), X2(NB),
        fb(10), fbn(18), fg(10), gi(48), gh(48), gbin(NB), spart(N),
        mu((size_t)N*NB), meanP(NB), fbins((size_t)NB*10);

    // analysis
    std::memcpy(xn.data(), fe.x_old.data(), M * sizeof(float));
    std::memcpy(xn.data() + M, x_cur, M * sizeof(float));
    std::memcpy(fe.x_old.data(), x_cur, M * sizeof(float));
    rfft(xn.data(), NF, Xr_.data(), Xi_.data());

    // push into ring: newest at partition 0 (shift down)
    std::memmove(fe.Xr_r.data() + NB, fe.Xr_r.data(), (size_t)(N-1)*NB*sizeof(float));
    std::memmove(fe.Xr_i.data() + NB, fe.Xr_i.data(), (size_t)(N-1)*NB*sizeof(float));
    std::memcpy(fe.Xr_r.data(), Xr_.data(), NB * sizeof(float));
    std::memcpy(fe.Xr_i.data(), Xi_.data(), NB * sizeof(float));

    // predict echo — k-tiled: accumulators stay in registers across the
    // 128-partition reduction (was: Yr/Yi round-tripped L1 every p).
    {
        float* __restrict yr = Yr.data(); float* __restrict yi = Yi.data();
        for (int k0 = 0; k0 < NB; k0 += 16) {
            const int kn = std::min(16, NB - k0);
            float ar[16] = {0}, ai[16] = {0};
            for (int p = 0; p < N; p++) {
                const float* __restrict hr = &fe.Hr[(size_t)p*NB] + k0;
                const float* __restrict hi = &fe.Hi[(size_t)p*NB] + k0;
                const float* __restrict xr = &fe.Xr_r[(size_t)p*NB] + k0;
                const float* __restrict xi = &fe.Xr_i[(size_t)p*NB] + k0;
                for (int k = 0; k < 16; k++) {
                    ar[k] += hr[k]*xr[k] - hi[k]*xi[k];
                    ai[k] += hr[k]*xi[k] + hi[k]*xr[k];
                }
            }
            for (int k = 0; k < kn; k++) { yr[k0+k] = ar[k]; yi[k0+k] = ai[k]; }
        }
    }
    irfft(Yr.data(), Yi.data(), NF, buf.data());
    for (int i = 0; i < M; i++) e_out[i] = d_cur[i] - buf[M + i];

    // spectra of e and d (zero-padded first half)
    std::memset(xn.data(), 0, M * sizeof(float));
    std::memcpy(xn.data() + M, e_out, M * sizeof(float));
    rfft(xn.data(), NF, Er.data(), Ei.data());
    std::memcpy(xn.data() + M, d_cur, M * sizeof(float));
    rfft(xn.data(), NF, Dr.data(), Di.data());

    // X2 + meanP — k-tiled like the echo prediction.
    {
        float* __restrict x2 = X2.data(); float* __restrict mp_ = meanP.data();
        for (int k0 = 0; k0 < NB; k0 += 16) {
            const int kn = std::min(16, NB - k0);
            float a2[16] = {0}, am[16] = {0};
            for (int p = 0; p < N; p++) {
                const float* __restrict xr = &fe.Xr_r[(size_t)p*NB] + k0;
                const float* __restrict xi = &fe.Xr_i[(size_t)p*NB] + k0;
                const float* __restrict pp = &fe.P[(size_t)p*NB] + k0;
                for (int k = 0; k < 16; k++) {
                    a2[k] += xr[k]*xr[k] + xi[k]*xi[k];
                    am[k] += pp[k];
                }
            }
            for (int k = 0; k < kn; k++) { x2[k0+k] = a2[k]; mp_[k0+k] = am[k]; }
        }
    }
    for (int k = 0; k < NB; k++) meanP[k] /= (float)N;

    // ── controller features ──
    const float ce = 1e-9f;
    for (int k = 0; k < NB; k++) {
        const float Xa = sqrtf(Xr_[k]*Xr_[k] + Xi_[k]*Xi_[k]);
        const float Ea = sqrtf(Er[k]*Er[k] + Ei[k]*Ei[k]);
        const float Ya = sqrtf(Yr[k]*Yr[k] + Yi[k]*Yi[k]);
        const float Da = sqrtf(Dr[k]*Dr[k] + Di[k]*Di[k]);
        float* f = &fbins[(size_t)k*10];
        f[0] = log10e(Xa*Xa); f[1] = log10e(X2[k]); f[2] = log10e(Da*Da);
        f[3] = log10e(Ea*Ea); f[4] = log10e(Ya*Ya); f[5] = log10e(meanP[k]);
        f[6] = (Er[k]*Dr[k] + Ei[k]*Di[k]) / (Ea*Da + ce);
        f[7] = (Dr[k]*Yr[k] + Di[k]*Yi[k]) / (Da*Ya + ce);
        f[8] = (Er[k]*Yr[k] + Ei[k]*Yi[k]) / (Ea*Ya + ce);
        f[9] = (Xr_[k]*Dr[k] + Xi_[k]*Di[k]) / (Xa*Da + ce);
    }
    // global branch: fg = mean over bins, LN6+raw -> GRU(10->8)
    std::fill(fg.begin(), fg.end(), 0.0f);
    for (int k = 0; k < NB; k++)
        for (int j = 0; j < 10; j++) fg[j] += fbins[(size_t)k*10 + j];
    for (int j = 0; j < 10; j++) fg[j] /= (float)NB;
    cos_ln(fg.data(), fb.data(), fe.g_ln_w.data(), fe.g_ln_b.data(), 10);
    gru_cell(fb.data(), 10, fe.hg.data(), 8,
             fe.g_wih.data(), fe.g_whh.data(), fe.g_bih.data(), fe.g_bhh.data(),
             gi.data(), gh.data());
    // per-bin backbone: [LN6(f)|cos|ctx8] -> GRU(18->16) -> g = 2σ(head)
    for (int k = 0; k < NB; k++) {
        cos_ln(&fbins[(size_t)k*10], fbn.data(), fe.b_ln_w.data(), fe.b_ln_b.data(), 10);
        std::memcpy(fbn.data() + 10, fe.hg.data(), 8 * sizeof(float));
        gru_cell(fbn.data(), 18, &fe.hb[(size_t)k*16], 16,
                 fe.b_wih.data(), fe.b_whh.data(), fe.b_bih.data(), fe.b_bhh.data(),
                 gi.data(), gh.data());
        float z = fe.head_b[0];
        for (int i = 0; i < 16; i++) z += fe.head_w[i] * fe.hb[(size_t)k*16 + i];
        gbin[k] = 2.0f / (1.0f + expf(-z));
    }
    // per-partition branch: [LN2(logH2,logP)|ctx8] -> GRU(10->8) -> s = 2σ
    for (int p = 0; p < N; p++) {
        float h2 = 0, pm = 0;
        const float* hr = &fe.Hr[(size_t)p*NB]; const float* hi = &fe.Hi[(size_t)p*NB];
        const float* pp = &fe.P[(size_t)p*NB];
        for (int k = 0; k < NB; k++) { h2 += hr[k]*hr[k] + hi[k]*hi[k]; pm += pp[k]; }
        float f2[2] = { log10e(h2 / NB), log10e(pm / NB) };
        float mean = 0.5f*(f2[0]+f2[1]);
        float var = 0.5f*((f2[0]-mean)*(f2[0]-mean) + (f2[1]-mean)*(f2[1]-mean));
        const float inv = 1.0f / sqrtf(var + 1e-5f);
        float in10[10];
        in10[0] = (f2[0]-mean)*inv*fe.p_ln_w[0] + fe.p_ln_b[0];
        in10[1] = (f2[1]-mean)*inv*fe.p_ln_w[1] + fe.p_ln_b[1];
        std::memcpy(in10 + 2, fe.hg.data(), 8 * sizeof(float));
        gru_cell(in10, 10, &fe.hp[(size_t)p*8], 8,
                 fe.p_wih.data(), fe.p_whh.data(), fe.p_bih.data(), fe.p_bhh.data(),
                 gi.data(), gh.data());
        float z = fe.p_head_b[0];
        for (int i = 0; i < 8; i++) z += fe.p_head_w[i] * fe.hp[(size_t)p*8 + i];
        spart[p] = 2.0f / (1.0f + expf(-z));
    }

    // ── Kalman update ──
    const float A2 = daf_frontend::A_DECAY * daf_frontend::A_DECAY;
    for (int p = 0; p < N; p++) {
        float* __restrict pp = &fe.P[(size_t)p*NB];
        const float* __restrict hr = &fe.Hr[(size_t)p*NB];
        const float* __restrict hi = &fe.Hi[(size_t)p*NB];
        float* __restrict mp = &mu[(size_t)p*NB];
        const float* __restrict er_ = Er.data(); const float* __restrict ei_ = Ei.data();
        const float* __restrict x2_ = X2.data(); const float* __restrict gb_ = gbin.data();
        for (int k = 0; k < NB; k++) {
            const float Rt = (er_[k]*er_[k] + ei_[k]*ei_[k]) / (float)N;
            const float Pe = 0.5f * pp[k] * x2_[k] + Rt;
            float m_ = pp[k] / (Pe + 1e-10f);
            m_ *= gb_[k] * spart[p];
            const float s = m_ * x2_[k];
            if (s > 2.0f) m_ *= 2.0f / s;
            const float fac = std::max(1.0f - 0.5f * m_ * x2_[k], 0.0f);
            pp[k] = std::max(A2 * fac * pp[k] + (1.0f - A2) * (hr[k]*hr[k] + hi[k]*hi[k]),
                             1e-12f);
            mp[k] = m_;
        }
    }
    // H += E * mu * conj(Xr)   (+ K_ITER-1 data-reuse repeats with refreshed E)
    for (int it = 0; it < daf_frontend::K_ITER; it++) {
        const float* er = Er.data(); const float* ei = Ei.data();
        static thread_local std::vector<float> E2r(NB), E2i(NB);
        if (it > 0) {
            std::fill(Yr.begin(), Yr.end(), 0.0f); std::fill(Yi.begin(), Yi.end(), 0.0f);
            for (int p = 0; p < N; p++) {
                const float* hr = &fe.Hr[(size_t)p*NB]; const float* hi = &fe.Hi[(size_t)p*NB];
                const float* xr = &fe.Xr_r[(size_t)p*NB]; const float* xi = &fe.Xr_i[(size_t)p*NB];
                float* __restrict yr = Yr.data(); float* __restrict yi = Yi.data();
                for (int k = 0; k < NB; k++) {
                    yr[k] += hr[k]*xr[k] - hi[k]*xi[k];
                    yi[k] += hr[k]*xi[k] + hi[k]*xr[k];
                }
            }
            irfft(Yr.data(), Yi.data(), NF, buf.data());
            std::memset(xn.data(), 0, M * sizeof(float));
            for (int i = 0; i < M; i++) xn[M + i] = d_cur[i] - buf[M + i];
            rfft(xn.data(), NF, E2r.data(), E2i.data());
            er = E2r.data(); ei = E2i.data();
        }
        for (int p = 0; p < N; p++) {
            float* __restrict hr = &fe.Hr[(size_t)p*NB];
            float* __restrict hi = &fe.Hi[(size_t)p*NB];
            const float* __restrict xr = &fe.Xr_r[(size_t)p*NB];
            const float* __restrict xi = &fe.Xr_i[(size_t)p*NB];
            const float* __restrict mp = &mu[(size_t)p*NB];
            const float* __restrict err = er; const float* __restrict eii = ei;
            for (int k = 0; k < NB; k++) {
                // E * mu * conj(X)
                hr[k] += mp[k] * (err[k]*xr[k] + eii[k]*xi[k]);
                hi[k] += mp[k] * (eii[k]*xr[k] - err[k]*xi[k]);
            }
        }
    }
    // round-robin overlap-save constraint
    {
        float* hr = &fe.Hr[(size_t)fe.p_idx*NB]; float* hi = &fe.Hi[(size_t)fe.p_idx*NB];
        irfft(hr, hi, NF, buf.data());
        std::memset(buf.data() + M, 0, M * sizeof(float));
        rfft(buf.data(), NF, hr, hi);
        fe.p_idx = (fe.p_idx + 1) % N;
    }
}

void daf_process(daf_frontend& fe, const float* mic, const float* ref,
                 int hop, float* e_out, float* yhat_out) {
    const int M = daf_frontend::M, W = daf_frontend::G_WIN;
    const int DL = (int)fe.ref_dline.size();
    for (int o = 0; o < hop; o += M) {
        // GCC rings + hop cadence
        std::memmove(fe.mic_ring.data(), fe.mic_ring.data() + M, (W - M) * sizeof(float));
        std::memcpy(fe.mic_ring.data() + W - M, mic + o, M * sizeof(float));
        std::memmove(fe.ref_ring.data(), fe.ref_ring.data() + M, (W - M) * sizeof(float));
        std::memcpy(fe.ref_ring.data() + W - M, ref + o, M * sizeof(float));
        // ref delay line (linear ring)
        for (int i = 0; i < M; i++)
            fe.ref_dline[(fe.ref_dpos + i) % DL] = ref[o + i];
        fe.n_seen += M;
        if (fe.enable_prealign && fe.n_seen >= W &&
            ((fe.n_seen - W) % daf_frontend::G_HOP) < M)
            gcc_update(fe);
        // shifted ref block
        float xs[daf_frontend::M];
        for (int i = 0; i < M; i++) {
            const long long idx = (long long)fe.ref_dpos + i - fe.cur_shift;
            xs[i] = (fe.n_seen - M + i >= fe.cur_shift)
                ? fe.ref_dline[((idx % DL) + DL) % DL] : 0.0f;
        }
        fe.ref_dpos = (fe.ref_dpos + M) % DL;
        daf_block(fe, mic + o, xs, e_out + o);
    }
    if (yhat_out)
        for (int i = 0; i < hop; i++) yhat_out[i] = mic[i] - e_out[i];
}

void daf_prime_delay(daf_frontend& fe, const float* mic, const float* ref, int n) {
    // Estimate the bulk echo delay once over the whole signal (same gated
    // growing-sum GCC-PHAT the online path uses), keep the highest-confidence
    // shift, then reset and freeze it so the filter is aligned from frame 0.
    // This removes the ~1-3 s online acquisition window — the dominant cause
    // of the standalone front-end's onset residual ("reverb"), which the
    // cascade's mask otherwise hides. The filter is never run here.
    daf_reset(fe);
    const int M = daf_frontend::M, W = daf_frontend::G_WIN;
    int best_shift = 0;
    float best_conf = 0.0f;
    for (int o = 0; o + M <= n; o += M) {
        std::memmove(fe.mic_ring.data(), fe.mic_ring.data() + M, (W - M) * sizeof(float));
        std::memcpy(fe.mic_ring.data() + W - M, mic + o, M * sizeof(float));
        std::memmove(fe.ref_ring.data(), fe.ref_ring.data() + M, (W - M) * sizeof(float));
        std::memcpy(fe.ref_ring.data() + W - M, ref + o, M * sizeof(float));
        fe.n_seen += M;
        if (fe.n_seen >= W && ((fe.n_seen - W) % daf_frontend::G_HOP) < M) {
            gcc_update(fe);
            fe.gcc_locked = false;  // keep refining over the entire signal
            if (fe.gcc_conf > best_conf) {
                best_conf = fe.gcc_conf;
                best_shift = fe.cur_shift;
            }
        }
    }
    daf_reset(fe);
    if (best_conf > daf_frontend::G_CONF_THR) {
        fe.cur_shift = best_shift;
        fe.gcc_locked = true;       // freeze: no online re-estimation
    }
}
