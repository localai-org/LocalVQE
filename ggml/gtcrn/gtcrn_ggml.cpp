/**
 * GTCRN-AEC as a real GGML compute graph.
 *
 * The entire forward is built from ggml ops and dispatched through a ggml
 * backend (CPU SIMD variants). Feed-forward layers (convs, ERB matmuls, SFE,
 * FC, LayerNorm, mask) are single batched ggml ops over all T frames; the GRU
 * recurrences are unrolled in the graph builder (intra over the 33 freq steps,
 * inter and TRA over the T time steps), each step reusing the prior step's
 * tensor — so every operation is a genuine ggml op.
 *
 * Layout convention: ggml [F, T, C, 1] (ne0=F fastest), which is byte-identical
 * to the PyTorch (1,C,T,F) row-major fixtures. Weights are uploaded with ne =
 * reversed numpy shape (so PyTorch (OC,IC,kT,kF) -> ggml [kF,kT,IC,OC], exactly
 * what ggml_conv_2d wants). GRU weight (3H,I) -> ggml [I,3H] for mul_mat.
 *
 * Parity to the scalar reference / PyTorch fixtures is gated by test_gtcrn
 * --ggml; F32 throughout (no quantization).
 */

#include "gtcrn.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using T = ggml_tensor;

// ── load: backend + weight tensors ──────────────────────────────────────────

bool GtcrnGraph::load(const char* path, int n_threads, bool verbose) {
    GtcrnModel host;
    if (!host.load(path, false)) return false;

    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (!reg) { fprintf(stderr, "gtcrn: CPU backend not registered\n"); return false; }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "gtcrn: backend init failed\n"); return false; }
    auto setth = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (setth) setth(backend, n_threads);

    struct ggml_init_params wp;
    wp.mem_size = host.W.size() * ggml_tensor_overhead() + (1u << 20);
    wp.mem_buffer = nullptr;
    wp.no_alloc = true;
    wctx = ggml_init(wp);

    for (auto& kv : host.W) {
        const NpyArray& a = kv.second;
        int nd = a.ndim() ? a.ndim() : 1;
        int64_t ne[4] = {1, 1, 1, 1};
        for (int i = 0; i < nd; ++i) ne[i] = a.shape[nd - 1 - i];  // reverse to ggml order
        T* t = ggml_new_tensor(wctx, GGML_TYPE_F32, nd, ne);
        ggml_set_name(t, kv.first.c_str());
        wt[kv.first] = t;
    }
    wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) { fprintf(stderr, "gtcrn: weight buffer alloc failed\n"); return false; }
    for (auto& kv : host.W) {
        T* t = wt[kv.first];
        ggml_backend_tensor_set(t, kv.second.data.data(), 0, ggml_nbytes(t));
    }
    if (verbose) printf("gtcrn(ggml): %zu weight tensors, backend=%s\n",
                        wt.size(), ggml_backend_name(backend));
    return true;
}

GtcrnGraph::~GtcrnGraph() {
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

// ── graph builder ────────────────────────────────────────────────────────────

namespace {

struct GB {
    ggml_context* ctx;
    const std::map<std::string, T*>* wt;
    T* zeros;  // 1D zero buffer for GRU initial hidden states

    T* W(const std::string& n) const {
        auto it = wt->find(n);
        if (it == wt->end()) { fprintf(stderr, "gtcrn: missing weight %s\n", n.c_str()); abort(); }
        return it->second;
    }
    T* cont(T* x) { return ggml_cont(ctx, x); }
    T* perm(T* x, int a0, int a1, int a2, int a3) { return cont(ggml_permute(ctx, x, a0, a1, a2, a3)); }

    T* addbias(T* y, T* b) {  // b:[OC] -> broadcast over F,T
        return ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, y->ne[2], 1));
    }
    T* prelu(T* x, T* slope) {  // slope:[1]
        T* s = ggml_reshape_4d(ctx, slope, 1, 1, 1, 1);
        T* r = ggml_relu(ctx, x);
        T* nr = ggml_relu(ctx, ggml_neg(ctx, x));
        return ggml_sub(ctx, r, ggml_mul(ctx, nr, s));
    }

    // general conv. p1top = causal left-pad in time (then conv time-pad 0).
    T* conv(T* x, T* w, int groups, int s0, int p0, int p1top, int d0, int d1, bool dw) {
        if (p1top > 0) x = ggml_pad_ext(ctx, x, 0, 0, p1top, 0, 0, 0, 0, 0);
        // F32-native depthwise (ggml_conv_2d_dw forces F16 im2col); kernel [KW,KH,1,C].
        if (dw) return ggml_conv_2d_dw_direct(ctx, w, x, s0, 1, p0, 0, d0, d1);
        if (groups == 1) return ggml_conv_2d(ctx, w, x, s0, 1, p0, 0, d0, d1);
        int OC = (int)w->ne[3], ipg = (int)w->ne[2], opg = OC / groups;
        T* out = nullptr;
        for (int g = 0; g < groups; ++g) {
            T* wg = cont(ggml_view_4d(ctx, w, w->ne[0], w->ne[1], ipg, opg,
                                      w->nb[1], w->nb[2], w->nb[3], (size_t)g * opg * w->nb[3]));
            T* xg = cont(ggml_view_4d(ctx, x, x->ne[0], x->ne[1], ipg, 1,
                                      x->nb[1], x->nb[2], x->nb[3], (size_t)g * ipg * x->nb[2]));
            T* yg = ggml_conv_2d(ctx, wg, xg, s0, 1, p0, 0, d0, d1);
            out = out ? ggml_concat(ctx, out, yg, 2) : yg;
        }
        return out;
    }

    // SFE 3-tap freq unfold -> 3C channels, order c*3+k (k: f-1,f,f+1, zero pad).
    T* sfe(T* x) {
        int F = (int)x->ne[0], Tt = (int)x->ne[1], C = (int)x->ne[2];
        T* xpadL = ggml_pad_ext(ctx, x, 1, 0, 0, 0, 0, 0, 0, 0);          // [F+1,T,C]
        T* sm = cont(ggml_view_4d(ctx, xpadL, F, Tt, C, 1, xpadL->nb[1], xpadL->nb[2], xpadL->nb[3], 0));
        T* xpadR = ggml_pad_ext(ctx, x, 0, 1, 0, 0, 0, 0, 0, 0);          // [F+1,T,C]
        T* sp = cont(ggml_view_4d(ctx, xpadR, F, Tt, C, 1, xpadR->nb[1], xpadR->nb[2], xpadR->nb[3],
                                  (size_t)1 * sizeof(float)));            // [1:F+1]: out[f]=in[f+1]
        // stack [sm,x,sp] as [F,T,1,C] each -> concat ne2 -> [F,T,3,C] -> [F,T,3C]
        T* a = ggml_reshape_4d(ctx, sm, F, Tt, 1, C);
        T* b = ggml_reshape_4d(ctx, x, F, Tt, 1, C);
        T* c = ggml_reshape_4d(ctx, sp, F, Tt, 1, C);
        T* s = ggml_concat(ctx, ggml_concat(ctx, a, b, 2), c, 2);  // [F,T,3,C]
        return ggml_reshape_4d(ctx, cont(s), F, Tt, 3 * C, 1);     // channel = k + 3c
    }

    // band op: low `low` bins pass; high (in->out) via matmul Wmat [in,out].
    T* band(T* x, T* Wmat, int low) {
        int F = (int)x->ne[0], Tt = (int)x->ne[1], C = (int)x->ne[2];
        int in = (int)Wmat->ne[0];
        T* lo = cont(ggml_view_4d(ctx, x, low, Tt, C, 1, x->nb[1], x->nb[2], x->nb[3], 0));
        T* hi = cont(ggml_view_4d(ctx, x, in, Tt, C, 1, x->nb[1], x->nb[2], x->nb[3],
                                  (size_t)low * sizeof(float)));
        T* ho = ggml_mul_mat(ctx, Wmat, hi);   // [out,T,C]
        (void)F;
        return ggml_concat(ctx, lo, ho, 0);
    }

    // Whole-sequence GRU via the fused ggml_gru op: precompute the input
    // projection W_ih·x+b_ih for all L steps in ONE matmul, then the op runs the
    // entire recurrence (W_hh·h + gates, per step) inside a single graph node —
    // collapsing L×~13 tiny ops to 1. seq:[I,N,L] -> [H,N,L].
    T* gru_unroll(T* seq, const std::string& p, bool reverse) {
        std::string sfx = reverse ? "_reverse" : "";
        T* wih = W(p + ".weight_ih_l0" + sfx);
        T* whh = W(p + ".weight_hh_l0" + sfx);
        T* bih = W(p + ".bias_ih_l0" + sfx);
        T* bhh = W(p + ".bias_hh_l0" + sfx);
        int I = (int)seq->ne[0], N = (int)seq->ne[1], L = (int)seq->ne[2];
        int H = (int)whh->ne[0];
        T* gi = ggml_add(ctx, ggml_mul_mat(ctx, wih, ggml_reshape_2d(ctx, seq, I, N * L)),
                         ggml_reshape_2d(ctx, bih, 3 * H, 1));        // [3H, N*L]
        gi = ggml_reshape_3d(ctx, gi, 3 * H, N, L);
        T* h0 = cont(ggml_view_2d(ctx, zeros, H, N, (size_t)H * sizeof(float), 0));  // zeros [H,N]
        return ggml_gru(ctx, gi, whh, bhh, h0, reverse ? 1 : 0);      // [H,N,L]
    }

    // single GRU step with an explicit carried hidden (streaming inter/TRA), via
    // the same fused op (L=1). xt:[I,N], h:[H,N] -> [H,N].
    T* gru_step(T* xt, T* h, T* wih, T* whh, T* bih, T* bhh) {
        int H = (int)whh->ne[0], N = (int)xt->ne[1];
        T* gi = ggml_add(ctx, ggml_mul_mat(ctx, wih, xt), ggml_reshape_2d(ctx, bih, 3 * H, 1));
        gi = ggml_reshape_3d(ctx, gi, 3 * H, N, 1);
        T* o = ggml_gru(ctx, gi, whh, bhh, h, 0);                     // [H,N,1]
        return ggml_reshape_2d(ctx, o, H, N);
    }

    // grouped RNN: split feature, rnn1/rnn2, optional bidirectional. [I,N,L]->[16,N,L].
    T* grnn(T* x, const std::string& d, bool bidir) {
        int I = (int)x->ne[0], N = (int)x->ne[1], L = (int)x->ne[2], half = I / 2;
        auto sub = [&](int off) {
            return cont(ggml_view_3d(ctx, x, half, N, L, x->nb[1], x->nb[2], (size_t)off * sizeof(float)));
        };
        T* x1 = sub(0); T* x2 = sub(half);
        auto run = [&](T* xs, const std::string& rp) {
            T* f = gru_unroll(xs, rp, false);
            if (bidir) f = ggml_concat(ctx, f, gru_unroll(xs, rp, true), 0);
            return f;
        };
        T* o1 = run(x1, d + ".rnn1");
        T* o2 = run(x2, d + ".rnn2");
        return ggml_concat(ctx, o1, o2, 0);  // [16,N,L]
    }

    T* lin(T* x, const std::string& wn, const std::string& bn) {  // over ne0
        T* y = ggml_mul_mat(ctx, W(wn), x);
        return ggml_add(ctx, y, ggml_reshape_2d(ctx, W(bn), W(bn)->ne[0], 1));
    }

    // LayerNorm over a contiguous [528,N] tensor; affine w/b [528].
    T* layernorm(T* x528, const std::string& wn, const std::string& bn) {
        T* xn = ggml_norm(ctx, x528, 1e-8f);
        T* w = ggml_reshape_2d(ctx, W(wn), 528, 1);
        T* b = ggml_reshape_2d(ctx, W(bn), 528, 1);
        return ggml_add(ctx, ggml_mul(ctx, xn, w), b);
    }

    T* tra(T* x, const std::string& p) {
        int F = (int)x->ne[0], Tt = (int)x->ne[1], C = (int)x->ne[2];
        T* m = ggml_mean(ctx, ggml_sqr(ctx, x));        // [1,T,C,1]
        T* tc = ggml_reshape_2d(ctx, cont(m), Tt, C);   // [T,C]
        T* seq = perm(ggml_reshape_4d(ctx, tc, Tt, C, 1, 1), 2, 0, 1, 3);  // [C,1,T]
        T* g = gru_unroll(seq, p + ".tra.gru", false);  // [2C,1,T]
        T* at = lin(g, p + ".tra.fc.w", p + ".tra.fc.b");  // [C,1,T]
        at = ggml_sigmoid(ctx, at);
        T* atb = perm(at, 2, 0, 1, 3);                  // [1,T,C]
        (void)F;
        return ggml_mul(ctx, x, atb);                   // broadcast over F
    }

    T* shuffle(T* h, T* x2) {
        int F = (int)h->ne[0], Tt = (int)h->ne[1], C = (int)h->ne[2];
        T* a = ggml_reshape_4d(ctx, h, F, Tt, 1, C);
        T* b = ggml_reshape_4d(ctx, x2, F, Tt, 1, C);
        T* s = ggml_concat(ctx, a, b, 2);               // [F,T,2,C]
        // cont() the result: it's a block output that may be captured/aliased; a
        // bare reshape view's backing buffer can be reused by gallocr post-compute.
        return cont(ggml_reshape_4d(ctx, cont(s), F, Tt, 2 * C, 1));  // channel = s + 2c
    }

    T* chunk(T* x, bool second) {
        int F = (int)x->ne[0], Tt = (int)x->ne[1], C = (int)x->ne[2], h = C / 2;
        size_t off = second ? (size_t)h * x->nb[2] : 0;
        return cont(ggml_view_4d(ctx, x, F, Tt, h, 1, x->nb[1], x->nb[2], x->nb[3], off));
    }

    T* upsample(T* x, int factor) {
        int F = (int)x->ne[0], Tt = (int)x->ne[1], C = (int)x->ne[2];
        T* r = ggml_reshape_4d(ctx, x, 1, F, Tt, C);    // leading 1
        r = ggml_pad_ext(ctx, r, 0, factor - 1, 0, 0, 0, 0, 0, 0);  // [factor,F,T,C]
        r = ggml_reshape_4d(ctx, cont(r), factor * F, Tt, C, 1);    // ne0 = k + factor*f
        int Fout = (F - 1) * factor + 1;
        return cont(ggml_view_4d(ctx, r, Fout, Tt, C, 1, r->nb[1], r->nb[2], r->nb[3], 0));
    }

    T* gt_block(T* x, const std::string& p, int dil) {
        T* x1 = chunk(x, false); T* x2 = chunk(x, true);
        T* s = sfe(x1);
        T* h = conv(s, W(p + ".pc1.w"), 1, 1, 0, 0, 1, 1, false);
        h = prelu(addbias(h, W(p + ".pc1.b")), W(p + ".pc1.prelu"));
        int hidden = (int)h->ne[2];
        h = conv(h, W(p + ".dw.w"), hidden, 1, 1, 2 * dil, 1, dil, true);
        h = prelu(addbias(h, W(p + ".dw.b")), W(p + ".dw.prelu"));
        h = addbias(conv(h, W(p + ".pc2.w"), 1, 1, 0, 0, 1, 1, false), W(p + ".pc2.b"));
        h = tra(h, p);
        return shuffle(h, x2);
    }

    T* conv_block(T* x, const std::string& p, int groups, int stride_f, bool deconv, bool last) {
        T* y;
        if (deconv) { x = upsample(x, stride_f); y = conv(x, W(p + ".w"), groups, 1, 2, 0, 1, 1, false); }
        else        { y = conv(x, W(p + ".w"), groups, stride_f, 2, 0, 1, 1, false); }
        y = addbias(y, W(p + ".b"));
        return last ? ggml_tanh(ctx, y) : prelu(y, W(p + ".prelu"));
    }

    T* dpgrnn(T* x, const std::string& d) {
        int Tt = (int)x->ne[1];  // x:[F=33,T,C=16]
        T* xp = perm(x, 2, 1, 0, 3);                        // [C,T,F] = [16,T,33] (intra in)
        T* intra = grnn(xp, d + ".intra_rnn", true);        // [16,T,33]
        intra = lin(intra, d + ".intra_fc.w", d + ".intra_fc.b");
        T* in528 = ggml_reshape_2d(ctx, perm(intra, 0, 2, 1, 3), 528, Tt);  // [C,F,T]->[528,T]
        in528 = layernorm(in528, d + ".intra_ln.w", d + ".intra_ln.b");
        T* intra_ln = perm(ggml_reshape_4d(ctx, in528, 16, 33, Tt, 1), 0, 2, 1, 3);  // ->[C,T,F]
        T* intra_out = ggml_add(ctx, xp, intra_ln);          // [C,T,F]

        T* xq = perm(intra_out, 0, 2, 1, 3);                 // [C,F,T] (inter in: I=16,N=33,L=T)
        T* inter = grnn(xq, d + ".inter_rnn", false);        // [16,33,T]
        inter = lin(inter, d + ".inter_fc.w", d + ".inter_fc.b");
        T* it528 = ggml_reshape_2d(ctx, cont(inter), 528, Tt);  // [C,F,T]->[528,T]
        it528 = layernorm(it528, d + ".inter_ln.w", d + ".inter_ln.b");
        T* inter_ln = ggml_reshape_4d(ctx, it528, 16, 33, Tt, 1);   // [C,F,T]
        inter_ln = perm(inter_ln, 0, 2, 1, 3);               // [C,T,F]
        T* inter_out = ggml_add(ctx, intra_out, inter_ln);   // [C,T,F]
        return perm(inter_out, 2, 1, 0, 3);                  // [F,T,C]
    }

    T* feat(T* re, T* im) {  // re/im:[257,T,1] -> [129,T,9]
        T* mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)));
        T* f3 = ggml_concat(ctx, ggml_concat(ctx, mag, re, 2), im, 2);  // [257,T,3]
        T* banded = band(f3, W("erb.bm"), 65);                          // [129,T,3]
        return sfe(banded);                                            // [129,T,9]
    }

    // ── streaming (T=1) variants: register (in,out) state pairs into `st` ──────
    std::vector<std::pair<T*, T*>>* st = nullptr;

    T* tra_stream(T* x, const std::string& p) {  // x:[F,1,C]
        int C = (int)x->ne[2];
        T* m = ggml_mean(ctx, ggml_sqr(ctx, x));                 // [1,1,C,1]
        T* seq = ggml_reshape_2d(ctx, cont(m), C, 1);            // [C,1]
        T* hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * C, 1); ggml_set_input(hin);
        T* hout = gru_step(seq, hin, W(p + ".tra.gru.weight_ih_l0"), W(p + ".tra.gru.weight_hh_l0"),
                           W(p + ".tra.gru.bias_ih_l0"), W(p + ".tra.gru.bias_hh_l0"));  // [2C,1]
        // hout is consumed downstream (lin) AND carried; register a standalone copy
        // so gallocr can't reuse its buffer before the post-compute read-back.
        st->push_back({hin, cont(hout)});
        T* at = ggml_sigmoid(ctx, lin(hout, p + ".tra.fc.w", p + ".tra.fc.b"));  // [C,1]
        T* atb = ggml_reshape_4d(ctx, at, 1, 1, C, 1);
        return ggml_mul(ctx, x, atb);
    }

    T* gt_block_stream(T* x, const std::string& p, int dil) {
        T* x1 = chunk(x, false); T* x2 = chunk(x, true);
        T* s = sfe(x1);
        T* h = prelu(addbias(conv(s, W(p + ".pc1.w"), 1, 1, 0, 0, 1, 1, false), W(p + ".pc1.b")),
                     W(p + ".pc1.prelu"));                       // [F,1,hidden]
        int F = (int)h->ne[0], hidden = (int)h->ne[2];
        T* hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, 2 * dil, hidden); ggml_set_input(hist_in);
        T* hcat = ggml_concat(ctx, hist_in, h, 1);              // [F,2dil+1,hidden]
        T* dw = ggml_conv_2d_dw_direct(ctx, W(p + ".dw.w"), hcat, 1, 1, 1, 0, 1, dil);  // [F,1,hidden]
        dw = prelu(addbias(dw, W(p + ".dw.b")), W(p + ".dw.prelu"));
        T* hist_out = cont(ggml_view_3d(ctx, hcat, F, 2 * dil, hidden,
                                        hcat->nb[1], hcat->nb[2], (size_t)1 * hcat->nb[1]));  // drop frame 0
        st->push_back({hist_in, hist_out});
        T* hh = addbias(conv(dw, W(p + ".pc2.w"), 1, 1, 0, 0, 1, 1, false), W(p + ".pc2.b"));
        hh = tra_stream(hh, p);
        return shuffle(hh, x2);
    }

    T* dpgrnn_stream(T* x, const std::string& d) {  // x:[33,1,16]
        T* xp = perm(x, 2, 1, 0, 3);                            // [16,1,33] (intra in, resets)
        T* intra = grnn(xp, d + ".intra_rnn", true);            // [16,1,33]
        intra = lin(intra, d + ".intra_fc.w", d + ".intra_fc.b");
        T* in528 = ggml_reshape_2d(ctx, perm(intra, 0, 2, 1, 3), 528, 1);
        in528 = layernorm(in528, d + ".intra_ln.w", d + ".intra_ln.b");
        T* intra_ln = perm(ggml_reshape_4d(ctx, in528, 16, 33, 1, 1), 0, 2, 1, 3);  // [16,1,33]
        T* intra_out = ggml_add(ctx, xp, intra_ln);

        T* xq = perm(intra_out, 0, 2, 1, 3);                    // [16,33,1] (inter in, carried)
        auto sub = [&](int off) {
            return cont(ggml_view_3d(ctx, xq, 8, 33, 1, xq->nb[1], xq->nb[2], (size_t)off * sizeof(float)));
        };
        auto step = [&](T* xs, const std::string& rp) {
            T* xt = ggml_reshape_2d(ctx, xs, 8, 33);
            T* hin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 33); ggml_set_input(hin);
            T* hout = gru_step(xt, hin, W(rp + ".weight_ih_l0"), W(rp + ".weight_hh_l0"),
                               W(rp + ".bias_ih_l0"), W(rp + ".bias_hh_l0"));  // [8,33]
            T* hsav = cont(hout);  // standalone copy for the carried-state read-back
            st->push_back({hin, hsav});
            return hout;
        };
        T* o1 = step(sub(0), d + ".inter_rnn.rnn1");
        T* o2 = step(sub(8), d + ".inter_rnn.rnn2");
        T* inter = ggml_reshape_3d(ctx, ggml_concat(ctx, o1, o2, 0), 16, 33, 1);  // [C,F,T=1]
        inter = lin(inter, d + ".inter_fc.w", d + ".inter_fc.b");
        T* it528 = ggml_reshape_2d(ctx, cont(inter), 528, 1);
        it528 = layernorm(it528, d + ".inter_ln.w", d + ".inter_ln.b");
        T* inter_ln = perm(ggml_reshape_4d(ctx, it528, 16, 33, 1, 1), 0, 2, 1, 3);  // [16,1,33]
        T* inter_out = ggml_add(ctx, intra_out, inter_ln);
        return perm(inter_out, 2, 1, 0, 3);                     // [33,1,16]
    }
};

}  // namespace

std::vector<float> GtcrnGraph::forward(const float* spec_e, const float* spec_y, int Tt,
                                       std::map<std::string, NpyArray>* cap,
                                       double* compute_ms) const {
    const int n_op = 600 * Tt + 200000;
    struct ggml_init_params cp;
    cp.mem_size = (size_t)n_op * (ggml_tensor_overhead() + 32) + ggml_graph_overhead_custom(n_op, false);
    cp.mem_buffer = nullptr;
    cp.no_alloc = true;
    ggml_context* ctx = ggml_init(cp);

    GB gb{ctx, &wt, nullptr};

    // inputs (deinterleaved re/im per spec, + a zero buffer for GRU h0)
    T* re_e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, Tt, 1);
    T* im_e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, Tt, 1);
    T* re_y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, Tt, 1);
    T* im_y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, Tt, 1);
    int zlen = 16 * (Tt > 33 ? Tt : 33);
    T* zeros = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, zlen);
    for (T* t : {re_e, im_e, re_y, im_y, zeros}) ggml_set_input(t);
    gb.zeros = zeros;

    // build
    T* fe = gb.feat(re_e, im_e);
    T* fy = gb.feat(re_y, im_y);
    T* ft = ggml_concat(ctx, fe, fy, 2);                  // [129,T,18]

    T* en0 = gb.conv_block(ft, "encoder.en_convs.0", 1, 2, false, false);
    T* en1 = gb.conv_block(en0, "encoder.en_convs.1", 2, 2, false, false);
    T* en2 = gb.gt_block(en1, "encoder.en_convs.2", 1);
    T* en3 = gb.gt_block(en2, "encoder.en_convs.3", 2);
    T* en4 = gb.gt_block(en3, "encoder.en_convs.4", 5);

    T* d1 = gb.dpgrnn(en4, "dpgrnn1");
    T* d2 = gb.dpgrnn(d1, "dpgrnn2");

    T* x = gb.gt_block(ggml_add(ctx, d2, en4), "decoder.de_convs.0", 5);
    T* dc0 = x;
    x = gb.gt_block(ggml_add(ctx, x, en3), "decoder.de_convs.1", 2); T* dc1 = x;
    x = gb.gt_block(ggml_add(ctx, x, en2), "decoder.de_convs.2", 1); T* dc2 = x;
    x = gb.conv_block(ggml_add(ctx, x, en1), "decoder.de_convs.3", 2, 2, true, false); T* dc3 = x;
    x = gb.conv_block(ggml_add(ctx, x, en0), "decoder.de_convs.4", 1, 2, true, true);  T* dc4 = x;

    T* mask = gb.band(x, gb.W("erb.bs"), 65);             // [257,T,2]

    T* mr = gb.cont(ggml_view_4d(ctx, mask, 257, Tt, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3], 0));
    T* mi = gb.cont(ggml_view_4d(ctx, mask, 257, Tt, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3],
                                 (size_t)1 * mask->nb[2]));
    T* outr = ggml_sub(ctx, ggml_mul(ctx, re_e, mr), ggml_mul(ctx, im_e, mi));
    T* outi = ggml_add(ctx, ggml_mul(ctx, im_e, mr), ggml_mul(ctx, re_e, mi));
    // interleave to [2,T,257] (== (257,T,2) bytes)
    T* outr_p = gb.perm(outr, 2, 1, 0, 3);   // [1,T,257]
    T* outi_p = gb.perm(outi, 2, 1, 0, 3);
    T* out = ggml_concat(ctx, outr_p, outi_p, 0);  // [2,T,257]
    ggml_set_output(out);

    // optional per-stage captures
    std::vector<std::pair<std::string, T*>> stages;
    if (cap) {
        stages = {{"feat", ft}, {"enc0", en0}, {"enc1", en1}, {"enc2", en2}, {"enc3", en3},
                  {"enc4", en4}, {"dpgrnn1", d1}, {"dpgrnn2", d2}, {"dec0", dc0}, {"dec1", dc1},
                  {"dec2", dc2}, {"dec3", dc3}, {"dec4", dc4}, {"mask", mask}};
        for (auto& s : stages) ggml_set_output(s.second);
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, n_op, false);
    ggml_build_forward_expand(graph, out);
    if (cap) for (auto& s : stages) ggml_build_forward_expand(graph, s.second);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, graph)) {
        fprintf(stderr, "gtcrn: gallocr alloc failed (n_op=%d)\n", n_op);
        ggml_gallocr_free(ga); ggml_free(ctx); return {};
    }

    // upload inputs: host spec is (257,T,2) row-major (F-slowest); the ggml input
    // tensors are [F,T,1] (F-fastest), so transpose (f,t) while deinterleaving.
    std::vector<float> re(257 * Tt), im(257 * Tt), rey(257 * Tt), imy(257 * Tt), zbuf(zlen, 0.0f);
    for (int f = 0; f < 257; ++f)
        for (int t = 0; t < Tt; ++t) {
            size_t si = ((size_t)f * Tt + t) * 2, gi = (size_t)t * 257 + f;
            re[gi] = spec_e[si]; im[gi] = spec_e[si + 1];
            rey[gi] = spec_y[si]; imy[gi] = spec_y[si + 1];
        }
    ggml_backend_tensor_set(re_e, re.data(), 0, re.size() * sizeof(float));
    ggml_backend_tensor_set(im_e, im.data(), 0, im.size() * sizeof(float));
    ggml_backend_tensor_set(re_y, rey.data(), 0, rey.size() * sizeof(float));
    ggml_backend_tensor_set(im_y, imy.data(), 0, imy.size() * sizeof(float));
    ggml_backend_tensor_set(zeros, zbuf.data(), 0, zbuf.size() * sizeof(float));

    auto t0 = std::chrono::high_resolution_clock::now();
    ggml_backend_graph_compute(backend, graph);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (compute_ms) *compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cap) {
        for (auto& s : stages) {
            T* t = s.second;
            NpyArray a; a.shape = {1, t->ne[2], t->ne[1], t->ne[0]};
            a.data.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, a.data.data(), 0, ggml_nbytes(t));
            (*cap)[s.first] = a;
        }
        NpyArray o; o.shape = {1, 257, Tt, 2}; o.data.resize((size_t)257 * Tt * 2);
        ggml_backend_tensor_get(out, o.data.data(), 0, ggml_nbytes(out));
        (*cap)["out_spec"] = o;
    }

    std::vector<float> result((size_t)257 * Tt * 2);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return result;
}

std::vector<float> GtcrnGraph::forward_stream(const float* spec_e, const float* spec_y, int Tt,
                                              std::map<std::string, NpyArray>* cap,
                                              double* compute_ms) const {
    // Build the T=1 graph ONCE (recurrent state exposed as input/output tensors).
    const int n_op = 60000;
    struct ggml_init_params cp;
    cp.mem_size = (size_t)n_op * (ggml_tensor_overhead() + 32) + ggml_graph_overhead_custom(n_op, false);
    cp.mem_buffer = nullptr;
    cp.no_alloc = true;
    ggml_context* ctx = ggml_init(cp);

    std::vector<std::pair<ggml_tensor*, ggml_tensor*>> states;
    GB gb{ctx, &wt, nullptr};
    gb.st = &states;

    ggml_tensor* re_e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, 1, 1);
    ggml_tensor* im_e = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, 1, 1);
    ggml_tensor* re_y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, 1, 1);
    ggml_tensor* im_y = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 257, 1, 1);
    ggml_tensor* zeros = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16 * 33);
    for (ggml_tensor* t : {re_e, im_e, re_y, im_y, zeros}) ggml_set_input(t);
    gb.zeros = zeros;

    ggml_tensor* ft = ggml_concat(ctx, gb.feat(re_e, im_e), gb.feat(re_y, im_y), 2);
    ggml_tensor* en0 = gb.conv_block(ft, "encoder.en_convs.0", 1, 2, false, false);
    ggml_tensor* en1 = gb.conv_block(en0, "encoder.en_convs.1", 2, 2, false, false);
    ggml_tensor* en2 = gb.gt_block_stream(en1, "encoder.en_convs.2", 1);
    ggml_tensor* en3 = gb.gt_block_stream(en2, "encoder.en_convs.3", 2);
    ggml_tensor* en4 = gb.gt_block_stream(en3, "encoder.en_convs.4", 5);
    ggml_tensor* d1 = gb.dpgrnn_stream(en4, "dpgrnn1");
    ggml_tensor* d2 = gb.dpgrnn_stream(d1, "dpgrnn2");
    ggml_tensor* x = gb.gt_block_stream(ggml_add(ctx, d2, en4), "decoder.de_convs.0", 5);
    x = gb.gt_block_stream(ggml_add(ctx, x, en3), "decoder.de_convs.1", 2);
    x = gb.gt_block_stream(ggml_add(ctx, x, en2), "decoder.de_convs.2", 1);
    x = gb.conv_block(ggml_add(ctx, x, en1), "decoder.de_convs.3", 2, 2, true, false);
    x = gb.conv_block(ggml_add(ctx, x, en0), "decoder.de_convs.4", 1, 2, true, true);
    ggml_tensor* mask = gb.band(x, gb.W("erb.bs"), 65);   // [257,1,2]
    ggml_tensor* mr = gb.cont(ggml_view_4d(ctx, mask, 257, 1, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3], 0));
    ggml_tensor* mi = gb.cont(ggml_view_4d(ctx, mask, 257, 1, 1, 1, mask->nb[1], mask->nb[2], mask->nb[3], (size_t)1 * mask->nb[2]));
    ggml_tensor* outr = ggml_sub(ctx, ggml_mul(ctx, re_e, mr), ggml_mul(ctx, im_e, mi));
    ggml_tensor* outi = ggml_add(ctx, ggml_mul(ctx, im_e, mr), ggml_mul(ctx, re_e, mi));
    ggml_tensor* out = ggml_concat(ctx, gb.perm(outr, 2, 1, 0, 3), gb.perm(outi, 2, 1, 0, 3), 0);  // [2,1,257]
    ggml_set_output(out);
    for (auto& s : states) ggml_set_output(s.second);
    // optional debug captures (single-frame stages, assembled across frames)
    ggml_tensor* cap_en4 = nullptr, *cap_d1 = nullptr;
    if (cap) {
        cap_en4 = gb.cont(en4); ggml_set_output(cap_en4);
        cap_d1 = gb.cont(d1);   ggml_set_output(cap_d1);
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, n_op, false);
    ggml_build_forward_expand(graph, out);
    for (auto& s : states) ggml_build_forward_expand(graph, s.second);
    if (cap) { ggml_build_forward_expand(graph, cap_en4); ggml_build_forward_expand(graph, cap_d1); }

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, graph)) {
        fprintf(stderr, "gtcrn(stream): gallocr alloc failed\n");
        ggml_gallocr_free(ga); ggml_free(ctx); return {};
    }

    // host-side state buffers (carried frame-to-frame), init zero
    std::vector<std::vector<float>> sbuf(states.size());
    for (size_t i = 0; i < states.size(); ++i)
        sbuf[i].assign(ggml_nelements(states[i].second), 0.0f);
    std::vector<float> zbuf(16 * 33, 0.0f);

    std::vector<float> result((size_t)257 * Tt * 2);
    std::vector<float> rf(257), imf(257), ryf(257), iyf(257), of((size_t)2 * 257);
    std::vector<float> acc_en4(cap ? (size_t)16 * Tt * 33 : 0), acc_d1(cap ? (size_t)16 * Tt * 33 : 0);
    std::vector<float> frbuf((size_t)16 * 33);
    double total_ms = 0;
    for (int t = 0; t < Tt; ++t) {
        for (int f = 0; f < 257; ++f) {
            size_t si = ((size_t)f * Tt + t) * 2;
            rf[f] = spec_e[si]; imf[f] = spec_e[si + 1]; ryf[f] = spec_y[si]; iyf[f] = spec_y[si + 1];
        }
        ggml_backend_tensor_set(re_e, rf.data(), 0, 257 * sizeof(float));
        ggml_backend_tensor_set(im_e, imf.data(), 0, 257 * sizeof(float));
        ggml_backend_tensor_set(re_y, ryf.data(), 0, 257 * sizeof(float));
        ggml_backend_tensor_set(im_y, iyf.data(), 0, 257 * sizeof(float));
        ggml_backend_tensor_set(zeros, zbuf.data(), 0, zbuf.size() * sizeof(float));  // intra h0
        for (size_t i = 0; i < states.size(); ++i)
            ggml_backend_tensor_set(states[i].first, sbuf[i].data(), 0, sbuf[i].size() * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, graph);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (size_t i = 0; i < states.size(); ++i)
            ggml_backend_tensor_get(states[i].second, sbuf[i].data(), 0, sbuf[i].size() * sizeof(float));
        ggml_backend_tensor_get(out, of.data(), 0, of.size() * sizeof(float));  // [2,1,257]: ri + f*2
        for (int f = 0; f < 257; ++f) {
            size_t ri = ((size_t)f * Tt + t) * 2;
            result[ri] = of[(size_t)f * 2 + 0];
            result[ri + 1] = of[(size_t)f * 2 + 1];
        }
        if (cap) {  // assemble en4/d1 ([F=33,1,C=16] per frame) into [C,T,F]
            ggml_backend_tensor_get(cap_en4, frbuf.data(), 0, frbuf.size() * sizeof(float));
            for (int c = 0; c < 16; ++c) for (int f = 0; f < 33; ++f)
                acc_en4[((size_t)c * Tt + t) * 33 + f] = frbuf[(size_t)c * 33 + f];  // ggml [F,1,C]: f + c*33
            ggml_backend_tensor_get(cap_d1, frbuf.data(), 0, frbuf.size() * sizeof(float));
            for (int c = 0; c < 16; ++c) for (int f = 0; f < 33; ++f)
                acc_d1[((size_t)c * Tt + t) * 33 + f] = frbuf[(size_t)c * 33 + f];
        }
    }
    if (compute_ms) *compute_ms = total_ms;
    if (cap) {
        NpyArray o; o.shape = {1, 257, Tt, 2}; o.data = result; (*cap)["out_spec"] = o;
        NpyArray e4; e4.shape = {1, 16, Tt, 33}; e4.data = acc_en4; (*cap)["enc4"] = e4;
        NpyArray dd; dd.shape = {1, 16, Tt, 33}; dd.data = acc_d1; (*cap)["dpgrnn1"] = dd;
    }

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return result;
}
