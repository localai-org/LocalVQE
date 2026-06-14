/**
 * LocalVQE inference using GGML computational graphs.
 *
 * Each network block builds a small GGML graph that is dispatched
 * to the selected backend (CPU with SIMD, or CUDA).
 */

#include "localvqe_graph.h"
#include "gguf.h"
#include "model_hash.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

// ── Helpers ───────────────────────────────────────────────────────────────

// Auto thread count for CPU backends. Capped at LVQE_AUTO_THREADS_MAX:
// the model is small enough that thread-launch + sync overhead dominates
// past ~4 threads on the CPUs we've measured (see README benchmark
// tables). On Linux we respect sched_getaffinity so taskset/cgroup/VM
// limits aren't exceeded — falling back to hardware_concurrency() on
// other platforms.
static constexpr int LVQE_AUTO_THREADS_MAX = 4;
static int auto_n_threads() {
    int avail = 0;
#if defined(__linux__)
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        avail = CPU_COUNT(&mask);
    }
#endif
    if (avail <= 0) avail = (int)std::thread::hardware_concurrency();
    if (avail <= 0) avail = 1;
    return std::min(LVQE_AUTO_THREADS_MAX, avail);
}

// Allocate a compute context (for graph node metadata, not tensor data).
static struct ggml_context* make_ctx(size_t mem = 256 * 1024) {
    struct ggml_init_params p = { mem, nullptr, true };
    return ggml_init(p);
}

// Create a 3D input tensor. Marks it as input.
static struct ggml_tensor* input_3d(struct ggml_context* ctx,
                                     int64_t ne0, int64_t ne1, int64_t ne2) {
    struct ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_input(t);
    return t;
}

// ── Block graphs ──────────────────────────────────────────────────────────

// Feature extraction: STFT (ne0=2, ne1=T, ne2=F) → (ne0=F, ne1=T, ne2=2)
// ggml_conv_2d with the im2col dst type pinned to F32: required for BF16
// weights (ggml's im2col kernel has no BF16 dst; mixed-type mul_mat handles
// the bf16 x f32 product by converting src1 rows). Identical to ggml_conv_2d
// otherwise.
static struct ggml_tensor* lvq_conv_2d(struct ggml_context* ctx,
                                        struct ggml_tensor* a,
                                        struct ggml_tensor* b,
                                        int s0, int s1, int p0, int p1,
                                        int d0, int d1) {
    struct ggml_tensor* im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1,
                                             true, GGML_TYPE_F32);
    // weight as src0 (the position that admits BF16/quantized types);
    // src1 (im2col) stays F32. Transpose restores ggml_conv_2d's layout.
    struct ggml_tensor* result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]),
        ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                        im2col->ne[3] * im2col->ne[2] * im2col->ne[1]));
    result = ggml_cont(ctx, ggml_transpose(ctx, result));
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2],
                             im2col->ne[3], a->ne[3]);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
    return result;
}

// Applies power-law compression: out = stft * mag^(c-1) / (1+eps)
// where mag = sqrt(re² + im² + eps).
static struct ggml_tensor* build_fe(struct ggml_context* ctx,
                                     struct ggml_tensor* stft, float power_c) {
    // stft: (ne0=2, ne1=T, ne2=F) — complex pair fastest
    int64_t T = stft->ne[1];
    int64_t F = stft->ne[2];
    float eps = 1e-12f;

    // mag² = sum over re²+im² along ne0 → (1, T, F)
    struct ggml_tensor* sqr = ggml_sqr(ctx, stft);
    struct ggml_tensor* mag2 = ggml_sum_rows(ctx, sqr);  // sum ne0
    struct ggml_tensor* mag2_eps = ggml_scale_bias(ctx, mag2, 1.0f, eps);
    struct ggml_tensor* mag = ggml_sqrt(ctx, mag2_eps);

    // s = mag^(c-1) / (1+eps)
    struct ggml_tensor* log_mag = ggml_log(ctx, mag);
    struct ggml_tensor* s = ggml_exp(ctx, ggml_scale(ctx, log_mag, power_c - 1.0f));
    s = ggml_scale(ctx, s, 1.0f / (1.0f + eps));
    // s: (1, T, F) — broadcasts over ne0=2 when multiplied with stft

    // Scale both re and im by s
    struct ggml_tensor* scaled = ggml_mul(ctx, stft, s);
    // scaled: (2, T, F)

    // Permute to (F, T, 2) to match Buf(C=2, T, F) memory layout
    struct ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, scaled, 2, 1, 0, 3));
    ggml_set_output(out);
    return out;
}

// Causal conv: pad_ext + conv2d + bias.
// Padding is computed from weight shape: causal in time, symmetric in freq.
// Input: (ne0=F, ne1=T, ne2=C_in, ne3=1)
// Output: (ne0=F_out, ne1=T, ne2=C_out, ne3=1)
static struct ggml_tensor* build_causal_conv(
    struct ggml_context* ctx,
    struct ggml_tensor* x,           // (F, T, C_in)
    struct ggml_tensor* weight,      // (kW, kH, C_in, C_out)
    struct ggml_tensor* bias,        // (C_out)
    int sF
) {
    int kW = (int)weight->ne[0];  // freq kernel
    int kH = (int)weight->ne[1];  // time kernel

    // Causal padding: pad_left = (kW-1)/2, pad_right = kW-1-pad_left
    int pad_left  = (kW - 1) / 2;
    int pad_right = kW - 1 - pad_left;
    int pad_top   = kH - 1;  // causal: all padding in past

    // Add batch dim if needed
    if (ggml_n_dims(x) == 3) {
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 1);
    }

    struct ggml_tensor* padded = ggml_pad_ext(ctx, x,
                                               pad_left, pad_right,  // dim0 (freq)
                                               pad_top, 0,           // dim1 (time): causal
                                               0, 0,                 // dim2 (channel)
                                               0, 0);                // dim3 (batch)

    struct ggml_tensor* conv = lvq_conv_2d(ctx, weight, padded,
                                             sF, 1, 0, 0, 1, 1);

    struct ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    conv = ggml_add(ctx, conv, b);

    return conv;
}

// Encoder block: causal_conv(stride 1,2) → ELU → causal_conv(stride 1,1) → ELU + skip
static struct ggml_tensor* build_encoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* conv_w, struct ggml_tensor* conv_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b
) {
    // Main branch: stride_F=2
    struct ggml_tensor* y = build_causal_conv(ctx, x, conv_w, conv_b, 2);
    y = ggml_elu(ctx, y);

    // Residual: stride_F=1
    struct ggml_tensor* res = build_causal_conv(ctx, y, res_w, res_b, 1);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    return res;
}

// Decoder block: skip_conv + residual + SubpixelConv2d + ChannelAffine + ELU
static struct ggml_tensor* build_decoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,        // (F, T, C)
    struct ggml_tensor* x_en,     // (F, T, C) encoder skip
    struct ggml_tensor* skip_w, struct ggml_tensor* skip_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b,
    struct ggml_tensor* deconv_w, struct ggml_tensor* deconv_b,
    struct ggml_tensor* bn_scale, struct ggml_tensor* bn_bias,  // null if is_last
    bool is_last
) {
    int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2];

    // skip_conv(x_en): 1x1 conv
    struct ggml_tensor* x_en_4d = ggml_reshape_4d(ctx, x_en, F, T, C, 1);
    struct ggml_tensor* skip = lvq_conv_2d(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1);
    // Bias
    struct ggml_tensor* sb = ggml_reshape_4d(ctx, skip_b, 1, 1, skip_b->ne[0], 1);
    skip = ggml_add(ctx, skip, sb);
    skip = ggml_reshape_3d(ctx, skip, F, T, C);

    // y = x + skip
    struct ggml_tensor* y = ggml_add(ctx, x, skip);

    // Residual
    struct ggml_tensor* res = build_causal_conv(ctx, y, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, F, T, C);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    // SubpixelConv2d: causal_conv → pixel shuffle
    int64_t C_out = deconv_w->ne[3] / 2;
    struct ggml_tensor* deconv = build_causal_conv(ctx, res, deconv_w, deconv_b, 1);
    // deconv: (F, T, C_out*2, 1) → strip batch dim
    deconv = ggml_reshape_3d(ctx, deconv, F, T, C_out * 2);

    // Pixel shuffle: (ne0=F, ne1=T, ne2=2*C_out) → (ne0=2*F, ne1=T, ne2=C_out)
    // Split channels: reshape to (F, T, C_out, 2)
    struct ggml_tensor* r1 = ggml_reshape_4d(ctx, deconv, F, T, C_out, 2);
    // Permute to (F, 2, T, C_out): old[0]→0, old[1]→2, old[2]→3, old[3]→1
    struct ggml_tensor* r2 = ggml_permute(ctx, r1, 0, 2, 3, 1);
    // Make contiguous and reshape to (2*F, T, C_out)
    struct ggml_tensor* shuffled = ggml_reshape_3d(ctx, ggml_cont(ctx, r2), 2 * F, T, C_out);

    if (!is_last && bn_scale && bn_bias) {
        // ChannelAffine: x * scale + bias, broadcast over F and T
        struct ggml_tensor* sc = ggml_reshape_3d(ctx, bn_scale, 1, 1, C_out);
        struct ggml_tensor* bi = ggml_reshape_3d(ctx, bn_bias, 1, 1, C_out);
        shuffled = ggml_elu(ctx, ggml_add(ctx, ggml_mul(ctx, shuffled, sc), bi));
    }

    return shuffled;
}

// Freq trim: take first target_F elements of dim 0
static struct ggml_tensor* build_freq_trim(struct ggml_context* ctx,
                                            struct ggml_tensor* x,
                                            int64_t target_F) {
    if (x->ne[0] <= target_F) return x;
    // ggml_view_3d creates non-contiguous tensor; wrap in cont for downstream reshape
    return ggml_cont(ctx,
        ggml_view_3d(ctx, x, target_F, x->ne[1], x->ne[2],
                      x->nb[1], x->nb[2], 0));
}

// AlignBlock: cross-attention soft delay estimation.
// x_mic, x_ref: (ne0=F, ne1=T, ne2=C)
// Returns: (ne0=F, ne1=T, ne2=C) — aligned reference.
static struct ggml_tensor* build_align(
    struct ggml_context* ctx,
    struct ggml_tensor* x_mic,   // (F, T, C)
    struct ggml_tensor* x_ref,   // (F, T, C)
    struct ggml_tensor* pmw, struct ggml_tensor* pmb,  // pconv_mic: (1, 1, C, H)
    struct ggml_tensor* prw, struct ggml_tensor* prb,  // pconv_ref: (1, 1, C, H)
    struct ggml_tensor* sw, struct ggml_tensor* sb,    // smooth conv: (3, 5, H, 1)
    int dmax
) {
    int64_t F = x_mic->ne[0], T = x_mic->ne[1], C = x_mic->ne[2];
    int64_t H = pmw->ne[3];  // align_hidden

    // 1x1 conv projections → Q, K: (F, T, H)
    struct ggml_tensor* mic4 = ggml_reshape_4d(ctx, x_mic, F, T, C, 1);
    struct ggml_tensor* ref4 = ggml_reshape_4d(ctx, x_ref, F, T, C, 1);
    struct ggml_tensor* Q4 = ggml_add(ctx, lvq_conv_2d(ctx, pmw, mic4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, pmb, 1,1,H,1));
    struct ggml_tensor* K4 = ggml_add(ctx, lvq_conv_2d(ctx, prw, ref4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, prb, 1,1,H,1));
    struct ggml_tensor* Q = ggml_reshape_3d(ctx, Q4, F, T, H);
    struct ggml_tensor* K = ggml_reshape_3d(ctx, K4, F, T, H);

    // Pad K in time: prepend dmax-1 zeros → (F, T+dmax-1, H)
    struct ggml_tensor* Kp = ggml_pad_ext(ctx, K, 0,0, dmax-1,0, 0,0, 0,0);

    // Similarity: V[d,t,h] = sum_f Q[f,t,h] * Kp[f,t+d,h] / sqrt(F)
    // Build by iterating over delays and concatenating
    float scale = 1.0f / std::sqrt((float)F);
    struct ggml_tensor* V = nullptr;  // will be (1, T, H) then grow to (dmax, T, H)

    for (int d = 0; d < dmax; d++) {
        // View of Kp at time offset d: (F, T, H) — make contiguous for mul
        struct ggml_tensor* Kd = ggml_cont(ctx,
            ggml_view_3d(ctx, Kp, F, T, H,
                          Kp->nb[1], Kp->nb[2],
                          d * Kp->nb[1]));
        // Element-wise multiply Q * Kd → (F, T, H), then sum over F → (1, T, H)
        struct ggml_tensor* qk = ggml_mul(ctx, Q, Kd);
        struct ggml_tensor* sim = ggml_sum_rows(ctx, qk);  // sum over ne0 → (1, T, H)
        sim = ggml_scale(ctx, sim, scale);

        if (V == nullptr) {
            V = sim;  // (1, T, H)
        } else {
            V = ggml_concat(ctx, V, sim, 0);  // concat along dim0 → (d+1, T, H)
        }
    }
    // V: (dmax, T, H)

    // Smooth conv: pad(1,1,4,0) on (dmax, T) dims, then Conv2d(H,1,(5,3))
    // Permute V to match conv input: (dmax, T, H) = (ne0=dmax, ne1=T, ne2=H)
    // Pad: freq(dmax) +1/+1, time(T) +4/+0 → (dmax+2, T+4, H, 1)
    struct ggml_tensor* Vp = ggml_reshape_4d(ctx, V, dmax, T, H, 1);
    Vp = ggml_pad_ext(ctx, Vp, 1,1, 4,0, 0,0, 0,0);
    // Conv2d with sw: kernel (kF=3, kT=5, C_in=H, C_out=1), stride (1,1)
    struct ggml_tensor* Vc = lvq_conv_2d(ctx, sw, Vp, 1,1, 0,0, 1,1);
    // Add bias
    struct ggml_tensor* s_bias = ggml_reshape_4d(ctx, sb, 1,1,1,1);
    Vc = ggml_add(ctx, Vc, s_bias);
    // Vc: (dmax, T, 1, 1) → reshape to (dmax, T)
    Vc = ggml_reshape_2d(ctx, Vc, dmax, T);

    // Softmax over delay (dim0=dmax)
    struct ggml_tensor* attn = ggml_soft_max(ctx, Vc);  // softmax over ne0
    // attn: (dmax, T)

    // Pad x_ref in time: prepend dmax-1 zeros → (F, T+dmax-1, C)
    struct ggml_tensor* rp = ggml_pad_ext(ctx, x_ref, 0,0, dmax-1,0, 0,0, 0,0);

    // Weighted sum: aligned[f,t,c] = sum_d attn[d,t] * rp[f,t+d,c]
    struct ggml_tensor* aligned = nullptr;
    for (int d = 0; d < dmax; d++) {
        // View of rp at time offset d: (F, T, C) — make contiguous
        struct ggml_tensor* rd = ggml_cont(ctx,
            ggml_view_3d(ctx, rp, F, T, C,
                          rp->nb[1], rp->nb[2],
                          d * rp->nb[1]));
        // attn[d, t]: element at ne0 offset d. View (1, T) slice, make contiguous
        struct ggml_tensor* ad = ggml_cont(ctx,
            ggml_view_2d(ctx, attn, 1, T,
                          attn->nb[1],
                          d * sizeof(float)));
        // Reshape ad to (1, T, 1) for broadcasting over F and C
        struct ggml_tensor* ad3 = ggml_reshape_3d(ctx, ad, 1, T, 1);
        // Multiply: (F, T, C) * (1, T, 1) → (F, T, C)
        struct ggml_tensor* contrib = ggml_mul(ctx, rd, ad3);

        if (aligned == nullptr) {
            aligned = contrib;
        } else {
            aligned = ggml_add(ctx, aligned, contrib);
        }
    }
    // aligned: (F, T, C)
    return aligned;
}

// S4D Bottleneck: flatten (C,T,F) → input_proj → complex diagonal SSM → output_proj + skip
// Precomputed constants a_real, a_imag stored in GGUF (avoids sin/cos/softplus in graph).
// h_real_init_out/h_imag_init_out: if non-null, receive initial state tensors.
static struct ggml_tensor* build_bottleneck(
    struct ggml_context* ctx,
    struct ggml_tensor* x,           // (F, T, C)
    struct ggml_tensor* in_w,        // input_proj weight: (input_size, H)
    struct ggml_tensor* in_b,        // input_proj bias: (H)
    struct ggml_tensor* out_w,       // output_proj weight: (H, input_size)
    struct ggml_tensor* out_b,       // output_proj bias: (input_size)
    struct ggml_tensor* a_real,      // precomputed: r*cos(theta), (H)
    struct ggml_tensor* a_imag,      // precomputed: r*sin(theta), (H)
    struct ggml_tensor* B_real,      // input coupling real, (H)
    struct ggml_tensor* B_imag,      // input coupling imag, (H)
    struct ggml_tensor* C_real,      // output coupling real, (H)
    struct ggml_tensor* C_imag,      // output coupling imag, (H)
    struct ggml_tensor* D,           // skip connection, (input_size)
    struct ggml_tensor** h_real_init_out = nullptr,
    struct ggml_tensor** h_imag_init_out = nullptr
) {
    int64_t F = x->ne[0], T = x->ne[1], C = x->ne[2];
    int64_t input_size = C * F;
    int64_t H = in_w->ne[1];  // hidden_size

    // Flatten: (F, T, C) → (C*F, T)
    struct ggml_tensor* flat = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    flat = ggml_reshape_2d(ctx, flat, C * F, T);

    // Initial hidden state (caller must zero after gallocr)
    struct ggml_tensor* hr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    struct ggml_tensor* hi = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_set_input(hr);
    ggml_set_input(hi);
    if (h_real_init_out) *h_real_init_out = hr;
    if (h_imag_init_out) *h_imag_init_out = hi;

    // Collect S4D outputs
    struct ggml_tensor* ssm_out = nullptr;

    for (int64_t t = 0; t < T; t++) {
        // x_t: (input_size, 1)
        struct ggml_tensor* x_t = ggml_cont(ctx,
            ggml_view_2d(ctx, flat, input_size, 1,
                          flat->nb[1], t * flat->nb[1]));

        // v = input_proj(x_t): (H, 1)
        struct ggml_tensor* v = ggml_add(ctx,
            ggml_mul_mat(ctx, in_w, x_t),
            ggml_reshape_2d(ctx, in_b, H, 1));
        struct ggml_tensor* v1d = ggml_reshape_1d(ctx, v, H);

        // Complex diagonal recurrence:
        //   hr' = a_real*hr - a_imag*hi + B_real*v
        //   hi' = a_real*hi + a_imag*hr + B_imag*v
        struct ggml_tensor* hr_new = ggml_add(ctx,
            ggml_sub(ctx,
                ggml_mul(ctx, a_real, hr),
                ggml_mul(ctx, a_imag, hi)),
            ggml_mul(ctx, B_real, v1d));

        struct ggml_tensor* hi_new = ggml_add(ctx,
            ggml_add(ctx,
                ggml_mul(ctx, a_real, hi),
                ggml_mul(ctx, a_imag, hr)),
            ggml_mul(ctx, B_imag, v1d));

        hr = hr_new;
        hi = hi_new;

        // y_t = C_real*hr - C_imag*hi: (H)
        struct ggml_tensor* y_t = ggml_sub(ctx,
            ggml_mul(ctx, C_real, hr),
            ggml_mul(ctx, C_imag, hi));
        struct ggml_tensor* y_2d = ggml_reshape_2d(ctx, y_t, H, 1);

        ssm_out = ssm_out ? ggml_concat(ctx, ssm_out, y_2d, 1) : y_2d;
    }
    // ssm_out: (H, T)

    // Output projection: (H, T) → (input_size, T)
    struct ggml_tensor* proj_out = ggml_add(ctx,
        ggml_mul_mat(ctx, out_w, ssm_out),
        ggml_reshape_2d(ctx, out_b, input_size, 1));

    // Skip connection: + D * flat
    struct ggml_tensor* D_2d = ggml_reshape_2d(ctx, D, input_size, 1);
    struct ggml_tensor* skip = ggml_mul(ctx, D_2d, flat);
    struct ggml_tensor* fc_out = ggml_add(ctx, proj_out, skip);

    // Reshape: (C*F, T) → (F, C, T) → permute to (F, T, C)
    struct ggml_tensor* reshaped = ggml_reshape_3d(ctx, fc_out, F, C, T);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 2, 1, 3));

    return out;
}

// Concat along channel dim: (F, T, C1) + (F, T, C2) → (F, T, C1+C2)
static struct ggml_tensor* build_concat_channels(
    struct ggml_context* ctx,
    struct ggml_tensor* a,
    struct ggml_tensor* b
) {
    return ggml_concat(ctx, a, b, 2);  // concat along ne2 (channel)
}

// DFT basis vectors for 3-tap CCM
static const float CCM_VR[3] = {1.0f, -0.5f, -0.5f};
static const float CCM_VI[3] = {0.0f, 0.86602540378f, -0.86602540378f};

// Complex Convolving Mask: mask (27ch) + mic_stft → enhanced STFT
// mask: (F, T, 27), mic_stft input: (2, T, F) — original STFT
// Returns: (F, T, 2) enhanced STFT
static struct ggml_tensor* build_ccm(
    struct ggml_context* ctx,
    struct ggml_tensor* mask,     // (F, T, 27)
    struct ggml_tensor* stft_in,  // (2, T, F) — original mic STFT
    struct ggml_tensor* vr_tensor, struct ggml_tensor* vi_tensor  // DFT basis (3)
) {
    int64_t F = mask->ne[0], T = mask->ne[1];

    // Build H_real and H_imag: (F, T, 9) from mask channels
    // mask has 27 channels = 3 (DFT) × 9 (spatial)
    // H_real[c] = sum_r VR[r] * mask[r*9 + c], H_imag[c] = sum_r VI[r] * mask[r*9 + c]

    struct ggml_tensor* Hr = nullptr;  // (F, T, 9)
    struct ggml_tensor* Hi = nullptr;

    for (int r = 0; r < 3; r++) {
        // Extract 9 channels for this DFT component: mask[:,:,r*9:(r+1)*9]
        struct ggml_tensor* m_r = ggml_cont(ctx,
            ggml_view_3d(ctx, mask, F, T, 9,
                          mask->nb[1], mask->nb[2],
                          r * 9 * mask->nb[2]));
        // Scale by VR[r] and VI[r]
        struct ggml_tensor* scaled_r = ggml_scale(ctx, m_r, CCM_VR[r]);
        struct ggml_tensor* scaled_i = ggml_scale(ctx, m_r, CCM_VI[r]);

        Hr = Hr ? ggml_add(ctx, Hr, scaled_r) : scaled_r;
        Hi = Hi ? ggml_add(ctx, Hi, scaled_i) : scaled_i;
    }
    // Hr, Hi: (F, T, 9) — 3x3 spatial filter, real and imaginary parts

    // Permute input STFT: (2, T, F) → (F, T, 2)
    struct ggml_tensor* stft = ggml_cont(ctx, ggml_permute(ctx, stft_in, 2, 1, 0, 3));

    // Pad input: (F, T, 2) → pad freq +1/+1, time +2/+0 → (F+2, T+2, 2)
    struct ggml_tensor* xp = ggml_pad_ext(ctx, stft, 1,1, 2,0, 0,0, 0,0);

    // Complex convolution: for each 3x3 position (m,n):
    //   e_r += Hr[mn] * xp_r[t+m, f+n] - Hi[mn] * xp_i[t+m, f+n]
    //   e_i += Hr[mn] * xp_i[t+m, f+n] + Hi[mn] * xp_r[t+m, f+n]
    struct ggml_tensor* er = nullptr;
    struct ggml_tensor* ei = nullptr;

    for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
            int ki = m * 3 + n;

            // View of Hr/Hi for this kernel position: (F, T, 1) = channel ki
            struct ggml_tensor* hr = ggml_cont(ctx,
                ggml_view_3d(ctx, Hr, F, T, 1,
                              Hr->nb[1], Hr->nb[2],
                              ki * Hr->nb[2]));
            struct ggml_tensor* hi = ggml_cont(ctx,
                ggml_view_3d(ctx, Hi, F, T, 1,
                              Hi->nb[1], Hi->nb[2],
                              ki * Hi->nb[2]));

            // View of padded input at offset (n, m): re and im channels
            // xp is (F+2, T+2, 2). We want (F, T) starting at freq=n, time=m
            struct ggml_tensor* xr = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, T, 1,
                              xp->nb[1], xp->nb[2],
                              m * xp->nb[1] + n * xp->nb[0]));
            struct ggml_tensor* xi = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, T, 1,
                              xp->nb[1], xp->nb[2],
                              xp->nb[2] + m * xp->nb[1] + n * xp->nb[0]));

            // Complex multiply: er += hr*xr - hi*xi, ei += hr*xi + hi*xr
            struct ggml_tensor* cr = ggml_sub(ctx, ggml_mul(ctx, hr, xr),
                                               ggml_mul(ctx, hi, xi));
            struct ggml_tensor* ci = ggml_add(ctx, ggml_mul(ctx, hr, xi),
                                               ggml_mul(ctx, hi, xr));

            er = er ? ggml_add(ctx, er, cr) : cr;
            ei = ei ? ggml_add(ctx, ei, ci) : ci;
        }
    }
    // er, ei: (F, T, 1)

    // Concat re and im → (F, T, 2)
    struct ggml_tensor* enhanced = ggml_concat(ctx, er, ei, 2);
    return enhanced;
}

// ── Model loading ─────────────────────────────────────────────────────────

static uint32_t gguf_u32(struct gguf_context* ctx, const char* key) {
    int idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_u32(ctx, idx) : 0;
}

static void ensure_backends_loaded() {
    static std::once_flag f;
    std::call_once(f, []() {
        ggml_backend_load_all();
        // ggml's default search looks next to the host executable and in
        // cwd — wrong when liblocalvqe.so is bundled inside an embedder
        // (e.g. an OBS plugin tree) where the ggml-cpu-*.so variants
        // live next to liblocalvqe.so. Self-locate via dladdr (POSIX)
        // or GetModuleHandleEx (Windows) and also load from that dir;
        // ggml dedupes by score so the call above isn't wasted. Every
        // failure path is loud — silent skipping here just produces an
        // empty registry and a confusing crash later.
        // Resolve liblocalvqe.so's own path as a UTF-8 std::string `p`,
        // then split off the parent dir. Symmetric across platforms so
        // failure messages always quote the actual path (essential for
        // Windows bug reports — we have no machine to reproduce on).
        std::string p;
#if defined(_WIN32)
        // FROM_ADDRESS lets us pass an in-module function pointer cast
        // to LPCWSTR; UNCHANGED_REFCOUNT means we don't have to free.
        HMODULE hmod = nullptr;
        if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&ensure_backends_loaded),
                &hmod)) {
            DWORD err = GetLastError();
            fprintf(stderr, "localvqe[win]: GetModuleHandleExW failed "
                "for self-symbol (GetLastError=%lu); skipping "
                "embedded-backend self-discovery\n", err);
            return;
        }
        wchar_t wpath[MAX_PATH];
        DWORD wlen = GetModuleFileNameW(hmod, wpath, MAX_PATH);
        DWORD wlen_err = GetLastError();  // captured before next call
        if (wlen == 0 || wlen >= MAX_PATH) {
            fprintf(stderr, "localvqe[win]: GetModuleFileNameW failed "
                "or path exceeds MAX_PATH=%d (returned len=%lu, "
                "GetLastError=%lu); skipping embedded-backend "
                "self-discovery\n",
                static_cast<int>(MAX_PATH), wlen, wlen_err);
            return;
        }
        // Convert wide → UTF-8 immediately so all subsequent failure
        // paths and the success log can quote the actual path.
        int u8_len = WideCharToMultiByte(CP_UTF8, 0, wpath, -1,
                                          nullptr, 0, nullptr, nullptr);
        if (u8_len <= 0) {
            DWORD err = GetLastError();
            fprintf(stderr, "localvqe[win]: WideCharToMultiByte sizing "
                "failed (GetLastError=%lu, wide-len=%lu); skipping "
                "embedded-backend self-discovery — the module path "
                "contains a wide character that won't round-trip to "
                "UTF-8 (vanishingly unlikely on Windows 10+)\n",
                err, wlen);
            return;
        }
        p.resize(static_cast<size_t>(u8_len - 1));  // excl. trailing NUL
        if (WideCharToMultiByte(CP_UTF8, 0, wpath, -1, p.data(), u8_len,
                                nullptr, nullptr) == 0) {
            DWORD err = GetLastError();
            fprintf(stderr, "localvqe[win]: WideCharToMultiByte "
                "conversion failed (GetLastError=%lu, expected %d "
                "bytes); skipping embedded-backend self-discovery\n",
                err, u8_len);
            return;
        }
        // Windows accepts both separators in paths; the last one wins.
        const size_t slash = p.find_last_of("\\/");
#else
        Dl_info info{};
        if (dladdr((void*)&ensure_backends_loaded, &info) == 0) {
            fprintf(stderr, "localvqe: dladdr failed for self-symbol; "
                "skipping embedded-backend self-discovery\n");
            return;
        }
        if (!info.dli_fname) {
            fprintf(stderr, "localvqe: dladdr returned NULL dli_fname; "
                "skipping embedded-backend self-discovery\n");
            return;
        }
        p = info.dli_fname;
        const size_t slash = p.find_last_of('/');
#endif
        if (slash == std::string::npos) {
            fprintf(stderr, "localvqe: module path '%s' has no path "
                "separator; skipping embedded-backend self-discovery\n",
                p.c_str());
            return;
        }
        if (slash == 0) {
            fprintf(stderr, "localvqe: module path '%s' has no parent "
                "directory; skipping embedded-backend self-discovery\n",
                p.c_str());
            return;
        }
        const std::string dir = p.substr(0, slash);
        // Loud success line: if ggml later reports "Registered backends
        // (0)", the user can see exactly what directory we scanned, and
        // ggml itself prints which .so files it tried to load.
        fprintf(stderr, "localvqe: loading embedded backends from '%s' "
            "(self-resolved from '%s')\n", dir.c_str(), p.c_str());
        ggml_backend_load_all_from_path(dir.c_str());
    });
}

void dvqe_list_devices(FILE* out) {
    ensure_backends_loaded();
    size_t n_reg = ggml_backend_reg_count();
    fprintf(out, "Registered backends (%zu):\n", n_reg);
    for (size_t i = 0; i < n_reg; i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        const char* rname = ggml_backend_reg_name(reg);
        size_t n_dev = ggml_backend_reg_dev_count(reg);
        fprintf(out, "  %s (%zu device%s)\n",
                rname ? rname : "?", n_dev, n_dev == 1 ? "" : "s");
        for (size_t j = 0; j < n_dev; j++) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
            const char* dn = ggml_backend_dev_name(dev);
            const char* dd = ggml_backend_dev_description(dev);
            const char* dt = "?";
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:   dt = "CPU";   break;
                case GGML_BACKEND_DEVICE_TYPE_GPU:   dt = "GPU";   break;
                case GGML_BACKEND_DEVICE_TYPE_IGPU:  dt = "iGPU";  break;
                case GGML_BACKEND_DEVICE_TYPE_ACCEL: dt = "ACCEL"; break;
            }
            fprintf(out, "    [%zu] %s — %s (%s)\n",
                    j, dn ? dn : "?", dd ? dd : "?", dt);
        }
    }
}

bool load_graph_model(const char* path, dvqe_graph_model& model,
                      bool verbose, int n_threads) {
    return load_graph_model_ex(path, model, "CPU", 0, verbose, n_threads);
}

bool load_graph_model_ex(const char* path, dvqe_graph_model& model,
                         const char* backend_name, int device_index,
                         bool verbose, int n_threads) {
    // Refuse any GGUF that isn't on the released-models allowlist. The
    // GGUF parser below is not hardened against malicious inputs (the
    // vendored ggml calls ggml_abort on bad metadata), so we treat the
    // hash check as the integrity boundary. See model_hash.cpp.
    if (!localvqe::verify_model_hash(path)) return false;

    // Load GGUF metadata only — we'll allocate tensors on the backend buffer
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &model.weight_ctx;

    struct gguf_context* gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "Failed to load: %s\n", path); return false; }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gctx_guard(gctx, gguf_free);
    auto fail = [&]() { free_graph_model(model); return false; };

    // Read hyperparameters
    auto& hp = model.hparams;
    hp.n_fft        = (int)gguf_u32(gctx, "localvqe.n_fft");
    hp.hop_length   = (int)gguf_u32(gctx, "localvqe.hop_length");
    hp.n_freq_bins  = (int)gguf_u32(gctx, "localvqe.n_freq_bins");
    hp.sample_rate  = (int)gguf_u32(gctx, "localvqe.sample_rate");
    hp.dmax         = (int)gguf_u32(gctx, "localvqe.dmax");
    hp.align_hidden = (int)gguf_u32(gctx, "localvqe.align_hidden");
    int idx = gguf_find_key(gctx, "localvqe.power_law_c");
    hp.power_law_c = idx >= 0 ? gguf_get_val_f32(gctx, idx) : 0.3f;

    // Kernel size (default to 4x3 for backward compat with old GGUF files)
    idx = gguf_find_key(gctx, "localvqe.kernel_size_h");
    hp.kernel_size_h = idx >= 0 ? (int)gguf_get_val_u32(gctx, idx) : 4;
    idx = gguf_find_key(gctx, "localvqe.kernel_size_w");
    hp.kernel_size_w = idx >= 0 ? (int)gguf_get_val_u32(gctx, idx) : 3;

    // Bottleneck hidden size (0 = auto)
    idx = gguf_find_key(gctx, "localvqe.bottleneck_hidden");
    hp.bottleneck_hidden = idx >= 0 ? (int)gguf_get_val_u32(gctx, idx) : 0;

    // Architecture version: 1 = post-conv BN+ELU (legacy), 2 = pre-norm
    // CausalGroupNorm + ReLU6 (v1.1 onward).
    idx = gguf_find_key(gctx, "localvqe.version");
    hp.version = idx >= 0 ? (int)gguf_get_val_u32(gctx, idx) : 1;

    // Front-end-only build (2.7K release): daf.* tensors, no mask backend.
    idx = gguf_find_key(gctx, "localvqe.daf.standalone");
    hp.daf_standalone = idx >= 0 && gguf_get_val_bool(gctx, idx);

    int mic_n = (int)gguf_u32(gctx, "localvqe.mic_channels.count");
    hp.mic_channels.resize(mic_n);
    for (int i = 0; i < mic_n; i++) {
        char k[64]; snprintf(k, sizeof(k), "localvqe.mic_channels.%d", i);
        hp.mic_channels[i] = (int)gguf_u32(gctx, k);
    }
    int far_n = (int)gguf_u32(gctx, "localvqe.far_channels.count");
    hp.far_channels.resize(far_n);
    for (int i = 0; i < far_n; i++) {
        char k[64]; snprintf(k, sizeof(k), "localvqe.far_channels.%d", i);
        hp.far_channels[i] = (int)gguf_u32(gctx, k);
    }

    // Index weight tensors by name
    int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor* t = ggml_get_tensor(model.weight_ctx, name);
        if (t) model.weights[name] = t;
    }

    if (verbose) {
        printf("Graph model: %zu tensors, n_fft=%d, dmax=%d\n",
               model.weights.size(), hp.n_fft, hp.dmax);
    }

    ensure_backends_loaded();

    ggml_backend_reg_t reg = (backend_name && *backend_name)
        ? ggml_backend_reg_by_name(backend_name) : nullptr;
    if (!reg) {
        fprintf(stderr, "localvqe: backend '%s' not registered\n",
                backend_name ? backend_name : "(null)");
        dvqe_list_devices(stderr);
        return fail();
    }
    size_t n_dev = ggml_backend_reg_dev_count(reg);
    if (device_index < 0 || (size_t)device_index >= n_dev) {
        fprintf(stderr,
            "localvqe: device %d out of range for backend '%s' (valid: 0..%zu)\n",
            device_index, backend_name, n_dev ? n_dev - 1 : 0);
        return fail();
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, (size_t)device_index);
    model.backend = ggml_backend_dev_init(dev, nullptr);
    if (!model.backend) {
        fprintf(stderr, "localvqe: failed to init %s device %d\n",
                backend_name, device_index);
        return fail();
    }

    if (n_threads <= 0) {
        n_threads = auto_n_threads();
    }
    auto fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(model.backend, n_threads);

    if (verbose) {
        const char* dn = ggml_backend_dev_name(dev);
        const char* dd = ggml_backend_dev_description(dev);
        fprintf(stderr, "localvqe: backend=%s device[%d]=%s (%s) threads=%d\n",
                ggml_backend_name(model.backend), device_index,
                dn ? dn : "?", dd ? dd : "?", n_threads);
    }

    // Quantized tensors (Q4_K, Q8_0) are kept in their native format for inference.
    model.weight_buf = ggml_backend_alloc_ctx_tensors(model.weight_ctx, model.backend);
    if (!model.weight_buf) {
        fprintf(stderr, "Failed to allocate weight buffer\n");
        return fail();
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return fail();
    }
    size_t data_offset = gguf_get_data_offset(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor* t = ggml_get_tensor(model.weight_ctx, name);
        if (!t) continue;
        size_t offset = gguf_get_tensor_offset(gctx, i);
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> buf(nbytes);
        fseek(f, data_offset + offset, SEEK_SET);
        if (fread(buf.data(), 1, nbytes, f) != nbytes) {
            fprintf(stderr, "Short read for tensor: %s\n", name);
            fclose(f);
            return fail();
        }
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    }
    fclose(f);
    return true;
}

void free_graph_model(dvqe_graph_model& model) {
    if (model.weight_buf) { ggml_backend_buffer_free(model.weight_buf); model.weight_buf = nullptr; }
    if (model.backend) { ggml_backend_free(model.backend); model.backend = nullptr; }
    if (model.weight_ctx) { ggml_free(model.weight_ctx); model.weight_ctx = nullptr; }
    model.weights.clear();
}

// ── Streaming graph (T=1 with history buffers) ──────────────────────────────

// Causal conv for streaming — persistent-window scheme:
// a persistent PRE-PADDED input window (Fp, kH, C_in) lives across frames as
// a graph input; the graph writes only the current frame into the window's
// last row (set_inplace, F*C floats), and the host shifts rows up by one
// after each frame. Replaces the previous two whole-window ggml_concat
// copies + ggml_cont + per-frame freq ggml_pad (≈30% of hop time on the
// 200K-class model).
static struct ggml_tensor* build_causal_conv_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,           // (F, 1, C_in) — current frame
    struct ggml_tensor* weight,      // (kF=3, kT=4, C_in, C_out)
    struct ggml_tensor* bias,        // (C_out)
    int sF
) {
    int64_t F = x->ne[0], C_in = x->ne[2];
    int kW = (int)weight->ne[0];
    int kH = (int)weight->ne[1];

    // Causal padding for freq dimension — baked into the window layout.
    int pad_left  = (kW - 1) / 2;
    int pad_right = kW - 1 - pad_left;
    int64_t Fp = F + pad_left + pad_right;

    int64_t C_w = weight->ne[2];  // weight IC (may be padded for quantization)
    struct ggml_tensor* win = ggml_new_tensor_3d(sg.win_ctx, GGML_TYPE_F32,
                                                  Fp, kH, C_w);
    sg.conv_win.push_back(win);

    // Write the current frame into the last row, center columns. The window
    // keeps permanent zeros in the pad columns (and in channels >= C_in for
    // quant-padded weights) — reset_stream_graph zeroes the whole buffer.
    struct ggml_tensor* xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    struct ggml_tensor* wset = ggml_set_inplace(
        ctx, win, xc,
        /*nb1=*/win->nb[1] /* unused for 1-row writes; row stride of view */,
        /*nb2=*/win->nb[2],
        /*nb3=*/win->nb[3],
        /*offset=*/(size_t)pad_left * win->nb[0] + (size_t)(kH - 1) * win->nb[1]);

    struct ggml_tensor* x4d = ggml_reshape_4d(ctx, wset, Fp, kH, C_w, 1);
    struct ggml_tensor* conv = lvq_conv_2d(ctx, weight, x4d,
                                             sF, 1, 0, 0, 1, 1);
    struct ggml_tensor* b = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    conv = ggml_add(ctx, conv, b);

    return conv;
}

// Streaming encoder block: main conv (stride 2) + residual conv (stride 1).
static struct ggml_tensor* build_encoder_block_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,
    struct ggml_tensor* conv_w, struct ggml_tensor* conv_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b
) {
    struct ggml_tensor* y = build_causal_conv_s(ctx, sg, x, conv_w, conv_b, 2);
    y = ggml_elu(ctx, y);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], y->ne[2]);

    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, y, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, res->ne[0], res->ne[1], res->ne[2]);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    return res;
}

// CausalGroupNorm: per-frame normalization over (F, C). Streaming-safe —
// stats at frame t depend only on frame t's channels and freqs.
//
//   y = gamma * (x - mean(F, C)) / sqrt(var(F, C) + eps) + beta
//
// Input  x:     (F, T=1, C)
// Params gamma: (C,), beta: (C,)
// Output:       (F, 1, C)
static struct ggml_tensor* build_causal_groupnorm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    float eps = 1e-5f
) {
    int64_t F = x->ne[0];
    int64_t T = x->ne[1];
    int64_t C = x->ne[2];

    // Permute (F, T, C) -> (F, C, T) so all (F, C) values for a given T
    // become contiguous along ne[0]*ne[1].
    struct ggml_tensor* perm = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    // Flatten F*C into ne[0] so ggml_norm reduces over both.
    struct ggml_tensor* flat = ggml_reshape_2d(ctx, perm, F * C, T);
    // ggml_norm: (x - mean(ne0)) / sqrt(var(ne0) + eps) per row.
    struct ggml_tensor* normed = ggml_norm(ctx, flat, eps);
    // Restore (F, C, T).
    struct ggml_tensor* y = ggml_reshape_3d(ctx, normed, F, C, T);
    // Per-channel affine: gamma, beta have shape (C,) → broadcast over (F, _, T).
    struct ggml_tensor* g = ggml_reshape_3d(ctx, gamma, 1, C, 1);
    struct ggml_tensor* b = ggml_reshape_3d(ctx, beta,  1, C, 1);
    y = ggml_add(ctx, ggml_mul(ctx, y, g), b);
    // Permute back to (F, T, C).
    return ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));
}

// ReLU6: clamp(x, 0, 6). Bounded activation makes INT8 quantisation α
// analytical (= 6/127) and removes outlier-driven scale drift. Used by v1.1.
static inline struct ggml_tensor* build_relu6(struct ggml_context* ctx,
                                              struct ggml_tensor* x) {
    return ggml_clamp(ctx, x, 0.0f, 6.0f);
}

// v1.2 onward uses SiLU (smooth, gradient-everywhere). Pre-norm bounds the
// input so quantisation α stays well-conditioned despite the unbounded
// activation range.
enum class lvq_activation { RELU6, SILU };

static inline struct ggml_tensor* build_act(struct ggml_context* ctx,
                                            struct ggml_tensor* x,
                                            lvq_activation act) {
    if (act == lvq_activation::SILU) {
        return ggml_silu(ctx, x);
    }
    return build_relu6(ctx, x);
}

// v1.1+ streaming encoder block: pre-norm + conv + activation, with a
// residual sub-block that is itself pre-normed:
//   y = act(conv_main(pad(norm_main(x))))              stride 2 in F
//   res = act(conv_res(pad(norm_res(y))))              stride 1
//   out = res + y
// Activation is ReLU6 for version=2 (v1.1) or SiLU for version>=3 (v1.2).
static struct ggml_tensor* build_encoder_block_s_v11(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,
    struct ggml_tensor* norm_w,    struct ggml_tensor* norm_b,
    struct ggml_tensor* conv_w,    struct ggml_tensor* conv_b,
    struct ggml_tensor* res_norm_w,struct ggml_tensor* res_norm_b,
    struct ggml_tensor* res_w,     struct ggml_tensor* res_b,
    lvq_activation act
) {
    struct ggml_tensor* xn = build_causal_groupnorm(ctx, x, norm_w, norm_b);
    struct ggml_tensor* y = build_causal_conv_s(ctx, sg, xn, conv_w, conv_b, 2);
    y = build_act(ctx, y, act);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], y->ne[2]);

    struct ggml_tensor* yn = build_causal_groupnorm(ctx, y, res_norm_w, res_norm_b);
    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, yn, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, res->ne[0], res->ne[1], res->ne[2]);
    res = ggml_add(ctx, build_act(ctx, res, act), y);

    return res;
}

// Streaming S4D bottleneck: single step with persistent complex hidden state.
static struct ggml_tensor* build_bottleneck_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,         // (F, 1, C)
    struct ggml_tensor* in_w,      // input_proj weight: (input_size, H)
    struct ggml_tensor* in_b,      // input_proj bias: (H)
    struct ggml_tensor* out_w,     // output_proj weight: (H, input_size)
    struct ggml_tensor* out_b,     // output_proj bias: (input_size)
    struct ggml_tensor* a_real,    // (H)
    struct ggml_tensor* a_imag,    // (H)
    struct ggml_tensor* B_real,    // (H)
    struct ggml_tensor* B_imag,    // (H)
    struct ggml_tensor* C_real,    // (H)
    struct ggml_tensor* C_imag,    // (H)
    struct ggml_tensor* D          // (input_size)
) {
    int64_t F = x->ne[0], C = x->ne[2];
    int64_t input_size = C * F;
    int64_t H = in_w->ne[1];

    // Flatten: (F, 1, C) → (C*F, 1)
    struct ggml_tensor* flat = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    flat = ggml_reshape_2d(ctx, flat, C * F, 1);

    // Hidden state inputs
    sg.s4d_h_real_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    sg.s4d_h_imag_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_set_input(sg.s4d_h_real_in);
    ggml_set_input(sg.s4d_h_imag_in);

    // Input projection: v = in_w @ flat + in_b → (H, 1)
    struct ggml_tensor* v = ggml_add(ctx,
        ggml_mul_mat(ctx, in_w, flat),
        ggml_reshape_2d(ctx, in_b, H, 1));
    struct ggml_tensor* v1d = ggml_reshape_1d(ctx, v, H);

    // Complex diagonal recurrence
    struct ggml_tensor* hr = sg.s4d_h_real_in;
    struct ggml_tensor* hi = sg.s4d_h_imag_in;

    struct ggml_tensor* term_ar_hr = ggml_mul(ctx, a_real, hr);
    struct ggml_tensor* term_ai_hi = ggml_mul(ctx, a_imag, hi);
    struct ggml_tensor* term_Br_v  = ggml_mul(ctx, B_real, v1d);
    ggml_set_output(term_ar_hr); ggml_set_output(term_ai_hi); ggml_set_output(term_Br_v);
    sg.dbg_taps.push_back(term_ar_hr); sg.dbg_names.push_back("s4d_ar_hr");
    sg.dbg_taps.push_back(term_ai_hi); sg.dbg_names.push_back("s4d_ai_hi");
    sg.dbg_taps.push_back(term_Br_v);  sg.dbg_names.push_back("s4d_Br_v");
    struct ggml_tensor* v_out = ggml_cont(ctx, v1d);
    ggml_set_output(v_out);
    sg.dbg_taps.push_back(v_out); sg.dbg_names.push_back("s4d_v");
    struct ggml_tensor* hr_new = ggml_add(ctx,
        ggml_sub(ctx, term_ar_hr, term_ai_hi),
        term_Br_v);

    struct ggml_tensor* hi_new = ggml_add(ctx,
        ggml_add(ctx,
            ggml_mul(ctx, a_real, hi),
            ggml_mul(ctx, a_imag, hr)),
        ggml_mul(ctx, B_imag, v1d));

    // Output hidden state
    sg.s4d_h_real_out = ggml_cont(ctx, hr_new);
    sg.s4d_h_imag_out = ggml_cont(ctx, hi_new);
    ggml_set_output(sg.s4d_h_real_out);
    ggml_set_output(sg.s4d_h_imag_out);

    // y = C_real*hr_new - C_imag*hi_new: (H)
    struct ggml_tensor* y = ggml_sub(ctx,
        ggml_mul(ctx, C_real, hr_new),
        ggml_mul(ctx, C_imag, hi_new));

    // Output projection + skip: out_w @ y + out_b + D * flat
    struct ggml_tensor* y_2d = ggml_reshape_2d(ctx, y, H, 1);
    struct ggml_tensor* proj = ggml_add(ctx,
        ggml_mul_mat(ctx, out_w, y_2d),
        ggml_reshape_2d(ctx, out_b, input_size, 1));

    struct ggml_tensor* D_2d = ggml_reshape_2d(ctx, D, input_size, 1);
    struct ggml_tensor* fc_out = ggml_add(ctx, proj, ggml_mul(ctx, D_2d, flat));

    // Reshape: (C*F, 1) → (F, C, 1) → permute to (F, 1, C)
    struct ggml_tensor* reshaped = ggml_reshape_3d(ctx, fc_out, F, C, 1);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_permute(ctx, reshaped, 0, 2, 1, 3));

    return out;
}

// Streaming align block: cross-attention with K/ref/smooth histories.
static struct ggml_tensor* build_align_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x_mic,   // (F, 1, C)
    struct ggml_tensor* x_ref,   // (F, 1, C)
    struct ggml_tensor* pmw, struct ggml_tensor* pmb,   // pconv_mic: (1, 1, C, H)
    struct ggml_tensor* prw, struct ggml_tensor* prb,   // pconv_ref: (1, 1, C, H)
    struct ggml_tensor* sw, struct ggml_tensor* sb,     // smooth conv: (3, 5, H, 1)
    int dmax
) {
    int64_t F = x_mic->ne[0], C = x_mic->ne[2];
    int64_t H = pmw->ne[3];

    // 1x1 projections → Q, K: (F, 1, H)
    struct ggml_tensor* mic4 = ggml_reshape_4d(ctx, x_mic, F, 1, C, 1);
    struct ggml_tensor* ref4 = ggml_reshape_4d(ctx, x_ref, F, 1, C, 1);
    struct ggml_tensor* Q4 = ggml_add(ctx, lvq_conv_2d(ctx, pmw, mic4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, pmb, 1,1,H,1));
    struct ggml_tensor* K4 = ggml_add(ctx, lvq_conv_2d(ctx, prw, ref4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, prb, 1,1,H,1));
    struct ggml_tensor* Q = ggml_reshape_3d(ctx, Q4, F, 1, H);
    struct ggml_tensor* K_cur = ggml_reshape_3d(ctx, K4, F, 1, H);

    // K window (A1b): persistent (F, dmax, H); graph writes K_cur into the
    // last row, host shifts rows. Replaces concat+cont+concat per frame.
    struct ggml_tensor* K_win = ggml_new_tensor_3d(sg.win_ctx, GGML_TYPE_F32,
                                                   F, dmax, H);
    sg.conv_win.push_back(K_win);
    struct ggml_tensor* K_full = ggml_set_inplace(ctx, K_win, ggml_is_contiguous(K_cur) ? K_cur : ggml_cont(ctx, K_cur), K_win->nb[1], K_win->nb[2], K_win->nb[3], (size_t)(0) * K_win->nb[0] + (size_t)(dmax - 1) * K_win->nb[1]);

    // Similarity (batched across d): qk[f,d,h] = Q[f,0,h] * K_full[f,d,h].
    // Q's ne[1]=1 broadcasts against K_full's ne[1]=dmax automatically.
    // Then sum over f and scale. Reshape reinterprets the trailing (1, dmax, H)
    // as (dmax, 1, H) — same memory order since ne[0]=1 in the intermediate.
    float scale = 1.0f / std::sqrt((float)F);
    struct ggml_tensor* QK = ggml_mul(ctx, K_full, Q);              // (F, dmax, H)
    struct ggml_tensor* sim_all = ggml_sum_rows(ctx, QK);           // (1, dmax, H)
    sim_all = ggml_scale(ctx, sim_all, scale);
    struct ggml_tensor* V = ggml_reshape_3d(ctx, sim_all, dmax, 1, H);

    // Smooth window (A1b): persistent PRE-PADDED (dmax+2, 5, H); graph writes
    // V into the last row at column offset 1; host shifts rows. Replaces
    // concat+cont+concat AND the per-frame pad.
    struct ggml_tensor* S_win = ggml_new_tensor_3d(sg.win_ctx, GGML_TYPE_F32,
                                                   dmax + 2, 5, H);
    sg.conv_win.push_back(S_win);
    struct ggml_tensor* V_full = ggml_set_inplace(ctx, S_win, ggml_is_contiguous(V) ? V : ggml_cont(ctx, V), S_win->nb[1], S_win->nb[2], S_win->nb[3], (size_t)(1) * S_win->nb[0] + (size_t)(4) * S_win->nb[1]);
    struct ggml_tensor* Vp = ggml_reshape_4d(ctx, V_full, dmax + 2, 5, H, 1);

    // Conv2d: kernel (3, 5, H, 1) → (dmax, 1, 1, 1)
    struct ggml_tensor* Vc = lvq_conv_2d(ctx, sw, Vp, 1,1, 0,0, 1,1);
    struct ggml_tensor* s_bias = ggml_reshape_4d(ctx, sb, 1,1,1,1);
    Vc = ggml_add(ctx, Vc, s_bias);
    Vc = ggml_reshape_2d(ctx, Vc, dmax, 1);

    // Softmax over delay dim
    struct ggml_tensor* attn = ggml_soft_max(ctx, Vc);  // (dmax, 1)

    // Ref window (A1b): persistent (F, dmax, C), newest row last.
    struct ggml_tensor* R_win = ggml_new_tensor_3d(sg.win_ctx, GGML_TYPE_F32,
                                                   F, dmax, C);
    sg.conv_win.push_back(R_win);
    struct ggml_tensor* ref_full = ggml_set_inplace(ctx, R_win, ggml_is_contiguous(x_ref) ? x_ref : ggml_cont(ctx, x_ref), R_win->nb[1], R_win->nb[2], R_win->nb[3], (size_t)(0) * R_win->nb[0] + (size_t)(dmax - 1) * R_win->nb[1]);

    // Weighted sum (batched): aligned[f, 0, c] = sum_d attn[d] * ref_full[f, d, c].
    // Reshape attn from (dmax, 1) → (1, dmax, 1) so it broadcasts across F and C.
    // ggml_sum_rows reduces ne[0] only, so permute dmax into dim 0 first.
    struct ggml_tensor* attn_3d = ggml_reshape_3d(ctx, attn, 1, dmax, 1);
    struct ggml_tensor* weighted = ggml_mul(ctx, ref_full, attn_3d);   // (F, dmax, C)
    struct ggml_tensor* w_perm = ggml_cont(ctx,
        ggml_permute(ctx, weighted, 1, 0, 2, 3));                      // (dmax, F, C)
    struct ggml_tensor* summed = ggml_sum_rows(ctx, w_perm);           // (1, F, C)
    // (1, F, C) → (F, 1, C): identical memory layout since leading dim is 1.
    return ggml_reshape_3d(ctx, summed, F, 1, C);
}

// Streaming decoder block: skip + residual + SubpixelConv + optional BN.
static struct ggml_tensor* build_decoder_block_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,        // (F, 1, C)
    struct ggml_tensor* x_en,     // (F, 1, C) encoder skip
    struct ggml_tensor* skip_w, struct ggml_tensor* skip_b,
    struct ggml_tensor* res_w, struct ggml_tensor* res_b,
    struct ggml_tensor* deconv_w, struct ggml_tensor* deconv_b,
    struct ggml_tensor* bn_scale, struct ggml_tensor* bn_bias,
    bool is_last
) {
    int64_t F = x->ne[0], C = x->ne[2];

    // Skip conv (1x1, no history) — pad channels if weight IC is padded
    struct ggml_tensor* x_en_4d = ggml_reshape_4d(ctx, x_en, F, 1, C, 1);
    if (skip_w->ne[2] > C) {
        x_en_4d = ggml_pad_ext(ctx, x_en_4d, 0, 0, 0, 0, 0, skip_w->ne[2] - C, 0, 0);
    }
    struct ggml_tensor* skip = lvq_conv_2d(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1);
    struct ggml_tensor* sb = ggml_reshape_4d(ctx, skip_b, 1, 1, skip_b->ne[0], 1);
    skip = ggml_add(ctx, skip, sb);
    skip = ggml_reshape_3d(ctx, skip, F, 1, C);

    struct ggml_tensor* y = ggml_add(ctx, x, skip);

    // Residual (causal conv with history)
    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, y, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, F, 1, C);
    res = ggml_add(ctx, ggml_elu(ctx, res), y);

    // SubpixelConv2d (causal conv with history) → pixel shuffle
    int64_t C_out = deconv_w->ne[3] / 2;
    struct ggml_tensor* deconv = build_causal_conv_s(ctx, sg, res, deconv_w, deconv_b, 1);
    deconv = ggml_reshape_3d(ctx, deconv, F, 1, C_out * 2);

    // Pixel shuffle: (F, 1, 2*C_out) → (2*F, 1, C_out)
    struct ggml_tensor* r1 = ggml_reshape_4d(ctx, deconv, F, 1, C_out, 2);
    struct ggml_tensor* r2 = ggml_permute(ctx, r1, 0, 2, 3, 1);
    struct ggml_tensor* shuffled = ggml_reshape_3d(ctx, ggml_cont(ctx, r2), 2 * F, 1, C_out);

    if (!is_last && bn_scale && bn_bias) {
        struct ggml_tensor* sc = ggml_reshape_3d(ctx, bn_scale, 1, 1, C_out);
        struct ggml_tensor* bi = ggml_reshape_3d(ctx, bn_bias, 1, 1, C_out);
        shuffled = ggml_elu(ctx, ggml_add(ctx, ggml_mul(ctx, shuffled, sc), bi));
    }

    return shuffled;
}

// v1.1+ streaming decoder block:
//   skip_proj = skip_conv(skip_norm(x_en))
//   y = x + skip_proj
//   y = y + act(conv_res(norm_res(y)))               ResidualBlock
//   y = pixel_shuffle(conv_dec(norm_dec(y)))         SubpixelConv2d
//   if not last: y = act(y)                          (no ChannelAffine)
// Activation is ReLU6 for version=2 (v1.1) or SiLU for version>=3 (v1.2).
static struct ggml_tensor* build_decoder_block_s_v11(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* x,             // (F, 1, C)
    struct ggml_tensor* x_en,          // (F, 1, C) encoder skip
    struct ggml_tensor* skip_norm_w,   struct ggml_tensor* skip_norm_b,
    struct ggml_tensor* skip_w,        struct ggml_tensor* skip_b,
    struct ggml_tensor* res_norm_w,    struct ggml_tensor* res_norm_b,
    struct ggml_tensor* res_w,         struct ggml_tensor* res_b,
    struct ggml_tensor* deconv_norm_w, struct ggml_tensor* deconv_norm_b,
    struct ggml_tensor* deconv_w,      struct ggml_tensor* deconv_b,
    bool is_last,
    lvq_activation act
) {
    int64_t F = x->ne[0], C = x->ne[2];

    // Pre-norm the encoder skip, then 1x1 project.
    struct ggml_tensor* x_en_n = build_causal_groupnorm(ctx, x_en,
                                                        skip_norm_w, skip_norm_b);
    struct ggml_tensor* x_en_4d = ggml_reshape_4d(ctx, x_en_n, F, 1, C, 1);
    if (skip_w->ne[2] > C) {
        x_en_4d = ggml_pad_ext(ctx, x_en_4d, 0, 0, 0, 0, 0, skip_w->ne[2] - C, 0, 0);
    }
    struct ggml_tensor* skip = lvq_conv_2d(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1);
    struct ggml_tensor* sb = ggml_reshape_4d(ctx, skip_b, 1, 1, skip_b->ne[0], 1);
    skip = ggml_add(ctx, skip, sb);
    skip = ggml_reshape_3d(ctx, skip, F, 1, C);

    struct ggml_tensor* y = ggml_add(ctx, x, skip);

    // ResidualBlock with internal pre-norm.
    struct ggml_tensor* yn = build_causal_groupnorm(ctx, y, res_norm_w, res_norm_b);
    struct ggml_tensor* res = build_causal_conv_s(ctx, sg, yn, res_w, res_b, 1);
    res = ggml_reshape_3d(ctx, res, F, 1, C);
    res = ggml_add(ctx, build_act(ctx, res, act), y);

    // SubpixelConv2d with internal pre-norm.
    int64_t C_out = deconv_w->ne[3] / 2;
    struct ggml_tensor* res_n = build_causal_groupnorm(ctx, res,
                                                       deconv_norm_w, deconv_norm_b);
    struct ggml_tensor* deconv = build_causal_conv_s(ctx, sg, res_n, deconv_w, deconv_b, 1);
    deconv = ggml_reshape_3d(ctx, deconv, F, 1, C_out * 2);

    // Pixel shuffle: (F, 1, 2*C_out) -> (2*F, 1, C_out).
    struct ggml_tensor* r1 = ggml_reshape_4d(ctx, deconv, F, 1, C_out, 2);
    struct ggml_tensor* r2 = ggml_permute(ctx, r1, 0, 2, 3, 1);
    struct ggml_tensor* shuffled = ggml_reshape_3d(ctx, ggml_cont(ctx, r2), 2 * F, 1, C_out);

    if (!is_last) {
        shuffled = build_act(ctx, shuffled, act);
    }
    return shuffled;
}

// Streaming CCM: complex convolving mask with STFT history.
static struct ggml_tensor* build_ccm_s(
    struct ggml_context* ctx,
    dvqe_stream_graph& sg,
    struct ggml_tensor* mask,       // (F, 1, 27)
    struct ggml_tensor* stft_in     // (2, 1, F) — original mic STFT
) {
    int64_t F = mask->ne[0];

    // Permute STFT: (2, 1, F) → (F, 1, 2)
    struct ggml_tensor* stft_cur = ggml_cont(ctx, ggml_permute(ctx, stft_in, 2, 1, 0, 3));

    // CCM window (A1b): persistent PRE-PADDED (F+2, 3, 2); graph writes the
    // current STFT frame into the last row at column offset 1.
    struct ggml_tensor* C_win = ggml_new_tensor_3d(sg.win_ctx, GGML_TYPE_F32,
                                                   F + 2, 3, 2);
    sg.conv_win.push_back(C_win);
    struct ggml_tensor* xp = ggml_set_inplace(ctx, C_win, ggml_is_contiguous(stft_cur) ? stft_cur : ggml_cont(ctx, stft_cur), C_win->nb[1], C_win->nb[2], C_win->nb[3], (size_t)(1) * C_win->nb[0] + (size_t)(2) * C_win->nb[1]);

    // Build H_real, H_imag from mask (same as batch CCM)
    struct ggml_tensor* Hr = nullptr;
    struct ggml_tensor* Hi = nullptr;

    for (int r = 0; r < 3; r++) {
        struct ggml_tensor* m_r = ggml_cont(ctx,
            ggml_view_3d(ctx, mask, F, 1, 9,
                          mask->nb[1], mask->nb[2],
                          r * 9 * mask->nb[2]));
        struct ggml_tensor* scaled_r = ggml_scale(ctx, m_r, CCM_VR[r]);
        struct ggml_tensor* scaled_i = ggml_scale(ctx, m_r, CCM_VI[r]);

        Hr = Hr ? ggml_add(ctx, Hr, scaled_r) : scaled_r;
        Hi = Hi ? ggml_add(ctx, Hi, scaled_i) : scaled_i;
    }
    // Hr, Hi: (F, 1, 9)

    // Complex convolution over 3x3 kernel
    struct ggml_tensor* er = nullptr;
    struct ggml_tensor* ei = nullptr;

    for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
            int ki = m * 3 + n;

            struct ggml_tensor* hr = ggml_cont(ctx,
                ggml_view_3d(ctx, Hr, F, 1, 1,
                              Hr->nb[1], Hr->nb[2],
                              ki * Hr->nb[2]));
            struct ggml_tensor* hi = ggml_cont(ctx,
                ggml_view_3d(ctx, Hi, F, 1, 1,
                              Hi->nb[1], Hi->nb[2],
                              ki * Hi->nb[2]));

            // xp at time=m, freq offset=n: (F, 1, 1)
            struct ggml_tensor* xr = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, 1, 1,
                              xp->nb[1], xp->nb[2],
                              m * xp->nb[1] + n * xp->nb[0]));
            struct ggml_tensor* xi = ggml_cont(ctx,
                ggml_view_3d(ctx, xp, F, 1, 1,
                              xp->nb[1], xp->nb[2],
                              xp->nb[2] + m * xp->nb[1] + n * xp->nb[0]));

            struct ggml_tensor* cr = ggml_sub(ctx, ggml_mul(ctx, hr, xr),
                                               ggml_mul(ctx, hi, xi));
            struct ggml_tensor* ci = ggml_add(ctx, ggml_mul(ctx, hr, xi),
                                               ggml_mul(ctx, hi, xr));

            er = er ? ggml_add(ctx, er, cr) : cr;
            ei = ei ? ggml_add(ctx, ei, ci) : ci;
        }
    }

    // (F, 1, 1) + (F, 1, 1) → (F, 1, 2)
    return ggml_concat(ctx, er, ei, 2);
}

// ── Build/process/reset/free ────────────────────────────────────────────────

bool build_stream_graph(dvqe_graph_model& m, dvqe_stream_graph& sg) {
    auto& hp = m.hparams;
    int F = hp.n_freq_bins;
    const int K = hp.n_fft;  // DCT kernel == n_fft (512); 2*F must equal K
    if (K != 2 * F) {
        fprintf(stderr, "build_stream_graph: n_fft (%d) != 2*n_freq_bins (%d)\n", K, 2 * F);
        return false;
    }

    // Allocate context for streaming graph (T=1 per frame)
    sg.ctx = make_ctx(64 * 1024 * 1024);
    auto* ctx = sg.ctx;
    // Persistent conv-window tensors (see header): own context, allocated on
    // the backend right before gallocr runs.
    {
        struct ggml_init_params wp = { ggml_tensor_overhead() * 64, nullptr, true };
        sg.win_ctx = ggml_init(wp);
    }

    // DCT-II analysis head: (512,) PCM window → (2, 1, F) STFT-compatible frame.
    // Encoder weight is loaded from GGUF with pytorch shape (out=K, 1, kernel=K);
    // ggml gets ne=(K, 1, K). Squeeze the singleton middle dim as a view.
    struct ggml_tensor* W_enc = m.w("encoder.conv.weight");
    struct ggml_tensor* W_dec = m.w("decoder.linear.weight");
    if (!W_enc || !W_dec) {
        fprintf(stderr, "GGUF missing encoder.conv.weight / decoder.linear.weight\n");
        return false;
    }
    struct ggml_tensor* W_enc_2d = ggml_reshape_2d(ctx, W_enc, K, K);

    sg.mic_pcm_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    sg.ref_pcm_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_set_input(sg.mic_pcm_in);
    ggml_set_input(sg.ref_pcm_in);

    // Pytorch conv output filter index 2*f+c corresponds to bin f, channel c;
    // ggml tensor with ne=(2, 1, F) has element (c, 0, f) at offset 2*f+c,
    // matching exactly.
    // Keep the post-analysis spectra alive after compute: tap_debug reads them
    // for PyTorch parity verification. The output flag must sit on the actual
    // compute node (mul_mat result) — a flag on the reshape VIEW does not stop
    // the gallocr from reusing the underlying buffer mid-graph.
    struct ggml_tensor* mic_mm = ggml_mul_mat(ctx, W_enc_2d, sg.mic_pcm_in);
    struct ggml_tensor* ref_mm = ggml_mul_mat(ctx, W_enc_2d, sg.ref_pcm_in);
    ggml_set_output(mic_mm);
    ggml_set_output(ref_mm);
    sg.mic_in = ggml_reshape_3d(ctx, mic_mm, 2, 1, F);
    sg.ref_in = ggml_reshape_3d(ctx, ref_mm, 2, 1, F);
    ggml_set_output(sg.mic_in);
    ggml_set_output(sg.ref_in);

    // 1. Feature extraction (no time dependency)
    struct ggml_tensor* mic_fe = build_fe(ctx, sg.mic_in, hp.power_law_c);
    struct ggml_tensor* ref_fe = build_fe(ctx, sg.ref_in, hp.power_law_c);

    // 2-3. Encoder blocks.
    //
    // v1   (version=1): post-conv BN folded into conv weights, ELU activation.
    //                   Builder takes only main + residual conv tensors.
    // v1.1 (version=2): pre-norm CausalGroupNorm + ReLU6. Builder takes two
    //                   extra norm tensor pairs (one for main, one for res).
    // v1.2 (version=3): same structure as v1.1, SiLU activation.
    const bool is_v11 = (m.hparams.version >= 2);
    const lvq_activation act = (m.hparams.version >= 3)
        ? lvq_activation::SILU
        : lvq_activation::RELU6;
    auto enc = [&](struct ggml_tensor* x, const char* prefix) -> struct ggml_tensor* {
        std::string p = prefix;
        if (is_v11) {
            return build_encoder_block_s_v11(ctx, sg, x,
                m.w((p + ".norm.weight").c_str()),
                m.w((p + ".norm.bias").c_str()),
                m.w((p + ".conv.weight").c_str()),
                m.w((p + ".conv.bias").c_str()),
                m.w((p + ".resblock.norm.weight").c_str()),
                m.w((p + ".resblock.norm.bias").c_str()),
                m.w((p + ".resblock.conv.weight").c_str()),
                m.w((p + ".resblock.conv.bias").c_str()),
                act);
        }
        return build_encoder_block_s(ctx, sg, x,
            m.w((p + ".conv.weight").c_str()),
            m.w((p + ".conv.bias").c_str()),
            m.w((p + ".resblock.conv.weight").c_str()),
            m.w((p + ".resblock.conv.bias").c_str()));
    };

    // Debug taps for parity verification (tap_debug): the output flag keeps
    // each block-boundary tensor's buffer alive after compute.
    auto dbg = [&](struct ggml_tensor* t, const char* nm) {
        ggml_set_output(t);
        sg.dbg_taps.push_back(t);
        sg.dbg_names.push_back(nm);
        return t;
    };
    dbg(mic_fe, "fe_mic"); dbg(ref_fe, "fe_far");

    struct ggml_tensor* mic_e1 = dbg(enc(mic_fe, "mic_enc1"), "mic_e1");
    struct ggml_tensor* mic_e2 = dbg(enc(mic_e1, "mic_enc2"), "mic_e2");
    struct ggml_tensor* far_e1 = dbg(enc(ref_fe, "far_enc1"), "far_e1");
    struct ggml_tensor* far_e2 = dbg(enc(far_e1, "far_enc2"), "far_e2");

    // 4. Alignment
    struct ggml_tensor* aligned = dbg(build_align_s(ctx, sg, mic_e2, far_e2,
        m.w("align.pconv_mic.weight"), m.w("align.pconv_mic.bias"),
        m.w("align.pconv_ref.weight"), m.w("align.pconv_ref.bias"),
        m.w("align.conv.1.weight"), m.w("align.conv.1.bias"),
        hp.dmax), "aligned");

    // 5. Concat + encoder 3-5
    struct ggml_tensor* cat = build_concat_channels(ctx, mic_e2, aligned);
    struct ggml_tensor* mic_e3 = dbg(enc(cat,    "mic_enc3"), "mic_e3");
    struct ggml_tensor* mic_e4 = dbg(enc(mic_e3, "mic_enc4"), "mic_e4");
    struct ggml_tensor* mic_e5 = dbg(enc(mic_e4, "mic_enc5"), "mic_e5");

    // 6. S4D Bottleneck (single step)
    struct ggml_tensor* bn = dbg(build_bottleneck_s(ctx, sg, mic_e5,
        m.w("bottleneck.input_proj.weight"), m.w("bottleneck.input_proj.bias"),
        m.w("bottleneck.output_proj.weight"), m.w("bottleneck.output_proj.bias"),
        m.w("bottleneck.a_real"), m.w("bottleneck.a_imag"),
        m.w("bottleneck.B_real"), m.w("bottleneck.B_imag"),
        m.w("bottleneck.C_real"), m.w("bottleneck.C_imag"),
        m.w("bottleneck.D")), "bottleneck");

    // 7. Decoder with skip connections + frequency trimming
    auto dec = [&](struct ggml_tensor* x, struct ggml_tensor* x_en,
                   const char* prefix, bool is_last) -> struct ggml_tensor* {
        std::string p = prefix;
        if (is_v11) {
            return build_decoder_block_s_v11(ctx, sg, x, x_en,
                m.w((p + ".skip_norm.weight").c_str()),
                m.w((p + ".skip_norm.bias").c_str()),
                m.w((p + ".skip_conv.weight").c_str()),
                m.w((p + ".skip_conv.bias").c_str()),
                m.w((p + ".resblock.norm.weight").c_str()),
                m.w((p + ".resblock.norm.bias").c_str()),
                m.w((p + ".resblock.conv.weight").c_str()),
                m.w((p + ".resblock.conv.bias").c_str()),
                m.w((p + ".deconv.norm.weight").c_str()),
                m.w((p + ".deconv.norm.bias").c_str()),
                m.w((p + ".deconv.conv.weight").c_str()),
                m.w((p + ".deconv.conv.bias").c_str()),
                is_last, act);
        }
        return build_decoder_block_s(ctx, sg, x, x_en,
            m.w((p + ".skip_conv.weight").c_str()),
            m.w((p + ".skip_conv.bias").c_str()),
            m.w((p + ".resblock.conv.weight").c_str()),
            m.w((p + ".resblock.conv.bias").c_str()),
            m.w((p + ".deconv.conv.weight").c_str()),
            m.w((p + ".deconv.conv.bias").c_str()),
            is_last ? nullptr : m.w((p + ".bn.scale").c_str()),
            is_last ? nullptr : m.w((p + ".bn.bias").c_str()),
            is_last);
    };

    struct ggml_tensor* d5 = dbg(dec(bn, mic_e5, "dec5", false), "d5");
    d5 = build_freq_trim(ctx, d5, mic_e4->ne[0]);
    struct ggml_tensor* d4 = dbg(dec(d5, mic_e4, "dec4", false), "d4");
    d4 = build_freq_trim(ctx, d4, mic_e3->ne[0]);
    struct ggml_tensor* d3 = dbg(dec(d4, mic_e3, "dec3", false), "d3");
    d3 = build_freq_trim(ctx, d3, mic_e2->ne[0]);
    struct ggml_tensor* d2 = dbg(dec(d3, mic_e2, "dec2", false), "d2");
    d2 = build_freq_trim(ctx, d2, mic_e1->ne[0]);
    struct ggml_tensor* d1 = dbg(dec(d2, mic_e1, "dec1", true), "d1");
    d1 = build_freq_trim(ctx, d1, mic_fe->ne[0]);

    // 8. CCM
    struct ggml_tensor* ccm_out = build_ccm_s(ctx, sg, d1, sg.mic_in);
    // ccm_out: (F, 1, 2) → permute to (2, 1, F) to match I/O format
    sg.enhanced_out = ggml_cont(ctx,
        ggml_permute(ctx, ccm_out, 2, 1, 0, 3));
    ggml_set_output(sg.enhanced_out);

    // DCT-II synthesis tail: (2, 1, F) → (512,) PCM frame for caller OLA.
    // enhanced_out memory is [f0_c0, f0_c1, f1_c0, f1_c1, ...] so the 1-D
    // reshape directly matches pytorch's (F,2) → (2F,) flatten.
    struct ggml_tensor* enh_flat = ggml_reshape_1d(ctx, sg.enhanced_out, K);
    sg.enhanced_pcm_out = ggml_mul_mat(ctx, W_dec, enh_flat);
    ggml_set_output(sg.enhanced_pcm_out);

    // Build graph with all outputs
    sg.graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(sg.graph, sg.enhanced_pcm_out);
    // Add all history outputs to graph
    for (auto* h : sg.conv_hist_out)
        ggml_build_forward_expand(sg.graph, h);
    if (sg.s4d_h_real_out)
        ggml_build_forward_expand(sg.graph, sg.s4d_h_real_out);
    if (sg.s4d_h_imag_out)
        ggml_build_forward_expand(sg.graph, sg.s4d_h_imag_out);
    if (sg.align_K_hist_out)
        ggml_build_forward_expand(sg.graph, sg.align_K_hist_out);
    if (sg.align_ref_hist_out)
        ggml_build_forward_expand(sg.graph, sg.align_ref_hist_out);
    if (sg.align_smooth_hist_out)
        ggml_build_forward_expand(sg.graph, sg.align_smooth_hist_out);
    if (sg.ccm_hist_out)
        ggml_build_forward_expand(sg.graph, sg.ccm_hist_out);
    for (auto* t : sg.dbg_taps)
        ggml_build_forward_expand(sg.graph, t);

    // Allocate
    sg.win_buf = ggml_backend_alloc_ctx_tensors(sg.win_ctx, m.backend);
    sg.galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(m.backend));
    if (!ggml_gallocr_alloc_graph(sg.galloc, sg.graph)) {
        fprintf(stderr, "ERROR: stream graph allocation failed\n");
        free_stream_graph(sg);
        return false;
    }

    // Size persistent scratch to the largest history tensor
    size_t max_hist_bytes = 0;
    for (auto* h : sg.conv_hist_out)
        max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(h));
    if (sg.s4d_h_real_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.s4d_h_real_out));
    if (sg.s4d_h_imag_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.s4d_h_imag_out));
    if (sg.align_K_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.align_K_hist_out));
    if (sg.align_ref_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.align_ref_hist_out));
    if (sg.align_smooth_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.align_smooth_hist_out));
    if (sg.ccm_hist_out) max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(sg.ccm_hist_out));
    for (auto* w : sg.conv_win)
        max_hist_bytes = std::max(max_hist_bytes, ggml_nbytes(w));
    sg.hist_scratch.resize(max_hist_bytes, 0);

    // Zero all history
    reset_stream_graph(sg, m);

    return true;
}

void process_frame_graph(dvqe_stream_graph& sg, dvqe_graph_model& m,
                         const float* mic_pcm_window,
                         const float* ref_pcm_window,
                         float* enhanced_pcm_window) {
    const int K = m.hparams.n_fft;  // DCT kernel / PCM window size (512)

    ggml_backend_tensor_set(sg.mic_pcm_in, mic_pcm_window, 0, K * sizeof(float));
    ggml_backend_tensor_set(sg.ref_pcm_in, ref_pcm_window, 0, K * sizeof(float));

    ggml_backend_graph_compute(m.backend, sg.graph);

    ggml_backend_tensor_get(sg.enhanced_pcm_out, enhanced_pcm_window, 0, K * sizeof(float));

    // Update histories: copy each output back to corresponding input.
    // ggml_gallocr places persistent in/out tensors at overlapping offsets
    // inside its reuse pool; ggml_backend_tensor_copy reduces to memcpy on
    // host backends, which is UB on overlap. Use memmove directly on host
    // buffers (single copy, overlap-safe); fall back to scratch for
    // non-host backends where ->data isn't dereferenceable.
    // TWO-PHASE copy: the gallocr may place ANY in/out pair at overlapping
    // offsets in its reuse pool, and the overlap is not necessarily between a
    // tensor and its own counterpart — writing input i can clobber output j's
    // bytes before j's copy runs (layout- and shape-dependent; observed in
    // practice on S4D state and on the v1.1 graph). Snapshot every output
    // into the scratch arena first, then write all inputs.
    struct pair_t { struct ggml_tensor* in; struct ggml_tensor* out; size_t off; };
    static thread_local std::vector<pair_t> pairs;
    pairs.clear();
    size_t off = 0;
    auto stage = [&](struct ggml_tensor* in, struct ggml_tensor* out) {
        if (!in || !out) return;
        pairs.push_back({in, out, off});
        off += ggml_nbytes(out);
    };
    for (size_t i = 0; i < sg.conv_hist_in.size(); i++)
        stage(sg.conv_hist_in[i], sg.conv_hist_out[i]);
    stage(sg.s4d_h_real_in, sg.s4d_h_real_out);
    stage(sg.s4d_h_imag_in, sg.s4d_h_imag_out);
    stage(sg.align_K_hist_in, sg.align_K_hist_out);
    stage(sg.align_ref_hist_in, sg.align_ref_hist_out);
    stage(sg.align_smooth_hist_in, sg.align_smooth_hist_out);
    stage(sg.ccm_hist_in, sg.ccm_hist_out);
    if (sg.hist_scratch.size() < off) sg.hist_scratch.resize(off);
    for (auto& p : pairs)
        ggml_backend_tensor_get(p.out, sg.hist_scratch.data() + p.off, 0,
                                ggml_nbytes(p.out));
    for (auto& p : pairs)
        ggml_backend_tensor_set(p.in, sg.hist_scratch.data() + p.off, 0,
                                ggml_nbytes(p.out));

    // A1 conv windows: shift rows up by one (row r <- r+1); the graph wrote
    // the current frame into the last row during compute, so after the shift
    // rows 0..kh-2 hold the kh-1 newest frames and the last row is a stale
    // copy that the next frame's set_inplace overwrites.
    for (auto* w : sg.conv_win) {
        const size_t row = (size_t)w->nb[1];
        const int64_t kh = w->ne[1], C = w->ne[2];
        if (ggml_backend_buffer_is_host(w->buffer)) {
            char* base = (char*)w->data;
            for (int64_t c = 0; c < C; c++)
                std::memmove(base + (size_t)c * w->nb[2],
                             base + (size_t)c * w->nb[2] + row,
                             (size_t)(kh - 1) * row);
        } else {
            size_t n = ggml_nbytes(w);
            if (sg.hist_scratch.size() < n) sg.hist_scratch.resize(n);
            ggml_backend_tensor_get(w, sg.hist_scratch.data(), 0, n);
            char* base = (char*)sg.hist_scratch.data();
            for (int64_t c = 0; c < C; c++)
                std::memmove(base + (size_t)c * w->nb[2],
                             base + (size_t)c * w->nb[2] + row,
                             (size_t)(kh - 1) * row);
            ggml_backend_tensor_set(w, sg.hist_scratch.data(), 0, n);
        }
    }
}

void reset_stream_graph(dvqe_stream_graph& sg, dvqe_graph_model& m) {
    (void)m;
    // Zero the scratch buffer once, then use it to zero all history tensors.
    std::memset(sg.hist_scratch.data(), 0, sg.hist_scratch.size());

    auto zero_tensor = [&sg](struct ggml_tensor* t) {
        if (!t) return;
        ggml_backend_tensor_set(t, sg.hist_scratch.data(), 0, ggml_nbytes(t));
    };

    for (auto* h : sg.conv_hist_in)
        zero_tensor(h);
    for (auto* w : sg.conv_win)
        zero_tensor(w);   // also restores permanent zero pad columns
    zero_tensor(sg.s4d_h_real_in);
    zero_tensor(sg.s4d_h_imag_in);
    zero_tensor(sg.align_K_hist_in);
    zero_tensor(sg.align_ref_hist_in);
    zero_tensor(sg.align_smooth_hist_in);
    zero_tensor(sg.ccm_hist_in);
}

void print_memory_budget(const dvqe_graph_model& m, const dvqe_stream_graph& sg) {
    auto kib = [](size_t b) { return (double)b / 1024.0; };
    size_t weights = m.weight_buf ? ggml_backend_buffer_get_size(m.weight_buf) : 0;
    size_t acts    = sg.galloc ? ggml_gallocr_get_buffer_size(sg.galloc, 0) : 0;
    size_t scratch = sg.hist_scratch.size();

    // Sum of persistent history tensor sizes (double-copied per frame today).
    size_t hist_total = 0;
    auto add_hist = [&](const struct ggml_tensor* t) {
        if (t) hist_total += ggml_nbytes(t);
    };
    for (auto* t : sg.conv_hist_in) add_hist(t);
    for (auto* t : sg.conv_hist_out) add_hist(t);
    add_hist(sg.s4d_h_real_in);       add_hist(sg.s4d_h_real_out);
    add_hist(sg.s4d_h_imag_in);       add_hist(sg.s4d_h_imag_out);
    add_hist(sg.align_K_hist_in);     add_hist(sg.align_K_hist_out);
    add_hist(sg.align_ref_hist_in);   add_hist(sg.align_ref_hist_out);
    add_hist(sg.align_smooth_hist_in);add_hist(sg.align_smooth_hist_out);
    add_hist(sg.ccm_hist_in);         add_hist(sg.ccm_hist_out);

    printf("Memory budget:\n");
    printf("  weights buffer:    %8.1f KiB\n", kib(weights));
    printf("  activation buffer: %8.1f KiB\n", kib(acts));
    printf("  history scratch:   %8.1f KiB  (one-shot, reused)\n", kib(scratch));
    printf("  history tensors:   %8.1f KiB  (in+out, per-frame memcpy)\n", kib(hist_total));
    printf("  total resident:    %8.1f KiB\n", kib(weights + acts + scratch));
}

void print_op_histogram(const struct ggml_cgraph* graph) {
    if (!graph) return;
    int n = ggml_graph_n_nodes((struct ggml_cgraph*)graph);
    std::map<std::string, int> counts;
    for (int i = 0; i < n; i++) {
        struct ggml_tensor* node = ggml_graph_node((struct ggml_cgraph*)graph, i);
        counts[ggml_op_name(node->op)]++;
    }
    std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    printf("Graph ops (%d nodes):\n", n);
    for (const auto& [name, cnt] : sorted)
        printf("  %-20s %4d\n", name.c_str(), cnt);
}

void free_stream_graph(dvqe_stream_graph& sg) {
    if (sg.win_buf) { ggml_backend_buffer_free(sg.win_buf); sg.win_buf = nullptr; }
    if (sg.win_ctx) { ggml_free(sg.win_ctx); sg.win_ctx = nullptr; }
    if (sg.galloc) { ggml_gallocr_free(sg.galloc); sg.galloc = nullptr; }
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.conv_hist_in.clear();
    sg.conv_hist_out.clear();
    sg.graph = nullptr;
    sg.mic_pcm_in = sg.ref_pcm_in = sg.enhanced_pcm_out = nullptr;
    sg.mic_in = sg.ref_in = sg.enhanced_out = nullptr;
    sg.s4d_h_real_in = sg.s4d_h_real_out = nullptr;
    sg.s4d_h_imag_in = sg.s4d_h_imag_out = nullptr;
    sg.align_K_hist_in = sg.align_K_hist_out = nullptr;
    sg.align_ref_hist_in = sg.align_ref_hist_out = nullptr;
    sg.align_smooth_hist_in = sg.align_smooth_hist_out = nullptr;
    sg.ccm_hist_in = sg.ccm_hist_out = nullptr;
}
