/**
 * LocalVQE inference using GGML computational graphs.
 *
 * Each network block builds a small GGML graph that is dispatched
 * to the selected backend (CPU with SIMD, or CUDA).
 */

#include "localvqe_graph.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

// ── Helpers ───────────────────────────────────────────────────────────────

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

    struct ggml_tensor* conv = ggml_conv_2d(ctx, weight, padded,
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
    struct ggml_tensor* skip = ggml_conv_2d(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1);
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
    struct ggml_tensor* Q4 = ggml_add(ctx, ggml_conv_2d(ctx, pmw, mic4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, pmb, 1,1,H,1));
    struct ggml_tensor* K4 = ggml_add(ctx, ggml_conv_2d(ctx, prw, ref4, 1,1,0,0,1,1),
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
    struct ggml_tensor* Vc = ggml_conv_2d(ctx, sw, Vp, 1,1, 0,0, 1,1);
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
    std::call_once(f, ggml_backend_load_all);
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
        n_threads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
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

// Causal conv for streaming: concat history + current frame, then conv.
// Pushes history in/out tensors to sg.conv_hist_in/conv_hist_out.
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
    int hist_T = kH - 1;  // number of history frames needed

    // Causal padding for freq dimension
    int pad_left  = (kW - 1) / 2;
    int pad_right = kW - 1 - pad_left;

    // History input: last hist_T frames (F, hist_T, C_in)
    struct ggml_tensor* hist = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, hist_T, C_in);
    ggml_set_input(hist);
    sg.conv_hist_in.push_back(hist);

    // Concat: (F, hist_T, C_in) + (F, 1, C_in) → (F, kH, C_in)
    struct ggml_tensor* cat = ggml_concat(ctx, hist, x, 1);

    // New history = [hist[1:], x] — shift and append current frame
    struct ggml_tensor* hist_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, hist, F, hist_T - 1, C_in,
                      hist->nb[1], hist->nb[2],
                      1 * hist->nb[1]));
    struct ggml_tensor* new_hist = ggml_concat(ctx, hist_tail, x, 1);
    ggml_set_output(new_hist);
    sg.conv_hist_out.push_back(new_hist);

    // Reshape to 4D, pad freq only, conv
    int64_t C_w = weight->ne[2];  // weight IC (may be padded for quantization)
    struct ggml_tensor* x4d = ggml_reshape_4d(ctx, cat, F, kH, C_in, 1);
    if (C_w > C_in) {
        x4d = ggml_pad_ext(ctx, x4d, 0, 0, 0, 0, 0, C_w - C_in, 0, 0);
    }
    struct ggml_tensor* padded = ggml_pad_ext(ctx, x4d,
                                               pad_left, pad_right,  // freq
                                               0, 0,                 // time: history provides
                                               0, 0, 0, 0);
    struct ggml_tensor* conv = ggml_conv_2d(ctx, weight, padded,
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
    struct ggml_tensor* Q4 = ggml_add(ctx, ggml_conv_2d(ctx, pmw, mic4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, pmb, 1,1,H,1));
    struct ggml_tensor* K4 = ggml_add(ctx, ggml_conv_2d(ctx, prw, ref4, 1,1,0,0,1,1),
                                       ggml_reshape_4d(ctx, prb, 1,1,H,1));
    struct ggml_tensor* Q = ggml_reshape_3d(ctx, Q4, F, 1, H);
    struct ggml_tensor* K_cur = ggml_reshape_3d(ctx, K4, F, 1, H);

    // K history: (F, dmax-1, H)
    sg.align_K_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, dmax - 1, H);
    ggml_set_input(sg.align_K_hist_in);

    // K_full = concat(K_hist, K_cur) → (F, dmax, H)
    struct ggml_tensor* K_full = ggml_concat(ctx, sg.align_K_hist_in, K_cur, 1);

    // New K history = [K_hist[1:], K_cur] — built from inputs
    struct ggml_tensor* K_hist_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.align_K_hist_in, F, dmax - 2, H,
                      sg.align_K_hist_in->nb[1], sg.align_K_hist_in->nb[2],
                      1 * sg.align_K_hist_in->nb[1]));
    sg.align_K_hist_out = ggml_concat(ctx, K_hist_tail, K_cur, 1);
    ggml_set_output(sg.align_K_hist_out);

    // Similarity (batched across d): qk[f,d,h] = Q[f,0,h] * K_full[f,d,h].
    // Q's ne[1]=1 broadcasts against K_full's ne[1]=dmax automatically.
    // Then sum over f and scale. Reshape reinterprets the trailing (1, dmax, H)
    // as (dmax, 1, H) — same memory order since ne[0]=1 in the intermediate.
    float scale = 1.0f / std::sqrt((float)F);
    struct ggml_tensor* QK = ggml_mul(ctx, K_full, Q);              // (F, dmax, H)
    struct ggml_tensor* sim_all = ggml_sum_rows(ctx, QK);           // (1, dmax, H)
    sim_all = ggml_scale(ctx, sim_all, scale);
    struct ggml_tensor* V = ggml_reshape_3d(ctx, sim_all, dmax, 1, H);

    // Smooth conv with history
    sg.align_smooth_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dmax, 4, H);
    ggml_set_input(sg.align_smooth_hist_in);

    // V_full = concat(smooth_hist, V) → (dmax, 5, H)
    struct ggml_tensor* V_full = ggml_concat(ctx, sg.align_smooth_hist_in, V, 1);

    // New smooth history = [smooth_hist[1:], V] — built from inputs
    struct ggml_tensor* smooth_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.align_smooth_hist_in, dmax, 3, H,
                      sg.align_smooth_hist_in->nb[1], sg.align_smooth_hist_in->nb[2],
                      1 * sg.align_smooth_hist_in->nb[1]));
    sg.align_smooth_hist_out = ggml_concat(ctx, smooth_tail, V, 1);
    ggml_set_output(sg.align_smooth_hist_out);

    // Pad freq(dmax) +1/+1, no time padding → (dmax+2, 5, H, 1)
    struct ggml_tensor* Vp = ggml_reshape_4d(ctx, V_full, dmax, 5, H, 1);
    Vp = ggml_pad_ext(ctx, Vp, 1,1, 0,0, 0,0, 0,0);

    // Conv2d: kernel (3, 5, H, 1) → (dmax, 1, 1, 1)
    struct ggml_tensor* Vc = ggml_conv_2d(ctx, sw, Vp, 1,1, 0,0, 1,1);
    struct ggml_tensor* s_bias = ggml_reshape_4d(ctx, sb, 1,1,1,1);
    Vc = ggml_add(ctx, Vc, s_bias);
    Vc = ggml_reshape_2d(ctx, Vc, dmax, 1);

    // Softmax over delay dim
    struct ggml_tensor* attn = ggml_soft_max(ctx, Vc);  // (dmax, 1)

    // Ref history: (F, dmax-1, C)
    sg.align_ref_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, dmax - 1, C);
    ggml_set_input(sg.align_ref_hist_in);

    // ref_full = concat(ref_hist, x_ref) → (F, dmax, C)
    struct ggml_tensor* ref_full = ggml_concat(ctx, sg.align_ref_hist_in, x_ref, 1);

    // New ref history = [ref_hist[1:], x_ref] — built from inputs
    struct ggml_tensor* ref_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.align_ref_hist_in, F, dmax - 2, C,
                      sg.align_ref_hist_in->nb[1], sg.align_ref_hist_in->nb[2],
                      1 * sg.align_ref_hist_in->nb[1]));
    sg.align_ref_hist_out = ggml_concat(ctx, ref_tail, x_ref, 1);
    ggml_set_output(sg.align_ref_hist_out);

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
    struct ggml_tensor* skip = ggml_conv_2d(ctx, skip_w, x_en_4d, 1, 1, 0, 0, 1, 1);
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

    // STFT history: (F, 2, 2)
    sg.ccm_hist_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, 2, 2);
    ggml_set_input(sg.ccm_hist_in);

    // Concat: (F, 2, 2) + (F, 1, 2) → (F, 3, 2)
    struct ggml_tensor* stft_full = ggml_concat(ctx, sg.ccm_hist_in, stft_cur, 1);

    // New history = [ccm_hist[1:], stft_cur] — built from inputs
    struct ggml_tensor* ccm_tail = ggml_cont(ctx,
        ggml_view_3d(ctx, sg.ccm_hist_in, F, 1, 2,
                      sg.ccm_hist_in->nb[1], sg.ccm_hist_in->nb[2],
                      1 * sg.ccm_hist_in->nb[1]));
    sg.ccm_hist_out = ggml_concat(ctx, ccm_tail, stft_cur, 1);
    ggml_set_output(sg.ccm_hist_out);

    // Pad freq: (F, 3, 2) → (F+2, 3, 2)
    struct ggml_tensor* xp = ggml_pad_ext(ctx, stft_full, 1,1, 0,0, 0,0, 0,0);

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
    sg.mic_in = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W_enc_2d, sg.mic_pcm_in), 2, 1, F);
    sg.ref_in = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W_enc_2d, sg.ref_pcm_in), 2, 1, F);

    // 1. Feature extraction (no time dependency)
    struct ggml_tensor* mic_fe = build_fe(ctx, sg.mic_in, hp.power_law_c);
    struct ggml_tensor* ref_fe = build_fe(ctx, sg.ref_in, hp.power_law_c);

    // 2. Mic encoder 1-2
    struct ggml_tensor* mic_e1 = build_encoder_block_s(ctx, sg, mic_fe,
        m.w("mic_enc1.conv.weight"), m.w("mic_enc1.conv.bias"),
        m.w("mic_enc1.resblock.conv.weight"), m.w("mic_enc1.resblock.conv.bias"));

    struct ggml_tensor* mic_e2 = build_encoder_block_s(ctx, sg, mic_e1,
        m.w("mic_enc2.conv.weight"), m.w("mic_enc2.conv.bias"),
        m.w("mic_enc2.resblock.conv.weight"), m.w("mic_enc2.resblock.conv.bias"));

    // 3. Far-end encoder 1-2
    struct ggml_tensor* far_e1 = build_encoder_block_s(ctx, sg, ref_fe,
        m.w("far_enc1.conv.weight"), m.w("far_enc1.conv.bias"),
        m.w("far_enc1.resblock.conv.weight"), m.w("far_enc1.resblock.conv.bias"));

    struct ggml_tensor* far_e2 = build_encoder_block_s(ctx, sg, far_e1,
        m.w("far_enc2.conv.weight"), m.w("far_enc2.conv.bias"),
        m.w("far_enc2.resblock.conv.weight"), m.w("far_enc2.resblock.conv.bias"));

    // 4. Alignment
    struct ggml_tensor* aligned = build_align_s(ctx, sg, mic_e2, far_e2,
        m.w("align.pconv_mic.weight"), m.w("align.pconv_mic.bias"),
        m.w("align.pconv_ref.weight"), m.w("align.pconv_ref.bias"),
        m.w("align.conv.1.weight"), m.w("align.conv.1.bias"),
        hp.dmax);

    // 5. Concat + encoder 3-5
    struct ggml_tensor* cat = build_concat_channels(ctx, mic_e2, aligned);
    struct ggml_tensor* mic_e3 = build_encoder_block_s(ctx, sg, cat,
        m.w("mic_enc3.conv.weight"), m.w("mic_enc3.conv.bias"),
        m.w("mic_enc3.resblock.conv.weight"), m.w("mic_enc3.resblock.conv.bias"));

    struct ggml_tensor* mic_e4 = build_encoder_block_s(ctx, sg, mic_e3,
        m.w("mic_enc4.conv.weight"), m.w("mic_enc4.conv.bias"),
        m.w("mic_enc4.resblock.conv.weight"), m.w("mic_enc4.resblock.conv.bias"));

    struct ggml_tensor* mic_e5 = build_encoder_block_s(ctx, sg, mic_e4,
        m.w("mic_enc5.conv.weight"), m.w("mic_enc5.conv.bias"),
        m.w("mic_enc5.resblock.conv.weight"), m.w("mic_enc5.resblock.conv.bias"));

    // 6. S4D Bottleneck (single step)
    struct ggml_tensor* bn = build_bottleneck_s(ctx, sg, mic_e5,
        m.w("bottleneck.input_proj.weight"), m.w("bottleneck.input_proj.bias"),
        m.w("bottleneck.output_proj.weight"), m.w("bottleneck.output_proj.bias"),
        m.w("bottleneck.a_real"), m.w("bottleneck.a_imag"),
        m.w("bottleneck.B_real"), m.w("bottleneck.B_imag"),
        m.w("bottleneck.C_real"), m.w("bottleneck.C_imag"),
        m.w("bottleneck.D"));

    // 7. Decoder with skip connections + frequency trimming
    struct ggml_tensor* d5 = build_decoder_block_s(ctx, sg, bn, mic_e5,
        m.w("dec5.skip_conv.weight"), m.w("dec5.skip_conv.bias"),
        m.w("dec5.resblock.conv.weight"), m.w("dec5.resblock.conv.bias"),
        m.w("dec5.deconv.conv.weight"), m.w("dec5.deconv.conv.bias"),
        m.w("dec5.bn.scale"), m.w("dec5.bn.bias"), false);
    d5 = build_freq_trim(ctx, d5, mic_e4->ne[0]);

    struct ggml_tensor* d4 = build_decoder_block_s(ctx, sg, d5, mic_e4,
        m.w("dec4.skip_conv.weight"), m.w("dec4.skip_conv.bias"),
        m.w("dec4.resblock.conv.weight"), m.w("dec4.resblock.conv.bias"),
        m.w("dec4.deconv.conv.weight"), m.w("dec4.deconv.conv.bias"),
        m.w("dec4.bn.scale"), m.w("dec4.bn.bias"), false);
    d4 = build_freq_trim(ctx, d4, mic_e3->ne[0]);

    struct ggml_tensor* d3 = build_decoder_block_s(ctx, sg, d4, mic_e3,
        m.w("dec3.skip_conv.weight"), m.w("dec3.skip_conv.bias"),
        m.w("dec3.resblock.conv.weight"), m.w("dec3.resblock.conv.bias"),
        m.w("dec3.deconv.conv.weight"), m.w("dec3.deconv.conv.bias"),
        m.w("dec3.bn.scale"), m.w("dec3.bn.bias"), false);
    d3 = build_freq_trim(ctx, d3, mic_e2->ne[0]);

    struct ggml_tensor* d2 = build_decoder_block_s(ctx, sg, d3, mic_e2,
        m.w("dec2.skip_conv.weight"), m.w("dec2.skip_conv.bias"),
        m.w("dec2.resblock.conv.weight"), m.w("dec2.resblock.conv.bias"),
        m.w("dec2.deconv.conv.weight"), m.w("dec2.deconv.conv.bias"),
        m.w("dec2.bn.scale"), m.w("dec2.bn.bias"), false);
    d2 = build_freq_trim(ctx, d2, mic_e1->ne[0]);

    struct ggml_tensor* d1 = build_decoder_block_s(ctx, sg, d2, mic_e1,
        m.w("dec1.skip_conv.weight"), m.w("dec1.skip_conv.bias"),
        m.w("dec1.resblock.conv.weight"), m.w("dec1.resblock.conv.bias"),
        m.w("dec1.deconv.conv.weight"), m.w("dec1.deconv.conv.bias"),
        nullptr, nullptr, true);
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

    // Allocate
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
    // ggml_backend_tensor_copy is a direct memcpy when both buffers are on
    // the same host backend (CPU), halving memory traffic vs get+set via
    // scratch. For CUDA it's a device-to-device copy on the backend.
    auto copy_hist = [](struct ggml_tensor* in, struct ggml_tensor* out) {
        if (in && out) ggml_backend_tensor_copy(out, in);
    };

    for (size_t i = 0; i < sg.conv_hist_in.size(); i++)
        copy_hist(sg.conv_hist_in[i], sg.conv_hist_out[i]);
    copy_hist(sg.s4d_h_real_in, sg.s4d_h_real_out);
    copy_hist(sg.s4d_h_imag_in, sg.s4d_h_imag_out);
    copy_hist(sg.align_K_hist_in, sg.align_K_hist_out);
    copy_hist(sg.align_ref_hist_in, sg.align_ref_hist_out);
    copy_hist(sg.align_smooth_hist_in, sg.align_smooth_hist_out);
    copy_hist(sg.ccm_hist_in, sg.ccm_hist_out);
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
