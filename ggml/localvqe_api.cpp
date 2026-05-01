/**
 * LocalVQE C API implementation.
 *
 * Streaming graph analysis/synthesis uses an orthonormal DCT-II basis
 * (baked into the graph as mul_mat nodes). This file keeps a 256-sample
 * PCM history ring per channel to construct the 512-sample analysis window
 * and performs 50%-overlap synthesis with divide-by-2 normalization.
 */

#include "localvqe_api.h"
#include "localvqe_graph.h"
#include "noise_gate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// ── Internal context ─────────────────────────────────────────────────────────

struct localvqe_ctx {
    dvqe_graph_model graph_model;
    dvqe_stream_graph stream_graph;
    std::string last_error;

    // 256-sample PCM history per channel (kept for next analysis window).
    std::vector<float> pcm_hist_mic;
    std::vector<float> pcm_hist_ref;

    // Per-frame scratch: the 512-sample analysis window fed to the graph.
    std::vector<float> mic_window;
    std::vector<float> ref_window;
    std::vector<float> enh_window;  // 512-sample graph output before OLA

    // Synthesis OLA accumulator (512 floats, shifts by hop each frame).
    std::vector<float> ola;

    int frame_count = 0;

    // Residual-echo noise gate. Off by default; user opts in via
    // localvqe_set_noise_gate. Operates per-hop on the post-OLA output.
    bool noise_gate_enabled = false;
    float noise_gate_threshold_dbfs = -45.0f;

    // s16 conversion scratch (3 * hop for frame API).
    std::vector<float> s16_conv_buf;

    // Batch API scratch (grow-only).
    std::vector<float> batch_s16_a;
    std::vector<float> batch_s16_b;
    std::vector<float> batch_s16_out;
};

static void ensure_size(std::vector<float>& v, size_t n) {
    if (v.size() < n) v.resize(n);
}

// ── C API ────────────────────────────────────────────────────────────────────

extern "C" {

LOCALVQE_API uintptr_t localvqe_new_ex(const char* model_path,
                                       const char* backend_name,
                                       int device_index) {
    auto* ctx = new (std::nothrow) localvqe_ctx;
    if (!ctx) return 0;

    int n_threads = 0;
    const char* env_threads = std::getenv("GGML_NTHREADS");
    if (env_threads) n_threads = std::atoi(env_threads);

    if (!load_graph_model_ex(model_path, ctx->graph_model,
                             backend_name, device_index, true, n_threads)) {
        delete ctx;
        return 0;
    }
    if (!build_stream_graph(ctx->graph_model, ctx->stream_graph)) {
        free_graph_model(ctx->graph_model);
        delete ctx;
        return 0;
    }

    int n_fft = ctx->graph_model.hparams.n_fft;
    int hop = ctx->graph_model.hparams.hop_length;

    ctx->pcm_hist_mic.assign(hop, 0.0f);
    ctx->pcm_hist_ref.assign(hop, 0.0f);
    ctx->mic_window.assign(n_fft, 0.0f);
    ctx->ref_window.assign(n_fft, 0.0f);
    ctx->enh_window.assign(n_fft, 0.0f);
    ctx->ola.assign(n_fft, 0.0f);
    ctx->frame_count = 0;
    ctx->s16_conv_buf.assign(3 * hop, 0.0f);

    return reinterpret_cast<uintptr_t>(ctx);
}

LOCALVQE_API uintptr_t localvqe_new(const char* model_path) {
    return localvqe_new_ex(model_path, "CPU", 0);
}

LOCALVQE_API void localvqe_list_devices(void) {
    dvqe_list_devices(stderr);
}

LOCALVQE_API void localvqe_print_profile(uintptr_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    print_memory_budget(ctx->graph_model, ctx->stream_graph);
    putchar('\n');
    print_op_histogram(ctx->stream_graph.graph);
    putchar('\n');
}

LOCALVQE_API void localvqe_free(uintptr_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    free_stream_graph(ctx->stream_graph);
    free_graph_model(ctx->graph_model);
    delete ctx;
}

// Push one hop of new PCM into the history ring, filling `window[0..n_fft)`
// with [prev_hop | new_hop]. Leaves history updated to new_hop for next frame.
static void build_window(std::vector<float>& hist, const float* new_hop,
                         int hop, float* window) {
    std::memcpy(window,            hist.data(), hop * sizeof(float));
    std::memcpy(window + hop,      new_hop,     hop * sizeof(float));
    std::memcpy(hist.data(),       new_hop,     hop * sizeof(float));
}

// Stream one hop through the graph, producing one hop of OLA'd output.
// Returns hop samples to `out`. Frame 0 outputs zeros (matches the decoder's
// front-pad trim in the reference PyTorch implementation).
static void stream_one_frame(localvqe_ctx* ctx, const float* mic,
                             const float* ref, float* out) {
    auto& hp = ctx->graph_model.hparams;
    int n_fft = hp.n_fft;
    int hop   = hp.hop_length;

    build_window(ctx->pcm_hist_mic, mic, hop, ctx->mic_window.data());
    build_window(ctx->pcm_hist_ref, ref, hop, ctx->ref_window.data());

    process_frame_graph(ctx->stream_graph, ctx->graph_model,
                        ctx->mic_window.data(), ctx->ref_window.data(),
                        ctx->enh_window.data());

    // OLA: add into accumulator; every interior sample receives exactly
    // two 512-frame contributions at 50% overlap, so /2 in steady state.
    for (int i = 0; i < n_fft; i++) ctx->ola[i] += ctx->enh_window[i];

    if (ctx->frame_count == 0) {
        // Drop the first hop — it only carries the encoder's zero-padded
        // prefix. Mirrors `output[:, self.pad:]` in DCTDecoder.
        std::memset(out, 0, hop * sizeof(float));
    } else {
        const float scale = 0.5f;
        for (int i = 0; i < hop; i++) out[i] = ctx->ola[i] * scale;
        if (ctx->noise_gate_enabled) {
            localvqe::apply_noise_gate(out, hop,
                                       ctx->noise_gate_threshold_dbfs);
        }
    }

    // Slide accumulator left by hop, zero-fill the tail.
    std::memmove(ctx->ola.data(), ctx->ola.data() + hop,
                 (n_fft - hop) * sizeof(float));
    std::memset(ctx->ola.data() + (n_fft - hop), 0, hop * sizeof(float));

    ctx->frame_count++;
}

LOCALVQE_API int localvqe_process_f32(uintptr_t handle,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    ctx->last_error.clear();

    auto& hp = ctx->graph_model.hparams;
    int n_fft = hp.n_fft;
    int hop   = hp.hop_length;

    if (n_samples < n_fft) {
        ctx->last_error = "Input too short: need at least "
                          + std::to_string(n_fft) + " samples";
        return -2;
    }

    reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    std::fill(ctx->pcm_hist_mic.begin(), ctx->pcm_hist_mic.end(), 0.0f);
    std::fill(ctx->pcm_hist_ref.begin(), ctx->pcm_hist_ref.end(), 0.0f);
    std::fill(ctx->ola.begin(), ctx->ola.end(), 0.0f);
    ctx->frame_count = 0;

    int n_frames = n_samples / hop;  // drop the trailing partial hop
    for (int t = 0; t < n_frames; t++) {
        stream_one_frame(ctx,
                         mic + t * hop,
                         ref + t * hop,
                         out + t * hop);
    }
    int tail = n_samples - n_frames * hop;
    if (tail > 0) std::memset(out + n_frames * hop, 0, tail * sizeof(float));
    return 0;
}

// ── s16/f32 conversion helpers ────────────────────────────────────────────

static void s16_to_f32(const int16_t* in, float* out, int n) {
    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < n; i++) out[i] = in[i] * scale;
}

static void f32_to_s16(const float* in, int16_t* out, int n) {
    for (int i = 0; i < n; i++) {
        float v = std::max(-32768.0f, std::min(32767.0f, in[i] * 32768.0f));
        out[i] = (int16_t)v;
    }
}

LOCALVQE_API int localvqe_process_s16(uintptr_t handle,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);

    ensure_size(ctx->batch_s16_a, n_samples);
    ensure_size(ctx->batch_s16_b, n_samples);
    ensure_size(ctx->batch_s16_out, n_samples);

    s16_to_f32(mic, ctx->batch_s16_a.data(), n_samples);
    s16_to_f32(ref, ctx->batch_s16_b.data(), n_samples);

    int ret = localvqe_process_f32(handle, ctx->batch_s16_a.data(),
                                   ctx->batch_s16_b.data(), n_samples,
                                   ctx->batch_s16_out.data());
    if (ret != 0) return ret;

    f32_to_s16(ctx->batch_s16_out.data(), out, n_samples);
    return 0;
}

LOCALVQE_API const char* localvqe_last_error(uintptr_t handle) {
    if (!handle) return "null context";
    return reinterpret_cast<localvqe_ctx*>(handle)->last_error.c_str();
}

LOCALVQE_API int localvqe_sample_rate(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.sample_rate;
}

LOCALVQE_API int localvqe_hop_length(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.hop_length;
}

LOCALVQE_API int localvqe_fft_size(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.n_fft;
}

// ── Streaming C API ──────────────────────────────────────────────────────

LOCALVQE_API int localvqe_process_frame_f32(uintptr_t handle,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (hop_samples != ctx->graph_model.hparams.hop_length) return -2;
    stream_one_frame(ctx, mic, ref, out);
    return 0;
}

LOCALVQE_API int localvqe_process_frame_s16(uintptr_t handle,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);

    float* mic_f = ctx->s16_conv_buf.data();
    float* ref_f = mic_f + hop_samples;
    float* out_f = ref_f + hop_samples;
    s16_to_f32(mic, mic_f, hop_samples);
    s16_to_f32(ref, ref_f, hop_samples);

    int ret = localvqe_process_frame_f32(handle, mic_f, ref_f, hop_samples, out_f);
    if (ret != 0) return ret;

    f32_to_s16(out_f, out, hop_samples);
    return 0;
}

LOCALVQE_API void localvqe_reset(uintptr_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    std::fill(ctx->pcm_hist_mic.begin(), ctx->pcm_hist_mic.end(), 0.0f);
    std::fill(ctx->pcm_hist_ref.begin(), ctx->pcm_hist_ref.end(), 0.0f);
    std::fill(ctx->ola.begin(), ctx->ola.end(), 0.0f);
    ctx->frame_count = 0;
    // Note: gate config is intentionally NOT reset — it's caller-set
    // policy, not stream state.
}

LOCALVQE_API int localvqe_set_noise_gate(uintptr_t handle,
                                        int enabled,
                                        float threshold_dbfs) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    ctx->noise_gate_enabled = (enabled != 0);
    ctx->noise_gate_threshold_dbfs = threshold_dbfs;
    return 0;
}

LOCALVQE_API int localvqe_get_noise_gate(uintptr_t handle,
                                        int* enabled_out,
                                        float* threshold_dbfs_out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (enabled_out) *enabled_out = ctx->noise_gate_enabled ? 1 : 0;
    if (threshold_dbfs_out) *threshold_dbfs_out = ctx->noise_gate_threshold_dbfs;
    return 0;
}

} // extern "C"
