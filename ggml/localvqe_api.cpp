/**
 * LocalVQE C API implementation.
 *
 * Streaming graph analysis/synthesis uses an orthonormal DCT-II basis
 * (baked into the graph as mul_mat nodes). This file keeps a 256-sample
 * PCM history ring per channel to construct the 512-sample analysis window
 * and performs 50%-overlap synthesis with divide-by-2 normalization.
 */

#include "localvqe_api.h"
#include "daf_frontend.h"
#include "localvqe_graph.h"
#include "noise_gate.h"
#include "gguf.h"

#ifdef LOCALVQE_HAS_GTCRN
#include "gtcrn.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// Read a string-valued KV from a gguf, or "" if absent/unreadable.
static std::string gguf_str_kv(const char* path, const char* key) {
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context* gc = gguf_init_from_file(path, p);
    if (!gc) return "";
    std::string v;
    int64_t kid = gguf_find_key(gc, key);
    if (kid >= 0 && gguf_get_kv_type(gc, kid) == GGUF_TYPE_STRING)
        v = gguf_get_val_str(gc, kid);
    gguf_free(gc);
    return v;
}

// general.architecture — lets the C API route GTCRN (compact / low-power)
// models down a different path than the v1.x streaming graph.
static std::string gguf_architecture(const char* path) {
    return gguf_str_kv(path, "general.architecture");
}

// ── Internal context ─────────────────────────────────────────────────────────

struct localvqe_ctx {
    // Optional DAF front-end (203K cascade): present when the gguf embeds
    // daf.* tensors. Runs per hop before the mask graph; the graph then
    // consumes (e, yhat) instead of (mic, ref).
    daf_frontend daf;
    std::vector<float> daf_e, daf_yhat;

    dvqe_graph_model graph_model;
    dvqe_stream_graph stream_graph;
    std::string last_error;

    // Compact / low-power line (arch="gtcrn"): the DAF front-end above runs on
    // a separate v1.4-AEC `graph_model`, and the GTCRN backend (its own STFT)
    // consumes the (e, yhat). Whole-clip via localvqe_process_f32; the per-hop
    // frame API is not applicable (the GTCRN STFT is non-streaming here).
    bool is_gtcrn = false;
    bool fe_model_loaded = false;  // gtcrn: graph_model holds a separate front-end
#ifdef LOCALVQE_HAS_GTCRN
    GtcrnGraph* gt_graph = nullptr;
    GtcrnModel* gt_host  = nullptr;
    // GTCRN streaming-frame state (built lazily on the first frame call).
    GtcrnStream* gt_stream = nullptr;
    std::vector<float> gt_bufE, gt_bufY;   // 512-sample STFT analysis buffers
    std::vector<float> gt_acc, gt_wenv;    // 512-sample ISTFT overlap-add accumulators
    float gt_pow = 0.0f;                   // running e-power (EMA) for the gain
    bool gt_pow_init = false;
#endif

    // Front-end-only build (2.7K release): mask graph never built; each
    // hop emits the adaptive filter's output `e` directly.
    bool daf_standalone = false;

    // 256-sample PCM history per channel (kept for next analysis window).
    std::vector<float> pcm_hist_mic;
    std::vector<float> pcm_hist_ref;

    // Per-frame scratch: the 512-sample analysis window fed to the graph.
    std::vector<float> mic_window;
    std::vector<float> ref_window;
    std::vector<float> enh_window;  // 512-sample graph output before OLA

    // Synthesis OLA accumulator (512 floats, shifts by hop each frame).
    std::vector<float> ola;

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

struct localvqe_options {
    std::string model_path;
    std::string frontend_path;  // GTCRN / low-power line: separate DAF gguf
    std::string backend_name = "CPU";
    int device_index = 0;
    int n_threads = 0;  // 0 = auto / honour GGML_NTHREADS env var
};

#ifdef LOCALVQE_HAS_GTCRN
static bool file_exists(const std::string& p) {
    if (p.empty()) return false;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

static std::string dir_of(const char* path) {
    std::string s(path ? path : "");
    size_t pos = s.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : s.substr(0, pos + 1);
}

// Decide the DAF front-end for a GTCRN (compact-line) model without the caller
// naming it. The whole compact line shares the same v1.4-AEC DAF, so this needs
// no per-version logic: an explicit "localvqe.frontend" hint baked into the
// gguf wins (forward-compatible), otherwise a v1.4-AEC gguf sitting next to the
// model. Returns "" if none is found.
static std::string resolve_gtcrn_frontend(const char* model_path) {
    std::string dir = dir_of(model_path);
    std::string hint = gguf_str_kv(model_path, "localvqe.frontend");
    if (!hint.empty()) {
        if (file_exists(hint)) return hint;          // absolute / cwd-relative
        if (file_exists(dir + hint)) return dir + hint;  // next to the model
    }
    // Released v1.4-AEC front-ends, lightest first (the 2.7K is the DAF alone).
    const char* names[] = {
        "localvqe-v1.4-aec-2.7K-f32.gguf",
        "localvqe-v1.4-aec-200K-f32.gguf",
        "localvqe-v1.4-aec-200K-bf16.gguf",
    };
    for (const char* n : names)
        if (file_exists(dir + n)) return dir + n;
    return "";
}

// Build a GTCRN (compact / low-power) context. The compact net runs on a
// v1.4-AEC DAF residual; the front-end comes from, in order: (1) daf.* tensors
// embedded in the model gguf itself (a self-contained file — nothing else
// needed), (2) an explicit frontend_path, (3) a v1.4-AEC gguf auto-detected
// next to the model. Returns the populated ctx (early-out from make_ctx), or 0.
static localvqe_ctx_t make_ctx_gtcrn(localvqe_ctx* ctx, const char* model_path,
                                     const char* frontend_path,
                                     const char* backend_name, int device_index,
                                     int n_threads) {
    // Register ggml backends self-resolving from this module's dir — the
    // GTCRN path uses the backend registry directly, and the self-contained
    // case never calls load_graph_model_ex (which would otherwise do this).
    ensure_backends_loaded();

    // GTCRN backend first — loading the host weights also exposes any embedded
    // daf.* tensors, so we can tell a self-contained model from one that needs
    // a separate front-end.
    ctx->gt_graph = new (std::nothrow) GtcrnGraph();
    ctx->gt_host  = new (std::nothrow) GtcrnModel();
    int gt_threads = n_threads > 0 ? n_threads : 1;
    if (!ctx->gt_graph || !ctx->gt_host ||
        !ctx->gt_graph->load(model_path, gt_threads, false) ||
        !ctx->gt_host->load(model_path, false)) {
        fprintf(stderr, "localvqe: failed to load GTCRN gguf: %s\n", model_path);
        delete ctx->gt_graph;
        delete ctx->gt_host;
        delete ctx;
        return 0;
    }

    // (1) Self-contained: the model embeds its own DAF front-end.
    if (daf_init_tensors(ctx->daf, ctx->gt_host->W)) {
        ctx->graph_model.hparams.sample_rate = 16000;
        ctx->graph_model.hparams.hop_length  = 256;
        ctx->graph_model.hparams.n_fft       = 512;
        ctx->is_gtcrn = true;
        ctx->daf_e.assign(256, 0.0f);
        ctx->daf_yhat.assign(256, 0.0f);
        fprintf(stderr, "localvqe: GTCRN backend active (compact / low-power "
                "line), embedded DAF front-end\n");
        return reinterpret_cast<localvqe_ctx_t>(ctx);
    }

    // (2)/(3) Need a separate front-end gguf: explicit, else auto-detect.
    std::string resolved;
    if (!frontend_path || !*frontend_path) {
        resolved = resolve_gtcrn_frontend(model_path);
        if (resolved.empty()) {
            fprintf(stderr, "localvqe: GTCRN model '%s' needs a v1.4-AEC DAF "
                    "front-end and none was found next to it. Pass one explicitly "
                    "(--fe / localvqe_options_set_frontend_path), or place e.g. "
                    "localvqe-v1.4-aec-2.7K-f32.gguf in the same directory.\n",
                    model_path);
            delete ctx->gt_graph; delete ctx->gt_host;
            delete ctx;
            return 0;
        }
        frontend_path = resolved.c_str();
        fprintf(stderr, "localvqe: auto-selected DAF front-end %s\n", frontend_path);
    }
    if (!load_graph_model_ex(frontend_path, ctx->graph_model,
                             backend_name, device_index, true, n_threads)) {
        fprintf(stderr, "localvqe: failed to load front-end gguf: %s\n", frontend_path);
        delete ctx->gt_graph; delete ctx->gt_host;
        delete ctx;
        return 0;
    }
    ctx->fe_model_loaded = true;
    if (!daf_init(ctx->daf, ctx->graph_model)) {
        fprintf(stderr, "localvqe: front-end gguf %s has no DAF tensors "
                "(use a v1.4-AEC model)\n", frontend_path);
        free_graph_model(ctx->graph_model);
        delete ctx->gt_graph; delete ctx->gt_host;
        delete ctx;
        return 0;
    }
    ctx->is_gtcrn = true;
    int hop = ctx->graph_model.hparams.hop_length;
    ctx->daf_e.assign(hop, 0.0f);
    ctx->daf_yhat.assign(hop, 0.0f);
    fprintf(stderr, "localvqe: GTCRN backend active (compact / low-power line), "
            "DAF front-end from %s\n", frontend_path);
    return reinterpret_cast<localvqe_ctx_t>(ctx);
}
#endif  // LOCALVQE_HAS_GTCRN

static localvqe_ctx_t make_ctx(const char* model_path,
                               const char* backend_name,
                               int device_index,
                               int n_threads_override = 0,
                               const char* frontend_path = nullptr) {
    auto* ctx = new (std::nothrow) localvqe_ctx;
    if (!ctx) return 0;

    int n_threads = n_threads_override;
    if (n_threads == 0) {
        // Fallback chain: explicit option (above) → env var → auto.
        if (const char* env_threads = std::getenv("GGML_NTHREADS")) {
            n_threads = std::atoi(env_threads);
        }
    }

#ifdef LOCALVQE_HAS_GTCRN
    if (gguf_architecture(model_path) == "gtcrn") {
        return make_ctx_gtcrn(ctx, model_path, frontend_path,
                              backend_name, device_index, n_threads);
    }
#else
    (void)frontend_path;
#endif

    if (!load_graph_model_ex(model_path, ctx->graph_model,
                             backend_name, device_index, true, n_threads)) {
        delete ctx;
        return 0;
    }
    ctx->daf_standalone = ctx->graph_model.hparams.daf_standalone;
    if (!ctx->daf_standalone &&
        !build_stream_graph(ctx->graph_model, ctx->stream_graph)) {
        free_graph_model(ctx->graph_model);
        delete ctx;
        return 0;
    }

    int n_fft = ctx->graph_model.hparams.n_fft;
    int hop = ctx->graph_model.hparams.hop_length;

    bool daf_ok = daf_init(ctx->daf, ctx->graph_model);
    if (ctx->daf_standalone) {
        if (!daf_ok) {  // a standalone build with no filter tensors is invalid
            fprintf(stderr, "localvqe: daf.standalone set but no daf.* tensors\n");
            free_graph_model(ctx->graph_model);
            delete ctx;
            return 0;
        }
        fprintf(stderr, "localvqe: front-end-only build (2.7K adaptive filter)\n");
    } else if (daf_ok) {
        fprintf(stderr, "localvqe: DAF front-end active (203K cascade)\n");
    }
    ctx->daf_e.assign(hop, 0.0f);
    ctx->daf_yhat.assign(hop, 0.0f);
    ctx->pcm_hist_mic.assign(hop, 0.0f);
    ctx->pcm_hist_ref.assign(hop, 0.0f);
    ctx->mic_window.assign(n_fft, 0.0f);
    ctx->ref_window.assign(n_fft, 0.0f);
    ctx->enh_window.assign(n_fft, 0.0f);
    ctx->ola.assign(n_fft, 0.0f);
    ctx->s16_conv_buf.assign(3 * hop, 0.0f);

    return reinterpret_cast<localvqe_ctx_t>(ctx);
}

extern "C" {

LOCALVQE_API localvqe_ctx_t localvqe_new(const char* model_path) {
    return make_ctx(model_path, "CPU", 0);
}

LOCALVQE_API localvqe_ctx_t localvqe_new_with_frontend(const char* model_path,
                                                       const char* frontend_path) {
    return make_ctx(model_path, "CPU", 0, 0, frontend_path);
}

LOCALVQE_API localvqe_options_t localvqe_options_new(void) {
    return reinterpret_cast<localvqe_options_t>(new (std::nothrow) localvqe_options);
}

LOCALVQE_API void localvqe_options_free(localvqe_options_t handle) {
    delete reinterpret_cast<localvqe_options*>(handle);
}

LOCALVQE_API int localvqe_options_set_model_path(localvqe_options_t handle,
                                                 const char* model_path) {
    if (!handle) return -1;
    if (!model_path || !*model_path) return -2;
    reinterpret_cast<localvqe_options*>(handle)->model_path = model_path;
    return 0;
}

LOCALVQE_API int localvqe_options_set_backend(localvqe_options_t handle,
                                              const char* backend_name) {
    if (!handle) return -1;
    if (!backend_name || !*backend_name) return -2;
    reinterpret_cast<localvqe_options*>(handle)->backend_name = backend_name;
    return 0;
}

LOCALVQE_API int localvqe_options_set_frontend_path(localvqe_options_t handle,
                                                    const char* frontend_path) {
    if (!handle) return -1;
    if (!frontend_path || !*frontend_path) return -2;
    reinterpret_cast<localvqe_options*>(handle)->frontend_path = frontend_path;
    return 0;
}

LOCALVQE_API int localvqe_options_set_device(localvqe_options_t handle,
                                             int device_index) {
    if (!handle) return -1;
    if (device_index < 0) return -2;
    reinterpret_cast<localvqe_options*>(handle)->device_index = device_index;
    return 0;
}

LOCALVQE_API int localvqe_options_set_threads(localvqe_options_t handle,
                                              int n_threads) {
    if (!handle) return -1;
    if (n_threads < 0 || n_threads > 32) return -2;
    reinterpret_cast<localvqe_options*>(handle)->n_threads = n_threads;
    return 0;
}

LOCALVQE_API localvqe_ctx_t localvqe_new_with_options(localvqe_options_t handle) {
    if (!handle) return 0;
    auto* opts = reinterpret_cast<localvqe_options*>(handle);
    if (opts->model_path.empty()) {
        fprintf(stderr, "localvqe: options missing model_path\n");
        return 0;
    }
    return make_ctx(opts->model_path.c_str(),
                    opts->backend_name.c_str(),
                    opts->device_index,
                    opts->n_threads,
                    opts->frontend_path.empty() ? nullptr
                                                : opts->frontend_path.c_str());
}

LOCALVQE_API void localvqe_list_devices(void) {
    dvqe_list_devices(stderr);
}

LOCALVQE_API void localvqe_print_profile(localvqe_ctx_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
#ifdef LOCALVQE_HAS_GTCRN
    if (ctx->is_gtcrn) {
        printf("GTCRN backend (compact / low-power line); no streaming graph profile.\n");
        return;
    }
#endif
    print_memory_budget(ctx->graph_model, ctx->stream_graph);
    putchar('\n');
    print_op_histogram(ctx->stream_graph.graph);
    putchar('\n');
}

LOCALVQE_API void localvqe_free(localvqe_ctx_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
#ifdef LOCALVQE_HAS_GTCRN
    if (ctx->is_gtcrn) {
        delete ctx->gt_stream;
        delete ctx->gt_graph;
        delete ctx->gt_host;
        if (ctx->fe_model_loaded) free_graph_model(ctx->graph_model);
        delete ctx;
        return;
    }
#endif
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
// Returns hop samples to `out`. The OLA accumulator is emitted on every
// call including frame 0 — synthesis-window-tapered samples at the left
// edge of the first frame's reconstruction start near zero and ramp up,
// so callers can pre-pend silence (or simply concatenate) without a
// click at the boundary. v1's DCT codec previously zeroed the first hop
// and got away with it because its OLA divisor halved the next hop; the
// COLA=1 sqrt-Hann² codec lands the next hop's first sample at full
// synthesis-window peak, so emitting zeros for the first hop and
// peak-amplitude content for the second creates an audible step.
static void stream_one_frame(localvqe_ctx* ctx, const float* mic,
                             const float* ref, float* out) {
    auto& hp = ctx->graph_model.hparams;
    int n_fft = hp.n_fft;
    int hop   = hp.hop_length;

    // Front-end-only build: emit the adaptive filter's output `e` directly,
    // no mask graph / codec round-trip (and so no one-hop codec latency).
    if (ctx->daf_standalone) {
        daf_process(ctx->daf, mic, ref, hop,
                    ctx->daf_e.data(), ctx->daf_yhat.data());
        std::memcpy(out, ctx->daf_e.data(), hop * sizeof(float));
        if (ctx->noise_gate_enabled) {
            localvqe::apply_noise_gate(out, hop,
                                       ctx->noise_gate_threshold_dbfs);
        }
        return;
    }

    if (ctx->daf.loaded) {
        daf_process(ctx->daf, mic, ref, hop,
                    ctx->daf_e.data(), ctx->daf_yhat.data());
        mic = ctx->daf_e.data();
        ref = ctx->daf_yhat.data();
    }
    build_window(ctx->pcm_hist_mic, mic, hop, ctx->mic_window.data());
    build_window(ctx->pcm_hist_ref, ref, hop, ctx->ref_window.data());

    process_frame_graph(ctx->stream_graph, ctx->graph_model,
                        ctx->mic_window.data(), ctx->ref_window.data(),
                        ctx->enh_window.data());

    // OLA scale below: v1's DCT codec has no analysis window so the two
    // overlapping frame contributions must be halved; v1.1's sqrt-Hann²
    // STFT-256 is COLA=1 at hop=N/2, no divisor.
    for (int i = 0; i < n_fft; i++) ctx->ola[i] += ctx->enh_window[i];

    const float scale = (hp.version >= 2) ? 1.0f : 0.5f;
    for (int i = 0; i < hop; i++) out[i] = ctx->ola[i] * scale;
    if (ctx->noise_gate_enabled) {
        localvqe::apply_noise_gate(out, hop,
                                   ctx->noise_gate_threshold_dbfs);
    }

    // Slide accumulator left by hop, zero-fill the tail.
    std::memmove(ctx->ola.data(), ctx->ola.data() + hop,
                 (n_fft - hop) * sizeof(float));
    std::memset(ctx->ola.data() + (n_fft - hop), 0, hop * sizeof(float));
}

#ifdef LOCALVQE_HAS_GTCRN
// Whole-clip GTCRN (compact / low-power line): DAF echo-cancel front-end ->
// residual e + echo estimate yhat; per-utterance RMS-normalise to 0.05; GTCRN
// complex-ratio mask on its own 512/256 STFT; iSTFT; undo the gain. Mirrors the
// PyTorch reference the gguf was exported from. The trailing partial DAF hop
// (< 128 samples) is zero-filled to keep the output sample-aligned.
static int process_gtcrn_clip(localvqe_ctx* ctx, const float* mic,
                              const float* ref, int n_samples, float* out) {
    const int M = daf_frontend::M;
    int n = n_samples / M * M;
    if (n <= 0) {
        ctx->last_error = "Input too short for the GTCRN front-end";
        return -2;
    }
    daf_reset(ctx->daf);
    ctx->daf.enable_prealign = true;
    daf_prime_delay(ctx->daf, mic, ref, n);   // file mode: lock the bulk delay
    ensure_size(ctx->daf_e, n);
    ensure_size(ctx->daf_yhat, n);
    daf_process(ctx->daf, mic, ref, n, ctx->daf_e.data(), ctx->daf_yhat.data());

    float* e  = ctx->daf_e.data();
    float* yh = ctx->daf_yhat.data();
    double sq = 0.0;
    for (int i = 0; i < n; ++i) sq += (double)e[i] * e[i];
    float gain = 0.05f / ((float)std::sqrt(sq / n) + 1e-6f);
    gain = std::min(50.0f, std::max(0.05f, gain));
    for (int i = 0; i < n; ++i) { e[i] *= gain; yh[i] *= gain; }

    int T = ctx->gt_host->n_frames(n);
    std::vector<float> spec_e = ctx->gt_host->stft(e, n);
    std::vector<float> spec_y = ctx->gt_host->stft(yh, n);
    std::vector<float> enh = ctx->gt_graph->forward(spec_e.data(), spec_y.data(), T);
    std::vector<float> y = ctx->gt_host->istft(enh.data(), T, n);
    float inv = 1.0f / gain;
    for (int i = 0; i < n; ++i) out[i] = y[i] * inv;
    if (n < n_samples) std::memset(out + n, 0, (size_t)(n_samples - n) * sizeof(float));

    if (ctx->noise_gate_enabled) {
        int hop = ctx->graph_model.hparams.hop_length;
        for (int t = 0; t + hop <= n_samples; t += hop)
            localvqe::apply_noise_gate(out + t, hop, ctx->noise_gate_threshold_dbfs);
    }
    return 0;
}

// Zero all GTCRN streaming state (new utterance / first frame).
static void gtcrn_stream_reset(localvqe_ctx* ctx) {
    if (ctx->daf.loaded) daf_reset(ctx->daf);
    ctx->daf.enable_prealign = true;          // online GCC acquisition (no file-mode prime)
    if (ctx->gt_stream) ctx->gt_stream->reset();
    std::fill(ctx->gt_bufE.begin(), ctx->gt_bufE.end(), 0.0f);
    std::fill(ctx->gt_bufY.begin(), ctx->gt_bufY.end(), 0.0f);
    std::fill(ctx->gt_acc.begin(), ctx->gt_acc.end(), 0.0f);
    std::fill(ctx->gt_wenv.begin(), ctx->gt_wenv.end(), 0.0f);
    ctx->gt_pow = 0.0f; ctx->gt_pow_init = false;
}

// One streaming hop (256 samples) for a GTCRN model: DAF front-end (online) ->
// running RMS-normalise -> 512-sample STFT frame -> GtcrnStream::step (carried
// recurrent state) -> inverse frame, undo gain, overlap-add -> 256 samples out.
// Real-time / low-latency counterpart of process_gtcrn_clip; not bit-identical
// to it (online DAF acquisition + running gain vs whole-clip).
static int process_gtcrn_frame(localvqe_ctx* ctx, const float* mic,
                               const float* ref, float* out) {
    const int H = 256, N = 512;
    if (!ctx->gt_stream) {
        ctx->gt_stream = new (std::nothrow) GtcrnStream();
        if (!ctx->gt_stream || !ctx->gt_stream->begin(*ctx->gt_graph)) {
            ctx->last_error = "GTCRN streaming session init failed";
            delete ctx->gt_stream; ctx->gt_stream = nullptr; return -1;
        }
        ctx->gt_bufE.assign(N, 0.0f); ctx->gt_bufY.assign(N, 0.0f);
        ctx->gt_acc.assign(N, 0.0f);  ctx->gt_wenv.assign(N, 0.0f);
        gtcrn_stream_reset(ctx);
    }

    // 1. DAF front-end, one hop (online acquisition — no file-mode prime).
    ensure_size(ctx->daf_e, H); ensure_size(ctx->daf_yhat, H);
    daf_process(ctx->daf, mic, ref, H, ctx->daf_e.data(), ctx->daf_yhat.data());
    float* e = ctx->daf_e.data(); float* yh = ctx->daf_yhat.data();

    // 2. running RMS gain (EMA) toward the training level (0.05 RMS).
    double p = 0.0;
    for (int i = 0; i < H; ++i) p += (double)e[i] * e[i];
    p /= H;
    const float A = 0.95f;
    ctx->gt_pow = ctx->gt_pow_init ? (A * ctx->gt_pow + (1.0f - A) * (float)p) : (float)p;
    ctx->gt_pow_init = true;
    float gain = 0.05f / ((float)std::sqrt(ctx->gt_pow) + 1e-6f);
    gain = std::min(50.0f, std::max(0.05f, gain));

    // 3. slide the 512-sample analysis buffers, append this hop RAW. The gain is
    //    applied uniformly to the whole window below, so a frame is never split
    //    across two different per-hop gains (which would corrupt the spectrum).
    float* bE = ctx->gt_bufE.data(); float* bY = ctx->gt_bufY.data();
    std::memmove(bE, bE + H, (size_t)H * sizeof(float));
    std::memmove(bY, bY + H, (size_t)H * sizeof(float));
    for (int i = 0; i < H; ++i) { bE[H + i] = e[i]; bY[H + i] = yh[i]; }

    // 4. uniform gain over the full window, then one STFT frame each.
    float winE[512], winY[512];
    for (int n = 0; n < N; ++n) { winE[n] = bE[n] * gain; winY[n] = bY[n] * gain; }
    float re_e[257], im_e[257], re_y[257], im_y[257], spe[514], spy[514], osp[514];
    ctx->gt_host->stft_frame(winE, re_e, im_e);
    ctx->gt_host->stft_frame(winY, re_y, im_y);
    for (int f = 0; f < 257; ++f) {
        spe[f * 2] = re_e[f]; spe[f * 2 + 1] = im_e[f];
        spy[f * 2] = re_y[f]; spy[f * 2 + 1] = im_y[f];
    }

    // 5. GTCRN net step (recurrent state carried across calls).
    ctx->gt_stream->step(spe, spy, osp);

    // 6. inverse frame, undo the gain so the OLA mixes original-scale frames.
    float ore[257], oim[257], ft[512];
    for (int f = 0; f < 257; ++f) { ore[f] = osp[f * 2]; oim[f] = osp[f * 2 + 1]; }
    ctx->gt_host->istft_frame(ore, oim, ft);
    float inv = 1.0f / gain;
    for (int n = 0; n < N; ++n) ft[n] *= inv;

    // 7. overlap-add (50%), emit the now-complete hop, slide the accumulators.
    const std::vector<float>& w2 = ctx->gt_host->win2();
    float* acc = ctx->gt_acc.data(); float* we = ctx->gt_wenv.data();
    for (int n = 0; n < N; ++n) { acc[n] += ft[n]; we[n] += w2[n]; }
    for (int i = 0; i < H; ++i) out[i] = (we[i] > 1e-11f) ? acc[i] / we[i] : 0.0f;
    std::memmove(acc, acc + H, (size_t)H * sizeof(float)); std::memset(acc + H, 0, (size_t)H * sizeof(float));
    std::memmove(we, we + H, (size_t)H * sizeof(float));   std::memset(we + H, 0, (size_t)H * sizeof(float));

    // 8. optional residual-echo gate.
    if (ctx->noise_gate_enabled)
        localvqe::apply_noise_gate(out, H, ctx->noise_gate_threshold_dbfs);
    return 0;
}
#endif  // LOCALVQE_HAS_GTCRN

LOCALVQE_API int localvqe_process_f32(localvqe_ctx_t handle,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    ctx->last_error.clear();

#ifdef LOCALVQE_HAS_GTCRN
    if (ctx->is_gtcrn) return process_gtcrn_clip(ctx, mic, ref, n_samples, out);
#endif

    auto& hp = ctx->graph_model.hparams;
    int n_fft = hp.n_fft;
    int hop   = hp.hop_length;

    if (n_samples < n_fft) {
        ctx->last_error = "Input too short: need at least "
                          + std::to_string(n_fft) + " samples";
        return -2;
    }

    if (!ctx->daf_standalone)
        reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    if (ctx->daf.loaded) daf_reset(ctx->daf);
    // Standalone front-end: the whole clip is available here (callers pass the
    // full signal), so prime the bulk delay once and lock it — the filter is
    // aligned from frame 0 instead of acquiring over the first ~1-3 s. The
    // cascade keeps online acquisition (its mask hides the transient).
    if (ctx->daf_standalone && ctx->daf.loaded)
        daf_prime_delay(ctx->daf, mic, ref, n_samples);
    std::fill(ctx->pcm_hist_mic.begin(), ctx->pcm_hist_mic.end(), 0.0f);
    std::fill(ctx->pcm_hist_ref.begin(), ctx->pcm_hist_ref.end(), 0.0f);
    std::fill(ctx->ola.begin(), ctx->ola.end(), 0.0f);

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

LOCALVQE_API int localvqe_process_s16(localvqe_ctx_t handle,
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

LOCALVQE_API const char* localvqe_last_error(localvqe_ctx_t handle) {
    if (!handle) return "null context";
    return reinterpret_cast<localvqe_ctx*>(handle)->last_error.c_str();
}

LOCALVQE_API int localvqe_sample_rate(localvqe_ctx_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.sample_rate;
}

LOCALVQE_API int localvqe_hop_length(localvqe_ctx_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.hop_length;
}

LOCALVQE_API int localvqe_fft_size(localvqe_ctx_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<localvqe_ctx*>(handle)->graph_model.hparams.n_fft;
}

// ── Streaming C API ──────────────────────────────────────────────────────

LOCALVQE_API int localvqe_process_frame_f32(localvqe_ctx_t handle,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
#ifdef LOCALVQE_HAS_GTCRN
    if (ctx->is_gtcrn) {
        if (hop_samples != 256) {
            ctx->last_error = "GTCRN streaming hop must be 256 samples";
            return -2;
        }
        return process_gtcrn_frame(ctx, mic, ref, out);
    }
#endif
    if (hop_samples != ctx->graph_model.hparams.hop_length) return -2;
    stream_one_frame(ctx, mic, ref, out);
    return 0;
}

LOCALVQE_API int localvqe_process_frame_s16(localvqe_ctx_t handle,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (hop_samples <= 0) { ctx->last_error = "hop_samples must be > 0"; return -2; }

    // The s16 scratch (3*hop) is sized here rather than at construction so it
    // is valid for every model type — GTCRN contexts skip the v1.x buffer init.
    ensure_size(ctx->s16_conv_buf, (size_t)3 * hop_samples);
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

LOCALVQE_API void localvqe_reset(localvqe_ctx_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
#ifdef LOCALVQE_HAS_GTCRN
    if (ctx->is_gtcrn) {
        gtcrn_stream_reset(ctx);  // streaming state; the whole-clip path re-primes per call
        return;
    }
#endif
    if (!ctx->daf_standalone)
        reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    if (ctx->daf.loaded) daf_reset(ctx->daf);
    std::fill(ctx->pcm_hist_mic.begin(), ctx->pcm_hist_mic.end(), 0.0f);
    std::fill(ctx->pcm_hist_ref.begin(), ctx->pcm_hist_ref.end(), 0.0f);
    std::fill(ctx->ola.begin(), ctx->ola.end(), 0.0f);
    // Note: gate config is intentionally NOT reset — it's caller-set
    // policy, not stream state.
}

LOCALVQE_API int localvqe_set_noise_gate(localvqe_ctx_t handle,
                                        int enabled,
                                        float threshold_dbfs) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    ctx->noise_gate_enabled = (enabled != 0);
    ctx->noise_gate_threshold_dbfs = threshold_dbfs;
    return 0;
}

LOCALVQE_API int localvqe_get_noise_gate(localvqe_ctx_t handle,
                                        int* enabled_out,
                                        float* threshold_dbfs_out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<localvqe_ctx*>(handle);
    if (enabled_out) *enabled_out = ctx->noise_gate_enabled ? 1 : 0;
    if (threshold_dbfs_out) *threshold_dbfs_out = ctx->noise_gate_threshold_dbfs;
    return 0;
}

} // extern "C"
