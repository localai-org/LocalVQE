/**
 * GTCRN-AEC parity + benchmark harness.
 *
 *   test_gtcrn --gguf m.gguf --fixtures <dir> [--ggml]   # per-stage parity
 *   test_gtcrn --gguf m.gguf --bench [frames] [reps] [--ggml]
 *
 * Default uses the scalar reference (gtcrn.cpp). --ggml uses the real ggml
 * compute graph (gtcrn_ggml.cpp). Parity diffs every stage against the
 * export_gtcrn_ggml.py .npy fixtures (the validated PyTorch reference). Bench
 * reports single-thread CPU time -> real-time factor (frames*256/16000 s).
 */

#include "gtcrn.h"
#include "common.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using Fwd = std::function<std::vector<float>(const float*, const float*, int,
                                             std::map<std::string, NpyArray>*, double*)>;

static int parity(const Fwd& fwd, const std::string& fdir, bool only_out) {
    NpyArray e = npy_load(fdir + "/in_spec_e.npy");
    NpyArray y = npy_load(fdir + "/in_spec_y.npy");
    if (e.ndim() != 4) { fprintf(stderr, "bad spec fixture\n"); return 1; }
    int T = (int)e.dim(2);
    printf("Parity: T=%d frames\n\n", T);

    std::map<std::string, NpyArray> cap;
    fwd(e.data.data(), y.data.data(), T, &cap, nullptr);

    const char* all[] = {"feat", "enc0", "enc1", "enc2", "enc3", "enc4",
                         "dpgrnn1", "dpgrnn2", "dec0", "dec1", "dec2",
                         "dec3", "dec4", "mask", "out_spec"};
    const char* out_only[] = {"enc4", "dpgrnn1", "out_spec"};
    std::vector<const char*> stages = only_out
        ? std::vector<const char*>(out_only, out_only + 3)
        : std::vector<const char*>(all, all + 15);
    int fails = 0;
    for (const char* s : stages) {
        NpyArray ref = npy_load(fdir + "/" + std::string(s) + ".npy");
        auto it = cap.find(s);
        if (it == cap.end()) { printf("  %-9s MISSING capture\n", s); fails++; continue; }
        const NpyArray& got = it->second;
        if (got.numel() != ref.numel()) {
            printf("  %-9s SHAPE numel got=%lld ref=%lld\n", s,
                   (long long)got.numel(), (long long)ref.numel());
            fails++; continue;
        }
        float mx = max_abs_diff(got.data.data(), ref.data.data(), ref.numel());
        float mn = mean_abs_diff(got.data.data(), ref.data.data(), ref.numel());
        if (!print_result(s, mx, mn)) fails++;
    }
    printf("\n%s (%d/%zu stages failed)\n", fails == 0 ? "PARITY PASS" : "PARITY FAIL",
           fails, stages.size());
    return fails == 0 ? 0 : 1;
}

static int bench(const Fwd& fwd, int frames, int reps, bool have_ms) {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> e((size_t)257 * frames * 2), y((size_t)257 * frames * 2);
    for (auto& v : e) v = nd(rng) * 0.05f;
    for (auto& v : y) v = nd(rng) * 0.05f;

    double ms = 0;
    fwd(e.data(), y.data(), frames, nullptr, have_ms ? &ms : nullptr);  // warmup/build
    double best = 1e30, sum = 0;
    for (int r = 0; r < reps; ++r) {
        double cms = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = fwd(e.data(), y.data(), frames, nullptr, have_ms ? &cms : nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double m = have_ms ? cms : wall;  // ggml: pure compute time; scalar: wall
        best = std::min(best, m); sum += m;
        (void)out;
    }
    double avg = sum / reps;
    double secs = frames * 256.0 / 16000.0;
    printf("GTCRN-AEC forward: frames=%d (%.2f s audio), 1 thread, n=%d%s\n",
           frames, secs, reps, have_ms ? " [ggml graph compute]" : " [scalar wall]");
    printf("  per-clip: %.1f ms (avg)  %.1f ms (min)\n", avg, best);
    printf("  RTF: %.4f (avg)  %.4f (min)   [<1 = real-time on 1 core]\n",
           avg / 1000.0 / secs, best / 1000.0 / secs);
    return 0;
}

static int stft_test(GtcrnModel& m, const std::string& fdir) {
    NpyArray sin = npy_load(fdir + "/stft_in.npy");     // (1,L)
    NpyArray sref = npy_load(fdir + "/stft_out.npy");    // (1,257,T,2)
    NpyArray iref = npy_load(fdir + "/istft_out.npy");   // (1,L)
    int L = (int)sin.dim(1), T = (int)sref.dim(2);
    auto spec = m.stft(sin.data.data(), L);
    float se = max_abs_diff(spec.data(), sref.data.data(), sref.numel());
    bool ok1 = print_result("stft", se, mean_abs_diff(spec.data(), sref.data.data(), sref.numel()));
    auto y = m.istft(sref.data.data(), T, L);
    float ie = max_abs_diff(y.data(), iref.data.data(), L);
    bool ok2 = print_result("istft", ie, mean_abs_diff(y.data(), iref.data.data(), L));
    // round-trip: istft(stft(x)) vs x (interior; edges differ by design)
    auto rt = m.istft(spec.data(), T, L);
    double e = 0; for (int i = 512; i < L - 512; ++i) e = std::max(e, (double)std::fabs(rt[i] - sin.data[i]));
    printf("  stft->istft round-trip (interior) max=%.3e\n", e);
    printf("\n%s\n", (ok1 && ok2) ? "STFT PASS" : "STFT FAIL");
    return (ok1 && ok2) ? 0 : 1;
}

static int cmp_stream(GtcrnGraph& g, const std::string& fdir) {
    NpyArray e = npy_load(fdir + "/in_spec_e.npy");
    NpyArray y = npy_load(fdir + "/in_spec_y.npy");
    int T = (int)e.dim(2);
    auto a = g.forward(e.data.data(), y.data.data(), T, nullptr, nullptr);
    auto b = g.forward_stream(e.data.data(), y.data.data(), T, nullptr, nullptr);
    double mx = 0; int af = -1, at = -1, ari = -1;
    for (int f = 0; f < 257; ++f)
        for (int t = 0; t < T; ++t)
            for (int ri = 0; ri < 2; ++ri) {
                size_t i = ((size_t)f * T + t) * 2 + ri;
                double d = std::fabs(a[i] - b[i]);
                if (d > mx) { mx = d; af = f; at = t; ari = ri; }
            }
    printf("stream vs batched: max=%.3e at f=%d t=%d ri=%d (batched=%.4f stream=%.4f)\n",
           mx, af, at, ari, a[((size_t)af * T + at) * 2 + ari], b[((size_t)af * T + at) * 2 + ari]);
    // per-frame max
    for (int t = 0; t < T; ++t) {
        double fm = 0;
        for (int f = 0; f < 257; ++f) for (int ri = 0; ri < 2; ++ri) {
            size_t i = ((size_t)f * T + t) * 2 + ri; fm = std::max(fm, (double)std::fabs(a[i] - b[i]));
        }
        printf("  frame %d max=%.3e\n", t, fm);
    }
    return 0;
}

int main(int argc, char** argv) {
    const char* gguf = nullptr;
    std::string fixtures;
    bool do_bench = false, use_ggml = false, use_stream = false, do_cmp = false, do_stft = false;
    int frames = 500, reps = 10, threads = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--fixtures" && i + 1 < argc) fixtures = argv[++i];
        else if (a == "--ggml") use_ggml = true;
        else if (a == "--stream") { use_ggml = true; use_stream = true; }
        else if (a == "--cmp-stream") { use_ggml = true; do_cmp = true; }
        else if (a == "--stft") { do_stft = true; }
        else if (a == "--bench") {
            do_bench = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') frames = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') reps = atoi(argv[++i]);
        }
    }
    if (!gguf) { fprintf(stderr, "Usage: test_gtcrn --gguf m.gguf [--ggml] [--fixtures dir | --bench [frames] [reps]]\n"); return 1; }

    if (do_stft) {
        GtcrnModel m;
        if (!m.load(gguf, true)) { fprintf(stderr, "load failed\n"); return 1; }
        return stft_test(m, fixtures.empty() ? "../tests/gtcrn" : fixtures);
    }

    Fwd fwd;
    GtcrnModel scalar;
    GtcrnGraph graph;
    if (use_ggml) {
        if (!graph.load(gguf, threads, true)) { fprintf(stderr, "ggml load failed\n"); return 1; }
        if (use_stream)
            fwd = [&](const float* e, const float* y, int T, std::map<std::string, NpyArray>* c, double* ms) {
                return graph.forward_stream(e, y, T, c, ms);
            };
        else
            fwd = [&](const float* e, const float* y, int T, std::map<std::string, NpyArray>* c, double* ms) {
                return graph.forward(e, y, T, c, ms);
            };
    } else {
        if (!scalar.load(gguf, true)) { fprintf(stderr, "load failed\n"); return 1; }
        fwd = [&](const float* e, const float* y, int T, std::map<std::string, NpyArray>* c, double*) {
            return scalar.forward(e, y, T, c);
        };
    }

    if (do_cmp) return cmp_stream(graph, fixtures.empty() ? "../tests/gtcrn" : fixtures);
    if (do_bench) return bench(fwd, frames, reps, use_ggml);
    if (!fixtures.empty()) return parity(fwd, fixtures, use_stream);
    fprintf(stderr, "nothing to do — pass --fixtures or --bench\n");
    return 1;
}
