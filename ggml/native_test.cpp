/**
 * Development harness for the native engine: run it on npy inputs, dump
 * per-frame taps for parity comparison against the graph implementation,
 * benchmark, and pin regression fixtures (--reg-input mode).
 *
 *   native_test model.gguf mic.npy ref.npy out_dir [n_frames] [--bench N]
 */
#include "native_engine.h"
#include "daf_frontend.h"
#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: native_test gguf mic ref outdir [T] [--bench N]\n"
                            "       native_test gguf --reg-input in.f32 [--save out.f32] [--expected exp.f32]\n"); return 1; }
    localvqe_model m;
    if (!load_model(argv[1], m, false)) { fprintf(stderr, "load failed\n"); return 1; }
    native_engine ne;
    if (!ne_init(ne, m)) { fprintf(stderr, "init failed\n"); return 1; }

    // Regression mode: mirrors tests/test_regression.cpp for the native
    // engine. Input is the shared fixture format (16000 mic then 16000
    // ref samples, raw little-endian f32); output is the 16000-sample
    // enhanced stream. Cascade (DAF) runs when the gguf carries daf
    // tensors. NOTE: fixtures produced here are kernel-path specific —
    // native_test is -march=znver4 (AVX512BF16 conv kernels) and
    // native_test_f32 masks bf16 off (AVX512F kernels); they guard this
    // development CPU only.
    {
        std::string reg_in, reg_save, reg_exp;
        float atol = 1e-3f, rtol = 1e-2f;
        for (int i = 2; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--reg-input" && i + 1 < argc) reg_in = argv[++i];
            else if (a == "--save" && i + 1 < argc) reg_save = argv[++i];
            else if (a == "--expected" && i + 1 < argc) reg_exp = argv[++i];
            else if (a == "--atol" && i + 1 < argc) atol = (float)atof(argv[++i]);
            else if (a == "--rtol" && i + 1 < argc) rtol = (float)atof(argv[++i]);
        }
        if (!reg_in.empty()) {
            const int N = 16000, hop = 256, K = 512;
            std::vector<float> buf(2 * N);
            FILE* f = fopen(reg_in.c_str(), "rb");
            if (!f || fread(buf.data(), 4, 2 * N, f) != (size_t)2 * N) {
                fprintf(stderr, "bad --reg-input %s\n", reg_in.c_str());
                if (f) fclose(f);
                return 1;
            }
            fclose(f);
            const float* mic = buf.data();
            const float* ref = buf.data() + N;

            daf_frontend fe;
            bool daf = daf_init_npy(fe, m);
            std::vector<float> e(N), yh(N);
            if (daf) {
                daf_process(fe, mic, ref, N, e.data(), yh.data());
            } else {
                memcpy(e.data(), mic, N * 4);
                memcpy(yh.data(), ref, N * 4);
            }
            ne_debug_taps(false);
            ne_reset(ne);
            std::vector<float> hm(hop, 0), hr(hop, 0), mw(K, 0), rw(K, 0), ow(K, 0);
            std::vector<float> out(N, 0.0f);
            // Full hops only, and emit hop-quantized samples — mirrors
            // localvqe_process_f32, which leaves the partial tail zero.
            const int n_hops = N / hop;
            const int n_emit = n_hops * hop;
            for (int t = 0; t < n_hops; t++) {
                memcpy(mw.data(), hm.data(), hop * 4);
                memcpy(mw.data() + hop, e.data() + t * hop, hop * 4);
                memcpy(rw.data(), hr.data(), hop * 4);
                memcpy(rw.data() + hop, yh.data() + t * hop, hop * 4);
                memcpy(hm.data(), mw.data() + hop, hop * 4);
                memcpy(hr.data(), rw.data() + hop, hop * 4);
                ne_process_frame(ne, mw.data(), rw.data(), ow.data());
                for (int j = 0; j < K && t * hop + j < n_emit; j++)
                    out[t * hop + j] += ow[j];
            }
            printf("native %s: daf=%s\n", reg_in.c_str(), daf ? "on" : "off");
            if (!reg_save.empty()) {
                FILE* o = fopen(reg_save.c_str(), "wb");
                if (!o || fwrite(out.data(), 4, N, o) != (size_t)N) {
                    fprintf(stderr, "failed to write %s\n", reg_save.c_str());
                    if (o) fclose(o);
                    return 1;
                }
                fclose(o);
                printf("Saved %s (%d samples)\n", reg_save.c_str(), N);
                if (reg_exp.empty()) return 0;
            }
            std::vector<float> exp(N);
            FILE* x = fopen(reg_exp.c_str(), "rb");
            if (!x || fread(exp.data(), 4, N, x) != (size_t)N) {
                fprintf(stderr, "bad --expected %s\n", reg_exp.c_str());
                if (x) fclose(x);
                return 1;
            }
            fclose(x);
            float max_abs = 0, sum_abs = 0;
            int viol = 0;
            for (int i = 0; i < N; i++) {
                float d = fabsf(out[i] - exp[i]);
                if (d > atol + rtol * fabsf(exp[i])) viol++;
                if (d > max_abs) max_abs = d;
                sum_abs += d;
            }
            printf("max abs diff:  %.3e\nmean abs diff: %.3e\n", max_abs, sum_abs / N);
            printf("violations (>atol+rtol*|ref|): %d / %d (atol=%.1e, rtol=%.1e)\n",
                   viol, N, atol, rtol);
            printf(viol == 0 ? "PASS\n" : "FAIL\n");
            return viol == 0 ? 0 : 1;
        }
    }

    if (argc < 5) { fprintf(stderr, "usage: native_test gguf mic ref outdir [T] [--bench N]\n"); return 1; }
    std::string outdir = argv[4];
    int T = argc > 5 ? atoi(argv[5]) : 200;
    int bench = 0;
    for (int i = 5; i < argc - 1; i++)
        if (std::string(argv[i]) == "--bench") bench = atoi(argv[i + 1]);

    const int hop = 256, K = 512;
    std::vector<float> mw(K, 0), rw(K, 0), ow(K, 0), hm(hop, 0), hr(hop, 0);

    bool cascade = false;
    const char* outdir2 = nullptr;
    for (int i = 5; i < argc; i++) {
        if (std::string(argv[i]) == "--cascade") cascade = true;
        if (std::string(argv[i]) == "--dir" && i + 1 < argc) outdir2 = argv[i + 1];
    }
    daf_frontend fe;
    if (cascade && !daf_init_npy(fe, m)) { fprintf(stderr, "no daf tensors\n"); return 1; }

    if (cascade && outdir2) {
        // batch mode: indir holds mic_%04d.npy / ref_%04d.npy
        for (int idx = 0;; idx++) {
            char mp[512], rp[512], op[512];
            snprintf(mp, sizeof(mp), "%s/mic_%04d.npy", argv[2], idx);
            snprintf(rp, sizeof(rp), "%s/ref_%04d.npy", argv[2], idx);
            NpyArray mm, rr;
            try { mm = npy_load(mp); rr = npy_load(rp); } catch (...) { break; }
            if (mm.data.empty()) break;
            int n = (int)std::min(mm.numel(), rr.numel()) / hop * hop;
            std::vector<float> e(n), yh(n), out(n + hop, 0.0f);
            ne_reset(ne); daf_reset(fe);
            std::vector<float> hm2(hop, 0), hr2(hop, 0), mw2(K, 0), rw2(K, 0), ow2(K, 0);
            daf_process(fe, mm.data.data(), rr.data.data(), n, e.data(), yh.data());
            for (int t = 0; t * hop < n; t++) {
                memcpy(mw2.data(), hm2.data(), hop*4);
                memcpy(mw2.data()+hop, e.data()+t*hop, hop*4);
                memcpy(rw2.data(), hr2.data(), hop*4);
                memcpy(rw2.data()+hop, yh.data()+t*hop, hop*4);
                memcpy(hm2.data(), mw2.data()+hop, hop*4);
                memcpy(hr2.data(), rw2.data()+hop, hop*4);
                ne_process_frame(ne, mw2.data(), rw2.data(), ow2.data());
                for (int j = 0; j < K && t*hop + j < n + hop; j++)
                    out[t*hop + j] += ow2[j];
            }
            snprintf(op, sizeof(op), "%s/enhanced_%04d.npy", outdir2, idx);
            npy_save(op, out.data(), {(int64_t)n});
            if (idx % 100 == 0) printf("  clip %d\n", idx);
        }
        printf("batch done\n");
        return 0;
    }

    NpyArray mic = npy_load(argv[2]), ref = npy_load(argv[3]);
    int Tmax = (int)std::min(mic.numel(), ref.numel()) / hop;
    T = std::min(T, Tmax);

    if (bench && cascade) {
        ne_debug_taps(false);
        int n = Tmax * hop;
        std::vector<float> e(n), yh(n);
        auto t0 = std::chrono::steady_clock::now();
        int hops = 0;
        for (int it = 0; it < bench; it++) {
            ne_reset(ne); daf_reset(fe);
            std::vector<float> hm2(hop, 0), hr2(hop, 0), mw2(K, 0), rw2(K, 0), ow2(K, 0);
            daf_process(fe, mic.data.data(), ref.data.data(), n, e.data(), yh.data());
            for (int t = 0; t < Tmax; t++, hops++) {
                memcpy(mw2.data(), hm2.data(), hop*4);
                memcpy(mw2.data()+hop, e.data()+t*hop, hop*4);
                memcpy(rw2.data(), hr2.data(), hop*4);
                memcpy(rw2.data()+hop, yh.data()+t*hop, hop*4);
                memcpy(hm2.data(), mw2.data()+hop, hop*4);
                memcpy(hr2.data(), rw2.data()+hop, hop*4);
                ne_process_frame(ne, mw2.data(), rw2.data(), ow2.data());
            }
        }
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        printf("native CASCADE: %.3f ms/hop -> %.2fx realtime\n", ms / hops, 16.0 / (ms / hops));
        return 0;
    }

    if (bench) {
        ne_debug_taps(false);
        // warmup
        for (int t = 0; t < 50 && t < Tmax; t++) {
            memcpy(mw.data(), hm.data(), hop*4); memcpy(mw.data()+hop, mic.data.data()+t*hop, hop*4);
            memcpy(rw.data(), hr.data(), hop*4); memcpy(rw.data()+hop, ref.data.data()+t*hop, hop*4);
            memcpy(hm.data(), mw.data()+hop, hop*4); memcpy(hr.data(), rw.data()+hop, hop*4);
            ne_process_frame(ne, mw.data(), rw.data(), ow.data());
        }
        ne_reset(ne);
        std::fill(hm.begin(), hm.end(), 0.0f); std::fill(hr.begin(), hr.end(), 0.0f);
        auto t0 = std::chrono::steady_clock::now();
        int n = 0;
        for (int it = 0; it < bench; it++) {
            ne_reset(ne);
            for (int t = 0; t < Tmax; t++, n++) {
                memcpy(mw.data(), hm.data(), hop*4); memcpy(mw.data()+hop, mic.data.data()+t*hop, hop*4);
                memcpy(rw.data(), hr.data(), hop*4); memcpy(rw.data()+hop, ref.data.data()+t*hop, hop*4);
                memcpy(hm.data(), mw.data()+hop, hop*4); memcpy(hr.data(), rw.data()+hop, hop*4);
                ne_process_frame(ne, mw.data(), rw.data(), ow.data());
            }
        }
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        printf("native: %.3f ms/hop (%d hops) -> %.2fx realtime\n",
               ms / n, n, 16.0 / (ms / n));
        return 0;
    }

    ne_debug_taps(true);
    std::map<std::string, std::vector<float>> acc;
    std::vector<float> enh_all;
    for (int t = 0; t < T; t++) {
        memcpy(mw.data(), hm.data(), hop*4); memcpy(mw.data()+hop, mic.data.data()+t*hop, hop*4);
        memcpy(rw.data(), hr.data(), hop*4); memcpy(rw.data()+hop, ref.data.data()+t*hop, hop*4);
        memcpy(hm.data(), mw.data()+hop, hop*4); memcpy(hr.data(), rw.data()+hop, hop*4);
        ne_process_frame(ne, mw.data(), rw.data(), ow.data());
        for (auto& kv : ne_taps()) {
            auto& v = acc[kv.first];
            v.insert(v.end(), kv.second.begin(), kv.second.end());
        }
        enh_all.insert(enh_all.end(), ow.begin(), ow.end());
    }
    for (auto& kv : acc) {
        int64_t per = (int64_t)(kv.second.size() / T);
        npy_save(outdir + "/ne_" + kv.first + ".npy", kv.second.data(),
                 {(int64_t)T, per});
    }
    npy_save(outdir + "/ne_outwin.npy", enh_all.data(), {(int64_t)T, 512});
    printf("native taps dumped: %d frames, %zu taps\n", T, acc.size());
    return 0;
}
