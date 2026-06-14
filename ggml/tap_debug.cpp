/**
 * Per-layer tap tool: run the streaming graph frame-by-frame and dump the
 * time series of internal tensors as .npy for PyTorch parity comparison.
 *
 * Dumps (per frame, concatenated over T):
 *   tap_micin.npy      (T, 2, F)   post-analysis mic spectrum (graph input)
 *   tap_refin.npy      (T, 2, F)   post-analysis ref spectrum
 *   tap_enh.npy        (T, 2, F)   pre-synthesis enhanced spectrum
 *   tap_conv<i>.npy    (T, F_i, C_i)  newest frame of conv layer i's INPUT
 *                                     history (== that layer's input at t)
 *
 * Usage: tap_debug model.gguf mic.npy ref.npy out_dir [n_frames]
 */
#include "localvqe_graph.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Stride-aware fetch: several stream-graph tensors (mic_in, enhanced_out)
// are VIEWS — a flat tensor_get would read raw underlying memory in the
// wrong order. Read element-by-element via nb[] offsets (debug tool; speed
// is irrelevant). Output is C-order with ne0 fastest.
static void fetch(struct ggml_tensor* t, std::vector<float>& buf) {
    size_t n = ggml_nelements(t);
    buf.resize(n);
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                    size_t off = i0 * t->nb[0] + i1 * t->nb[1] +
                                 i2 * t->nb[2] + i3 * t->nb[3];
                    ggml_backend_tensor_get(t, &buf[idx++], off, sizeof(float));
                }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: tap_debug model.gguf mic.npy ref.npy out_dir [n_frames]\n");
        return 1;
    }
    const char* model_path = argv[1];
    NpyArray mic = npy_load(argv[2]);
    NpyArray ref = npy_load(argv[3]);
    std::string outdir = argv[4];
    int max_frames = argc > 5 ? atoi(argv[5]) : 1 << 30;

    dvqe_graph_model m;
    if (!load_graph_model(model_path, m, false)) { fprintf(stderr, "load failed\n"); return 1; }
    // Dump loaded bottleneck weights for parity checks (the loader may
    // reinterpret shapes; this shows the bytes the graph actually uses).
    for (auto& kv : m.weights) {
        if (kv.first.rfind("bottleneck.", 0) == 0) {
            std::vector<float> wbuf;
            fetch(kv.second, wbuf);
            std::string nm = kv.first;
            for (auto& ch : nm) if (ch == '.') ch = '_';
            npy_save(outdir + "/w_" + nm + ".npy", wbuf.data(),
                     {(int64_t)wbuf.size()});
        }
    }

    dvqe_stream_graph sg;
    if (!build_stream_graph(m, sg)) { fprintf(stderr, "graph failed\n"); return 1; }
    reset_stream_graph(sg, m);

    const int hop = m.hparams.hop_length;     // 256
    const int K   = m.hparams.n_fft;          // 512
    int n = (int)std::min(mic.numel(), ref.numel());
    int T = std::min(n / hop, max_frames);

    std::vector<float> micw(K, 0.0f), refw(K, 0.0f), outw(K, 0.0f);
    std::vector<float> hist_mic(hop, 0.0f), hist_ref(hop, 0.0f);

    std::vector<float> micin_all, refin_all, enh_all, tmp;
    std::vector<float> hr_all, hi_all;
    std::vector<std::vector<float>> dbg_all;
    std::vector<std::vector<int64_t>> dbg_shape;
    size_t n_conv = sg.conv_hist_in.size();
    std::vector<std::vector<float>> conv_all(n_conv);
    std::vector<std::vector<int64_t>> conv_shape(n_conv);

    for (int t = 0; t < T; t++) {
        memcpy(micw.data(), hist_mic.data(), hop * sizeof(float));
        memcpy(micw.data() + hop, mic.data.data() + t * hop, hop * sizeof(float));
        memcpy(refw.data(), hist_ref.data(), hop * sizeof(float));
        memcpy(refw.data() + hop, ref.data.data() + t * hop, hop * sizeof(float));
        memcpy(hist_mic.data(), micw.data() + hop, hop * sizeof(float));
        memcpy(hist_ref.data(), refw.data() + hop, hop * sizeof(float));

        process_frame_graph(sg, m, micw.data(), refw.data(), outw.data());

        fetch(sg.mic_in, tmp);  micin_all.insert(micin_all.end(), tmp.begin(), tmp.end());
        if (sg.s4d_h_real_out) {
            fetch(sg.s4d_h_real_out, tmp); hr_all.insert(hr_all.end(), tmp.begin(), tmp.end());
            fetch(sg.s4d_h_imag_out, tmp); hi_all.insert(hi_all.end(), tmp.begin(), tmp.end());
        }
        for (size_t i = 0; i < sg.dbg_taps.size(); i++) {
            fetch(sg.dbg_taps[i], tmp);
            if (dbg_all.size() <= i) { dbg_all.resize(i + 1); dbg_shape.resize(i + 1); }
            if (dbg_shape[i].empty())
                dbg_shape[i] = {sg.dbg_taps[i]->ne[0], sg.dbg_taps[i]->ne[1],
                                sg.dbg_taps[i]->ne[2], sg.dbg_taps[i]->ne[3]};
            dbg_all[i].insert(dbg_all[i].end(), tmp.begin(), tmp.end());
        }
        fetch(sg.ref_in, tmp);  refin_all.insert(refin_all.end(), tmp.begin(), tmp.end());
        fetch(sg.enhanced_out, tmp); enh_all.insert(enh_all.end(), tmp.begin(), tmp.end());
        for (size_t i = 0; i < n_conv; i++) {
            struct ggml_tensor* h = sg.conv_hist_out[i];   // (F, kh-1, C)
            fetch(h, tmp);
            int64_t F = h->ne[0], KH1 = h->ne[1], C = h->ne[2];
            if (conv_shape[i].empty()) conv_shape[i] = {F, KH1, C};
            // newest frame = index KH1-1 along ne[1]
            for (int64_t c = 0; c < C; c++)
                for (int64_t f = 0; f < F; f++)
                    conv_all[i].push_back(tmp[c * KH1 * F + (KH1 - 1) * F + f]);
        }
    }

    int64_t F = sg.mic_in->ne[2];
    npy_save(outdir + "/tap_micin.npy", micin_all.data(), {(int64_t)T, 2, 1, F});
    npy_save(outdir + "/tap_refin.npy", refin_all.data(), {(int64_t)T, 2, 1, F});
    npy_save(outdir + "/tap_enh.npy",   enh_all.data(),   {(int64_t)T, 2, 1, F});
    for (size_t i = 0; i < n_conv; i++) {
        char nm[64]; snprintf(nm, sizeof(nm), "/tap_conv%02zu.npy", i);
        npy_save(outdir + nm, conv_all[i].data(),
                 {(int64_t)T, conv_shape[i][2], conv_shape[i][0]});
        printf("conv%02zu: F=%lld kh-1=%lld C=%lld\n", i,
               (long long)conv_shape[i][0], (long long)conv_shape[i][1],
               (long long)conv_shape[i][2]);
    }
    for (size_t i = 0; i < dbg_all.size(); i++) {
        auto& sh = dbg_shape[i];
        npy_save(outdir + "/dbg_" + sg.dbg_names[i] + ".npy", dbg_all[i].data(),
                 {(int64_t)T, sh[3], sh[2], sh[1], sh[0]});
        printf("dbg %-10s ne=(%lld,%lld,%lld,%lld)\n", sg.dbg_names[i].c_str(),
               (long long)sh[0], (long long)sh[1], (long long)sh[2], (long long)sh[3]);
    }
    if (!hr_all.empty()) {
        int64_t H = sg.s4d_h_real_out->ne[0];
        npy_save(outdir + "/dbg_s4d_hr.npy", hr_all.data(), {(int64_t)T, H});
        npy_save(outdir + "/dbg_s4d_hi.npy", hi_all.data(), {(int64_t)T, H});
    }
    printf("dumped %d frames to %s (%zu conv taps)\n", T, outdir.c_str(), n_conv);
    free_stream_graph(sg);
    free_graph_model(m);
    return 0;
}
