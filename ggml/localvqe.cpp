/**
 * LocalVQE AEC inference CLI (C API).
 *
 * Usage:
 *   ./build/bin/localvqe model.gguf --input-npy mic.npy ref.npy --output enhanced.npy
 *   ./build/bin/localvqe model.gguf --in-wav mic.wav ref.wav --out-wav enhanced.wav
 *
 * Compact / low-power line (GTCRN-AEC, arch="gtcrn"): the net runs on a v1.4-AEC
 * DAF residual, but the front-end is resolved automatically (embedded in the
 * model gguf, else a localvqe-v1.4-aec-*.gguf beside it), so the command is
 * identical to any other model:
 *   ./build/bin/localvqe model.gguf --in-wav mic.wav ref.wav --out-wav enhanced.wav
 * Override the front-end explicitly with --fe <v1.4-aec.gguf>.
 */

#include "localvqe_api.h"
#include "common.h"
#include "audio_io.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* mic_path   = nullptr;
    const char* ref_path   = nullptr;
    const char* fe_path    = nullptr;   // DAF front-end gguf (GTCRN models)
    std::string out_path   = "enhanced.npy";
    bool wav_mode          = false;
    int  threads           = 0;         // 0 = library default

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "--input-npy" || a == "--in-wav") && i + 2 < argc) {
            wav_mode = (a == "--in-wav");
            mic_path = argv[++i];
            ref_path = argv[++i];
        } else if ((a == "--output" || a == "--out-wav") && i + 1 < argc) {
            out_path = argv[++i];
            if (a == "--out-wav") wav_mode = true;
        } else if (a == "--fe" && i + 1 < argc) {
            fe_path = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            fprintf(stderr,
                "Usage: localvqe <model.gguf> --input-npy mic.npy ref.npy [--output enhanced.npy]\n"
                "       localvqe <model.gguf> --in-wav mic.wav ref.wav [--out-wav enhanced.wav]\n"
                "GTCRN (compact / low-power) models use a v1.4-AEC DAF front-end, found\n"
                "automatically (embedded in the model, or a localvqe-v1.4-aec-*.gguf beside\n"
                "it); override with --fe <v1.4-aec.gguf>.\n");
            return 0;
        } else if (!model_path) {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (!model_path || !mic_path || !ref_path) {
        fprintf(stderr,
            "Usage: localvqe <model.gguf> --input-npy mic.npy ref.npy [--output enhanced.npy]\n"
            "       localvqe <model.gguf> --in-wav mic.wav ref.wav [--out-wav enhanced.wav]\n");
        return 1;
    }

    // Load PCM inputs
    std::vector<float> mic_pcm, ref_pcm;
    if (wav_mode) {
        mic_pcm = audio_load_mono(mic_path);
        ref_pcm = audio_load_mono(ref_path);
        if (mic_pcm.empty() || ref_pcm.empty()) return 1;
    } else {
        NpyArray mic = npy_load(mic_path);
        NpyArray ref = npy_load(ref_path);
        mic_pcm.assign(mic.data.begin(), mic.data.end());
        ref_pcm.assign(ref.data.begin(), ref.data.end());
    }

    int n_mic = (int)mic_pcm.size();
    int n_ref = (int)ref_pcm.size();
    if (n_mic != n_ref) {
        fprintf(stderr, "Warning: mic (%d) and ref (%d) sample counts differ; "
                "truncating to min\n", n_mic, n_ref);
        int n_min = std::min(n_mic, n_ref);
        mic_pcm.resize(n_min);
        ref_pcm.resize(n_min);
        n_mic = n_min;
    }
    printf("Input: %d samples (%.2f s)\n", n_mic, n_mic / 16000.0f);

    // Init model. GTCRN models carry no front-end, so pass one with --fe; the
    // C API detects the arch and wires the DAF front-end + GTCRN backend.
    localvqe_ctx_t ctx;
    if (fe_path || threads) {
        localvqe_options_t opts = localvqe_options_new();
        localvqe_options_set_model_path(opts, model_path);
        if (fe_path) localvqe_options_set_frontend_path(opts, fe_path);
        if (threads) localvqe_options_set_threads(opts, threads);
        ctx = localvqe_new_with_options(opts);
        localvqe_options_free(opts);
    } else {
        ctx = localvqe_new(model_path);
    }
    if (!ctx) {
        fprintf(stderr, "Error: failed to load model: %s\n", model_path);
        return 1;
    }

    // Process
    std::vector<float> enhanced(n_mic);
    int ret = localvqe_process_f32(ctx, mic_pcm.data(), ref_pcm.data(), n_mic, enhanced.data());
    if (ret != 0) {
        fprintf(stderr, "Error: localvqe_process_f32 returned %d: %s\n",
                ret, localvqe_last_error(ctx));
        localvqe_free(ctx);
        return 1;
    }

    // Save output
    if (wav_mode || (out_path.size() >= 4 &&
        out_path.compare(out_path.size() - 4, 4, ".wav") == 0)) {
        if (!audio_save_wav(out_path, enhanced.data(), n_mic)) {
            localvqe_free(ctx);
            return 1;
        }
    } else {
        npy_save(out_path, enhanced.data(), {(int64_t)n_mic});
    }
    printf("Saved enhanced to: %s\n", out_path.c_str());

    localvqe_free(ctx);
    return 0;
}
