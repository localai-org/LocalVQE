/**
 * Run GGML model on validation data, save enhanced audio as .npy.
 *
 * Quality scoring (DNSMOS, AECMOS, PESQ, STOI) is done in Python:
 *   make -C train eval-ggml
 *
 * Usage:
 *   eval_graph model.gguf --val-dir eval_output/val_audio --save-dir eval_output/ggml_enhanced [--max N]
 */

#include "localvqe_api.h"
#include "common.h"
#include "audio_io.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* val_dir = nullptr;
    const char* save_dir = nullptr;
    int max_samples = 10000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--val-dir" && i + 1 < argc) val_dir = argv[++i];
        else if (arg == "--save-dir" && i + 1 < argc) save_dir = argv[++i];
        else if (arg == "--max" && i + 1 < argc) max_samples = std::stoi(argv[++i]);
        else if (!model_path) model_path = argv[i];
    }

    if (!model_path || !val_dir || !save_dir) {
        fprintf(stderr, "Usage: eval_graph model.gguf --val-dir DIR --save-dir DIR [--max N]\n");
        return 1;
    }

    uintptr_t ctx = localvqe_new(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    int n_total = 0;

    for (int idx = 0; idx < max_samples; idx++) {
        char mic_path[512], ref_path[512];
        snprintf(mic_path, sizeof(mic_path), "%s/mic_%04d.npy", val_dir, idx);
        snprintf(ref_path, sizeof(ref_path), "%s/ref_%04d.npy", val_dir, idx);

        NpyArray mic_npy, ref_npy;
        try {
            mic_npy = npy_load(mic_path);
            ref_npy = npy_load(ref_path);
        } catch (...) { break; }
        if (mic_npy.data.empty()) break;

        int len = (int)std::min(mic_npy.numel(), ref_npy.numel());
        if (len < 512) continue;

        std::vector<float> enhanced(len);
        localvqe_reset(ctx);
        int ret = localvqe_process_f32(ctx, mic_npy.data.data(), ref_npy.data.data(),
                                      len, enhanced.data());
        if (ret != 0) {
            fprintf(stderr, "  [%d] process error: %s\n", idx, localvqe_last_error(ctx));
            continue;
        }

        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s/enhanced_%04d.npy", save_dir, idx);
        npy_save(out_path, enhanced.data(), {(int64_t)len});
        n_total++;

        if (n_total % 50 == 0 || n_total <= 5)
            printf("  [%d] saved\n", n_total);
    }

    if (n_total == 0) {
        fprintf(stderr, "No samples found in %s\n", val_dir);
        localvqe_free(ctx);
        return 1;
    }

    printf("Saved %d enhanced files to %s\n", n_total, save_dir);
    printf("Score with: make -C train eval-ggml\n");

    localvqe_free(ctx);
    return 0;
}
