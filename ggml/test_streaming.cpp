/**
 * Streaming equivalence test: batch localvqe_process_f32() vs
 * frame-by-frame localvqe_process_frame_f32() on real audio.
 *
 * Usage:
 *   test_streaming model.gguf --audio-dirs mic_dir/ ref_dir/ [--n-pairs 5]
 */

#include "localvqe_api.h"
#include "common.h"
#include "audio_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Test 1: PCM end-to-end batch vs streaming ────────────────────────────

#include <dirent.h>

// Collect all .flac files under a directory, recursively.
static void collect_flac(const std::string& dir, std::vector<std::string>& out) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        std::string path = dir + "/" + e->d_name;
        if (e->d_type == DT_DIR) {
            collect_flac(path, out);
        } else if (e->d_type == DT_REG || e->d_type == DT_LNK) {
            size_t len = strlen(e->d_name);
            if (len > 5 && strcmp(e->d_name + len - 5, ".flac") == 0)
                out.push_back(path);
        }
    }
    closedir(d);
}

// Pick n indices spread evenly across [0, total).
static std::vector<size_t> spread_indices(size_t total, int n) {
    std::vector<size_t> idx;
    if (total == 0 || n <= 0) return idx;
    if ((size_t)n >= total) {
        for (size_t i = 0; i < total; i++) idx.push_back(i);
        return idx;
    }
    for (int i = 0; i < n; i++)
        idx.push_back((size_t)((double)i / n * total));
    return idx;
}

// Run one pair through batch + streaming, return true if they match.
static bool run_one_pcm_pair(uintptr_t ctx, const std::string& mic_path,
                              const std::string& ref_path, int pair_idx) {
    int hop = localvqe_hop_length(ctx);
    int sr = localvqe_sample_rate(ctx);
    int n_fft = localvqe_fft_size(ctx);

    std::vector<float> mic = audio_load_mono(mic_path, sr);
    std::vector<float> ref = audio_load_mono(ref_path, sr);
    if (mic.empty() || ref.empty()) {
        fprintf(stderr, "  [%d] Failed to load audio\n", pair_idx);
        return false;
    }

    int n_samples = (int)std::min(mic.size(), ref.size());
    if (n_samples < n_fft) {
        printf("  [%d] SKIP (too short: %d samples)\n", pair_idx, n_samples);
        return true;
    }
    mic.resize(n_samples);
    ref.resize(n_samples);

    // Batch
    std::vector<float> batch_out(n_samples, 0.0f);
    localvqe_reset(ctx);
    int ret = localvqe_process_f32(ctx, mic.data(), ref.data(),
                                   n_samples, batch_out.data());
    if (ret != 0) {
        fprintf(stderr, "  [%d] Batch failed: %s\n", pair_idx, localvqe_last_error(ctx));
        return false;
    }

    // Streaming
    localvqe_reset(ctx);
    int n_hops = n_samples / hop;
    std::vector<float> stream_out(n_samples, 0.0f);
    for (int h = 0; h < n_hops; h++) {
        ret = localvqe_process_frame_f32(ctx,
                                         mic.data() + h * hop,
                                         ref.data() + h * hop,
                                         hop,
                                         stream_out.data() + h * hop);
        if (ret != 0) {
            fprintf(stderr, "  [%d] Stream frame %d failed\n", pair_idx, h);
            return false;
        }
    }

    // Batch and streaming call the same per-frame path (stream_one_frame),
    // so outputs should be bit-identical aside from float roundoff. Skip
    // the first hop (warmup zero) and compare sample-for-sample.
    int skip = hop;
    int cmp_len = n_hops * hop - skip;
    if (cmp_len <= 0) {
        printf("  [%d] SKIP (too short after warmup)\n", pair_idx);
        return true;
    }

    float max_err = max_abs_diff(batch_out.data() + skip,
                                  stream_out.data() + skip, cmp_len);
    float mean_err = mean_abs_diff(batch_out.data() + skip,
                                    stream_out.data() + skip, cmp_len);
    float max_val = 0.0f;
    for (int i = 0; i < cmp_len; i++) {
        float v = std::fabs(stream_out[skip + i]);
        if (v > max_val) max_val = v;
    }

    bool pass = max_err < 1e-4f;
    printf("  [%d] %.1fs  max=%.2e mean=%.2e out=%.4f  %s\n",
           pair_idx, (float)n_samples / sr, max_err, mean_err, max_val,
           pass ? "OK" : "FAIL");
    return pass;
}

static bool test_pcm_e2e(const char* model_path,
                          const char* mic_dir, const char* ref_dir,
                          int n_pairs) {
    printf("=== Test 1: PCM end-to-end batch vs streaming (%d pairs) ===\n",
           n_pairs);

    // Collect files from both directories
    printf("  Scanning %s ...\n", mic_dir);
    std::vector<std::string> mic_files;
    collect_flac(mic_dir, mic_files);
    printf("  Scanning %s ...\n", ref_dir);
    std::vector<std::string> ref_files;
    collect_flac(ref_dir, ref_files);

    if (mic_files.empty() || ref_files.empty()) {
        fprintf(stderr, "  No FLAC files found (mic: %zu, ref: %zu)\n",
                mic_files.size(), ref_files.size());
        return false;
    }
    printf("  Found %zu mic files, %zu ref files\n",
           mic_files.size(), ref_files.size());

    // Sort for determinism, then pick n_pairs spread evenly
    std::sort(mic_files.begin(), mic_files.end());
    std::sort(ref_files.begin(), ref_files.end());
    auto mic_idx = spread_indices(mic_files.size(), n_pairs);
    auto ref_idx = spread_indices(ref_files.size(), n_pairs);

    uintptr_t ctx = localvqe_new(model_path);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return false; }

    bool all_pass = true;
    for (int i = 0; i < n_pairs; i++) {
        if (!run_one_pcm_pair(ctx, mic_files[mic_idx[i]], ref_files[ref_idx[i]], i))
            all_pass = false;
    }

    printf("  %s\n\n", all_pass ? "PASS" : "FAIL");
    localvqe_free(ctx);
    return all_pass;
}

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* audio_mic_dir = nullptr;
    const char* audio_ref_dir = nullptr;
    int n_pairs = 5;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--audio-dirs" && i + 2 < argc) {
            audio_mic_dir = argv[++i];
            audio_ref_dir = argv[++i];
        } else if (arg == "--n-pairs" && i + 1 < argc) {
            n_pairs = std::stoi(argv[++i]);
        } else if (!model_path) {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Usage: test_streaming model.gguf "
                    "--audio-dirs mic_dir/ ref_dir/ "
                    "[--n-pairs 5]\n");
            return 1;
        }
    }

    if (!model_path || !audio_mic_dir || !audio_ref_dir) {
        fprintf(stderr, "Usage: test_streaming model.gguf "
                "--audio-dirs mic_dir/ ref_dir/ "
                "[--n-pairs 5]\n");
        return 1;
    }

    bool pass = test_pcm_e2e(model_path, audio_mic_dir, audio_ref_dir, n_pairs);
    printf("=== %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
