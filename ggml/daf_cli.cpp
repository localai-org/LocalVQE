/**
 * Stage-1 (DAF) standalone CLI for parity tests and debugging.
 * Usage: daf_cli model.gguf mic.npy ref.npy e_out.npy [yhat_out.npy] [--no-prealign]
 */
#include "daf_frontend.h"
#include "common.h"
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: daf_cli gguf mic ref e_out [yhat_out] [--no-prealign]\n"); return 1; }
    bool prealign = true;
    const char* yout = nullptr;
    for (int i = 5; i < argc; i++) {
        if (std::string(argv[i]) == "--no-prealign") prealign = false;
        else yout = argv[i];
    }
    dvqe_graph_model m;
    if (!load_graph_model(argv[1], m, false)) return 1;
    daf_frontend fe;
    if (!daf_init(fe, m)) { fprintf(stderr, "no daf tensors in gguf\n"); return 1; }
    fe.enable_prealign = prealign;
    NpyArray mic = npy_load(argv[2]), ref = npy_load(argv[3]);
    int n = (int)std::min(mic.numel(), ref.numel());
    n = n / daf_frontend::M * daf_frontend::M;
    std::vector<float> e(n), yh(n);
    // Standalone: prime the bulk delay over the whole signal (file/batch use),
    // so the filter is aligned from frame 0 instead of acquiring over ~1-3 s.
    if (prealign) daf_prime_delay(fe, mic.data.data(), ref.data.data(), n);
    daf_process(fe, mic.data.data(), ref.data.data(), n, e.data(), yh.data());
    npy_save(argv[4], e.data(), {(int64_t)n});
    if (yout) npy_save(yout, yh.data(), {(int64_t)n});
    printf("daf: %d samples processed (prealign=%d, final shift=%d)\n", n, (int)prealign, fe.cur_shift);
    free_graph_model(m);
    return 0;
}
