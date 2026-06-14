// libFuzzer harness for the HAND-ROLLED implementations: the DAF front-end
// (daf_frontend.cpp) and the native streaming engine (native_engine.cpp).
// One gguf is loaded on first call (LOCALVQE_FUZZ_MODEL, must carry daf.*
// tensors). Fuzzer bytes are reinterpreted as PCM floats — this covers
// NaN/Inf/denormal injection for free — and drive variable-length hops
// through daf_process and ne_process_frame, with periodic resets.
#include "../native_engine.h"
#include "../daf_frontend.h"
#include "../common.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace {
localvqe_model* g_m = nullptr;
native_engine* g_ne = nullptr;
daf_frontend* g_fe = nullptr;

bool init() {
    const char* p = std::getenv("LOCALVQE_FUZZ_MODEL");
    if (!p) p = "bench_assets/localvqe-v1.4-aec-200K-f32.gguf";
    g_m = new localvqe_model();
    if (!load_model(p, *g_m, false)) return false;
    g_ne = new native_engine();
    g_fe = new daf_frontend();
    if (!ne_init(*g_ne, *g_m)) return false;
    if (!daf_init_npy(*g_fe, *g_m)) return false;
    return true;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static bool ok = init();
    if (!ok) return 0;
    if (size < 8) return 0;

    // control byte: reset behaviour + number of frames
    const uint8_t ctl = data[0];
    data++; size--;
    if (ctl & 1) { ne_reset(*g_ne); daf_reset(*g_fe); }

    const int hop = 256, K = 512;
    const size_t max_frames = 8;
    size_t n_float = size / 4;
    std::vector<float> pcm(n_float);
    std::memcpy(pcm.data(), data, n_float * 4);

    std::vector<float> mic(hop), ref(hop), e(hop), yh(hop);
    std::vector<float> mw(K, 0), rw(K, 0), ow(K, 0), hm(hop, 0), hr(hop, 0);
    size_t pos = 0;
    for (size_t f = 0; f < max_frames; f++) {
        for (int i = 0; i < hop; i++) {
            mic[i] = (pos < n_float) ? pcm[pos++] : 0.0f;
            ref[i] = (pos < n_float) ? pcm[pos++] : 0.0f;
        }
        daf_process(*g_fe, mic.data(), ref.data(), hop, e.data(), yh.data());
        std::memcpy(mw.data(), hm.data(), hop * 4);
        std::memcpy(mw.data() + hop, e.data(), hop * 4);
        std::memcpy(rw.data(), hr.data(), hop * 4);
        std::memcpy(rw.data() + hop, yh.data(), hop * 4);
        std::memcpy(hm.data(), mw.data() + hop, hop * 4);
        std::memcpy(hr.data(), rw.data() + hop, hop * 4);
        ne_process_frame(*g_ne, mw.data(), rw.data(), ow.data());
        if (pos >= n_float && (ctl & 2)) break;
    }
    return 0;
}
