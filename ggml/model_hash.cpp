// SHA-256 allowlist check for GGUF model files.
//
// The threat model is "we only ship a fixed set of released model
// files; any other GGUF is rejected before reaching the parser."
// This lets us treat the GGUF parsing path as trusted-input and avoid
// hardening it against malicious files.
//
// Hashes here mirror the ones in CMakeLists.txt's _dvqe_download calls
// for the released v1 / v1.1 / v1.2 F32 GGUFs. When releasing a new
// model, add its SHA-256 to kAllowed[] (and to CMakeLists.txt).

#include "model_hash.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace localvqe {

namespace {

constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

class Sha256 {
public:
    Sha256() { init(); }

    void update(const uint8_t* data, size_t len) {
        n_bits_ += (uint64_t)len * 8;
        if (buf_n_ > 0) {
            size_t take = std::min(len, 64 - buf_n_);
            std::memcpy(buf_ + buf_n_, data, take);
            buf_n_ += take; data += take; len -= take;
            if (buf_n_ == 64) { compress(buf_); buf_n_ = 0; }
        }
        while (len >= 64) { compress(data); data += 64; len -= 64; }
        if (len > 0) { std::memcpy(buf_, data, len); buf_n_ = len; }
    }

    void final(uint8_t out[32]) {
        uint64_t bits = n_bits_;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buf_n_ != 56) update(&zero, 1);
        uint8_t lenbuf[8];
        for (int i = 0; i < 8; i++) lenbuf[i] = (uint8_t)(bits >> (56 - 8*i));
        update(lenbuf, 8);
        for (int i = 0; i < 8; i++) {
            out[i*4]   = (uint8_t)(h_[i] >> 24);
            out[i*4+1] = (uint8_t)(h_[i] >> 16);
            out[i*4+2] = (uint8_t)(h_[i] >> 8);
            out[i*4+3] = (uint8_t)(h_[i]);
        }
    }

private:
    void init() {
        h_[0]=0x6a09e667; h_[1]=0xbb67ae85; h_[2]=0x3c6ef372; h_[3]=0xa54ff53a;
        h_[4]=0x510e527f; h_[5]=0x9b05688c; h_[6]=0x1f83d9ab; h_[7]=0x5be0cd19;
        n_bits_ = 0;
        buf_n_  = 0;
    }

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress(const uint8_t b[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t)b[i*4]   << 24 | (uint32_t)b[i*4+1] << 16
                 | (uint32_t)b[i*4+2] << 8  | (uint32_t)b[i*4+3];
        }
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19)  ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h_[0], b2=h_[1], c=h_[2], d=h_[3], e=h_[4], f=h_[5], g=h_[6], hh=h_[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b2) ^ (a & c) ^ (b2 & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b2; b2 = a; a = t1 + t2;
        }
        h_[0]+=a; h_[1]+=b2; h_[2]+=c; h_[3]+=d; h_[4]+=e; h_[5]+=f; h_[6]+=g; h_[7]+=hh;
    }

    uint32_t h_[8];
    uint64_t n_bits_;
    uint8_t  buf_[64];
    size_t   buf_n_;
};

bool sha256_file(const char* path, uint8_t out[32]) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "localvqe: cannot open '%s'\n", path);
        return false;
    }
    Sha256 s;
    uint8_t buf[64 * 1024];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.update(buf, n);
    bool ok = !std::ferror(f);
    std::fclose(f);
    if (!ok) {
        std::fprintf(stderr, "localvqe: read error on '%s'\n", path);
        return false;
    }
    s.final(out);
    return true;
}

void hex32(const uint8_t in[32], char out[65]) {
    static const char d[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[i*2]   = d[in[i] >> 4];
        out[i*2+1] = d[in[i] & 0xF];
    }
    out[64] = 0;
}

// SHA-256 digests of released LocalVQE GGUFs. Keep in sync with
// CMakeLists.txt:_dvqe_download(); fuzz_audio and the regression
// tests both load files whose hashes must appear here.
constexpr uint8_t kAllowed[][32] = {
    // localvqe-v1-1.3M-f32.gguf
    {0xd5,0xea,0xf5,0x77,0x44,0x9d,0x0f,0x92,0x0d,0x8e,0xe5,0xe1,0x04,0x2b,0x8d,0xdc,
     0x7b,0x66,0x27,0x31,0x3a,0x04,0x2c,0x62,0xe2,0xad,0xa1,0xb4,0x27,0x19,0xab,0x30},
    // localvqe-v1.1-1.3M-f32.gguf
    {0xc1,0x18,0x22,0x7c,0x6b,0x43,0x3d,0x6a,0xa3,0x6d,0x9e,0x4b,0x99,0x3e,0x0f,0x31,
     0xaa,0x60,0x78,0x7e,0xa3,0x8d,0x30,0x1d,0x04,0xdb,0x91,0x7a,0x4a,0x2b,0x0a,0x84},
    // localvqe-v1.2-1.3M-f32.gguf
    {0x48,0x56,0xec,0xf5,0xf5,0x22,0xb2,0x3f,0xb2,0xbc,0x5c,0xae,0xac,0x81,0xf3,0x23,
     0xc0,0xef,0x1c,0x4c,0x15,0x6a,0x9c,0x7d,0x40,0xa6,0xad,0xbe,0x09,0x2b,0xa9,0xce},
    // localvqe-v1.3-4.8M-f32.gguf
    {0xc4,0xf7,0x91,0x24,0x85,0xc3,0x2c,0xfc,0x20,0x6c,0x53,0x6f,0x2f,0x05,0x0b,0x52,
     0x51,0x3f,0x2f,0x61,0x3f,0xdb,0xc6,0x16,0x39,0x1f,0x6b,0x26,0xab,0x1d,0x51,0xec},
    // localvqe-v1.4-aec-200K-f32.gguf
    {0xb6,0xe4,0x31,0x38,0x58,0x8a,0x83,0xbf,0xe9,0x03,0xab,0x5e,0x14,0x3b,0x40,0x20,
     0xb9,0x1c,0x1e,0x16,0x29,0xf5,0xa5,0x75,0xac,0x58,0x55,0xff,0x00,0x03,0xc7,0x31},
    // localvqe-v1.4-aec-200K-bf16.gguf
    {0xc1,0x48,0x99,0xf4,0xde,0xc7,0x9e,0x1b,0xc3,0x36,0x0e,0xbb,0x98,0x45,0x62,0x1b,
     0xd3,0x2b,0x81,0xd7,0xc9,0xa6,0xc1,0xee,0x9e,0x56,0xfc,0x1d,0x7b,0x19,0x78,0xa6},
    // localvqe-v1.4-aec-2.7K-f32.gguf (front-end-only)
    {0xd7,0x9f,0x82,0x4f,0x6e,0xe6,0xf5,0x8b,0x2a,0x71,0x08,0xd1,0xdf,0x90,0xdb,0x51,
     0xc4,0x2c,0x15,0xa7,0x40,0x0a,0x5a,0xdd,0xbd,0x3b,0xe9,0x97,0x2b,0xa6,0x57,0xa4},
};

}  // namespace

bool verify_model_hash(const char* path) {
    if (const char* bypass = std::getenv("LOCALVQE_ALLOW_UNHASHED")) {
        if (bypass[0] == '1') return true;
    }
    uint8_t digest[32];
    if (!sha256_file(path, digest)) return false;
    for (const auto& allowed : kAllowed) {
        if (std::memcmp(digest, allowed, 32) == 0) return true;
    }
    char hex_str[65];
    hex32(digest, hex_str);
    std::fprintf(stderr,
        "localvqe: model file '%s' (sha256=%s) is not on the released-models"
        " allowlist. Set LOCALVQE_ALLOW_UNHASHED=1 for development.\n",
        path, hex_str);
    return false;
}

}  // namespace localvqe
