#pragma once

/**
 * LocalVQE model — internal C++ structs.
 * Used by both the CLI tool and the C API shared library.
 */

#include "common.h"

#include <map>
#include <string>
#include <vector>

struct localvqe_hparams {
    int n_fft            = 512;
    int hop_length       = 256;
    int n_freq_bins      = 256;
    int sample_rate      = 16000;
    int dmax             = 32;
    int align_hidden     = 32;
    float power_law_c    = 0.3f;
    bool bn_folded       = true;
    int kernel_size_h    = 4;
    int kernel_size_w    = 3;  // overridden by GGUF; default matches legacy
    int bottleneck_hidden = 0;  // 0 = auto (C * F / 2)
    // 1 = post-conv BN folded into conv weights, ELU activation (legacy v1).
    // 2 = pre-norm CausalGroupNorm + ReLU6 (v1.1); norm tensors live alongside
    //     conv weights in the GGUF and run at inference.
    // 3 = pre-norm CausalGroupNorm + SiLU (v1.2 onward); identical structure
    //     to v=2, only the activation changes.
    int version          = 1;
    // Front-end-only build (the 2.7K release): the GGUF carries daf.*
    // tensors but no mask-backend weights. The engine skips the mask
    // graph entirely and emits the adaptive filter's output `e`.
    bool daf_standalone  = false;
    std::vector<int> mic_channels;
    std::vector<int> far_channels;
};

struct localvqe_model {
    localvqe_hparams hparams;
    std::map<std::string, NpyArray> tensors;
};

// ── Public API ────────────────────────────────────────────────────────────

/// Load model from GGUF file (tensor data as NpyArray for block tests).
/// For inference, use load_graph_model() in localvqe_graph.h instead.
bool load_model(const char* path, localvqe_model& model, bool verbose = true);
