# LocalVQE

**Local Voice Quality Enhancement** — a compact neural model for joint
acoustic echo cancellation (AEC), noise suppression, and dereverberation of
16 kHz speech, designed to run on commodity CPUs in real time.

- ~0.9 M parameters (~3.5 MB F32)
- ~1.66 ms per 16 ms frame on Zen4 (24 threads) — **≈9.6× realtime**
- Causal, streaming: 256-sample hop, 16 ms algorithmic latency
- F32 reference inference in C++ via [GGML](https://github.com/ggml-org/ggml);
  PyTorch reference included for verification and research
- Quantization-friendly by design (power-of-2 channel widths, kernel area 16)
  to support future Q4_K / Q8_0 native inference
- Apache 2.0

LocalVQE is a derivative of **DeepVQE**
([Indenbom et al., Interspeech 2023](https://arxiv.org/abs/2306.03177)).
It keeps DeepVQE's overall topology (mic/far-end encoders, soft-delay cross
attention, decoder with sub-pixel upsampling, complex convolving mask) but
replaces the STFT with an in-graph DCT-II filterbank, swaps the GRU
bottleneck for a diagonal state-space model (S4D), and is ~9× smaller than
the reference DeepVQE. Everything specific to LocalVQE is original to this
repository — there is no LocalVQE paper.

## Why this, and not a classical AEC/NS stack?

Hand-tuned DSP pipelines (NLMS/AP/Kalman AEC, Wiener/spectral-subtraction
NS, MCRA noise tracking, RLS dereverb) can run in tens of microseconds per
frame and remain a strong baseline when the acoustic path is benign. LocalVQE
is interesting when you want:

- **Robustness to non-linear echo paths** (small loudspeakers, handheld
  devices, plastic laptop chassis) where linear AEC leaves residual echo.
- **Non-stationary noise suppression** (babble, keyboards, fans changing
  speed) that energy-based noise estimators struggle with.
- **One model, many conditions** — no per-device tuning of step sizes,
  forgetting factors, or VAD thresholds.
- **A single deterministic causal pass** — no double-talk detector, no
  adaptation state that can diverge.

The trade-off is CPU: a classical stack might cost ~0.1 ms/frame, LocalVQE
~1–2 ms/frame. On anything larger than a microcontroller that's still a
small fraction of a real-time budget.

## Why this, and not DeepVQE?

| | DeepVQE (reference) | LocalVQE |
|---|---|---|
| Parameters | ~7.5 M | ~0.9 M |
| Weights (F32) | ~30 MB | ~3.5 MB |
| Analysis | STFT (complex FFT) | DCT-II (real, in-graph) |
| Bottleneck | GRU | S4D (diagonal state space) |
| CCM arithmetic | Complex | Real-valued (GGML-friendly) |
| Streaming inference | Not published | Yes, in this repo |

The smaller parameter count comes from iterative channel pruning of the
full-width reference, not from distillation; S4D halves the bottleneck
parameter count vs GRU at similar quality.

## Model Weights

Pre-trained weights are published on Hugging Face at
[LocalAI-io/LocalVQE](https://huggingface.co/LocalAI-io/LocalVQE):

| Variant | File | Description |
|---|---|---|
| v1 F32 | `localvqe-v1-f32.gguf` | DNS5 pre-training + ICASSP 2022/2023 AEC Challenge fine-tune |

Only F32 GGUF is published today. A `quantize` tool is included in the C++
build (see below) and the architecture is designed to be Q4_K / Q8_0
friendly, but quantized weights have not yet been calibrated and released.

## Validation Results

Numbers below are from the best checkpoint of the AEC fine-tune
(`localvqe-v1-f32.gguf`), evaluated on a 1 000-clip validation split mixing
DNS5-synthesised near/far-end scenes and ICASSP AEC Challenge synthetic
data. AECMOS scores are computed over a 100-clip sub-sample per the standard
AEC Challenge protocol.

| Metric | Overall | Single-talk far-end | Double-talk |
|---|---:|---:|---:|
| ERLE | — | **+52.2 dB** | — |
| AECMOS echo (↑, 1–5) | 4.36 | 4.46 | 4.33 |
| AECMOS degradation (↑, 1–5) | 4.83 | 5.00 | 4.78 |

- **ERLE** (Echo Return Loss Enhancement) in dB — higher is better. Only
  reported for single-talk far-end, where the mic signal is pure echo and the
  ratio `10·log10(E[mic²] / E[enh²])` directly measures echo attenuation.
  Overall and double-talk ERLE are omitted because near-end speech in the
  mic and enhanced signals dominates the numerator/denominator and the
  number stops being a clean echo-removal measurement.
- **AECMOS** (Purin et al., ICASSP 2022) is Microsoft's non-intrusive AEC
  quality predictor. "Echo" rates how well the echo was removed; "degradation"
  rates how clean the resulting speech/residual is. Both are on a 1–5 MOS
  scale, higher is better.

### Why DNSMOS OVRL is not reported here

We track DNSMOS P.808 (`sig_bak_ovr.onnx`) in TensorBoard but are deliberately
*not* publishing OVRL numbers for this model. The scores we obtain (around 2.0
overall, 2.1 on single-talk far-end) contradict informal listening —
single-talk far-end with 52 dB of cancellation is audibly near-silent, not a
"2-out-of-5" output. We suspect our DNSMOS invocation (input normalisation,
silence handling, or ONNX model variant) is miscalibrated for AEC outputs
and in particular for near-silent clips, which are out of distribution for a
speech-quality predictor. Until we can reconcile the numbers with a
DeepVQE-matching protocol we consider our OVRL numbers untrustworthy and
omit them rather than publish misleading figures.

## Repository Layout

```
ggml/        C++ streaming inference (GGML graph, CLI, C API, tests)
pytorch/     PyTorch reference implementation (model definition only)
ARCHITECTURE.md
CITATION.cff
LICENSE
flake.nix
```

## Building the C++ Inference Engine

Requires CMake ≥ 3.20 and a C++17 compiler. A [Nix](https://nixos.org/)
flake is provided:

```bash
git clone --recursive https://github.com/LocalAI-io/LocalVQE.git
cd LocalVQE

# With Nix:
nix develop
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

# Without Nix — install cmake, gcc/clang, pkg-config, libsndfile, then:
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)
```

Binaries land in `ggml/build/bin/`. The CPU build produces multiple
`libggml-cpu-*.so` variants (SSE4.2 / AVX2 / AVX-512) selected at runtime.
Keep the binaries and `.so` files together.

### Vulkan backend (embedded / integrated-GPU targets)

Add `-DLOCALVQE_VULKAN=ON` to the configure step. This composes with the
CPU build — an additional `libggml-vulkan.so` is produced in
`ggml/build/bin/` and the runtime loader picks it up when a Vulkan ICD is
present, otherwise it falls back to the CPU variants.

```bash
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release -DLOCALVQE_VULKAN=ON
cmake --build ggml/build -j$(nproc)
```

The Nix flake's dev shell already includes `vulkan-loader`,
`vulkan-headers`, and `shaderc`. Without Nix, install the equivalents
from your distro (Debian: `libvulkan-dev vulkan-headers
glslc`/`shaderc`).

## Running Inference

### CLI

```bash
./ggml/build/bin/localvqe localvqe-v1-f32.gguf \
    --in-wav mic.wav ref.wav \
    --out-wav enhanced.wav
```

Expects 16 kHz mono PCM for both mic and far-end reference.

### Benchmark

```bash
./ggml/build/bin/bench localvqe-v1-f32.gguf \
    --in-wav mic.wav ref.wav --iters 10 --profile
```

### Shared Library (C API)

```bash
cmake -S ggml -B ggml/build -DLOCALVQE_BUILD_SHARED=ON
cmake --build ggml/build -j$(nproc)
```

Produces `liblocalvqe.so` with the API in `ggml/localvqe_api.h`. See
`ggml/example_purego_test.go` for a Go / `purego` integration.

### Quantizing (experimental)

The model was designed with quantization in mind — power-of-two channel
widths, kernel area 16, GGML-friendly real-valued arithmetic — but
calibrated Q4_K / Q8_0 weights are not yet published. The `quantize` tool
in the C++ build can produce GGUF variants from the F32 reference for
experimentation:

```bash
./ggml/build/bin/quantize localvqe-v1-f32.gguf localvqe-v1-q8.gguf Q8_0
```

Expect end-to-end quality loss until proper per-tensor selection and
calibration have been worked through.

## PyTorch Reference

`pytorch/` contains the model definition used to train and export the
weights. It's provided for verification, ablation, and downstream research
— not for end-user inference, which should go through the GGML build.

```bash
cd pytorch
pip install -r requirements.txt
python -c "
import yaml, torch
from localvqe.model import LocalVQE
cfg = yaml.safe_load(open('configs/default.yaml'))
model = LocalVQE(**cfg['model'], n_freqs=cfg['audio']['n_freqs'])
print(sum(p.numel() for p in model.parameters()))
"
```

## Citing LocalVQE

If you use LocalVQE in academic work, please cite the repository via the
`CITATION.cff` file at the root — GitHub renders a "Cite this repository"
button that produces APA and BibTeX entries automatically.

For a DOI, we recommend citing a specific release via
[Zenodo](https://zenodo.org), which mints a DOI per GitHub release. Please
also cite the upstream DeepVQE paper:

```bibtex
@inproceedings{indenbom2023deepvqe,
  title     = {DeepVQE: Real Time Deep Voice Quality Enhancement for Joint
               Acoustic Echo Cancellation, Noise Suppression and Dereverberation},
  author    = {Indenbom, Evgenii and Beltr{\'a}n, Nicolae-C{\u{a}}t{\u{a}}lin
               and Chernov, Mykola and Aichner, Robert},
  booktitle = {Interspeech},
  year      = {2023},
  doi       = {10.21437/Interspeech.2023-2176}
}
```

## Dataset Attribution

Published weights are trained on data from the
[ICASSP 2023 Deep Noise Suppression Challenge](https://github.com/microsoft/DNS-Challenge)
(Microsoft, CC BY 4.0) and fine-tuned on the
[ICASSP 2022/2023 Acoustic Echo Cancellation Challenge](https://github.com/microsoft/AEC-Challenge).

## Safety Note

Training data was filtered by DNSMOS perceived-quality scores, which can
misclassify distressed speech (screaming, crying) as noise. LocalVQE may
attenuate or distort such signals and must not be relied upon for emergency
call or safety-critical applications.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
