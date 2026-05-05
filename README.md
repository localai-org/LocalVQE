# LocalVQE

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/LocalAI-io/LocalVQE)

**Local Voice Quality Enhancement** — a compact neural model for joint
acoustic echo cancellation (AEC), noise suppression, and dereverberation of
16 kHz speech, designed to run on commodity CPUs in real time.

- 1.3 M parameters (~5 MB F32)
- ~1.66 ms per 16 ms frame on Zen4 (24 threads) — **≈9.6× realtime**
- Causal, streaming: 256-sample hop, 16 ms algorithmic latency
- F32 reference inference in C++ via [GGML](https://github.com/ggml-org/ggml);
  PyTorch reference included for verification and research

Try it: <https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo>.

LocalVQE is a derivative of **DeepVQE**
([Indenbom et al., Interspeech 2023](https://arxiv.org/abs/2306.03177)) —
smaller, GGML-native, and tuned for streaming CPU inference. The
architecture, training recipe, and reproducibility checks are
documented in the [technical report](paper/localvqe.tex) under
`paper/`; this README covers building and running the published
weights only.

## A concrete example

Picture a video call from a laptop. Your microphone picks up three things
alongside your voice:

1. The remote participant's voice, played back through your speakers and
   caught again by your mic — this is the **echo**. Without cancellation
   they hear themselves a fraction of a second later.
2. Your own voice bouncing off walls, desk, and monitor before reaching
   the mic — this is **reverberation**, the "tunnel" or "bathroom" sound
   that makes you feel far away from the listener.
3. A fan, keyboard clatter, a dog barking, or traffic outside — plain
   **background noise**.

LocalVQE removes all three in a single causal pass, frame by frame, on
the CPU, so only your voice reaches the far end.

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

Microsoft never released DeepVQE — no weights, no reference
implementation, no streaming runtime. We re-implemented it from the
paper as a GGML graph at
[richiejp/deepvqe-ggml](https://github.com/richiejp/deepvqe-ggml)
(the full-width ~7.5 M-parameter version) before starting LocalVQE.
LocalVQE is the same idea pruned and rebuilt to ~1.3 M parameters
(~5 MB F32), small enough to run on commodity CPUs in real time.

## Model Weights

Pre-trained weights are published on Hugging Face at
[LocalAI-io/LocalVQE](https://huggingface.co/LocalAI-io/LocalVQE):

| File | Description |
|---|---|
| `localvqe-v1.1-1.3M-f32.gguf` | F32 GGUF — what the C++ engine loads. |
| `localvqe-v1.1-1.3M.pt` | PyTorch checkpoint — for verification, ablation, and downstream research. |
| `localvqe-v1-1.3M-f32.gguf` | Previous release. |

The current release is **v1.1**, which fixes intermittent crackling
the previous release produced under heavy background noise.

Only F32 is published today. A `quantize` tool is included in the C++
build (see below); calibrated Q4_K / Q8_0 weights are not yet
released.

## Streaming latency

Per-hop, 16 kHz / 256-sample hop → 16 ms budget. Each hop is a full
`ggml_backend_graph_compute`. Run any of these locally with the
`bench-run` cmake target — see [Benchmark](#benchmark) below. 30
iters × 625 hops/iter = 18 750 hops per row.

| Hardware                              | Backend | Threads | Hop p50  | Hop p99  | Hop max    | RT factor |
|---------------------------------------|---------|--------:|---------:|---------:|-----------:|----------:|
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       1 |  3.40 ms |  3.57 ms |  5.06 ms   |     4.7×  |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       2 |  2.07 ms |  2.25 ms |  3.65 ms   |     7.7×  |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       4 |  1.32 ms |  1.57 ms |  6.91 ms ‡ |    12.0×  |
| Ryzen 9 7900 + RADV iGPU (Raphael)    | Vulkan  |       — |  4.43 ms |  4.62 ms |  5.07 ms   |     3.60× |
| Ryzen 9 7900 + RTX 5070 Ti (dGPU)     | Vulkan  |       — |  1.79 ms |  3.41 ms |  4.14 ms   |     8.63× |
| Apple M4 (4P + 6E, macOS 25.3)        | CPU     |       1 |  2.98 ms |  3.16 ms | 19.11 ms ‡ |     5.4×  |
| Apple M4 (4P + 6E, macOS 25.3)        | CPU     |       2 |  1.82 ms |  1.93 ms |  3.17 ms   |     8.8×  |
| Apple M4 (4P + 6E, macOS 25.3)        | CPU     |       4 |  1.11 ms |  1.81 ms | 10.41 ms ‡ |    14.4×  |
| Core i5-14500 (Alder Lake-S)          | CPU     |       1 |  3.25 ms |  3.53 ms |  6.73 ms   |     4.93× |
| Core i5-14500 (Alder Lake-S)          | CPU     |       2 |  2.55 ms |  2.81 ms |  5.20 ms   |     6.23× |
| Core i5-14500 (Alder Lake-S)          | CPU     |       3 |  2.26 ms |  3.09 ms |  3.85 ms   |     7.06× |
| Core i5-14500 (Alder Lake-S)          | CPU     |       4 |  2.02 ms |  2.89 ms |  3.59 ms   |     7.79× |
| Core i5-14500 + Arc A770 (dGPU)       | Vulkan  |       — | 10.90 ms | 12.00 ms | 13.38 ms   |     1.48× |
| Core i5-14500 + UHD 770 (iGPU)        | Vulkan  |       — |  9.02 ms | 11.77 ms | 17.93 ms   |     1.74× |

Adding cores hits diminishing returns quickly: the model is small
enough that thread-launch and synchronisation overhead start to
dominate beyond ≈4 threads on these CPUs. The Alder Lake sweep
shows it plainly — the 1→2 thread step gives a 1.27× speedup, but
3→4 only adds 1.10×. For deployment, two to four threads is
usually the sweet spot.

Zen4 rows are measured on the v1.1 model. Apple M4 and Alder Lake
rows were measured on the previous release; CPU latency is unchanged
to within measurement noise so they remain representative.

‡ Outliers are single hops early in the first iteration (cold
caches); p99 is representative of steady-state.

Vulkan p50/p95/p99 are typically tight, but worst-case single-hop
latency on a shared desktop is sensitive to external GPU clients
(display compositor, browser). On a dedicated embedded device with
no compositor contending for the queue, expect the quieter end of
the range.

The bench binary prints the top-10 slowest hops with
`(iteration, hop-in-iteration)` coordinates so you can check whether
outliers cluster at post-`localvqe_reset()` boundaries (cold path)
or scatter through the stream (external contention). In practice we
see the latter.

## Validation Results

Full 800-clip eval on the
[ICASSP 2022 AEC Challenge blind test set](https://github.com/microsoft/AEC-Challenge)
(real recordings, not synthetic mixes):

| Scenario                          |   n | AECMOS echo ↑ | AECMOS deg ↑ | blind ERLE ↑ | DNSMOS OVRL ↑ |
|-----------------------------------|----:|--------------:|-------------:|-------------:|--------------:|
| doubletalk                        | 115 |          4.70 |         2.35 |       8.4 dB |          2.85 |
| doubletalk-with-movement          | 185 |          4.63 |         2.35 |       8.3 dB |          2.80 |
| farend-singletalk                 | 107 |          2.98 |         4.91 |      44.7 dB |          1.93 |
| farend-singletalk-with-movement   | 193 |          3.40 |         4.95 |      45.0 dB |          1.91 |
| nearend-singletalk                | 200 |          4.99 |         4.05 |       2.5 dB |          3.13 |

- **AECMOS** (Purin et al., ICASSP 2022) is Microsoft's non-intrusive AEC
  quality predictor. "Echo" rates how well echo was removed; "degradation"
  rates how clean the resulting speech is. 1–5 MOS scale, higher is better.
- **Blind ERLE** is `10·log10(E[mic²] / E[enh²])`. Only meaningful on
  far-end single-talk where the input is echo-only; on scenes with active
  near-end speech it understates echo removal because both numerator and
  denominator are dominated by speech.

PyTorch checkpoint integrity (SHA256):

    76aabaa3bca3a9d7989463226312aa2344f978403c3e0e007e58a15922c97707  localvqe-v1.1-1.3M.pt

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
git clone --recursive https://github.com/localai-org/LocalVQE.git
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
./ggml/build/bin/localvqe localvqe-v1.1-1.3M-f32.gguf \
    --in-wav mic.wav ref.wav \
    --out-wav enhanced.wav
```

Expects 16 kHz mono PCM for both mic and far-end reference.

### Benchmark

The `bench-run` cmake target is the turnkey path: it builds `bench`,
downloads the released F32 model and a doubletalk mic/ref WAV pair from
HuggingFace into `ggml/build/bench_assets/`, and runs the benchmark.

```bash
# Configure once (Vulkan optional but recommended for GPU runs)
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release -DLOCALVQE_VULKAN=ON

# Discover backends + device indices
cmake --build ggml/build --target bench-list-devices

# Run on the default backend (CPU device 0, 10 iterations)
cmake --build ggml/build --target bench-run
```

To pick a specific backend or device, set the cache variables at
configure time and rebuild the target:

```bash
# Vulkan device 0 (e.g. dGPU) with 30 iterations
cmake -S ggml -B ggml/build -DBENCH_BACKEND=Vulkan -DBENCH_DEVICE=0 -DBENCH_ITERS=30
cmake --build ggml/build --target bench-run

# Vulkan device 1 (e.g. iGPU)
cmake -S ggml -B ggml/build -DBENCH_DEVICE=1
cmake --build ggml/build --target bench-run
```

Sweeping every backend/device on the box is just a shell loop over the
indices `bench-list-devices` printed:

```bash
for dev in 0 1; do
    cmake -S ggml -B ggml/build -DBENCH_BACKEND=Vulkan -DBENCH_DEVICE=$dev
    cmake --build ggml/build --target bench-run
done
```

Or invoke the binary directly against your own WAV pair:

```bash
./ggml/build/bin/bench localvqe-v1.1-1.3M-f32.gguf \
    --backend Vulkan --device 0 \
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

Calibrated Q4_K / Q8_0 weights are not yet published. The `quantize`
tool in the C++ build can produce GGUF variants from the F32 reference
for experimentation:

```bash
./ggml/build/bin/quantize localvqe-v1.1-1.3M-f32.gguf localvqe-v1.1-1.3M-q8.gguf Q8_0
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
