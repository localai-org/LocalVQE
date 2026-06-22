# LocalVQE

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/LocalAI-io/LocalVQE)

**Local Voice Quality Enhancement** — compact neural models for acoustic echo
cancellation (AEC), noise suppression (NS), and dereverberation of 16 kHz
speech, running on commodity CPUs in real time. Causal and streaming
(256-sample hop, 16 ms latency). F32 inference in C++ via
[GGML](https://github.com/ggml-org/ggml); a PyTorch reference is included for
research.

- **Try it:** <https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo>
- **Weights:** <https://huggingface.co/LocalAI-io/LocalVQE>

A streaming, CPU-tuned derivative of **DeepVQE**
([Indenbom et al., Interspeech 2023](https://arxiv.org/abs/2306.03177)).

## Models

Speed is per 16 ms hop on a Ryzen 9 7900 (Zen4), 4 threads; RT = realtime
factor (higher is faster than realtime).

| Version | Does | Params | Size (F32) | Speed | Pick it when |
|---|---|---:|---:|---|---|
| **v1.3** *(current)* | AEC + NS + dereverb | 4.8 M | ~19 MB | 3.2 ms · 5.0× RT | best joint quality, CPU budget available |
| **v1.2** | AEC + NS + dereverb | 1.3 M | ~5 MB | 1.7 ms · 8.9× RT | tight CPU / low-power devices |
| **v1.4-AEC** | echo only (keeps voice, noise, room) | 203 K | ~3 MB | 0.83 ms · 19× RT | NS is handled elsewhere, or you want the room kept |
| **v1.4-AEC 2.7K** | echo only, linear filter (no mask) | 2.7 K | ~17 KB | 0.36 ms · 44× RT | lightest echo canceller; echo isn't heavily reverberant |
| v1.1 / v1 | AEC + NS + dereverb | 1.3 M | ~5 MB | — | superseded by v1.2 |

- **Joint models (v1.2 / v1.3)** clean echo, noise, and reverb in one pass.
  v1.3 is wider and filters noise better; v1.2 is ~1/4 the per-hop cost.
- **v1.4-AEC** removes only the far-end echo and passes voice, room, and
  background through unchanged. It's a classical adaptive filter followed by a
  small neural mask. The **2.7K** build is that filter alone — cheaper and
  gentler, but it can't remove heavily reverberant echo the way the mask can.
- Every model needs a far-end **reference** signal (a loopback of what your
  speakers play) in addition to the mic.
- `bf16` GGUFs are ~12 % smaller with identical quality and speed; pick `f32`
  unless download size matters.
- A separate **Raspberry Pi line** (experimental) — a ~49 K-parameter GTCRN-AEC
  backend (a distinct architecture, not the v1.x graph) for single-board ARM;
  ≈21× realtime on one Pi 5 core. See [below](#raspberry-pi-line--gtcrn-aec-experimental).

### Weight files on [Hugging Face](https://huggingface.co/LocalAI-io/LocalVQE)

| File | Model |
|---|---|
| `localvqe-v1.3-4.8M-f32.gguf` / `.pt` | v1.3 joint (GGUF for inference, `.pt` for research) |
| `localvqe-v1.2-1.3M-f32.gguf` / `.pt` | v1.2 joint |
| `localvqe-v1.4-aec-200K-f32.gguf` / `-bf16.gguf` | v1.4-AEC (echo only) |
| `localvqe-v1.4-aec-2.7K-f32.gguf` | v1.4-AEC front-end only |
| `localvqe-v1.1-1.3M-f32.gguf`, `localvqe-v1-1.3M-f32.gguf` | older releases |
| `localvqe-pi-v1-49k-f32.gguf`, `localvqe-pi-aec-v1-49k-f32.gguf` | Raspberry Pi GTCRN-AEC line — experimental (full enhance / echo-only) |

v1.4-AEC is GGUF-only (no `.pt`). GGUF integrity is checked at load time against
a built-in SHA256 allowlist (`ggml/model_hash.cpp`). PyTorch checkpoint hashes:

    22d3e2f33bb8b25ec1c6a928cfb741bb631d45bae2b3759684818b101c95878e  localvqe-v1.3-4.8M.pt
    ff6885e7c8d7d29a8ce963303dcd668ae0f2a7bdafae28631292fe6f06f7cd77  localvqe-v1.2-1.3M.pt

## Performance

Full 800-clip eval on the
[ICASSP 2022 AEC Challenge blind test set](https://github.com/microsoft/AEC-Challenge)
(real recordings). AECMOS echo / deg are 1–5 (higher = more echo removed /
cleaner speech); blind ERLE is `10·log10(E[mic²]/E[enh²])`, only meaningful on
far-end-only clips. Unprocessed-mic echo MOS is 2.67 / 2.56 / 1.90 / 2.13 / 5.00
across the five scenarios.

**v1.4-AEC** — keeps background noise and room by design, so its ERLE and
far-end DNSMOS are intentionally lower than the joint models (it isn't deleting
the ambience):

| Scenario | n | echo ↑ | deg ↑ | ERLE ↑ | OVRL |
|---|--:|--:|--:|--:|--:|
| doubletalk | 115 | 4.20 | 2.45 | — | 2.59 |
| doubletalk-with-movement | 185 | 4.19 | 2.45 | — | 2.55 |
| farend-singletalk | 107 | 3.80 | 4.99 | 14.6 dB | 1.37 |
| farend-singletalk-with-movement | 193 | 3.86 | 4.95 | 11.1 dB | 1.31 |
| nearend-singletalk | 200 | 4.99 | 3.99 | — | 3.08 |

**v1.4-AEC 2.7K** (front-end only) — matches or beats the full model's
perceptual far-end echo at 1/74 the parameters; the mask's extra work shows up
as higher ERLE above, not higher echo MOS:

| Scenario | n | echo ↑ | deg ↑ | ERLE ↑ | OVRL |
|---|--:|--:|--:|--:|--:|
| doubletalk | 115 | 4.00 | 2.79 | — | 2.46 |
| doubletalk-with-movement | 185 | 3.90 | 2.92 | — | 2.42 |
| farend-singletalk | 107 | 4.06 | 5.00 | 6.5 dB | 1.24 |
| farend-singletalk-with-movement | 193 | 4.05 | 4.97 | 3.9 dB | 1.22 |
| nearend-singletalk | 200 | 4.98 | 3.77 | — | 3.03 |

**v1.3** (joint) and **v1.2** (joint) — these also delete the background, so
their far-end ERLE is much higher and not comparable to v1.4-AEC's:

| Scenario | n | v1.3 echo / deg / ERLE / OVRL | v1.2 echo / deg / ERLE / OVRL |
|---|--:|---|---|
| doubletalk | 115 | 4.73 / 2.62 / 8.5 dB / 2.89 | 4.72 / 2.37 / 8.4 dB / 2.83 |
| doubletalk-with-movement | 185 | 4.67 / 2.43 / 8.3 dB / 2.85 | 4.65 / 2.30 / 8.1 dB / 2.79 |
| farend-singletalk | 107 | 3.69 / 4.83 / 50.9 dB / 1.94 | 3.78 / 4.91 / 45.7 dB / 1.80 |
| farend-singletalk-with-movement | 193 | 3.88 / 4.98 / 49.9 dB / 1.96 | 4.12 / 4.96 / 40.6 dB / 1.75 |
| nearend-singletalk | 200 | 5.00 / 4.18 / 2.4 dB / 3.17 | 5.00 / 4.16 / 2.1 dB / 3.17 |

### Latency

Per-hop p50 / p99 and RT factor. 16 kHz, 256-sample hop, 16 ms budget.

**v1.4-AEC** (Ryzen 9 7900, CPU):

| Threads | p50 | p99 | RT |
|--:|--:|--:|--:|
| 1 | 1.29 ms | 1.89 ms | 12.2× |
| 4 | 0.83 ms | 1.30 ms | 18.6× |

The 2.7K front-end-only build runs at 0.36 ms p50 (≈44× RT), single-threaded by
nature. The adaptive front-end always runs on CPU; the neural stage is too small
for GPU offload to pay off, so run v1.4-AEC on CPU.

**v1.3** (joint):

| Hardware | Backend | Threads | p50 | p99 | RT |
|---|---|--:|--:|--:|--:|
| Ryzen 9 7900 | CPU | 1 | 9.73 ms | 14.48 ms | 1.58× |
| Ryzen 9 7900 | CPU | 4 | 3.21 ms | 3.42 ms | 4.97× |
| Ryzen 9 7900 + RTX 5070 Ti | Vulkan | — | 2.57 ms | 4.21 ms | 6.07× |

**v1.2** (joint):

| Hardware | Backend | Threads | p50 | p99 | RT |
|---|---|--:|--:|--:|--:|
| Ryzen 9 7900 | CPU | 1 | 4.28 ms | 4.85 ms | 3.72× |
| Ryzen 9 7900 | CPU | 4 | 1.65 ms | 2.91 ms | 8.90× |
| Ryzen 9 7900 + RTX 5070 Ti | Vulkan | — | 1.96 ms | 3.64 ms | 7.85× |
| Ryzen 7 6800U (laptop) | CPU | 4 | 2.11 ms | 2.77 ms | 7.44× |

These graphs are small, so threads hit diminishing returns past ~4. The library
defaults to `min(4, available CPUs)` (respects `taskset` / cgroup limits);
override with `localvqe_options_set_threads`. Run `bench-run` (below) to
reproduce on your hardware.

### Memory (CPU)

Working set the model adds on top of the ~7 MiB binary baseline:

| Model | Post-load delta | Peak RSS |
|---|--:|--:|
| v1.3 (4.8 M) | +24.4 MiB | 34.1 MiB |
| v1.2 (1.3 M) | +10.0 MiB | 19.6 MiB |
| v1.4-AEC (203 K) | +6.7 MiB | 17.0 MiB |

### Raspberry Pi line — GTCRN-AEC (experimental)

A separate, much smaller backend for single-board ARM: a ~49 K-parameter
**GTCRN-AEC** network — a distinct architecture based on
[GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (Rong et al., ICASSP 2024) — with
the project's DSP echo-cancellation front-end. Two variants share the
architecture: a full enhancer (echo + NS + dereverb) and an echo-only
"keep-noise" build. Cross-compile for aarch64 with `ggml/docker/Dockerfile.arm64`
(docker buildx + qemu, Cortex-A76 target).

A graphical demo of the speed ([`demo/`](demo/)): a scrolling spectrogram whose
processing frontier sweeps ahead of real-time playback, cleaning a 3-minute
track in ~9 s on one Pi 5 core. The two bars track playback vs processing over
the whole track.

![Pi AEC sweep: a scrolling spectrogram whose processing frontier reveals the cleaned output ~21x faster than real-time playback on one Raspberry Pi 5 core](demo/out/sweep.gif)

Whole-clip RTF on the real ggml graph (`test_gtcrn --bench` on a Raspberry Pi 5
Model B, Cortex-A76, Ubuntu 24.04), parity-verified to the PyTorch reference
within ~1e-6 on-device (~0.78 ms per 16 ms hop single-threaded). RTF is identical
for both variants:

| Threads | 8 s clip | RTF | RT |
|--:|--:|--:|--:|
| 1 | 388 ms | 0.048 | ~21× |
| 2 | 219 ms | 0.027 | ~37× |
| 4 | 163 ms | 0.020 | ~49× |

## Usage

### Build

Requires CMake ≥ 3.20 and a C++17 compiler. A [Nix](https://nixos.org/) flake is
provided (`nix develop`); without Nix, install cmake, gcc/clang, pkg-config, and
libsndfile.

```bash
git clone --recursive https://github.com/localai-org/LocalVQE.git
cd LocalVQE
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)
```

Binaries land in `ggml/build/bin/`. The CPU build produces several
`libggml-cpu-*.so` variants (SSE4.2 → AVX-512) selected at runtime — keep them
next to the binary. For GPU, add `-DLOCALVQE_VULKAN=ON` (the loader falls back
to CPU when no Vulkan ICD is present).

### Run (CLI)

```bash
./ggml/build/bin/localvqe localvqe-v1.3-4.8M-f32.gguf \
    --in-wav mic.wav ref.wav \
    --out-wav enhanced.wav
```

16 kHz mono PCM for both mic and far-end reference. Swap the GGUF to switch
models — same command for every version (the engine reads what to do from the
file).

### Embed (C API)

```bash
cmake -S ggml -B ggml/build -DLOCALVQE_BUILD_SHARED=ON
cmake --build ggml/build -j$(nproc)   # -> liblocalvqe.so
```

API in `ggml/localvqe_api.h`:

```c
localvqe_ctx_t ctx = localvqe_new("localvqe-v1.3-4.8M-f32.gguf");
localvqe_process_f32(ctx, mic, ref, n_samples, out);   // whole clip
// or per 256-sample hop for real-time: localvqe_process_frame_f32(...)
localvqe_free(ctx);
```

See `ggml/example_purego_test.go` for a Go / `purego` binding.

### Benchmark / test

```bash
cmake --build ggml/build --target bench-run          # downloads a model + clip, benches
cmake --build ggml/build --target test_regression regression-assets
ctest --test-dir ggml/build --output-on-failure      # SKIPs models not downloaded
```

`bench-run` honors `-DBENCH_BACKEND=Vulkan -DBENCH_DEVICE=N -DBENCH_ITERS=N` set
at configure time; `bench-list-devices` enumerates backends.

### OBS Studio plugin

`obs-plugin/` wraps `liblocalvqe.so` as an audio filter — appears as **"LocalVQE
(AEC + Noise + Dereverb)"** in any source's filter list, with the bundled v1.3
GGUF preselected. NS and dereverb work out of the box; for AEC, set a **Reference
source** (usually "Desktop Audio") so the model knows what's playing. Browse to
`localvqe-v1.4-aec-200K-f32.gguf` to switch to echo-only mode.

```bash
nix develop .#obs-plugin
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release -DLOCALVQE_BUILD_SHARED=ON
cmake --build ggml/build -j$(nproc)
cmake --build ggml/build --target regression-assets
cp ggml/build/bench_assets/localvqe-v1.3-4.8M-f32.gguf obs-plugin/data/
cmake -S obs-plugin -B obs-plugin/build -DCMAKE_BUILD_TYPE=Release
cmake --build obs-plugin/build -j$(nproc) && cmake --install obs-plugin/build
```

The install is self-contained (plugin `.so` + `liblocalvqe.so` + the
`libggml-cpu-*.so` variants under `~/.config/obs-studio/plugins/`). Tested on
Linux; macOS expected to work; Windows implemented but unverified.

### PyTorch reference

`pytorch/` holds the model definition used to train and export the weights — for
verification and research, not end-user inference (use the GGML build).

```bash
cd pytorch && pip install -r requirements.txt
python -c "import yaml, torch; from localvqe.model import LocalVQE; \
cfg = yaml.safe_load(open('configs/default.yaml')); \
m = LocalVQE(**cfg['model'], n_freqs=cfg['audio']['n_freqs']); \
print(sum(p.numel() for p in m.parameters()))"
```

## Repository layout

```
ggml/        C++ streaming inference (GGML graph, CLI, C API, tests)
pytorch/     PyTorch reference (model definition only)
obs-plugin/  OBS Studio audio filter wrapping liblocalvqe.so
```

## Citing

Cite the repository via `CITATION.cff` (GitHub's "Cite this repository" button
produces APA / BibTeX), and the upstream DeepVQE paper:

```bibtex
@inproceedings{indenbom2023deepvqe,
  title     = {DeepVQE: Real Time Deep Voice Quality Enhancement for Joint
               Acoustic Echo Cancellation, Noise Suppression and Dereverberation},
  author    = {Indenbom, Evgenii and Beltr{\'a}n, Nicolae-C{\u{a}}t{\u{a}}lin
               and Chernov, Mykola and Aichner, Robert},
  booktitle = {Interspeech}, year = {2023},
  doi       = {10.21437/Interspeech.2023-2176}
}
```

The experimental Raspberry Pi backend is based on **GTCRN** — please also cite:

```bibtex
@inproceedings{rong2024gtcrn,
  title     = {GTCRN: A Speech Enhancement Model Requiring Ultralow
               Computational Resources},
  author    = {Rong, Xiaobin and Sun, Tianchi and Zhang, Xu and Hu, Yuxiang
               and Zhu, Changbao and Lu, Jing},
  booktitle = {ICASSP 2024 - 2024 IEEE International Conference on Acoustics,
               Speech and Signal Processing (ICASSP)},
  pages     = {971--975}, year = {2024},
  doi       = {10.1109/ICASSP48485.2024.10448310}
}
```

Reference implementation: <https://github.com/Xiaobin-Rong/gtcrn>.

## Attribution, safety, license

Weights are trained on the
[ICASSP 2023 DNS Challenge](https://github.com/microsoft/DNS-Challenge)
(Microsoft, CC BY 4.0) and fine-tuned on the
[ICASSP 2022/2023 AEC Challenge](https://github.com/microsoft/AEC-Challenge).

**Safety:** training data was filtered by DNSMOS, which can misclassify
distressed speech (screaming, crying) as noise. LocalVQE may attenuate such
signals and must not be relied on for emergency or safety-critical use.

Licensed under Apache 2.0 — see [LICENSE](LICENSE).
