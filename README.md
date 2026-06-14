# LocalVQE

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/LocalAI-io/LocalVQE)

**Local Voice Quality Enhancement** — a compact neural model for joint
acoustic echo cancellation (AEC), noise suppression, and dereverberation of
16 kHz speech, designed to run on commodity CPUs in real time.

- Two sizes of the joint model — choose by CPU budget:
  - **v1.3 (current)** — 4.8 M parameters (~19 MB F32), ~3.3 ms per 16 ms
    frame on Zen4 (4 threads), **≈4.7× realtime**.
  - **v1.2** — 1.3 M parameters (~5 MB F32), ~1.6 ms per 16 ms frame on
    Zen4 (4 threads), **≈9.7× realtime**.
- **v1.4-AEC (new)** — echo cancellation *only*, 203 K parameters (~3 MB):
  removes the speaker echo and leaves everything else — your voice, the
  room, background sounds — untouched. See
  [v1.4-AEC: echo cancellation only](#v14-aec-echo-cancellation-only).
  A **2.7 K front-end-only** build (`-2.7K-f32.gguf`, ~0.36 ms/hop,
  ≈44× realtime) ships the adaptive filter without the neural mask, for
  the lightest possible linear echo canceller.
- Causal, streaming: 256-sample hop, 16 ms algorithmic latency
- F32 reference inference in C++ via [GGML](https://github.com/ggml-org/ggml);
  PyTorch reference included for verification and research

Try it: <https://huggingface.co/spaces/LocalAI-io/LocalVQE-demo>.

LocalVQE is a derivative of **DeepVQE**
([Indenbom et al., Interspeech 2023](https://arxiv.org/abs/2306.03177)) —
smaller, GGML-native, and tuned for streaming CPU inference.

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
LocalVQE is the same idea rebuilt for streaming CPU inference, and
published in two sizes: a 1.3 M-parameter compact build (v1.2, ~5 MB
F32) for tight CPU budgets, and a 4.8 M-parameter wider build (v1.3,
~19 MB F32) that filters noise better on some clips at ~2× the
per-hop cost. Both are small enough to run real time on commodity
CPUs.

## v1.4-AEC: echo cancellation only

v1.2 and v1.3 do three jobs at once — echo cancellation, noise
suppression, and dereverberation. **LocalVQE-v1.4-AEC does exactly one:
it removes the far-end echo and passes everything else through.** Your
voice, the room's natural acoustics, and the background stay as they
are. Choose it when:

- another component owns noise suppression (an ASR front-end, a
  conferencing stack with its own NS, a broadcast chain), and a second
  suppressor would double-process the signal;
- you *want* the ambience — full enhancement can sound "too processed"
  for podcasts, field recordings, or rooms that are already quiet;
- you need the smallest possible model: 203 K parameters, ~23× less
  than v1.3.

It is a two-stage cascade rather than a single network. A classical
adaptive stage — cross-correlation pre-alignment plus a
partitioned-block frequency-domain Kalman filter, steered per-frame by
a small learned controller — cancels the linear part of the echo with
unit gain on everything that isn't echo. A compact neural mask then
removes the residual the linear filter can't reach (loudspeaker
distortion, enclosure nonlinearity). Echo paths are searched over the
same ~1 s window as v1.2/v1.3, and streaming latency is identical
(16 ms hop).

Two files are published: `f32` (the reference) and `bf16`, which
stores the convolution weights in bfloat16 for a ~12 % smaller file.
Output quality is identical within validation noise and inference
speed is the same — pick `f32` unless download size matters.

The AEC-only release is the first half of the v1.4 line; a joint
variant that adds noise suppression on top of the same cascade is
planned as the full v1.4.

## Model Weights

Pre-trained weights are published on Hugging Face at
[LocalAI-io/LocalVQE](https://huggingface.co/LocalAI-io/LocalVQE):

| File | Description |
|---|---|
| `localvqe-v1.4-aec-200K-f32.gguf` | **New:** echo cancellation only — keeps voice, room, and background intact. |
| `localvqe-v1.4-aec-200K-bf16.gguf` | Same model, conv weights in BF16 (~12 % smaller file, same quality). |
| `localvqe-v1.4-aec-2.7K-f32.gguf` | **New:** the adaptive filter alone (2.7 K params, ~17 KB) — the cascade's front-end without the neural mask. Lighter and faster, less echo removal; same CLI/engine. |
| `localvqe-v1.3-4.8M-f32.gguf` | F32 GGUF — joint AEC + NS + dereverb (current default). |
| `localvqe-v1.3-4.8M.pt` | PyTorch checkpoint — for verification, ablation, and downstream research. |
| `localvqe-v1.2-1.3M-f32.gguf` | Compact joint alternative — same architecture family, ~1/4 the cost per hop. |
| `localvqe-v1.2-1.3M.pt` | PyTorch checkpoint for the compact variant. |
| `localvqe-v1.1-1.3M-f32.gguf` | Older release. |
| `localvqe-v1-1.3M-f32.gguf` | Original release. |

**v1.3** is the current joint-model release (the echo-only
**v1.4-AEC** is described in its own section above). It widens the
encoder/decoder (mic channels `[2,112,32,104,96,152]`, far `[2,64,32]`,
bottleneck 256) and trains from scratch under a noise-floor-aware loss
recipe.
On doubletalk it filters noise better than v1.2 (deg MOS +0.25 on
the stratified dev sample, with stronger ERLE). On far-end-only
echo it cancels harder but the residual rates rougher in AECMOS —
some users will prefer v1.2's gentler trade-off on FE-ST scenes.
v1.2 stays on the repo as the small/fast option (~1/4 the per-hop
cost). Both reuse v1.2's 1024 ms echo-search window.

## Streaming latency

Per-hop, 16 kHz / 256-sample hop → 16 ms budget. Each hop is a full
`ggml_backend_graph_compute`. Run any of these locally with the
`bench-run` cmake target — see [Benchmark](#benchmark) below. 30
iters × 625 hops/iter = 18 750 hops per row.

### v1.4-AEC (echo-only — 203 K two-stage cascade)

| Hardware                              | Backend | Threads | Hop p50  | Hop p99  | RT factor |
|---------------------------------------|---------|--------:|---------:|---------:|----------:|
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       1 |  1.29 ms |  1.89 ms |    12.2× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       2 |  1.04 ms |  1.55 ms |    15.1× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       4 |  0.83 ms |  1.30 ms |    18.6× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       8 |  1.13 ms |  1.45 ms |    14.1× |

The adaptive front-end runs on the host CPU regardless of the selected
GGML backend, and the neural stage is small enough that GPU offload
isn't worth the transfer overhead — run this model on CPU. The BF16
file benches identically at the 4-thread sweet spot (0.84 ms p50) and
~7 % faster single-threaded.

The **2.7 K front-end-only** build (`-2.7K-f32.gguf`) is just the
adaptive filter — no neural mask — at **0.36 ms p50 / hop (≈44×
realtime)** on the same Zen4 desktop. It is single-threaded by nature
(sequential block DSP), so thread count doesn't change the figure.

### v1.3 (current joint model — 4.8 M, wider encoder/decoder, bn 256)

| Hardware                              | Backend | Threads | Hop p50  | Hop p99  | RT factor |
|---------------------------------------|---------|--------:|---------:|---------:|----------:|
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       1 |  9.73 ms | 14.48 ms |     1.58× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       2 |  5.41 ms |  5.62 ms |     2.95× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       4 |  3.21 ms |  3.42 ms |     4.97× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       8 |  3.47 ms |  3.80 ms |     4.59× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |      16 |  3.79 ms |  4.06 ms |     4.19× |
| Ryzen 9 7900 + RADV iGPU (Raphael)    | Vulkan  |       — |  8.71 ms |  9.15 ms |     1.83× |
| Ryzen 9 7900 + RTX 5070 Ti (dGPU)     | Vulkan  |       — |  2.57 ms |  4.21 ms |     6.07× |

The wider model is ~2× the per-hop cost of v1.2 in matching
configurations — the dGPU (RTX 5070 Ti) ends up the fastest option
for v1.3 by ~1.25× vs 4-thread CPU. The 1-thread case is the
worst, still real-time (RT 1.58×) but with little margin; running
v1.3 on a low-core / power-constrained device should use v1.2
instead. Re-runs on other CPUs (Apple M4, Alder Lake, mobile Zen3+)
will be published as we collect them — until then the v1.2 sweep
below is representative shape-wise and expects roughly the same ~2×
multiplier.

### v1.2 (compact alternative — 1.3 M, 1024 ms echo-search window)

| Hardware                              | Backend | Threads | Hop p50  | Hop p99  | Hop max    | RT factor |
|---------------------------------------|---------|--------:|---------:|---------:|-----------:|----------:|
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       1 |  4.28 ms |  4.85 ms |  6.23 ms   |     3.72× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       2 |  2.59 ms |  3.80 ms |  3.81 ms   |     6.09× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       4 |  1.65 ms |  2.91 ms |  4.57 ms   |     8.90× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |       8 |  1.93 ms |  2.41 ms |  6.91 ms ‡ |     8.22× |
| Ryzen 9 7900 (Zen4 desktop)           | CPU     |      16 |  2.09 ms |  2.22 ms |  6.43 ms ‡ |     7.69× |
| Ryzen 9 7900 + RADV iGPU (Raphael)    | Vulkan  |       — |  6.10 ms |  6.53 ms |  6.24 ms   |     2.61× |
| Ryzen 9 7900 + RTX 5070 Ti (dGPU)     | Vulkan  |       — |  1.96 ms |  3.64 ms |  5.42 ms   |     7.85× |
| Ryzen 7 6800U (Zen3+ laptop)          | CPU     |       1 |  4.69 ms |  6.08 ms | 19.31 ms ‡ |     3.37× |
| Ryzen 7 6800U (Zen3+ laptop)          | CPU     |       4 |  2.11 ms |  2.77 ms |  4.90 ms   |     7.44× |
| Ryzen 7 6800U (Zen3+ laptop)          | CPU     |       8 |  1.94 ms |  2.60 ms |  5.52 ms   |     7.94× |
| Ryzen 7 6800U + RADV iGPU (Rembrandt) | Vulkan  |       — |  9.84 ms | 14.75 ms | 20.87 ms ‡ |     1.53× |

The wider echo-search window v1.2 introduced (1024 ms vs v1.1's 512 ms)
costs ~20–25 % per-hop on CPU vs v1.1.

### v1.1 (previous — 512 ms echo-search window)

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

Adding cores hits diminishing returns quickly: even the wider v1.3
graph is small enough that thread-launch and synchronisation
overhead start to dominate beyond ≈4 threads on these CPUs. The
Zen4 sweeps show it plainly on both versions — the 1→4 thread step
gives a 2.59× speedup on v1.2 and a 3.03× speedup on v1.3, but
4→8 is a regression on both and 8→16 worse still. The 6800U mobile
Zen3+ on v1.2 agrees: 1→4 is a 2.21× speedup, 4→8 only buys another
7%. **The library's default thread count is `min(4, sched_getaffinity)`** —
auto-capped at 4 with respect for `taskset`, cgroup, and VM CPU
limits, so over-subscription doesn't happen on resource-constrained
hosts. Pass a non-zero value to `localvqe_options_set_threads` to
override.

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

### Memory footprint (CPU)

`bench` reports process RSS from `/proc/self/status` alongside the
internal allocator accounting from `--profile`. The numbers are
essentially thread-count-invariant — both 1 and 16 threads land on
the same peak within a few hundred KiB — so one row per model
suffices.

| Model                | Post-load delta ¹ | Peak RSS (VmHWM) ² | Internal `total resident` ³ |
|----------------------|------------------:|-------------------:|----------------------------:|
| **v1.3** (4.8 M)     | +24.4 MiB         |  34.1 MiB          |  23.0 MiB                   |
| **v1.2** (1.3 M)     | +10.0 MiB         |  19.6 MiB          |   8.7 MiB                   |
| **v1.4-AEC** (203 K) |  +6.7 MiB         |  17.0 MiB          |   3.1 MiB                   |

¹ RSS added by `localvqe_new_with_options` + CPU backend init, on
top of the ~7 MiB binary/libs baseline measured by `bench` itself.
This is the portable "working set the model brings" number; the
absolute peak will depend on your host process baseline.

² `VmHWM` after warmup + sustained streaming on a Zen4 desktop
(Ryzen 9 7900). v1.3 is ~1.75× v1.2 in RSS terms despite carrying
~3.7× more parameters — activation, history-scratch, and per-frame
history buffers don't scale with channel width the way the weight
buffer does.

³ Backend-internal accounting from `bench --profile`: the sum of
the weights buffer, activation buffer (gallocr), and one-shot
history scratch. Excludes the double-buffered history-tensor swap
pages (already counted in the activation buffer for the read side).

For GPU backends (Vulkan), RSS understates real usage — VRAM isn't
visible in `/proc/self/status`. Use `bench --profile` on the GPU
build to read the same weights/activation/scratch breakdown from
the backend-internal allocator.

## Validation Results

Full 800-clip eval on the
[ICASSP 2022 AEC Challenge blind test set](https://github.com/microsoft/AEC-Challenge)
(real recordings, not synthetic mixes):

**v1.4-AEC** (echo cancellation only — background noise and room
acoustics are kept *by design*):

| Scenario                          |   n | AECMOS echo ↑ | AECMOS deg ↑ | blind ERLE ↑ | DNSMOS OVRL |
|-----------------------------------|----:|--------------:|-------------:|-------------:|------------:|
| doubletalk                        | 115 |          4.20 |         2.45 |            — |        2.59 |
| doubletalk-with-movement          | 185 |          4.19 |         2.45 |            — |        2.55 |
| farend-singletalk                 | 107 |          3.80 |         4.99 |      14.6 dB |        1.37 |
| farend-singletalk-with-movement   | 193 |          3.86 |         4.95 |      11.1 dB |        1.31 |
| nearend-singletalk                | 200 |          4.99 |         3.99 |            — |        3.08 |

The unprocessed microphone scores echo MOS 2.67 / 2.56 / 1.90 / 2.13 /
5.00 on the same five scenarios — v1.4-AEC buys **+1.5 to +1.9 echo
MOS** over doing nothing, and its far-end echo scores match v1.3
(3.80/3.86 vs 3.69/3.88) at 1/23 the parameters. Reading the other
columns needs the keep-noise design in mind:

- **Near-end preservation is the headline**: on nearend-singletalk the
  output is nearly the input (deg 3.98 vs 4.07 unprocessed, DNSMOS
  OVRL 3.08 vs 3.13) — where the joint models would denoise, this
  model deliberately does nothing.
- **Blind ERLE is bounded by the preserved background**: the output
  keeps the room's noise, so energy can only drop to the ambient
  floor — 14.6/11.1 dB here measures echo-above-ambient removed,
  not total attenuation. v1.3's ~50 dB on the same clips reflects
  that it deletes the background too; the numbers are not comparable.
- **DNSMOS OVRL on far-end clips drops below even the unprocessed
  mic** (1.37 vs 1.90): once the echo — the only speech-like content —
  is removed, what remains is intentionally-kept ambient noise, which
  DNSMOS rates as noise. Expected, not a defect.
- **Doubletalk echo trails the joint models** (4.20 vs v1.3's 4.73):
  suppressing the noise floor also removes echo cues buried in it —
  headroom the keep-noise constraint gives up. Doubletalk deg (2.45)
  sits between v1.2 (2.37) and v1.3 (2.62).

The BF16 file scores identically — the largest difference across all
five scenarios and four metrics is 0.1 dB ERLE and 0.01 MOS.

**v1.4-AEC front-end-only** (2.7 K — the adaptive filter alone, no
neural mask):

| Scenario                          |   n | AECMOS echo ↑ | AECMOS deg ↑ | blind ERLE ↑ | DNSMOS OVRL |
|-----------------------------------|----:|--------------:|-------------:|-------------:|------------:|
| doubletalk                        | 115 |          4.00 |         2.79 |            — |        2.46 |
| doubletalk-with-movement          | 185 |          3.90 |         2.92 |            — |        2.42 |
| farend-singletalk                 | 107 |          4.06 |         5.00 |       6.5 dB |        1.24 |
| farend-singletalk-with-movement   | 193 |          4.05 |         4.97 |       3.9 dB |        1.22 |
| nearend-singletalk                | 200 |          4.98 |         3.77 |            — |        3.03 |

Scored on the same 800-clip blind set and pipeline as the cascade above,
so the columns line up directly. The standalone build estimates the bulk
echo delay once over the whole clip and locks it, so the filter is
aligned from the first frame (no online-acquisition transient — see
[delay priming](#delay-priming-standalone-build) below). What the 2.7 K
filter buys, and what the neural mask adds on top:

- **Far-end echo MOS beats the full cascade** (4.06 / 4.05 vs the
  cascade's 3.80 / 3.86) at 1/74 the parameters, and **+1.9 to +2.2 over
  the unprocessed mic** (1.90 / 2.13). The aligned linear filter delivers
  essentially all of the cascade's *perceptual* far-end echo gain — and
  edges past it, because the standalone primes the delay while the
  cascade relies on its mask to cover online acquisition.
- **The mask's contribution is energy attenuation, not perceived
  quality**: far-end ERLE is 6.5 / 3.9 dB here vs 14.6 / 11.1 dB for the
  cascade. The filter removes the echo a listener notices; the mask
  removes more of what's left over.
- **Gentler on doubletalk near-end**: deg MOS stays *higher* than the
  cascade (2.79 / 2.92 vs 2.45 / 2.45) — the filter trades some
  doubletalk echo MOS (4.00 / 3.90 vs 4.20 / 4.19) for less near-end
  distortion, even with the delay aligned.
- **Near-end passthrough is unchanged** (4.98 / 3.77 / OVRL 3.03 vs the
  cascade's 4.99 / 3.99 / 3.08) — with no echo present, both builds
  correctly do almost nothing.

The headline: the front-end-only build is for callers who want a
near-zero-cost echo canceller that is perceptually competitive on
far-end echo and maximally transparent on near-end speech, and who don't
need the extra echo-energy attenuation the 200 K mask provides. Same CLI
and engine; it just reads the `localvqe.daf.standalone` flag and emits
the filter output directly.

<a name="delay-priming-standalone-build"></a>
**Delay priming (standalone build).** A classical adaptive filter must
first find the bulk echo delay before it can cancel anything, and a
robust GCC-PHAT estimate needs ~1 s of audio to lock — so a purely
streaming filter leaks echo (heard as reverb) for the first 1–3 s. The
cascade hides this behind its feed-forward mask; the bare filter cannot.
So when the whole clip is available (file/batch use, including the demo),
the standalone engine estimates the delay once over the entire signal and
aligns the filter from frame 0. On the blind far-end set this cuts the
time to reach 3 dB ERLE from ~3.3 s to ~0.9 s and lifts steady-state
cancellation, with no change to the cascade (which still acquires online).
Real-time callers that feed audio incrementally fall back to online
acquisition automatically.

**v1.3** (current joint model, 4.8 M):

| Scenario                          |   n | AECMOS echo ↑ | AECMOS deg ↑ | blind ERLE ↑ | DNSMOS OVRL ↑ |
|-----------------------------------|----:|--------------:|-------------:|-------------:|--------------:|
| doubletalk                        | 115 |          4.73 |     **2.62** |       8.5 dB |          2.89 |
| doubletalk-with-movement          | 185 |          4.67 |     **2.43** |       8.3 dB |          2.85 |
| farend-singletalk                 | 107 |          3.69 |         4.83 |  **50.9 dB** |          1.94 |
| farend-singletalk-with-movement   | 193 |          3.88 |         4.98 |  **49.9 dB** |          1.96 |
| nearend-singletalk                | 200 |          5.00 |         4.18 |       2.4 dB |          3.17 |

**v1.2** (compact alternative, 1.3 M):

| Scenario                          |   n | AECMOS echo ↑ | AECMOS deg ↑ | blind ERLE ↑ | DNSMOS OVRL ↑ |
|-----------------------------------|----:|--------------:|-------------:|-------------:|--------------:|
| doubletalk                        | 115 |          4.72 |         2.37 |       8.4 dB |          2.83 |
| doubletalk-with-movement          | 185 |          4.65 |         2.30 |       8.1 dB |          2.79 |
| farend-singletalk                 | 107 |          3.78 |         4.91 |      45.7 dB |          1.80 |
| farend-singletalk-with-movement   | 193 |          4.12 |         4.96 |      40.6 dB |          1.75 |
| nearend-singletalk                | 200 |          5.00 |         4.16 |       2.1 dB |          3.17 |

v1.3 vs v1.2 deltas (same 800-clip set, same eval pipeline):

- **Doubletalk deg MOS +0.25**, dt-with-movement deg MOS +0.13 —
  the wider model + noise-floor-aware loss recipe noticeably reduces
  perceived speech degradation when both talkers are active. The
  primary v1.3 release goal.
- **FE-ST-with-movement ERLE +9.3 dB**, FE-ST ERLE +5.2 dB — v1.3
  cancels far-end echo substantially harder. AECMOS echo MOS drops
  −0.24 / −0.09 at the same time: the residual after cancellation
  rates rougher on AECMOS's perceptual scale even though there's
  numerically less of it. Some users will prefer v1.2's gentler
  trade-off on far-end-only scenes.
- **Nearend-singletalk identical** within noise (deg +0.02,
  OVRL +0.00) — wider capacity doesn't help (or hurt) when there's
  nothing to cancel.
- DNSMOS OVRL is up 0.04–0.21 across all scenarios — the wider
  model produces consistently cleaner-rated output by DNS metrics.

For the original v1.2-vs-v1.1 release notes (the previous headline:
echo MOS +0.80 / +0.72 on FE-ST and FE-ST-with-movement, near-end
deg MOS +0.11), see the v1.2 git tag.

- **AECMOS** (Purin et al., ICASSP 2022) is Microsoft's non-intrusive AEC
  quality predictor. "Echo" rates how well echo was removed; "degradation"
  rates how clean the resulting speech is. 1–5 MOS scale, higher is better.
- **Blind ERLE** is `10·log10(E[mic²] / E[enh²])`. Only meaningful on
  far-end single-talk where the input is echo-only; on scenes with active
  near-end speech it understates echo removal because both numerator and
  denominator are dominated by speech.

PyTorch checkpoint integrity (SHA256):

    22d3e2f33bb8b25ec1c6a928cfb741bb631d45bae2b3759684818b101c95878e  localvqe-v1.3-4.8M.pt
    ff6885e7c8d7d29a8ce963303dcd668ae0f2a7bdafae28631292fe6f06f7cd77  localvqe-v1.2-1.3M.pt

(v1.4-AEC is GGUF-only — no PyTorch checkpoint is published for it.)

GGUF integrity is enforced at load time — the C++ engine checks every
model file against its built-in allowlist of released SHA256 digests
(`ggml/model_hash.cpp`).

## Repository Layout

```
ggml/        C++ streaming inference (GGML graph, CLI, C API, tests)
pytorch/     PyTorch reference implementation (model definition only)
obs-plugin/  OBS Studio audio filter wrapping liblocalvqe.so
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
./ggml/build/bin/localvqe localvqe-v1.2-1.3M-f32.gguf \
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
./ggml/build/bin/bench localvqe-v1.2-1.3M-f32.gguf \
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

### Regression test

`ggml/tests/test_regression.cpp` is an end-to-end check: it runs
`localvqe_process_f32` on a fixed seeded input through each published
`.gguf` and compares against a committed reference output, mirroring
the PyTorch suite under `pytorch/tests/`. Build, fetch the released
GGUFs from HuggingFace, and run via CTest:

```bash
cmake --build ggml/build --target test_regression regression-assets
ctest --test-dir ggml/build --output-on-failure
```

`regression-assets` reuses the same SHA256-verified download path as
`bench-assets`. Missing GGUFs make the corresponding test entry SKIP
rather than fail, so CI without network access still runs cleanly.

To refresh a reference output after an intentional graph change:

```bash
python ggml/tests/regenerate_fixtures.py \
    --gguf ggml/build/bench_assets/localvqe-v1.2-1.3M-f32.gguf
```

### Quantizing (experimental)

Calibrated Q4_K / Q8_0 weights are not yet published. The `quantize`
tool in the C++ build can produce GGUF variants from the F32 reference
for experimentation:

```bash
./ggml/build/bin/quantize localvqe-v1.2-1.3M-f32.gguf localvqe-v1.2-1.3M-q8.gguf Q8_0
```

Expect end-to-end quality loss until proper per-tensor selection and
calibration have been worked through.

## OBS Studio Plugin

`obs-plugin/` wraps `liblocalvqe.so` as an OBS Studio audio source
filter. Once installed it appears as **"LocalVQE (AEC + Noise +
Dereverb)"** in any audio source's filter list. The bundled v1.3 GGUF
is preselected on first use, so noise suppression and dereverberation
work out of the box; AEC additionally requires picking a reference
source — typically "Desktop Audio" — so the model knows what's
playing through the speakers.

The flake provides a dedicated dev shell with libobs alongside the
parent build deps:

```bash
nix develop .#obs-plugin

# Parent library (shared); the plugin links against it.
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release -DLOCALVQE_BUILD_SHARED=ON
cmake --build ggml/build -j$(nproc)

# Stage the bundled GGUF so the plugin's default-model resolver finds it.
cmake --build ggml/build --target regression-assets
cp ggml/build/bench_assets/localvqe-v1.3-4.8M-f32.gguf obs-plugin/data/

# Plugin
cmake -S obs-plugin -B obs-plugin/build -DCMAKE_BUILD_TYPE=Release
cmake --build obs-plugin/build -j$(nproc)
cmake --install obs-plugin/build
```

The install copies the plugin `.so` along with `liblocalvqe.so` and
every `libggml-cpu-*.so` variant into
`~/.config/obs-studio/plugins/obs-localvqe/`, so the tree is
self-contained — no `LD_LIBRARY_PATH`, no system-wide install of
LocalVQE required. Pass
`-DOBS_PLUGIN_DESTINATION=/usr/lib/x86_64-linux-gnu/obs-plugins/obs-localvqe`
to the plugin's configure step for a system-wide install instead.

Restart OBS, right-click any audio source → Filters → Add → **LocalVQE**.

| Property              | Default | Notes                                                                                              |
|-----------------------|---------|----------------------------------------------------------------------------------------------------|
| Model (.gguf)         | bundled | Auto-resolved to `data/localvqe-v1.3-4.8M-f32.gguf` if staged; otherwise browse to a path.        |
| Inference threads     | 4       | Sweet spot on Zen4 (see the benchmark table). Changing this rebuilds the model ctx.               |
| Residual noise gate   | off     | Mutes hops below an RMS threshold; cleans up quiet model residual during silence.                 |
| Gate threshold (dBFS) | -45     | Only used when the gate is on. -45 mutes the typical -60 dBFS residual but preserves speech.      |
| Reference source      | (none)  | For AEC: pick the OBS source feeding your speakers (usually "Desktop Audio"). Off → NS + dereverb only. |

Without a reference, the AEC head sees silence and contributes nothing
— the filter still runs noise suppression and dereverberation on the
mic alone. With a reference, the plugin time-aligns it to the mic
queue via OBS timestamps; the model's AlignBlock then absorbs the
remaining speaker→mic acoustic delay (up to ~1 s on v1.2 and v1.3).

The echo-only model works here too: browse to
`localvqe-v1.4-aec-200K-f32.gguf` as the model and the filter cancels
echo while leaving noise and room acoustics untouched (a reference
source is required — without one this model passes audio through
nearly unchanged).

Tested on Linux. macOS uses the same POSIX `dladdr` path and is
expected to work unchanged. The Windows path is implemented (via
`GetModuleHandleEx` + `GetModuleFileName` in `ensure_backends_loaded`)
but is currently unverified — please open an issue if you hit a
problem there.

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
