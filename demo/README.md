# Compact-line AEC sweep demo

A graphical "how fast is it" demo for the LocalVQE **compact / low-power line**
(GTCRN-AEC, ~49 K params), benchmarked here on a single **Raspberry Pi 5** core
(≈21× realtime) as one example of a low-power target.

![Pi AEC sweep](out/sweep.gif)

A page-at-a-time scrolling spectrogram of a noisy/echoey near-end mic track. A
vertical wipe — the model's **processing frontier** — sweeps each page
left→right at the true Pi speed; left of the wipe is the **enhanced** (clean)
spectrogram, right is the **raw** mic. The audio plays in real time and visibly
crawls behind: by the time you've heard ~9 s, the model has already cleaned the
whole 3-minute track. The two bottom bars (play vs proc, over the whole track)
are the point of reference for the speed gap.

Full-quality MP4: [`out/sweep.mp4`](out/sweep.mp4).

## How it's built

The demo is **baked once** and **rendered** from the baked assets (like
`privacy-filter.cpp/demo`), with the *timing* measured separately on a real Pi.

The bake cleans the audio with the PyTorch reference (parity-identical to the
ggml engine, ~1e-6). The GTCRN line is now also in the C++ `localvqe` CLI
(`localvqe model.gguf --fe v1.4-aec.gguf …`), so a torch-free bake is possible
too — but the C++ DAF front-end's first ~1-2 s convergence transient differs
slightly from the fast-converging PyTorch front-end, so the PyTorch bake is kept
for the cleanest before/after visuals.

```
gen_assets.py   bake (torch): example clips -> long track -> Pi model -> wavs + meta
bench_pi.sh     measure true single-core RTF on an actual Pi 5 -> engine.json
aec_sweep.py    render (matplotlib): spectrogram + wipe + bars -> out/frames/*.png
make.sh         orchestrate render -> H.264 -> mux clean audio -> GIF (host ffmpeg)
```

### 1. Bake the audio (once)

Needs torch + the (unpublished) **LocalVQE training repo** and its Pi
checkpoints, so it runs in the training Docker with *this* repo mounted
alongside:

```sh
cd <LocalVQE-train>
LOCALVQE_INF=<this repo>                  # e.g. ../LocalVQE
train/scripts/docker-run.sh \
  --docker-args "-v $LOCALVQE_INF:/workspace/localvqe_inf" \
  python /workspace/localvqe_inf/demo/gen_assets.py --minutes 3
```

Writes `assets/track_mic.wav`, `assets/track_enh.wav`, `traces/pi/meta.json`.
The full-enhance Pi model (`LocalVQE-Pi-v1-49k`: echo + noise + dereverb) is the
default; swap to the echo-only keep-noise build with
`--gtcrn-ckpt .../gtcrn_aec_keepnoise.pt`.

### 2. (Optional) Measure on a real Pi

`traces/pi/engine.json` ships the README's measured 21× as a fallback so the
demo renders standalone. To use your own Pi's number, cross-compile `test_gtcrn`
for aarch64 (`ggml/docker/Dockerfile.arm64`), copy it + the gguf to the Pi, and:

```sh
./bench_pi.sh LocalVQE-Pi-v1-49k-f32.gguf ./test_gtcrn   # overwrites engine.json
```

### 3. Render

On NixOS the Python wheels won't load on the host, so `make.sh` renders the
frames in the **training Docker** (which ships numpy/scipy/matplotlib/soundfile)
and does all H.264/AAC/GIF encoding with the **host ffmpeg**:

```sh
./make.sh                  # -> out/sweep.mp4, out/sweep.gif
FPS=30 LINGER=2.5 ./make.sh
SKIP_RENDER=1 ./make.sh    # re-encode existing out/frames only
RENDER_LOCAL=1 ./make.sh   # render with host python instead (needs requirements.txt deps)
```

## Notes

- `--minutes` sets the track length; at ≈21× it renders to ~`minutes·60·0.048` s
  of sweep (3 min → ~8.6 s), so a "decent sized track" keeps the wipe sweeping
  for a watchable while without slowing the visuals.
- The on-screen "21× realtime" is the GTCRN-AEC **net** figure (`test_gtcrn
  --bench`); the DSP echo-cancel front-end runs on top, on CPU. `engine.json`'s
  `measured` field states exactly what was timed.
- Pages follow the baked clip boundaries (`meta.json`), so the header names the
  scenario (double-talk, far-end single-talk, near-end + noise, …).
