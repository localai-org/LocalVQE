#!/usr/bin/env python3
"""
gen_assets.py — bake the audio assets for the Pi AEC sweep demo (run once).

The compact / low-power line (GTCRN-AEC, ~49 K params) is baked here with the
PyTorch reference, which is parity-identical to the ggml engine (~1e-6) and the
simplest source of clean before/after audio. (The line is also runnable from the
C++ `localvqe` CLI now — `--fe v1.4-aec.gguf` — but its DAF front-end's short
convergence transient differs from PyTorch's, so PyTorch is kept for the visuals.)
Timing is measured separately on a real Pi (bench_pi.sh).

This assembles a multi-minute near-end track from the Space's example clip
pairs, cleans it with the full-enhance Pi model (LocalVQE-Pi-v1-49k:
echo + noise + dereverb), and writes the raw ("before") and enhanced ("after")
tracks plus a meta.json the renderer (aec_sweep.py) consumes.

It depends on torch + the (unpublished) training tree, so it runs inside the
LocalVQE training Docker with THIS (inference) repo mounted alongside it:

    cd <LocalVQE-train>                       # the training repo
    LOCALVQE_INF=<LocalVQE>                    # this inference repo
    train/scripts/docker-run.sh \
        --docker-args "-v $LOCALVQE_INF:/workspace/localvqe_inf" \
        python /workspace/localvqe_inf/demo/gen_assets.py --minutes 3

Outputs (next to this script): assets/track_mic.wav, assets/track_enh.wav,
traces/pi/meta.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# The training tree (mounted at /workspace/localvqe) holds the Space packages
# that wrap the Pi model + DAF front-end. Make them importable.
TRAIN_ROOT = Path("/workspace/localvqe")
SPACE_DIR = TRAIN_ROOT / "space"
sys.path.insert(0, str(SPACE_DIR))

SR = 16000
# Example pairs, ordered dirtiest-first so the "before" spectrogram opens on
# visibly noisy/echoey content (the full-enhance model then visibly cleans it).
CLIP_ORDER = [
    ("ne_st_noisy", "near-end + noise (5 dB)"),
    ("fe_st",       "far-end single-talk (echo)"),
    ("dt",          "double-talk"),
    ("fe_st2",      "far-end single-talk"),
    ("ne_st_clean", "near-end + light noise (20 dB)"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    if sr != SR:
        raise SystemExit(f"{path}: expected {SR} Hz, got {sr}")
    return x


def _edge_fade(x: np.ndarray, n: int) -> np.ndarray:
    """In-place short linear fade in/out at both ends (declick at joins)."""
    n = min(n, len(x) // 2)
    if n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=3.0,
                    help="target near-end track length (default 3.0)")
    ap.add_argument("--examples", default=str(SPACE_DIR / "examples"),
                    help="dir of <name>_mic.wav / <name>_ref.wav pairs")
    ap.add_argument("--gtcrn-ckpt", default=None,
                    help="full-enhance Pi GTCRN-AEC .pt (default: os400_move)")
    ap.add_argument("--fe-ckpt", default=None,
                    help="v1.4 cascade .pt for the DAF front-end (default: nsdr-200K)")
    ap.add_argument("--out", default=str(here / "assets"),
                    help="output dir for the baked wavs")
    args = ap.parse_args()

    ex = Path(args.examples)
    gtcrn_ckpt = Path(args.gtcrn_ckpt) if args.gtcrn_ckpt else _default_gtcrn()
    fe_ckpt = Path(args.fe_ckpt) if args.fe_ckpt else _default_fe()
    for p in (ex, gtcrn_ckpt, fe_ckpt):
        if not Path(p).exists():
            raise SystemExit(f"missing: {p}")

    # The Pi model lives behind the Space's GTCRN_AEC wrapper (front-end → e,
    # yhat=mic−e → complex mask → iSTFT). Same code path the demo Space uses.
    from localvqe_gtcrn import load_gtcrn_aec  # noqa: E402
    print(f"loading Pi model: gtcrn={gtcrn_ckpt.name}  fe={fe_ckpt.name}")
    model = load_gtcrn_aec(str(fe_ckpt), str(gtcrn_ckpt))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GTCRN-AEC {n_params:,} params")

    # Build the ordered playlist long enough to hit the target length.
    target = args.minutes * 60.0
    playlist = []
    acc = 0.0
    i = 0
    while acc < target:
        name, label = CLIP_ORDER[i % len(CLIP_ORDER)]
        mic_p, ref_p = ex / f"{name}_mic.wav", ex / f"{name}_ref.wav"
        if not mic_p.exists() or not ref_p.exists():
            i += 1
            if i > len(CLIP_ORDER) * 4:
                raise SystemExit(f"no usable example pairs under {ex}")
            continue
        dur = len(_load_mono(mic_p)) / SR
        playlist.append((name, label, mic_p, ref_p, dur))
        acc += dur
        i += 1

    mic_parts, enh_parts, clips = [], [], []
    fade = int(0.005 * SR)
    t = 0.0
    src_sha = {}
    for name, label, mic_p, ref_p, dur in playlist:
        mic = _load_mono(mic_p)
        ref = _load_mono(ref_p)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]
        # Per-clip processing matches the Space (each example is an independent
        # recording; the DAF front-end re-acquires alignment per clip).
        enh = np.asarray(model.process(mic, ref), dtype=np.float32)[:n]
        mic = _edge_fade(mic.copy(), fade)
        enh = _edge_fade(enh.copy(), fade)
        mic_parts.append(mic)
        enh_parts.append(enh)
        seg = n / SR
        clips.append({"label": label, "name": name,
                      "start_s": round(t, 4), "end_s": round(t + seg, 4)})
        t += seg
        src_sha[mic_p.name] = _sha256(mic_p)
        src_sha[ref_p.name] = _sha256(ref_p)
        print(f"  + {name:<12} {seg:5.1f}s  → {t:6.1f}s")

    mic_track = np.concatenate(mic_parts).astype(np.float32)
    enh_track = np.concatenate(enh_parts).astype(np.float32)
    # Shared gain so the before/after loudness relationship is preserved.
    peak = float(max(np.abs(mic_track).max(), np.abs(enh_track).max(), 1e-6))
    g = 0.99 / peak
    mic_track *= g
    enh_track *= g

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "track_mic.wav"), mic_track, SR, subtype="PCM_16")
    sf.write(str(out / "track_enh.wav"), enh_track, SR, subtype="PCM_16")

    meta = {
        "sample_rate": SR,
        "total_s": round(len(mic_track) / SR, 4),
        "n_samples": int(len(mic_track)),
        "clips": clips,
        "model": {
            "label": "LocalVQE-Pi-v1-49k",
            "n_params": int(n_params),
            "gtcrn_ckpt": gtcrn_ckpt.name,
            "gtcrn_sha256": _sha256(gtcrn_ckpt),
            "fe_ckpt": fe_ckpt.name,
            "fe_sha256": _sha256(fe_ckpt),
        },
        "sources_sha256": src_sha,
        "generator": "demo/gen_assets.py",
    }
    meta_p = here / "traces" / "pi" / "meta.json"
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out/'track_mic.wav'}, {out/'track_enh.wav'}")
    print(f"      {meta_p}  ({meta['total_s']:.1f}s, {len(clips)} segments)")


def _default_gtcrn() -> Path:
    for c in ("gtcrn_ns_aec_os400_move.pt", "gtcrn_ns_aec_os400.pt"):
        p = TRAIN_ROOT / "checkpoints" / c
        if p.exists():
            return p
    return TRAIN_ROOT / "checkpoints" / "gtcrn_ns_aec_os400_move.pt"


def _default_fe() -> Path:
    return TRAIN_ROOT / "checkpoints" / "release" / "localvqe-v1.4-nsdr-200K.pt"


if __name__ == "__main__":
    main()
