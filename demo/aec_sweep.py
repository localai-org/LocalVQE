#!/usr/bin/env python3
"""
aec_sweep.py — render the compact-line AEC sweep demo frames (PNG) for the
LocalVQE compact / low-power line (GTCRN-AEC), benchmarked on a Raspberry Pi 5.

A page-at-a-time scrolling spectrogram of the near-end mic. A vertical wipe —
the model's PROCESSING frontier — sweeps each page left→right at the true Pi
speed (~21× realtime). Left of the wipe is the *enhanced* (clean) spectrogram,
right is the *raw* mic. Audio plays in real time and visibly crawls behind; two
bottom bars (play vs proc) over the whole track quantify the speed gap.

Self-contained: numpy / scipy / soundfile / matplotlib + the baked assets. No
torch, no training tree. (On this NixOS host the wheels won't load, so it's run
inside the LocalVQE training Docker, which ships all four — see make.sh.)

Writes out/frames/f_NNNNN.png + out/render.json. The host then encodes/muxes
(make.sh); kept apart so all H.264/AAC/GIF work uses the full host ffmpeg.

    python aec_sweep.py [--fps 30] [--linger 2.5] [--card 3.0]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib import gridspec          # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Dark palette (shared shape with privacy-filter.cpp/demo).
BG = "#0d1117"; INK = "#d7dde5"; DIM = "#6e7681"; FAINT = "#3b424c"
ACC = "#3ec8e0"; ACC_DK = "#16414b"; GOLD = "#e3b341"; GOLD_DK = "#4d3f17"
GREEN = "#46c266"; ROSE = "#e0709a"; WIPE = "#bdf3ff"


def mmss(x):
    x = max(0, int(round(x)))
    return f"{x // 60}:{x % 60:02d}"


def spec_db(x, sr, nperseg=512, hop=160):
    _, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
                   boundary=None, padded=False)
    return t, (20.0 * np.log10(np.abs(Z) + 1e-6)).astype(np.float32)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=str(here / "assets"))
    ap.add_argument("--traces", default=str(here / "traces" / "pi"))
    ap.add_argument("--out", default=str(here / "out"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--linger", type=float, default=2.5,
                    help="seconds to hold after processing completes (audio keeps playing)")
    ap.add_argument("--card", type=float, default=3.0, help="end-card seconds")
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args()

    assets, traces, out = Path(a.assets), Path(a.traces), Path(a.out)
    meta = json.loads((traces / "meta.json").read_text())
    eng = json.loads((traces / "engine.json").read_text())
    sr = meta["sample_rate"]; total_s = float(meta["total_s"])
    rtf = float(eng["rtf"]); rt = int(round(eng.get("rt_factor") or (1.0 / rtf)))
    per_hop = eng.get("per_hop_ms"); hop_ms = float(eng.get("hop_ms", 16.0))
    device = eng.get("device", ""); label = eng.get("label", "LocalVQE-Pi")
    clips = meta.get("clips") or [{"label": "", "start_s": 0.0, "end_s": total_s}]

    mic, _ = sf.read(str(assets / "track_mic.wav"), dtype="float32")
    enh, _ = sf.read(str(assets / "track_enh.wav"), dtype="float32")
    tk, Smic = spec_db(mic, sr)
    _, Senh = spec_db(enh, sr)
    fmax = sr / 2000.0                                   # kHz
    vmax = float(np.percentile(Smic, 99.5)); vmin = vmax - 72.0

    # Time model: rtf = wall/audio (<1 → faster than realtime).
    proc_wall = total_s * rtf
    ab = proc_wall + a.linger                            # audible span
    total_v = ab + a.card
    fps = a.fps
    n_ab = int(round(ab * fps))
    n = int(round(total_v * fps))

    frames_dir = out / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(1280 / a.dpi, 720 / a.dpi), dpi=a.dpi)
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.6, 5.0, 1.5], hspace=0.42,
                           left=0.055, right=0.975, top=0.92, bottom=0.11)
    ax_h = fig.add_subplot(gs[0]); ax_s = fig.add_subplot(gs[1]); ax_b = fig.add_subplot(gs[2])

    def clip_at(p):
        for i, c in enumerate(clips):
            if c["start_s"] <= p < c["end_s"]:
                return i, c
        return len(clips) - 1, clips[-1]

    def blank(ax):
        ax.cla(); ax.set_facecolor(BG); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def draw_header(page_i, clip, done):
        blank(ax_h)
        ax_h.text(0, 0.62, label, color=ACC, fontsize=15, fontweight="bold", va="center")
        ax_h.text(0, 0.02, f"GTCRN-AEC  ·  {device}", color=DIM, fontsize=10, va="center")
        ax_h.text(1, 0.62, f"{rt}× realtime", color=GOLD, fontsize=14,
                  fontweight="bold", va="center", ha="right")
        tail = "done" if done else clip["label"]
        ax_h.text(1, 0.02, f"page {page_i + 1}/{len(clips)}  ·  {tail}",
                  color=DIM, fontsize=10, va="center", ha="right")

    def draw_spec(clip, proc_pos):
        ax_s.cla(); ax_s.set_facecolor(BG)
        a0, b0 = clip["start_s"], clip["end_s"]
        c0 = int(np.searchsorted(tk, a0)); c1 = int(np.searchsorted(tk, b0))
        split = int(np.clip(np.searchsorted(tk, proc_pos), c0, c1))
        s = split - c0
        comp = np.empty((Smic.shape[0], c1 - c0), dtype=np.float32)
        comp[:, :s] = Senh[:, c0:split]
        comp[:, s:] = Smic[:, split:c1]
        ax_s.imshow(comp, origin="lower", aspect="auto", cmap="magma",
                    vmin=vmin, vmax=vmax, extent=[a0, b0, 0, fmax], interpolation="nearest")
        ax_s.axvline(proc_pos, color=WIPE, lw=2.0, alpha=0.95)
        ax_s.set_xlim(a0, b0); ax_s.set_ylim(0, fmax)
        ax_s.set_ylabel("kHz", color=DIM, fontsize=9)
        ax_s.tick_params(colors=DIM, labelsize=8)
        for sp in ax_s.spines.values():
            sp.set_color(FAINT)
        w = b0 - a0
        if proc_pos > a0 + 0.18 * w:
            ax_s.text(a0 + 0.012 * w, fmax * 0.93, "enhanced ✓", color=GREEN,
                      fontsize=11, fontweight="bold", va="top")
        if proc_pos < b0 - 0.16 * w:
            ax_s.text(b0 - 0.012 * w, fmax * 0.93, "raw mic", color=ROSE,
                      fontsize=11, fontweight="bold", va="top", ha="right")
        ax_s.set_title(f"near-end mic spectrogram   ·   {mmss(a0)}–{mmss(b0)}",
                       color=INK, fontsize=11, loc="left", pad=6)

    def bar(y, h, frac, fill, dk, left, right):
        ax_b.add_patch(Rectangle((0, y), 1, h, color=dk, lw=0))
        ax_b.add_patch(Rectangle((0, y), max(0.0, min(1.0, frac)), h, color=fill, lw=0))
        ax_b.text(0, y + h + 0.06, left, color=INK, fontsize=11, va="bottom", fontfamily="monospace")
        if right:
            ax_b.text(1, y + h + 0.06, right, color=DIM, fontsize=10, va="bottom", ha="right")

    def draw_bars(play_pos, proc_pos):
        blank(ax_b)
        bar(0.60, 0.18, proc_pos / total_s, GOLD, GOLD_DK,
            f"proc   {mmss(proc_pos)} / {mmss(total_s)}", f"{rt}× realtime")
        extra = f"{per_hop:.2f} ms / {hop_ms:.0f} ms hop  ·  1 core" if per_hop else "1 core"
        bar(0.12, 0.18, play_pos / total_s, ACC, ACC_DK,
            f"play   {mmss(play_pos)} / {mmss(total_s)}", extra)

    def save(k):
        fig.savefig(frames_dir / f"f_{k:05d}.png", facecolor=BG)

    # Sweep + linger frames.
    for k in range(n_ab):
        wall = k / fps
        play_pos = min(total_s, wall)
        if wall <= proc_wall:
            proc_pos, done = min(total_s, wall / rtf), False
        else:
            proc_pos, done = total_s, True
        pi, clip = clip_at(min(proc_pos, total_s - 1e-3))
        draw_header(pi, clip, done)
        draw_spec(clip, proc_pos)
        draw_bars(play_pos, proc_pos)
        save(k)
        if k % 50 == 0:
            print(f"  frame {k}/{n}  proc={mmss(proc_pos)} play={mmss(play_pos)}")

    # End card — render once, reuse for every card frame.
    for ax in (ax_h, ax_b):
        blank(ax)
    blank(ax_s)
    cx = 0.5
    ax_s.text(cx, 0.78, label, color=ACC, fontsize=30, fontweight="bold", ha="center")
    ax_s.text(cx, 0.58, f"{mmss(total_s)} of audio cleaned in {proc_wall:.1f} s",
              color=INK, fontsize=19, ha="center")
    ax_s.text(cx, 0.45, f"{rt}× realtime   ·   GTCRN-AEC (49K)   ·   one {device}",
              color=DIM, fontsize=13, ha="center")
    ax_s.text(cx, 0.20, "github.com/localai-org/LocalVQE", color=DIM, fontsize=13, ha="center")
    card = frames_dir / "card.png"
    fig.savefig(card, facecolor=BG)
    for k in range(n_ab, n):
        shutil.copyfile(card, frames_dir / f"f_{k:05d}.png")
    card.unlink()

    render = {
        "fps": fps, "n_frames": n, "n_ab_frames": n_ab,
        "proc_wall_s": round(proc_wall, 4), "ab_s": round(ab, 4),
        "total_v_s": round(total_v, 4), "audio_fade_st": round(ab - 0.5, 4),
        "total_s": total_s, "rtf": rtf, "rt_factor": rt,
    }
    (out / "render.json").write_text(json.dumps(render, indent=2))
    print(f"\nwrote {n} frames to {frames_dir}  ({total_v:.1f}s @ {fps}fps; "
          f"sweep {proc_wall:.1f}s)")
    print(f"      {out / 'render.json'}")


if __name__ == "__main__":
    main()
