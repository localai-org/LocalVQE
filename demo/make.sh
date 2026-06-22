#!/usr/bin/env bash
# Render the Pi AEC sweep demo end to end: matplotlib frames (in the LocalVQE
# training Docker, which has numpy/scipy/matplotlib/soundfile) -> H.264 ->
# muxed with the clean audio -> GIF. All encoding uses the host ffmpeg.
#
#   ./make.sh                     # full render at 30 fps
#   FPS=30 LINGER=2.5 ./make.sh
#   SKIP_RENDER=1 ./make.sh       # reuse existing out/frames (re-encode only)
#   RENDER_LOCAL=1 ./make.sh      # run aec_sweep.py with host python (needs deps)
#
# Prereqs: a built demo (gen_assets.py has run -> assets/, traces/pi/meta.json),
# host ffmpeg, and (unless RENDER_LOCAL=1) the training repo + its Docker image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
INF_ROOT="$(cd "$HERE/.." && pwd)"
TRAIN="${LOCALVQE_TRAIN:-$(cd "$HERE/../../LocalVQE-train" 2>/dev/null && pwd || true)}"
OUT="$HERE/out"
FPS="${FPS:-30}"; LINGER="${LINGER:-2.5}"; CARD="${CARD:-3.0}"
GIF_W="${GIF_W:-960}"; GIF_FPS="${GIF_FPS:-15}"

[ -f "$HERE/assets/track_mic.wav" ] || { echo "no baked assets — run gen_assets.py first (see README)"; exit 1; }

# 1. Render frames.
if [ "${SKIP_RENDER:-0}" != "1" ]; then
  ARGS="--fps $FPS --linger $LINGER --card $CARD"
  if [ "${RENDER_LOCAL:-0}" = "1" ]; then
    echo "[make] rendering frames (host python)"
    python3 "$HERE/aec_sweep.py" $ARGS
  else
    [ -n "$TRAIN" ] && [ -x "$TRAIN/train/scripts/docker-run.sh" ] || {
      echo "training repo not found (set LOCALVQE_TRAIN, or RENDER_LOCAL=1)"; exit 1; }
    echo "[make] rendering frames (training Docker)"
    "$TRAIN/train/scripts/docker-run.sh" \
      --docker-args "-v $INF_ROOT:/workspace/localvqe_inf" \
      python /workspace/localvqe_inf/demo/aec_sweep.py $ARGS
  fi
fi

[ -f "$OUT/render.json" ] || { echo "no out/render.json — render failed"; exit 1; }
read -r RFPS AB FADEST TOTAL < <(python3 -c "import json;d=json.load(open('$OUT/render.json'));print(d['fps'],d['ab_s'],d['audio_fade_st'],d['total_v_s'])")
echo "[make] fps=$RFPS audible=${AB}s total=${TOTAL}s"

# 2. Frames -> silent H.264.
ffmpeg -y -loglevel error -framerate "$RFPS" -i "$OUT/frames/f_%05d.png" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "$OUT/sweep_silent.mp4"

# 3. Mux the clean (enhanced) audio: real-time playback for the audible span,
#    fade out, silence under the end card.
ffmpeg -y -loglevel error -i "$OUT/sweep_silent.mp4" -i "$HERE/assets/track_enh.wav" \
  -filter_complex "[1:a]atrim=0:${AB},afade=t=out:st=${FADEST}:d=0.5,apad[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 160k -t "$TOTAL" \
  -movflags +faststart "$OUT/sweep.mp4"

# 4. GIF (palette for clean colour).
ffmpeg -y -loglevel error -i "$OUT/sweep.mp4" \
  -vf "fps=$GIF_FPS,scale=$GIF_W:-1:flags=lanczos,palettegen=stats_mode=diff" "$OUT/_pal.png"
ffmpeg -y -loglevel error -i "$OUT/sweep.mp4" -i "$OUT/_pal.png" \
  -lavfi "fps=$GIF_FPS,scale=$GIF_W:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" "$OUT/sweep.gif"
rm -f "$OUT/_pal.png" "$OUT/sweep_silent.mp4"

echo "[make] done:"
ls -lh "$OUT/sweep.mp4" "$OUT/sweep.gif" | awk '{print "   "$5"  "$9}'
