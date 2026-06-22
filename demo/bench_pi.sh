#!/usr/bin/env bash
# Measure the true single-core RTF of the Pi GTCRN-AEC model on THIS machine
# (run it on a Raspberry Pi 5) and overwrite traces/pi/engine.json so the demo
# renders against your own measured number instead of the README fallback.
#
#   ./bench_pi.sh /path/to/LocalVQE-Pi-v1-49k-f32.gguf [test_gtcrn] [frames] [reps]
#
# Build test_gtcrn for aarch64 first (see ggml/docker/Dockerfile.arm64), copy it
# + the gguf to the Pi, then run this. frames=500 ≈ 8 s of audio (256-sample hop).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GGUF="${1:?usage: bench_pi.sh <pi.gguf> [test_gtcrn] [frames] [reps]}"
BIN="${2:-test_gtcrn}"
FRAMES="${3:-500}"; REPS="${4:-20}"
LABEL="${LABEL:-LocalVQE-Pi-v1-49k}"
DEVICE="${DEVICE:-Raspberry Pi 5 · Cortex-A76 · 1 core}"

command -v "$BIN" >/dev/null 2>&1 || [ -x "$BIN" ] || { echo "test_gtcrn not found: $BIN"; exit 1; }
PIN=""; command -v taskset >/dev/null 2>&1 && PIN="taskset -c 0"

echo "[bench] $LABEL  frames=$FRAMES reps=$REPS  (1 thread, pinned)"
LOG="$($PIN "$BIN" --gguf "$GGUF" --bench "$FRAMES" "$REPS" --ggml --threads 1 2>&1)"
echo "$LOG"

# Parse: "  RTF: 0.0480 (avg)  0.0470 (min) ..." and
#        "  per-clip: 388.0 ms (avg)  380.0 ms (min)"
RTF=$(printf '%s\n' "$LOG" | sed -n 's/.*RTF:[^0-9]*[0-9.]*[^0-9]*\([0-9.]*\) (min).*/\1/p' | head -1)
MS=$(printf '%s\n'  "$LOG" | sed -n 's/.*per-clip:[^0-9]*[0-9.]*[^0-9]*\([0-9.]*\) ms (min).*/\1/p' | head -1)
[ -n "$RTF" ] || { echo "could not parse RTF from bench output"; exit 1; }

# hop is 256 samples @ 16 kHz = 16 ms; per_hop = per-clip / frames.
PERHOP=$(python3 -c "print(round(${MS:-0}/${FRAMES}.0, 3))" 2>/dev/null || echo "")
RTFAC=$(python3 -c "print(round(1.0/${RTF}, 1))")

mkdir -p "$HERE/traces/pi"
cat > "$HERE/traces/pi/engine.json" <<EOF
{
  "label": "$LABEL",
  "arch": "GTCRN-AEC (~49K params)",
  "device": "$DEVICE",
  "rtf": $RTF,
  "rt_factor": $RTFAC,
  "per_hop_ms": ${PERHOP:-null},
  "hop_ms": 16.0,
  "measured": "test_gtcrn --bench $FRAMES $REPS --ggml --threads 1, pinned to one core (min of $REPS reps).",
  "source": "bench_pi.sh on $(uname -srm)"
}
EOF
echo "[bench] wrote $HERE/traces/pi/engine.json  (rtf=$RTF → ${RTFAC}× realtime)"
