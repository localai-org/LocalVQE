#!/usr/bin/env bash
# Render the full AEC blind test set through a release gguf with the
# localvqe CLI (parallel), producing the *_enh.wav mirror layout that
# train/scripts/score_blind_prerendered.py consumes.
set -uo pipefail
GGUF="${1:?gguf}"
BLIND="${2:?blind_dir}"
OUT="${3:?out_dir}"
JOBS="${4:-6}"
BIN="$(dirname "$0")/build-release/bin/localvqe"

mkdir -p "$OUT"
for d in "$BLIND"/*/; do mkdir -p "$OUT/$(basename "$d")"; done

find "$BLIND" -name '*_mic.flac' | sort | \
xargs -P "$JOBS" -I{} bash -c '
    mic="$1"; stem="${mic%_mic.flac}"
    scen="$(basename "$(dirname "$mic")")"
    out="'"$OUT"'/$scen/$(basename "$stem")_enh.wav"
    [ -f "$out" ] && exit 0
    "'"$BIN"'" "'"$GGUF"'" --in-wav "$mic" "${stem}_lpb.flac" --out-wav "$out" >/dev/null 2>&1 \
        || echo "FAIL: $mic" >&2
' _ {}
echo "done: $(find "$OUT" -name '*_enh.wav' | wc -l) files"
