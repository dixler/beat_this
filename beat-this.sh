#!/bin/bash
# run_beat_this.sh (inspiration: your stemgen runner)

set -euo pipefail

BASENAME="$(basename "$1")"
OUTDIR="$(mktemp -d)"
cp "$1" "$OUTDIR/"

# Build once:
#   podman build -t beat-this .

podman run \
  --rm \
  --network=none \
  --gpus all \
  -v "$OUTDIR:/data" \
  -v beat_this_cache:/cache \
  beat-this \
  "/data/$BASENAME" "/data"

# Collect output
OUTFILE="$(find "$OUTDIR" -maxdepth 1 -type f -name '*.beats' | head -n 1)"
mv "$OUTFILE" "./"

if [[ "$OUTDIR" == /tmp/* ]]; then
  rm -rf "$OUTDIR"
fi

