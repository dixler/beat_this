# syntax=docker/dockerfile:1.4

# Dockerfile
# GPU-enabled base (PyTorch already installed). Pick a tag that matches your host+driver comfort level.
# Current tags are listed on Docker Hub; example uses a recent CUDA runtime image. :contentReference[oaicite:1]{index=1}
ARG BASE_IMAGE=docker.io/pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/cache

# System deps: ffmpeg for reading non-wav formats, git for optional installs
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps listed in the README “Requirements” section :contentReference[oaicite:2]{index=2}
RUN pip install -U pip \
 && pip install tqdm einops soxr rotary-embedding-torch \
 && pip install "git+https://github.com/CPJKU/beat_this"

RUN curl 'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt' -o /usr/local/bin/final0.ckpt

RUN pip install soundfile

# Simple entrypoint wrapper:
#   beat-this-run INPUT_AUDIO OUTPUT_PATH
# OUTPUT_PATH can be either a directory or a full .beats filename.
RUN cat > /usr/local/bin/beat-this-run << "EOF"
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo \"Usage: beat-this-run INPUT_AUDIO [OUTPUT_PATH]\" >&2
  exit 2
fi

IN=$1
OUT=${2:-}

if [[ -z \"$OUT\" ]]; then
  # Default: write next to the input, same basename + .beats
  base=\"$(basename \"$IN\")\"
  OUT=\"$(dirname \"$IN\")/${base%.*}.beats\"
fi

# If OUT is a directory, pass it directly; beat_this will create per-file outputs. :contentReference[oaicite:3]{index=3}
exec beat_this "$IN" -o "$OUT" --model /usr/local/bin/final0.ckpt
EOF
RUN chmod +x /usr/local/bin/beat-this-run

# Cache for downloaded checkpoints, etc.
RUN mkdir -p /cache
VOLUME ["/cache"]

ENTRYPOINT ["/usr/local/bin/beat-this-run"]
