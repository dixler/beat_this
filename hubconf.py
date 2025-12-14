dependencies = [
    "torch",
    "torchaudio",
    "numpy",
    "rotary_embedding_torch",
    "einops",
    "soxr",
]

from beat_this.inference import (
    load_model as beat_this,
    BeatThis,
    Spect2Frames,
    Audio2Frames,
    Audio2Boundaries,
    File2Boundaries,
    File2File,
)

# Backward compatibility aliases
Audio2Beats = Audio2Boundaries
File2Beats = File2Boundaries
