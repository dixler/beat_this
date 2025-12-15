from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class Postprocessor:
    """Postprocess framewise boundary logits into time stamps."""

    def __init__(self, type: str = "minimal", fps: int = 50):
        assert type == "minimal"
        self.type = type
        self.fps = fps

    def __call__(
        self, boundary: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[np.ndarray, ...] | np.ndarray:
        batched = boundary.ndim > 1
        if padding_mask is None:
            padding_mask = torch.ones_like(boundary, dtype=torch.bool)
        if not batched:
            boundary = boundary.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        boundary_peaks = boundary.masked_fill(~padding_mask, -1000)
        boundary_peaks = boundary_peaks.masked_fill(
            boundary_peaks != F.max_pool1d(boundary_peaks, 7, 1, 3), -1000
        )
        boundary_peaks = boundary_peaks > 0
        results = [
            self._postp_minimal_item(peaks, mask)
            for peaks, mask in zip(boundary_peaks, padding_mask)
        ]

        if not batched:
            return results[0]
        return tuple(results)

    def _postp_minimal_item(self, padded_peaks, mask):
        boundary_frame = torch.nonzero(padded_peaks[mask]).cpu().numpy()[:, 0]
        boundary_frame = deduplicate_peaks(boundary_frame, width=1)
        return boundary_frame / self.fps


def deduplicate_peaks(peaks, width=1) -> np.ndarray:
    """
    Replaces groups of adjacent peak frame indices that are each not more
    than `width` frames apart by the average of the frame indices.
    """
    result = []
    peaks = map(int, peaks)  # ensure we get ordinary Python int objects
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)
