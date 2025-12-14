from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class Postprocessor:
    """Postprocessor for the phrase boundary predictions of the model.
    The postprocessor takes the (framewise) model predictions (boundaries) and the padding mask,
    and returns the postprocessed boundaries as list of times in seconds.
    The boundaries can be 1D arrays (for only 1 piece) or 2D arrays, if a batch of pieces is considered.
    The output dimensionality is the same as the input dimensionality.
    Only minimal postprocessing is implemented:
        - minimal: a simple postprocessing that takes the maximum of the framewise predictions,
        and removes adjacent peaks.
    Args:
        type (str): the type of postprocessing to apply. Only "minimal" is supported. Default is "minimal".
        fps (int): the frames per second of the model framewise predictions. Default is 50.
    """

    def __init__(self, type: str = "minimal", fps: int = 50):
        assert type in ["minimal", "dbn"]
        self.type = type
        self.fps = fps
        if type == "dbn":
            # DBN not applicable for phrase boundaries
            print("Warning: DBN postprocessing not applicable for phrase boundaries, using minimal instead")
            self.type = "minimal"

    def __call__(
        self,
        boundary: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Apply postprocessing to the input boundary tensor. Works with batched and unbatched inputs.
        The output is a list of times in seconds, or a list of lists of times in seconds, if the input is batched.

        Args:
            boundary (torch.Tensor): The input boundary tensor.
            padding_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            np.ndarray: The postprocessed boundary tensor.
        """
        batched = False if boundary.ndim == 1 else True
        if padding_mask is None:
            padding_mask = torch.ones_like(boundary, dtype=torch.bool)

        # if boundary is 1D tensor, add a batch dimension
        if not batched:
            boundary = boundary.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        if self.type == "minimal":
            postp_boundary = self.postp_minimal(boundary, padding_mask)
        else:
            raise ValueError("Invalid postprocessing type")

        # remove the batch dimension if it was added
        if not batched:
            postp_boundary = postp_boundary[0]

        # return the postprocessed boundary
        return postp_boundary

    def postp_minimal(self, boundary, padding_mask):
        # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
        pred_logits = boundary.masked_fill(~padding_mask, -1000)
        # pick maxima within +/- 70ms
        pred_peaks = pred_logits.masked_fill(
            pred_logits != F.max_pool1d(pred_logits, 7, 1, 3), -1000
        )
        # keep maxima with over 0.5 probability (logit > 0)
        pred_peaks = pred_peaks > 0
        # run the piecewise operations
        with ThreadPoolExecutor() as executor:
            postp_boundary = tuple(
                executor.map(
                    self._postp_minimal_item, pred_peaks, padding_mask
                )
            )
        return postp_boundary

    def _postp_minimal_item(self, padded_boundary_peaks, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        boundary_peaks = padded_boundary_peaks[mask]
        # pass from a boolean array to a list of times in frames.
        boundary_frame = torch.nonzero(boundary_peaks).cpu().numpy()[:, 0]
        # remove adjacent peaks
        boundary_frame = deduplicate_peaks(boundary_frame, width=1)
        # convert from frame to seconds
        boundary_time = boundary_frame / self.fps
        return boundary_time


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
