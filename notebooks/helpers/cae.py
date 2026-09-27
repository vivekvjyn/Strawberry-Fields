"""Melodic features from a complex autoencoder over the constant-Q transform.

A window of CQT magnitudes is projected onto a bank of complex basis functions, and
only the *magnitude* of each projection is kept:

.. math::

    a_{i,k} = \\sqrt{(\\mathbf{w}_k^\\top \\mathbf{x}_i)^2
                     + (\\mathbf{w}_{k+d}^\\top \\mathbf{x}_i)^2},
    \\qquad k = 1, \\dots, d

Because the magnitude discards the phase of the projection, a window and the same
window shifted by a fraction of a cycle look alike; what remains is *how much* of each
learned shape of spectrum is present. The bases are trained to reconstruct the CQT, so
they span the spectral shapes that actually recur in this repertoire — a plain
pitch-class histogram has to be told what matters, this one is told by the data.

The model reads ``models/model_complex_auto_cqt.save``: one linear layer of
``2 * N_BASES`` rows over ``N_BINS * N_NGRAM`` inputs, split into real and imaginary
halves.
"""

from pathlib import Path

import librosa
import numpy as np

__all__ = ["extract_features", "window_seconds", "frame_step"]

SR = 44100                 # analysis sample rate, fixed by the training configuration
HOP_LENGTH = 1984          # CQT frames, 45 ms apart
N_BINS = 120               # CQT bins, 24 per octave from 80 Hz
BINS_PER_OCTAVE = 24
FMIN = 80.0
N_NGRAM = 32               # CQT frames per window
N_BASES = 256              # complex basis functions = 512 real outputs

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "model_complex_auto_cqt.save"

# CQT frames summarised into one feature frame, chosen so the grid lands near the
# quarter-second the rest of the pipeline was written against. Successive CAE rows
# share 97 % of their window, so nothing musically is lost by stepping over them.
DEFAULT_FRAME_SECONDS = 0.25


def window_seconds():
    """Seconds of audio one feature frame summarises."""
    return N_NGRAM * HOP_LENGTH / SR


def frame_step(frame_seconds=DEFAULT_FRAME_SECONDS):
    """Seconds between feature frames, snapped to a whole number of CQT frames."""
    step = max(1, round(frame_seconds / (HOP_LENGTH / SR)))
    return step * HOP_LENGTH / SR


def _standardize(x, axis=-1):
    """Zero mean, unit variance along ``axis`` — the contrast normalisation."""
    x = x - x.mean(axis=axis, keepdims=True)
    return x / (x.std(axis=axis, keepdims=True) + 1e-8)


def _basis(path=WEIGHTS):
    """The trained layer, as a ``(2 * N_BASES, N_BINS * N_NGRAM)`` array."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return state["layer.weight"].detach().cpu().numpy().astype(np.float32)


def cqt_frames(path):
    """Standardised CQT magnitudes of a recording, as ``(frames, N_BINS)``."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    cqt = librosa.cqt(y, sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
                      fmin=FMIN, hop_length=HOP_LENGTH)
    return _standardize(np.abs(cqt), axis=0).T.astype(np.float32)


def extract_features(path, *, weights=None, frame_seconds=DEFAULT_FRAME_SECONDS,
                     rows=4096, verbose=False):
    """Encode a recording into a feature frame per ``frame_seconds`` of audio.

    The CQT is standardised twice, as the model expects: once across the bins of each
    frame, so the transform describes the *shape* of the spectrum rather than its
    level, and once across each window, so a loud passage and a quiet one that share a
    contour are the same feature.

    :param path: Audio file to encode.
    :type path: str or pathlib.Path
    :param weights: Trained layer to project with; the packaged one if ``None``.
    :type weights: numpy.ndarray or None
    :param frame_seconds: Spacing of the returned feature frames, rounded to a whole
        number of CQT frames.
    :type frame_seconds: float
    :param rows: Rows encoded per matrix multiply, bounding the working memory.
    :type rows: int
    :param verbose: Print the recording length before encoding.
    :type verbose: bool
    :return: ``(features, times, window)`` — a ``(frames, N_BASES)`` array of
        magnitudes, the centre time of each frame in seconds, and the number of
        seconds of audio each frame summarises.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, float]
    """
    frames = cqt_frames(path)
    basis = _basis() if weights is None else np.asarray(weights, dtype=np.float32)
    window = window_seconds()
    if verbose:
        print(f"{len(frames) * HOP_LENGTH / SR:,.0f} s of audio -> "
              f"{len(frames)} CQT frames")

    n_rows = max(0, len(frames) - N_NGRAM)
    if not n_rows:
        return np.zeros((0, N_BASES), dtype=np.float32), np.zeros(0), window

    view = np.lib.stride_tricks.sliding_window_view(frames, N_NGRAM, axis=0)[:n_rows]
    half = basis.shape[0] // 2
    encoded = np.empty((n_rows, N_BASES), dtype=np.float32)
    for lo in range(0, n_rows, rows):
        hi = min(lo + rows, n_rows)
        # One window per row, standardised across its whole extent.
        grams = _standardize(np.ascontiguousarray(view[lo:hi]).reshape(hi - lo, -1))
        projected = grams @ basis.T
        encoded[lo:hi] = np.sqrt(projected[:, :half] ** 2 + projected[:, half:] ** 2)

    step = max(1, round(frame_step(frame_seconds) / (HOP_LENGTH / SR)))
    keep = np.arange(0, n_rows, step)
    times = (keep.astype(np.float64) + N_NGRAM / 2.0) * HOP_LENGTH / SR
    return encoded[keep], times, window
