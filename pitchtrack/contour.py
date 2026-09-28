"""Pitch contours: from the analysis grid to the form the database stores.

The analysis leaves a recording as a pitch track on a fine, uniform grid — one
frequency in hertz per frame, ``0.0`` where the frame carries no pitch. The database
stores a *contour*: the same performance on a coarse grid, measured in cents relative
to a reference frequency, with ``nan`` where a frame carries no pitch.

Three properties make the stored contour directly usable by the application:

* **cents, not hertz.** Pitch is logarithmic to the ear, and folding a contour into a
  pitch-class profile is a modulo-1200 operation on cents;
* **a reference frequency that matches the application's.** The application converts a
  hummed query with the same reference, so both sides of a search are on one scale;
* **``nan`` for the unvoiced frames.** :func:`strawberryfields.utils.to_pitch_class_profile`
  renders those frames as all-zero columns, which is how a silence says *nothing was
  sung here* rather than naming a note.

The grid's spacing is :data:`HOP_SECONDS`, the same value the application resamples a
query to, so a stored contour and a query contour are built the same way.
"""

import warnings

import numpy as np

from .phrases import frame_step

__all__ = [
    "REFERENCE_HZ",
    "HOP_SECONDS",
    "hz_to_cents",
    "median_pool",
    "contour_from_track",
]

REFERENCE_HZ = 55.0

HOP_SECONDS = 0.2


def hz_to_cents(frequencies_hz, reference_hz=REFERENCE_HZ):
    """Convert frequencies in hertz to cents above a reference frequency.

    Frames without a pitch become ``nan`` rather than a number, so downstream steps can
    still tell *no pitch* from *a very low pitch*.

    :param frequencies_hz: Frequencies in hertz; ``0.0`` or less means unvoiced.
    :type frequencies_hz: numpy.ndarray
    :param reference_hz: Frequency in hertz that maps to 0 cents.
    :type reference_hz: float
    :return: Pitch in cents, same shape as ``frequencies_hz``.
    :rtype: numpy.ndarray
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    cents = np.full(frequencies.shape, np.nan)
    voiced = frequencies > 0
    cents[voiced] = 1200.0 * np.log2(frequencies[voiced] / reference_hz)
    return cents


def median_pool(values, frames_per_hop):
    """Reduce a fine grid to a coarse one by taking the median of each block.

    The median of a block is used rather than its mean so that a single frame of jitter
    cannot move the stored pitch, and a block with no voiced frame at all pools to
    ``nan`` so that the silence survives the reduction.

    :param values: Samples to reduce, ``nan`` where undefined.
    :type values: numpy.ndarray
    :param frames_per_hop: How many samples of the fine grid make one of the coarse.
    :type frames_per_hop: int
    :return: The pooled samples; the last block may be padded with ``nan`` to fill.
    :rtype: numpy.ndarray
    """
    if frames_per_hop < 1:
        raise ValueError("frames_per_hop must be at least 1")
    values = np.asarray(values, dtype=np.float64)
    padding = (-len(values)) % frames_per_hop
    padded = np.pad(values, (0, padding), constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(padded.reshape(-1, frames_per_hop), axis=1)


def contour_from_track(times, f0, hop_seconds=HOP_SECONDS):
    """Turn an analysis pitch track into the contour that is stored.

    :param times: Frame times in seconds, uniformly spaced.
    :type times: numpy.ndarray
    :param f0: Pitch in hertz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param hop_seconds: Spacing wanted of the resulting contour, in seconds.
    :type hop_seconds: float
    :return: ``(contour, grid_hop_seconds)`` — the contour in cents with ``nan`` where a
        frame holds no pitch, and the spacing it actually landed on, which is the
        nearest whole number of analysis frames to ``hop_seconds``.
    :rtype: tuple[numpy.ndarray, float]
    """
    frames_per_hop = max(1, round(hop_seconds / frame_step(times)))
    return median_pool(hz_to_cents(f0), frames_per_hop), frames_per_hop * frame_step(times)
