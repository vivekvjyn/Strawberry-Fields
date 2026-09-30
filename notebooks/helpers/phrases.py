"""Cut the discarded repeats out of a recording, as a waveform.

The search for repeated phrases is compIAM's own, run the way the original Sancara
notebook runs it — see :mod:`helpers.reference` — and what it yields is a mask over the
pitch frames: which moments belong to a repeat that should go. compIAM stops at the
mask; this module is the listening end. It lifts the frame-level mask onto the
waveform's sample grid, then either mutes the marked spans in place or cuts them out
and crossfades each join, so the result states every phrase once, in the order it was
first heard. Nothing is written to disk — the waveform comes back for the caller to
play, plot or save.
"""

import librosa
import numpy as np

__all__ = ["filter_audio_array"]


def _frame_step(times):
    """Seconds between consecutive pitch frames.

    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :return: The spacing, or the default 25 ms if there is not a second frame.
    :rtype: float
    """
    times = np.asarray(times, dtype=np.float64)
    return float(times[1] - times[0]) if len(times) > 1 else 0.025


def _sample_mask(n_samples, sample_rate, times, duplicates):
    """Lift a pitch-frame mask onto the waveform's sample grid.

    The mask is on the pitch grid, which is far too coarse to cut a waveform on, so
    each sample takes its flag from the frame whose span covers it: a cut lands on a
    frame boundary instead of anywhere in between.

    :param n_samples: Length of the waveform in samples.
    :type n_samples: int
    :param sample_rate: Sample rate of the waveform.
    :type sample_rate: int
    :param times: Frame times of the pitch track.
    :type times: numpy.ndarray
    :param duplicates: Boolean per pitch frame; ``True`` where it is a discarded repeat.
    :type duplicates: numpy.ndarray
    :return: Boolean per sample, ``True`` where the sample is kept.
    :rtype: numpy.ndarray
    """
    step = _frame_step(times)
    sample_times = np.arange(n_samples) / sample_rate
    flags = np.asarray(duplicates, dtype=bool)
    frame_index = np.clip((sample_times / step).astype(int), 0, len(flags) - 1)
    return ~flags[frame_index]


def _join_runs(audio, keep, fade):
    """Concatenate the kept runs of a waveform, crossfading across each gap.

    :param audio: The original waveform.
    :type audio: numpy.ndarray
    :param keep: Boolean per sample, ``True`` where the sample is kept.
    :type keep: numpy.ndarray
    :param fade: Crossfade length in samples.
    :type fade: int
    :return: The shortened waveform.
    :rtype: numpy.ndarray
    """
    edges = np.flatnonzero(np.diff(keep.astype(np.int8)))
    starts = list(edges[keep[edges + 1]] + 1)
    stops = list(edges[~keep[edges + 1]] + 1)
    if keep[0]:
        starts.insert(0, 0)
    if keep[-1]:
        stops.append(len(audio))

    if fade <= 0 or not starts:
        return np.concatenate([audio[a:b] for a, b in zip(starts, stops)])

    pieces = []
    for i, (start, stop) in enumerate(zip(starts, stops)):
        chunk = audio[start:stop].copy()
        if i > 0:
            width = min(fade, len(chunk))
            chunk[:width] *= np.linspace(0.0, 1.0, width)
        if i < len(starts) - 1:
            width = min(fade, len(chunk))
            chunk[-width:] *= np.linspace(1.0, 0.0, width)
        pieces.append(chunk)
    return np.concatenate(pieces) if pieces else audio[:0]


def filter_audio_array(audio_path, times, duplicates, *, mode="cut",
                       fade_seconds=0.008, sample_rate=None):
    """Return a recording with the discarded repeats taken out, as a waveform.

    This is where the search turns into something you can listen to: the repeat regions
    found in the pitch track are mapped back onto the waveform and removed, leaving each
    phrase present once, in the order it was first stated. Nothing is written to disk —
    the waveform comes back so it can be played, plotted or saved by the caller.

    :param audio_path: The original recording.
    :type audio_path: str or pathlib.Path
    :param times: Frame times of the pitch track the mask was built on.
    :type times: numpy.ndarray
    :param duplicates: Boolean per pitch frame; ``True`` where it is a discarded repeat.
    :type duplicates: numpy.ndarray
    :param mode: ``"cut"`` removes the repeats and closes the gap, so the result is
        shorter. ``"mute"`` silences them in place, so the length and every surviving
        moment's timestamp are unchanged — the right choice when the result has to line
        up against the original.
    :type mode: str
    :param fade_seconds: Length of the crossfade applied at each join in ``"cut"`` mode.
        Without it the waveform steps abruptly at every cut and the result clicks; a few
        milliseconds is enough to hide it.
    :type fade_seconds: float
    :param sample_rate: Resample the original to this rate first; ``None`` keeps it.
    :type sample_rate: int or None
    :return: ``(waveform, kept_fraction)`` — the filtered audio, and the share of the
        original duration that survived.
    :rtype: tuple[numpy.ndarray, float]
    """
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    keep = _sample_mask(len(audio), sr, times, duplicates)
    kept_fraction = float(keep.mean())

    if mode == "mute":
        return np.where(keep, audio, 0.0), kept_fraction
    if mode == "cut":
        return _join_runs(audio, keep, int(round(fade_seconds * sr))), kept_fraction
    raise ValueError(f"mode must be 'cut' or 'mute', not {mode!r}")
