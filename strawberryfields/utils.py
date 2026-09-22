import librosa
import numpy as np
from rich.progress import Progress

from strawberryfields.dtw import dtw
from strawberryfields.pyin import pyin
from strawberryfields.salience import rasterise


def load_audio(path):
    """Decode an audio file to a mono waveform at its native sampling rate.

    :param path: Path to the audio file (any format librosa/ffmpeg can decode).
    :type path: str
    :return: The waveform and its sampling rate in Hz.
    :rtype: tuple[numpy.ndarray, int]
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float64), sr


def extract_f0(y, sr, fmin, fmax, frame_length, hop_length):
    """Estimate the fundamental frequency of a waveform with pYIN.

    Unvoiced frames are returned as ``0.0`` Hz.

    :param y: Mono waveform.
    :type y: numpy.ndarray
    :param sr: Sampling rate of ``y`` in Hz.
    :type sr: int
    :param fmin: Lowest frequency to search for, in Hz.
    :type fmin: float
    :param fmax: Highest frequency to search for, in Hz.
    :type fmax: float
    :param frame_length: Analysis window length in samples.
    :type frame_length: int
    :param hop_length: Number of samples between consecutive frames.
    :type hop_length: int
    :return: Frame times in seconds and the f0 value of each frame in Hz.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    f0, voiced_flag, _ = pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~voiced_flag] = 0.0
    times = np.arange(len(f0)) * hop_length / sr
    return times, f0


def hz_to_cents(f0, f_ref):
    """Convert frequencies in Hz to cents relative to a reference frequency.

    Non-positive (unvoiced) values become ``nan``.

    :param f0: Frequencies in Hz.
    :type f0: numpy.ndarray
    :param f_ref: Reference frequency in Hz that maps to 0 cents.
    :type f_ref: float
    :return: Pitch in cents, same shape as ``f0``.
    :rtype: numpy.ndarray
    """
    f0 = np.asarray(f0, dtype=np.float64)
    cents = np.full_like(f0, np.nan)
    voiced = f0 > 0
    cents[voiced] = 1200.0 * np.log2(f0[voiced] / f_ref)
    return cents


def resample_uniform(times, values, hop_seconds, duration=None):
    """Linearly resample a series onto a uniform time grid, ignoring ``nan`` samples.

    Grid points outside the range of valid samples are ``nan``.

    :param times: Sample times in seconds.
    :type times: numpy.ndarray
    :param values: Sample values, ``nan`` where undefined.
    :type values: numpy.ndarray
    :param hop_seconds: Spacing of the output grid in seconds.
    :type hop_seconds: float
    :param duration: Length of the grid in seconds; defaults to the last time in ``times``.
    :type duration: float or None
    :return: The grid times and the resampled values.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if duration is None:
        duration = float(times[-1]) if len(times) else 0.0
    grid = np.arange(0.0, max(duration, hop_seconds), hop_seconds)

    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return grid, np.full_like(grid, np.nan)

    resampled = np.interp(grid, times[valid], values[valid], left=np.nan, right=np.nan)
    return grid, resampled


def fill_gaps(contour):
    """Fill ``nan`` gaps in a contour by linear interpolation.

    Leading and trailing gaps take the nearest valid value; a contour with no
    valid values becomes all zeros.

    :param contour: Pitch contour with ``nan`` for missing values.
    :type contour: numpy.ndarray
    :return: The contour with no ``nan`` values.
    :rtype: numpy.ndarray
    """
    contour = np.asarray(contour, dtype=np.float64)
    idx = np.arange(len(contour))
    voiced = ~np.isnan(contour)
    if voiced.sum() == 0:
        return np.zeros_like(contour)
    if voiced.sum() < 2:
        return np.full_like(contour, contour[voiced][0])
    return np.interp(idx, idx[voiced], contour[voiced])


def center(contour):
    """Subtract the median so that the contour is independent of the singer's key.

    :param contour: Pitch contour in cents.
    :type contour: numpy.ndarray
    :return: The median-centred contour.
    :rtype: numpy.ndarray
    """
    contour = np.asarray(contour, dtype=np.float64)
    finite = contour[np.isfinite(contour)]
    if finite.size == 0:
        return contour
    return contour - np.median(finite)


def contour_from_audio(y, sr, pitch_config):
    """Turn a waveform into a gap-free pitch contour in cents on a uniform time grid.

    :param y: Mono waveform.
    :type y: numpy.ndarray
    :param sr: Sampling rate of ``y`` in Hz.
    :type sr: int
    :param pitch_config: Settings with the keys ``fmin``, ``fmax``, ``frame_length``,
        ``hop_seconds`` and ``ref_hz``.
    :type pitch_config: dict
    :return: Median-centred pitch contour in cents, one value per ``hop_seconds``.
    :rtype: numpy.ndarray
    """
    times, f0 = extract_f0(
        y, sr, pitch_config["fmin"], pitch_config["fmax"], pitch_config["frame_length"],
        hop_length=round(pitch_config["hop_seconds"] * sr),
    )
    cents = hz_to_cents(f0, pitch_config["ref_hz"])
    duration = len(y) / sr
    _, cents = resample_uniform(times, cents, pitch_config["hop_seconds"], duration=duration)
    cents = fill_gaps(cents)
    return center(cents)


def salience_from_audio(y, sr, pitch_config):
    """Turn a waveform into a key-normalised pitch-salience image.

    Unlike :func:`contour_from_audio`, unvoiced gaps are kept (not interpolated)
    so they render as all-zero columns in the image.

    :param y: Mono waveform.
    :type y: numpy.ndarray
    :param sr: Sampling rate of ``y`` in Hz.
    :type sr: int
    :param pitch_config: Settings with the keys ``fmin``, ``fmax``, ``frame_length``,
        ``hop_seconds``, ``ref_hz``, ``bin_cents``, ``range_cents`` and ``sigma_cents``.
    :type pitch_config: dict
    :return: Salience image of shape ``(bins, frames)``.
    :rtype: numpy.ndarray
    """
    times, f0 = extract_f0(
        y, sr, pitch_config["fmin"], pitch_config["fmax"], pitch_config["frame_length"],
        hop_length=round(pitch_config["hop_seconds"] * sr),
    )
    cents = hz_to_cents(f0, pitch_config["ref_hz"])
    duration = len(y) / sr
    _, cents = resample_uniform(times, cents, pitch_config["hop_seconds"], duration=duration)
    cents = center(cents)
    return rasterise(cents, pitch_config["bin_cents"], pitch_config["range_cents"], pitch_config["sigma_cents"])


def best_match(query_image, tracks, pitch_config):
    """Find the track whose pitch-salience image contains the best subsequence match.

    Each track's stored cents contour is rasterised into a salience image and
    compared against the query image with subsequence DTW; the cost is normalised
    by the shorter length.

    :param query_image: Salience image of the query, as returned by :func:`salience_from_audio`.
    :type query_image: numpy.ndarray
    :param tracks: Track ids paired with their pitch contours in cents.
    :type tracks: collections.abc.Iterable[tuple[int, numpy.ndarray]]
    :param pitch_config: Settings with the keys ``bin_cents``, ``range_cents`` and ``sigma_cents``.
    :type pitch_config: dict
    :return: The id of the best-matching track (``None`` if there are no tracks), and the
        10 lowest-cost ``(track_id, cost)`` pairs, best first.
    :rtype: tuple[int or None, list[tuple[int, float]]]
    """
    tracks = list(tracks)
    best_id, best_cost = None, np.inf
    results = []

    with Progress() as progress:
        task = progress.add_task("Searching tracks...", total=len(tracks))
        for track_id, contour in tracks:
            if len(contour) == 0:
                progress.advance(task)
                continue
            track_image = rasterise(contour, pitch_config["bin_cents"], pitch_config["range_cents"],
                                     pitch_config["sigma_cents"])
            D = dtw(X=query_image, Y=track_image, metric="euclidean", subseq=True, backtrack=False)
            cost = D[-1, :].min() / min(query_image.shape[1], track_image.shape[1])
            results.append((track_id, cost))
            if cost < best_cost:
                best_id, best_cost = track_id, cost
            progress.advance(task)

    return best_id, sorted(results, key=lambda r: r[1])[:10]
