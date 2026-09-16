import librosa
import numpy as np

from strawberryfields.pyin import pyin


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def extract_f0(y, sr, fmin, fmax, frame_length, hop_length):
    f0, voiced_flag, _ = pyin(
        y.astype(np.float64), sr, fmin, fmax,
        frame_length=frame_length, hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~voiced_flag] = 0.0
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    return times, f0


def hz_to_cents(f0, f_ref):
    f0 = np.asarray(f0, dtype=np.float64)
    cents = np.full_like(f0, np.nan)
    voiced = f0 > 0
    cents[voiced] = 1200.0 * np.log2(f0[voiced] / f_ref)
    return cents


def resample_uniform(times, values, hop_seconds, duration=None):
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
    contour = np.asarray(contour, dtype=np.float64)
    idx = np.arange(len(contour))
    voiced = ~np.isnan(contour)
    if voiced.sum() == 0:
        return np.zeros_like(contour)
    if voiced.sum() < 2:
        return np.full_like(contour, contour[voiced][0])
    return np.interp(idx, idx[voiced], contour[voiced])


def center(contour):
    contour = np.asarray(contour, dtype=np.float64)
    finite = contour[np.isfinite(contour)]
    if finite.size == 0:
        return contour
    return contour - np.median(finite)


def contour_from_audio(y, sr, pitch_config):
    times, f0 = extract_f0(
        y, sr, pitch_config["fmin"], pitch_config["fmax"], pitch_config["frame_length"],
        hop_length=round(pitch_config["hop_seconds"] * sr),
    )
    cents = hz_to_cents(f0, pitch_config["ref_hz"])
    duration = len(y) / sr
    _, cents = resample_uniform(times, cents, pitch_config["hop_seconds"], duration=duration)
    cents = fill_gaps(cents)
    return center(cents)
