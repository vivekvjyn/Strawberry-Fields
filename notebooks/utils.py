from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm


_separator = None


def _get_separator(model):
    """Get a cached Demucs :class:`~demucs.api.Separator`, loading it if needed.

    :param model: Name of the pretrained Demucs model to use.
    :type model: str
    :return: A separator using ``model``, reused across calls with the same model.
    :rtype: demucs.api.Separator
    """
    global _separator
    if _separator is None or _separator.model_name != model:
        import torch
        from demucs.api import Separator

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _separator = Separator(model=model, device=device, progress=False)
        _separator.model_name = model
    return _separator


def separate_vocals(path, out_dir, model="htdemucs"):
    """Isolate the vocal stem of an audio file with Demucs, run in-process.

    :param path: Path to the mixed audio file.
    :type path: str or pathlib.Path
    :param out_dir: Directory the separated vocal stem is written under.
    :type out_dir: str or pathlib.Path
    :param model: Name of the pretrained Demucs model to use.
    :type model: str
    :return: Path to the separated vocal stem.
    :rtype: pathlib.Path
    """
    from demucs.api import save_audio

    path, out_dir = Path(path), Path(out_dir)
    vocals_path = out_dir / model / path.stem / "vocals.wav"
    if not vocals_path.exists():
        vocals_path.parent.mkdir(parents=True, exist_ok=True)
        separator = _get_separator(model)
        _, stems = separator.separate_audio_file(path)
        save_audio(stems["vocals"], vocals_path, samplerate=separator.samplerate)
    return vocals_path


def extract_f0(path, cache_dir, pitch_cfg):
    """Run pYIN on an audio file, caching the result.

    :param path: Path to the audio file.
    :type path: str or pathlib.Path
    :param cache_dir: Directory to read/write the cached ``.npz`` result in.
    :type cache_dir: str or pathlib.Path
    :param pitch_cfg: Settings with the keys ``sr``, ``fmin``, ``fmax``, ``frame_length``
        and ``hop_seconds``.
    :type pitch_cfg: dict
    :return: Frame times in seconds and the f0 value of each frame in Hz, ``nan``
        where unvoiced. Together these are a song's pitch track.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    cache = Path(cache_dir) / (Path(path).stem + ".npz")
    if cache.exists():
        data = np.load(cache)
        return data["times"], data["f0"]
    y, sr = librosa.load(path, sr=pitch_cfg["sr"], mono=True)
    hop_length = round(pitch_cfg["hop_seconds"] * sr)
    f0, voiced_flag, _ = librosa.pyin(y.astype(np.float64), fmin=pitch_cfg["fmin"], fmax=pitch_cfg["fmax"],
                                       sr=sr, frame_length=pitch_cfg["frame_length"], hop_length=hop_length)
    f0 = np.where(voiced_flag, f0, np.nan)
    times = np.arange(len(f0)) * hop_length / sr
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=times, f0=f0)
    return times, f0


def extract_f0_batch(paths, cache_dir, pitch_cfg, n_jobs=-1):
    """Extract and cache pYIN pitch tracks for many audio files in parallel.

    :param paths: Paths to the audio files.
    :type paths: collections.abc.Iterable[str or pathlib.Path]
    :param cache_dir: Directory to read/write cached results in.
    :type cache_dir: str or pathlib.Path
    :param pitch_cfg: Settings passed to :func:`extract_f0`.
    :type pitch_cfg: dict
    :param n_jobs: Number of worker processes; -1 uses all available cores.
    :type n_jobs: int
    :return: Mapping of file stem to its ``(times, f0)`` pitch track.
    :rtype: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    """
    paths = [Path(p) for p in paths]
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_f0)(p, cache_dir, pitch_cfg) for p in tqdm(paths, desc="pYIN"))
    return {p.stem: r for p, r in zip(paths, results)}


def hz_to_cents(f0, f_ref):
    """Convert frequencies in Hz to cents relative to a reference frequency.

    :param f0: Frequencies in Hz.
    :type f0: numpy.ndarray
    :param f_ref: Reference frequency in Hz that maps to 0 cents.
    :type f_ref: float
    :return: Pitch in cents, same shape as ``f0``.
    :rtype: numpy.ndarray
    """
    return 1200.0 * np.log2(np.asarray(f0, dtype=np.float64) / f_ref)


def downsample(contour, factor):
    """Median-pool a contour by a fixed number of frames, ignoring ``nan`` values.

    :param contour: Contour to downsample; its length must be a multiple of ``factor``
        for all of it to be used.
    :type contour: numpy.ndarray
    :param factor: Number of input frames per output frame.
    :type factor: int
    :return: The downsampled contour.
    :rtype: numpy.ndarray
    """
    n = len(contour) // factor * factor
    blocks = np.asarray(contour[:n], dtype=np.float64).reshape(-1, factor)
    with np.errstate(all="ignore"):
        out = np.nanmedian(blocks, axis=1)
    return out


def fill_gaps(contour):
    """Fill ``nan`` gaps in a contour by linear interpolation.

    Leading and trailing gaps take the nearest valid value; a contour with no valid
    values becomes all zeros.

    :param contour: Pitch contour with ``nan`` for missing values.
    :type contour: numpy.ndarray
    :return: The contour with no ``nan`` values.
    :rtype: numpy.ndarray
    """
    contour = np.asarray(contour, dtype=np.float64)
    ok = ~np.isnan(contour)
    if ok.sum() == 0:
        return np.zeros_like(contour)
    idx = np.arange(len(contour))
    return np.interp(idx, idx[ok], contour[ok])


def normalise_query(cents):
    """Remove the singer's key from a query contour.

    :param cents: Query pitch contour in cents.
    :type cents: numpy.ndarray
    :return: The contour with the median of its voiced frames subtracted.
    :rtype: numpy.ndarray
    """
    return cents - np.nanmedian(cents)


def normalise_reference(cents, window):
    """Remove the local key of a reference contour with a sliding median.

    :param cents: Reference pitch contour in cents.
    :type cents: numpy.ndarray
    :param window: Width of the sliding window, in frames.
    :type window: int
    :return: The contour with its local median subtracted.
    :rtype: numpy.ndarray
    """
    local = pd.Series(cents).rolling(window, center=True, min_periods=1).median().to_numpy()
    local = pd.Series(local).ffill().bfill().to_numpy()
    return cents - local


def to_contour(pitch_track, is_query, eval_cfg, pitch_cfg):
    """Turn a raw pitch track into a key-normalised cents contour at the evaluation hop.

    :param pitch_track: Pitch track as returned by :func:`extract_f0`.
    :type pitch_track: tuple[numpy.ndarray, numpy.ndarray]
    :param is_query: Whether ``pitch_track`` is a query (median-centred as a whole) or
        a reference (key removed with a sliding median).
    :type is_query: bool
    :param eval_cfg: Evaluation settings with the keys ``hop_seconds``, ``norm_window_s``
        and ``f_ref``.
    :type eval_cfg: dict
    :param pitch_cfg: Pitch-extraction settings with the key ``hop_seconds``.
    :type pitch_cfg: dict
    :return: Key-normalised pitch contour in cents, ``nan`` where unvoiced.
    :rtype: numpy.ndarray
    """
    _, f0 = pitch_track
    factor = max(1, round(eval_cfg["hop_seconds"] / pitch_cfg["hop_seconds"]))
    cents = downsample(hz_to_cents(f0, eval_cfg["f_ref"]), factor)
    if is_query:
        return normalise_query(cents)
    window = max(3, int(eval_cfg["norm_window_s"] / eval_cfg["hop_seconds"]) | 1)
    return normalise_reference(cents, window)


def to_salience_image(contour, eval_cfg, sigma_cents=0.0, unit_sum=True):
    """Render a pitch contour as a 2D salience image, frequency on y and time on x.

    A voiced frame is 1 at its pitch bin and 0 elsewhere; unvoiced frames are
    all-zero columns. ``sigma_cents`` blurs each column along the frequency axis
    with a Gaussian, so nearby pitches partially overlap.

    :param contour: Pitch contour in cents, ``nan`` where unvoiced.
    :type contour: numpy.ndarray
    :param eval_cfg: Settings with the keys ``bin_cents`` and ``range_cents``.
    :type eval_cfg: dict
    :param sigma_cents: Width of the Gaussian blur in cents; 0 disables blurring.
    :type sigma_cents: float
    :param unit_sum: If blurring, rescale each voiced column to sum to 1.
    :type unit_sum: bool
    :return: Image of shape ``(bins, len(contour))``.
    :rtype: numpy.ndarray
    """
    n_bins = int(2 * eval_cfg["range_cents"] / eval_cfg["bin_cents"]) + 1
    image = np.zeros((n_bins, len(contour)))
    voiced = ~np.isnan(contour)
    bins = np.rint((contour[voiced] + eval_cfg["range_cents"]) / eval_cfg["bin_cents"]).astype(int)
    keep = (bins >= 0) & (bins < n_bins)
    image[bins[keep], np.flatnonzero(voiced)[keep]] = 1.0
    if sigma_cents > 0:
        image = gaussian_filter1d(image, sigma_cents / eval_cfg["bin_cents"], axis=0, mode="constant")
        if unit_sum:
            sums = image.sum(axis=0, keepdims=True)
            image = np.divide(image, sums, out=np.zeros_like(image), where=sums > 0)
    return image


def subsequence_cost(query, reference, metric="euclidean"):
    """Compute the subsequence-DTW cost of matching a query inside a reference.

    :param query: Query representation, shape ``(features, frames)``.
    :type query: numpy.ndarray
    :param reference: Reference representation, shape ``(features, frames)``.
    :type reference: numpy.ndarray
    :param metric: Distance metric passed to :func:`librosa.sequence.dtw`.
    :type metric: str
    :return: DTW cost normalised by the shorter of the two lengths.
    :rtype: float
    """
    D = librosa.sequence.dtw(X=query, Y=reference, metric=metric, subseq=True, backtrack=False)
    return D[-1, :].min() / min(query.shape[1], reference.shape[1])


def subsequence_match(query, reference, metric="euclidean"):
    """Find where a query best matches within a reference via subsequence DTW.

    Unlike :func:`subsequence_cost`, this backtracks the optimal warping path to
    locate the matched region, so it's meant for inspecting a single query/reference
    pair rather than for the ranking loop.

    :param query: Query representation, shape ``(features, frames)``.
    :type query: numpy.ndarray
    :param reference: Reference representation, shape ``(features, frames)``.
    :type reference: numpy.ndarray
    :param metric: Distance metric passed to :func:`librosa.sequence.dtw`.
    :type metric: str
    :return: The first and last reference frame covered by the match.
    :rtype: tuple[int, int]
    """
    _, wp = librosa.sequence.dtw(X=query, Y=reference, metric=metric, subseq=True, backtrack=True)
    ref_idx = wp[:, 1]
    return int(ref_idx.min()), int(ref_idx.max())


def rank_queries(query_reprs, ref_reprs, true_ids, metric="euclidean", desc="matching"):
    """Match every query against every reference and rank the true song's cost.

    :param query_reprs: Query representations, each shape ``(features, frames)``.
    :type query_reprs: collections.abc.Sequence[numpy.ndarray]
    :param ref_reprs: Reference representations keyed by song id.
    :type ref_reprs: dict[int, numpy.ndarray]
    :param true_ids: The correct song id for each entry in ``query_reprs``.
    :type true_ids: collections.abc.Sequence[int]
    :param metric: Distance metric passed to :func:`subsequence_cost`.
    :type metric: str
    :param desc: Label shown on the progress bar.
    :type desc: str
    :return: The rank of the true song for each query (1 means it scored best), the
        full cost matrix, and the song ids in the order used for its columns.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, list[int]]
    """
    song_ids = list(ref_reprs)
    costs = np.full((len(query_reprs), len(song_ids)), np.inf)
    for i, q in enumerate(tqdm(query_reprs, desc=desc)):
        for j, sid in enumerate(song_ids):
            costs[i, j] = subsequence_cost(q, ref_reprs[sid], metric)
    true_cols = np.array([song_ids.index(t) for t in true_ids])
    true_costs = costs[np.arange(len(costs)), true_cols]
    ranks = 1 + (costs < true_costs[:, None]).sum(axis=1)
    return ranks, costs, song_ids


def summarise(ranks, ks=(1, 3, 5, 10)):
    """Summarise a set of ranks as top-k accuracy and reciprocal-rank statistics.

    :param ranks: Rank of the true song for each query (1 is best).
    :type ranks: numpy.ndarray
    :param ks: The k values to report top-k accuracy for.
    :type ks: collections.abc.Sequence[int]
    :return: Metric name to value, including ``top-k`` for each ``k``, ``MRR``,
        ``median rank`` and ``mean rank``.
    :rtype: dict[str, float]
    """
    ranks = np.asarray(ranks)
    out = {f"top-{k}": float(np.mean(ranks <= k)) for k in ks}
    out["MRR"] = float(np.mean(1.0 / ranks))
    out["median rank"] = float(np.median(ranks))
    out["mean rank"] = float(np.mean(ranks))
    return out


def cmc_curve(ranks, max_k):
    """Compute the Cumulative Match Characteristic curve up to ``max_k``.

    The CMC curve is the standard evaluation curve for closed-set ranked
    identification: the fraction of queries whose true match is within the
    top ``k`` candidates, as a function of ``k``.

    :param ranks: Rank of the true song for each query (1 is best).
    :type ranks: numpy.ndarray
    :param max_k: Largest k to compute accuracy for.
    :type max_k: int
    :return: The k values and the corresponding accuracy at each one.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    ranks = np.asarray(ranks)
    ks = np.arange(1, max_k + 1)
    return ks, np.array([np.mean(ranks <= k) for k in ks])
