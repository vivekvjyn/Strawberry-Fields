import csv
from pathlib import Path

import librosa
import numpy as np
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
    out = np.full(len(blocks), np.nan)
    voiced = ~np.isnan(blocks).all(axis=1)
    out[voiced] = np.nanmedian(blocks[voiced], axis=1)
    return out


def downsampled_cents(pitch_track, eval_cfg, pitch_cfg):
    """Turn a raw pitch track into an un-normalised cents contour at the evaluation hop.

    :param pitch_track: Pitch track as returned by :func:`extract_f0`.
    :type pitch_track: tuple[numpy.ndarray, numpy.ndarray]
    :param eval_cfg: Evaluation settings with the keys ``hop_seconds`` and ``f_ref``.
    :type eval_cfg: dict
    :param pitch_cfg: Pitch-extraction settings with the key ``hop_seconds``.
    :type pitch_cfg: dict
    :return: Pitch contour in cents relative to ``f_ref``, ``nan`` where unvoiced.
    :rtype: numpy.ndarray
    """
    _, f0 = pitch_track
    factor = max(1, round(eval_cfg["hop_seconds"] / pitch_cfg["hop_seconds"]))
    return downsample(hz_to_cents(f0, eval_cfg["f_ref"]), factor)


def to_pitch_class_profile(contour, n_classes, sigma_cents=0.0):
    """Fold a cents contour into an octave-invariant pitch-class salience profile.

    Every pitch is reduced mod 1200 cents (its position within an octave,
    regardless of which octave), then rendered the same way as a salience image: a
    voiced frame is 1 at its pitch-class bin and 0 elsewhere, blurred along the
    pitch-class axis with a Gaussian that wraps around the octave (0 and 1200 cents
    are the same point). Because pitch class is transposition- and octave-invariant
    by construction, matching against 24 shifts of this profile (see
    :func:`transposed_subsequence_cost`) covers every possible tonic difference
    with a single un-windowed subsequence DTW per shift -- no re-centring or tempo
    resampling needed, since ordinary DTW already warps through tempo differences.

    :param contour: Pitch contour in cents, ``nan`` where unvoiced.
    :type contour: numpy.ndarray
    :param n_classes: Number of pitch classes per octave (24 for quarter-tone/sruti
        resolution, i.e. 50 cents per class).
    :type n_classes: int
    :param sigma_cents: Width of the Gaussian blur in cents; 0 disables blurring.
    :type sigma_cents: float
    :return: Profile of shape ``(n_classes, len(contour))``.
    :rtype: numpy.ndarray
    """
    bin_cents = 1200.0 / n_classes
    contour = np.asarray(contour, dtype=np.float64)
    image = np.zeros((n_classes, len(contour)))
    voiced = ~np.isnan(contour)
    bins = np.rint(np.mod(contour[voiced], 1200.0) / bin_cents).astype(int) % n_classes
    image[bins, np.flatnonzero(voiced)] = 1.0
    if sigma_cents > 0:
        image = gaussian_filter1d(image, sigma_cents / bin_cents, axis=0, mode="wrap")
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


def transposed_subsequence_cost(query_profile, reference_profile, n_classes, metric="euclidean", shift_step=1):
    """Match a query pitch-class profile inside a reference, trying several transpositions.

    Circularly shifting a pitch-class profile by one class is exactly a transposition
    by ``1200 / n_classes`` cents, so trying ``n_classes`` shifts covers every
    possible tonic difference at the profile's own resolution. ``shift_step`` searches
    coarser: e.g. with 24 classes (50 cents each), ``shift_step=2`` tries every other
    shift -- semitone (100-cent) transpositions -- for half the cost, relying on the
    profile's Gaussian blur to absorb the skipped in-between shifts. Each shift tried
    is one ordinary, un-windowed subsequence DTW over the whole reference -- the
    profile's octave/transposition invariance means no per-window re-centring is needed.

    :param query_profile: Query profile, as returned by :func:`to_pitch_class_profile`.
    :type query_profile: numpy.ndarray
    :param reference_profile: Reference profile, as returned by
        :func:`to_pitch_class_profile`.
    :type reference_profile: numpy.ndarray
    :param n_classes: Number of pitch classes per octave; must match how both
        profiles were built.
    :type n_classes: int
    :param metric: Distance metric passed to :func:`subsequence_cost`.
    :type metric: str
    :param shift_step: Try every ``shift_step``-th class shift instead of all of them.
    :type shift_step: int
    :return: The lowest subsequence-DTW cost over the transpositions tried.
    :rtype: float
    """
    best = np.inf
    for shift in range(0, n_classes, shift_step):
        cost = subsequence_cost(query_profile, np.roll(reference_profile, shift, axis=0), metric)
        if cost < best:
            best = cost
    return best


def transposed_subsequence_match(query_profile, reference_profile, n_classes, metric="euclidean", shift_step=1):
    """Locate where a query pitch-class profile best matches within a reference.

    Same search as :func:`transposed_subsequence_cost`, but the best transposition's
    warping path is also backtracked to find the matched frames, for inspecting one
    query/reference pair.

    :param query_profile: Query profile, as returned by :func:`to_pitch_class_profile`.
    :type query_profile: numpy.ndarray
    :param reference_profile: Reference profile, as returned by
        :func:`to_pitch_class_profile`.
    :type reference_profile: numpy.ndarray
    :param n_classes: Number of pitch classes per octave; must match how both
        profiles were built.
    :type n_classes: int
    :param metric: Distance metric passed to :func:`subsequence_cost`.
    :type metric: str
    :param shift_step: Try every ``shift_step``-th class shift instead of all of them.
    :type shift_step: int
    :return: The lowest cost and the first and last reference frame of the match.
    :rtype: tuple[float, int, int]
    """
    best_cost, best_shifted = np.inf, None
    for shift in range(0, n_classes, shift_step):
        shifted = np.roll(reference_profile, shift, axis=0)
        cost = subsequence_cost(query_profile, shifted, metric)
        if cost < best_cost:
            best_cost, best_shifted = cost, shifted
    lo, hi = subsequence_match(query_profile, best_shifted, metric)
    return best_cost, lo, hi


def rank_queries(query_reprs, ref_reprs, true_ids, metric="euclidean", desc="matching", cost=None,
                 query_ids=None, cache_path=None):
    """Match every query against every reference and rank the true song's cost.

    :param query_reprs: Query representations, each shape ``(features, frames)``.
    :type query_reprs: collections.abc.Sequence[numpy.ndarray]
    :param ref_reprs: Reference representations keyed by song id (or whatever ``cost``
        expects as its second argument).
    :type ref_reprs: dict[int, numpy.ndarray]
    :param true_ids: The correct song id for each entry in ``query_reprs``.
    :type true_ids: collections.abc.Sequence[int]
    :param metric: Distance metric passed to :func:`subsequence_cost`; ignored when
        ``cost`` is given.
    :type metric: str
    :param desc: Label shown on the progress bar.
    :type desc: str
    :param cost: Scores one query against one reference; defaults to
        :func:`subsequence_cost` with ``metric``.
    :type cost: collections.abc.Callable[[numpy.ndarray, numpy.ndarray], float] or None
    :param query_ids: A stable id per entry of ``query_reprs``, used as the cache key
        alongside each song id; defaults to its position. Needed for resuming to work
        if ``query_reprs`` might be reordered or subsampled between runs.
    :type query_ids: collections.abc.Sequence or None
    :param cache_path: CSV file of ``query_id, song_id, cost`` rows. Existing rows are
        loaded and skipped; each cost computed this run is appended immediately (not
        batched), so interrupting and rerunning with the same path resumes from
        whatever was already computed instead of starting over.
    :type cache_path: str or pathlib.Path or None
    :return: The rank of the true song for each query (1 means it scored best), the
        full cost matrix, and the song ids in the order used for its columns.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, list[int]]
    """
    if cost is None:
        cost = lambda q, r: subsequence_cost(q, r, metric)
    song_ids = list(ref_reprs)
    if query_ids is None:
        query_ids = list(range(len(query_reprs)))
    costs = np.full((len(query_reprs), len(song_ids)), np.inf)

    cached, cache_file, writer = {}, None, None
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, newline="") as f:
                for row in csv.DictReader(f):
                    cached[(row["query_id"], row["song_id"])] = float(row["cost"])
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_file = open(cache_path, "a", newline="")
        writer = csv.writer(cache_file)
        if cache_file.tell() == 0:
            writer.writerow(["query_id", "song_id", "cost"])

    try:
        for i in tqdm(range(len(query_reprs)), desc=desc):
            qid = query_ids[i]
            for j, sid in enumerate(song_ids):
                key = (str(qid), str(sid))
                if key in cached:
                    costs[i, j] = cached[key]
                    continue
                c = cost(query_reprs[i], ref_reprs[sid])
                costs[i, j] = c
                if writer is not None:
                    writer.writerow([qid, sid, c])
                    cache_file.flush()
    finally:
        if cache_file is not None:
            cache_file.close()

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
