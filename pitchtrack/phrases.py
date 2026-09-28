"""Find repeated melodic phrases in a song, then discard the repeats.

The method is the classic self-similarity recipe for repeated-section detection, with
melodic features instead of timbral ones:

1. Track the predominant pitch of the solo line.
2. Leave out the frames that cannot start or end a phrase — silence, and sustained
   notes, which are the natural phrase boundaries.
3. Describe each remaining moment by the melodic content around it — how the window's
   voiced frames are shared between the pitch classes, and how much of it is voiced at
   all — and compare every moment with every other one to get a *self-similarity
   matrix*.
4. Sharpen the matrix so that two similar regions show up as a bright streak running
   diagonally, then read each streak as a *segment*: a pair of regions that sound alike.
5. Collect the segments that point at the same regions into a *group* — one phrase,
   heard several times.
6. Keep one occurrence of each group and cut the rest out of the pitch track.

Two design points are worth stating up front, because they are what make the method
survive a 45-minute performance rather than a 3-minute excerpt:

* **The self-similarity matrix is computed one window at a time.** It is dense and
  quadratic in the number of frames that survive the mask, so a whole concert does not
  fit. Groups found in overlapping windows are merged afterwards, which means a phrase
  straddling a window edge is still recognised as one phrase.
* **Pitch is extracted in chunks.** Tracking pitch over tens of minutes of audio at one
  go needs several times the memory of the audio itself; doing it a few minutes at a
  time with a small overlap costs nothing and keeps the footprint flat.

Every stage can be cached, so a re-run only redoes what changed.

The method and its parameters are those of ``notebooks/similar_phrases.ipynb`` in the
Strawberry Fields repository; this module is the same algorithm with the audio output
and the notebook's diagnostics taken out, leaving the pitch track and the mask of the
frames to drop from it.
"""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, generate_binary_structure, label
from scipy.signal import convolve2d, fftconvolve
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "PitchConfig",
    "PhraseConfig",
    "PhraseGroup",
    "extract_pitch",
    "voiced_fraction",
    "frame_step",
    "exclusion_mask",
    "pitch_features",
    "self_similarity",
    "emphasise_diagonals",
    "extract_segments",
    "find_repeated_phrases",
    "discard_duplicate_phrases",
    "deduplicate_pitch",
    "remove_repeated_phrases",
]


@dataclass(frozen=True)
class PitchConfig:
    """Predominant-pitch extraction settings.

    :param sr: Analysis sample rate in Hz.
    :param fmin: Lowest frequency the tracker will report in Hz.
    :param fmax: Highest frequency the tracker will report in Hz.
    :param frame_length: Analysis window in samples.
    :param hop_seconds: Time between pitch frames in seconds.
    :param chunk_seconds: Audio analysed per pYIN call. Bounded so that a long
        performance does not need the whole decoded signal in memory at once; a
        chunk's first and last moments are discarded so the chunks do not disagree
        about the frames straddling a join.
    """

    sr: int = 22050
    fmin: float = 80.0
    fmax: float = 1000.0
    frame_length: int = 2048
    hop_seconds: float = 0.025
    chunk_seconds: float = 120.0

    @property
    def cache_key(self):
        """Short string identifying the settings that change a track's output."""
        return f"pyin{self.sr}_{self.fmin:g}_{self.fmax:g}_h{self.hop_seconds:g}"


@dataclass(frozen=True)
class PhraseConfig:
    """Parameters of the repeated-phrase search and of the discard step.

    :param window_seconds: Length of one self-similarity window in seconds. It bounds
        how far apart two statements of a phrase may be and still be compared, so a
        window much shorter than the song will miss the long-range recurrences that
        give a raga its shape. The cost is quadratic in the window length, but at these
        settings it is the least of the memory used: on a 4-minute song a 240 s window
        holds about 1 500 surviving frames, a 26 MB matrix.
    :param window_hop_seconds: Stride between consecutive windows in seconds. Half the
        window, so every phrase lies wholly inside at least one window.
    :param bin_thresh: Binarisation threshold of the self-similarity matrix. The single
        most important parameter; low values keep the speckle, high values break real
        segments up.
    :param segment_thresh_fraction: Fraction of ``bin_thresh`` used to grow a segment
        into the surrounding matrix once its path is known.
    :param min_pattern_length_seconds: Shortest occurrence kept, in seconds.
    :param break_seconds: How long a silence or a held note must last before it counts
        as a phrase boundary and splits a segment, as opposed to merely trimming it.
    :param min_in_group: Fewest occurrences that make a group a repeat.
    :param length_tolerance: How much two statements of the same phrase may differ in
        duration, as a fraction of the longer. Repeated statements of one phrase are
        close in length, and this is what stops a two-second fragment and a ten-second
        passage from being collected into one group.
    :param merge_overlap: Two occurrences count as the same moment when they overlap by
        at least this fraction *and* are within ``length_tolerance`` in length.
    :param group_agreement: Two groups are the same phrase when this fraction of their
        occurrences pair up.
    :param max_gap_seconds: Longest silence bridged over a dropout inside a phrase.
    :param stability_min_seconds: Minimum run of stable pitch that counts as a phrase
        boundary.
    :param stability_var_cents: Deviation from the window mean, in cents, above which a
        window counts as *not* stable.
    :param stability_hop_seconds: Stride of the stability test in seconds.
    :param n_classes: Pitch classes per octave in the melodic features.
    :param feature_window_seconds: Length of the window summarised into one feature.
    """

    window_seconds: float = 300.0
    window_hop_seconds: float = 150.0
    bin_thresh: float = 0.1
    segment_thresh_fraction: float = 0.5
    min_pattern_length_seconds: float = 2.0
    break_seconds: float = 0.4
    min_in_group: int = 2
    length_tolerance: float = 0.45
    merge_overlap: float = 0.55
    group_agreement: float = 0.5
    max_gap_seconds: float = 0.35
    stability_min_seconds: float = 1.0
    stability_var_cents: float = 60.0
    stability_hop_seconds: float = 0.2
    n_classes: int = 24
    feature_window_seconds: float = 0.5

    def __post_init__(self):
        if self.window_hop_seconds > self.window_seconds:
            raise ValueError("window_hop_seconds must not exceed window_seconds")
        if self.min_in_group < 2:
            raise ValueError("min_in_group must be at least 2 for a group to be a repeat")


@dataclass(frozen=True)
class PhraseGroup:
    """Occurrences of one melodic phrase, as absolute ``(start, end)`` second spans.

    :param index: Identifier of the group within its song.
    :param spans: ``(start_s, end_s)`` of each occurrence, ordered in time.
    """

    index: int
    spans: tuple

    def __len__(self):
        return len(self.spans)

    @property
    def duration_s(self):
        """Mean length of the group's occurrences, in seconds."""
        return float(np.mean([end - start for start, end in self.spans])) if self.spans else 0.0


def extract_pitch(path, name, cache_dir, pitch_cfg=None):
    """Extract a predominant-pitch track from a solo vocal stem, caching the result.

    Frames sit on a uniform grid at ``pitch_cfg.hop_seconds`` and unvoiced frames are
    ``0.0``. The audio is tracked in chunks of ``pitch_cfg.chunk_seconds`` with a small
    overlap, and each chunk's outermost frames are dropped, so the result is the same
    pitch track a single pass would give while the memory stays proportional to one
    chunk rather than to the whole song.

    :param path: Audio file to analyse.
    :type path: str or pathlib.Path
    :param name: Cache file stem, normally the song id.
    :type name: str
    :param cache_dir: Directory the ``.npz`` result is read from and written to.
    :type cache_dir: str or pathlib.Path
    :param pitch_cfg: Tracker settings; the :class:`PitchConfig` default if ``None``.
    :type pitch_cfg: PitchConfig or None
    :return: Frame times in seconds and the frequency of each frame in Hz, ``0.0``
        where unvoiced.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    path, pitch_cfg = Path(path), pitch_cfg or PitchConfig()
    cache = Path(cache_dir) / f"{name}-{pitch_cfg.cache_key}.npz"
    if cache.exists():
        data = np.load(cache)
        return data["times"], data["f0"]

    y, sr = librosa.load(path, sr=pitch_cfg.sr, mono=True)
    hop_length = round(pitch_cfg.hop_seconds * sr)
    chunk = max(hop_length * 4, round(pitch_cfg.chunk_seconds * sr))


    trim = max(1, pitch_cfg.frame_length // hop_length + 1)

    tracks = []
    for start in range(0, max(1, len(y)), chunk):
        stop = min(start + chunk, len(y))
        f0, voiced, _ = librosa.pyin(
            y[start:stop].astype(np.float64), fmin=pitch_cfg.fmin, fmax=pitch_cfg.fmax,
            sr=sr, frame_length=pitch_cfg.frame_length, hop_length=hop_length)
        f0 = np.where(voiced, f0, 0.0)
        if tracks:
            f0 = f0[trim:]
        if stop < len(y):
            f0 = f0[:-trim]
        tracks.append(f0)
        if stop >= len(y):
            break

    f0 = np.nan_to_num(np.concatenate(tracks) if tracks else np.zeros(1),
                       nan=0.0, posinf=0.0, neginf=0.0)
    times = np.arange(len(f0)) * hop_length / sr
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, times=times, f0=f0)
    return times, f0


def voiced_fraction(f0):
    """Fraction of pitch frames that carry a pitch.

    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :return: Value in ``[0, 1]``.
    :rtype: float
    """
    return float(np.mean(np.asarray(f0) > 0))


def frame_step(times):
    """Seconds between consecutive pitch frames.

    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :return: The spacing, or the default 25 ms if there is not a second frame.
    :rtype: float
    """
    times = np.asarray(times, dtype=np.float64)
    return float(times[1] - times[0]) if len(times) > 1 else 0.025


def _interpolate_gaps(f0, max_gap_frames):
    """Bridge unvoiced runs no longer than ``max_gap_frames`` by linear interpolation.

    Without this a single glottal dropout would split a phrase in two. The mask used
    downstream is still built from the raw track, so a bridged region is never mistaken
    for fresh material.

    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param max_gap_frames: Longest unvoiced run to bridge, in frames.
    :type max_gap_frames: float
    :return: The bridged track, of the same length.
    :rtype: numpy.ndarray
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.flatnonzero(f0 > 0)
    if len(voiced) < 2:
        return f0.copy()
    bridged = f0.copy()
    bridged[:voiced[0]] = f0[voiced[0]]
    bridged[voiced[-1] + 1:] = f0[voiced[-1]]
    for lo, hi in zip(voiced[:-1], voiced[1:]):
        if 1 < hi - lo <= max_gap_frames:
            bridged[lo:hi] = np.interp(np.arange(lo, hi), [lo, hi], [f0[lo], f0[hi]])
    return bridged


def exclusion_mask(f0, times, cfg=None):
    """Mark the pitch frames that cannot take part in a phrase.

    Two kinds of frame are excluded:

    * **Silence**, which carries no melody and would otherwise fill the matrix with a
      block of trivial self-similarity.
    * **Sustained notes** — a note held steady enough to be one svara. These are the
      natural place for a phrase to begin or end, so excluding them both shrinks the
      matrix and gives the segment extractor its break points.

    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param cfg: Search settings; the :class:`PhraseConfig` default if ``None``.
    :type cfg: PhraseConfig or None
    :return: ``(silence, sustained, exclusion)`` boolean arrays. ``exclusion`` is the
        union, and ``True`` means *skip*.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    cfg = cfg or PhraseConfig()
    f0, times = np.asarray(f0, dtype=np.float64), np.asarray(times, dtype=np.float64)
    step = frame_step(times)
    silence = f0 <= 0
    if silence.all():
        return silence, np.zeros(len(f0), dtype=bool), silence

    with np.errstate(invalid="ignore", divide="ignore"):
        cents = 1200.0 * np.log2(np.where(f0 > 0, f0, np.nan) / 55.0)


    hop = max(1, round(cfg.stability_hop_seconds / step))
    n = len(cents) // hop * hop
    blocks = cents[:n].reshape(-1, hop)
    voiced = ~np.all(np.isnan(blocks), axis=1)
    deviation = np.full(len(blocks), np.inf)
    if voiced.any():
        with np.errstate(invalid="ignore"):
            deviation[voiced] = np.nanmean(np.abs(blocks[voiced] - np.nanmean(blocks[voiced],
                                                                              axis=1, keepdims=True)),
                                           axis=1)


    steady = deviation <= cfg.stability_var_cents
    need = max(1, round(cfg.stability_min_seconds / cfg.stability_hop_seconds))
    sustained = np.zeros(len(f0), dtype=bool)
    run = 0
    for i, is_steady in enumerate(steady):
        run = run + 1 if is_steady else 0
        if run >= need:
            sustained[i * hop:(i + 1) * hop] = True
    sustained &= ~silence
    return silence, sustained, np.logical_or(silence, sustained)


def pitch_features(times, f0, cfg=None):
    """Describe each moment of the performance by the pitch content around it.

    One feature frame covers a short window and holds the share of the window's voiced
    samples falling into each of ``n_classes`` pitch classes, plus the voiced fraction
    itself. The two are complementary: pitch class says *which* notes, the voiced
    fraction says *how sustained*, so a held svara and a fast oscillation over the same
    notes are not confused. Each frame is L2-normalised, so comparing frames with a
    plain cosine distance ignores how much singing there was and looks only at the
    shape of the line.

    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param cfg: Search settings; the :class:`PhraseConfig` default if ``None``.
    :type cfg: PhraseConfig or None
    :return: ``(features, hop_seconds)`` — a ``(frames, n_classes + 1)`` array and its
        spacing in seconds.
    :rtype: tuple[numpy.ndarray, float]
    """
    cfg = cfg or PhraseConfig()
    n_classes = cfg.n_classes
    f0 = np.asarray(f0, dtype=np.float64)
    if len(f0) < 2:
        return np.zeros((0, n_classes + 1), dtype=np.float32), cfg.feature_window_seconds / 2

    step = frame_step(times)
    stride = max(1, round(cfg.feature_window_seconds / step / 2))
    width = max(stride, round(cfg.feature_window_seconds / step))
    edges = np.linspace(0.0, 1200.0, n_classes + 1)

    rows = []
    for start in range(0, max(1, len(f0) - width + 1), stride):
        block = f0[start:start + width]
        voiced = block[block > 0]
        row = np.zeros(n_classes + 1, dtype=np.float64)
        row[-1] = len(voiced) / max(1, len(block))
        if len(voiced):
            cents = np.mod(1200.0 * np.log2(voiced / 55.0), 1200.0)
            bins = np.clip(np.digitize(cents, edges) - 1, 0, n_classes - 1)
            row[:-1] = np.bincount(bins, minlength=n_classes)[:n_classes]
            row[:-1] /= row[:-1].sum()
        rows.append(row)

    features = np.asarray(rows, dtype=np.float64)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    return features.astype(np.float32), stride * step


def self_similarity(features, keep, *, feature_times=None, hop_seconds=None,
                    band_seconds=2.0, smooth_seconds=2.5):
    """Compare every moment of the performance with every other moment.

    The cosine distance between two feature frames is turned into a *similarity* by
    inverting it, so alike regions score high. Three things then happen to that matrix:

    * a band around the main diagonal is zeroed, because a frame is trivially similar to
      itself and that band carries no information about repeats;
    * the matrix is smoothed along both axes with a short three-tap kernel, which
      fills in the gaps a single frame comparison leaves behind;
    * the frames ``keep`` marks false are dropped, and the times of the ones that
      remain are returned alongside the matrix so every downstream step can talk in
      seconds rather than in frame indices.

    Both widths are given in seconds rather than in frames, so the matrix means the same
    thing however finely the features happen to be sampled.

    :param features: ``(frames, coefficients)`` feature array.
    :type features: numpy.ndarray
    :param keep: Boolean per feature frame, ``True`` where it takes part. The caller
        builds this from the exclusion mask, because only the caller knows which slice
        of the mask goes with which slice of the features.
    :type keep: numpy.ndarray
    :param feature_times: Centre time in seconds of each feature frame.
    :type feature_times: numpy.ndarray or None
    :param hop_seconds: Seconds per feature frame, used when ``feature_times`` is
        ``None``.
    :type hop_seconds: float or None
    :param band_seconds: Half-width of the band zeroed around the main diagonal. It has
        to cover the span a single feature frame summarises, or the trivial self-match
        comes back as a bright band down the middle.
    :type band_seconds: float
    :param smooth_seconds: Width of the smoothing kernel along each axis.
    :type smooth_seconds: float
    :return: ``(matrix, times)`` — the normalised matrix over the kept frames, and the
        time in seconds of each of them.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    features = np.asarray(features, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    if len(keep) != len(features):
        raise ValueError(f"keep has {len(keep)} entries for {len(features)} features")
    if feature_times is None:
        if hop_seconds is None:
            raise ValueError("pass feature_times or hop_seconds")
        feature_times = (np.arange(len(features)) + 1) * hop_seconds
    feature_times = np.asarray(feature_times, dtype=np.float64)
    if keep.sum() < 4:
        return np.zeros((0, 0)), np.zeros(0)

    spacing = (float(np.median(np.diff(feature_times))) if len(feature_times) > 1
               else float(hop_seconds or 1.0))
    kept = np.flatnonzero(keep)
    matrix = squareform(pdist(features[kept], metric="cosine"))
    matrix = 1.0 / (matrix + 1e-6)

    n = matrix.shape[0]
    band = max(1, int(round(band_seconds / spacing)))
    rows = np.arange(n)
    matrix[np.abs(rows[:, None] - rows[None, :]) <= band] = 0.0


    width = max(3, int(round(smooth_seconds / spacing)))
    taper = np.eye(width) + np.eye(width, k=1) + np.eye(width, k=-1)
    weight = fftconvolve(np.ones_like(matrix), taper, mode="same")
    matrix = fftconvolve(matrix, taper, mode="same") / np.maximum(weight, 1e-9)
    return matrix, feature_times[kept]


def emphasise_diagonals(matrix, bin_thresh, *, spacing=None, band_seconds=7.5,
                        close_size=10, open_diagonal=3):
    """Sharpen the self-similarity matrix so repeated regions stand out as streaks.

    A repeat is a *diagonal streak* and not a blob, because two statements of the same
    phrase are never aligned frame for frame — the tempo differs, so the matched frames
    drift apart as the repeat goes on. The pipeline is:

    * a Scharr convolution, which responds to the edges of the similarity structure;
    * rescaling to ``[0, 1]`` and binarising at ``bin_thresh``;
    * zeroing a band along the main diagonal;
    * making the matrix symmetric, so a streak and its mirror count once;
    * a morphological *closing* with a square kernel, which fills the interior of each
      streak between its two edges;
    * a morphological *opening* with a diagonal structure, which erases the speckle and
      the round blobs that are not streaks at all.

    :param matrix: Self-similarity matrix, as returned by :func:`self_similarity`.
    :type matrix: numpy.ndarray
    :param bin_thresh: Binarisation threshold, the parameter that matters most.
    :type bin_thresh: float
    :param spacing: Seconds per matrix element; ``None`` keeps the band at 30 elements.
    :type spacing: float or None
    :param band_seconds: Half-width of the band zeroed along the main diagonal.
    :type band_seconds: float
    :param close_size: Side of the square closing kernel.
    :type close_size: int
    :param open_diagonal: Side of the diagonal opening structure.
    :type open_diagonal: int
    :return: ``(processed, binarised)`` — the cleaned matrix the segments are read from,
        and the raw binarised one, which is what carries the streak geometry.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    if matrix.size == 0:
        return matrix, matrix

    scharr = np.array([[-3 - 3j, 0 - 10j, 3 - 3j],
                       [-10 + 0j, 0 + 0j, 10 + 0j],
                       [-3 + 3j, 0 + 10j, 3 + 3j]])
    edges = np.abs(convolve2d(matrix, scharr, boundary="symm", mode="same"))
    edges = edges - edges.min()
    edges /= edges.max() + 1e-8

    binary = (edges >= bin_thresh).astype(np.uint8)
    n = binary.shape[0]
    band = 30 if spacing is None else max(1, int(round(band_seconds / spacing)))
    rows = np.arange(n)
    for offset in range(-band, band + 1):
        cols = rows + offset
        valid = (cols >= 0) & (cols < n)
        keep_x = rows[valid][abs(offset):n - abs(offset)] if offset else rows
        keep_y = cols[valid][abs(offset):n - abs(offset)] if offset else cols
        binary[keep_x, keep_y] = 0

    symmetric = np.maximum(binary, binary.T)
    footprint = np.ones((close_size, close_size), dtype=bool)
    filled = binary_closing(symmetric, structure=footprint)

    streak = np.zeros((open_diagonal, open_diagonal), dtype=bool)
    np.fill_diagonal(streak, True)
    cleaned = binary_opening(filled, structure=streak)
    return np.maximum(cleaned, cleaned.T), binary


def _fit_line(p0, p1):
    """Least-squares line through two points, as ``y = m x + c``."""
    (x0, y0), (x1, y1) = p0, p1
    if x1 == x0:
        m, c = 0.0, y0
    else:
        m = (y1 - y0) / (x1 - x0)
        c = y0 - m * x0
    return m, c


def _segments_from_matrix(binary, min_length):
    """Read one path through each connected region of a binarised matrix.

    A streak is several pixels thick, so a connected component is an area rather than a
    line. For each one the component is split at its centroid into a top-left and a
    bottom-right half, the centroid of each half is taken, and a line is fitted through
    those two points — which is exactly the shape a repeat makes, since it drifts away
    from the diagonal. The line is then run out to the sides or the ends of the
    component's bounding box, whichever it reaches first.

    :param binary: Binarised self-similarity matrix.
    :type binary: numpy.ndarray
    :param min_length: Shortest segment worth keeping, in matrix elements.
    :type min_length: int
    :return: Segments as ``[((x0, y0), (x1, y1)), ...]``.
    :rtype: list
    """
    if binary.size == 0:
        return []
    connectivity = generate_binary_structure(2, 2)
    labels, count = label(binary, structure=connectivity)

    segments = []
    for region in range(1, count + 1):
        ys, xs = (labels == region).nonzero()
        if len(xs) < 2:
            continue
        centre = (int(xs.mean()), int(ys.mean()))

        top_left = (xs <= centre[0]) & (ys <= centre[1])
        bottom_right = (xs > centre[0]) & (ys > centre[1])
        if not top_left.any() or not bottom_right.any():
            continue

        first = (int(xs[top_left].mean()), int(ys[top_left].mean()))
        last = (int(xs[bottom_right].mean()), int(ys[bottom_right].mean()))
        m, c = _fit_line(first, last)
        to_x = (lambda y: (y - c) / m) if m else (lambda y: first[0])
        to_y = lambda x: m * x + c

        north, south = int(ys.min()), int(ys.max())
        west, east = int(xs.min()), int(xs.max())

        if to_y(west) > north:
            corners = [(to_x(north), north), (to_x(south), south)]
        else:
            corners = [(west, to_y(west)), (east, to_y(east))]

        (x0, y0), (x1, y1) = corners
        span_x, span_y = int(round(x1)) - int(round(x0)), int(round(y1)) - int(round(y0))
        if span_x <= min_length or span_y <= min_length:
            continue
        x0, x1 = int(np.clip(round(x0), 0, binary.shape[0] - 1)), int(np.clip(round(x1), 0, binary.shape[0] - 1))
        y0, y1 = int(np.clip(round(y0), 0, binary.shape[1] - 1)), int(np.clip(round(y1), 0, binary.shape[1] - 1))
        segments.append(((x0, y0), (x1, y1)))
    return segments


def _overlap_fraction(a, b):
    """Fraction of ``a`` covered by ``b``, in ``[0, 1]``."""
    shared = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return shared / max(a[1] - a[0], 1e-9)


def _grow_segment(segment, soft, threshold, min_length):
    """Extend a segment's two ends while the softer matrix still supports it.

    The streak's path is known but its ends are only approximately placed. Walking
    outwards from each end along the fitted line and stopping where the raw similarity
    drops below a lower threshold recovers the true start and end of the phrase.

    :param segment: ``((x0, y0), (x1, y1))`` in matrix coordinates.
    :type segment: tuple
    :param soft: The un-binarised similarity matrix, rescaled to ``[0, 1]``.
    :type soft: numpy.ndarray
    :param threshold: Value below which the extension stops.
    :type threshold: float
    :param min_length: Shortest result worth keeping, in matrix elements.
    :type min_length: int
    :return: The grown segment, or the original if growing did not help.
    :rtype: tuple
    """
    (x0, y0), (x1, y1) = segment
    m, c = _fit_line((x0, y0), (x1, y1))
    n = soft.shape[0]

    def walk(x, y, dx, dy):
        while 0 <= x < n and 0 <= y < n and soft[x, y] >= threshold:
            x, y = x + dx, y + dy
        return x - dx, y - dy

    start = walk(x0, y0, -1, -1 if m >= 0 else 1)
    end = walk(x1, y1, 1, 1 if m >= 0 else -1)
    grown = (start, (end[0], m * end[0] + c))
    if (grown[1][0] - grown[0][0]) > min_length:
        return grown
    return segment


def extract_segments(binary, soft, exclusion_times, feature_times, cfg, *, window_s=0.0):
    """Turn the sharpened matrix into candidate repeats, then prune them.

    Segments shorter than ``min_pattern_length_seconds`` go first. The survivors are
    grown into the softer matrix, then any that crosses an excluded frame — a silence or
    a held note — is *split* at that frame, because a phrase does not continue through a
    phrase boundary.

    :param binary: Binarised self-similarity matrix.
    :type binary: numpy.ndarray
    :param soft: The rescaled-but-unbinarised similarity matrix.
    :type soft: numpy.ndarray
    :param exclusion_times: Times, in seconds, of every excluded pitch frame.
    :type exclusion_times: numpy.ndarray
    :param feature_times: Centre time in seconds of each matrix axis element.
    :type feature_times: numpy.ndarray
    :param cfg: Search settings.
    :type cfg: PhraseConfig
    :param window_s: Seconds of audio one feature frame summarises. A frame's centre
        sits a half window away from either edge of what it describes, so each span is
        widened by half a window at each end to name the audio rather than the frames.
    :type window_s: float
    :return: Candidate repeats, as a pair of ``(start_s, end_s)`` occurrences each.
    :rtype: list[tuple[tuple[float, float], tuple[float, float]]]
    """
    feature_times = np.asarray(feature_times, dtype=np.float64)
    exclusion_times = np.asarray(exclusion_times, dtype=np.float64)
    spacing = float(np.median(np.diff(feature_times))) if len(feature_times) > 1 else 1.0

    min_length = max(1, int(round((cfg.min_pattern_length_seconds - window_s) / spacing)))

    segments = _segments_from_matrix(binary, min_length)
    if not segments:
        return []

    grown = [_grow_segment(segment, soft, cfg.bin_thresh * cfg.segment_thresh_fraction,
                           min_length) for segment in segments]

    def to_span(first, last):
        first = int(np.clip(first, 0, len(feature_times) - 1))
        last = int(np.clip(last, 0, len(feature_times) - 1))
        return (max(0.0, float(feature_times[first]) - window_s / 2.0),
                float(feature_times[last]) + window_s / 2.0)

    pairs = []
    for ((x0, y0), (x1, y1)) in grown:


        first = _split_at_exclusions(x0, x1, feature_times, exclusion_times, cfg.break_seconds)
        second = _split_at_exclusions(y0, y1, feature_times, exclusion_times, cfg.break_seconds)
        if not first or not second:
            continue
        for i in range(max(len(first), len(second))):
            a = first[min(i, len(first) - 1)]
            b = second[min(i, len(second) - 1)]
            if not (a and b):
                continue
            left, right = to_span(*a), to_span(*b)


            if min(right[1] - right[0], left[1] - left[0]) < cfg.min_pattern_length_seconds:
                continue
            if _length_ratio(left, right) > cfg.length_tolerance:
                continue
            pairs.append((left, right))
    return pairs


def _length_ratio(a, b):
    """Relative difference between two spans' durations, in ``[0, 1]``."""
    longest, shortest = max(a[1] - a[0], b[1] - b[0]), min(a[1] - a[0], b[1] - b[0])
    return (longest - shortest) / longest if longest > 0 else 1.0


def _split_at_exclusions(start, end, feature_times, exclusion_times, break_seconds):
    """Cut one span at the phrase boundaries inside it.

    A phrase begins after a silence or a held note and ends before the next one, so
    leading and trailing excluded frames are simply trimmed off. A *split* happens only
    where an excluded run is long enough to be a genuine boundary — a breath, a pause,
    a held note that clearly ends one thought and starts another.

    Trimming rather than splitting at every excluded frame is what keeps this honest.
    Sustained notes are common, and cutting at every one of them shreds an otherwise
    good segment into fragments too short to be phrases, which loses most of the real
    repeats before they are ever grouped.

    :param start: First matrix element of the span.
    :type start: int
    :param end: Last matrix element of the span.
    :type end: int
    :param feature_times: Time in seconds of each matrix element.
    :type feature_times: numpy.ndarray
    :param exclusion_times: Times, in seconds, of the excluded frames.
    :type exclusion_times: numpy.ndarray
    :param break_seconds: Length an excluded run must reach before it splits a span.
    :type break_seconds: float
    :return: The pieces the span breaks into, as ``(first_element, last_element)``.
    :rtype: list[tuple[int, int]]
    """
    start, end = int(start), int(end)
    if end <= start:
        return []
    last = min(end, len(feature_times) - 1)
    if not len(exclusion_times):
        return [(start, last)]

    inside = exclusion_times[(exclusion_times >= feature_times[start])
                             & (exclusion_times <= feature_times[last])]
    if not len(inside):
        return [(start, last)]

    breaks = _long_runs(inside, break_seconds)

    first_piece = int(np.searchsorted(feature_times, breaks[0], side="right")) if breaks else start
    if first_piece > start and (last - first_piece) > 1:
        pieces = [(start, first_piece - 1)]
    else:
        pieces = []
        first_piece = start

    for cut in breaks:
        edge = int(np.searchsorted(feature_times, cut, side="right"))
        if edge - first_piece > 1:
            pieces.append((first_piece, edge - 1))
        first_piece = edge
    if last - first_piece > 1:
        pieces.append((first_piece, last))
    return [(a, b) for a, b in pieces if b > a]


def _long_runs(times, min_seconds):
    """Group sorted times into runs, keeping only those at least ``min_seconds`` long."""
    if not len(times):
        return []
    breaks, run = [], [times[0]]
    for value in times[1:]:
        if value - run[-1] > min_seconds * 0.5:
            if run[-1] - run[0] >= min_seconds:
                breaks.append((run[0] + run[-1]) / 2.0)
            run = [value]
        else:
            run.append(value)
    if run[-1] - run[0] >= min_seconds:
        breaks.append((run[0] + run[-1]) / 2.0)
    return breaks


def _group_segments(segments, cfg):
    """Collect the segments that point at the same regions into phrases.

    A segment names two occurrences of the same material. Two segments belong to the
    same phrase when they share an occurrence, so this walks the segments and unions
    them whenever an occurrence overlaps. The connected components of that union are
    the phrases: one group per phrase, holding every occurrence found for it.

    "Overlaps" is deliberately strict. Two spans have to overlap by
    ``merge_overlap`` *and* be within ``length_tolerance`` of each other in duration.
    Overlap alone is far too permissive — neighbouring phrases touch constantly, and a
    union-find over "anything that overlaps at all" chains a whole performance into one
    enormous group. Requiring comparable durations is what keeps phrases separate,
    because two statements of the same phrase really do run for about as long as each
    other, while a two-second fragment and a ten-second passage never do.

    :param segments: Candidate repeats, as pairs of ``(start_s, end_s)`` occurrences.
    :type segments: collections.abc.Sequence
    :param cfg: Search settings.
    :type cfg: PhraseConfig
    :return: Groups of non-overlapping occurrences, in seconds.
    :rtype: list[tuple[tuple[float, float], ...]]
    """
    parent = list(range(len(segments)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    def same_moment(a, b):
        return (_overlap_fraction(a, b) >= cfg.merge_overlap
                and _length_ratio(a, b) <= cfg.length_tolerance)

    for i, (a, b) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            c, d = segments[j]
            if same_moment(a, c) or same_moment(b, d) or same_moment(a, d) or same_moment(b, c):
                union(i, j)

    grouped = {}
    for i, occurrence in enumerate(segments):
        for span in occurrence:
            grouped.setdefault(find(i), []).append(span)

    return [_dedupe_spans(spans) for spans in grouped.values()]


def _dedupe_spans(spans, tolerance=0.0):
    """Merge touching or overlapping spans into a sorted, non-overlapping list."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _agreement(a, b, min_iou=0.5):
    """How much two sets of occurrences correspond, as a fraction in ``[0, 1]``.

    Occurrences are paired greedily by intersection over union and the score is the
    number of good pairs divided by the size of the smaller set. Pairing occurrence by
    occurrence, rather than comparing total duration, is what stops a phrase nested
    inside a longer one from being counted as a repeat of it: a nested group matches
    only one of the outer group's occurrences, so it scores low.

    :param a: Occurrences of one group.
    :type a: collections.abc.Sequence[tuple[float, float]]
    :param b: Occurrences of another group.
    :type b: collections.abc.Sequence[tuple[float, float]]
    :param min_iou: Overlap over union a pairing must reach.
    :type min_iou: float
    :return: The agreement, in ``[0, 1]``.
    :rtype: float
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0

    def iou(p, q):
        shared = _overlap_fraction(p, q) * (p[1] - p[0])
        return shared / ((p[1] - p[0]) + (q[1] - q[0]) - shared) if shared > 0 else 0.0

    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a, used_b, matched = set(), set(), 0
    for score, i, j in pairs:
        if score < min_iou or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
    return matched / min(len(a), len(b))


def _merge_groups(groups, cfg):
    """Fuse groups found in overlapping windows, and drop the ones too small to matter.

    Windows overlap by half their length, so a phrase near a window edge comes back
    twice with much the same occurrences. Two groups are the same phrase when enough of
    their occurrences pair up; a fused group keeps the union.

    :param groups: Groups of occurrence spans, in seconds.
    :type groups: collections.abc.Sequence
    :param cfg: Search settings.
    :type cfg: PhraseConfig
    :return: Merged groups with at least ``min_in_group`` occurrences.
    :rtype: list[tuple[tuple[float, float], ...]]
    """
    clusters = []
    for spans in groups:
        spans = _dedupe_spans(spans)
        if len(spans) < cfg.min_in_group:
            continue
        for cluster in clusters:
            if any(_agreement(spans, other) >= cfg.group_agreement for other in cluster):
                cluster.append(spans)
                break
        else:
            clusters.append([spans])

    merged = []
    for cluster in clusters:
        combined = _dedupe_spans([span for member in cluster for span in member])


        combined = tuple(s for s in combined if s[1] - s[0] >= cfg.min_pattern_length_seconds)


        if len(combined) >= cfg.min_in_group:
            durations = np.array([end - start for start, end in combined])
            middle = float(np.median(durations))
            combined = tuple(span for span, span_s in zip(combined, durations)
                             if _length_ratio((0.0, middle), (0.0, span_s)) <= cfg.length_tolerance)
        if len(combined) >= cfg.min_in_group:
            merged.append(combined)
    return merged


def _windows(n_frames, step, cfg):
    """Yield the ``(lo, hi)`` pitch-frame bounds of each self-similarity window."""
    width = max(2, round(cfg.window_seconds / step))
    stride = max(1, round(cfg.window_hop_seconds / step))
    for lo in range(0, n_frames, stride):
        yield lo, min(lo + width, n_frames)


def find_repeated_phrases(times, f0, cfg=None, *, features=None, feature_times=None,
                          feature_window=0.0):
    """Find the melodic phrases that occur more than once in a single song.

    Runs the pipeline on one self-similarity window at a time and merges the groups
    found in overlapping windows, so a phrase that straddles a window edge is
    recognised as one phrase rather than two.

    :param times: Pitch frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param cfg: Search settings; the :class:`PhraseConfig` default if ``None``.
    :type cfg: PhraseConfig or None
    :param features: Precomputed feature frames; the pitch-based ones are computed from
        ``f0`` when this is ``None``.
    :type features: numpy.ndarray or None
    :param feature_times: Centre time in seconds of each feature frame.
    :type feature_times: numpy.ndarray or None
    :param feature_window: Seconds of audio each feature frame summarises, widened off
        the reported spans so they name audio rather than frames.
    :type feature_window: float
    :return: Repeated phrases, ordered by first occurrence.
    :rtype: list[PhraseGroup]
    """
    cfg = cfg or PhraseConfig()
    times, f0 = np.asarray(times, dtype=np.float64), np.asarray(f0, dtype=np.float64)
    if len(times) < 2:
        return []

    step = frame_step(times)


    bridged = _interpolate_gaps(f0, cfg.max_gap_seconds / step)
    if features is None:
        features, hop = pitch_features(times, bridged, cfg)
        feature_times = (np.arange(len(features)) + 1) * hop
        feature_window = cfg.feature_window_seconds
    features = np.asarray(features, dtype=np.float64)
    feature_times = np.asarray(feature_times, dtype=np.float64)
    if len(features) < 4 or len(features) != len(feature_times):
        return []

    spacing = float(np.median(np.diff(feature_times)))
    _, _, exclusion = exclusion_mask(f0, times, cfg)
    exclusion_times = times[exclusion]

    found = []
    for lo, hi in _windows(len(times), step, cfg):
        window_exclusion = exclusion[lo:hi]
        if window_exclusion.all():
            continue

        lo_t, hi_t = lo * step, hi * step
        first = max(0, int(np.searchsorted(feature_times, lo_t, side="left")) - 1)
        last = min(len(feature_times), int(np.searchsorted(feature_times, hi_t, side="right")) + 1)
        if last - first < 4:
            continue


        frame_index = np.clip(
            ((feature_times[first:last] - lo_t) / step).astype(int), 0, len(window_exclusion) - 1)
        keep = ~window_exclusion[frame_index]

        matrix, matrix_times = self_similarity(features[first:last], keep,
                                               feature_times=feature_times[first:last])
        if matrix.shape[0] < 4:
            continue

        processed, binary = emphasise_diagonals(matrix, cfg.bin_thresh, spacing=spacing)
        if not processed.any():
            continue
        segments = extract_segments(binary, processed.astype(np.float64),
                                    exclusion_times - lo_t, matrix_times - lo_t, cfg,
                                    window_s=feature_window)
        segments = [[(s + lo_t, e + lo_t) for s, e in side] for side in segments]
        found.extend(_group_segments(segments, cfg))

    merged = _merge_groups(found, cfg)
    merged.sort(key=lambda spans: (spans[0][0], -len(spans)))
    return [PhraseGroup(index=i, spans=spans) for i, spans in enumerate(merged)]


def discard_duplicate_phrases(groups, times):
    """Keep one occurrence of every repeated phrase and mark the rest as duplicates.

    A phrase belongs in a study of the raga once, not once per time the performer
    circles back to it, so of every group of occurrences exactly one is kept — the
    earliest statement — and the rest are cut from the pitch track.

    The groups are claimed longest first, so the most substantial repeat is the one
    allowed to claim a stretch of the track and a brief incidental one cannot mask it.
    A repeat that overlaps something already kept is discarded only *outside* that
    overlap: cutting a hole through a phrase being kept would be worse than the repeat.

    :param groups: Repeated phrases found by :func:`find_repeated_phrases`.
    :type groups: collections.abc.Sequence[PhraseGroup]
    :param times: Pitch frame times in seconds; the mask is returned on this grid.
    :type times: numpy.ndarray
    :return: Boolean per pitch frame, ``True`` where the frame belongs to a discarded
        repeat.
    :rtype: numpy.ndarray
    """
    times = np.asarray(times, dtype=np.float64)
    step = frame_step(times)
    duplicates = np.zeros(len(times), dtype=bool)
    if not groups:
        return duplicates

    ordered = sorted(groups, key=lambda group: -group.duration_s)
    kept = [sorted(group.spans)[0] for group in ordered if group.spans]
    repeats = [span for group in ordered for span in sorted(group.spans)[1:]]

    claimed = np.zeros(len(times), dtype=bool)
    for start, end in kept:
        lo, hi = _span_frames(start, end, step, len(times))
        claimed[lo:hi] = True
    for start, end in repeats:
        lo, hi = _span_frames(start, end, step, len(times))
        duplicates[lo:hi] |= ~claimed[lo:hi]
    return duplicates


def _span_frames(start_s, end_s, step, n_frames):
    """Convert a second span to a half-open frame range clipped to the track."""
    lo = int(np.clip(round(start_s / step), 0, max(n_frames - 1, 0)))
    hi = int(np.clip(round(end_s / step), lo, n_frames))
    return lo, hi


def deduplicate_pitch(times, f0, duplicates):
    """Cut the discarded repeats out of a pitch track.

    Frames marked as duplicates are removed and the ones either side are joined, so the
    result is a contiguous, shortened track rather than one with holes in it.

    :param times: Frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param duplicates: Boolean per frame; ``True`` where it is a discarded repeat.
    :type duplicates: numpy.ndarray
    :return: ``(times, f0)`` of the shortened track, on a uniform grid.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    times, f0 = np.asarray(times), np.asarray(f0)
    keep = ~np.asarray(duplicates, dtype=bool)
    if keep.all():
        return times, f0
    kept = np.flatnonzero(keep)
    return np.arange(len(kept)) * frame_step(times), f0[kept]


def remove_repeated_phrases(times, f0, cfg=None):
    """Find every phrase stated more than once and cut all but its first statement.

    This is the whole search as one call: :func:`find_repeated_phrases` locates the
    phrases, :func:`discard_duplicate_phrases` decides which statements go, and
    :func:`deduplicate_pitch` closes the gaps they leave behind.

    :param times: Pitch frame times in seconds.
    :type times: numpy.ndarray
    :param f0: Pitch in Hz, ``0.0`` where unvoiced.
    :type f0: numpy.ndarray
    :param cfg: Search settings; the :class:`PhraseConfig` default if ``None``.
    :type cfg: PhraseConfig or None
    :return: ``(times, f0, duplicates)`` — the shortened track on the same frame grid,
        and the boolean mask of the frames removed from it, on the original grid.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    groups = find_repeated_phrases(times, f0, cfg)
    duplicates = discard_duplicate_phrases(groups, times)
    kept_times, kept_f0 = deduplicate_pitch(times, f0, duplicates)
    return kept_times, kept_f0, duplicates


