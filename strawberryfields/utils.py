import librosa
import numpy as np
import psycopg2
from rich.progress import Progress
from scipy.ndimage import gaussian_filter1d

from strawberryfields import codec
from strawberryfields.dtw import dtw


_profile_cache = {}


def get_tracks(database_url):
    """Yield every track's stored pitch contour, one row at a time.

    The table is never pulled down whole: each row is fetched on its own, unpacked
    with :func:`strawberryfields.codec.decode_contour` and yielded, so only one
    stored blob is in memory at a time.

    :param database_url: Connection string for the app database.
    :type database_url: str
    :return: Track ids paired with their pitch contours in cents, one row at a time.
    :rtype: collections.abc.Iterator[tuple[int, numpy.ndarray]]
    """
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            conn = psycopg2.connect(database_url)
            break
        except psycopg2.OperationalError:
            if attempt == attempts:
                raise
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, pitch_track FROM tracks")
            while True:
                row = cur.fetchone()
                if row is None:
                    break
                track_id, blob = row
                yield track_id, codec.decode_contour(bytes(blob))[0]
    finally:
        conn.close()


def load_audio(path):
    """Decode an audio file to a mono waveform at its native sampling rate.

    :param path: Path to the audio file (any format librosa/ffmpeg can decode).
    :type path: str
    :return: The waveform and its sampling rate in Hz.
    :rtype: tuple[numpy.ndarray, int]
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float64), sr


def hz_to_cents(f0, f_ref):
    """Convert frequencies in Hz to cents relative to a reference frequency.

    Unvoiced ``nan`` and non-positive values stay ``nan``.

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


def pitch_class_profile(contour, n_classes, sigma_cents=0.0):
    """Fold a cents contour into an octave-invariant pitch-class salience profile.

    Every pitch is reduced mod 1200 cents (its position within an octave, regardless
    of which octave), then rendered as a salience image: a voiced frame is 1 at its
    pitch-class bin and 0 elsewhere, blurred along the pitch-class axis with a
    Gaussian that wraps around the octave (0 and 1200 cents are the same point).
    Because pitch class is octave-invariant by construction, matching against several
    circular shifts of this profile (see :func:`dtw_cost`) covers
    a range of tonic differences with plain, un-windowed subsequence DTW -- no
    per-window re-centring is needed, and ordinary DTW warping absorbs tempo
    differences on its own.

    :param contour: Pitch contour in cents, ``nan`` where unvoiced.
    :type contour: numpy.ndarray
    :param n_classes: Number of pitch classes per octave.
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


def dtw_cost(query_profile, reference_profile, n_classes, metric="euclidean", shift_step=1):
    """Match a query pitch-class profile inside a reference, over several transpositions.

    Circularly shifting a pitch-class profile by one class is exactly a transposition
    by ``1200 / n_classes`` cents. ``shift_step`` trades transposition coverage for
    speed: with ``shift_step=1`` every class shift is tried; with ``shift_step=2``
    only every other one is (semitone steps, if ``n_classes=24``), relying on the
    profile's Gaussian blur to absorb the skipped in-between shifts. Each shift tried
    is one ordinary, un-windowed subsequence DTW over the whole reference: the last
    row of the accumulated cost matrix is the matching function, normalised by the
    shorter of the two lengths.

    :param query_profile: Query profile, as returned by :func:`pitch_class_profile`.
    :type query_profile: numpy.ndarray
    :param reference_profile: Reference profile, as returned by :func:`pitch_class_profile`.
    :type reference_profile: numpy.ndarray
    :param n_classes: Number of pitch classes per octave; must match how both
        profiles were built.
    :type n_classes: int
    :param metric: Distance metric passed to :func:`strawberryfields.dtw.dtw`.
    :type metric: str
    :param shift_step: Try every ``shift_step``-th class shift instead of all of them.
    :type shift_step: int
    :return: The lowest subsequence-DTW cost over the transpositions tried.
    :rtype: float
    """
    best = np.inf
    for shift in range(0, n_classes, shift_step):
        shifted = np.roll(reference_profile, shift, axis=0)
        D = dtw(X=query_profile, Y=shifted, metric=metric, subseq=True, backtrack=False)
        cost = D[-1, :].min() / min(query_profile.shape[1], reference_profile.shape[1])
        if cost < best:
            best = cost
    return best


def top_matches(query_profile, tracks, pitch_config, k=3):
    """Find the tracks whose pitch-class profiles contain the best subsequence matches.

    Each track's stored contour is turned into a pitch-class profile the first time
    it's seen and memoised, so across searches every song is profiled exactly once;
    the profile is compared against the query with :func:`dtw_cost`.
    Memoised profiles are stored as ``float16`` so the whole catalogue fits in a
    512 MB instance, and :func:`dtw_cost` upcasts one track at a time as it
    compares.

    :param query_profile: Profile of the query, as returned by
        :func:`pitch_class_profile`.
    :type query_profile: numpy.ndarray
    :param tracks: Track ids paired with their pitch contours in cents, fetched one
        row at a time by :func:`get_tracks`.
    :type tracks: collections.abc.Iterator[tuple[int, numpy.ndarray]]
    :param pitch_config: Settings with the keys ``n_classes``, ``sigma_cents`` and
        ``shift_step``.
    :type pitch_config: dict
    :param k: How many matches to return.
    :type k: int
    :return: The ``k`` lowest-cost ``(track_id, cost)`` pairs, cheapest first; every pair
        when the catalogue holds fewer than ``k`` tracks.
    :rtype: list[tuple[int, float]]
    """
    results = []

    with Progress() as progress:
        task = progress.add_task("Searching tracks...")
        for track_id, contour in tracks:
            if len(contour) == 0:
                progress.advance(task)
                continue
            track_profile = _profile_cache.get(track_id)
            if track_profile is None:
                track_profile = pitch_class_profile(
                    contour, pitch_config["n_classes"], pitch_config["sigma_cents"]).astype(np.float16)
                _profile_cache[track_id] = track_profile
            cost = dtw_cost(query_profile, track_profile, pitch_config["n_classes"],
                            "euclidean", pitch_config["shift_step"])
            results.append((track_id, cost))
            progress.advance(task)

    return sorted(results, key=lambda r: r[1])[:k]
