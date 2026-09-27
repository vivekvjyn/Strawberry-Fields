"""The reference implementation of repeated-phrase discovery, as one importable module.

Every algorithmic step in the notebook comes from here: the trained complex autoencoder
that turns audio into a feature frame per moment, the self-similarity matrix computed
while skipping the frames the mask excludes, the extractor that reads diagonal streaks
out of that matrix, and the grouping that turns those streaks into occurrences of a
phrase. None of it is re-implemented in this repository — the helpers only make the
pieces importable, keep the one heavyweight model loaded once per session, and keep the
expensive results of each stage on disk so a re-run only redoes what changed.
"""

from hashlib import sha1
from pathlib import Path

import numpy as np  # noqa: E402
import compiam  # noqa: E402
from compiam.melody.pattern import self_similarity as _compiam_self_similarity  # noqa: E402
from compiam.melody.pattern import segmentExtractor  # noqa: E402
from compiam.utils import add_center_to_mask, run_or_cache  # noqa: E402
from compiam.utils.pitch import (  # noqa: E402
    extract_stability_mask,
    interpolate_below_length,
    pitch_seq_to_cents,
)

__all__ = [
    "digest",
    "extract_features",
    "extract_pitch",
    "load_encoder",
    "load_pitch_model",
    "self_similarity",
    "segmentExtractor",
    "add_center_to_mask",
    "extract_stability_mask",
    "interpolate_below_length",
    "pitch_seq_to_cents",
    "tonic",
]

# Tonic of the recording, only used to name the axis of the pitch plots.
tonic = 195.99

_ENCODER = None
_PITCH_MODEL = None


def load_encoder():
    """The trained complex autoencoder, loaded once and reused.

    :return: Wrapper whose ``extract_features`` gives the feature frame of a recording and
        whose ``hop_length`` and ``sr`` say how much audio one frame covers.
    :rtype: object
    """
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = compiam.load_model("melody:cae-carnatic")
    return _ENCODER


def load_pitch_model():
    """The trained neural pitch tracker, loaded once and reused.

    :return: Wrapper whose ``predict`` gives the predominant pitch of a recording as an
        ``(n, 2)`` array of ``[time_s, pitch_hz]`` rows.
    :rtype: object
    """
    global _PITCH_MODEL
    if _PITCH_MODEL is None:
        _PITCH_MODEL = compiam.load_model("melody:ftanet-carnatic")
    return _PITCH_MODEL


def digest(*arrays):
    """A short tag that changes as soon as one of the arrays does.

    Cache files are named with it, so an intermediate built from a different mask is
    recomputed rather than quietly read from the file the old mask left behind.

    :param arrays: Arrays whose contents decide the tag.
    :type arrays: numpy.ndarray
    :return: Eight hex characters.
    :rtype: str
    """
    sha = sha1()
    for array in arrays:
        array = np.ascontiguousarray(array)
        sha.update(array.dtype.str.encode())
        sha.update(str(array.shape).encode())
        sha.update(array.tobytes())
    return sha.hexdigest()[:8]


def _cache_file(cache_dir, path):
    """Where an intermediate of ``path`` is kept, or ``None`` when there is no caching.

    :param cache_dir: Directory to read and write in; ``None`` disables the cache.
    :type cache_dir: str or pathlib.Path or None
    :param path: Recording whose intermediate is being cached.
    :type path: str or pathlib.Path
    :return: Path of the ``.npz`` file, or ``None``.
    :rtype: pathlib.Path or None
    """
    if cache_dir is None:
        return None
    return Path(cache_dir) / (Path(path).stem + ".npz")


def extract_pitch(path, cache_dir=None, model=None):
    """The pitch track of a recording, frame by frame, cached.

    Tracking is the slowest stage of the notebook — around forty seconds for a
    four-minute recording — and depends on nothing but the audio and the weights, so the
    rows are kept as one compressed ``.npz`` per recording. Reading one also skips
    loading the tracker.

    :param path: Audio file to track.
    :type path: str or pathlib.Path
    :param cache_dir: Directory to read and write the ``.npz`` in; ``None`` computes
        without touching the disk.
    :type cache_dir: str or pathlib.Path or None
    :param model: Tracker to run on a miss; ``None`` loads the default one.
    :type model: object or None
    :return: One ``[time_s, pitch_hz]`` row per frame of the track.
    :rtype: numpy.ndarray
    """
    cache = _cache_file(cache_dir, path)
    if cache is not None and cache.exists():
        return np.load(cache)["track"]
    tracker = load_pitch_model() if model is None else model
    track = tracker.predict(str(path))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, track=track)
    return track


def extract_features(path, cache_dir=None, encoder=None):
    """The feature frame of every moment of a recording, cached.

    Only the magnitudes are kept — the notebook throws the phase away — as one
    compressed ``.npz`` per recording. A miss returns the encoder's own tensor and a hit
    wraps what was stored back into one, since compIAM calls ``.detach()`` on it.

    :param path: Audio file to describe.
    :type path: str or pathlib.Path
    :param cache_dir: Directory to read and write the ``.npz`` in; ``None`` computes
        without touching the disk.
    :type cache_dir: str or pathlib.Path or None
    :param encoder: Encoder to run on a miss; ``None`` loads the default one.
    :type encoder: object or None
    :return: ``n_frames x n_bases`` magnitudes.
    :rtype: torch.Tensor
    """
    cache = _cache_file(cache_dir, path)
    if cache is not None and cache.exists():
        import torch

        return torch.from_numpy(np.load(cache)["ampl"])
    encoder = load_encoder() if encoder is None else encoder
    ampl, _ = encoder.extract_features(str(path))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ampl=ampl.detach().cpu().numpy())
    return ampl


def self_similarity(features, exclusion_mask=None, timestep=None, hop_length=None, sr=44100,
                    cache_dir=None, key="matrix"):
    """compIAM's self-similarity, with the matrix kept between runs.

    Everything comes back as compIAM returns it — matrix, lookup tables, boundary lists
    — but the matrix, the only part that grows with the square of the recording, is read
    from or written to ``cache_dir/<key>.pkl`` through compIAM's own cache utility.
    ``key`` must name what the matrix was computed from, the recording and the mask, so
    that a different mask lands in a different file. ``cache_dir=None`` computes without
    touching the disk.

    :param features: Feature frames of the recording.
    :type features: torch.Tensor
    :param exclusion_mask: Region not to compare, one flag per pitch frame.
    :type exclusion_mask: numpy.ndarray or None
    :param timestep: Time in seconds between elements of ``exclusion_mask``.
    :type timestep: float or None
    :param hop_length: Audio frames covered by one feature frame.
    :type hop_length: int or None
    :param sr: Sample rate of the audio.
    :type sr: int
    :param cache_dir: Directory to read and write ``<key>.pkl`` in; ``None`` for none.
    :type cache_dir: str or pathlib.Path or None
    :param key: Name for what the matrix was computed from.
    :type key: str
    :return: ``(matrix, orig_sparse_lookup, sparse_orig_lookup, boundaries_orig,
        boundaries_sparse)``.
    :rtype: tuple
    """
    if cache_dir is None:
        return _compiam_self_similarity(features, exclusion_mask=exclusion_mask,
                                        timestep=timestep, hop_length=hop_length, sr=sr)
    return run_or_cache(
        _compiam_self_similarity,
        [features, exclusion_mask, timestep, hop_length, sr],
        str(Path(cache_dir) / (key + ".pkl")),
    )
