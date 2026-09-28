"""The Saraga corpus on disk: one folder per recording, one JSON file of metadata.

The corpus is laid out as::

    data/saraga/<folder>/<folder>.mp3
    data/saraga/<folder>/<folder>.json

and the JSON carries the three fields this database stores::

    {"title": "Kalaye", "raaga": ["Kedaragaula"], "taala": ["Adi"]}

``raaga`` and ``taala`` are lists of the names a recording is known by, which are
joined into the single string the ``raga`` and ``tala`` columns hold.

A folder's own name is the corpus's identifier for the recording, because titles are
not unique — the corpus holds ten different *Thillana*s — while folder names are. It
names the cached pitch track and the built artefact, and it is not stored in the
database.
"""

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Recording",
    "scan_recordings",
]


@dataclass(frozen=True)
class Recording:
    """One recording of the corpus, as it sits on disk with its metadata read.

    :param identifier: The corpus folder's name, unique within the corpus.
    :type identifier: str
    :param audio_path: Path of the recording's audio file.
    :type audio_path: pathlib.Path
    :param title: Title of the piece performed.
    :type title: str
    :param raga: Raga the performance is set in, several names joined by ``, ``.
    :type raga: str
    :param tala: Tala the performance is set in, several names joined by ``, ``.
    :type tala: str
    """

    identifier: str
    audio_path: Path
    title: str
    raga: str
    tala: str


def scan_recordings(dataset_root):
    """Read every recording of a corpus folder.

    :param dataset_root: Directory holding one subdirectory per recording.
    :type dataset_root: str or pathlib.Path
    :return: The recordings, in the corpus' own order.
    :rtype: list[Recording]
    :raises ValueError: if a subdirectory holds no audio file or no metadata file.
    """
    root = Path(dataset_root)
    recordings = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        audio = _only(directory.glob("*.mp3"))
        metadata = _only(directory.glob("*.json"))
        if audio is None or metadata is None:
            raise ValueError(f"{directory} needs one .mp3 and one .json file")
        title, raga, tala = _read_metadata(metadata)
        recordings.append(Recording(identifier=directory.name, audio_path=audio,
                                    title=title, raga=raga, tala=tala))
    return recordings


def _only(paths):
    """Return the single path among ``paths``, or ``None`` when there is not exactly one."""
    matches = sorted(paths)
    return matches[0] if len(matches) == 1 else None


def _read_metadata(path):
    """Read a recording's JSON file into the three fields the database stores.

    :param path: Path of the JSON file.
    :type path: pathlib.Path
    :return: ``(title, raga, tala)``, each empty when the file does not carry it.
    :rtype: tuple[str, str, str]
    """
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    title = str(metadata.get("title") or path.stem)
    return title, _join_names(metadata.get("raga") or metadata.get("raaga")), \
        _join_names(metadata.get("tala") or metadata.get("taala"))


def _join_names(names):
    """Join the names a metadata field lists into one string, keeping first-seen order.

    :param names: A single name, a list of names, or anything absent.
    :type names: str or list or None
    :return: The names joined by ``, ``, or ``""`` when there are none.
    :rtype: str
    """
    if isinstance(names, str):
        return names
    if not isinstance(names, list):
        return ""
    return ", ".join(dict.fromkeys(str(name) for name in names))
