"""The corpus on disk: one folder per song, an audio file and a metadata file in each.

The corpus folder is passed on the command line, so it can hold any collection of
recordings, laid out as::

    <data folder>/<song folder>/audio.mp3
    <data folder>/<song folder>/metadata.json

A song folder's name is only its corpus's identifier for that song — it may be named
anything, because titles are not unique across recordings. The title the database
stores comes from the metadata file.

``metadata.json`` carries the three fields every row holds::

    {"title": "Song 1", "raga": "Raga 1", "tala": "Tala 1"}

``title``, ``raga`` and ``tala`` are each one string — a song has one raga and one
tala, so no field ever holds several. Saraga's export wraps ``raga`` and ``tala`` in a
list of one, and a list of one is read as that one name; a list of several is refused.
Its spelling of those two fields, ``raaga`` and ``taala``, is read as the same field.

:func:`verify_corpus` walks a whole corpus and reports everything wrong with it at
once, so a bad copy is found in one pass rather than one fault per run;
:func:`scan_recordings` refuses to hand back songs from a corpus that did not pass.
"""

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AUDIO_FILENAME",
    "METADATA_FILENAME",
    "ERROR",
    "WARNING",
    "Problem",
    "Recording",
    "layout_problems",
    "song_folders",
    "verify_song",
    "verify_corpus",
    "scan_recordings",
]

AUDIO_FILENAME = "audio.mp3"

METADATA_FILENAME = "metadata.json"

ERROR = "error"

WARNING = "warning"


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a corpus, as :func:`verify_corpus` found it.

    :param song: The song folder's name, or ``""`` for the data folder itself.
    :type song: str
    :param field: What the problem is about — a filename, or a metadata field.
    :type field: str
    :param severity: :data:`ERROR` when the corpus cannot be built from, or
        :data:`WARNING` when it can be built but something looks wrong.
    :type severity: str
    :param detail: What is wrong, phrased to follow the field.
    :type detail: str
    """

    song: str
    field: str
    severity: str
    detail: str


@dataclass(frozen=True)
class Recording:
    """One song of the corpus, as it sits on disk with its metadata read.

    :param identifier: The song folder's name, unique within the corpus.
    :type identifier: str
    :param audio_path: Path of the song's audio file.
    :type audio_path: pathlib.Path
    :param title: Title of the piece performed.
    :type title: str
    :param raga: The recording's one raga.
    :type raga: str
    :param tala: The recording's one tala.
    :type tala: str
    """

    identifier: str
    audio_path: Path
    title: str
    raga: str
    tala: str


def layout_problems(data_folder):
    """Check the data folder itself, before any song inside it is looked at.

    What is checked: that ``data folder`` exists and is a directory; that no plain
    files sit beside the song folders, which would be recordings no build can see; and
    that the folder holds song folders at all. Hidden entries are left out of the check
    — ``.DS_Store`` and ``.git`` are the operating system's business, not the corpus's.

    :param data_folder: Directory holding one subdirectory per song.
    :type data_folder: str or pathlib.Path
    :return: One problem per fault found; empty when the folder's shape is sound.
    :rtype: list[Problem]
    """
    root = Path(data_folder)
    if not root.is_dir():
        return [Problem("", "data folder", ERROR, f"{root} is not a folder")]

    problems = [Problem(path.name, "data folder", ERROR, "is a file, not a song folder")
                for path in sorted(root.iterdir())
                if path.is_file() and not path.name.startswith(".")]
    if not _song_folders(root):
        problems.append(Problem("", "data folder", ERROR, f"{root} holds no song folders"))
    return problems


def song_folders(data_folder):
    """The song folders of a corpus, in name order, hidden folders left out.

    :param data_folder: Directory holding one subdirectory per song.
    :type data_folder: str or pathlib.Path
    :return: One path per song folder; empty when ``data folder`` is not a folder.
    :rtype: list[pathlib.Path]
    """
    root = Path(data_folder)
    if not root.is_dir():
        return []
    return _song_folders(root)


def verify_song(directory):
    """Check one song folder: its audio file, then its metadata file.

    What is checked: that ``audio.mp3`` exists, is a file, and is not empty; that
    ``metadata.json`` exists and parses; and that the metadata carries a usable
    ``title``, ``raga`` and ``tala``. An MP3 that does not look like one is a warning,
    not an error — a strange first byte is worth a look but is no reason to stop.

    :param directory: The song folder.
    :type directory: pathlib.Path
    :return: One problem per fault found; empty when the folder is sound.
    :rtype: list[Problem]
    """
    song = directory.name
    problems = []
    audio = directory / AUDIO_FILENAME
    metadata = directory / METADATA_FILENAME

    if not audio.is_file():
        problems.append(Problem(song, AUDIO_FILENAME, ERROR, "is missing"))
    elif audio.stat().st_size == 0:
        problems.append(Problem(song, AUDIO_FILENAME, ERROR, "is empty"))
    elif not _looks_like_mp3(audio):
        problems.append(Problem(song, AUDIO_FILENAME, WARNING,
                                "does not start like an MP3 file"))

    if not metadata.is_file():
        problems.append(Problem(song, METADATA_FILENAME, ERROR, "is missing"))
    else:
        try:
            with metadata.open(encoding="utf-8") as stream:
                parsed = json.load(stream)
        except json.JSONDecodeError as error:
            problems.append(Problem(song, METADATA_FILENAME, ERROR,
                                    f"is not valid JSON: {error}"))
        except OSError as error:
            problems.append(Problem(song, METADATA_FILENAME, ERROR,
                                    f"cannot be read: {error}"))
        else:
            _, found = _parse_metadata(parsed, song)
            problems.extend(found)
    return problems


def verify_corpus(data_folder, on_song=None):
    """Check a whole corpus, without stopping at the first fault.

    What is checked: the shape of ``data folder`` itself, and then every song folder
    in it. A corpus with a hundred broken folders reports a hundred problems, so one
    run is enough to know what has to be fixed.

    :param data_folder: Directory holding one subdirectory per song.
    :type data_folder: str or pathlib.Path
    :param on_song: Called with each song folder once its check has finished, so a
        caller can draw a progress bar; ``None`` draws nothing.
    :type on_song: callable or None
    :return: ``(songs, problems)`` — how many song folders were checked, and every
        problem found, in folder order.
    :rtype: tuple[int, list[Problem]]
    """
    problems = layout_problems(data_folder)
    folders = song_folders(data_folder)
    for directory in folders:
        problems.extend(verify_song(directory))
        if on_song is not None:
            on_song(directory)
    return len(folders), problems


def scan_recordings(data_folder):
    """Read every song of a corpus folder that passes :func:`verify_corpus`.

    :param data_folder: Directory holding one subdirectory per song.
    :type data_folder: str or pathlib.Path
    :return: The songs, in the folder's own order.
    :rtype: list[Recording]
    :raises ValueError: if verification found errors — the message carries the first
        few, and :func:`verify_corpus` carries all of them.
    """
    songs, problems = verify_corpus(data_folder)
    errors = [problem for problem in problems if problem.severity == ERROR]
    if errors:
        found = "; ".join(f"{problem.song or 'data folder'}: "
                          f"{problem.field} {problem.detail}"
                          for problem in errors[:3])
        raise ValueError(f"{Path(data_folder)} failed verification with "
                         f"{len(errors)} problems in {songs} song folders — {found}")

    recordings = []
    for directory in song_folders(data_folder):
        title, raga, tala = _read_metadata(directory / METADATA_FILENAME)
        recordings.append(Recording(identifier=directory.name,
                                    audio_path=directory / AUDIO_FILENAME,
                                    title=title, raga=raga, tala=tala))
    return recordings


def _song_folders(root):
    """The song folders of a corpus, in name order, leaving hidden folders out."""
    return sorted(path for path in root.iterdir()
                  if path.is_dir() and not path.name.startswith("."))


def _looks_like_mp3(path):
    """Whether a file opens as an MP3 does: an ID3 tag, or a frame sync."""
    try:
        with path.open("rb") as stream:
            head = stream.read(3)
    except OSError:
        return False
    return head.startswith(b"ID3") or (
        len(head) == 3 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def _read_metadata(path):
    """Read a song's metadata file into the three fields the database stores.

    :param path: Path of the ``metadata.json`` file.
    :type path: pathlib.Path
    :return: ``(title, raga, tala)``.
    :rtype: tuple[str, str, str]
    :raises ValueError: if the file does not parse or does not carry usable fields.
    """
    with path.open(encoding="utf-8") as stream:
        parsed = json.load(stream)
    values, problems = _parse_metadata(parsed, path.parent.name)
    if problems:
        raise ValueError(f"{path} — {problems[0].field} {problems[0].detail}")
    return values


def _parse_metadata(metadata, song):
    """Take the three fields out of a parsed metadata file, noting what is unusable.

    :param metadata: Whatever the JSON file held.
    :type metadata: object
    :param song: The song folder's name, for the problems found.
    :type song: str
    :return: ``(values, problems)`` — ``(title, raga, tala)`` with each unusable field
        left empty, and one problem per field that is unusable.
    :rtype: tuple[tuple[str, str, str], list[Problem]]
    """
    if not isinstance(metadata, dict):
        return ("", "", ""), [Problem(song, METADATA_FILENAME, ERROR,
                                      "must hold a JSON object")]

    problems = []
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        problems.append(Problem(song, "title", ERROR, "needs a non-empty string"))
        title = ""

    raga, raga_problem = _one_name(metadata, "raga", "raaga")
    if raga_problem:
        problems.append(Problem(song, "raga", ERROR, raga_problem))
    tala, tala_problem = _one_name(metadata, "tala", "taala")
    if tala_problem:
        problems.append(Problem(song, "tala", ERROR, tala_problem))

    return (title, raga, tala), problems


def _one_name(metadata, key, alias):
    """Read one of the single-name fields, which hold one name and never several.

    :param metadata: The parsed metadata file.
    :type metadata: dict
    :param key: The field's name in this project's format.
    :type key: str
    :param alias: The field's name in Saraga's export.
    :type alias: str
    :return: ``(value, problem)`` — the one name the field holds, and what to say when
        it does not hold exactly one.
    :rtype: tuple[str, str or None]
    """
    value = metadata[key] if key in metadata else metadata.get(alias)
    if isinstance(value, str):
        if value.strip():
            return value, None
    elif isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], str) and value[0].strip():
            return value[0], None
        if len(value) > 1:
            return "", f"holds {len(value)} names, but a song has one {key}"
    return "", f'needs "{key}" as a string'
