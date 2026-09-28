"""Verify a corpus folder's format before anything is built from it.

    python scripts/verify_data.py your_data

The layout every song folder must have:

    <data folder>/<song folder>/audio.mp3
    <data folder>/<song folder>/metadata.json

and the metadata file's three fields:

    {"title": "Song 1", "raga": "Raga 1", "tala": "Tala 1"}

``title``, ``raga`` and ``tala`` are each one string: a song has one raga and one tala.
Saraga's export spells those two fields ``raaga`` and ``taala`` and wraps each in a list
of one, and both spellings and a list of one are read as that one name; a list of
several is an error.

Every song folder is checked, not just the first broken one, and the findings are
printed as a table: errors first, in red, then warnings, in yellow. Warnings describe
a corpus that can still be built from; errors describe one that cannot.

Exit status is 0 when the corpus is usable, 1 when it is not.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pitchtrack.console import check_corpus, console, show, summary_table
from pitchtrack.corpus import ERROR, scan_recordings


def describe(recordings):
    """Summarise what a corpus holds, from its metadata and its files' sizes.

    :param recordings: The corpus's songs, as :func:`scan_recordings` returns them.
    :type recordings: list[pitchtrack.corpus.Recording]
    :return: Rows describing the corpus: its size, and how its ragas, talas and titles
        are spread across its songs.
    :rtype: list[tuple[str, str]]
    """
    titles = Counter(recording.title for recording in recordings)
    gigabytes = sum(recording.audio_path.stat().st_size
                    for recording in recordings) / 2 ** 30
    ragas = {recording.raga for recording in recordings if recording.raga}
    talas = {recording.tala for recording in recordings if recording.tala}
    return [
        ("songs", f"{len(recordings):,}"),
        ("audio", f"{gigabytes:,.2f} GiB"),
        ("ragas", f"{len(ragas):,}"),
        ("talas", f"{len(talas):,}"),
        ("titles shared by more than one song",
         f"{sum(1 for count in titles.values() if count > 1):,}"),
    ]


def main(argv=None):
    """Verify a corpus folder and describe it when it passes.

    :param argv: Command line arguments; ``None`` reads them from the command line.
    :type argv: list[str] or None
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="corpus folder: one subdirectory per song, "
                                     "each holding audio.mp3 and metadata.json")
    args = parser.parse_args(argv)

    songs, problems = check_corpus(args.data)
    errors = [problem for problem in problems if problem.severity == ERROR]
    if errors:
        return 1

    recordings = scan_recordings(args.data)
    show(summary_table(describe(recordings),
                       title=f"{args.data} — {songs} songs"))
    if problems:
        console.print(f"{args.data} is buildable, but {len(problems)} of its files "
                      f"are worth a look")
    return 0


if __name__ == "__main__":
    sys.exit(main())
