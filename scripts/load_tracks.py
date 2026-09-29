"""Replace the contents of the ``tracks`` table with the built pitch tracks.

    python scripts/load_tracks.py

Every ``.cache/tracks/*.npz`` written by ``scripts/build_pitch_tracks.py`` becomes
one row: the title, raga and tala it was built with, and the packed contour. The
credentials come from ``.env`` in the project root, so the same file points the
application and this loader at the same database.

Every artefact is read and checked — its contour is unpacked, so a truncated file is
found here rather than as a corrupt row later — before the table is touched. One
unreadable artefact stops the load, listed with the others that could not be read, and
leaves the table as it was.

The load is one transaction — see :func:`pitchtrack.repository.replace_all` — so a
failure leaves the table holding the corpus it held before.

Exit status is 0 when the table holds the recordings it was given, 1 otherwise.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pitchtrack import codec
from pitchtrack.config import database_url
from pitchtrack.console import (console, problems_table, progress, show,
                                summary_table)
from pitchtrack.corpus import ERROR, Problem
from pitchtrack.repository import Track, count_tracks, replace_all

CACHE_DIR = ".cache"


def read_track(artefact_path):
    """Read one built artefact back into a row of the ``tracks`` table.

    :param artefact_path: Path of the ``.npz`` file written by the build.
    :type artefact_path: str or pathlib.Path
    :return: The recording's row, with its contour unpacked only as far as checking it.
    :rtype: pitchtrack.repository.Track
    """
    with np.load(artefact_path) as artefact:
        packed = artefact["pitch_track"].tobytes()
        codec.decode_contour(packed)
        return Track(title=str(artefact["title"]),
                     raga=str(artefact["raga"]),
                     tala=str(artefact["tala"]),
                     pitch_track=packed)


def main(argv=None):
    """Load every built artefact into the database.

    :param argv: Command line arguments; ``None`` reads them from the command line.
    :type argv: list[str] or None
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="load only the first N recordings")
    args = parser.parse_args(argv)

    artefacts = sorted(Path(CACHE_DIR).glob("tracks/*.npz"))
    if args.limit is not None:
        artefacts = artefacts[:args.limit]
    if not artefacts:
        console.print(f"[red]no built pitch tracks under {CACHE_DIR}[/red]")
        return 1

    started = time.perf_counter()
    tracks = []
    problems = []
    with progress() as bar:
        task = bar.add_task("reading artefacts", total=len(artefacts))
        for artefact_path in artefacts:
            try:
                tracks.append(read_track(artefact_path))
            except Exception as error:
                problems.append(Problem(artefact_path.stem, "artefact", ERROR,
                                        f"{type(error).__name__}: {error}"))
            bar.advance(task)

    if problems:
        show(problems_table(problems,
                            title=f"{len(problems)} artefacts could not be read"))
        console.print(f"[red]nothing was loaded[/red] — rebuild "
                      f"{CACHE_DIR} with --force, the table was not touched")
        return 1

    total_bytes = sum(len(track.pitch_track) for track in tracks)
    show(summary_table([("artefacts read", f"{len(tracks):,}"),
                        ("pitch tracks", f"{total_bytes / 1024:,.1f} KiB"),
                        ("seconds spent reading", f"{time.perf_counter() - started:,.1f}")],
                       title="reading"))

    connection_url = database_url()
    with console.status("storing rows in the tracks table"):
        written = replace_all(tracks, connection_url)
        stored = count_tracks(connection_url)

    show(summary_table([("rows written", f"{written:,}"),
                        ("rows in the table", f"{stored:,}"),
                        ("seconds storing", f"{time.perf_counter() - started:,.1f}")],
                       title="tracks"))
    return 0 if stored == len(tracks) else 1


if __name__ == "__main__":
    sys.exit(main())
