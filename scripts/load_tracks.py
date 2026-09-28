"""Replace the contents of the ``tracks`` table with the built pitch tracks.

    python scripts/load_tracks.py --artifacts artifacts

Every ``<artifacts>/tracks/*.npz`` written by ``scripts/build_pitch_tracks.py`` becomes
one row: the title, raga and tala it was built with, and the packed contour. The
credentials come from ``.env`` in the project root, so the same file points the
application and this loader at the same database.

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
from pitchtrack.repository import Track, count_tracks, replace_all


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
    parser.add_argument("--artifacts", default="artifacts",
                        help="directory the built pitch tracks were written to")
    parser.add_argument("--limit", type=int, default=None,
                        help="load only the first N recordings")
    args = parser.parse_args(argv)

    artefacts = sorted(Path(args.artifacts).glob("tracks/*.npz"))
    if args.limit is not None:
        artefacts = artefacts[:args.limit]
    if not artefacts:
        print(f"no built pitch tracks under {args.artifacts}")
        return 1

    started = time.perf_counter()
    tracks = [read_track(path) for path in artefacts]
    total_bytes = sum(len(track.pitch_track) for track in tracks)
    print(f"read {len(tracks)} recordings, "
          f"{total_bytes / 1024:,.1f} KiB of pitch tracks")

    connection_url = database_url()
    written = replace_all(tracks, connection_url)
    stored = count_tracks(connection_url)
    print(f"wrote {written} rows in {time.perf_counter() - started:,.1f}s; "
          f"the table now holds {stored}")
    return 0 if stored == len(tracks) else 1


if __name__ == "__main__":
    sys.exit(main())
