"""Build the stored pitch track of every recording of a corpus.

    python scripts/build_pitch_tracks.py --dataset data/saraga --artifacts artifacts

Every recording goes through the same four steps:

1. :func:`pitchtrack.phrases.extract_pitch` tracks the predominant pitch out of the
   audio, and caches it under ``<artifacts>/pitch`` so a re-run starts past the slow
   part;
2. :func:`pitchtrack.phrases.remove_repeated_phrases` finds the phrases the performer
   states more than once and cuts every statement after the first of each;
3. :func:`pitchtrack.contour.contour_from_track` folds what is left onto the storage
   grid, in cents relative to the application's reference frequency;
4. :func:`pitchtrack.codec.encode_contour` packs the contour, and it is written with
   the recording's metadata to ``<artifacts>/tracks/<identifier>.npz``.

A recording whose artefact already exists is skipped, so an interrupted sweep resumes
where it stopped; ``--force`` rebuilds it from the audio instead.

Exit status is 0 when every recording was built, 1 when any of them failed.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pitchtrack import codec, contour, phrases
from pitchtrack.corpus import scan_recordings

MIN_VOICED_FRACTION = 0.05


def artefact_path(artifacts_root, identifier):
    """Where the built artefact of a recording is written.

    :param artifacts_root: Directory the build writes into.
    :type artifacts_root: str or pathlib.Path
    :param identifier: The recording's corpus folder name.
    :type identifier: str
    :return: Path of the recording's ``.npz`` file.
    :rtype: pathlib.Path
    """
    return Path(artifacts_root) / "tracks" / f"{identifier}.npz"


def build_recording(recording, artifacts_root, force=False):
    """Run the four steps over one recording and write its artefact.

    :param recording: The recording to build.
    :type recording: pitchtrack.corpus.Recording
    :param artifacts_root: Directory the artefacts and the pitch cache live in.
    :type artifacts_root: str or pathlib.Path
    :param force: Rebuild an artefact that already exists instead of skipping it.
    :type force: bool
    :return: What the build measured, for the sweep's summary.
    :rtype: dict
    :raises ValueError: if too little of the recording carries a pitch at all.
    """
    started = time.perf_counter()
    target = artefact_path(artifacts_root, recording.identifier)
    if target.exists() and not force:
        return {"identifier": recording.identifier, "outcome": "skipped", "seconds": 0.0}

    artifacts_root = Path(artifacts_root)
    times, f0 = phrases.extract_pitch(recording.audio_path, recording.identifier,
                                      artifacts_root / "pitch")
    voiced = phrases.voiced_fraction(f0)
    if voiced < MIN_VOICED_FRACTION:
        raise ValueError(f"only {voiced:.1%} of the pitch frames carry a pitch")

    kept_times, kept_f0, duplicates = phrases.remove_repeated_phrases(times, f0)
    contour_cents, grid_hop = contour.contour_from_track(kept_times, kept_f0)
    packed = codec.encode_contour(contour_cents, grid_hop)

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target,
                        title=recording.title,
                        raga=recording.raga,
                        tala=recording.tala,
                        pitch_track=np.frombuffer(packed, dtype=np.uint8))

    return {
        "identifier": recording.identifier,
        "outcome": "built",
        "frames": len(contour_cents),
        "bytes": len(packed),
        "voiced": voiced,
        "kept": 1.0 - float(np.mean(duplicates)) if len(duplicates) else 1.0,
        "seconds": time.perf_counter() - started,
    }


def _report(results):
    """Print what the sweep did, and return the number of recordings that failed."""
    built = [row for row in results if row["outcome"] == "built"]
    skipped = [row for row in results if row["outcome"] == "skipped"]
    failed = [row for row in results if row["outcome"] == "failed"]

    print(f"built {len(built)}, skipped {len(skipped)}, failed {len(failed)}")
    if built:
        total_bytes = sum(row["bytes"] for row in built)
        print(f"frames {sum(row['frames'] for row in built):,} "
              f"packed to {total_bytes / 1024:,.1f} KiB "
              f"({total_bytes / sum(row['frames'] for row in built):.2f} bytes a frame)")
        print(f"voiced {np.mean([row['voiced'] for row in built]):.1%}, "
              f"kept after phrase removal "
              f"{np.mean([row['kept'] for row in built]):.1%}")
    for row in failed:
        print(f"  {row['identifier']}: {row['error']}")
    return len(failed)


def main(argv=None):
    """Build every recording of a corpus.

    :param argv: Command line arguments; ``None`` reads them from the command line.
    :type argv: list[str] or None
    :return: Process exit status.
    :rtype: int
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/saraga",
                        help="corpus directory, one subdirectory per recording")
    parser.add_argument("--artifacts", default="artifacts",
                        help="directory to write the built pitch tracks into")
    parser.add_argument("--jobs", type=int, default=1,
                        help="recordings to build at once")
    parser.add_argument("--limit", type=int, default=None,
                        help="build only the first N recordings")
    parser.add_argument("--force", action="store_true",
                        help="rebuild recordings whose artefact already exists")
    args = parser.parse_args(argv)

    recordings = scan_recordings(args.dataset)
    if args.limit is not None:
        recordings = recordings[:args.limit]

    results = []
    if args.jobs <= 1:
        for recording in tqdm(recordings, desc="building"):
            results.append(_build_safely(recording, args.artifacts, args.force))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(build_recording, recording, args.artifacts, args.force):
                       recording for recording in recordings}
            for future in tqdm(as_completed(futures), total=len(futures), desc="building"):
                recording = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append({"identifier": recording.identifier, "outcome": "failed",
                                    "error": f"{type(error).__name__}: {error}"})

    return 1 if _report(results) else 0


def _build_safely(recording, artifacts_root, force):
    """Build one recording, turning a failure into a result the summary can print."""
    try:
        return build_recording(recording, artifacts_root, force)
    except Exception as error:
        return {"identifier": recording.identifier, "outcome": "failed",
                "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    sys.exit(main())
