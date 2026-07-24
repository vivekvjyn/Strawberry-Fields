import sys
import os
import csv
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))

import soundfile as sf
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()

CHAD_DIR = Path(__file__).parent / ".cache/datasets/chad_hummings_subset"
CSV_PATH = CHAD_DIR / "metadata" / "dataset.csv"


def get_missing():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)

    disk_groups = {d.name for d in CHAD_DIR.iterdir() if d.is_dir()}

    missing = []
    for gid, recs in groups.items():
        if gid in disk_groups:
            continue
        for r in recs:
            if r["audio_type"] == "original":
                missing.append({
                    "group_id": gid,
                    "youtube_id": r["youtube_id"],
                    "interval": r["interval"],
                    "fragment_id": r["fragment_id"],
                })
                break
    return missing


def download_one(item):
    gid = item["group_id"]
    yid = item["youtube_id"]
    interval = item["interval"]
    frag_id = item["fragment_id"]

    group_dir = CHAD_DIR / gid
    group_dir.mkdir(parents=True, exist_ok=True)

    tmp_wav = group_dir / f"_tmp_{yid}.wav"
    url = f"https://www.youtube.com/watch?v={yid}"

    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
        "-o", str(group_dir / f"_tmp_{yid}.%(ext)s"),
        "--no-playlist", "--quiet", "--no-warnings",
        "--socket-timeout", "30", "--retries", "3",
        url,
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)

    for ext in ["webm", "opus", "m4a", "mp3", "ogg"]:
        tmp_other = group_dir / f"_tmp_{yid}.{ext}"
        if tmp_other.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_other), "-ar", "48000", "-ac", "1", str(tmp_wav)],
                capture_output=True, timeout=120,
            )
            tmp_other.unlink()
            break

    if not tmp_wav.exists():
        return False

    try:
        audio, sr = sf.read(str(tmp_wav))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        start_s = float(interval.strip("()").split(",")[0])
        end_s = float(interval.strip("()").split(",")[1])
        start = int(start_s * sr)
        end = int(end_s * sr)
        segment = audio[start:end]

        out_path = group_dir / f"original_{frag_id}_{yid}.wav"
        sf.write(str(out_path), segment, sr)
        tmp_wav.unlink()
        return True
    except Exception:
        if tmp_wav.exists():
            tmp_wav.unlink()
        return False


def main():
    missing = get_missing()
    console.print(f"[cyan]{len(missing)} originals to download[/]")

    if not missing:
        console.print("[green]All done[/]")
        return

    with Progress(
        TextColumn("[bold cyan]Downloading"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("download", total=len(missing))
        ok = 0
        fail = 0

        for item in missing:
            try:
                success = download_one(item)
                if success:
                    ok += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            progress.advance(task)

    console.print(f"[green]{ok} ok, [red]{fail} failed[/]")

    disk_groups = {d.name for d in CHAD_DIR.iterdir() if d.is_dir()}
    has_orig = sum(1 for g in disk_groups if list((CHAD_DIR / g).glob("original*.wav")))
    console.print(f"[green]{has_orig} groups with originals now[/]")


if __name__ == "__main__":
    main()
