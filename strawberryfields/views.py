import tempfile
from pathlib import Path

import psycopg2
import psycopg2.extras
from flask import Blueprint, current_app, render_template, request
from rich.console import Console
from rich.table import Table

from strawberryfields import utils
from strawberryfields.pyin import pyin

bp = Blueprint("search", __name__)


@bp.route("/")
def index():
    """Render the search page.

    :return: The upload form.
    :rtype: str
    """
    return render_template("index.html")


@bp.route("/result", methods=["POST"])
def search():
    """Match an uploaded recording against the stored catalogue.

    The posted audio is decoded, turned into a pitch-class profile and compared
    against the stored contours as they are read from the database one row at a
    time; the three cheapest matches and their metadata are printed as a
    table. A request carrying no audio, audio that fails to decode, or audio
    shorter than a second renders the results page with no tracks.

    :return: The results page for the best-matching tracks, or with no tracks set.
    :rtype: str
    """
    if "audio" not in request.files:
        return render_template("results.html", tracks=[])

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename or "query.webm").suffix or ".webm"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            y, sr = utils.load_audio(tmp.name)
    except Exception:
        return render_template("results.html", tracks=[])

    if len(y) < sr:
        return render_template("results.html", tracks=[])

    pitch_config = current_app.config["PITCH"]
    f0, _, _ = pyin(
        y, fmin=pitch_config["fmin"], fmax=pitch_config["fmax"], sr=sr,
        frame_length=pitch_config["frame_length"],
        hop_length=round(pitch_config["hop_seconds"] * sr),
    )
    cents = utils.hz_to_cents(f0, pitch_config["ref_hz"])
    query_profile = utils.pitch_class_profile(
        cents, pitch_config["n_classes"], pitch_config["sigma_cents"])

    contours = utils.get_tracks(current_app.config["DATABASE_URL"])
    matches = utils.top_matches(query_profile, contours, pitch_config)

    tracks = []
    if matches:
        conn = psycopg2.connect(current_app.config["DATABASE_URL"])
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, title, raga, tala FROM tracks WHERE id = ANY(%s)",
                    ([track_id for track_id, _ in matches],),
                )
                rows = {row["id"]: row for row in cur.fetchall()}

            for rank, (track_id, cost) in enumerate(matches, start=1):
                row = rows.get(track_id)
                if row is not None:
                    tracks.append({"rank": rank, "cost": cost, **row})

            if tracks:
                table = Table(title="Top matches")
                table.add_column("#", style="bold cyan", justify="right")
                table.add_column("Title", style="bold")
                table.add_column("Raga")
                table.add_column("Tala")
                table.add_column("Cost", justify="right")
                for entry in tracks:
                    table.add_row(str(entry["rank"]), entry["title"] or "",
                                  entry["raga"] or "", entry["tala"] or "",
                                  f"{entry['cost']:.4f}")
                Console().print(table)
        finally:
            conn.close()

    return render_template("results.html", tracks=tracks)
