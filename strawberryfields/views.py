import tempfile
from pathlib import Path

import psycopg2
import psycopg2.extras
from flask import Blueprint, current_app, render_template, request
from rich.console import Console
from rich.table import Table

from strawberryfields import utils

bp = Blueprint("search", __name__)


@bp.route("/")
def index():
    """Render the search page.

    :return: The upload form.
    :rtype: str
    """
    return render_template("index.html")


@bp.route("/api/search", methods=["POST"])
def search():
    """Match an uploaded recording against the stored catalogue.

    The posted audio is decoded, turned into a pitch-class profile and compared
    against every stored contour; the metadata of the best match is printed as a
    table. A request carrying no audio, audio that fails to decode, or audio
    shorter than a second renders the results page with no track.

    :return: The results page for the best-matching track, or with no track set.
    :rtype: str
    """
    if "audio" not in request.files:
        return render_template("results.html", track=None)

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename or "query.webm").suffix or ".webm"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            y, sr = utils.load_audio(tmp.name)
    except Exception:
        return render_template("results.html", track=None)

    if len(y) < sr:
        return render_template("results.html", track=None)

    pitch_config = current_app.config["PITCH"]
    cents = utils.contour_from_audio(y, sr, pitch_config)
    query_profile = utils.to_pitch_class_profile(
        cents, pitch_config["n_classes"], pitch_config["sigma_cents"])

    contours = utils.get_track_contours(current_app.config["DATABASE_URL"])
    best_id, _ = utils.best_match(query_profile, contours, pitch_config)

    conn = psycopg2.connect(current_app.config["DATABASE_URL"])
    try:
        track = None
        if best_id is not None:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT title, raga, tala FROM tracks WHERE id = %s",
                    (best_id,),
                )
                track = cur.fetchone()

            if track:
                table = Table(title="Best match")
                table.add_column("Field", style="bold cyan")
                table.add_column("Value")
                table.add_row("Title", track["title"] or "")
                table.add_row("Raga", track["raga"] or "")
                table.add_row("Tala", track["tala"] or "")
                Console().print(table)
    finally:
        conn.close()

    return render_template("results.html", track=track)
