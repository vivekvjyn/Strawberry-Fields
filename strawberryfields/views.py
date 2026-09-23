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
    return render_template("index.html")


@bp.route("/api/search", methods=["POST"])
def search():
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

    query_profile = utils.pitch_class_profile_from_audio(y, sr, current_app.config["PITCH"])

    contours = utils.get_track_contours(current_app.config["DATABASE_URL"])
    best_id, _ = utils.best_match(query_profile, contours, current_app.config["PITCH"])

    conn = psycopg2.connect(current_app.config["DATABASE_URL"])
    try:
        track = None
        if best_id is not None:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT title, raag, taal, laya, form FROM tracks WHERE id = %s",
                    (best_id,),
                )
                track = cur.fetchone()

            if track:
                table = Table(title="Best match")
                table.add_column("Field", style="bold cyan")
                table.add_column("Value")
                table.add_row("Title", track["title"] or "")
                table.add_row("Raag", track["raag"] or "")
                table.add_row("Taal", track["taal"] or "")
                table.add_row("Laya", track["laya"] or "")
                table.add_row("Form", track["form"] or "")
                Console().print(table)
    finally:
        conn.close()

    return render_template("results.html", track=track)
