import tempfile
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from flask import Blueprint, current_app, jsonify, render_template, request

from strawberryfields import utils

bp = Blueprint("search", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/search", methods=["POST"])
def search():
    if "audio" not in request.files:
        return jsonify({"result": None})

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename or "query.webm").suffix or ".webm"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            y, sr = utils.load_audio(tmp.name)
    except Exception:
        return jsonify({"result": None})

    if len(y) < sr:
        return jsonify({"result": None})

    query_image = utils.salience_from_audio(y, sr, current_app.config["PITCH"])

    conn = psycopg2.connect(current_app.config["DATABASE_URL"])
    try:
        with conn.cursor(name="contours") as cur:
            cur.itersize = 50
            cur.execute("SELECT id, pitch_cents FROM tracks")
            contours = ((track_id, np.asarray(pitch_cents, dtype=np.float64))
                        for track_id, pitch_cents in cur)
            track_id = utils.best_match(query_image, contours, current_app.config["PITCH"])

        track = None
        if track_id is not None:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT title, artists, raag, taal, laya, form FROM tracks WHERE id = %s",
                    (track_id,),
                )
                track = cur.fetchone()
    finally:
        conn.close()

    return jsonify({"result": track})
