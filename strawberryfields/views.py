import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from strawberryfields import pitch
from strawberryfields.alignment import search as dtw_search

bp = Blueprint("search", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/api/health")
def health():
    return jsonify({"status": "ok", "indexed_tracks": len(current_app.config["TRACKS"])})


@bp.route("/api/search", methods=["POST"])
def search():
    tracks = current_app.config["TRACKS"]
    if not tracks:
        return jsonify({"error": "No tracks in the database yet. Run ./scripts/setup.sh first."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename or "query.webm").suffix or ".webm"
    sample_rate = current_app.config["PITCH"]["sample_rate"]

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            audio_file.save(tmp.name)
            y = pitch.load_audio(tmp.name, sample_rate)
    except Exception as exc:
        return jsonify({"error": f"Failed to read audio: {exc}"}), 400

    if len(y) < sample_rate:
        return jsonify({"error": "Recording too short - hum for at least a second"}), 400

    query_contour = pitch.contour_from_audio(y, sample_rate, current_app.config["PITCH"]["pitch"])
    top_k = current_app.config["PITCH"]["search"]["top_k"]
    matches = dtw_search(query_contour, ((tid, t["contour"]) for tid, t in tracks.items()), top_k)

    results = []
    for rank, (track_id, score) in enumerate(matches, start=1):
        track = tracks[track_id]
        results.append({
            "rank": rank,
            "score": round(score, 4),
            "title": track["title"],
            "artists": track["artists"],
            "raaga": track["raaga"],
            "taala": track["taala"],
            "work": track["work"],
            "concert": track["concert"],
        })

    return jsonify({"results": results})
