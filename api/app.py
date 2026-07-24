import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import json
import numpy as np
import torch
import yaml
import tempfile
import soundfile as sf
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import SiameseNetwork
from libs.utils import load_audio, segment_audio

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

ROOT = Path(__file__).parent.parent

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_dir = ROOT / cfg["checkpoint_dir"]
ckpt_path = checkpoint_dir / "best_model.pt"

model = None
db_embeddings = None
db_names = None
db_songs = None


def load_model():
    global model, db_embeddings, db_names, db_songs

    if not ckpt_path.exists():
        return False

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SiameseNetwork(
        embedding_dim=ckpt["config"]["embedding_dim"],
        sample_rate=ckpt["config"]["sample_rate"],
        n_mels=ckpt["config"]["n_mels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    db_dir = ROOT / cfg["dataset_dir"]
    if not db_dir.exists():
        return False

    db_embeddings = []
    db_names = []

    for group_dir in sorted(db_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        wavs = list(group_dir.rglob("*.wav"))
        if not wavs:
            continue

        embeddings = []
        for wav in wavs:
            audio = load_audio(wav, sr=cfg["sample_rate"])
            audio = segment_audio(audio, int(cfg["segment_length"] * cfg["sample_rate"]))
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            emb = model.embed(audio_tensor)
            embeddings.append(emb)

        mean_emb = torch.stack(embeddings).mean(dim=0)
        db_embeddings.append(mean_emb)
        db_names.append(group_dir.name)

    db_embeddings = torch.cat(db_embeddings, dim=0)
    return True


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "db_size": len(db_names) if db_names else 0,
    })


@app.route("/api/search", methods=["POST"])
def search():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        audio = load_audio(tmp_path, sr=cfg["sample_rate"])
        os.unlink(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load audio: {str(e)}"}), 400

    audio = segment_audio(audio, int(cfg["segment_length"] * cfg["sample_rate"]))
    query_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = model.embed(query_tensor)

    sims = torch.mm(query_emb, db_embeddings.t()).squeeze(0)
    top_k = min(10, len(db_names))
    top_sims, top_idx = sims.topk(top_k)

    results = []
    for i in range(top_k):
        results.append({
            "group_id": db_names[top_idx[i].item()],
            "score": round(top_sims[i].item(), 4),
            "rank": i + 1,
        })

    return jsonify({
        "results": results,
        "query_duration": round(len(audio) / cfg["sample_rate"], 2),
    })


if __name__ == "__main__":
    print("Loading model and indexing database...")
    if load_model():
        print(f"Ready. {len(db_names)} songs indexed.")
    else:
        print("Warning: Model or database not found. Run train.py and ingest.py first.")
    app.run(host="0.0.0.0", port=5000, debug=False)
