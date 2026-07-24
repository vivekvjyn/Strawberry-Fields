import os
import glob
import csv
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba
import psycopg2
import psycopg2.extras
from pathlib import Path
from dotenv import load_dotenv
from mido import MidiFile, tick2second
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.rule import Rule
from scipy.signal import resample_poly
from math import gcd

load_dotenv()

console = Console()

PLOT_DIR = Path(__file__).parent.parent.parent / ".cache/plots"
AUDIO_DIR = Path(__file__).parent.parent.parent / ".cache/audio"
PITCH_FLOOR = 36
PITCH_CEIL = 96
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_KEYS = {1, 3, 6, 8, 10}
AUDIO_SR = 44100

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    album TEXT,
    singer TEXT,
    composer TEXT,
    lyricist TEXT,
    link TEXT,
    melody_notes JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_songs_title ON songs (title);
"""


def get_db():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432"),
    )
    conn.autocommit = True
    return conn


def setup_schema():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute("SELECT COUNT(*) FROM songs")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def clear_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("TRUNCATE songs RESTART IDENTITY")
    cur.close()
    conn.close()


def insert_song(cur, title, album, singer, composer, lyricist, link, melody_notes):
    cur.execute(
        """INSERT INTO songs (title, album, singer, composer, lyricist, link, melody_notes)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (title, album, singer, composer, lyricist, link, json.dumps(melody_notes)),
    )


def get_all_songs(cur):
    cur.execute("SELECT id, title, album, singer, composer, lyricist, link, melody_notes FROM songs")
    cols = ["id", "title", "album", "singer", "composer", "lyricist", "link", "melody_notes"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def parse_midi_events(midi_path):
    try:
        mid = MidiFile(midi_path)
    except Exception:
        return []

    events = []
    for track in mid.tracks:
        abs_tick = 0
        abs_sec = 0.0
        tempo = 500000
        for msg in track:
            abs_sec += tick2second(msg.time, mid.ticks_per_beat, tempo)
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            if msg.type in ("note_on", "note_off"):
                events.append((abs_sec, msg.type, msg.note, msg.velocity))

    events.sort(key=lambda x: x[0])

    note_starts = {}
    notes = []

    for time, kind, note, vel in events:
        if kind == "note_on" and vel > 0:
            note_starts[note] = (time, vel)
        elif kind == "note_off" or (kind == "note_on" and vel == 0):
            if note in note_starts:
                start, velocity = note_starts.pop(note)
                if time > start:
                    notes.append([start, time, note, velocity])

    return notes


def extract_melody(notes):
    if not notes:
        return []

    notes_sorted = sorted(notes, key=lambda x: (x[0], -x[2]))

    t_min = notes_sorted[0][0]
    t_max = max(n[1] for n in notes_sorted)

    step = 0.01
    times = np.arange(t_min, t_max, step)

    melody_pitches = np.zeros(len(times))
    for i, t in enumerate(times):
        active = [(n[2], n[3]) for n in notes_sorted if n[0] <= t < n[1]]
        if active:
            melody_pitches[i] = max(active, key=lambda x: x[0])[0]

    melody_notes = []
    if melody_pitches[0] > 0:
        current_pitch = melody_pitches[0]
        start_t = times[0]

        for i in range(1, len(times)):
            if melody_pitches[i] != current_pitch:
                if current_pitch > 0:
                    melody_notes.append([round(start_t, 4), round(times[i], 4), int(current_pitch), 80])
                current_pitch = melody_pitches[i]
                start_t = times[i]

        if current_pitch > 0:
            melody_notes.append([round(start_t, 4), round(times[-1], 4), int(current_pitch), 80])

    return clean_melody(melody_notes)


def clean_melody(notes, min_duration=0.05, octave_threshold=12):
    if not notes:
        return []

    notes = sorted(notes, key=lambda x: x[0])

    notes = [n for n in notes if (n[1] - n[0]) >= min_duration]
    if not notes:
        return []

    pitches = [n[2] for n in notes]
    median_pitch = np.median(pitches)

    cleaned = []
    for i, note in enumerate(notes):
        start, end, pitch, vel = note
        duration = end - start

        if i > 0 and i < len(notes) - 1:
            prev_pitch = notes[i - 1][2]
            next_pitch = notes[i + 1][2]
            jump_prev = abs(pitch - prev_pitch)
            jump_next = abs(pitch - next_pitch)

            if jump_prev > octave_threshold and jump_next > octave_threshold and duration < 0.15:
                new_pitch = round((prev_pitch + next_pitch) / 2)
                note = [start, end, new_pitch, vel]

        cleaned.append(note)

    merged = [cleaned[0]]
    for note in cleaned[1:]:
        prev = merged[-1]
        if note[2] == prev[2] and note[0] - prev[1] < 0.03:
            merged[-1] = [prev[0], note[1], prev[2], max(prev[3], note[3])]
        else:
            merged.append(note)

    return merged


def melody_notes_to_pitches(melody_notes, sr=AUDIO_SR):
    if not melody_notes:
        return np.array([])

    total_time_s = max(n[1] for n in melody_notes)
    n_samples = int(total_time_s * sr) + 1
    pitches = np.zeros(n_samples)

    for start, end, pitch, vel in melody_notes:
        s = int(start * sr)
        e = int(end * sr)
        pitches[s:e] = pitch

    return pitches


def parse_maestro_csv(dataset_dir):
    csv_path = os.path.join(dataset_dir, "maestro-v3.0.0.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(dataset_dir, "maestro-v2.0.0.csv")

    metadata = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                full_key = os.path.splitext(row.get("midi_filename", ""))[0]
                stem_key = Path(full_key).name
                entry = {
                    "title": row.get("canonical_title", "Unknown"),
                    "composer": row.get("canonical_composer", "Unknown"),
                    "year": row.get("year", ""),
                }
                metadata[full_key] = entry
                metadata[stem_key] = entry
    return metadata


def find_midi_files(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, "**", "*.mid"), recursive=True)
    files += glob.glob(os.path.join(dataset_dir, "**", "*.midi"), recursive=True)
    return files


def note_label(midi_num):
    return f"{NOTE_NAMES[int(midi_num) % 12]}{int(midi_num) // 12 - 1}"


def midi_to_freq(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def melody_to_audio(melody_notes, sr=AUDIO_SR):
    if not melody_notes:
        return np.array([])

    total_time_s = max(n[1] for n in melody_notes)
    n_samples = int(total_time_s * sr) + 1
    audio = np.zeros(n_samples, dtype=np.float64)

    harmonics = [1.0, 0.5, 0.25, 0.125, 0.06, 0.03]
    attack_s = 0.01
    decay_s = 0.08
    sustain_level = 0.6
    release_s = 0.15

    for start, end, pitch, vel in melody_notes:
        freq = midi_to_freq(pitch)
        note_dur = end - start
        s = int(start * sr)
        e = int(end * sr)
        n_note = e - s
        if n_note <= 0:
            continue

        amp = (vel / 127.0) * 0.7
        t = np.arange(n_note) / sr

        waveform = np.zeros(n_note, dtype=np.float64)
        for h, gain in enumerate(harmonics, 1):
            waveform += gain * np.sin(2 * np.pi * freq * h * t)

        a_samples = min(int(attack_s * sr), n_note)
        d_samples = min(int(decay_s * sr), n_note - a_samples)
        r_samples = min(int(release_s * sr), n_note)

        env = np.ones(n_note)
        env[:a_samples] = np.linspace(0, 1, a_samples)
        if a_samples + d_samples < n_note:
            env[a_samples:a_samples + d_samples] = np.linspace(1, sustain_level, d_samples)
            env[a_samples + d_samples:] = sustain_level
        if r_samples > 0:
            env[-r_samples:] *= np.linspace(1, 0, r_samples)

        waveform *= env * amp
        audio[s:e] += waveform

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    return audio


def plot_pianoroll(melody_notes, title, save_path):
    if not melody_notes:
        return

    pitches = [n[2] for n in melody_notes]
    p_min = max(min(pitches) - 1, PITCH_FLOOR)
    p_max = min(max(pitches) + 1, PITCH_CEIL)

    total_time_s = max(n[1] for n in melody_notes)
    fig_width = max(12, min(40, total_time_s * 0.5))

    ticks = list(range(p_min, p_max + 1))
    white_ticks = [p for p in ticks if p % 12 not in BLACK_KEYS]
    black_ticks = [p for p in ticks if p % 12 in BLACK_KEYS]

    fig, ax = plt.subplots(figsize=(fig_width, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    colors = [
        "#4a90d9", "#5b9bd5", "#6fa3e0", "#7eb3ea",
        "#3a7cc2", "#2e6aab", "#4a90d9", "#5b9bd5",
    ]

    for start, end, pitch, velocity in melody_notes:
        if pitch < p_min or pitch > p_max:
            continue
        duration = end - start
        x = start
        color_idx = pitch % len(colors)
        alpha = 0.4 + 0.6 * (velocity / 127)
        ax.barh(pitch, duration, left=x, height=0.7,
                color=to_rgba(colors[color_idx], alpha),
                edgecolor=to_rgba(colors[color_idx], 0.9),
                linewidth=0.3)

    ax.set_yticks(white_ticks)
    ax.set_yticklabels([note_label(p) for p in white_ticks], fontsize=7, fontfamily="monospace")
    ax.set_yticks(black_ticks, minor=True)
    ax.set_yticklabels([note_label(p) for p in black_ticks], fontsize=6, fontfamily="monospace", minor=True)

    for label in ax.get_yticklabels():
        if label.get_text() and " #" in label.get_text():
            label.set_color("#888888")

    ax.set_xlabel("Time (s)", fontsize=9, color="#333333")
    ax.set_ylabel("Pitch", fontsize=9, color="#333333")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="x", colors="#555555", labelsize=8)
    ax.tick_params(axis="y", colors="#555555", length=0, labelsize=7)
    ax.tick_params(axis="y", which="minor", colors="#555555", length=0, labelsize=6)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.5)

    ax.grid(axis="x", color="#e0e0e0", linewidth=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.text(0.02, 0.96, title, fontsize=10, color="#222222", fontweight="bold",
             fontfamily="monospace", va="top", transform=fig.transFigure)
    fig.text(0.02, 0.92, f"{len(melody_notes)} notes | {total_time_s:.1f}s | {p_min}–{p_max} MIDI",
             fontsize=8, color="#888888", fontfamily="monospace", va="top", transform=fig.transFigure)

    plt.tight_layout(pad=1.5)
    fig.savefig(save_path, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def save_artifacts(name, melody_notes):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    plot_pianoroll(melody_notes, name, PLOT_DIR / f"{name}_pianoroll.png")

    audio = melody_to_audio(melody_notes)
    sf.write(AUDIO_DIR / f"{name}.wav", audio, AUDIO_SR)


def melody_to_pitch_vector(melody_notes, hop_ms=10):
    if not melody_notes:
        return np.array([])

    total_time_s = max(n[1] for n in melody_notes)
    hop_s = hop_ms / 1000.0
    n_steps = int(total_time_s / hop_s) + 1
    pitches = np.zeros(n_steps)
    times = np.arange(n_steps) * hop_s

    for start, end, pitch, vel in melody_notes:
        mask = (times >= start) & (times < end)
        pitches[mask] = pitch

    return pitches


def dtw_match(candidates, query, window_size, hop_length):
    min_cost = np.inf
    result = None

    for doc in candidates:
        melody = json.loads(doc["melody_notes"]) if isinstance(doc["melody_notes"], str) else doc["melody_notes"]
        vector = melody_to_pitch_vector(melody)
        if len(vector) < window_size:
            continue
        cost = np.inf

        for l in range(0, len(vector) - window_size, hop_length):
            Y = vector[l: l + window_size]
            Y = Y - np.mean(Y)
            D = librosa.sequence.dtw(query, Y, subseq=True, global_constraints=True, band_rad=0.1, backtrack=False)
            cost = min(D[-1, -1], cost)

        if cost < min_cost:
            min_cost = cost
            result = doc

    return result


def hz_to_midi(frequencies):
    note_nums = librosa.hz_to_midi(frequencies)
    note_nums = note_nums - np.mean(note_nums)
    return note_nums


def parse_request(request):
    if hasattr(request, "POST"):
        arr = np.fromstring(request.POST["signal"], sep=",")
        arr = arr / np.max(abs(arr))
        fs = int(request.POST["sample-rate"])
    else:
        arr = np.fromstring(request.form["signal"], sep=",")
        arr = arr / np.max(abs(arr))
        fs = int(request.form["sample-rate"])
    return arr, fs


def pyin_f0(y, sr):
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("E2"), fmax=librosa.note_to_hz("C5"), sr=sr
    )

    nan_mask = np.isnan(f0)
    for i in range(1, len(f0)):
        if nan_mask[i]:
            f0[i] = f0[i - 1]

    f0 = f0[~np.isnan(f0)]
    f0 = f0[0::4]
    return f0


def load_audio(path, sr=16000):
    audio, orig_sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        g = gcd(orig_sr, sr)
        audio = resample_poly(audio, sr // g, orig_sr // g).astype(np.float32)
    return audio


def segment_audio(audio, n_samples):
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    start = (len(audio) - n_samples) // 2
    return audio[start:start + n_samples]


def load_config():
    import yaml
    root = Path(__file__).parent.parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return root, cfg


def load_model(cfg, device):
    import torch
    from model import SiameseNetwork
    root, _ = load_config()
    ckpt_path = root / cfg["checkpoint_dir"] / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SiameseNetwork(
        embedding_dim=ckpt["config"]["embedding_dim"],
        sample_rate=ckpt["config"]["sample_rate"],
        n_mels=ckpt["config"]["n_mels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_audio_chunk(path, sample_rate, n_samples):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return audio


def load_audio_sliding(path, sample_rate, n_samples, hop=None):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    if hop is None:
        hop = n_samples // 2
    chunks = []
    for start in range(0, len(audio) - n_samples + 1, hop):
        chunks.append(audio[start:start + n_samples])
    if not chunks:
        chunks.append(audio[:n_samples])
    return np.stack(chunks)
