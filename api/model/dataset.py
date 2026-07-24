import random
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio(path, sample_rate):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = resample(audio, sr, sample_rate)
    return audio


def random_segment(audio, n_samples):
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    start = random.randint(0, max(0, len(audio) - n_samples))
    return audio[start:start + n_samples]


def augment(audio, sample_rate, n_samples):
    if random.random() < 0.5:
        shift = random.uniform(-0.1, 0.1)
        audio = np.roll(audio, int(shift * sample_rate))
    if random.random() < 0.3:
        gain = random.uniform(0.7, 1.3)
        audio = audio * gain
    if random.random() < 0.2:
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
    if random.random() < 0.3:
        speed = random.uniform(0.9, 1.1)
        indices = np.round(np.arange(0, len(audio), speed)).astype(int)
        indices = indices[indices < len(audio)]
        audio = audio[indices]
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        else:
            audio = audio[:n_samples]
    return audio


class TripletDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000, segment_length=4.0):
        self.sample_rate = sample_rate
        self.n_samples = int(segment_length * sample_rate)
        self.groups = {}

        for group_dir in sorted(Path(data_dir).iterdir()):
            if not group_dir.is_dir():
                continue
            gid = group_dir.name
            corpus = None
            keys = []
            for w in group_dir.rglob("*.wav"):
                if w.stem.startswith("original"):
                    corpus = w
                else:
                    keys.append(w)
            if corpus and keys:
                self.groups[gid] = {"corpus": corpus, "keys": keys}

        self.gids = list(self.groups.keys())

    def __len__(self):
        return len(self.gids)

    def __getitem__(self, idx):
        gid = self.gids[idx]
        data = self.groups[gid]

        corpus = load_audio(data["corpus"], self.sample_rate)
        corpus = random_segment(corpus, self.n_samples)

        pos_key = load_audio(random.choice(data["keys"]), self.sample_rate)
        pos_key = augment(pos_key, self.sample_rate, self.n_samples)
        pos_key = random_segment(pos_key, self.n_samples)

        neg_gid = random.choice(self.gids)
        while neg_gid == gid:
            neg_gid = random.choice(self.gids)
        neg_key = load_audio(random.choice(self.groups[neg_gid]["keys"]), self.sample_rate)
        neg_key = augment(neg_key, self.sample_rate, self.n_samples)
        neg_key = random_segment(neg_key, self.n_samples)

        return (
            torch.tensor(corpus, dtype=torch.float32),
            torch.tensor(pos_key, dtype=torch.float32),
            torch.tensor(neg_key, dtype=torch.float32),
        )
