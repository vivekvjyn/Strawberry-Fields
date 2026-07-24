import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SpectrogramEncoder, LogMelSpectrogram
from .augment import SpecAugmentor


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128, sample_rate=16000, n_mels=128):
        super().__init__()
        self.encoder = SpectrogramEncoder(embedding_dim)
        self.mel_spec = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
            f_min=60,
            f_max=7000,
        )
        self.augmentor = SpecAugmentor()

    def forward(self, audio, augment=False):
        spec = self.mel_spec(audio)
        spec = spec.unsqueeze(1)
        if augment:
            spec = self.augmentor(spec)
        return self.encoder(spec)

    def embed(self, audio):
        with torch.no_grad():
            return self.forward(audio, augment=False)
