import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def mel_filterbank(sample_rate, n_mels, n_fft, f_min, f_max):
    n_freqs = n_fft // 2 + 1
    mel_min = 2595 * math.log10(1 + f_min / 700)
    mel_max = 2595 * math.log10(1 + f_max / 700)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    filters = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        if f_center > f_left:
            filters[m - 1, f_left:f_center] = torch.arange(f_left, f_center).float() - f_left.float()
            filters[m - 1, f_left:f_center] /= (f_center - f_left).float()

        if f_right > f_center:
            filters[m - 1, f_center:f_right] = f_right.float() - torch.arange(f_center, f_right).float()
            filters[m - 1, f_center:f_right] /= (f_right - f_center).float()

    return filters


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512, f_min=60, f_max=7000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("mel_fb", mel_filterbank(sample_rate, n_mels, n_fft, f_min, f_max))

    def forward(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        spec = torch.stft(audio, self.n_fft, self.hop_length, window=self.window, return_complex=True)
        spec = spec.abs() ** 2
        mel = torch.matmul(self.mel_fb, spec)
        mel = torch.log(mel.clamp(min=1e-10))
        return mel


class SpectrogramEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)
