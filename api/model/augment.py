import torch
import torch.nn as nn
import numpy as np


def pink_noise(length):
    uneven = length % 2
    x = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
    s = np.sqrt(np.arange(len(x)) + 1.)
    y = np.real(np.fft.irfft(x / s, length))
    return y


def brown_noise(length):
    x = np.random.randn(length)
    return np.cumsum(x) - np.mean(np.cumsum(x))


class PitchShift(nn.Module):
    def __init__(self, max_shift_bins=4, n_segments=8, p=0.5):
        super().__init__()
        self.max_shift_bins = max_shift_bins
        self.n_segments = n_segments
        self.p = p

    def forward(self, spec):
        if torch.rand(1).item() > self.p:
            return spec

        batch, channels, freq, time = spec.shape
        out = spec.clone()

        seg_len = time // self.n_segments
        for i in range(self.n_segments):
            if torch.rand(1).item() < 0.5:
                continue

            start = i * seg_len
            end = min(start + seg_len, time)
            shift = torch.randint(-self.max_shift_bins, self.max_shift_bins + 1, (1,)).item()

            if shift > 0:
                out[:, :, shift:, start:end] = spec[:, :, :-shift, start:end]
                out[:, :, :shift, start:end] = 0
            elif shift < 0:
                out[:, :, :shift, start:end] = spec[:, :, -shift:, start:end]
                out[:, :, shift:, start:end] = 0

        return out


class TimeWarp(nn.Module):
    def __init__(self, max_warp_ratio=0.15, n_segments=8, p=0.5):
        super().__init__()
        self.max_warp_ratio = max_warp_ratio
        self.n_segments = n_segments
        self.p = p

    def forward(self, spec):
        if torch.rand(1).item() > self.p:
            return spec

        batch, channels, freq, time = spec.shape

        seg_len = time // self.n_segments
        boundaries = [0]
        for i in range(1, self.n_segments):
            offset = int(seg_len * self.max_warp_ratio * (2 * torch.rand(1).item() - 1))
            boundaries.append(boundaries[-1] + seg_len + offset)
        boundaries.append(time)

        out = torch.zeros_like(spec)
        for i in range(self.n_segments):
            src_start = int(boundaries[i] * time / boundaries[-1])
            src_end = int(boundaries[i + 1] * time / boundaries[-1])
            dst_start = i * seg_len
            dst_end = min(dst_start + seg_len, time)

            src_len = src_end - src_start
            dst_len = dst_end - dst_start

            if src_len <= 0 or dst_len <= 0:
                continue

            indices = torch.linspace(0, src_len - 1, dst_len).long()
            indices = indices.clamp(0, src_len - 1)

            out[:, :, :, dst_start:dst_end] = spec[:, :, :, src_start:src_end][:, :, :, indices]

        return out


class AmplitudeScale(nn.Module):
    def __init__(self, min_gain=0.5, max_gain=1.5, n_segments=8, p=0.5):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.n_segments = n_segments
        self.p = p

    def forward(self, spec):
        if torch.rand(1).item() > self.p:
            return spec

        batch, channels, freq, time = spec.shape
        out = spec.clone()

        seg_len = time // self.n_segments
        for i in range(self.n_segments):
            start = i * seg_len
            end = min(start + seg_len, time)
            gain = torch.empty(1).uniform_(self.min_gain, self.max_gain).item()
            out[:, :, :, start:end] = spec[:, :, :, start:end] * gain

        return out


class NoiseAdder(nn.Module):
    def __init__(self, pink_prob=0.5, brown_prob=0.3, snr_range=(10, 30)):
        super().__init__()
        self.pink_prob = pink_prob
        self.brown_prob = brown_prob
        self.snr_range = snr_range

    def forward(self, spec):
        if torch.rand(1).item() > (self.pink_prob + self.brown_prob):
            return spec

        batch, channels, freq, time = spec.shape
        out = spec.clone()

        noise = torch.zeros(batch, channels, freq, time, device=spec.device)
        for b in range(batch):
            noise_len = freq * time
            if torch.rand(1).item() < self.pink_prob:
                n = pink_noise(noise_len)
            else:
                n = brown_noise(noise_len)
            n = torch.tensor(n, dtype=torch.float32, device=spec.device).view(1, 1, freq, time)
            noise[b] = n

        snr_db = torch.empty(1).uniform_(self.snr_range[0], self.snr_range[1]).item()
        signal_power = spec.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = (signal_power / (snr_linear * noise_power + 1e-10)).sqrt()

        out = spec + noise * noise_scale * 0.3
        return out


class SpecAugmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pitch_shift = PitchShift(max_shift_bins=4, n_segments=8, p=0.7)
        self.time_warp = TimeWarp(max_warp_ratio=0.15, n_segments=8, p=0.6)
        self.amplitude_scale = AmplitudeScale(min_gain=0.5, max_gain=1.5, n_segments=8, p=0.5)
        self.noise_adder = NoiseAdder(pink_prob=0.5, brown_prob=0.3, snr_range=(10, 30))

    def forward(self, spec):
        spec = self.pitch_shift(spec)
        spec = self.time_warp(spec)
        spec = self.amplitude_scale(spec)
        spec = self.noise_adder(spec)
        return spec
