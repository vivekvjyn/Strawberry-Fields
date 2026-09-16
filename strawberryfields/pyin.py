import librosa
import numpy as np
import scipy.stats

from strawberryfields import pyin_core, yin_core


def _trough_probabilities(yin_frames, parabolic_shifts, sr, thresholds, boltzmann_parameter,
                           beta_probs, no_trough_prob, min_period, fmin, n_pitch_bins,
                           n_bins_per_semitone):
    yin_probs = np.zeros_like(yin_frames)

    for i in range(yin_frames.shape[1]):
        yin_frame = yin_frames[:, i]
        is_trough = librosa.util.localmin(yin_frame)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        trough_index = np.nonzero(is_trough)[0]

        if len(trough_index) == 0:
            continue

        trough_heights = yin_frame[trough_index]
        probs = pyin_core.trough_probabilities(
            trough_heights, thresholds, boltzmann_parameter, beta_probs, no_trough_prob,
        )

        yin_probs[trough_index, i] = probs

    yin_period, frame_index = np.nonzero(yin_probs)

    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    return observation_probs, voiced_prob


def pyin(y, sr, fmin, fmax, frame_length=2048, hop_length=None,
         n_thresholds=100, beta_parameters=(2, 18), boltzmann_parameter=2,
         resolution=0.1, max_transition_rate=35.92, switch_prob=0.01,
         no_trough_prob=0.01, fill_na=np.nan):
    if hop_length is None:
        hop_length = frame_length // 4

    padding = frame_length // 2
    y_padded = np.pad(y, padding, mode="constant")
    y_frames = librosa.util.frame(y_padded, frame_length=frame_length, hop_length=hop_length)
    y_frames = np.ascontiguousarray(y_frames, dtype=np.float64)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)

    yin_frames = yin_core.cumulative_mean_normalized_difference(y_frames, min_period, max_period)
    parabolic_shifts = yin_core.parabolic_interpolation(np.ascontiguousarray(yin_frames))

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    observation_probs, voiced_prob = _trough_probabilities(
        yin_frames, parabolic_shifts, sr, thresholds, boltzmann_parameter,
        beta_probs, no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone,
    )

    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    transition = librosa.sequence.transition_local(
        n_pitch_bins, transition_width, window="triangle", wrap=False
    )

    t_switch = librosa.sequence.transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)

    p_init = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)

    states = librosa.sequence.viterbi(observation_probs, transition, p_init=p_init)

    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins

    if fill_na is not None:
        f0 = f0.copy()
        f0[~voiced_flag] = fill_na

    return f0, voiced_flag, voiced_prob
