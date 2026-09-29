# cython: language_level=3, cdivision=True
cimport cython
import numpy as np
import scipy.signal
import scipy.stats
from libc.math cimport exp, fabs, INFINITY
from numpy.lib.stride_tricks import as_strided

TINY = np.finfo(np.float64).tiny


cdef double boltzmann_pmf(Py_ssize_t k, double lambda_, Py_ssize_t n) noexcept:
    cdef double denominator
    if n <= 0 or k < 0 or k >= n:
        return 0.0
    denominator = 1.0 - exp(-lambda_ * n)
    if denominator == 0.0:
        return 0.0
    return (1.0 - exp(-lambda_)) * exp(-lambda_ * k) / denominator


@cython.boundscheck(False)
@cython.wraparound(False)
def trough_probabilities(double[:] trough_heights, double[:] thresholds,
                         double boltzmann_parameter, double[:] beta_probs,
                         double no_trough_prob):
    cdef Py_ssize_t n_troughs = trough_heights.shape[0]
    cdef Py_ssize_t n_thresholds = beta_probs.shape[0]
    cdef double[:] probs = np.zeros(n_troughs, dtype=np.float64)
    cdef Py_ssize_t[:] below_count = np.zeros(n_thresholds, dtype=np.intp)

    cdef Py_ssize_t i, j, k, position
    cdef double prior, global_min_height, boost
    cdef Py_ssize_t global_min_index, n_thresholds_below_min

    for j in range(n_thresholds):
        for i in range(n_troughs):
            if trough_heights[i] < thresholds[j + 1]:
                below_count[j] += 1

    for i in range(n_troughs):
        for j in range(n_thresholds):
            if not trough_heights[i] < thresholds[j + 1]:
                continue
            position = 0
            for k in range(i + 1):
                if trough_heights[k] < thresholds[j + 1]:
                    position += 1
            position -= 1
            prior = boltzmann_pmf(position, boltzmann_parameter, below_count[j])
            probs[i] += prior * beta_probs[j]

    global_min_index = 0
    global_min_height = trough_heights[0]
    for i in range(1, n_troughs):
        if trough_heights[i] < global_min_height:
            global_min_height = trough_heights[i]
            global_min_index = i

    n_thresholds_below_min = 0
    for j in range(n_thresholds):
        if not (trough_heights[global_min_index] < thresholds[j + 1]):
            n_thresholds_below_min += 1

    boost = 0.0
    for j in range(n_thresholds_below_min):
        boost += beta_probs[j]
    probs[global_min_index] += no_trough_prob * boost

    return np.asarray(probs)


@cython.boundscheck(False)
@cython.wraparound(False)
def cumulative_mean_normalized_difference(double[:, :] frames, Py_ssize_t min_period,
                                          Py_ssize_t max_period):
    cdef Py_ssize_t frame_length = frames.shape[0]
    cdef Py_ssize_t n_frames = frames.shape[1]
    cdef Py_ssize_t n_lags = max_period - min_period + 1
    cdef double[:] difference = np.empty(max_period + 1, dtype=np.float64)
    cdef double[:, :] cmnd = np.empty((n_lags, n_frames), dtype=np.float64)
    cdef Py_ssize_t t, lag, j
    cdef double delta, accumulator, running_sum, mean, shifted

    for t in range(n_frames):
        difference[0] = 0.0
        for lag in range(1, max_period + 1):
            accumulator = 0.0
            for j in range(frame_length):
                shifted = frames[j + lag, t] if j + lag < frame_length else 0.0
                delta = frames[j, t] - shifted
                accumulator += delta * delta
            difference[lag] = accumulator

        running_sum = 0.0
        for lag in range(1, max_period + 1):
            running_sum += difference[lag]
            if lag >= min_period:
                mean = running_sum / lag
                if mean == 0.0:
                    mean = 1e-12
                cmnd[lag - min_period, t] = difference[lag] / mean

    return np.asarray(cmnd)


@cython.boundscheck(False)
@cython.wraparound(False)
def parabolic_interpolation(double[:, :] cmnd):
    cdef Py_ssize_t n_lags = cmnd.shape[0]
    cdef Py_ssize_t n_frames = cmnd.shape[1]
    cdef double[:, :] shifts = np.zeros((n_lags, n_frames), dtype=np.float64)
    cdef Py_ssize_t lag, t
    cdef double a, b

    for t in range(n_frames):
        for lag in range(1, n_lags - 1):
            a = cmnd[lag + 1, t] + cmnd[lag - 1, t] - 2.0 * cmnd[lag, t]
            b = (cmnd[lag + 1, t] - cmnd[lag - 1, t]) / 2.0
            if fabs(b) < fabs(a):
                shifts[lag, t] = -b / a

    return np.asarray(shifts)


@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi_core(double[:, :] log_prob, double[:, :] log_trans, double[:] log_p_init,
                 Py_ssize_t[:] pred_offsets, Py_ssize_t[:] pred_indices, bint use_threshold):
    cdef Py_ssize_t n_steps = log_prob.shape[0]
    cdef Py_ssize_t n_states = log_prob.shape[1]
    cdef unsigned short[:] state = np.zeros(n_steps, dtype=np.uint16)
    cdef double[:, :] value = np.zeros((n_steps, n_states), dtype=np.float64)
    cdef unsigned short[:, :] ptr = np.zeros((n_steps, n_states), dtype=np.uint16)
    cdef Py_ssize_t t, j, k, idx
    cdef double best_cost, cost
    cdef double best_last

    for j in range(n_states):
        value[0, j] = log_prob[0, j] + log_p_init[j]

    if use_threshold:
        for t in range(1, n_steps):
            for j in range(n_states):
                best_cost = -INFINITY
                for idx in range(pred_offsets[j], pred_offsets[j + 1]):
                    k = pred_indices[idx]
                    cost = value[t - 1, k] + log_trans[k, j]
                    if cost > best_cost:
                        ptr[t, j] = k
                        best_cost = cost
                value[t, j] = log_prob[t, j] + best_cost
    else:
        for t in range(1, n_steps):
            for j in range(n_states):
                best_cost = -INFINITY
                for k in range(n_states):
                    cost = value[t - 1, k] + log_trans[k, j]
                    if cost > best_cost:
                        ptr[t, j] = k
                        best_cost = cost
                value[t, j] = log_prob[t, j] + best_cost

    k = 0
    best_last = value[n_steps - 1, 0]
    for j in range(1, n_states):
        if value[n_steps - 1, j] > best_last:
            best_last = value[n_steps - 1, j]
            k = j
    state[n_steps - 1] = k

    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]

    return np.asarray(state), np.array([value[n_steps - 1, state[n_steps - 1]]])

def valid_audio(y):
    if not isinstance(y, np.ndarray):
        raise TypeError("Audio data must be of type numpy.ndarray")
    if not np.issubdtype(y.dtype, np.floating):
        raise ValueError("Audio data must be floating-point")
    if y.ndim != 1:
        raise ValueError("Only mono audio is supported")
    if not np.isfinite(y).all():
        raise ValueError("Audio buffer is not finite everywhere")
    return True


def frame(x, *, frame_length, hop_length):
    x = np.asarray(x)
    if x.shape[-1] < frame_length:
        raise ValueError(f"Input is too short (n={x.shape[-1]}) for frame_length={frame_length}")
    if hop_length < 1:
        raise ValueError(f"Invalid hop_length: {hop_length}")

    out_shape = (x.shape[-1] - frame_length + 1, frame_length)
    out_strides = (x.strides[-1], x.strides[-1])
    xw = as_strided(x, strides=out_strides, shape=out_shape, writeable=False)
    return np.moveaxis(xw, -1, -2)[:, ::hop_length]


def localmin(x):
    lmin = np.zeros(x.shape, dtype=bool)
    lmin[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    lmin[-1] = x[-1] < x[-2]
    return lmin


def pad_center(data, size):
    lpad = int((size - data.shape[-1]) // 2)
    return np.pad(data, (lpad, size - data.shape[-1] - lpad), mode="constant")


def transition_local(n_states, width, *, window="triangle", wrap=False):
    if not (int(n_states) == n_states and n_states > 1):
        raise ValueError(f"n_states={n_states} must be a positive integer > 1")

    width = np.asarray(width, dtype=int)
    if width.ndim == 0:
        width = np.tile(width, n_states)
    if width.shape != (n_states,):
        raise ValueError(f"width={width} must have length equal to n_states={n_states}")
    if np.any(width < 1):
        raise ValueError(f"width={width} must be at least 1")

    transition = np.zeros((n_states, n_states), dtype=np.float64)

    for i, width_i in enumerate(width):
        trans_row = pad_center(scipy.signal.get_window(window, width_i, fftbins=False), n_states)
        trans_row = np.roll(trans_row, n_states // 2 + i + 1)

        if not wrap:
            trans_row[min(n_states, i + width_i // 2 + 1):] = 0
            trans_row[:max(0, i - width_i // 2)] = 0

        transition[i] = trans_row

    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def transition_loop(n_states, prob):
    if not (int(n_states) == n_states and n_states > 1):
        raise ValueError(f"n_states={n_states} must be a positive integer > 1")

    transition = np.empty((n_states, n_states), dtype=np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 0:
        prob = np.tile(prob, n_states)
    if prob.shape != (n_states,):
        raise ValueError(f"prob={prob} must have length equal to n_states={n_states}")
    if np.any(prob < 0) or np.any(prob > 1):
        raise ValueError(f"prob={prob} must have values in the range [0, 1]")

    for i, prob_i in enumerate(prob):
        transition[i] = (1.0 - prob_i) / (n_states - 1)
        transition[i, i] = prob_i

    return transition


def viterbi(prob, transition, *, p_init=None, return_logp=False, transition_min_prob=None):
    n_states, _ = prob.shape[-2:]

    if transition.shape != (n_states, n_states):
        raise ValueError(f"transition.shape={transition.shape}, must be (n_states, n_states)={n_states, n_states}")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1):
        raise ValueError("Invalid transition matrix: must be non-negative and sum to 1 on each row.")
    if np.any(prob < 0) or np.any(prob > 1):
        raise ValueError("Invalid probability values: must be between 0 and 1.")

    epsilon = TINY

    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(1.0 / n_states)
    elif np.any(p_init < 0) or not np.allclose(p_init.sum(), 1) or p_init.shape != (n_states,):
        raise ValueError(f"Invalid initial state distribution: p_init={p_init}")

    log_trans = np.log(transition + epsilon)
    log_prob = np.log(prob + epsilon)
    log_p_init = np.log(p_init + epsilon)

    if transition_min_prob is not None and transition_min_prob > 0:
        log_trans_threshold = np.log(transition_min_prob + epsilon)
    elif transition_min_prob is None or transition_min_prob == 0:
        log_trans_threshold = -np.inf
    else:
        raise ValueError(f"Invalid transition_min_prob={transition_min_prob}, must be None or non-negative.")

    use_threshold = bool(np.isfinite(log_trans_threshold))
    offsets = np.zeros(n_states + 1, dtype=np.intp)
    indices = np.zeros(0, dtype=np.intp)
    if use_threshold:
        pred_states = []
        for j in range(n_states):
            possible_states = np.flatnonzero(log_trans[:, j] >= log_trans_threshold)
            if len(possible_states) == 0:
                raise ValueError(
                    f"Empty transition matrix detected for state {j} in Viterbi. "
                    "Try reducing your minimum transition probability threshold."
                )
            pred_states.append(possible_states)
        offsets[1:] = np.cumsum([len(p) for p in pred_states])
        indices = np.concatenate(pred_states).astype(np.intp)

    states, logp = viterbi_core(
        np.ascontiguousarray(log_prob.T), np.ascontiguousarray(log_trans),
        np.ascontiguousarray(log_p_init), offsets, indices, use_threshold,
    )

    if return_logp:
        return states, logp
    return states


def _check_yin_params(*, sr, fmax, fmin, frame_length):
    if fmax > sr / 2:
        raise ValueError(f"fmax={fmax:.3f} cannot exceed Nyquist frequency {sr / 2}")
    if fmin >= fmax:
        raise ValueError(f"fmin={fmin:.3f} must be less than fmax={fmax:.3f}")
    if fmin <= 0:
        raise ValueError(f"fmin={fmin:.3f} must be strictly positive")
    if sr / fmin >= frame_length - 1:
        raise ValueError(
            f"fmin={fmin:.3f} is too small for frame_length={frame_length} and sr={sr}. "
            f"Either increase to fmin={sr / (frame_length - 1):.3f} "
            f"or set frame_length={int(np.ceil(sr / fmin) + 1)} or higher."
        )


def _pyin_helper(yin_frames, parabolic_shifts, sr, thresholds, boltzmann_parameter, beta_probs,
                 no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone):
    yin_probs = np.zeros_like(yin_frames)

    for i in range(yin_frames.shape[1]):
        yin_frame = yin_frames[:, i]
        is_trough = localmin(yin_frame)
        is_trough[0] = yin_frame[0] < yin_frame[1]
        (trough_index,) = np.nonzero(is_trough)

        if len(trough_index) == 0:
            continue

        yin_probs[trough_index, i] = trough_probabilities(
            np.ascontiguousarray(yin_frame[trough_index]), thresholds,
            boltzmann_parameter, beta_probs, no_trough_prob,
        )

    yin_period, frame_index = np.nonzero(yin_probs)

    period_candidates = min_period + yin_period
    period_candidates = period_candidates + parabolic_shifts[yin_period, frame_index]
    f0_candidates = sr / period_candidates

    bin_index = 12 * n_bins_per_semitone * np.log2(f0_candidates / fmin)
    bin_index = np.clip(np.round(bin_index), 0, n_pitch_bins).astype(int)

    observation_probs = np.zeros((2 * n_pitch_bins, yin_frames.shape[1]))
    observation_probs[bin_index, frame_index] = yin_probs[yin_period, frame_index]

    voiced_prob = np.clip(np.sum(observation_probs[:n_pitch_bins, :], axis=0, keepdims=True), 0, 1)
    observation_probs[n_pitch_bins:, :] = (1 - voiced_prob) / n_pitch_bins

    return observation_probs, voiced_prob


def pyin(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, n_thresholds=100,
         beta_parameters=(2, 18), boltzmann_parameter=2, resolution=0.1,
         max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01, fill_na=np.nan,
         center=True, pad_mode="constant"):
    if fmin is None or fmax is None:
        raise ValueError('both "fmin" and "fmax" must be provided')

    _check_yin_params(sr=sr, fmax=fmax, fmin=fmin, frame_length=frame_length)

    if hop_length is None:
        hop_length = frame_length // 4

    valid_audio(y)

    if center:
        y = np.pad(y, (frame_length // 2, frame_length // 2), mode=pad_mode)

    y_frames = np.ascontiguousarray(frame(y, frame_length=frame_length, hop_length=hop_length),
                                    dtype=np.float64)

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)

    yin_frames = cumulative_mean_normalized_difference(y_frames, min_period, max_period)
    parabolic_shifts = parabolic_interpolation(yin_frames)

    thresholds = np.linspace(0, 1, n_thresholds + 1)
    beta_cdf = scipy.stats.beta.cdf(thresholds, beta_parameters[0], beta_parameters[1])
    beta_probs = np.diff(beta_cdf)

    n_bins_per_semitone = int(np.ceil(1.0 / resolution))
    n_pitch_bins = int(np.floor(12 * n_bins_per_semitone * np.log2(fmax / fmin))) + 1

    observation_probs, voiced_prob = _pyin_helper(
        yin_frames, parabolic_shifts, sr, thresholds, boltzmann_parameter, beta_probs,
        no_trough_prob, min_period, fmin, n_pitch_bins, n_bins_per_semitone,
    )

    max_semitones_per_frame = round(max_transition_rate * 12 * hop_length / sr)
    transition_width = max_semitones_per_frame * n_bins_per_semitone + 1
    transition = transition_local(n_pitch_bins, transition_width, window="triangle", wrap=False)

    t_switch = transition_loop(2, 1 - switch_prob)
    transition = np.kron(t_switch, transition)

    p_init = np.ones(2 * n_pitch_bins) / (2 * n_pitch_bins)

    states = viterbi(observation_probs, transition, p_init=p_init)

    freqs = fmin * 2 ** (np.arange(n_pitch_bins) / (12 * n_bins_per_semitone))
    f0 = freqs[states % n_pitch_bins]
    voiced_flag = states < n_pitch_bins

    if fill_na is not None:
        f0[~voiced_flag] = fill_na

    return f0, voiced_flag, voiced_prob[0]
