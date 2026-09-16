# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from libc.math cimport fabs


def cumulative_mean_normalized_difference(double[:, :] frames, Py_ssize_t min_period, Py_ssize_t max_period):
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
