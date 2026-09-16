# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from libc.math cimport exp


cdef double boltzmann_pmf(Py_ssize_t k, double lambda_, Py_ssize_t n) noexcept:
    cdef double denominator
    if n <= 0 or k < 0 or k >= n:
        return 0.0
    denominator = 1.0 - exp(-lambda_ * n)
    if denominator == 0.0:
        return 0.0
    return (1.0 - exp(-lambda_)) * exp(-lambda_ * k) / denominator


def trough_probabilities(double[:] trough_heights, double[:] thresholds,
                          double boltzmann_parameter, double[:] beta_probs,
                          double no_trough_prob):
    cdef Py_ssize_t n_troughs = trough_heights.shape[0]
    cdef Py_ssize_t n_thresholds = beta_probs.shape[0]
    cdef double[:] probs = np.zeros(n_troughs, dtype=np.float64)
    cdef Py_ssize_t[:] below_count = np.zeros(n_thresholds, dtype=np.intp)

    cdef Py_ssize_t i, j
    cdef bint below
    cdef Py_ssize_t position
    cdef double prior, global_min_height
    cdef Py_ssize_t global_min_index
    cdef Py_ssize_t n_thresholds_below_min

    for j in range(n_thresholds):
        for i in range(n_troughs):
            if trough_heights[i] < thresholds[j + 1]:
                below_count[j] += 1

    for i in range(n_troughs):
        for j in range(n_thresholds):
            below = trough_heights[i] < thresholds[j + 1]
            if not below:
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

    cdef double boost = 0.0
    for j in range(n_thresholds_below_min):
        boost += beta_probs[j]
    probs[global_min_index] += no_trough_prob * boost

    return np.asarray(probs)
