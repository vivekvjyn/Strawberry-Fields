# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
from libc.math cimport fabs, floor, INFINITY


cdef Py_ssize_t round_half_to_even(double value):
    cdef Py_ssize_t floor_value = <Py_ssize_t> floor(value)
    cdef double fraction = value - floor_value

    if fraction < 0.5:
        return floor_value
    if fraction > 0.5:
        return floor_value + 1
    if floor_value % 2 == 0:
        return floor_value
    return floor_value + 1


def cost_matrix(double[:] query, double[:] reference):
    cdef Py_ssize_t n = query.shape[0]
    cdef Py_ssize_t m = reference.shape[0]
    cdef double[:, :] cost = np.empty((n, m), dtype=np.float64)
    cdef Py_ssize_t i, j

    for i in range(n):
        for j in range(m):
            cost[i, j] = fabs(query[i] - reference[j])

    return np.asarray(cost)


def apply_sakoe_chiba_band(double[:, :] cost, double band_rad):
    cdef Py_ssize_t n = cost.shape[0]
    cdef Py_ssize_t m = cost.shape[1]
    cdef Py_ssize_t radius = round_half_to_even(band_rad * min(n, m))
    cdef Py_ssize_t offset = n - m
    cdef Py_ssize_t upper_k, lower_k
    cdef Py_ssize_t i, j, diff

    if offset < 0:
        offset = -offset

    if n < m:
        upper_k = radius + offset
        lower_k = -radius
    else:
        upper_k = radius
        lower_k = -radius - offset

    for i in range(n):
        for j in range(m):
            diff = j - i
            if diff >= upper_k or diff <= lower_k:
                cost[i, j] = INFINITY


def accumulate_cost(double[:, :] cost, bint subseq):
    cdef Py_ssize_t n = cost.shape[0]
    cdef Py_ssize_t m = cost.shape[1]
    cdef double[:, :] accumulated = np.full((n + 1, m + 1), INFINITY, dtype=np.float64)
    cdef Py_ssize_t i, j
    cdef double diagonal, up, left, best

    if subseq:
        for j in range(m + 1):
            accumulated[0, j] = 0.0
    else:
        accumulated[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = accumulated[i - 1, j - 1]
            up = accumulated[i - 1, j]
            left = accumulated[i, j - 1]

            best = diagonal
            if up < best:
                best = up
            if left < best:
                best = left

            accumulated[i, j] = cost[i - 1, j - 1] + best

    return np.asarray(accumulated)


def backtrack_path(double[:, :] accumulated, bint subseq):
    cdef Py_ssize_t n = accumulated.shape[0] - 1
    cdef Py_ssize_t m = accumulated.shape[1] - 1
    cdef Py_ssize_t i = n
    cdef Py_ssize_t j
    cdef Py_ssize_t candidate
    cdef double diagonal, up, left, best
    cdef list path = []

    if subseq:
        j = 0
        best = accumulated[n, 0]
        for candidate in range(1, m + 1):
            if accumulated[n, candidate] < best:
                best = accumulated[n, candidate]
                j = candidate
    else:
        j = m

    path.append((i, j))

    while not (i == 0 and (subseq or j == 0)):
        diagonal = accumulated[i - 1, j - 1] if j > 0 else INFINITY
        up = accumulated[i - 1, j]
        left = accumulated[i, j - 1] if j > 0 else INFINITY

        best = diagonal
        step = (i - 1, j - 1)
        if up < best:
            best = up
            step = (i - 1, j)
        if left < best:
            best = left
            step = (i, j - 1)

        i, j = step
        path.append((i, j))

    path.reverse()
    return np.asarray(path, dtype=np.intp)


def dtw(double[:] query, double[:] reference, bint subseq=True,
        bint backtrack=False, bint global_constraints=False, double band_rad=0.25):
    cdef bint transposed = False

    if subseq and query.shape[0] > reference.shape[0]:
        query, reference = reference, query
        transposed = True

    cost = cost_matrix(query, reference)
    if global_constraints:
        apply_sakoe_chiba_band(cost, band_rad)

    accumulated = accumulate_cost(cost, subseq)

    if backtrack:
        path = backtrack_path(accumulated, subseq)
        path = path - 1
        if transposed:
            path = path[:, ::-1]
        return accumulated[1:, 1:], path

    return accumulated[1:, 1:]


def subsequence_dtw_cost(double[:] query, double[:] reference):
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return INFINITY

    cost = cost_matrix(query, reference)
    accumulated = accumulate_cost(cost, True)

    cdef Py_ssize_t n = query.shape[0]
    cdef Py_ssize_t m = reference.shape[0]
    cdef double min_cost = accumulated[n, 0]
    cdef Py_ssize_t j

    for j in range(1, m + 1):
        if accumulated[n, j] < min_cost:
            min_cost = accumulated[n, j]

    return min_cost / n
