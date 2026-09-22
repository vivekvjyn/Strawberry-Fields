# cython: language_level=3, cdivision=True
cimport cython
import numpy as np
from libc.math cimport exp


@cython.boundscheck(False)
@cython.wraparound(False)
def rasterise(double[:] contour, double bin_cents, double range_cents, double sigma_cents):
    cdef Py_ssize_t n_frames = contour.shape[0]
    cdef Py_ssize_t n_bins = <Py_ssize_t>(2 * range_cents / bin_cents) + 1
    cdef double sigma_bins = sigma_cents / bin_cents
    cdef Py_ssize_t radius = <Py_ssize_t>(4.0 * sigma_bins + 0.5)
    cdef Py_ssize_t kernel_size = 2 * radius + 1

    cdef double[:] kernel = np.empty(kernel_size, dtype=np.float64)
    cdef Py_ssize_t k
    cdef double weight, kernel_sum = 0.0
    for k in range(kernel_size):
        weight = exp(-0.5 * ((k - radius) / sigma_bins) ** 2)
        kernel[k] = weight
        kernel_sum += weight
    for k in range(kernel_size):
        kernel[k] /= kernel_sum

    image_np = np.zeros((n_bins, n_frames), dtype=np.float64)
    cdef double[:, :] image = image_np

    cdef Py_ssize_t t, b
    cdef double value

    for t in range(n_frames):
        value = contour[t]
        if value != value:  # NaN check
            continue
        b = <Py_ssize_t>((value + range_cents) / bin_cents + 0.5)
        if 0 <= b < n_bins:
            image[b, t] = 1.0

    blurred_np = np.zeros((n_bins, n_frames), dtype=np.float64)
    cdef double[:, :] blurred = blurred_np
    cdef Py_ssize_t src_bin

    for t in range(n_frames):
        for b in range(n_bins):
            if image[b, t] == 0.0:
                continue
            for k in range(kernel_size):
                src_bin = b + (k - radius)
                if 0 <= src_bin < n_bins:
                    blurred[src_bin, t] += kernel[k]

    cdef double col_sum
    for t in range(n_frames):
        col_sum = 0.0
        for b in range(n_bins):
            col_sum += blurred[b, t]
        if col_sum > 0.0:
            for b in range(n_bins):
                blurred[b, t] /= col_sum

    return blurred_np
