"""Zigzag, the fold that keeps the small differences small.

A signed difference spends its top bit on a sign it rarely needs, and a run of
differences taken backwards would therefore cost full thirty-two bit words. Zigzag
folds the sign into the low bit instead: 0 stays 0, -1 becomes 1, 1 becomes 2, -2
becomes 3, and the near-zero differences a pitch contour is made of stay near zero.
The reader folds the stream back the other way.

The fold is exact both ways: every unsigned value has one signed image and no two
values share one, so nothing is lost on either side of it.

"""

import numpy as np

__all__ = ["decode"]


def decode(values):
    """Unfold unsigned values back into the signed differences they stand for.

    :param values: The unsigned values to fold back.
    :type values: numpy.ndarray
    :return: The signed differences, one per frame.
    :rtype: numpy.ndarray

    """
    cdef:
        const unsigned long long[:] data
        long long[:] out_view
        Py_ssize_t index
        Py_ssize_t count
        unsigned long long value
        long long sign
    data = np.ascontiguousarray(values, dtype=np.uint64)
    count = data.shape[0]
    out = np.empty(count, dtype=np.int64)
    out_view = out
    for index in range(count):
        value = data[index]
        sign = 0 - (value & 1)
        out_view[index] = (value >> 1) ^ sign
    return out
