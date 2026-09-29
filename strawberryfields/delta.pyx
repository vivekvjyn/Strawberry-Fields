"""Delta, running the differences of a stored pitch track back into a contour.

What the column holds is not the cents themselves but the step from one frame to the
next, so a contour that wanders slowly stores small numbers throughout while its actual
values climb. Running the steps back is a prefix sum: the first frame is a difference
against nothing and stands as itself, and every frame after it adds one more step.

The loop is the shape of that run: two states, the running total and the step to add
to it. The total wraps in sixteen bits, the same ring the writer packed, so a contour
encoded from any source reads back to the same cent it held.

"""

import numpy as np

__all__ = ["cumulate"]


def cumulate(differences):
    """Run differences back into the sequence they were taken from.

    :param differences: The step from one frame to the next.
    :type differences: numpy.ndarray
    :return: The frame values the differences add up to, wrapped to sixteen bits.
    :rtype: numpy.ndarray

    """
    cdef:
        Py_ssize_t index
        Py_ssize_t count
        long long total = 0
        long long step
    cdef long long[:] data = np.ascontiguousarray(differences, dtype=np.int64)
    count = data.shape[0]
    out = np.empty(count, dtype=np.int16)
    cdef short[:] view = out
    for index in range(count):
        step = data[index]
        total = (total + step) & 0xFFFF
        if total >= 0x8000:
            total -= 0x10000
        view[index] = <short>total
    return out
