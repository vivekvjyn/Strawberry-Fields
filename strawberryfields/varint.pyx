"""Varint, reading back the groups the stored pitch track is packed as.

A varint spends seven bits of every byte on the number and the eighth on whether the
number continues, so an ordinary difference costs one byte and a large one costs a few
more. The stream the header points at is a run of these, one per frame, read into an
array. The version 1 layout skipped the variable step and stored plain sixteen-bit
words, so :func:`decode_words` reads that layout still.

A payload that runs short, or runs long, raises :class:`ValueError` rather than
returning a contour that does not match its header.

"""

import numpy as np

__all__ = ["decode", "decode_words"]


def decode(payload, frames):
    """Read a stream of varints into the unsigned values it holds.

    :param payload: The bytes after the header: one varint per frame.
    :type payload: bytes
    :param frames: How many frames the header says the stream carries.
    :type frames: int
    :return: One unsigned value per frame, wrapped to sixteen bits.
    :rtype: numpy.ndarray
    :raises ValueError: if the stream ends early or does not hold the frames the
        header promises.

    """
    cdef:
        const unsigned char[:] data = payload
        Py_ssize_t index = 0
        Py_ssize_t count = len(payload)
        Py_ssize_t position
        int byte
        unsigned long long value
        unsigned int shift
        unsigned short[:] view
    out = np.empty(frames, dtype=np.uint16)
    view = out
    for position in range(frames):
        value = 0
        shift = 0
        while True:
            if index >= count:
                raise ValueError(
                    f"packed pitch track ends after {position} of {frames} frames")
            byte = data[index]
            index += 1
            value |= <unsigned long long>(byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
            if shift > 63:
                raise ValueError(
                    f"packed pitch track holds an oversized integer at frame "
                    f"{position}")
        view[position] = <unsigned short>(value & 0xFFFF)
    if index != count:
        raise ValueError(
            f"packed pitch track holds {count - index} trailing payload bytes")
    return out


def decode_words(payload, frames):
    """Read the sixteen-bit differences of the version 1 layout.

    :param payload: The inflated payload.
    :type payload: bytes
    :param frames: How many frames the header promises.
    :type frames: int
    :return: One difference per frame, wrapped to sixteen bits.
    :rtype: numpy.ndarray
    :raises ValueError: if the payload is not a whole number of sixteen-bit words or
        does not hold the frames the header promises.

    """
    if len(payload) % 2:
        raise ValueError(
            f"packed pitch track payload of {len(payload)} bytes is not a whole "
            f"number of sixteen-bit differences")
    if len(payload) // 2 != frames:
        raise ValueError(
            f"packed pitch track holds {len(payload) // 2} frames, header says "
            f"{frames}")
    return np.frombuffer(payload, dtype=np.uint16)
