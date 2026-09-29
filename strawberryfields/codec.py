"""Reading the packed pitch track the database stores.

The ``pitch_track`` column holds a contour as a self-describing blob instead of an
array of numbers: eight bytes a frame would dominate the database, so the contour is
packed before it is stored and taken apart again on the way out. This module does the
taking apart.

A contour is a sequence of pitch values in cents on a uniform time grid, with ``nan``
marking the frames that carry no pitch. The blob begins with a thirteen-byte header:

.. list-table::
   :header-rows: 1

   * - Field
     - Size
     - Value
   * - ``magic``
     - 4 bytes
     - ``b"SFPT"``, the format's tag
   * - ``version``
     - 1 byte
     - ``2``; ``1`` is still readable, see :func:`decode_contour`
   * - ``frames``
     - 4 bytes
     - unsigned count of pitch frames
   * - ``hop_microseconds``
     - 4 bytes
     - unsigned spacing of the frames, in microseconds
   * - ``payload``
     - remainder
     - deflate stream of the delta-coded samples: variable-length integers from
       version 2, sixteen-bit words from version 1

The payload is read back in the order the writer applied the algorithms:

1. :func:`zlib.decompress` inflates the stream — the same call the writer packed it
   with;
2. the seven-bit groups are reassembled into values and the zigzag fold comes off,
   giving one difference per frame;
3. the differences are summed modulo ``2**16``, which restores the whole cents;
4. :func:`dequantise`, the helper in this module, turns whole cents into a contour.

Nothing but the standard library and :mod:`numpy` is involved, so a contour packed by
the writer and a contour read here describe exactly the same values.

:func:`decode_contour` is what the application calls on every row of ``tracks``.

"""

import struct
import zlib

import numpy as np

__all__ = [
    "MAGIC",
    "VERSION",
    "SUPPORTED_VERSIONS",
    "UNVOICED",
    "HEADER",
    "quantise",
    "dequantise",
    "decode_contour",
]

MAGIC = b"SFPT"

VERSION = 2

SUPPORTED_VERSIONS = (1, 2)

UNVOICED = -32768

HEADER = struct.Struct("<4sBII")

_MODULUS = 1 << 16


def quantise(contour):
    """Round a contour to whole cents, one ``int16`` per frame.

    A whole cent is an error of at most half a cent against a pitch-class grid fifty
    cents wide, and ``-32768`` is reserved for a frame that carries no pitch, so
    silence travels as ``nan`` instead of being dropped.

    :param contour: Pitch contour in cents, ``nan`` where the frame is unvoiced.
    :type contour: numpy.ndarray
    :return: One whole cent per frame, :data:`UNVOICED` where there was no pitch.
    :rtype: numpy.ndarray
    """
    values = np.ascontiguousarray(contour, dtype=np.float64).reshape(-1)
    unvoiced = ~np.isfinite(values)
    rounded = np.rint(np.where(unvoiced, 0.0, values))
    rounded = np.clip(rounded, UNVOICED + 1, np.iinfo(np.int16).max)
    rounded[unvoiced] = UNVOICED
    return np.ascontiguousarray(rounded, dtype=np.int16)


def dequantise(samples):
    """Read whole cents back into a contour, ``nan`` where there was no pitch.

    :param samples: Whole cents, :data:`UNVOICED` where the frame was unvoiced.
    :type samples: numpy.ndarray
    :return: The contour in cents as floating point values.
    :rtype: numpy.ndarray
    """
    packed = np.ascontiguousarray(samples, dtype=np.int16).reshape(-1)
    values = packed.astype(np.float64)
    values[packed == UNVOICED] = np.nan
    return values


def decode_contour(blob):
    """Unpack a stored contour back into cents on its own time grid.

    Both layouts of the format are read: version 1 stored its deltas as sixteen-bit
    words, version 2 stores them as variable-length integers, so a row written before
    the format changed still reads back unchanged.

    :param blob: The packed contour, header and payload together.
    :type blob: bytes
    :return: ``(contour, hop_seconds)`` — the contour in cents with ``nan`` where the
        frame is unvoiced, and the spacing of its frames in seconds.
    :rtype: tuple[numpy.ndarray, float]
    :raises ValueError: if the blob is not a packed pitch track, if its version is not
        one this module knows how to read, or if its payload does not inflate.
    """
    magic, version, frames, hop_microseconds = HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a packed pitch track: magic {magic!r}")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"packed pitch track version {version}, "
                         f"expected one of {SUPPORTED_VERSIONS}")

    try:
        payload = zlib.decompress(blob[HEADER.size:])
    except zlib.error as error:
        raise ValueError(f"packed pitch track payload does not inflate: {error}") from error

    differences = (_decode_varints(payload, frames) if version == 2
                   else _decode_words(payload, frames))
    if differences.size != frames:
        raise ValueError(f"packed pitch track holds {differences.size} frames, "
                         f"header says {frames}")

    samples = np.cumsum(differences, dtype=np.uint64) % _MODULUS
    samples = samples.astype(np.uint16).view(np.int16)
    values = dequantise(samples)
    return values, hop_microseconds / 1e6


def _decode_varints(data, frames):
    """Read a stream of seven-bit groups back as the differences it holds.

    The zigzag fold comes off here too: each value is turned back into its signed
    form and wrapped to sixteen bits, which is how the writer stored it.

    :param data: The inflated payload.
    :type data: bytes
    :param frames: How many values the header promises.
    :type frames: int
    :return: One difference per frame, wrapped to sixteen bits.
    :rtype: numpy.ndarray
    :raises ValueError: if the payload is truncated, malformed, or longer than the
        frame count asks for.
    """
    differences = np.empty(frames, dtype=np.uint16)
    position = 0
    for index in range(frames):
        value, shift = 0, 0
        while True:
            if position >= len(data):
                raise ValueError(f"packed pitch track ends after {index} of {frames} frames")
            byte = data[position]
            position += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
            if shift > 63:
                raise ValueError("packed pitch track holds an oversized integer")
        signed = (value >> 1) ^ -(value & 1)
        differences[index] = signed & (_MODULUS - 1)

    if position != len(data):
        raise ValueError(f"packed pitch track holds {len(data) - position} "
                         f"trailing payload bytes")
    return differences


def _decode_words(payload, frames):
    """Read the sixteen-bit differences of the version 1 layout.

    :param payload: The inflated payload.
    :type payload: bytes
    :param frames: How many values the header promises.
    :type frames: int
    :return: One difference per frame.
    :rtype: numpy.ndarray
    :raises ValueError: if the payload is not a whole number of sixteen-bit words.
    """
    if len(payload) % np.dtype(np.uint16).itemsize:
        raise ValueError(f"packed pitch track payload of {len(payload)} bytes "
                         f"is not a whole number of sixteen-bit differences")
    return np.frombuffer(payload, dtype=np.uint16)
