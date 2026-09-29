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

The payload is read back in the order the writer applied the algorithms, the first one
from :mod:`zlib` — the same call the writer packed it with — and the rest from the
modules beside this file:

1. :mod:`zlib` inflates the stream;
2. :mod:`strawberryfields.varint` turns seven-bit groups into differences;
3. :mod:`strawberryfields.zigzag` folds the sign back out;
4. :mod:`strawberryfields.delta` sums the differences into whole cents;
5. :func:`dequantise`, the helper in this module, turns whole cents into a contour.

:func:`decode_contour` is what the application calls on every row of ``tracks``.

"""

import struct
import zlib

import numpy as np

from strawberryfields import delta, varint, zigzag

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

    if version == 2:
        differences = zigzag.decode(varint.decode(payload, frames))
    else:
        differences = varint.decode_words(payload, frames)
    if differences.size != frames:
        raise ValueError(f"packed pitch track holds {differences.size} frames, "
                         f"header says {frames}")

    samples = delta.cumulate(differences)
    values = dequantise(samples)
    return values, hop_microseconds / 1e6
