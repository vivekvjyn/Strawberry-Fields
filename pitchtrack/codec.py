"""The compact wire format of a stored pitch contour.

A contour is a sequence of pitch values in cents on a uniform time grid, with ``nan``
marking the frames that carry no pitch. Storing it as it stands would cost eight bytes
a frame and dominate the database, so this module packs it into a self-describing blob
that fits the ``pitch_track`` column:

* the samples are rounded to whole cents and kept as ``int16`` — two bytes, and an
  error of at most half a cent against a pitch-class grid that is fifty cents wide;
* an unvoiced frame is kept as the reserved value ``-32768`` rather than being dropped,
  so nothing about the shape of the track is lost;
* the samples are delta-coded, because a contour moves by a few cents from frame to
  frame and the differences are far more repetitive than the values;
* the result is deflated with :mod:`zlib`, which every Python has, so the application
  needs nothing beyond the standard library to read the column back.

Encoding is lossless apart from the rounding to whole cents, which cannot move a frame
across a pitch-class boundary on its own.

The blob begins with a thirteen-byte header:

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
     - ``1``
   * - ``frames``
     - 4 bytes
     - unsigned count of pitch frames
   * - ``hop_microseconds``
     - 4 bytes
     - unsigned spacing of the frames, in microseconds
   * - ``payload``
     - remainder
     - :mod:`zlib` stream of the delta-coded ``int16`` samples

Use :func:`encode_contour` to write a blob and :func:`decode_contour` to read one.
"""

import struct
import zlib

import numpy as np

__all__ = [
    "MAGIC",
    "VERSION",
    "UNVOICED",
    "encode_contour",
    "decode_contour",
]

MAGIC = b"SFPT"

VERSION = 1

UNVOICED = int(np.iinfo(np.int16).min)

_HEADER = struct.Struct("<4sBII")

_MODULUS = 1 << 16


def encode_contour(contour, hop_seconds):
    """Pack a pitch contour into the bytes stored in the ``pitch_track`` column.

    :param contour: Pitch contour in cents, ``nan`` where the frame is unvoiced.
    :type contour: numpy.ndarray
    :param hop_seconds: Spacing of the contour's frames, in seconds.
    :type hop_seconds: float
    :return: The packed contour, header included.
    :rtype: bytes
    :raises ValueError: if the contour holds no frames.
    """
    samples = _quantise(contour)
    if samples.size == 0:
        raise ValueError("cannot pack an empty contour")
    bits = samples.view(np.uint16)
    deltas = np.empty_like(bits)
    deltas[0] = bits[0]
    np.subtract(bits[1:], bits[:-1], out=deltas[1:])
    header = _HEADER.pack(MAGIC, VERSION, samples.size, int(round(hop_seconds * 1e6)))
    return header + zlib.compress(deltas.tobytes(), level=9)


def decode_contour(blob):
    """Unpack a stored contour back into cents on its own time grid.

    :param blob: Packed contour, as written by :func:`encode_contour`.
    :type blob: bytes
    :return: ``(contour, hop_seconds)`` — the contour in cents with ``nan`` where the
        frame is unvoiced, and the spacing of its frames in seconds.
    :rtype: tuple[numpy.ndarray, float]
    """
    magic, version, frames, hop_microseconds = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a packed pitch track: magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"packed pitch track version {version}, expected {VERSION}")

    deltas = np.frombuffer(zlib.decompress(blob[_HEADER.size:]), dtype=np.uint16)
    if deltas.size != frames:
        raise ValueError(f"packed pitch track holds {deltas.size} frames, header says {frames}")

    bits = np.cumsum(deltas, dtype=np.uint32) % _MODULUS
    samples = bits.astype(np.uint16).view(np.int16)
    contour = samples.astype(np.float64)
    contour[samples == UNVOICED] = np.nan
    return contour, hop_microseconds / 1e6


def _quantise(contour):
    """Round a contour to whole cents, reserving ``UNVOICED`` for the frames without one."""
    values = np.asarray(contour, dtype=np.float64)
    unvoiced = ~np.isfinite(values)
    rounded = np.rint(np.where(unvoiced, 0.0, values))
    rounded = np.clip(rounded, UNVOICED + 1, np.iinfo(np.int16).max)
    rounded[unvoiced] = UNVOICED
    return np.ascontiguousarray(rounded, dtype=np.int16)
