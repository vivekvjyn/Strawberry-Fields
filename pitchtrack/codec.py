"""The compact on-disk and in-database form of a pitch contour.

A contour is a sequence of pitch values in cents on a uniform time grid, with ``nan``
marking the frames that carry no pitch. Storing it as it stands would cost eight bytes
a frame and dominate the database, so this module packs it into a self-describing blob
that fits the ``pitch_track`` column.

The stages are the shape every well-established codec has — remove what the consumer
cannot use, then remove the redundancy, then entropy-code what is left:

1. **round to whole cents** and keep the samples as ``int16``: two bytes a frame, and
   an error of at most half a cent against a pitch-class grid that is fifty cents wide;
2. **reserve** ``-32768`` for a frame without a pitch, so unvoiced frames survive as
   ``nan`` instead of being dropped;
3. **delta code** — each frame minus the one before it, wrapped to sixteen bits, since
   a contour moves only a few cents from frame to frame;
4. **zigzag + variable-length integers** — a signed difference of a few cents becomes
   one byte instead of two. This is the encoding Apache Parquet calls
   ``DELTA_BINARY_PACKED`` and Protocol Buffers uses for its ``sint`` fields;
5. **deflate** the result with :mod:`zlib`, which every Python has, so the application
   needs nothing beyond the standard library to read the column back.

Nothing but :mod:`numpy` and :mod:`zlib` is involved, and every stage is reversible: the
only loss anywhere in the format is the rounding in step 1, which cannot move a frame
across a pitch-class boundary on its own. A packed contour measures about **0.87 bytes
a frame**, against eight bytes a frame before packing and 1.14 for the layout of
version 1.

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
     - ``2``; ``1`` is still readable, see :func:`decode_contour`
   * - ``frames``
     - 4 bytes
     - unsigned count of pitch frames
   * - ``hop_microseconds``
     - 4 bytes
     - unsigned spacing of the frames, in microseconds
   * - ``payload``
     - remainder
     - :mod:`zlib` stream of the delta-coded samples: variable-length integers from
       version 2, sixteen-bit words from version 1

Use :func:`encode_contour` to write a blob and :func:`decode_contour` to read one.
"""

import struct
import zlib

import numpy as np

__all__ = [
    "MAGIC",
    "VERSION",
    "SUPPORTED_VERSIONS",
    "UNVOICED",
    "encode_contour",
    "decode_contour",
]

MAGIC = b"SFPT"

VERSION = 2

SUPPORTED_VERSIONS = (1, 2)

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

    payload = zlib.compress(_encode_varints(_zigzag(_differences(samples))), level=9)
    header = _HEADER.pack(MAGIC, VERSION, samples.size, int(round(hop_seconds * 1e6)))
    return header + payload


def decode_contour(blob):
    """Unpack a stored contour back into cents on its own time grid.

    Both layouts of the format are read: version 1 stored its deltas as sixteen-bit
    words, version 2 stores them as variable-length integers, so a blob written before
    the format changed still reads back unchanged.

    :param blob: Packed contour, as written by :func:`encode_contour`.
    :type blob: bytes
    :return: ``(contour, hop_seconds)`` — the contour in cents with ``nan`` where the
        frame is unvoiced, and the spacing of its frames in seconds.
    :rtype: tuple[numpy.ndarray, float]
    :raises ValueError: if the blob is not a packed pitch track, if its version is not
        one this module knows how to read, or if its payload does not inflate.
    """
    magic, version, frames, hop_microseconds = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"not a packed pitch track: magic {magic!r}")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"packed pitch track version {version}, "
                         f"expected one of {SUPPORTED_VERSIONS}")

    try:
        payload = zlib.decompress(blob[_HEADER.size:])
    except zlib.error as error:
        raise ValueError(f"packed pitch track payload does not inflate: {error}") from error

    differences = (_decode_varints(payload, frames) if version == 2
                   else _decode_words(payload, frames))
    if differences.size != frames:
        raise ValueError(f"packed pitch track holds {differences.size} frames, "
                         f"header says {frames}")

    samples = np.cumsum(differences, dtype=np.uint64) % _MODULUS
    samples = samples.astype(np.uint16).view(np.int16)
    values = samples.astype(np.float64)
    values[samples == UNVOICED] = np.nan
    return values, hop_microseconds / 1e6


def _quantise(contour):
    """Round a contour to whole cents, reserving ``UNVOICED`` for the frames without one."""
    values = np.asarray(contour, dtype=np.float64)
    unvoiced = ~np.isfinite(values)
    rounded = np.rint(np.where(unvoiced, 0.0, values))
    rounded = np.clip(rounded, UNVOICED + 1, np.iinfo(np.int16).max)
    rounded[unvoiced] = UNVOICED
    return np.ascontiguousarray(rounded, dtype=np.int16)


def _differences(samples):
    """Subtract each sample from the one before it, wrapped to signed sixteen bits.

    :param samples: Quantised contour.
    :type samples: numpy.ndarray
    :return: One value per frame — the first frame's own value, then the differences;
        every value fits in an ``int16``.
    :rtype: numpy.ndarray
    """
    bits = samples.view(np.uint16).astype(np.int64)
    differences = np.empty(bits.size, dtype=np.int64)
    differences[0] = bits[0]
    differences[1:] = bits[1:] - bits[:-1]
    wrapped = np.mod(differences, _MODULUS)
    return np.where(wrapped > _MODULUS // 2, wrapped - _MODULUS, wrapped)


def _zigzag(differences):
    """Map signed differences to unsigned ones, so small magnitudes stay small.

    :param differences: Signed differences from :func:`_differences`.
    :type differences: numpy.ndarray
    :return: Each value ``2n`` for ``n >= 0`` and ``-2n - 1`` for ``n < 0``.
    :rtype: numpy.ndarray
    """
    values = differences.astype(np.int64)
    return ((values << 1) ^ (values >> 63)).astype(np.uint64)


def _encode_varints(values):
    """Write unsigned values as seven bits a byte, the last byte of each unflagged.

    :param values: Unsigned values to write.
    :type values: numpy.ndarray
    :return: The encoded bytes, values back to back.
    :rtype: bytes
    """
    out = bytearray()
    for value in values.tolist():
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                break
    return bytes(out)


def _decode_varints(data, frames):
    """Read back what :func:`_encode_varints` wrote, as sixteen-bit differences.

    :param data: The encoded bytes.
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
