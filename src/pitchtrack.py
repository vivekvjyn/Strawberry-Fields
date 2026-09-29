"""The compact on-disk form of a pitch contour.

A contour is a sequence of pitch values in cents on a uniform time grid, with ``nan``
marking the frames that carry no pitch. Stored as it stands it costs eight bytes a
frame, so this module packs it into a self-describing blob instead — the same layout,
the same stages and the same :mod:`zlib` call the ``pitch_track`` column is written
with, so a contour packed here and a contour packed there are the same bytes.

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
5. **deflate** the result with :mod:`zlib` at level 9, so nothing beyond the standard
   library and :mod:`numpy` is needed to read a packed contour back.

Nothing but :mod:`numpy` and :mod:`zlib` is involved, and every stage is reversible: the
only loss anywhere in the format is the rounding in step 1, which cannot move a frame
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

Use :func:`encode_contour` to pack a contour and :func:`decode_contour` to read one
back; the individual stages are exported as well, so a pack can be taken apart by hand.
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
    "MODULUS",
    "quantise",
    "dequantise",
    "differences",
    "cumulate",
    "zigzag",
    "unzigzag",
    "encode_varints",
    "decode_varints",
    "decode_words",
    "encode_contour",
    "decode_contour",
]

MAGIC = b"SFPT"

VERSION = 2

SUPPORTED_VERSIONS = (1, 2)

UNVOICED = int(np.iinfo(np.int16).min)

HEADER = struct.Struct("<4sBII")

MODULUS = 1 << 16


def quantise(contour):
    """Round a contour to whole cents, reserving :data:`UNVOICED` for the frames
    without one.

    :param contour: Pitch contour in cents, ``nan`` where the frame is unvoiced.
    :type contour: numpy.ndarray
    :return: One whole cent per frame, :data:`UNVOICED` where there was no pitch.
    :rtype: numpy.ndarray
    """
    values = np.asarray(contour, dtype=np.float64)
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


def differences(samples):
    """Subtract each sample from the one before it, wrapped to signed sixteen bits.

    :param samples: Quantised contour.
    :type samples: numpy.ndarray
    :return: One value per frame — the first frame's own value, then the differences;
        every value fits in an ``int16``.
    :rtype: numpy.ndarray
    """
    bits = np.ascontiguousarray(samples, dtype=np.int16).view(np.uint16).astype(np.int64)
    signed = np.empty(bits.size, dtype=np.int64)
    signed[0] = bits[0]
    signed[1:] = bits[1:] - bits[:-1]
    wrapped = np.mod(signed, MODULUS)
    return np.where(wrapped > MODULUS // 2, wrapped - MODULUS, wrapped)


def cumulate(differences):
    """Sum the differences back into whole cents, wrapping to sixteen bits.

    :param differences: Signed differences, as :func:`differences` writes them.
    :type differences: numpy.ndarray
    :return: One whole cent per frame.
    :rtype: numpy.ndarray
    """
    values = np.asarray(differences, dtype=np.int64)
    samples = np.cumsum(values, dtype=np.int64) % MODULUS
    return samples.astype(np.uint16).view(np.int16)


def zigzag(differences):
    """Map signed differences to unsigned ones, so small magnitudes stay small.

    :param differences: Signed differences from :func:`differences`.
    :type differences: numpy.ndarray
    :return: Each value ``2n`` for ``n >= 0`` and ``-2n - 1`` for ``n < 0``.
    :rtype: numpy.ndarray
    """
    values = np.asarray(differences, dtype=np.int64)
    return ((values << 1) ^ (values >> 63)).astype(np.uint64)


def unzigzag(values):
    """Turn what :func:`zigzag` wrote back into signed differences.

    :param values: Unsigned differences from :func:`zigzag` or :func:`decode_varints`.
    :type values: numpy.ndarray
    :return: The signed differences they stood for.
    :rtype: numpy.ndarray
    """
    unsigned = np.asarray(values, dtype=np.int64)
    return (unsigned >> 1) ^ -(unsigned & 1)


def encode_varints(values):
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


def decode_varints(data, frames):
    """Read back what :func:`encode_varints` wrote.

    :param data: The encoded bytes.
    :type data: bytes
    :param frames: How many values the header promises.
    :type frames: int
    :return: The unsigned values, one per frame.
    :rtype: numpy.ndarray
    :raises ValueError: if the payload is truncated, malformed, or longer than the
        frame count asks for.
    """
    values = np.zeros(frames, dtype=np.uint64)
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
        values[index] = value & 0xFFFFFFFFFFFFFFFF

    if position != len(data):
        raise ValueError(f"packed pitch track holds {len(data) - position} "
                         f"trailing payload bytes")
    return values


def encode_contour(contour, hop_seconds):
    """Pack a pitch contour into a self-describing blob.

    :param contour: Pitch contour in cents, ``nan`` where the frame is unvoiced.
    :type contour: numpy.ndarray
    :param hop_seconds: Spacing of the contour's frames, in seconds.
    :type hop_seconds: float
    :return: The packed contour, header included.
    :rtype: bytes
    :raises ValueError: if the contour holds no frames.
    """
    samples = quantise(contour)
    if samples.size == 0:
        raise ValueError("cannot pack an empty contour")

    payload = zlib.compress(encode_varints(zigzag(differences(samples))), level=9)
    header = HEADER.pack(MAGIC, VERSION, samples.size, int(round(hop_seconds * 1e6)))
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
        samples = cumulate(unzigzag(decode_varints(payload, frames)))
    else:
        samples = cumulate(decode_words(payload, frames))
    if samples.size != frames:
        raise ValueError(f"packed pitch track holds {samples.size} frames, "
                         f"header says {frames}")

    values = dequantise(samples)
    return values, hop_microseconds / 1e6


def decode_words(payload, frames):
    """Read the sixteen-bit differences of the version 1 layout.

    :param payload: The inflated payload.
    :type payload: bytes
    :param frames: How many values the header promises.
    :type frames: int
    :return: One difference per frame, as the sixteen-bit words hold them.
    :rtype: numpy.ndarray
    :raises ValueError: if the payload is not a whole number of sixteen-bit words, or
        holds a different number of frames than the header says.
    """
    if len(payload) % np.dtype(np.uint16).itemsize:
        raise ValueError(f"packed pitch track payload of {len(payload)} bytes "
                         f"is not a whole number of sixteen-bit differences")
    words = np.frombuffer(payload, dtype=np.uint16)
    if words.size != frames:
        raise ValueError(f"packed pitch track holds {words.size} frames, "
                         f"header says {frames}")
    return words
