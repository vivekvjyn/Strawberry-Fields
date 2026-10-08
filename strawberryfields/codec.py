import struct
import zlib

import numpy as np
from google.protobuf.internal.decoder import _DecodeError, _DecodeVarint
from google.protobuf.internal.wire_format import ZigZagDecode

def quantise(contour):
    """Round a contour to whole cents, one ``int16`` per frame.

    A whole cent is an error of at most half a cent against a pitch-class grid fifty
    cents wide, and ``-32768`` is reserved for a frame that carries no pitch, so
    silence travels as ``nan`` instead of being dropped.

    :param contour: Pitch contour in cents, ``nan`` where the frame is unvoiced.
    :type contour: numpy.ndarray
    :return: One whole cent per frame, ``-32768`` where there was no pitch.
    :rtype: numpy.ndarray
    """
    values = np.ascontiguousarray(contour, dtype=np.float64).reshape(-1)
    unvoiced = ~np.isfinite(values)
    rounded = np.rint(np.where(unvoiced, 0.0, values))
    rounded = np.clip(rounded, -32767, np.iinfo(np.int16).max)
    rounded[unvoiced] = -32768
    return np.ascontiguousarray(rounded, dtype=np.int16)


def dequantise(samples):
    """Read whole cents back into a contour, ``nan`` where there was no pitch.

    :param samples: Whole cents, ``-32768`` where the frame was unvoiced.
    :type samples: numpy.ndarray
    :return: The contour in cents as floating point values.
    :rtype: numpy.ndarray
    """
    packed = np.ascontiguousarray(samples, dtype=np.int16).reshape(-1)
    values = packed.astype(np.float64)
    values[packed == -32768] = np.nan
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
    header = struct.Struct("<4sBII")
    magic, version, frames, hop_microseconds = header.unpack_from(blob, 0)
    if magic != b"SFPT":
        raise ValueError(f"not a packed pitch track: magic {magic!r}")
    if version not in (1, 2):
        raise ValueError(f"packed pitch track version {version}, "
                         "expected one of (1, 2)")

    try:
        payload = zlib.decompress(blob[header.size:])
    except zlib.error as error:
        raise ValueError(f"packed pitch track payload does not inflate: {error}") from error

    differences = (_decode_varints(payload, frames) if version == 2
                   else _decode_words(payload, frames))
    if differences.size != frames:
        raise ValueError(f"packed pitch track holds {differences.size} frames, "
                         f"header says {frames}")

    samples = np.cumsum(differences, dtype=np.uint64) % 65536
    samples = samples.astype(np.uint16).view(np.int16)
    values = dequantise(samples)
    return values, hop_microseconds / 1e6


def _decode_varints(data, frames):
    """Read a stream of seven-bit groups back as the differences it holds.

    Both steps are protobuf's own originals, the wire format the writer folded and
    packed with: :func:`google.protobuf.internal.decoder._DecodeVarint` reassembles
    one group at a time and :func:`google.protobuf.internal.wire_format.ZigZagDecode`
    takes the zigzag fold back off, the value then wrapped to sixteen bits as the
    writer stored it.

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
        try:
            value, position = _DecodeVarint(data, position)
        except IndexError:
            raise ValueError(f"packed pitch track ends after {index} of {frames} frames") from None
        except _DecodeError:
            raise ValueError("packed pitch track holds an oversized integer") from None
        differences[index] = ZigZagDecode(value) & 0xFFFF

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
