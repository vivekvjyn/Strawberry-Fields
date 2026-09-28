"""Tests for :mod:`pitchtrack.codec`."""

import struct
import unittest
import zlib

import numpy as np

from pitchtrack import codec


def version_one_blob(contour, hop_seconds):
    """Pack a contour the way version 1 did, to prove the older layout still reads."""
    samples = codec._quantise(contour)
    bits = samples.view(np.uint16)
    deltas = np.empty_like(bits)
    deltas[0] = bits[0]
    np.subtract(bits[1:], bits[:-1], out=deltas[1:])
    header = struct.pack("<4sBII", codec.MAGIC, 1, samples.size,
                         int(round(hop_seconds * 1e6)))
    return header + zlib.compress(deltas.tobytes())


class RoundTripTest(unittest.TestCase):
    def test_contour_survives_packing(self):
        contour = np.array([0.0, 12.0, 25.0, np.nan, -400.0, 1199.0, 0.0])

        restored, hop = codec.decode_contour(codec.encode_contour(contour, 0.2))

        self.assertEqual(hop, 0.2)
        np.testing.assert_array_equal(restored, contour)

    def test_long_smooth_contour_round_trips(self):
        rng = np.random.default_rng(7)
        steps = rng.normal(0.0, 30.0, 4000)
        contour = np.cumsum(steps)
        contour[rng.random(4000) < 0.2] = np.nan

        restored, _ = codec.decode_contour(codec.encode_contour(contour, 0.2))

        np.testing.assert_array_equal(np.isnan(restored), np.isnan(contour))
        np.testing.assert_array_equal(restored[~np.isnan(contour)],
                                      np.round(contour[~np.isnan(contour)]))

    def test_rounding_is_within_half_a_cent(self):
        contour = np.linspace(-1500.0, 3500.0, 500)

        restored, _ = codec.decode_contour(codec.encode_contour(contour, 0.2))

        self.assertLessEqual(np.abs(restored - contour).max(), 0.5)

    def test_extreme_values_are_kept(self):
        contour = np.array([float(np.iinfo(np.int16).max),
                            float(np.iinfo(np.int16).min + 1)])

        restored, _ = codec.decode_contour(codec.encode_contour(contour, 0.2))

        np.testing.assert_array_equal(restored, contour)


class SizeTest(unittest.TestCase):
    def test_packing_saves_space(self):
        rng = np.random.default_rng(3)
        contour = np.cumsum(rng.normal(0.0, 25.0, 3000))

        packed = codec.encode_contour(contour, 0.2)

        self.assertLess(len(packed), contour.nbytes)

    def test_version_two_packs_tighter_than_version_one(self):
        rng = np.random.default_rng(5)
        contour = np.cumsum(rng.normal(0.0, 25.0, 3000))
        contour[rng.random(3000) < 0.15] = np.nan

        packed = codec.encode_contour(contour, 0.2)
        previous = version_one_blob(contour, 0.2)

        self.assertEqual(packed[4], 2)
        self.assertLess(len(packed), len(previous))


class VersionTest(unittest.TestCase):
    def test_version_one_blobs_still_decode(self):
        contour = np.array([0.0, 12.0, np.nan, -400.0, 1199.0, -1.0])

        restored, hop = codec.decode_contour(version_one_blob(contour, 0.2))

        self.assertEqual(hop, 0.2)
        np.testing.assert_array_equal(restored, contour)

    def test_a_contour_survives_both_layouts(self):
        rng = np.random.default_rng(13)
        contour = np.cumsum(rng.normal(0.0, 40.0, 900))
        contour[rng.random(900) < 0.2] = np.nan

        from_two, _ = codec.decode_contour(codec.encode_contour(contour, 0.2))
        from_one, _ = codec.decode_contour(version_one_blob(contour, 0.2))

        np.testing.assert_array_equal(from_two, from_one)

    def test_an_unknown_version_is_refused(self):
        blob = bytearray(codec.encode_contour(np.zeros(4), 0.2))
        blob[4] = 99

        with self.assertRaises(ValueError):
            codec.decode_contour(bytes(blob))

    def test_a_truncated_payload_is_refused(self):
        blob = codec.encode_contour(np.arange(500, dtype=float), 0.2)
        payload = blob[codec._HEADER.size:]

        with self.assertRaises(ValueError):
            codec.decode_contour(blob[:codec._HEADER.size] + payload[:len(payload) // 2])


class HeaderTest(unittest.TestCase):
    def test_foreign_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            codec.decode_contour(b"nope" + b"\x00" * 32)

    def test_empty_contour_is_rejected(self):
        with self.assertRaises(ValueError):
            codec.encode_contour(np.zeros(0), 0.2)


if __name__ == "__main__":
    unittest.main()
