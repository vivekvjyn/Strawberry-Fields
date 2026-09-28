"""Tests for :mod:`pitchtrack.codec`."""

import unittest

import numpy as np

from pitchtrack import codec


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


class HeaderTest(unittest.TestCase):
    def test_foreign_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            codec.decode_contour(b"nope" + b"\x00" * 32)

    def test_empty_contour_is_rejected(self):
        with self.assertRaises(ValueError):
            codec.encode_contour(np.zeros(0), 0.2)


if __name__ == "__main__":
    unittest.main()
