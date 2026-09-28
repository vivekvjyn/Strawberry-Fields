"""Tests for :mod:`pitchtrack.contour`."""

import unittest

import numpy as np

from pitchtrack import contour


class HertzToCentsTest(unittest.TestCase):
    def test_reference_maps_to_zero(self):
        cents = contour.hz_to_cents(np.array([55.0]))

        self.assertAlmostEqual(cents[0], 0.0)

    def test_octave_up_maps_to_twelve_hundred(self):
        cents = contour.hz_to_cents(np.array([110.0]))

        self.assertAlmostEqual(cents[0], 1200.0)

    def test_unvoiced_frames_become_nan(self):
        cents = contour.hz_to_cents(np.array([0.0, -1.0, 220.0]))

        self.assertTrue(np.isnan(cents[0]))
        self.assertTrue(np.isnan(cents[1]))
        self.assertAlmostEqual(cents[2], 2400.0)


class MedianPoolTest(unittest.TestCase):
    def test_blocks_are_pooled_by_their_median(self):
        pooled = contour.median_pool(np.arange(8, dtype=float), 4)

        np.testing.assert_array_equal(pooled, [1.5, 5.5])

    def test_a_block_without_a_value_pools_to_nan(self):
        pooled = contour.median_pool(
            np.array([1.0, np.nan, np.nan, np.nan,
                      np.nan, np.nan, np.nan, np.nan,
                      5.0, 6.0]), 4)

        self.assertAlmostEqual(pooled[0], 1.0)
        self.assertTrue(np.isnan(pooled[1]))
        self.assertAlmostEqual(pooled[2], 5.5)

    def test_a_partial_final_block_is_padded(self):
        pooled = contour.median_pool(np.arange(10, dtype=float), 4)

        self.assertEqual(len(pooled), 3)
        self.assertAlmostEqual(pooled[2], 8.5)

    def test_frame_count_below_one_block_is_refused(self):
        with self.assertRaises(ValueError):
            contour.median_pool(np.zeros(4), 0)


class ContourFromTrackTest(unittest.TestCase):
    def test_track_is_folded_onto_the_storage_grid(self):
        times = np.arange(80) * 0.025
        f0 = np.full(80, 220.0)
        f0[:8] = 0.0

        values, hop = contour.contour_from_track(times, f0)

        self.assertEqual(hop, 0.2)
        self.assertEqual(len(values), 10)
        self.assertTrue(np.isnan(values[0]))
        self.assertAlmostEqual(values[1], 2400.0)


if __name__ == "__main__":
    unittest.main()
