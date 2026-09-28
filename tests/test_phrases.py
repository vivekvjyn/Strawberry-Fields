"""Tests for :mod:`pitchtrack.phrases`."""

import unittest

import numpy as np

from pitchtrack.phrases import (PhraseConfig, PhraseGroup, deduplicate_pitch,
                                discard_duplicate_phrases, remove_repeated_phrases)

STEP_SECONDS = 0.025

PHRASE = np.array([220.0, 247.0, 262.0, 294.0, 330.0, 294.0, 262.0, 247.0,
                   220.0, 196.0, 220.0, 262.0, 294.0, 262.0, 247.0, 262.0])

FILLER = np.array([349.0, 370.0, 415.0, 440.0, 466.0, 440.0, 415.0, 392.0])


def build_track(length_seconds=60.0, repeats_at=(10.0, 30.0), seed=11):
    """Build a pitch track on which one phrase is stated twice.

    The background is notes drawn at random from a small pool with random lengths, so
    nothing in it comes back; on top of that the same phrase — the same notes, in the
    same order, a quarter of a second each — is written at every start time, which is
    what makes those stretches a repeat of one another.
    """
    rng = np.random.default_rng(seed)
    frames = int(length_seconds / STEP_SECONDS)
    times = np.arange(frames) * STEP_SECONDS
    f0 = np.zeros(frames)

    index = 0
    while index < frames:
        length = int(rng.integers(4, 16))
        f0[index:index + length] = rng.choice(FILLER)
        index += length

    statement = int(4.0 / STEP_SECONDS)
    for start in repeats_at:
        first = int(start / STEP_SECONDS)
        for position in range(statement):
            f0[first + position] = PHRASE[position % len(PHRASE)]

    return times, f0


class DiscardTest(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(2400) * STEP_SECONDS

    def test_the_later_statement_is_the_one_marked(self):
        groups = [PhraseGroup(index=0, spans=((10.0, 18.0), (30.0, 38.0)))]

        duplicates = discard_duplicate_phrases(groups, self.times)

        self.assertFalse(duplicates[400:720].any())
        self.assertTrue(duplicates[1200:1520].all())

    def test_a_track_without_repeats_keeps_everything(self):
        duplicates = discard_duplicate_phrases([], self.times)

        self.assertFalse(duplicates.any())


class DeduplicatePitchTest(unittest.TestCase):
    def test_marked_frames_are_cut_and_the_rest_joined(self):
        times = np.arange(10) * STEP_SECONDS
        f0 = np.arange(10, dtype=float)
        duplicates = np.zeros(10, dtype=bool)
        duplicates[4:6] = True

        kept_times, kept_f0 = deduplicate_pitch(times, f0, duplicates)

        np.testing.assert_array_equal(kept_f0, [0, 1, 2, 3, 6, 7, 8, 9])
        self.assertEqual(len(kept_times), len(kept_f0))


class RemoveRepeatedPhrasesTest(unittest.TestCase):
    def test_the_second_statement_of_a_phrase_is_cut(self):
        times, f0 = build_track()

        kept_times, kept_f0, duplicates = remove_repeated_phrases(
            times, f0, PhraseConfig(window_seconds=60.0, window_hop_seconds=30.0))

        self.assertGreater(duplicates[1200:1520].mean(), 0.5)
        self.assertFalse(duplicates[400:720].any())
        self.assertEqual(len(kept_times), len(kept_f0))
        self.assertLess(len(kept_f0), len(f0))


if __name__ == "__main__":
    unittest.main()
