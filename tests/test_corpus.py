"""Tests for :mod:`pitchtrack.corpus`."""

import json
import tempfile
import unittest
from pathlib import Path

from pitchtrack.corpus import Recording, scan_recordings


def write_recording(root, folder, title, raaga, taala, with_metadata=True):
    """Create one corpus folder, so a test can build the corpus it needs."""
    directory = Path(root) / folder
    directory.mkdir(parents=True)
    (directory / f"{folder}.mp3").write_bytes(b"")
    if with_metadata:
        (directory / f"{folder}.json").write_text(
            json.dumps({"title": title, "raaga": raaga, "taala": taala}),
            encoding="utf-8")


class ScanRecordingsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_metadata_is_read_into_the_row(self):
        write_recording(self.root, "Kalaye", "Kalaye", ["Kedaragaula"], ["Adi"])

        recordings = scan_recordings(self.root)

        self.assertEqual(recordings, [Recording(
            identifier="Kalaye",
            audio_path=self.root / "Kalaye" / "Kalaye.mp3",
            title="Kalaye",
            raga="Kedaragaula",
            tala="Adi")])

    def test_several_names_are_joined_once_each(self):
        write_recording(self.root, "Ragamalika", "Ragamalika", ["Kapi", "Kapi", "Kapi"],
                        ["Adi", "Rupakam"])

        recording = scan_recordings(self.root)[0]

        self.assertEqual(recording.raga, "Kapi")
        self.assertEqual(recording.tala, "Adi, Rupakam")

    def test_folders_sharing_a_title_stay_separate_recordings(self):
        write_recording(self.root, "Thillana", "Thillana", ["Bhairavi"], ["Adi"])
        write_recording(self.root, "Thillana (Kruthi Bhat)", "Thillana", ["Bhairavi"],
                        ["Adi"])

        recordings = scan_recordings(self.root)

        self.assertEqual([recording.identifier for recording in recordings],
                         ["Thillana", "Thillana (Kruthi Bhat)"])

    def test_a_folder_without_metadata_is_refused(self):
        write_recording(self.root, "Broken", "Broken", ["Kapi"], ["Adi"], with_metadata=False)

        with self.assertRaises(ValueError):
            scan_recordings(self.root)


if __name__ == "__main__":
    unittest.main()
