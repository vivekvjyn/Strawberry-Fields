"""Tests for :mod:`pitchtrack.corpus`: the layout, and the checks over it."""

import json
import tempfile
import unittest
from pathlib import Path

from pitchtrack.corpus import (ERROR, WARNING, Recording, layout_problems,
                               scan_recordings, song_folders, verify_corpus, verify_song)

MP3_HEAD = b"ID3\x04\x00\x00\x00\x00\x00\x00"

SONG = {"title": "Song 1", "raga": "Raga 1", "tala": "Tala 1"}


def make_song(root, folder, payload=None, audio=MP3_HEAD):
    """Create one song folder of the layout the corpus requires.

    :param root: The corpus folder to create the song folder in.
    :type root: str or pathlib.Path
    :param folder: Name of the song folder.
    :type folder: str
    :param payload: Metadata to write: a dict is written as JSON, a string is written
        as it stands, and ``None`` writes no metadata file at all.
    :type payload: dict or str or None
    :param audio: Bytes to write as ``audio.mp3``, or ``None`` for no audio file.
    :type audio: bytes or None
    :return: The song folder.
    :rtype: pathlib.Path
    """
    directory = Path(root) / folder
    directory.mkdir(parents=True)
    if audio is not None:
        (directory / "audio.mp3").write_bytes(audio)
    if payload is not None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        (directory / "metadata.json").write_text(text, encoding="utf-8")
    return directory


class ScanRecordingsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_metadata_is_read_into_the_row(self):
        make_song(self.root, "Song-001", SONG)

        recordings = scan_recordings(self.root)

        self.assertEqual(recordings, [Recording(
            identifier="Song-001",
            audio_path=self.root / "Song-001" / "audio.mp3",
            title="Song 1",
            raga="Raga 1",
            tala="Tala 1")])

    def test_the_export_spellings_are_read_as_the_same_field(self):
        make_song(self.root, "Song-001",
                  {"title": "Song 1", "raaga": "Raga 1", "taala": "Tala 1"})

        recording = scan_recordings(self.root)[0]

        self.assertEqual(recording.raga, "Raga 1")
        self.assertEqual(recording.tala, "Tala 1")

    def test_a_list_of_one_name_is_read_as_that_one_name(self):
        make_song(self.root, "Song-001",
                  {"title": "Song 1", "raga": ["Raga 1"], "tala": ["Tala 1"]})

        recording = scan_recordings(self.root)[0]

        self.assertEqual(recording.raga, "Raga 1")
        self.assertEqual(recording.tala, "Tala 1")

    def test_a_list_of_several_names_is_refused(self):
        make_song(self.root, "Song-001",
                  {"title": "Song 1", "raga": ["Raga 1", "Raga 2"],
                   "tala": "Tala 1"})

        with self.assertRaisesRegex(ValueError, "holds 2 names"):
            scan_recordings(self.root)

    def test_folders_sharing_a_title_stay_separate_recordings(self):
        make_song(self.root, "Song-001", dict(SONG, title="Shared"))
        make_song(self.root, "Song-002", dict(SONG, title="Shared"))

        recordings = scan_recordings(self.root)

        self.assertEqual([recording.identifier for recording in recordings],
                         ["Song-001", "Song-002"])

    def test_a_folder_without_audio_is_refused(self):
        make_song(self.root, "Song-001", SONG, audio=None)

        with self.assertRaisesRegex(ValueError, "audio.mp3 is missing"):
            scan_recordings(self.root)

    def test_a_folder_without_metadata_is_refused(self):
        make_song(self.root, "Song-001", None)

        with self.assertRaisesRegex(ValueError, "metadata.json is missing"):
            scan_recordings(self.root)

    def test_an_unusable_title_is_refused(self):
        make_song(self.root, "Song-001", {"raga": "Raga 1", "tala": "Tala 1"})

        with self.assertRaisesRegex(ValueError, "title needs a non-empty string"):
            scan_recordings(self.root)

    def test_a_corpus_that_is_not_a_folder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "is not a folder"):
            scan_recordings(self.root / "nowhere")

    def test_a_file_beside_the_song_folders_is_refused(self):
        make_song(self.root, "Song-001", SONG)
        (self.root / "notes.txt").write_text("half of it is missing", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not a song folder"):
            scan_recordings(self.root)

    def test_an_empty_corpus_folder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "holds no song folders"):
            scan_recordings(self.root)

    def test_a_suspicious_audio_file_is_still_read(self):
        make_song(self.root, "Song-001", SONG, audio=b"not really an MP3")

        recordings = scan_recordings(self.root)

        self.assertEqual(len(recordings), 1)


class VerifyCorpusTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fields(self, problems):
        return [problem.field for problem in problems]

    def test_a_sound_corpus_reports_nothing(self):
        make_song(self.root, "Song-001", SONG)

        songs, problems = verify_corpus(self.root)

        self.assertEqual(songs, 1)
        self.assertEqual(problems, [])

    def test_every_broken_folder_is_reported_at_once(self):
        make_song(self.root, "One", {"title": "One"})
        make_song(self.root, "Two", {"title": "Two"})
        make_song(self.root, "Three", SONG, audio=None)

        songs, problems = verify_corpus(self.root)

        self.assertEqual(songs, 3)
        self.assertEqual([(problem.song, problem.field) for problem in problems],
                         [("One", "raga"), ("One", "tala"), ("Three", "audio.mp3"),
                          ("Two", "raga"), ("Two", "tala")])
        self.assertTrue(all(problem.severity == ERROR for problem in problems))

    def test_a_list_of_several_names_is_reported_by_name(self):
        make_song(self.root, "Song-001",
                  dict(SONG, raga=["Raga 1", "Raga 2", "Raga 3"]))

        problems = verify_song(self.root / "Song-001")

        self.assertEqual(self.fields(problems), ["raga"])
        self.assertEqual(problems[0].detail,
                         "holds 3 names, but a song has one raga")

    def test_an_empty_audio_file_is_an_error(self):
        make_song(self.root, "Song-001", SONG, audio=b"")

        problems = verify_song(self.root / "Song-001")

        self.assertEqual(self.fields(problems), ["audio.mp3"])
        self.assertIn("is empty", problems[0].detail)

    def test_audio_that_is_not_an_mp3_is_only_a_warning(self):
        make_song(self.root, "Song-001", SONG, audio=b"wav in disguise")

        problems = verify_song(self.root / "Song-001")

        self.assertEqual([problem.severity for problem in problems], [WARNING])

    def test_metadata_that_is_not_json_is_reported(self):
        make_song(self.root, "Song-001", "{not json")

        problems = verify_song(self.root / "Song-001")

        self.assertEqual(self.fields(problems), ["metadata.json"])
        self.assertIn("not valid JSON", problems[0].detail)

    def test_metadata_that_is_not_an_object_is_reported(self):
        make_song(self.root, "Song-001", "[]")

        problems = verify_song(self.root / "Song-001")

        self.assertEqual(self.fields(problems), ["metadata.json"])
        self.assertIn("JSON object", problems[0].detail)

    def test_a_missing_title_is_reported_by_name(self):
        make_song(self.root, "Song-001", {"raga": "Raga 1", "tala": "Tala 1"})

        problems = verify_song(self.root / "Song-001")

        self.assertEqual(self.fields(problems), ["title"])

    def test_a_name_field_of_the_wrong_type_is_reported_by_name(self):
        make_song(self.root, "Song-001", dict(SONG, raga=42))

        problems = verify_song(self.root / "Song-001")

        self.assertEqual(self.fields(problems), ["raga"])
        self.assertIn("needs", problems[0].detail)

    def test_a_file_beside_the_song_folders_is_reported(self):
        make_song(self.root, "Song-001", SONG)
        (self.root / "README").write_text("hello", encoding="utf-8")

        problems = layout_problems(self.root)

        self.assertEqual([(problem.song, problem.field) for problem in problems],
                         [("README", "data folder")])

    def test_a_corpus_folder_without_song_folders_is_reported(self):
        problems = layout_problems(self.root)

        self.assertEqual(self.fields(problems), ["data folder"])
        self.assertIn("no song folders", problems[0].detail)

    def test_a_corpus_folder_that_does_not_exist_is_reported(self):
        songs, problems = verify_corpus(self.root / "nowhere")

        self.assertEqual(songs, 0)
        self.assertEqual(self.fields(problems), ["data folder"])
        self.assertIn("is not a folder", problems[0].detail)

    def test_hidden_entries_are_left_alone(self):
        make_song(self.root, "Song-001", SONG)
        (self.root / ".DS_Store").write_bytes(b"\x00")
        (self.root / ".git").mkdir()

        songs, problems = verify_corpus(self.root)

        self.assertEqual(songs, 1)
        self.assertEqual(problems, [])

    def test_song_folders_are_listed_in_name_order(self):
        make_song(self.root, "Song-002", SONG)
        make_song(self.root, "Song-001", SONG)

        folders = song_folders(self.root)

        self.assertEqual([folder.name for folder in folders],
                         ["Song-001", "Song-002"])

    def test_a_callback_sees_every_song_folder(self):
        make_song(self.root, "Song-002", SONG)
        make_song(self.root, "Song-001", SONG)
        seen = []

        verify_corpus(self.root, on_song=seen.append)

        self.assertEqual([folder.name for folder in seen],
                         ["Song-001", "Song-002"])


if __name__ == "__main__":
    unittest.main()
