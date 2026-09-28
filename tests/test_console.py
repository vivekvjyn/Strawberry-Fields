"""Tests for :mod:`pitchtrack.console`: what the commands print about a corpus."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from rich.console import Console

from pitchtrack.console import check_corpus, problems_table, show, summary_table
from pitchtrack.corpus import ERROR, WARNING, Problem


def capture():
    """A console writing into a string, for a test to assert on the output.

    :return: The console, and the stream it writes to.
    :rtype: tuple[rich.console.Console, io.StringIO]
    """
    stream = io.StringIO()
    return Console(file=stream, width=100, force_terminal=False), stream


def make_song(root, folder, payload, audio=b"ID3\x04\x00"):
    """Create one song folder of the layout the corpus requires.

    :param root: The corpus folder to create the song folder in.
    :type root: str or pathlib.Path
    :param folder: Name of the song folder.
    :type folder: str
    :param payload: Metadata to write as JSON.
    :type payload: dict
    :param audio: Bytes to write as ``audio.mp3``.
    :type audio: bytes
    :return: The song folder.
    :rtype: pathlib.Path
    """
    directory = Path(root) / folder
    directory.mkdir(parents=True)
    (directory / "audio.mp3").write_bytes(audio)
    (directory / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


class ProblemsTableTest(unittest.TestCase):
    def test_every_problem_lands_in_a_row(self):
        console, stream = capture()
        problems = [Problem("Song-001", "audio.mp3", ERROR, "is missing"),
                    Problem("Song-002", "raga", WARNING, "needs a name")]

        show(problems_table(problems, title="2 problems"), target=console)

        output = stream.getvalue()
        self.assertIn("2 problems", output)
        self.assertIn("Song-001", output)
        self.assertIn("audio.mp3", output)
        self.assertIn("is missing", output)
        self.assertIn("Song-002", output)
        self.assertIn("needs a name", output)

    def test_errors_are_printed_before_warnings(self):
        console, stream = capture()
        problems = [Problem("Song-002", "raga", WARNING, "needs a name"),
                    Problem("Song-001", "audio.mp3", ERROR, "is missing")]

        show(problems_table(problems), target=console)

        output = stream.getvalue()
        self.assertLess(output.index("Song-001"), output.index("Song-002"))

    def test_a_problem_about_the_corpus_itself_shows_no_song(self):
        console, stream = capture()

        show(problems_table([Problem("", "data folder", ERROR, "is not a folder")]),
             target=console)

        self.assertIn("data folder", stream.getvalue())


class SummaryTableTest(unittest.TestCase):
    def test_rows_come_back_in_order(self):
        console, stream = capture()

        show(summary_table([("songs", "2"), ("audio", "1.00 GiB")], title="corpus"),
             target=console)

        output = stream.getvalue()
        self.assertIn("corpus", output)
        self.assertLess(output.index("songs"), output.index("audio"))
        self.assertIn("1.00 GiB", output)


class CheckCorpusTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_a_sound_corpus_is_announced_as_passed(self):
        make_song(self.root, "Song-001",
                  {"title": "Song 1", "raga": "Raga 1", "tala": "Tala 1"})
        console, stream = capture()

        songs, problems = check_corpus(self.root, target=console)

        self.assertEqual((songs, problems), (1, []))
        output = stream.getvalue()
        self.assertIn("passed", output)
        self.assertIn("1 song folders", output)

    def test_a_broken_corpus_is_announced_as_failed(self):
        make_song(self.root, "Song-001", {"title": "Song-001"})
        console, stream = capture()

        songs, problems = check_corpus(self.root, target=console)

        self.assertEqual(songs, 1)
        self.assertEqual(len(problems), 2)
        output = stream.getvalue()
        self.assertIn("failed", output)
        self.assertIn("raga", output)
        self.assertIn("tala", output)

    def test_warnings_pass_the_corpus_but_are_still_shown(self):
        make_song(self.root, "Song-001",
                  {"title": "Song 1", "raga": "Raga 1", "tala": "Tala 1"},
                  audio=b"not an MP3")
        console, stream = capture()

        songs, problems = check_corpus(self.root, target=console)

        self.assertEqual(len(problems), 1)
        output = stream.getvalue()
        self.assertIn("passed", output)
        self.assertNotIn("failed", output)
        self.assertIn("does not start like an MP3", output)


if __name__ == "__main__":
    unittest.main()
