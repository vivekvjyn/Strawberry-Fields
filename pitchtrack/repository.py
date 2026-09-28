"""The ``tracks`` table, which is the only table this project writes.

One row holds one recording: its title, the raga and tala it is set in, and its pitch
track packed by :mod:`pitchtrack.codec` into the ``pitch_track`` column. Nothing else
is stored with it — the frame spacing travels inside the packed blob, and the corpus's
own folder name stays outside the database.

The table is emptied before every load rather than upserted into, because the corpus
identifies its recordings by folder name while the table has no column for one: ten
different *Thillana*s share a title, and reloading has to replace the whole corpus
rather than guess which of them a row belongs to.
"""

from dataclasses import dataclass

import psycopg2
import psycopg2.extras

__all__ = [
    "Track",
    "replace_all",
    "count_tracks",
]

_INSERT_ROWS = "INSERT INTO tracks (title, raga, tala, pitch_track) VALUES %s"


@dataclass(frozen=True)
class Track:
    """One row of the ``tracks`` table, before it is written.

    :param title: Title of the piece performed.
    :type title: str
    :param raga: Raga the performance is set in.
    :type raga: str
    :param tala: Tala the performance is set in.
    :type tala: str
    :param pitch_track: The recording's pitch track, packed by
        :func:`pitchtrack.codec.encode_contour`.
    :type pitch_track: bytes
    """

    title: str
    raga: str
    tala: str
    pitch_track: bytes


def replace_all(tracks, connection_url, batch_size=500):
    """Empty the ``tracks`` table and fill it with ``tracks``.

    The whole load runs in one transaction, so the table is never observed half
    rewritten: a failure leaves the corpus that was there before exactly as it was.

    :param tracks: The rows to store.
    :type tracks: collections.abc.Iterable[Track]
    :param connection_url: Connection string for the database.
    :type connection_url: str
    :param batch_size: Rows sent to the server per round trip.
    :type batch_size: int
    :return: The number of rows written.
    :rtype: int
    """
    with psycopg2.connect(connection_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE tracks RESTART IDENTITY")
            rows = [(track.title, track.raga, track.tala, psycopg2.Binary(track.pitch_track))
                    for track in tracks]
            psycopg2.extras.execute_values(cursor, _INSERT_ROWS, rows, page_size=batch_size)
            return cursor.rowcount


def count_tracks(connection_url):
    """Count the rows the ``tracks`` table holds.

    :param connection_url: Connection string for the database.
    :type connection_url: str
    :return: Number of stored recordings.
    :rtype: int
    """
    with psycopg2.connect(connection_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM tracks")
            return int(cursor.fetchone()[0])
