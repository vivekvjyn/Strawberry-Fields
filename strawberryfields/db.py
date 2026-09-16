import numpy as np
import psycopg2
import psycopg2.extras


def dict_cursor(db):
    return db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def load_tracks(app):
    db = psycopg2.connect(app.config["DATABASE_URL"])
    try:
        with dict_cursor(db) as cur:
            cur.execute(
                "SELECT id, title, artists, raaga, taala, work, concert, pitch_cents FROM tracks"
            )
            rows = cur.fetchall()
    finally:
        db.close()

    return {
        row["id"]: {**row, "contour": np.asarray(row.pop("pitch_cents"), dtype=np.float64)}
        for row in rows
    }
