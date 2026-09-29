import os

from dotenv import load_dotenv

load_dotenv()


def _database_url():
    """Build the PostgreSQL connection string for the app.

    ``DATABASE_URL`` is used as it stands, with an ``sslmode`` parameter appended
    when it does not already carry one; otherwise the string is assembled from
    ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER`` and ``DB_PASSWORD``, with
    ``DB_SSLMODE`` (default ``require``) added.

    :return: Connection string for psycopg2.
    :rtype: str
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        sslmode = os.environ.get("DB_SSLMODE", "require")
        separator = "&" if "?" in database_url else "?"
        return f"{database_url}{separator}sslmode={sslmode}"

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "strawberry_fields")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


class Config:
    """Settings the app is configured from.

    ``SECRET_KEY`` and ``DATABASE_URL`` come from the environment, and
    ``MAX_CONTENT_LENGTH`` caps uploads at 25 MB. ``PITCH`` holds the analysis
    and matching settings: ``analysis_hop_seconds`` is the pYIN hop on the query
    audio, ``hop_seconds`` the grid the contour is resampled to, ``n_classes``
    the pitch classes per octave (50 cents each), and ``shift_step`` the class
    shifts tried when matching (2 means every second class, i.e. 12
    transpositions).
    """

    SECRET_KEY = os.environ.get("SECRET_KEY")
    DATABASE_URL = _database_url()
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024

    PITCH = {
        "fmin": 65.0,
        "fmax": 1046.0,
        "frame_length": 2048,
        "analysis_hop_seconds": 0.04,
        "hop_seconds": 0.2,
        "ref_hz": 55.0,
        "n_classes": 24,
        "sigma_cents": 150,
        "shift_step": 2,
    }
