import os

from dotenv import load_dotenv

load_dotenv()


def _database_url():
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
    SECRET_KEY = os.environ.get("SECRET_KEY")
    DATABASE_URL = _database_url()
    MAX_CONTENT_LENGTH = 25 * 1024 * 1024

    PITCH = {
        "fmin": 65.0,
        "fmax": 1046.0,
        "frame_length": 2048,
        "hop_seconds": 0.025,
        "ref_hz": 55.0,
        "bin_cents": 25,
        "range_cents": 2400,
        "sigma_cents": 200,
    }
