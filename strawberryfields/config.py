import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent


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


def load_pitch_config():
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)
