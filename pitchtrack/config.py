"""Connection settings, read the way the Strawberry Fields application reads them.

The application builds its connection string in ``strawberryfields/config.py``; this
module answers the same questions by the same rules, so one ``.env`` describes the
database for both of them:

* ``DATABASE_URL`` when the platform hands back a whole connection string — Render
  does — with ``DB_SSLMODE`` appended to it, ``require`` by default, because Render
  only answers TLS connections;
* otherwise ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER`` and ``DB_PASSWORD``.

``.env`` in the project root is read on import, so the scripts need no flags to pick
up the credentials.
"""

import os

from dotenv import load_dotenv

load_dotenv()

__all__ = ["database_url"]


def database_url():
    """Build the connection string for the project's database.

    :return: A PostgreSQL URL, with its SSL mode set when it came from ``DATABASE_URL``.
    :rtype: str
    """
    supplied = os.environ.get("DATABASE_URL")
    if supplied:
        sslmode = os.environ.get("DB_SSLMODE", "require")
        separator = "&" if "?" in supplied else "?"
        return f"{supplied}{separator}sslmode={sslmode}"

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "strawberry_fields")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"
