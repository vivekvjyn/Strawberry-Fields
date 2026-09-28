"""Apply ``db/migrations`` to the database, one file at a time.

    python scripts/migrate.py

The files run in filename order and each one is recorded in the ``schema_migrations``
table the moment it succeeds, so a re-run applies only what has not been applied yet,
and every file is applied inside a transaction of its own: a migration either happens
in full or not at all.

The login role and the database itself come from ``db/bootstrap/``, which ``setup.sh``
runs through ``psql`` because creating a database needs a superuser and cannot be done
inside a transaction. A hosted database such as Render's already has both, so this
script never has to.
"""

import sys
from pathlib import Path

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pitchtrack.config import database_url

MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

_TRACKING_TABLE = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version    TEXT PRIMARY KEY,
        applied_at TIMESTAMP NOT NULL DEFAULT now()
    )
"""


def migrate(connection_url, migrations_dir=MIGRATIONS_DIR):
    """Apply every migration of ``migrations_dir`` that has not been applied yet.

    :param connection_url: Connection string for the database.
    :type connection_url: str
    :param migrations_dir: Directory holding the ``.sql`` files to apply.
    :type migrations_dir: str or pathlib.Path
    :return: The filenames that were applied, in the order they ran.
    :rtype: list[str]
    """
    files = sorted(Path(migrations_dir).glob("*.sql"))
    applied = []
    with psycopg2.connect(connection_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(_TRACKING_TABLE)
            cursor.execute("SELECT version FROM schema_migrations")
            done = {row[0] for row in cursor}
            for path in files:
                if path.name in done:
                    continue
                cursor.execute(path.read_text(encoding="utf-8"))
                cursor.execute("INSERT INTO schema_migrations (version) VALUES (%s)",
                               (path.name,))
                connection.commit()
                applied.append(path.name)
    return applied


def main():
    """Apply the pending migrations to the database named by the environment.

    :return: Process exit status.
    :rtype: int
    """
    applied = migrate(database_url())
    if not applied:
        print("the database is already up to date")
        return 0
    for version in applied:
        print(f"applied {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
