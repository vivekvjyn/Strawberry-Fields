-- Run once against the strawberry_fields database:
--   psql "$DATABASE_URL" -f sql/02_schema.sql
-- or:
--   PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f sql/02_schema.sql
--
-- Schema matches the README: one row per track, pitch contour in cents
-- sampled every 25 ms (see strawberryfields/config.py PITCH["hop_seconds"]).

CREATE TABLE IF NOT EXISTS tracks (
    id           SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    artists      TEXT,
    raag         TEXT,
    taal         TEXT,
    laya         TEXT,
    form         TEXT,
    pitch_cents  DOUBLE PRECISION[] NOT NULL
);
