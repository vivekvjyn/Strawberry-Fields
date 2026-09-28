CREATE TABLE IF NOT EXISTS tracks (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    raga        TEXT,
    tala        TEXT,
    pitch_track BYTEA NOT NULL
);

ALTER TABLE tracks ALTER COLUMN pitch_track SET STORAGE EXTERNAL;
