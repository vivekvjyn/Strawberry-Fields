CREATE TABLE IF NOT EXISTS tracks (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    raga        TEXT,
    tala        TEXT,
    pitch_track BYTEA NOT NULL
);

DO $$
BEGIN
    IF current_setting('server_version_num')::int >= 140000 THEN
        EXECUTE 'ALTER TABLE tracks ALTER COLUMN pitch_track SET COMPRESSION NONE';
    END IF;
END
$$;
