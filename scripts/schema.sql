DROP TABLE IF EXISTS tracks;

CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    mbid TEXT,
    title TEXT NOT NULL,
    artists TEXT,
    raaga TEXT,
    taala TEXT,
    work TEXT,
    concert TEXT,
    tonic REAL NOT NULL,
    duration REAL NOT NULL,
    hop_seconds REAL NOT NULL,
    pitch_cents DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_tracks_raaga ON tracks (raaga);
