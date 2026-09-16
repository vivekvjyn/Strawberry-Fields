# Strawberry Fields

Query-by-humming for Carnatic music.

Hum a melody into your microphone and find the matching kriti using pitch detection
(pYIN) + time-series matching (DTW), searched against pitch tracks and metadata from
the [Saraga Carnatic dataset](https://mtg.github.io/saraga/).

## Architecture

```
Query (hum) --> pYIN F0 --> cents contour --> DTW against every track --> Result
```

```
Saraga pitch tracks + metadata --> cents contours --> PostgreSQL
```

Both the hummed query and the database tracks are converted to a pitch contour in
cents (register-normalized, so absolute tonic/key doesn't matter). Both stages of
signal processing are our own from-scratch implementations, not `librosa.pyin` /
`librosa.sequence.dtw`, with their computational cores written in Cython and
validated to match librosa's output bit-for-bit:

- **pYIN** (`strawberryfields/pyin.py`): the YIN difference function and parabolic
  interpolation (`yin_core.pyx`) and the probabilistic multi-threshold trough
  weighting (`pyin_core.pyx`, exact Boltzmann/Beta-distribution priors) both run in
  Cython; Viterbi/HMM decoding reuses `librosa.sequence.viterbi`. Validated against
  `librosa.pyin` across randomized trials (noise, silence gaps, varied frequencies):
  bit-identical voicing decisions, zero cents error, differences at machine-epsilon
  only.
- **DTW** (`strawberryfields/alignment.py` + `strawberryfields/dtw_core.pyx`):
  subsequence and global DTW, Sakoe-Chiba band constraints, and path backtracking,
  entirely in Cython. Validated against `librosa.sequence.dtw` across 90+ randomized
  shapes/configurations, including the query-longer-than-reference transpose case.

At search time the query contour is scored against every track's contour and the
best matches are returned. No training step, no separate index to build.

## Prerequisites

- Python 3.10+
- A C compiler (gcc/clang) to build the Cython extensions
- PostgreSQL (local)
- ffmpeg (for decoding browser-recorded audio)
- jq, awk, curl, unzip (used by `scripts/setup.sh`)

## PostgreSQL Setup

### Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
brew install postgresql@16
brew services start postgresql@16
```

### Create Database

```bash
sudo -u postgres psql
```

```sql
CREATE USER vivek WITH PASSWORD 'your_password';
CREATE DATABASE strawberry_fields OWNER vivek;
\q
```

### Initialize Schema

`scripts/schema.sql` is applied directly via `psql` - no Python involved. Safe to
re-run: it drops and recreates the tables. `./scripts/setup.sh` does this plus
ingestion in one step (see below), or apply it on its own:

```bash
psql -h localhost -U vivek -d strawberry_fields -f scripts/schema.sql
```

## Environment Setup

```bash
git clone https://github.com/vivekvjyn/strawberry-fields.git
cd strawberry-fields
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py build_ext --inplace
```

The last step compiles `strawberryfields/dtw_core.pyx`, `strawberryfields/yin_core.pyx`,
and `strawberryfields/pyin_core.pyx` into native extensions that `strawberryfields`
imports directly. Re-run it any time you change any `.pyx` file.

Create `.env` file:

```makefile
DB_HOST=localhost
DB_NAME=strawberry_fields
DB_USER=vivek
DB_PASSWORD=your_password
DB_PORT=5432
SECRET_KEY=your_secret_key
```

## Adding Songs

`./scripts/setup.sh` does everything: applies `scripts/schema.sql`, then downloads
(or uses an already-extracted copy of) the Saraga Carnatic dataset and ingests its
pitch tracks and metadata (raaga, taala, artists, work, concert). Download, zip
extraction, JSON metadata parsing (`jq`), pitch contour resampling (`awk`), and
inserts (`psql`) all run as plain shell - no Python involved:

```bash
./scripts/setup.sh                      # download + extract from Zenodo (~several GB), then ingest everything
./scripts/setup.sh --data-home /path    # ingest an already-extracted copy instead of downloading
./scripts/setup.sh --limit 20           # ingest only the first N tracks, for dev/testing
```

## Running

```bash
python wsgi.py
```

Open http://localhost:5000 in your browser. In production, use a WSGI server
instead: `gunicorn wsgi:app`.

## Usage

1. Click the record button (or upload an audio file)
2. Hum a melody (15 second limit)
3. Wait for results
4. View the matched kriti with raaga, taala, artists and concert

## File Overview

```
wsgi.py                                Entry point: from strawberryfields import create_app
setup.py                               Builds the Cython extensions
config.yaml                            Pitch/DTW/search parameters
requirements.txt

strawberryfields/                      Flask application package
    __init__.py                        create_app() application factory
    config.py                          Flask Config class + config.yaml loader
    db.py                              PostgreSQL connection + track cache, loaded once at startup
    views.py                           Blueprint: index, health, search routes
    pitch.py                           Audio loading, cents conversion, contour normalization
    pyin.py                            pYIN orchestration (framing, Viterbi decoding, f0 extraction)
    yin_core.pyx                       Cython: YIN difference function, parabolic interpolation
    pyin_core.pyx                      Cython: probabilistic trough/threshold weighting (Boltzmann/Beta priors)
    alignment.py                       DTW-based melodic matching/search orchestration
    dtw_core.pyx                       Cython: subsequence DTW, Sakoe-Chiba banding, backtracking
    templates/index.html               Recording UI
    static/css/style.css
    static/js/app.js                   Mic recording (MediaRecorder) + results rendering

scripts/
    setup.sh                           Schema creation + Saraga Carnatic download/parse/ingest, in shell
    schema.sql                         Database schema, applied directly via psql
    pitch_contour.awk                  Pitch-to-cents conversion for the Saraga ingest, used by setup.sh
```
