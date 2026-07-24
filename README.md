# Strawberry Fields

Song retrieval using hummed query.

Hum a melody into your microphone and find the matching song using pitch detection (pYIN) + time-series matching (KNN with DTW).

## Architecture

```
Query (hum) --> pYIN F0 --> MIDI notes --> KNN candidates --> DTW --> Result
```

```
MIDI files --> Monophonic extraction --> Pitch vectors --> PostgreSQL + KNN Model
```

## Prerequisites

- Python 3.9+
- PostgreSQL (local)

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

**Windows:**
Download from https://www.postgresql.org/download/windows/

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

```bash
python setup_db.py
```

## Environment Setup

```bash
git clone https://github.com/vivekvjyn/strawberry-fields.git
cd strawberry-fields
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

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

### Option 1: Individual MIDI files

```bash
python features.py path/to/song.mid --title "Song Title" --composer "Composer Name"
```

### Option 2: MAESTRO dataset (batch)

1. Download MAESTRO v3.0.0 MIDI archive from https://magenta.tensorflow.org/datasets/maestro
2. Extract it
3. Run the ingestion script:

```bash
python ingest_maestro.py --dataset-dir /path/to/maestro --limit 100
```

Remove `--limit` to process all files.

## Training

After adding songs, train the KNN model:

```bash
python train.py
```

This computes pitch windows, assigns hash indices, and saves the model to `model/model.json`.

## Running

```bash
python -m flask run
```

Open http://localhost:5000 in your browser.

## Usage

1. Click the record button or drag an audio file
2. Hum a melody (15 second limit)
3. Wait for results
4. View the matched song with YouTube link

## File Overview

| File | Description |
|------|-------------|
| `app.py` | Flask application, routes |
| `utils.py` | Audio processing (pYIN, DTW, KNN) |
| `features.py` | Add songs from MIDI files |
| `train.py` | Train KNN model from database |
| `ingest_maestro.py` | Batch ingest MAESTRO dataset |
| `setup_db.py` | Initialize PostgreSQL schema |
| `templates/` | HTML templates (Jinja2) |
| `static/` | CSS, JS, images |
