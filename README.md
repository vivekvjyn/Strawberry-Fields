# Strawberry Fieldsdatabase

Scripts for setting up the [Strawberry Fields](https://github.com/vivekvjyn/Strawberry-Fields)
database:

| Column | Type |
|---|---|
| `title` | text |
| `raga` | text |
| `tala` | text |
| `pitch_track` | bytea |

## Pipeline

```mermaid
flowchart TD
    Data[/"your_data"/]
    Validate{"format validation"}
    Stop([stop])
    Extract["pYIN"]
    Dedupe["Self similarity score"]
    Quantise["Quantisation"]
    Compress["Compression"]
    Database[("database")]

    Data --> Validate
    Validate -->|errors| Stop
    Validate -->|valid| Extract
    Extract --> |Pitch track signal| Dedupe
    Dedupe --> |Phrase deduplicated signal| Quantise
    Quantise --> |Quantized signal| Compress
    Compress --> |Compressed signal| Database
```

## Setup

```bash
git clone https://github.com/vivekvjyn/Strawberry-Fields.git
cd Strawberry-Fields
git switch database
pip install -r requirements.txt
```

**Dataset format**: `your_data` is the folder you pass on the
command line. Each recording is one folder holding `audio.mp3` and `metadata.json`:

```text
your_data/
├── song-001/
│   ├── audio.mp3
│   └── metadata.json
└── song-002/
    ├── audio.mp3
    └── metadata.json
```
**Metadata format**
```json
{"title": "Song 1", "raga": "Raga 1", "tala": "Tala 1"}
```

Credentials go in `.env`:

```makefile
DB_HOST=your_host
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=your_port
SECRET_KEY=your_secret_key
```

## Run

```bash
python scripts/verify_data.py your_data
python scripts/migrate.py
python scripts/build_pitch_tracks.py your_data
python scripts/load_tracks.py
```

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE).
