# Strawberry Fields

Query-by-humming for Hindustani music. Hum a melody into your microphone and find the
matching piece, searched against the pitch tracks and metadata of the
[Saraga Hindustani dataset](https://mtg.github.io/saraga/).

## Prerequisites

- Python 3.10+
- A C compiler (gcc/clang) to build the Cython extensions
- A running PostgreSQL database that you have already created, and its credentials
- ffmpeg (for decoding browser-recorded audio)

## Setup

### 1. Install dependencies and build the Cython extensions

```bash
git clone https://github.com/vivekvjyn/strawberry-fields.git
cd strawberry-fields
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### 2. Create `.env`

```makefile
DB_HOST=localhost
DB_NAME=strawberry_fields
DB_USER=your_username
DB_PASSWORD=your_password
DB_PORT=5432
SECRET_KEY=your_secret_key
```

### 3. Provide the data

The app reads a `tracks` table from your database at startup. Create and populate it
yourself with these columns (one row per track):

| column        | type                       | notes                                   |
|---------------|----------------------------|-----------------------------------------|
| `id`          | `SERIAL PRIMARY KEY`       |                                         |
| `title`       | `TEXT NOT NULL`            |                                         |
| `artists`     | `TEXT`                     |                                         |
| `raag`        | `TEXT`                     |                                         |
| `taal`        | `TEXT`                     |                                         |
| `laya`        | `TEXT`                     |                                         |
| `form`        | `TEXT`                     |                                         |
| `pitch_cents` | `DOUBLE PRECISION[] NOT NULL` | pitch contour in cents, one value every 25 ms |

## Running

```bash
python wsgi.py
```

Open http://localhost:5000 in your browser. In production, use a WSGI server
instead: `gunicorn wsgi:app`.
