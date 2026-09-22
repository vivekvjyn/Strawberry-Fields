# Strawberry Fields

Query-by-humming for Indian art music. Hum a melody into your microphone and find
the matching piece, searched against the pitch tracks and metadata of a corpus.
Each match returns:

| Field | Description |
|---|---|
| Title | name of the piece |
| Artists | credited performers |
| *Rāga* | melodic framework it is set in |
| *Tāla* | metrical cycle |
| *Laya* | tempo class |
| Form | compositional form |

## Setup

```bash
git clone https://github.com/vivekvjyn/strawberry-fields.git
cd strawberry-fields
pip install -r requirements.txt
python setup.py build_ext --inplace
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

## Running

```bash
python wsgi.py
```

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE).
