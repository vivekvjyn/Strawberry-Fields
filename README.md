# QBH Database

The corpus of pitch tracks and metadata the Strawberry Fields query-by-humming search
runs against: 233 Saraga recordings, each one tracked, stripped of the phrases its
performer states more than once, and packed into one row of a PostgreSQL `tracks`
table.

This is the `database` branch of
[Strawberry Fields](https://github.com/vivekvjyn/Strawberry-Fields).

## Layout

| Path | What it holds |
|---|---|
| `data/saraga/` | the corpus: one folder per recording, audio and metadata (git-ignored) |
| `pitchtrack/` | the package — corpus, phrase removal, contour, packing, table access |
| `scripts/` | the three commands: `migrate`, `build_pitch_tracks`, `load_tracks` |
| `db/bootstrap/` | login role and database for a local PostgreSQL (needs `psql`) |
| `db/migrations/` | schema migrations, applied by `scripts/migrate.py` |
| `artifacts/` | what the build writes: cached pitch tracks and packed recordings (git-ignored) |
| `tests/` | unit tests, run against synthetic material only |

## Pipeline

1. **Scan** — `pitchtrack.corpus` reads `data/saraga/`, one folder per recording, and
   takes `title`, `raaga` and `taala` out of its JSON file.
2. **Track** — `pitchtrack.phrases.extract_pitch` runs pYIN over the audio in
   two-minute chunks and caches the result under `artifacts/pitch/`.
3. **De-repeat** — `pitchtrack.phrases.remove_repeated_phrases` finds the phrases the
   performer states more than once and cuts every statement after the first of each,
   the method of `notebooks/similar_phrases.ipynb`.
4. **Fold** — `pitchtrack.contour.contour_from_track` converts hertz to cents and
   median-pools the 25 ms analysis frames onto the storage grid.
5. **Pack** — `pitchtrack.codec.encode_contour` squeezes the contour into the blob the
   `pitch_track` column holds.
6. **Load** — `pitchtrack.repository.replace_all` empties the table and fills it in one
   transaction.

Steps 1–5 are `scripts/build_pitch_tracks.py`, which writes one `.npz` per recording
under `artifacts/tracks/`; step 6 is `scripts/load_tracks.py`. The two are separate so
that the hours of pitch tracking never have to be repeated to re-load the table.

## The stored pitch track

The column holds a self-describing blob, compressed in three steps:

1. whole cents kept as `int16` — two bytes a frame, and an error of at most half a cent
   against a pitch-class grid fifty cents wide;
2. delta coding — a contour moves only a few cents from frame to frame, so the
   differences repeat far more than the values do;
3. `zlib` at level 9, which the standard library already has, so reading the column
   back needs nothing beyond it.

A packed track measures about **1.5 bytes a frame** — five frames a second, so roughly
27 KiB of database per hour of recording, against 8 bytes a frame before packing. The
payload is already compressed by the time it reaches PostgreSQL, so the column is
created with `SET COMPRESSION NONE`: there is nothing left for TOAST's own pass to
squeeze, and every insert skips work that could not pay. The migration guards that
line on PostgreSQL 14+, where the option exists.

| Field | Size | Value |
|---|---|---|
| `magic` | 4 bytes | `b"SFPT"` |
| `version` | 1 byte | `1` |
| `frames` | 4 bytes | unsigned count of pitch frames |
| `hop_microseconds` | 4 bytes | unsigned spacing of the frames |
| `payload` | remainder | `zlib` stream of delta-coded `int16` cents |

Three properties are what make the column convertible to a pitch-class profile at
runtime, with no re-analysis:

* **cents against a 55 Hz reference**, the same reference the application converts a
  hummed query with;
* **a 0.2 s grid**, the same hop the application resamples a query to;
* **`nan` for an unvoiced frame**, which `to_pitch_class_profile` renders as an
  all-zero column — a silence says *nothing was sung* instead of naming a note.

## Setup

```bash
conda activate sf
pip install -r requirements.txt
```

`.env` in the project root already carries the Render credentials, in the same shape
the application reads: the `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` /
`DB_PASSWORD` parts (a whole `DATABASE_URL` works too, with `DB_SSLMODE=require`
appended to it). `.env.example` is the empty template for another machine.

## Running

```bash
python scripts/migrate.py
python scripts/build_pitch_tracks.py
python scripts/load_tracks.py
```

`migrate.py` applies `db/migrations/` and needs to run once per schema change;
`build_pitch_tracks.py` is the hours-long audio pass and writes `artifacts/tracks`;
`load_tracks.py` puts what it wrote into the table.

`build_pitch_tracks.py` takes `--jobs 8` to track several recordings at once, `--limit
5` for a smoke test, and `--force` to rebuild what it has already built. A recording
with too little pitch in it — under 5% of its frames voiced — is reported and left out
rather than stored as noise, and it exits 1 so that it cannot pass unnoticed.

`load_tracks.py` reads each artefact back, checks that its contour unpacks, replaces
the table, and prints what the table holds afterwards.

## Reading a track back

```python
from pitchtrack import codec
from pitchtrack.config import database_url
from strawberryfields.utils import to_pitch_class_profile

cursor.execute("SELECT pitch_track FROM tracks WHERE id = %s", (track_id,))
contour, hop_seconds = codec.decode_contour(cursor.fetchone()[0])
profile = to_pitch_class_profile(contour, n_classes=24, sigma_cents=150)
```

`contour` is what `strawberryfields.utils.contour_from_audio` produces for a query —
cents on a uniform grid with `nan` where nothing was sung — so the two sides of a
search can be built the same way. Where a query contour is gap-filled first, apply
`strawberryfields.utils.fill_gaps` to the decoded contour as well.

## Migrations

`db/migrations/` is applied in filename order and recorded in `schema_migrations`, so a
re-run applies only what is missing, and each file runs in a transaction of its own:

| File | What it does |
|---|---|
| `001_reset_public_schema.sql` | drops every view and table in `public`, except `schema_migrations` |
| `002_create_tracks_table.sql` | creates `tracks (id, title, raga, tala, pitch_track)` and turns off column compression for `pitch_track` |

The first migration is a deliberate, complete wipe of the previous tables. The login
role and the database itself live in `db/bootstrap/` and are created by `setup.sh`,
which needs a superuser and `psql`; a hosted database already has both.

## Tests

```bash
python -m unittest discover -s tests
```

## Notes

* Ten different pieces in the corpus are called *Thillana*, and titles in general are
  not unique, so the table has no unique key and a load replaces the whole table
  rather than upserting into it. The corpus folder's own name identifies a recording
  while it is on disk, and stays out of the database.
* Phrase removal shortens the track it stores: what is kept is each phrase once, in the
  order it was first stated. The search matches a query as a *subsequence*, so it still
  finds the recording wherever the query lands in it.

## Licence

MIT. See [LICENSE](LICENSE).
