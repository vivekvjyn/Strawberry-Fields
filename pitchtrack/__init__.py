"""Build the pitch tracks and metadata the Strawberry Fields search runs against.

The package is the middle part of a three-step pipeline, each step a script of its
own:

1. ``scripts/build_pitch_tracks.py`` reads the Saraga corpus, tracks the pitch of every
   recording, cuts the phrases the performance states more than once, and packs what
   is left into :mod:`pitchtrack.codec` blobs under ``artifacts/``;
2. ``scripts/migrate.py`` applies ``db/migrations/`` to the database;
3. ``scripts/load_tracks.py`` replaces the contents of the ``tracks`` table with the
   built recordings.

Modules:

* :mod:`pitchtrack.corpus` — the corpus on disk and the metadata it carries;
* :mod:`pitchtrack.phrases` — repeated phrases found in a pitch track, and cut out;
* :mod:`pitchtrack.contour` — hertz to cents, fine grid to storage grid;
* :mod:`pitchtrack.codec` — the packed form a contour is stored as;
* :mod:`pitchtrack.repository` — the ``tracks`` table;
* :mod:`pitchtrack.config` — the connection string, read from ``.env``.

Nothing is imported here, so importing :mod:`pitchtrack` costs nothing; reach for the
module whose job you need.
"""
