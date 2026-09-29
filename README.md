# Strawberry Fields

Query-by-humming for Indian art music. This branch holds the project's Jupyter
notebooks: pitch tracks come out of the audio in `notebooks/data/`, phrases are
searched against them, and the pitch-track format is demonstrated end to end.

## Layout

| Path | Contents |
|---|---|
| `notebooks/compression.ipynb` | packs an example pitch contour and unpacks it again, stage by stage |
| `notebooks/prepare.ipynb` | isolates vocals and extracts pYIN pitch tracks |
| `notebooks/evaluate.ipynb` | retrieval evaluation |
| `notebooks/similar_phrases.ipynb` | similar-phrase search |
| `notebooks/data/` | audio and the query index |
| `notebooks/helpers/` | shared notebook utilities |
| `src/` | the pitch-track codec — `numpy` and `zlib` only — imported by the notebooks |

## Setup

```bash
git clone https://github.com/vivekvjyn/strawberry-fields.git
cd strawberry-fields
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE).
