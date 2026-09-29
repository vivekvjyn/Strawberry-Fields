# Strawberry Fields

This branch contains Python notebooks demonstrating the QBH pipeline.

## Notebooks

| Path | Description |
|---|---|
| `notebooks/compression.ipynb` | Packs an example pitch contour and unpacks it again. |
| `notebooks/prepare.ipynb` | Isolates vocals and extracts pitch tracks from audio. |
| `notebooks/evaluate.ipynb` | QBH retrieval evaluation |
| `notebooks/similar_phrases.ipynb` | Similar-phrase search and deduplication. |

## Setup

```bash
git clone https://github.com/vivekvjyn/strawberry-fields.git
cd strawberry-fields
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Licence

This project is licensed under the MIT License. See [LICENSE](LICENSE).
