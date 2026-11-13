# LegalVerse Backend

FastAPI backend baseline for LegalVerse with CSV source access, keyword search, optional FAISS semantic retrieval, optional Neo4j graph insights, and graceful fallback.

## Place in repository

Copy the contents of this package into the existing `LegalVerse/backend/` folder. Existing project data stays in `LegalVerse/data/`, artifacts stay in `LegalVerse/artifacts/`, and documentation stays in `LegalVerse/docs/`.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload
```

Open `/docs` for Swagger UI.

## Tests

```bash
pytest
```

## Build FAISS index

```bash
python backend/scripts_build_indexes.py --data data/bns_sections.csv --output-dir backend/artifacts/faiss_indexes
```

## Load base Neo4j graph

```bash
python backend/scripts_load_neo4j.py --data data/bns_sections.csv --password '<your-password>'
```

The Neo4j bootstrap creates Chapter and Section nodes plus `BELONGS_TO` relationships. Concepts, references, and semantic relationships must be added only after their extraction pipelines are evaluated.
