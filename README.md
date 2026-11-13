# LegalVerse

**Bharatiya Nyaya Sanhita Intelligence Explorer**

LegalVerse is a source-first, ML-assisted research application built around the Bharatiya Nyaya Sanhita (BNS) section dataset. The product is designed to help users explore statutory provisions, search using legal or plain-language queries, discover related sections, and surface explainable smart insights while keeping original statutory text clearly separated from ML-derived information.

> LegalVerse is intended as a research and educational aid. It does not provide legal advice or determine whether a provision legally applies to a specific case.

---

## Current Project Status

- **Phase 1 - UI/UX Design:** Completed / frozen
- **Phase 2 - Architecture Design:** Completed / frozen
- **Phase 3 - Backend Development:** Backend baseline implemented; real dataset, FAISS artifact, and live Neo4j integration testing pending
- **Phase 4 - Frontend Development:** Not started

Detailed project status is maintained in `docs/PROJECT_STATUS.md`.

---

## Product Direction

The primary LegalVerse workflow is:

```text
Discover
   -> Smart Search
   -> Evaluate Relevant Sections
   -> Read Original BNS Text
   -> Explore Concepts / Related Sections
```

ML is embedded as an intelligence layer rather than exposed as a separate experiment screen.

Planned product capabilities include:

- Smart Search using keyword and semantic retrieval
- Direct section lookup
- Chapter browsing
- Related-section discovery
- Concept-based exploration
- Explainable "Why This Matched" insights
- Neo4j-powered legal knowledge relationships
- Bookmarking
- Graceful fallback when ML services are unavailable

---

## Architecture Overview

```text
React Native + Expo
        |
        v
     FastAPI
        |
  Application Services
        |
  +-----+--------------------+
  |                          |
  v                          v
Legal/Data Services    Intelligence Services
  |                          |
  +-------------+------------+
                |
      +---------+---------+
      |                   |
      v                   v
    Neo4j               FAISS
Knowledge Graph      Vector Retrieval
      |                   |
      +---------+---------+
                |
                v
        BNS Source Data
```

### Technology Direction

**Frontend**

- React Native
- Expo
- TypeScript
- NativeWind

**Backend**

- FastAPI
- Pydantic
- pandas
- scikit-learn

**Intelligence Layer**

- FAISS for semantic/vector retrieval
- Sentence Transformers for section/query embeddings
- Neo4j for graph relationships and traversal
- TF-IDF keyword retrieval as the initial lexical search baseline and fallback

---

## Repository Structure

```text
LegalVerse/
|
|-- frontend/
|-- backend/
|
|-- data/
|   |-- bns_processed.csv
|   |-- bns_sections.csv
|
|-- artifacts/
|-- notebooks/
|
|-- evaluation/
|   |-- eda_&_fe.ipynb
|   |-- model_experiments.ipynb
|
|-- docs/
|   |-- PROJECT_STATUS.md
|   |-- UI_ROADMAP.md
|   |-- ARCHITECTURE.md
|   |-- DEVELOPMENT_ROADMAP.md
|   |-- ML_EXPERIMENTS.md
|
|-- README.md
```

`bns_sections.csv` remains the raw statutory source. Derived datasets, embeddings, FAISS indexes, graph imports, and other generated intelligence artifacts must remain separate from the raw source.

---

# Backend

The Phase 3 backend baseline follows a layered FastAPI architecture.

```text
API Routes
    |
Pydantic Schemas
    |
Application Services
    |
Repositories / Intelligence Adapters
    |
CSV + FAISS + Neo4j + Artifacts
```

The backend is designed so that React Native does not need to know which embedding model, FAISS implementation, graph query, or ranking algorithm produces an insight.

## Current Backend Capabilities

The implemented backend baseline includes:

- BNS CSV repository
- Section retrieval
- Chapter listing
- Exact section lookup
- TF-IDF keyword search
- FAISS semantic-search adapter
- Hybrid retrieval path
- Neo4j graph adapter
- Section Smart Insights endpoint
- Health/dependency availability endpoint
- Partial-ML fallback handling
- Environment-based configuration
- Unit tests
- Validation tests
- API integration tests
- FAISS index-building utility
- Base Neo4j graph-loading utility

The isolated backend test suite currently passes **11 tests** against representative fixture data.

The real project `bns_sections.csv`, generated FAISS artifacts, and live Neo4j instance still require integration testing before the backend is considered production-ready.

---

## Backend Structure

```text
backend/
|-- app/
|   |-- api/
|   |   |-- v1/
|   |
|   |-- core/
|   |-- intelligence/
|   |-- models/
|   |-- repositories/
|   |-- schemas/
|   |-- services/
|   |-- main.py
|
|-- tests/
|   |-- unit/
|   |-- validation/
|   |-- integration/
|   |-- fixtures/
|
|-- requirements.txt
|-- pytest.ini
|-- .env.example
|-- scripts_build_indexes.py
|-- scripts_load_neo4j.py
```

---

## API Baseline

```text
GET  /api/v1/health
GET  /api/v1/sections
GET  /api/v1/sections/{section_number}
GET  /api/v1/chapters
POST /api/v1/search
GET  /api/v1/insights/sections/{section_number}
```

Additional API contracts will be introduced only when required by approved frontend modules and intelligence capabilities.

---

## Backend Setup

From the `backend/` directory:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Create local runtime configuration from the provided example:

```bash
cp .env.example .env
```

Important configuration areas include:

- BNS data paths
- artifact paths
- FAISS index paths
- embedding model selection
- Neo4j connection settings
- search defaults
- intelligence feature switches

Secrets such as the Neo4j password must not be committed to source control.

---

## Run the API

From `backend/`:

```bash
uvicorn app.main:app --reload
```

FastAPI exposes its generated API documentation through the standard application documentation routes when running locally.

---

## Run Backend Tests

```bash
pytest
```

Tests are organized into three required groups:

- `tests/unit/`
- `tests/validation/`
- `tests/integration/`

Code tests and ML-quality evaluation are intentionally kept separate. Passing API/unit tests does not by itself approve an ML insight for frontend use.

---

# FAISS

FAISS is responsible for vector-based retrieval and semantic similarity.

Primary planned uses:

- semantic query retrieval
- related-section discovery
- nearest-section search
- semantic candidate generation for hybrid search

## Build the FAISS Index

From `backend/`:

```bash
python scripts_build_indexes.py \
  --data ../data/bns_sections.csv \
  --output-dir ../artifacts/faiss_indexes
```

Expected generated artifacts are kept under `artifacts/`, not under `data/raw` or the original CSV location.

The production embedding model and retrieval configuration remain subject to ML evaluation recorded in `docs/ML_EXPERIMENTS.md`.

---

# Neo4j

Neo4j provides the graph intelligence layer.

The initial graph supports the relationship direction:

```text
(Section)-[:BELONGS_TO]->(Chapter)
```

The approved architecture also allows validated relationships such as:

```text
(Section)-[:REFERENCES]->(Section)
(Section)-[:HAS_CONCEPT]->(Concept)
(Section)-[:SEMANTICALLY_RELATED]->(Section)
(Concept)-[:RELATED_TO]->(Concept)
```

These additional relationships should only be populated after their extraction/generation logic is evaluated.

## Load the Base Neo4j Graph

From `backend/`:

```bash
python scripts_load_neo4j.py \
  --data ../data/bns_sections.csv \
  --password '<your-neo4j-password>'
```

Neo4j connection settings are maintained through environment configuration.

---

# Search Strategy

LegalVerse exposes product-oriented search modes:

```text
Recommended -> Hybrid retrieval
Keyword     -> Lexical retrieval
Semantic    -> FAISS vector retrieval
```

At a high level:

```text
User Query
   |-- Keyword Retrieval
   |-- FAISS Semantic Retrieval
              |
              v
       Candidate Fusion
              |
              v
       Ranking / Thresholds
              |
              v
       Neo4j Enrichment
              |
              v
       Smart Insights
```

The specific production ranking strategy and thresholds must be validated through retrieval experiments rather than assumed from initial implementation values.

---
# User Interface 
## Explore tabs
<img width="944" height="502" alt="UI1_explore" src="https://github.com/user-attachments/assets/03d5171d-9c13-4263-8b25-275b1f12ee1f" />

<img width="944" height="503" alt="UI2_explore_input" src="https://github.com/user-attachments/assets/28af4f87-1c05-40cf-9777-896d66a3dfc9" />

## Smart Search
<img width="941" height="497" alt="UI3_smart_search" src="https://github.com/user-attachments/assets/76285521-6edf-406f-8fe4-cbd4a702d8e4" />

<img width="950" height="500" alt="UI5_smart_insights" src="https://github.com/user-attachments/assets/e0391982-36d6-40fb-8772-bd78232ff4aa" />

## Chanpters
<img width="1368" height="741" alt="UI4_chapters" src="https://github.com/user-attachments/assets/79de8d3e-d68a-4a1e-bf2f-481501f81886" />

## Insights
<img width="1360" height="628" alt="UI6_insight1" src="https://github.com/user-attachments/assets/40a8665f-7ad2-42d7-9adf-9eaf62f127ac" />

<img width="1357" height="628" alt="UI8_insights2" src="https://github.com/user-attachments/assets/ba3241e5-835d-4192-8530-0732ee85da19" />

---

# Graceful Intelligence Fallback

LegalVerse treats statutory access and ML intelligence as separate availability layers.

If FAISS or Neo4j is unavailable:

- statutory sections should remain accessible;
- chapter browsing should remain accessible;
- exact lookup should remain accessible;
- keyword retrieval should remain available where the source dataset is loaded;
- unavailable Smart Insights should be clearly identified rather than fabricated.

---

# Project Documentation

Detailed documentation lives under `docs/`:

- `docs/PROJECT_STATUS.md` - current project state, progress, blockers, and next work
- `docs/UI_ROADMAP.md` - frozen UI/UX direction and flows
- `docs/ARCHITECTURE.md` - approved system architecture
- `docs/DEVELOPMENT_ROADMAP.md` - implementation sequence and milestones
- `docs/ML_EXPERIMENTS.md` - ML experiments, evaluation, results, and promotion decisions

The root `README.md` remains the high-level project entry point and is updated after major project milestones.

---

## Current Development Boundary

Phase 3 currently has a tested backend baseline. Before frontend module development begins, the next backend validation step is to run the implementation against:

1. the actual `data/bns_sections.csv`;
2. generated FAISS artifacts;
3. a live Neo4j instance;
4. real API integration scenarios;
5. retrieval/insight quality evaluation separate from code correctness.

Only validated ML-derived capabilities should be promoted to the React Native Smart Insights experience.
