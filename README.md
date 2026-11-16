# LegalVerse

**Bharatiya Nyaya Sanhita Intelligence Explorer**

LegalVerse is a desktop-first, source-first legal research demo built on Bharatiya Nyaya Sanhita (BNS) section data.

It combines:

- statutory section browsing;
- keyword retrieval;
- FAISS-based semantic search;
- hybrid search;
- Neo4j relationship intelligence;
- explainable search results;
- Smart Insights;
- chapter exploration;
- corpus-level insights;
- browser-local bookmarks.

> **Important:** LegalVerse is a research and educational demonstration. Search rankings, semantic relationships, and Smart Insights are not legal advice and do not establish that a provision applies to a particular matter.

---

# Project Status

The core LegalVerse demo implementation is complete.

| Phase | Status | Scope |
|---|---|---|
| Phase 1 | ✅ Complete | UI/UX design and interaction flows |
| Phase 2 | ✅ Complete | System architecture |
| Phase 3 | ✅ Complete | FastAPI backend, FAISS, Neo4j, tests |
| Phase 4.1 | ✅ Complete | React + Vite + Tailwind desktop SPA foundation |
| Phase 4.2 | ✅ Complete | Explore |
| Phase 4.3 | ✅ Complete | Smart Search |
| Phase 4.4 | ✅ Complete | Section Detail + Smart Insights |
| Phase 4.5 | ✅ Complete | Chapters |
| Phase 4.6A | ✅ Complete | Corpus Overview |
| Phase 4.6B | ✅ Complete | Semantic relationship generation, evaluation, and quality policy |
| Phase 4.7 | ✅ Complete | Bookmarks + About |

Detailed project status, architecture, experiments, and development decisions are maintained under `docs/`.

---

# Product Overview

The main LegalVerse research journey is:

```text
Explore
   ↓
Smart Search
   ↓
Relevant BNS Sections
   ↓
Why This Matched
   ↓
Section Detail
   ↓
Smart Insights
   ↓
Related Provisions
```

Users can also browse structurally:

```text
Chapters
   ↓
Chapter
   ↓
Sections
   ↓
Section Detail
```

or revisit saved provisions:

```text
Bookmarks
   ↓
Saved Section
   ↓
Section Detail
```

The guiding principle is:

> **Source law first, machine intelligence second.**

Original BNS statutory text always remains visually and functionally separate from ML-derived information.

---

# Product Capabilities

## Explore

The Explore page provides the primary entry point into LegalVerse.

Capabilities include:

- desktop legal-research landing experience;
- Smart Search input;
- example queries;
- navigation to Chapters;
- navigation to Insights;
- live backend capability information.

Example query:

```text
entrusted property
```

or:

```text
criminal breach of trust
```

---

# Smart Search

Smart Search supports three retrieval strategies:

```text
Recommended
Keyword
Semantic
```

## Recommended

Uses the application's hybrid retrieval strategy.

Conceptually:

```text
User Query
   |
   ├── Keyword Retrieval
   |
   └── Semantic Retrieval
             |
             v
       Candidate Fusion
             |
             v
        Final Ranking
```

## Keyword

Uses lexical matching against BNS section information.

## Semantic

Uses FAISS-backed vector retrieval.

Smart Search displays:

- section number;
- section title;
- chapter;
- statutory excerpt;
- relevance label;
- matched terms;
- search method;
- error/fallback state.

---

# Why This Matched

Search results include an explainability action:

```text
Why This Matched
```

The explanation panel can show:

- search method;
- matched terms;
- related concepts where available;
- research-use disclaimer.

Example:

```text
Search method
Hybrid

Matched terms
dishonestly
property
```

Raw similarity scores are not presented as legal probability or legal certainty.

---

# Section Detail

Section Detail is the core LegalVerse research screen.

It displays three primary information areas:

```text
┌─────────────────┬──────────────────────────┬────────────────────┐
│ Section Context │ Statutory Source         │ Smart Insights     │
│                 │                          │                    │
│ Chapter         │ Section Number           │ Key Concepts       │
│ Chapter Name    │ Section Name             │                    │
│ Subtype         │                          │ Related Sections   │
│                 │ Full Provision           │                    │
│                 │                          │ Cross References   │
└─────────────────┴──────────────────────────┴────────────────────┘
```

## Statutory Source

The central reading pane contains:

- section number;
- section title;
- chapter;
- chapter subtype;
- complete original BNS description.

The statutory provision uses document-style serif typography to visually distinguish source law from the general application interface.

The original statutory text is not rewritten by ML.

---

# Smart Insights

Smart Insights are contextual intelligence attached to a specific section.

Potential information includes:

## Key Concepts

```text
HAS_CONCEPT
```

Concept nodes are displayed only when their generation pipeline is validated and the graph contains corresponding data.

## Related Provisions

Related sections are generated from evaluated FAISS similarity relationships stored in Neo4j.

Example:

```text
Section A
   |
   └── SEMANTICALLY_RELATED
            |
            v
         Section B
```

Related provisions are clickable and reuse the same Section Detail screen.

## Cross References

Cross-references use:

```text
REFERENCES
```

relationships when explicit reference extraction data is available.

Cross-references are treated as deterministic/source-derived information rather than ML recommendations.

---

# Chapters

The Chapters module provides structural BNS browsing.

Capabilities include:

- real chapter list from the backend;
- chapter search;
- section count per chapter;
- selected chapter highlighting;
- sections filtered by selected chapter;
- navigation into Section Detail.

The core flow is:

```text
GET /api/chapters
        ↓
Select Chapter
        ↓
GET /api/sections?chapter_number=...
        ↓
Select Section
        ↓
Section Detail
```

---

# Insights

The Insights module currently exposes source-backed corpus intelligence.

## Corpus Overview

The screen displays:

- total sections;
- total chapters;
- chapter coverage;
- section distribution by chapter;
- keyword-search availability;
- semantic-search availability;
- graph-intelligence availability.

All values come from live backend APIs rather than being hardcoded.

The relevant endpoints are:

```text
GET /api/sections
GET /api/chapters
GET /api/health
```

---

# Semantic Relationships

Semantic section relationships are generated from the existing FAISS embeddings and written into Neo4j.

Graph representation:

```text
(Section)
   |
   | SEMANTICALLY_RELATED
   |
   v
(Section)
```

Relationship properties include:

```text
score
rank
source = "faiss"
updated_at
```

---

# Relationship Generation

The initial candidate generator reads:

```text
backend/artifacts/faiss_indexes/bns_sections.index

backend/artifacts/faiss_indexes/bns_sections.json
```

The current FAISS index contains:

```text
Index type: IndexFlatIP
Vectors:    358
Dimension:  384
```

The initial relationship-generation run produced:

```text
Candidate relationships:          1686
Sections with candidates:          354
Rows skipped due to missing nodes:   0
```

This initial generation used:

```text
top_k = 5
minimum candidate score = 0.55
```

The candidate threshold is not the production/display policy.

---

# Relationship Quality Evaluation

Semantic relationships were evaluated before being promoted to the LegalVerse Smart Insights experience.

A stratified review set of **60 section relationships** was created.

Relationships were labelled as:

```text
relevant
partially_relevant
not_relevant
```

The completed review distribution was:

```text
Relevant:             23
Partially relevant:   28
Not relevant:          9
Total:                60
```

The review considered research usefulness between the two statutory provisions rather than similarity score alone.

Examples of review reasons included:

```text
same_offence_family
same_legal_concept
shared_actor_or_object
cross_chapter_but_useful
generic_statutory_language
lexical_overlap_only
unrelated
```

---

# Approved Demo Relationship Policy

The reviewed data showed a clear drop in quality among lower-ranked and threshold-near relationships.

A conservative demo policy was therefore selected:

```text
Minimum similarity score: 0.70
Maximum neighbour rank:   3
Policy status:             approved_for_demo
```

Within the reviewed relationships matching that policy:

```text
Reviewed pairs:       27

Relevant:             18
Partially relevant:    9
Not relevant:          0

Strict precision:     0.6667
Useful precision:     1.0000
Weighted quality:     0.8333
```

These figures apply to the reviewed subset.

Because the original review set was stratified by similarity-score bands, the metrics must not be treated as an unbiased estimate across every generated semantic relationship.

The policy is intentionally marked:

```text
approved_for_demo
```

rather than production-approved.

---

# Bookmarks

LegalVerse includes lightweight bookmarking for the desktop demo.

Bookmarks are stored in browser:

```text
localStorage
```

No backend user database is required.

Capabilities include:

- bookmark a section from Section Detail;
- toggle Bookmark → Saved;
- remove bookmark;
- persist bookmarks across browser refresh;
- search saved sections;
- open bookmark in Section Detail;
- remove saved section from Bookmarks.

This keeps the demo architecture simple and avoids unnecessary identity/account infrastructure.

---

# About

The About module explains:

- LegalVerse purpose;
- source-first architecture;
- hybrid retrieval;
- FAISS;
- Neo4j;
- distinction between source information and ML-derived insights;
- research-only usage boundary.

---

# Architecture

```text
Desktop Browser
      |
      v
React + Vite + Tailwind SPA
      |
      | /api/*
      v
FastAPI
      |
      +-------------------+-------------------+
      |                   |                   |
      v                   v                   v
BNS CSV / Repository     FAISS              Neo4j
Source + Metadata        Vector Search       Knowledge Graph
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
                   Legal Intelligence
                          |
              +-----------+-----------+
              |                       |
              v                       v
         Smart Search            Smart Insights
```

The architecture uses a modular-monolith backend.

FAISS and Neo4j remain separate intelligence components behind FastAPI service abstractions.

---

# Technology Stack

## Frontend

```text
React
TypeScript
Vite
Tailwind CSS
Lucide React
Browser localStorage
```

LegalVerse is a **desktop-only single-page web application**.

It does not use:

```text
Expo
React Native
Android
iOS
Metro
NativeWind
```

---

## Backend

```text
Python
FastAPI
Pydantic
pandas
scikit-learn
Sentence Transformers
FAISS
Neo4j Python Driver
pytest
```

---

## Intelligence

```text
Keyword Retrieval
        +
Sentence Embeddings
        +
FAISS
        +
Neo4j
        ↓
LegalVerse Smart Intelligence
```

---

# Repository Structure

```text
LegalVerse/
|
|-- frontend/
|
|-- backend/
|   |-- .env
|   |-- __init__.py
|   |-- main.py
|   |-- README.md
|   |-- requirements.txt
|   |-- pytest.ini
|   |
|   |-- scripts_build_indexes.py
|   |-- scripts_load_neo4j.py
|   |-- scripts_build_relationships.py
|   |-- scripts_evaluate_relationships.py
|   |-- scripts_select_relationship_policy.py
|   |-- scripts_apply_relationship_policy.py
|   |
|   |-- api/
|   |   |-- __init__.py
|   |   `-- routes.py
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   `-- container.py
|   |
|   |-- intelligence/
|   |   |-- __init__.py
|   |   |-- faiss_store.py
|   |   `-- neo4j_store.py
|   |
|   |-- models/
|   |   `-- __init__.py
|   |
|   |-- repositories/
|   |   |-- __init__.py
|   |   `-- csv_repository.py
|   |
|   |-- schemas/
|   |   |-- __init__.py
|   |   |-- common.py
|   |   |-- insight.py
|   |   |-- search.py
|   |   `-- section.py
|   |
|   |-- services/
|   |   |-- __init__.py
|   |   |-- insight_service.py
|   |   `-- search_service.py
|   |
|   |-- artifacts/
|   |   |-- faiss_indexes/
|   |   |   |-- bns_sections.index
|   |   |   `-- bns_sections.json
|   |   |
|   |   `-- relationship_policy.json
|   |
|   `-- tests/
|       |-- __init__.py
|       |-- conftest.py
|       |
|       |-- fixtures/
|       |   |-- bns_sections.json
|       |   `-- json_repository.py
|       |
|       |-- integration/
|       |
|       |-- unit/
|       |
|       `-- validation/
|
|-- data/
|   |-- bns_processed.csv
|   `-- bns_sections.csv
|
|-- artifacts/
|
|-- notebooks/
|
|-- evaluation/
|   |-- eda_&_fe.ipynb
|   |-- model_experiments.ipynb
|   |
|   `-- relationships/
|       |-- relationship_review.csv
|       |-- relationship_review_labelled.csv
|       |-- relationship_quality_report.json
|       |-- relationship_policy_analysis.json
|       `-- selected_policy_evidence.json
|
|-- docs/
|   |-- PROJECT_STATUS.md
|   |-- UI_ROADMAP.md
|   |-- ARCHITECTURE.md
|   |-- DEVELOPMENT_ROADMAP.md
|   `-- ML_EXPERIMENTS.md
|
`-- README.md
```

The backend uses the flattened package structure.

There is no longer a:

```text
backend/app/
```

hierarchy.

---

# Data

The primary datasets are:

```text
data/bns_sections.csv
data/bns_processed.csv
```

`bns_sections.csv` remains the source statutory dataset.

Derived information such as:

- embeddings;
- FAISS indexes;
- graph relationships;
- ML artifacts;
- evaluation outputs;

must remain separate from the original source data.

---

# Backend Setup

## 1. Create/activate Python environment

Example:

```bash
python -m venv .venv
```

Activate the environment according to the operating system.

Then install dependencies:

```bash
pip install -r backend/requirements.txt
```

---

# Neo4j Configuration

Configure:

```text
backend/.env
```

Example:

```env
NEO4J_URI=bolt://localhost:7687

NEO4J_USERNAME=neo4j

NEO4J_PASSWORD=<your-password>

NEO4J_DATABASE=neo4j

ENABLE_NEO4J=true
```

Never commit actual credentials to source control.

---

# Run Backend

Use the current LegalVerse command from the project root:

```bash
uvicorn backend:main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health
```

FastAPI documentation is available while the application is running through the standard FastAPI documentation route.

---

# API

The application no longer uses the old:

```text
/api/v1
```

contract.

The current API prefix is:

```text
/api
```

## Current API endpoints

```text
GET  /api/health

GET  /api/sections

GET  /api/sections/{section_number}

GET  /api/sections?chapter_number={chapter_number}

GET  /api/chapters

POST /api/search

GET  /api/insights/sections/{section_number}
```

---

# API Examples

## Health

```bash
curl \
  http://127.0.0.1:8000/api/health
```

---

## Section

```bash
curl \
  http://127.0.0.1:8000/api/sections/316
```

---

## All Sections

```bash
curl \
  http://127.0.0.1:8000/api/sections
```

---

## Chapters

```bash
curl \
  http://127.0.0.1:8000/api/chapters
```

---

## Sections for Chapter

```bash
curl \
  "http://127.0.0.1:8000/api/sections?chapter_number=17"
```

---

## Smart Search

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "entrusted property",
    "mode": "recommended",
    "top_k": 5,
    "chapter_number": null
  }'
```

---

## Section Smart Insights

```bash
curl \
  http://127.0.0.1:8000/api/insights/sections/316
```

---

# FAISS

The current FAISS artifacts are stored under:

```text
backend/artifacts/faiss_indexes/
├── bns_sections.index
└── bns_sections.json
```

The current index is based on normalized sentence embeddings and uses:

```text
IndexFlatIP
```

for similarity retrieval.

Observed index properties:

```text
Vectors:   358
Dimension: 384
```

---

# Build FAISS Index

From the appropriate project environment:

```bash
python -m backend.scripts_build_indexes
```

or use the supported arguments defined by:

```text
backend/scripts_build_indexes.py
```

Generated artifacts belong under:

```text
backend/artifacts/faiss_indexes/
```

---

# Neo4j Graph

The initial Neo4j graph contains:

```text
(Section)-[:BELONGS_TO]->(Chapter)
```

The validated semantic intelligence adds:

```text
(Section)-[:SEMANTICALLY_RELATED]->(Section)
```

Future graph relationships are reserved for:

```text
(Section)-[:REFERENCES]->(Section)

(Section)-[:HAS_CONCEPT]->(Concept)

(Concept)-[:RELATED_TO]->(Concept)
```

These relationships should not be populated until their respective extraction/generation pipelines are validated.

---

# Build Candidate Semantic Relationships

Dry-run:

```bash
python -m backend.scripts_build_relationships \
  --top-k 5 \
  --min-score 0.55 \
  --dry-run
```

Initial candidate generation produced:

```text
FAISS vectors: 358

Metadata rows: 358

Candidate relationships: 1686

Sections with at least one candidate: 354
```

The initial candidate threshold exists for evaluation and should not automatically be treated as the display policy.

---

# Relationship Review

Generate a stratified relationship review worksheet:

```bash
python -m backend.scripts_evaluate_relationships export \
  --per-band 20 \
  --threshold 0.55 \
  --middle-score 0.65 \
  --high-score 0.80 \
  --seed 42
```

Review labels:

```text
relevant

partially_relevant

not_relevant
```

Generate metrics:

```bash
python -m backend.scripts_evaluate_relationships score \
  --input evaluation/relationships/relationship_review_labelled.csv
```

Evaluate threshold/rank policies:

```bash
python -m backend.scripts_select_relationship_policy \
  --input evaluation/relationships/relationship_review_labelled.csv
```

---

# Apply Relationship Quality Policy

The approved demo relationship policy is stored at:

```text
backend/artifacts/relationship_policy.json
```

Dry-run:

```bash
python -m backend.scripts_apply_relationship_policy \
  --dry-run
```

Apply the policy:

```bash
python -m backend.scripts_apply_relationship_policy
```

The script rebuilds only:

```text
SEMANTICALLY_RELATED
```

relationships.

It does not delete:

```text
Section

Chapter

BELONGS_TO

future REFERENCES relationships

future HAS_CONCEPT relationships
```

---

# Verify Semantic Relationships

In Neo4j:

```cypher
MATCH
  (source:Section)
  -[relationship:SEMANTICALLY_RELATED]->
  (target:Section)

RETURN
  count(relationship) AS relationship_count,
  min(relationship.score) AS minimum_score,
  max(relationship.rank) AS maximum_rank;
```

For the current approved demo policy:

```text
minimum_score >= 0.70

maximum_rank <= 3
```

Check for self-links:

```cypher
MATCH
  (section:Section)
  -[relationship:SEMANTICALLY_RELATED]->
  (section)

RETURN
  count(relationship) AS self_links;
```

Expected:

```text
0
```

---

# Backend Testing

Backend tests are organized by intent:

```text
backend/tests/
├── fixtures/
├── unit/
├── validation/
└── integration/
```

The test fixture uses:

```text
backend/tests/fixtures/bns_sections.json
```

rather than the full production CSV.

This keeps unit and integration tests:

- lightweight;
- deterministic;
- easier to maintain.

Run:

```bash
pytest backend/tests
```

Specific relationship tests can also be run independently.

Example:

```bash
pytest backend/tests/unit/test_build_relationships.py
```

```bash
pytest backend/tests/unit/test_evaluate_relationships.py
```

```bash
pytest backend/tests/unit/test_relationship_policy.py
```

Software correctness and ML-quality evaluation are deliberately treated as different concerns.

---

# Frontend

LegalVerse uses a simple desktop SPA architecture.

## Technology

```text
React
TypeScript
Vite
Tailwind CSS
Lucide React
```

The application does not require:

```text
Expo
React Native
Android
iOS
Metro
NativeWind
```

---

# Frontend Structure

A representative frontend structure is:

```text
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
│
└── src/
    ├── App.tsx
    ├── main.tsx
    ├── index.css
    ├── vite-env.d.ts
    │
    ├── components/
    │   ├── Header.tsx
    │   └── Sidebar.tsx
    │
    ├── features/
    │   ├── Explore.tsx
    │   ├── SmartSearch.tsx
    │   │
    │   ├── section/
    │   │   ├── SectionDetail.tsx
    │   │   └── useSectionDetail.ts
    │   │
    │   ├── chapters/
    │   │   ├── Chapters.tsx
    │   │   └── useChapters.ts
    │   │
    │   ├── insights/
    │   │   └── CorpusOverview.tsx
    │   │
    │   ├── bookmarks/
    │   │   └── Bookmarks.tsx
    │   │
    │   └── about/
    │       └── About.tsx
    │
    ├── hooks/
    │   └── useBookmarks.ts
    │
    ├── services/
    │   └── api.ts
    │
    └── types/
        ├── app.ts
        ├── bookmark.ts
        ├── chapter.ts
        ├── insight.ts
        ├── search.ts
        └── section.ts
```

---

# Install Frontend

From:

```text
frontend/
```

run:

```bash
npm install
```

Validate TypeScript:

```bash
npm run typecheck
```

Build:

```bash
npm run build
```

---

# Run Frontend

With the FastAPI backend already running:

```bash
cd frontend

npm run dev
```

Vite normally serves the application at:

```text
http://localhost:5173
```

---

# Vite Backend Proxy

The development server proxies:

```text
/api/*
```

to:

```text
http://127.0.0.1:8000
```

Conceptually:

```text
Browser
   |
   v
localhost:5173
   |
   | /api/*
   v
Vite Proxy
   |
   v
127.0.0.1:8000
   |
   v
FastAPI
```

Therefore frontend API code uses relative URLs:

```text
/api/health

/api/search

/api/sections/316
```

instead of hardcoding the backend host throughout React components.

---

# Recommended Local Startup

Use two terminals.

## Terminal 1: Backend

From the project root:

```bash
source .venv/bin/activate
```

Start FastAPI:

```bash
uvicorn backend:main:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

Verify:

```bash
curl \
  http://127.0.0.1:8000/api/health
```

---

## Terminal 2: Frontend

```bash
cd frontend

npm run dev
```

Then open:

```text
http://localhost:5173
```

Verify the proxy:

```bash
curl \
  http://localhost:5173/api/health
```

Both direct backend and proxy calls should return the LegalVerse health response.

---

# Frontend Validation

Before considering a change stable:

```bash
npm run typecheck
```

and:

```bash
npm run build
```

should both pass.

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



# End-to-End Demo Flow

A useful demo sequence is:

## 1. Explore

Start from:

```text
Explore
```

Enter:

```text
entrusted property
```

---

## 2. Smart Search

Show:

- Recommended search;
- ranked sections;
- relevance labels;
- matched terms.

Open:

```text
Why This Matched
```

to demonstrate explainability.

---

## 3. Section Detail

Open one result.

Show:

- complete source provision;
- legal typography;
- chapter context;
- Smart Insights;
- Related Provisions.

---

## 4. Semantic Relationship

Open a related section.

This demonstrates:

```text
FAISS
   ↓
evaluated similarity
   ↓
Neo4j relationship
   ↓
FastAPI
   ↓
Smart Insights
```

---

## 5. Chapters

Open:

```text
Chapters
```

Select a chapter.

Open a section and show that the same Section Detail experience is reused.

---

## 6. Insights

Open:

```text
Insights
```

Show:

- corpus section count;
- chapter count;
- section distribution;
- search capability availability;
- graph capability availability.

---

## 7. Bookmarks

Bookmark a section.

Open:

```text
Bookmarks
```

Show saved section.

Refresh the browser to demonstrate `localStorage` persistence.

---

## 8. About

Finish with:

```text
About
```

to explain:

- source-first principle;
- FAISS;
- Neo4j;
- ML/source distinction;
- research-use boundary.

---

# Graceful Degradation

LegalVerse separates source availability from intelligence availability.

## If FAISS is unavailable

Expected behavior:

```text
Source browsing       available

Chapter browsing      available

Keyword search        available

Semantic search       degraded/unavailable
```

## If Neo4j is unavailable

Expected behavior:

```text
Section source        available

Smart Insights        unavailable/degraded
```

The UI should never fabricate missing intelligence.

---

# Design Principles

## 1. Source First

Original statutory content has the highest authority.

---

## 2. Insight First, Algorithm Second

The UI communicates:

```text
Strong match

Related provision

Why this matched
```

rather than forcing users to interpret raw ML terminology.

---

## 3. No Artificial Certainty

Embedding similarity is not:

```text
probability that a law applies
```

and should never be presented that way.

---

## 4. Graceful ML Failure

LegalVerse must remain useful even when intelligence services are unavailable.

---

## 5. Evaluate Before Promotion

ML-generated information should not automatically become product intelligence.

The semantic-relationship flow demonstrates this explicitly:

```text
FAISS candidate
      ↓
human review
      ↓
quality analysis
      ↓
approved policy
      ↓
Neo4j relationship
      ↓
frontend Smart Insight
```

---

## 6. Keep the Demo Architecture Simple

The current application intentionally avoids unnecessary infrastructure such as:

- mobile native applications;
- authentication;
- user accounts;
- remote bookmark persistence;
- microservices.

The architecture should remain proportional to the desktop demonstration goal.

---

# Documentation

Detailed documentation lives under:

```text
docs/
├── PROJECT_STATUS.md
├── UI_ROADMAP.md
├── ARCHITECTURE.md
├── DEVELOPMENT_ROADMAP.md
└── ML_EXPERIMENTS.md
```

## `PROJECT_STATUS.md`

Tracks:

- current phase;
- completed work;
- pending work;
- blockers;
- important decisions.

## `UI_ROADMAP.md`

Tracks:

- design system;
- screen definitions;
- navigation;
- user flows;
- Smart Insight presentation;
- frontend operational states.

## `ARCHITECTURE.md`

Tracks:

- application architecture;
- backend layers;
- FAISS responsibility;
- Neo4j responsibility;
- frontend/backend contract;
- test boundaries.

## `DEVELOPMENT_ROADMAP.md`

Tracks:

```text
Phase
  ↓
Milestone
  ↓
Module
  ↓
Implementation
  ↓
Test
  ↓
Freeze
```

## `ML_EXPERIMENTS.md`

Tracks:

- experiment objective;
- dataset/version;
- technique;
- parameters;
- metrics;
- observations;
- limitations;
- decision;
- artifact/version.

---

# Current Completion State

The main LegalVerse demo workflow is complete:

```text
Explore                 ✅

Smart Search            ✅

Why This Matched        ✅

Section Detail          ✅

Smart Insights          ✅

Semantic Relationships ✅

Chapters                ✅

Corpus Overview         ✅

Bookmarks               ✅

About                   ✅

Backend tests           ✅

Frontend type checking  ✅

Frontend build          ✅
```

---

# Potential Future Enhancements

The following are intentionally outside the completed core demo.

## Concept Intelligence

Generate and validate:

```text
(Section)-[:HAS_CONCEPT]->(Concept)
```

Then introduce Concept Explorer.

---

## Explicit Cross References

Extract actual section references from statutory source text and create:

```text
(Section)-[:REFERENCES]->(Section)
```

---

## Chapter Intelligence

Aggregate validated section/concept relationships at chapter level.

---

## Graph Visualization

Visualize:

```text
Chapter
   ↓
Section
   ↓
Concept
   ↓
Related Section
```

only after the underlying relationships have sufficient quality.

---

## Larger Quality Evaluation

Expand beyond the current 60-pair semantic relationship review.

Potential future evaluation should include:

- additional random sampling;
- cross-chapter samples;
- hard-negative examples;
- rank-specific evaluation;
- multiple reviewers where appropriate.

---

## Deployment

Potential deployment work includes:

- production FastAPI server configuration;
- built Vite static assets;
- reverse proxy;
- secret management;
- Neo4j deployment;
- CI/CD;
- frontend/backend smoke tests.

These are deployment/stabilization concerns rather than requirements for the current local desktop demo.

---

# Final Product Principle

> **LegalVerse should help users discover and explore BNS provisions intelligently while always making it clear which information comes from the statute and which information is derived by algorithms.**