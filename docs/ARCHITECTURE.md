# LegalVerse Architecture

## Phase 2: Approved High-Level Architecture

**Product:** LegalVerse - Bharatiya Nyaya Sanhita Intelligence Explorer  
**Architecture style:** Modular monolith  
**Frontend:** React Native, Expo, TypeScript, NativeWind  
**Backend:** FastAPI  
**Vector retrieval:** FAISS  
**Knowledge graph:** Neo4j  
**Status:** Phase 2 architecture approved

---

## 1. Architecture Goal

LegalVerse will provide source-first access to Bharatiya Nyaya Sanhita sections and enrich the user experience with ML-powered search, concepts, section relationships, chapter intelligence, and explainable retrieval.

The architecture must support the approved UI and UX without coupling the frontend to a specific model, embedding library, or storage implementation.

The core principle is:

> LegalVerse must remain a useful BNS explorer when ML functionality is degraded or unavailable, while becoming significantly smarter when the validated intelligence layer is available.

---

## 2. Overall Architecture

LegalVerse V1 will use a modular-monolith backend with separate internal modules for legal data, search, ML intelligence, and graph intelligence.

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
      Source Data and Artifacts
```

Microservices are not required for V1. Internal service boundaries must remain clear enough to allow later separation if product scale or deployment requirements justify it.

---

## 3. Project Structure

```text
LegalVerse/
|
|-- frontend/
|-- backend/
|-- data/
|-- artifacts/
|-- notebooks/
|-- evaluation/
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

`README.md` remains in the project root. Detailed project, architecture, UI, roadmap, and ML documentation remains under `docs/`.

---

## 4. Frontend Architecture

The frontend will use:

- React Native
- Expo
- TypeScript
- NativeWind
- feature-oriented modules

Recommended high-level structure:

```text
frontend/
|-- app/
|-- components/
|-- features/
|   |-- explore/
|   |-- search/
|   |-- sections/
|   |-- chapters/
|   |-- insights/
|   |-- bookmarks/
|-- services/
|-- store/
|-- types/
```

### Frontend responsibilities

- render statutory and ML-derived information using the approved visual hierarchy;
- preserve search and navigation state;
- support partial-ML and fallback states;
- keep API access behind service modules;
- remain independent of FAISS, Neo4j, and specific ML model implementations.

Search context must survive navigation such as:

```text
Search -> Section A -> Related Section B -> Back to original results
```

---

## 5. FastAPI Backend Architecture

The backend will use layered separation:

```text
API Routes
    |
Request and Response Schemas
    |
Application Services
    |
Repositories and Intelligence Services
    |
Source Data, Neo4j, FAISS, and Artifacts
```

High-level backend modules:

- Sections
- Chapters
- Search
- Insights
- Graph Relationships
- Bookmarks, if server-side persistence is approved
- Health and System Status

API routes must not contain retrieval, graph, or ML business logic.

---

## 6. Dual Intelligence Architecture

LegalVerse V1 will use Neo4j and FAISS together. They solve different product needs.

### FAISS responsibility

FAISS will support vector-based intelligence including:

- semantic query retrieval;
- nearest-section search;
- section-to-section similarity;
- initial semantic candidate generation;
- embedding-based related-section discovery.

### Neo4j responsibility

Neo4j will store and query explicit and derived relationships including:

```text
(Section)-[:BELONGS_TO]->(Chapter)
(Section)-[:REFERENCES]->(Section)
(Section)-[:HAS_CONCEPT]->(Concept)
(Section)-[:SEMANTICALLY_RELATED]->(Section)
(Concept)-[:RELATED_TO]->(Concept)
```

Neo4j will support:

- Concept Explorer;
- Related Sections enrichment;
- cross-reference navigation;
- chapter relationships;
- section relationship traversal;
- chapter and concept intelligence;
- future graph-based experiences.

### Combined flow

```text
User Query
    |
    v
Keyword and Semantic Retrieval
    |
    v
FAISS Candidate Sections
    |
    v
Neo4j Relationship Enrichment
    |
    v
Ranking and Explanation
    |
    v
FastAPI Response
    |
    v
React Native Smart Insights
```

FAISS and Neo4j must remain behind service abstractions so either implementation can evolve independently.

---

## 7. Source and Data Architecture

The raw BNS dataset remains immutable.

```text
data/
|-- raw/
|   |-- bns_sections.csv
|-- processed/
|-- evaluation/
|-- indexes/
```

Rules:

- never overwrite the raw CSV;
- create deterministic processing pipelines;
- preserve original statutory text;
- store derived data separately;
- version processed data and generated artifacts;
- make source-to-artifact lineage traceable.

Derived data may include:

- normalized metadata;
- extracted cross-references;
- concepts;
- embeddings;
- FAISS indexes;
- graph import data;
- section relationships;
- chapter intelligence.

---

## 8. Online and Precomputed Intelligence

LegalVerse will use a hybrid precomputed/runtime approach.

### Precomputed

- section embeddings;
- FAISS vector index;
- section-to-section similarity candidates;
- extracted concepts;
- explicit cross-references;
- Neo4j section, chapter, and concept nodes;
- validated graph relationships;
- chapter-level intelligence.

### Runtime

- input validation;
- query embedding;
- keyword retrieval;
- FAISS semantic retrieval;
- hybrid ranking;
- Neo4j relationship enrichment;
- match explanation composition;
- relevance threshold and fallback handling.

Precomputation keeps runtime behavior fast and avoids repeatedly calculating corpus-level intelligence.

---

## 9. Search Architecture

Smart Search will support three user-facing strategies:

```text
Recommended -> hybrid retrieval
Keyword     -> lexical retrieval
Semantic    -> FAISS vector retrieval
```

Recommended search flow:

```text
User Query
   |-- Keyword Retrieval
   |-- Semantic Retrieval through FAISS
              |
              v
       Candidate Fusion
              |
              v
       Ranking and Thresholds
              |
              v
       Neo4j Enrichment
              |
              v
       Match Explanation
```

The frontend must not depend on the specific embedding model, lexical algorithm, or score-fusion implementation.

---

## 10. Insights Architecture

Approved frontend insight requirements map to backend capabilities as follows:

- **Smart Search:** lexical and FAISS-assisted hybrid retrieval
- **Related Sections:** FAISS similarity plus Neo4j relationships
- **Key Concepts:** validated concept extraction stored in Neo4j
- **Concept Explorer:** Neo4j concept-to-section and concept-to-concept traversal
- **Chapter Intelligence:** graph aggregation and precomputed relationships
- **Why This Matched:** lexical evidence, semantic evidence, and relationship context
- **Cross-References:** deterministic extraction and Neo4j `REFERENCES` relationships

The frontend exposes validated insights, not ML experiments.

---

## 11. High-Level API Groups

```text
/api/v1/sections
/api/v1/chapters
/api/v1/search
/api/v1/insights
/api/v1/relationships
/api/v1/bookmarks
/api/v1/health
```

Exact request and response models will be finalized before backend development.

The API contract must represent source data, deterministic metadata, and ML-derived intelligence as distinct fields.

---

## 12. Failure Isolation and Fallback

ML and graph failures must not prevent access to statutory content.

```text
FastAPI
  |
  |-- Legal and Source Data -> remains available
  |
  |-- FAISS Intelligence -> may degrade independently
  |
  |-- Neo4j Intelligence -> may degrade independently
```

Expected fallback behavior:

- exact section lookup remains available;
- chapters and statutory text remain available;
- keyword search remains available where supported;
- FAISS failure disables semantic retrieval and related-vector insights;
- Neo4j failure disables relationship enrichment and concept traversal;
- frontend receives explicit partial-availability information;
- no fabricated insight replaces an unavailable capability.

---

## 13. Testing Architecture

Code quality and ML quality will be evaluated separately.

### Code tests

```text
tests/
|-- unit/
|-- validation/
|-- integration/
|-- fixtures/
```

Coverage includes:

- service logic;
- schema validation;
- API behavior;
- repository behavior;
- Neo4j query adapters;
- FAISS adapters;
- failure fallback;
- integration across API, search, graph, and data layers.

### ML and retrieval evaluation

```text
evaluation/
|-- retrieval/
|-- similarity/
|-- concepts/
|-- graph_relationships/
```

Evaluation determines whether an insight is useful enough to enter the frontend. Passing code tests alone does not approve an ML capability.

---

## 14. Artifact and Version Strategy

```text
artifacts/
|-- embeddings/
|-- faiss_indexes/
|-- concepts/
|-- relationships/
|-- graph_imports/
|-- models/
|-- manifests/
```

Each production artifact must be traceable to:

```text
source dataset version
    -> processing version
    -> experiment or generation run
    -> artifact version
    -> active runtime configuration
```

MLflow may be evaluated during implementation but is not mandatory for the initial architecture.

---

## 15. Configuration

Environment-based configuration will manage:

- raw and processed data paths;
- artifact locations;
- embedding model selection;
- active FAISS index version;
- Neo4j connection settings;
- search strategy defaults;
- ranking weights;
- relevance thresholds;
- feature switches;
- logging level;
- timeout and fallback behavior.

Paths, credentials, thresholds, and model choices must not be hardcoded inside API routes.

---

## 16. Security and Operational Boundaries

- Neo4j credentials and environment secrets remain outside source control.
- User search text must not be logged without an explicit logging and privacy decision.
- API errors must not expose stack traces or credentials.
- Graph and vector artifacts must be treated as generated assets, not source truth.
- Original statutory text must remain unchanged and independently retrievable.
- Health checks should distinguish general API, FAISS, and Neo4j availability.

---

## 17. Approved Phase 2 Decisions

1. Use a modular monolith for V1.
2. Use React Native, Expo, TypeScript, and NativeWind for the frontend.
3. Use a layered FastAPI backend.
4. Organize frontend code by product feature.
5. Preserve `bns_sections.csv` as immutable raw source data.
6. Use FAISS for vector similarity and semantic retrieval.
7. Use Neo4j for explicit and derived legal-knowledge relationships.
8. Keep FAISS and Neo4j behind independent service abstractions.
9. Precompute corpus intelligence and perform query intelligence at runtime.
10. Use hybrid search as the default Smart Search strategy.
11. Keep statutory browsing operational when FAISS or Neo4j is unavailable.
12. Separate code testing from ML and insight-quality evaluation.
13. Version datasets, embeddings, indexes, relationships, and graph imports.
14. Keep `README.md` at the project root and detailed documentation under `docs/`.

---

## 18. Architecture Freeze Statement

Phase 2 is approved with a dual intelligence architecture:

> FAISS provides vector retrieval and semantic similarity, while Neo4j provides legal-knowledge relationships and graph traversal. FastAPI combines these capabilities behind stable application services, and React Native consumes only product-focused source and insight contracts.

Any later change to this architecture must be documented in `docs/ARCHITECTURE.md`, reflected in `docs/PROJECT_STATUS.md`, and added to `docs/DEVELOPMENT_ROADMAP.md` when it changes implementation scope or sequencing.
