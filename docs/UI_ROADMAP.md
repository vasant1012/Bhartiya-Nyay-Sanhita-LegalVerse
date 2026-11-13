# LegalVerse UI Roadmap

## Phase 1: UI/UX Design Freeze

**Product:** LegalVerse - Bharatiya Nyaya Sanhita Intelligence Explorer  
**Primary UI stack:** React Native, Expo, TypeScript, NativeWind  
**Primary experience:** Landscape-first legal research and intelligence workspace  
**Status:** UI-01 through UI-05 proposed for Phase 1 freeze

---

## 1. Product Direction

LegalVerse will be a source-first, ML-assisted Bharatiya Nyaya Sanhita research product. It will help users search, browse, connect, and inspect BNS provisions while clearly separating statutory content from machine-derived intelligence.

The product will not present itself as:

- a generic analytics dashboard;
- an ML experiment interface;
- a chatbot-first application;
- a system that determines guilt or gives legal advice.

The primary user journey is:

> Discover -> Search -> Evaluate candidate sections -> Read source text -> Explore related law

ML is embedded throughout the experience as a practical intelligence layer. Model experiments, parameters, and evaluation details remain in project documentation and backend tooling rather than becoming the main frontend experience.

---

# UI-01: Product Shell and Landscape Strategy

## 2. Design Principles

1. **Source before intelligence**  
   Original BNS section text has the highest visual and informational authority.

2. **Insight first, algorithm second**  
   The frontend explains the benefit of an ML result before exposing technical model details.

3. **ML must be actionable**  
   An insight must help users find, inspect, connect, or navigate provisions.

4. **No artificial certainty**  
   Relevance is not presented as legal applicability or legal certainty.

5. **Graceful ML degradation**  
   Basic chapter browsing, section lookup, and statutory reading must remain usable when ML functionality is unavailable.

6. **Research context is preserved**  
   Search query, filters, ordering, scroll position, and navigation history remain intact when users inspect sections and return.

## 3. Viewport Strategy

### Primary target

- Desktop and large-tablet landscape
- Optimized around 1280 x 800 and above
- Comfortable from approximately 1024 px landscape
- Supports information-dense, multi-column research layouts

### Secondary target

- Smaller landscape tablets
- Collapsible sidebar
- Collapsible contextual panel
- Statutory reading pane remains dominant

### Portrait/mobile policy

Portrait and mobile layouts will be supported through stacked layouts, drawers, and compact navigation. Mobile-first compromises must not weaken the primary landscape research experience.

## 4. Global Application Shell

The application shell contains:

- persistent left navigation;
- context-aware top bar;
- flexible page workspace;
- optional right contextual panel;
- responsive collapsed states.

### Left navigation

Expanded width: approximately 216 px  
Collapsed width: approximately 68 px

Primary navigation:

1. Explore
2. Smart Search
3. Chapters
4. Insights
5. Bookmarks

Secondary navigation:

- About

There is no generic Dashboard in V1. The product opens directly into useful discovery and research functionality.

### Top bar

The top bar contains:

- page title or breadcrumb;
- fast global lookup for section number, title, or source text;
- bookmark shortcut where appropriate;
- help/about access.

Example breadcrumb:

`Chapters / XVII / Section 316`

The global lookup is optimized for fast navigation. The larger Smart Search experience supports plain-language scenarios and intelligence-assisted retrieval.

## 5. Workspace Layouts

The shared shell supports one, two, or three-column pages.

- **Explore:** single or two-column discovery layout
- **Smart Search Results:** result list plus filters/context
- **Section Detail:** chapter navigation, statutory reader, Smart Insights
- **Chapters:** chapter index plus selected chapter workspace
- **Insights:** intelligence categories and exploration views

## 6. Visual Information Authority

### Tier A: Statutory source truth

Includes:

- section number;
- section title;
- chapter information;
- original statutory description.

Visual identity: navy and legal gold.

### Tier B: Deterministic derived information

Includes:

- extracted cross-references;
- structural metadata;
- word counts;
- chapter and subtype relationships directly present in data.

Visual identity: neutral slate and standard product colors.

### Tier C: ML-derived intelligence

Includes:

- semantic relevance;
- related provisions;
- extracted concepts;
- chapter intelligence;
- recommendation and relationship signals.

Visual identity: ML purple.

A user must be able to distinguish source text, deterministic metadata, and ML-derived information immediately.

---

# UI-02: Navigation and Screen Inventory

## 7. Navigation Hierarchy

```text
LegalVerse
|
|-- Explore
|
|-- Smart Search
|   |-- Search Results
|       |-- Why This Matched
|       |-- Section Detail
|
|-- Chapters
|   |-- Selected Chapter
|       |-- Section Detail
|
|-- Insights
|   |-- Concept Explorer
|   |-- Section Relationships
|   |-- Chapter Intelligence
|   |-- Corpus Overview
|
|-- Bookmarks
|   |-- Section Detail
|
|-- About
    |-- LegalVerse
    |-- Dataset
    |-- Smart Insights
    |-- Disclaimer
```

## 8. Screen Inventory

### P0: Core product

- **S01:** Application Shell
- **S02:** Explore
- **S03:** Smart Search
- **S04:** Smart Search Results
- **S05:** Section Detail
- **S06:** Chapters and selected Chapter Detail
- **S08:** Insights
- **S10:** Bookmarks
- **S11:** About and Dataset Information

### P1: Intelligence expansion

- **S09:** Concept Explorer
- Section Relationships detailed view
- Chapter Intelligence detailed view

### Deferred

- Collections
- Search History
- Experiment Comparison
- Chatbot or conversational legal assistant
- Interactive concept graph

## 9. Screen Responsibilities

### S01: Application Shell

Provides persistent navigation, page context, responsive behavior, global lookup, and shared layout boundaries.

### S02: Explore

Acts as the home screen. It contains:

- product value statement;
- large Smart Search entry;
- example ways to search;
- Browse Chapters entry;
- Discover Concepts entry;
- Explore Relationships entry;
- recently viewed sections when available.

There is no separate Home page.

### S03: Smart Search

Supports:

- ordinary-language situations;
- legal terminology;
- section identifiers;
- optional search strategy selection.

Default strategy label: **Recommended**

Advanced options:

- Recommended
- Keyword
- Semantic

Technical terms such as TF-IDF and cosine similarity do not appear in the primary search interaction.

### S04: Smart Search Results

Displays ranked candidate BNS sections with:

- section number;
- title;
- chapter;
- relevant source excerpt;
- qualitative relevance label;
- matched concepts when available;
- Why This Matched action;
- Read Section action;
- Bookmark action.

### S05: Section Detail

This is the flagship and convergence screen.

Landscape regions:

1. Chapter and nearby-section navigation
2. Original statutory section reader
3. Smart Insights contextual panel

The statutory reader always displays the original source wording. ML does not replace, rewrite, or silently summarize the provision.

### S06: Chapters

Uses a two-column master/detail layout:

- left: chapter index and chapter search;
- right: selected chapter details and section list.

Chapter Detail is a state of the Chapters screen rather than a separate route.

Optional Chapter Insights include prominent concepts, representative sections, and related chapters only after validation.

### S08: Insights

Provides four intelligence entry points:

1. Concept Explorer
2. Section Relationships
3. Chapter Intelligence
4. Corpus Overview

Corpus Overview is descriptive. The other areas may contain validated ML-derived insights.

### S09: Concept Explorer

Displays:

- selected concept;
- relevant provisions;
- related concepts;
- relevant chapters;
- navigation into Section Detail.

An interactive graph is deferred until the usefulness and quality of relationships are validated.

### S10: Bookmarks

Provides:

- saved sections;
- bookmark search;
- chapter filtering;
- direct navigation to Section Detail.

Collections are deferred from V1.

### S11: About

Contains:

- product purpose;
- dataset information;
- explanation of Smart Insights;
- distinction between source and ML-derived information;
- research and legal-advice disclaimer.

---

# UI-03: Design System

## 10. Color Palette

| Role | Color | Primary usage |
|---|---|---|
| Deep Navy | `#14283F` | Navigation and structural elements |
| Primary Navy | `#1E3A5F` | Primary actions and headings |
| Legal Gold | `#C9952E` | Section identity and statutory emphasis |
| ML Purple | `#6554C0` | ML-derived insights only |
| Canvas | `#F7F8FA` | Application background |
| Surface | `#FFFFFF` | Cards and panels |
| Primary Text | `#172033` | Main text |
| Secondary Text | `#667085` | Supporting metadata |
| Border | `#E3E7ED` | Dividers and boundaries |
| Success | `#2E7D5B` | Success states |
| Warning | `#B7791F` | Caution states |
| Error | `#B54747` | Error states |

### Color rule

- Gold identifies statutory/legal material.
- Purple identifies machine-derived intelligence.
- Purple must not be used as a general decorative accent.

## 11. Typography

Recommended fonts:

- **Inter:** interface, navigation, forms, cards, metadata
- **Source Serif 4:** long statutory text

Recommended hierarchy:

- Product name: 22 px / 700
- Page heading: 26-30 px / 700
- Section title: 24 px / 700
- Card heading: 16-18 px / 600
- Application body: 14-16 px
- Statutory body: 17 px with approximately 28 px line height
- Metadata: 12-13 px
- Badge: 11-12 px

## 12. Spacing

Base spacing system:

`4 / 8 / 12 / 16 / 24 / 32 / 48`

Guidance:

- card padding: 16-20 px;
- major panel padding: 24 px;
- screen edge: 24-32 px;
- card gap: 12-16 px;
- major section gap: 24-32 px.

The interface should be information-dense but readable. Excessive empty space should be avoided.

## 13. Component System

### Structural components

- `AppShell`
- `Sidebar`
- `TopBar`
- `PageHeader`
- `ContentPanel`
- `ContextPanel`
- `SplitPane`

### Search components

- `GlobalSearch`
- `SmartSearchBox`
- `SearchStrategySelector`
- `SearchFilter`
- `SearchResultCard`

### Legal components

- `SectionBadge`
- `ChapterBadge`
- `SectionTitle`
- `StatutoryText`
- `SectionNavigator`
- `CrossReference`
- `SectionMetadata`

### Intelligence components

- `InsightPanel`
- `InsightCard`
- `ConceptChip`
- `RelatedSectionCard`
- `MatchExplanation`
- `RelevanceIndicator`

### Utility components

- `BookmarkButton`
- `EmptyState`
- `ErrorState`
- `Skeleton`
- `Tooltip`
- `Drawer`

## 14. Smart Insight Language

Every ML-derived component uses a consistent heading such as:

> Smart Insight

The component may show:

- related concepts;
- related provisions;
- relevance explanation;
- semantic relationships;
- chapter intelligence.

It must include an information affordance:

> How was this generated?

The explanation must reflect the production technique eventually approved for that insight.

## 15. Relevance Presentation

Default qualitative labels:

- Very strong match
- Strong match
- Possible match
- Related provision

Raw retrieval scores must not be presented as legal probability or certainty. Technical ranking details may appear under an advanced disclosure.

Thresholds and label mapping remain subject to retrieval evaluation and calibration.

---

# UI-04: Detailed Core Screen Design

## 16. Explore Screen

### Hero content

**Understand BNS through intelligent exploration**

Supporting message:

> Search sections, describe a situation, or explore relationships across the Bharatiya Nyaya Sanhita.

Primary Smart Search input:

> Search sections or describe a situation...

Quick entry cards:

- Browse Chapters
- Discover Concepts
- Explore Relationships

Recently viewed sections appear only after real user activity. The initial state contains no fabricated recent content.

## 17. Smart Search Screen

Primary guidance:

> Describe what you are looking for using legal terms or ordinary language.

Primary action:

**Find Relevant Sections**

Advanced search strategy selector remains secondary.

## 18. Search Result Card

Required anatomy:

1. Section identifier
2. Section title
3. Chapter metadata
4. Relevant source excerpt
5. Qualitative relevance label
6. Matched concepts when reliable
7. Read Section
8. Why This Matched
9. Bookmark

## 19. Why This Matched Drawer

Contains:

- concept relationship;
- relevant text relationship;
- human-readable search method;
- optional technical details.

It does not claim that a provision legally applies to the user's situation.

## 20. Section Detail Screen

### Left region

- current chapter;
- nearby sections;
- selected-section indicator;
- sequential navigation.

### Center region

- section identifier;
- section title;
- chapter metadata;
- original statutory text;
- independent scrolling for long provisions.

### Right region: Smart Insights

Potential validated modules:

- key concepts;
- related provisions;
- Why Related explanation;
- search-match explanation when opened from results;
- cross-references displayed separately as deterministic information.

## 21. Insights Screen

Entry cards:

- Concept Explorer
- Section Relationships
- Chapter Intelligence
- Corpus Overview

The screen presents intelligence capabilities, not model experiments or algorithm metrics.

---

# UI-05: End-to-End UX Flows

## 22. F01 - Plain-Language Smart Search

```text
Explore
  -> Enter situation
  -> Smart Search processing
  -> Relevant Sections
  -> Why This Matched
  -> Read statutory source
```

### Required behavior

- Empty input prompts the user to enter a section, legal concept, or situation.
- Short but meaningful queries remain valid.
- Loading message uses product language such as "Finding relevant BNS provisions" rather than model terminology.
- Result cards appear as skeletons during loading.
- Back navigation restores query, filters, sort order, and scroll position.

### Acceptance criterion

A user can move from an ordinary-language description to original BNS source text without encountering unnecessary model terminology.

## 23. F02 - Direct Section Lookup

```text
Global Search
  -> Exact section detected
  -> Section Detail
```

An unambiguous section number should open directly. Partial titles may show exact/title matches before intelligence-assisted candidates.

## 24. F03 - Why This Matched

```text
Search Result
  -> Why This Matched
  -> Explanation Drawer
```

The drawer explains retrieval evidence in understandable terms. Model names and raw score composition remain under optional technical details.

## 25. F04 - Related Provision Exploration

```text
Search
  -> Section A
  -> Smart Insights
  -> Related Section B
  -> Related Section C
```

A lightweight research trail is preserved:

`Search -> Section A -> Section B`

## 26. F05 - Concept Discovery

```text
Section Detail
  -> Select Key Concept
  -> Concept Explorer
  -> Relevant Provisions
  -> Section Detail
```

Selecting an ML-derived concept opens a dedicated concept view rather than a basic substring search.

## 27. F06 - Browse by Chapter

```text
Explore
  -> Browse Chapters
  -> Select Chapter
  -> Select Section
  -> Section Detail
```

Chapter navigation works without ML. Validated Chapter Insights enhance the experience but do not block it.

## 28. F07 - Insights-First Exploration

```text
Insights
  -> Choose intelligence type
  -> Explore concept or relationship
  -> Open Section
```

The user can begin from a concept, section relationship, chapter pattern, or corpus overview.

## 29. F08 - Bookmarking

```text
Result or Section
  -> Toggle Bookmark
  -> Saved state updates immediately
```

No confirmation dialog is required for toggling. The storage strategy remains an architecture decision.

## 30. F09 - No Strong ML Result

The system must not present the nearest result as a strong match merely because the retrieval system always returns candidates.

Required state:

**No strong matches identified**

Recommended actions:

- Try keyword search
- Browse chapters
- Modify search

A separately labeled "Possibly related" area may be used only after evaluation establishes appropriate thresholds.

## 31. F10 - ML Service Unavailable

Expected behavior:

- statutory data remains accessible;
- exact/title lookup remains available where supported;
- chapter browsing remains available;
- Smart Insights displays an unavailable state;
- Smart Search offers keyword search as fallback.

ML failure must not create a full-product failure.

## 32. F11 - Backend/API Failure

User-facing message:

**We could not load this content.**

Available actions:

- Try again
- Return to Explore

Internal exceptions, stack traces, and HTTP details are not shown to ordinary users.

## 33. F12 - Offline State

User-facing message:

**You are offline.**

If local caching is later supported, previously loaded material may remain available. The caching strategy will be finalized during architecture design.

## 34. F13 - Long Statutory Provision

Required behavior:

- no automatic truncation on Section Detail;
- statutory pane scrolls independently in landscape;
- section identity remains accessible;
- chapter navigation remains usable;
- Smart Insights remains accessible;
- result cards show excerpts rather than complete provisions.

## 35. F14 - Cross-Reference Navigation

```text
Section A
  -> Referenced Provision
  -> Section B
```

Cross-references use source/deterministic styling, not ML purple.

## 36. F15 - Smart Insight Provenance

Every ML-derived insight includes:

> How was this generated?

Provenance explanations are centrally defined and must match the actual approved production implementation.

---

# 37. Mandatory Screen States

| Screen or component | Loading | Empty | Success | No Match | Partial ML | API Error |
|---|---:|---:|---:|---:|---:|---:|
| Explore | Yes | Yes | Yes | N/A | Yes | Yes |
| Smart Search | Yes | Yes | Yes | Yes | Yes | Yes |
| Search Results | Yes | Yes | Yes | Yes | Yes | Yes |
| Section Detail | Yes | N/A | Yes | N/A | Yes | Yes |
| Chapters | Yes | Yes | Yes | N/A | Yes | Yes |
| Insights | Yes | Yes | Yes | Yes | Yes | Yes |
| Bookmarks | Yes | Yes | Yes | N/A | N/A | Yes |

**Partial ML** means source data is available while one or more intelligence features are unavailable.

---

# 38. Frontend ML Insight Requirements

The UI creates requirements for these candidate intelligence capabilities:

1. semantic section retrieval;
2. hybrid lexical-semantic ranking;
3. key-concept extraction;
4. section-to-section similarity;
5. chapter-level aggregation;
6. relevance calibration;
7. explainable match evidence;
8. safe thresholds and no-strong-match handling;
9. keyword fallback;
10. precomputed insight support where appropriate.

These are product requirements, not final algorithm decisions. Each implementation must be validated through ML experiments before being promoted to production.

The existing chapter classifier may be used as a supporting component for chapter suggestions, filtering, query routing, or ranking. It is not the central frontend experience.

---

# 39. Accessibility and Interaction Requirements

- All interactive controls need visible labels or accessible names.
- Navigation and major workflows must support keyboard and screen-reader interaction where the platform permits.
- Color alone must not communicate insight type, relevance, error, or selection.
- Text and controls must maintain sufficient contrast.
- Touch targets must be suitable for tablet interaction.
- Drawers and modal surfaces need clear focus and dismissal behavior.
- Loading and dynamic result changes need accessible announcements.
- Statutory text should support comfortable zoom and text scaling.

---

# 40. Phase 1 Freeze Gates

## UI-01: Product shell and landscape strategy

- Landscape-first research workspace
- Persistent left navigation
- Context-aware top bar
- Optional contextual right panel
- Responsive degradation for smaller screens
- No generic dashboard
- No chatbot-first V1

## UI-02: Navigation and screen inventory

- Explore
- Smart Search
- Chapters
- Insights
- Bookmarks
- About
- Section Detail as the convergence screen

## UI-03: Design system

- Navy/gold legal identity
- Purple reserved for ML intelligence
- Inter for UI
- Source Serif 4 for statutory text
- 8 px spacing system
- Reusable component vocabulary
- Qualitative relevance presentation
- ML provenance available on demand

## UI-04: Core screens

- Application Shell
- Explore
- Smart Search
- Search Results
- Section Detail
- Chapters
- Insights
- Bookmarks
- About

## UI-05: UX flows and operational states

- plain-language research journey;
- direct section lookup;
- explainable relevance;
- related-section exploration;
- concept discovery;
- chapter browsing;
- intelligence-first discovery;
- bookmarking;
- no-strong-match handling;
- ML fallback;
- API and offline errors;
- long statutory text;
- cross-reference navigation;
- insight provenance.

---

# 41. Phase 1 Completion Statement

Phase 1 is considered functionally frozen when the UI-01 through UI-05 decisions in this document are approved and any later change is captured as a documented design decision.

The central architectural principle produced by Phase 1 is:

> LegalVerse must remain a useful BNS explorer when ML functionality is degraded or unavailable, while becoming significantly smarter when the validated intelligence layer is available.

---

# 42. Next Phase Boundary

The next phase is **Backend and Frontend Architecture Design**.

Architecture must be derived from the frozen UX and should map:

```text
React Native screen and interaction
  -> API contract
  -> FastAPI route and schema
  -> application service
  -> retrieval or ML service
  -> data and artifact layer
  -> unit, validation, and integration test boundaries
```

No backend capability should be introduced solely because it is technically interesting. It should support an approved UI flow, an operational requirement, testing, security, observability, or maintainability.

---

# 43. Documentation Update Rule

Major UI decisions and changes must be reflected in:

- `UI_ROADMAP.md`
- `PROJECT_STATUS.md`
- `ARCHITECTURE.md` when architectural impact exists
- `DEVELOPMENT_ROADMAP.md` when scope or sequencing changes
- `ML_EXPERIMENTS.md` when an ML-derived frontend insight is introduced or revised
- `README.md` for major stable milestones

No approved UI or UX decision should be changed silently during implementation.
