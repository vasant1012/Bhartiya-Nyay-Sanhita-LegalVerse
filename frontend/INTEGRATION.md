# LegalVerse Phase 4.3 - Smart Search Integration

## Files

Copy into the current SPA:

```text
src/types/search.ts
src/services/api.ts
src/features/SmartSearch.tsx
src/features/SectionHandoff.tsx
```

`src/services/api.ts` replaces the current version and retains `/api/health` while adding `POST /api/search`.

## Update App.tsx

Add imports:

```tsx
import { SmartSearch } from './features/SmartSearch';
import { SectionHandoff } from './features/SectionHandoff';
```

Keep the existing `searchQuery` from Phase 4.2 and add:

```tsx
const [selectedSection, setSelectedSection] = useState('');
```

Replace the search placeholder render with:

```tsx
page === 'search' ? (
  selectedSection ? (
    <SectionHandoff
      sectionNumber={selectedSection}
      onBack={() => setSelectedSection('')}
    />
  ) : (
    <SmartSearch
      initialQuery={searchQuery}
      onOpenSection={setSelectedSection}
    />
  )
) : ...
```

When leaving Smart Search through sidebar navigation, clear the temporary section handoff if desired:

```tsx
const navigate: Navigate = (nextPage, options) => {
  if (options?.query !== undefined) setSearchQuery(options.query);
  if (nextPage !== 'search') setSelectedSection('');
  setPage(nextPage);
};
```

## Required backend contract

Phase 4.3 expects:

```text
POST /api/search
```

Example request:

```json
{
  "query": "entrusted property",
  "mode": "recommended",
  "top_k": 8,
  "chapter_number": null
}
```

Expected response shape:

```json
{
  "query": "entrusted property",
  "mode_used": "keyword",
  "results": [
    {
      "section": {
        "section_number": "316",
        "section_name": "Criminal breach of trust.",
        "chapter_number": "17",
        "chapter_name": "..."
      },
      "excerpt": "...",
      "score": 0.5,
      "relevance": "strong",
      "evidence": {
        "matched_terms": ["entrusted", "property"],
        "concepts": [],
        "method": "keyword"
      }
    }
  ],
  "partial_ml": false,
  "warnings": []
}
```

## Testing

Run backend and Vite, then validate:

1. Explore query automatically executes when Smart Search opens.
2. Manual query + Search calls `/api/search`.
3. Enter submits.
4. Recommended, Keyword and Semantic can be selected.
5. Results show section, title, chapter, excerpt and relevance.
6. Matched terms render when returned.
7. Why This Matched opens and closes correctly.
8. Partial ML warning appears when backend returns `partial_ml=true`.
9. Backend/search error produces retry UI without breaking the page.
10. Read Section opens the temporary 4.4 handoff.
11. Back to search preserves the existing result state because SmartSearch stays mounted only while the handoff is not active; if App conditionally unmounts SmartSearch, cached result persistence will be completed in 4.4. For 4.3, confirm query preservation at minimum.

Build validation:

```bash
npm run typecheck
npm run build
```
