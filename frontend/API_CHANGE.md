# Required api.ts addition

Phase 4.6A requires the unfiltered sections endpoint in addition to the existing methods.

In `src/services/api.ts`, the `api` object should contain this method:

```ts
sections: () => request<SectionSummary[]>('/sections'),
```

`SectionSummary` is already available from `src/types/chapter.ts` in the 4.5 implementation.

Ensure the import contains:

```ts
import type { ChapterSummary, SectionSummary } from '../types/chapter';
```

Do not remove existing health/search/section/sectionInsights/chapters/sectionsByChapter methods.
