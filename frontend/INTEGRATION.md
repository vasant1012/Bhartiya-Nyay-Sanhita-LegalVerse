# Phase 4.5 Chapters integration

## Copy files

```text
src/types/chapter.ts
src/services/api.ts
src/features/chapters/useChapters.ts
src/features/chapters/Chapters.tsx
```

`api.ts` replaces the current version and preserves 4.3/4.4 methods while adding:

```text
GET /api/chapters
GET /api/sections?chapter_number={chapter_number}
```

## Update App.tsx

Add:

```tsx
import { Chapters } from './features/chapters/Chapters';
```

Add chapter section state:

```tsx
const [chapterSection, setChapterSection] = useState('');
```

In `navigate`, clear chapter detail when leaving Chapters:

```tsx
if (nextPage !== 'chapters') setChapterSection('');
```

Render Chapters before the generic Placeholder branch:

```tsx
if (page === 'chapters') {
  if (chapterSection) {
    return (
      <SectionDetail
        sectionNumber={chapterSection}
        onBack={() => setChapterSection('')}
        onOpenSection={setChapterSection}
      />
    );
  }

  return (
    <Chapters
      onOpenSection={setChapterSection}
    />
  );
}
```

This reuses the existing 4.4 SectionDetail and preserves the Chapters component/state when no section is open. Note that conditional replacement does unmount Chapters during SectionDetail; if exact scroll preservation becomes important, lift selectedChapter to App or keep Chapters mounted. For Phase 4.5, selected chapter preservation should be verified; if it resets after Back in your environment, use the optional persistence adjustment below.

## Optional selected chapter persistence

If Back resets to the first chapter, lift the selected chapter into App.tsx and pass controlled props to Chapters. The supplied component currently owns its selected chapter internally for simplicity.

## Acceptance tests

- `GET /api/chapters` appears in Network.
- First returned chapter auto-selects.
- Search chapters works by number/name.
- Clicking chapter triggers `GET /api/sections?chapter_number=...`.
- Section list matches the selected chapter response.
- Clicking a section opens the existing 4.4 SectionDetail.
- SectionDetail triggers source and insights APIs.
- Back returns to Chapters.
- Loading, empty, and retry states work.
- `npm run typecheck` passes.
- `npm run build` passes.
