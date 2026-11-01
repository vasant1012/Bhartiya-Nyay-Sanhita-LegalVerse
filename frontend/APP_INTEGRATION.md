# App.tsx integration

Add:

```tsx
import { CorpusOverview } from './features/insights/CorpusOverview';
```

In `renderContent()`, before the generic Placeholder return, add:

```tsx
if (page === 'insights') {
  return <CorpusOverview />;
}
```

Keep all existing Explore, Smart Search, Section Detail and Chapters branches unchanged.
