# Phase 4.6A acceptance tests

With FastAPI and Vite running:

1. Click `Insights` in the sidebar.
2. Network should show:
   - `GET /api/chapters`
   - `GET /api/sections`
   - `GET /api/health`
3. The Sections metric must equal the length of the `/api/sections` response.
4. The Chapters metric must equal the length of the `/api/chapters` response.
5. Chapter coverage is calculated by summing `section_count` from the chapter response.
6. The chapter distribution uses real `section_count` values.
7. Keyword/Semantic/Graph capability status mirrors `/api/health`.
8. Stopping the backend should show the retry error state.
9. No concepts, relationships or chapter-intelligence claims are fabricated.
10. Run:

```bash
npm run typecheck
npm run build
```
