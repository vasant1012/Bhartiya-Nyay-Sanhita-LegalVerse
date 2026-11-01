# LegalVerse Frontend

Desktop-only single-page demo built with React, Vite, TypeScript and Tailwind CSS. No Expo, React Native, Android, iOS or native build tooling.

## Structure

```text
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
└── src/
    ├── App.tsx
    ├── main.tsx
    ├── index.css
    ├── components/
    ├── config/
    ├── features/
    └── services/
```

## Backend

The Vite development server proxies `/api/*` to the FastAPI server at `http://127.0.0.1:8000`.

Current health call:

```text
GET /api/health
```

There is no `/v1` segment.

## Install

Requires Node.js and npm.

```bash
cd frontend
npm install
```

## Run backend

In a separate terminal from `LegalVerse/backend/`:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health
```

## Run frontend

```bash
cd frontend
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`.

## Validate

```bash
npm run typecheck
npm run build
```

The initial Phase 4.1 implementation includes the complete desktop shell, sidebar navigation, API health status and polished Explore landing view. Other feature modules remain placeholders until their Phase 4 step.
