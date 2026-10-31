from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FaissStore:
    def __init__(self, index_path: Path, metadata_path: Path, model_name: str, enabled: bool = True):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.enabled = enabled
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self.model = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self.enabled and self.index is not None and self.model is not None

    def load(self) -> None:
        if not self.enabled:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            if not self.index_path.exists() or not self.metadata_path.exists():
                self.error = 'FAISS artifacts not built'
                return
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(
                self.metadata_path.read_text(encoding='utf-8'))
            self.model = SentenceTransformer(self.model_name)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.index = None
            self.model = None

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self.available:
            return []
        vector = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({**self.metadata[idx], 'score': float(score)})
        return results
