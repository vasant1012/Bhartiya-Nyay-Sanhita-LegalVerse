import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ALIASES = {
    "section_number": ["Section", "section"],
    "section_name": ["Section Name", "Section_name", "section_name"],
    "chapter_name": ["Chapter Name", "Chapter_name", "chapter_name"],
    "description": ["Description", "description"],
}


def col(frame, name):
    return next(x for x in ALIASES[name] if x in frame.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bns_sections.csv")
    parser.add_argument(
        "--output-dir", default="backend/artifacts/faiss_indexes")
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    frame = pd.read_csv(args.data, dtype=str).fillna("")
    texts = (
        frame[col(frame, "section_name")]
        + ". "
        + frame[col(frame, "chapter_name")]
        + ". "
        + frame[col(frame, "description")]
    ).tolist()
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "bns_sections.index"))
    metadata = [
        {"section_number": str(x)} for x in frame[col(frame, "section_number")].tolist()
    ]
    (out / "bns_sections.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
