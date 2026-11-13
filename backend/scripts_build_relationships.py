"""Build FAISS-derived SEMANTICALLY_RELATED relationships in Neo4j.

Run from the LegalVerse project root, for example:

    python -m backend.scripts_build_relationships --dry-run
    python -m backend.scripts_build_relationships --replace

The script is idempotent because it uses MERGE for each directed Section-to-Section edge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase


@dataclass(frozen=True)
class RelationshipCandidate:
    source_section: str
    target_section: str
    score: float
    rank: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _score(value: str) -> float:
    parsed = float(value)
    if not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("score must be between -1 and 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Neo4j SEMANTICALLY_RELATED edges from the existing FAISS index."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("backend/artifacts/faiss_indexes/bns_sections.index"),
        help="FAISS index path.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("backend/artifacts/faiss_indexes/bns_sections.json"),
        help="FAISS metadata JSON path.",
    )
    parser.add_argument("--top-k", type=_positive_int, default=5)
    parser.add_argument(
        "--min-score",
        type=_score,
        default=0.55,
        help="Minimum similarity kept as an edge. Validate this experimentally.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=500)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing SEMANTICALLY_RELATED edges before writing the new set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report candidates without connecting to Neo4j.",
    )
    parser.add_argument("--neo4j-uri", default=None)
    parser.add_argument("--neo4j-user", default=None)
    parser.add_argument("--neo4j-password", default=None)
    parser.add_argument("--neo4j-database", default=None)
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("FAISS metadata must be a JSON array")
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Metadata row {position} is not an object")
        section_number = str(item.get("section_number", "")).strip()
        if not section_number:
            raise ValueError(f"Metadata row {position} has no section_number")
        rows.append({**item, "section_number": section_number})
    duplicated = _duplicates(row["section_number"] for row in rows)
    if duplicated:
        raise ValueError(f"Duplicate section_number values in metadata: {sorted(duplicated)}")
    return rows


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    return duplicated


def load_faiss_index(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    import faiss

    return faiss.read_index(str(path))


def build_candidates(
    index,
    metadata: list[dict[str, Any]],
    top_k: int,
    min_score: float,
) -> list[RelationshipCandidate]:
    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS/metadata size mismatch: index={index.ntotal}, metadata={len(metadata)}"
        )
    if index.ntotal == 0:
        return []

    # Reconstruct vectors from the persisted index. IndexFlatIP, used by the
    # existing builder, supports reconstruct_n. This also avoids reloading the
    # sentence-transformer model merely to generate graph edges.
    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except Exception as exc:  # pragma: no cover - depends on FAISS index type
        raise RuntimeError(
            "The FAISS index does not support vector reconstruction. "
            "Rebuild it with an index type that supports reconstruct_n."
        ) from exc

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != index.ntotal:
        raise ValueError("Reconstructed FAISS vectors have an unexpected shape")

    # Query one extra neighbour because each source should retrieve itself.
    search_k = min(index.ntotal, top_k + 1)
    scores, neighbours = index.search(vectors, search_k)

    candidates: list[RelationshipCandidate] = []
    for source_idx in range(index.ntotal):
        source_section = metadata[source_idx]["section_number"]
        accepted_rank = 0
        for raw_score, target_idx in zip(scores[source_idx], neighbours[source_idx]):
            if target_idx < 0 or target_idx >= len(metadata) or target_idx == source_idx:
                continue
            score = float(raw_score)
            if not math.isfinite(score) or score < min_score:
                continue
            target_section = metadata[target_idx]["section_number"]
            if target_section == source_section:
                continue
            accepted_rank += 1
            candidates.append(
                RelationshipCandidate(
                    source_section=source_section,
                    target_section=target_section,
                    score=score,
                    rank=accepted_rank,
                )
            )
            if accepted_rank >= top_k:
                break
    return candidates


def chunks(items: list[RelationshipCandidate], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_relationships(
    candidates: list[RelationshipCandidate],
    uri: str,
    user: str,
    password: str,
    database: str,
    batch_size: int,
    replace: bool,
) -> tuple[int, int]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    merged_total = 0
    missing_total = 0

    merge_query = """
    UNWIND $rows AS row
    OPTIONAL MATCH (source:Section {section_number: row.source_section})
    OPTIONAL MATCH (target:Section {section_number: row.target_section})
    FOREACH (_ IN CASE WHEN source IS NOT NULL AND target IS NOT NULL THEN [1] ELSE [] END |
      MERGE (source)-[relationship:SEMANTICALLY_RELATED]->(target)
      SET relationship.score = row.score,
          relationship.rank = row.rank,
          relationship.source = 'faiss',
          relationship.updated_at = datetime()
    )
    RETURN
      sum(CASE WHEN source IS NOT NULL AND target IS NOT NULL THEN 1 ELSE 0 END) AS merged,
      sum(CASE WHEN source IS NULL OR target IS NULL THEN 1 ELSE 0 END) AS missing
    """

    try:
        with driver.session(database=database) as session:
            if replace:
                session.run(
                    "MATCH ()-[relationship:SEMANTICALLY_RELATED]->() DELETE relationship"
                ).consume()

            for batch in chunks(candidates, batch_size):
                rows = [
                    {
                        "source_section": item.source_section,
                        "target_section": item.target_section,
                        "score": item.score,
                        "rank": item.rank,
                    }
                    for item in batch
                ]
                record = session.run(merge_query, rows=rows).single()
                merged_total += int(record["merged"] if record else 0)
                missing_total += int(record["missing"] if record else 0)
    finally:
        driver.close()

    return merged_total, missing_total


def main() -> None:
    load_dotenv(Path("backend/.env"))
    args = parse_args()

    metadata = load_metadata(args.metadata)
    index = load_faiss_index(args.index)
    candidates = build_candidates(index, metadata, args.top_k, args.min_score)

    unique_sources = len({item.source_section for item in candidates})
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Candidate relationships: {len(candidates)}")
    print(f"Sections with at least one candidate: {unique_sources}")
    print(f"Configuration: top_k={args.top_k}, min_score={args.min_score}")

    if args.dry_run:
        print("Dry run complete. Neo4j was not modified.")
        for item in candidates[:10]:
            print(
                f"  {item.source_section} -> {item.target_section} "
                f"score={item.score:.6f} rank={item.rank}"
            )
        return

    uri = args.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = args.neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    database = args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise ValueError("Neo4j password is missing. Set NEO4J_PASSWORD or pass --neo4j-password.")

    merged, missing = write_relationships(
        candidates=candidates,
        uri=uri,
        user=user,
        password=password,
        database=database,
        batch_size=args.batch_size,
        replace=args.replace,
    )
    print(f"Neo4j relationship rows merged: {merged}")
    print(f"Rows skipped because Section nodes were missing: {missing}")


if __name__ == "__main__":
    main()
