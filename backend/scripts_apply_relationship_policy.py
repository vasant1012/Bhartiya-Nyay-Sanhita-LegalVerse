"""Rebuild Neo4j SEMANTICALLY_RELATED edges using an approved policy file."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.scripts_build_relationships import (
    build_candidates,
    load_faiss_index,
    load_metadata,
    write_relationships,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=Path("backend/artifacts/relationship_policy.json"))
    parser.add_argument("--index", type=Path, default=Path("backend/artifacts/faiss_indexes/bns_sections.index"))
    parser.add_argument("--metadata", type=Path, default=Path("backend/artifacts/faiss_indexes/bns_sections.json"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_policy(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    policy = json.loads(path.read_text(encoding="utf-8"))
    required = {"min_score", "max_rank", "generation_top_k", "status"}
    missing = required - policy.keys()
    if missing:
        raise ValueError(f"Policy missing required fields: {sorted(missing)}")
    if policy["status"] not in {"approved_for_demo", "approved"}:
        raise ValueError(f"Policy status is not approved: {policy['status']}")
    if int(policy["max_rank"]) > int(policy["generation_top_k"]):
        raise ValueError("max_rank cannot exceed generation_top_k")
    return policy


def main() -> None:
    args = parse_args()
    load_dotenv(Path("backend/.env"))
    policy = load_policy(args.policy)
    metadata = load_metadata(args.metadata)
    index = load_faiss_index(args.index)

    # Build enough neighbours for evaluation, then enforce both approved score
    # and rank gates. Rank is based on accepted FAISS neighbours.
    candidates = build_candidates(
        index=index,
        metadata=metadata,
        top_k=int(policy["generation_top_k"]),
        min_score=float(policy["min_score"]),
    )
    filtered = [item for item in candidates if item.rank <= int(policy["max_rank"])]

    print(f"Policy version: {policy.get('policy_version', 'unknown')}")
    print(f"Policy status: {policy['status']}")
    print(f"Minimum score: {policy['min_score']}")
    print(f"Maximum rank: {policy['max_rank']}")
    print(f"Relationships passing policy: {len(filtered)}")
    print(f"Covered source sections: {len({item.source_section for item in filtered})}")

    if args.dry_run:
        print("Dry run complete. Neo4j was not modified.")
        for item in filtered[:10]:
            print(f"  {item.source_section} -> {item.target_section} score={item.score:.6f} rank={item.rank}")
        return

    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise ValueError("NEO4J_PASSWORD is missing")
    merged, missing = write_relationships(
        candidates=filtered,
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=password,
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        batch_size=args.batch_size,
        replace=True,
    )
    print(f"Neo4j relationship rows merged: {merged}")
    print(f"Rows skipped because Section nodes were missing: {missing}")


if __name__ == "__main__":
    main()
