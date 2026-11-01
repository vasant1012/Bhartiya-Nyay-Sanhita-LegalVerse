"""Export and score human evaluation of SEMANTICALLY_RELATED edges.

From project root:
  python -m backend.scripts_evaluate_relationships export
  python -m backend.scripts_evaluate_relationships score --input evaluation/relationships/relationship_review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

VALID_LABELS = {"relevant", "partially_relevant", "not_relevant"}


@dataclass(frozen=True)
class Relationship:
    source_section: str
    source_name: str
    source_description: str
    source_chapter: str
    target_section: str
    target_name: str
    target_description: str
    target_chapter: str
    score: float
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LegalVerse semantic relationships.")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser(
        "export", help="Export a stratified human-review worksheet.")
    export.add_argument("--output", type=Path,
                        default=Path("evaluation/relationships/relationship_review.csv"))
    export.add_argument("--per-band", type=int, default=20)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--threshold", type=float, default=0.55)
    export.add_argument("--high-score", type=float, default=0.80)
    export.add_argument("--middle-score", type=float, default=0.65)
    export.add_argument("--neo4j-uri", default=None)
    export.add_argument("--neo4j-user", default=None)
    export.add_argument("--neo4j-password", default=None)
    export.add_argument("--neo4j-database", default=None)

    score = sub.add_parser(
        "score", help="Score a completed human-review worksheet.")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path(
        "evaluation/relationships/relationship_quality_report.json"))
    return parser.parse_args()


def neo4j_settings(args: argparse.Namespace) -> tuple[str, str, str, str]:
    load_dotenv(Path("backend/.env"))
    uri = args.neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = args.neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    database = args.neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
    if not password:
        raise ValueError("Neo4j password is missing.")
    return uri, user, password, database


def fetch_relationships(uri: str, user: str, password: str, database: str) -> list[Relationship]:
    query = """
    MATCH (source:Section)-[rel:SEMANTICALLY_RELATED]->(target:Section)
    RETURN
      toString(source.section_number) AS source_section,
      coalesce(source.name, '') AS source_name,
      coalesce(source.description, '') AS source_description,
      coalesce(source.chapter_subtype, '') AS source_chapter,
      toString(target.section_number) AS target_section,
      coalesce(target.name, '') AS target_name,
      coalesce(target.description, '') AS target_description,
      coalesce(target.chapter_subtype, '') AS target_chapter,
      toFloat(rel.score) AS score,
      toInteger(rel.rank) AS rank
    ORDER BY score DESC
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    try:
        with driver.session(database=database) as session:
            return [Relationship(**dict(record)) for record in session.run(query)]
    finally:
        driver.close()


def score_band(score: float, threshold: float, middle: float, high: float) -> str:
    if score >= high:
        return "high"
    if score >= middle:
        return "middle"
    if score >= threshold:
        return "threshold_near"
    return "below_threshold"


def stratified_sample(
    relationships: list[Relationship], per_band: int, seed: int,
    threshold: float, middle: float, high: float,
) -> list[tuple[str, Relationship]]:
    if not threshold <= middle <= high:
        raise ValueError("Expected threshold <= middle-score <= high-score")
    grouped: dict[str, list[Relationship]] = defaultdict(list)
    for item in relationships:
        grouped[score_band(item.score, threshold, middle, high)].append(item)
    rng = random.Random(seed)
    selected: list[tuple[str, Relationship]] = []
    for band in ("high", "middle", "threshold_near"):
        pool = grouped[band]
        picked = rng.sample(pool, min(per_band, len(pool)))
        selected.extend((band, item) for item in picked)
    return selected


def export_review(rows: list[tuple[str, Relationship]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "score_band", "source_section", "source_name", "source_chapter",
        "source_description", "target_section", "target_name", "target_chapter",
        "target_description", "similarity_score", "rank", "review_label",
        "review_reason", "reviewer_notes",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (band, item) in enumerate(rows, start=1):
            writer.writerow({
                "sample_id": f"REL-{index:03d}",
                "score_band": band,
                "source_section": item.source_section,
                "source_name": item.source_name,
                "source_chapter": item.source_chapter,
                "source_description": item.source_description,
                "target_section": item.target_section,
                "target_name": item.target_name,
                "target_chapter": item.target_chapter,
                "target_description": item.target_description,
                "similarity_score": f"{item.score:.6f}",
                "rank": item.rank,
                "review_label": "",
                "review_reason": "",
                "reviewer_notes": "",
            })


def load_completed_review(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = []
    invalid = []
    for row in rows:
        label = (row.get("review_label") or "").strip().lower()
        if not label:
            continue
        if label not in VALID_LABELS:
            invalid.append(f"{row.get('sample_id')}: {label}")
            continue
        row["review_label"] = label
        row["similarity_score"] = float(row["similarity_score"])
        row["rank"] = int(row["rank"])
        completed.append(row)
    if invalid:
        raise ValueError("Invalid labels: " + ", ".join(invalid))
    if not completed:
        raise ValueError("No completed review rows found")
    return completed


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = {"relevant": 1.0, "partially_relevant": 0.5, "not_relevant": 0.0}

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        labels = Counter(row["review_label"] for row in group)
        useful = labels["relevant"] + labels["partially_relevant"]
        scores = [float(row["similarity_score"]) for row in group]
        return {
            "reviewed": len(group),
            "label_counts": dict(labels),
            "precision_strict": round(labels["relevant"] / len(group), 4),
            "precision_useful": round(useful / len(group), 4),
            "weighted_quality": round(mean(weights[row["review_label"]] for row in group), 4),
            "score_min": round(min(scores), 6),
            "score_median": round(median(scores), 6),
            "score_max": round(max(scores), 6),
        }

    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_rank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_band[str(row["score_band"])].append(row)
        by_rank[str(row["rank"])].append(row)

    return {
        "overall": metrics(rows),
        "by_score_band": {key: metrics(value) for key, value in sorted(by_band.items())},
        "by_rank": {key: metrics(value) for key, value in sorted(by_rank.items(), key=lambda item: int(item[0]))},
    }


def main() -> None:
    args = parse_args()
    if args.command == "export":
        uri, user, password, database = neo4j_settings(args)
        relationships = fetch_relationships(uri, user, password, database)
        if not relationships:
            raise ValueError(
                "No SEMANTICALLY_RELATED relationships found in Neo4j")
        sample = stratified_sample(
            relationships, args.per_band, args.seed,
            args.threshold, args.middle_score, args.high_score,
        )
        export_review(sample, args.output)
        counts = Counter(band for band, _ in sample)
        print(f"Relationships available: {len(relationships)}")
        print(f"Review rows exported: {len(sample)}")
        print(f"Rows by band: {dict(counts)}")
        print(f"Output: {args.output}")
    else:
        rows = load_completed_review(args.input)
        report = summarize(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
