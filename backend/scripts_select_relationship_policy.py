"""Evaluate threshold/rank policies against a labelled relationship review CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

VALID = {"relevant", "partially_relevant", "not_relevant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("evaluation/relationships/relationship_review_labelled.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/relationships/relationship_policy_analysis.json"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    clean = []
    for row in rows:
        label = (row.get("review_label") or "").strip().lower()
        if label not in VALID:
            continue
        clean.append({
            "label": label,
            "score": float(row["similarity_score"]),
            "rank": int(row["rank"]),
            "score_band": row["score_band"],
        })
    if not clean:
        raise ValueError("No valid labelled review rows found")
    return clean


def evaluate(rows: list[dict[str, Any]], min_score: float, max_rank: int) -> dict[str, Any]:
    selected = [row for row in rows if row["score"] >= min_score and row["rank"] <= max_rank]
    if not selected:
        return {"min_score": min_score, "max_rank": max_rank, "reviewed_pairs": 0}
    relevant = sum(row["label"] == "relevant" for row in selected)
    partial = sum(row["label"] == "partially_relevant" for row in selected)
    rejected = sum(row["label"] == "not_relevant" for row in selected)
    count = len(selected)
    return {
        "min_score": min_score,
        "max_rank": max_rank,
        "reviewed_pairs": count,
        "relevant": relevant,
        "partially_relevant": partial,
        "not_relevant": rejected,
        "strict_precision": round(relevant / count, 4),
        "useful_precision": round((relevant + partial) / count, 4),
        "weighted_quality": round((relevant + 0.5 * partial) / count, 4),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    policies = [evaluate(rows, threshold, rank) for threshold in args.thresholds for rank in args.ranks]
    report = {
        "reviewed_rows": len(rows),
        "important_note": "The worksheet was stratified by score band, so these metrics describe reviewed policy subsets and are not unbiased population estimates.",
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
