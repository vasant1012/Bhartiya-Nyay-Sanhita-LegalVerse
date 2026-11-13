import json

import pytest

from backend.scripts_select_relationship_policy import evaluate
from backend.scripts_apply_relationship_policy import load_policy


def test_evaluate_policy_metrics():
    rows = [
        {"label": "relevant", "score": 0.9, "rank": 1},
        {"label": "partially_relevant", "score": 0.75, "rank": 2},
        {"label": "not_relevant", "score": 0.6, "rank": 3},
    ]
    result = evaluate(rows, min_score=0.7, max_rank=3)
    assert result["reviewed_pairs"] == 2
    assert result["strict_precision"] == 0.5
    assert result["useful_precision"] == 1.0
    assert result["weighted_quality"] == 0.75


def test_load_policy_rejects_unapproved(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"min_score": 0.7, "max_rank": 3, "generation_top_k": 5, "status": "draft"}))
    with pytest.raises(ValueError, match="not approved"):
        load_policy(path)
