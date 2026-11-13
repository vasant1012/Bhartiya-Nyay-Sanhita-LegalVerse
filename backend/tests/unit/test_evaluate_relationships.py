from backend.scripts_evaluate_relationships import (
    Relationship,
    score_band,
    stratified_sample,
    summarize,
)


def item(number: int, score: float) -> Relationship:
    return Relationship(
        source_section=str(number), source_name=f"S{number}", source_description="source",
        source_chapter="chapter", target_section=str(number + 100), target_name="target",
        target_description="target", target_chapter="chapter", score=score, rank=1,
    )


def test_score_band_boundaries():
    assert score_band(0.80, 0.55, 0.65, 0.80) == "high"
    assert score_band(0.70, 0.55, 0.65, 0.80) == "middle"
    assert score_band(0.55, 0.55, 0.65, 0.80) == "threshold_near"
    assert score_band(0.54, 0.55, 0.65, 0.80) == "below_threshold"


def test_stratified_sample_takes_each_band():
    rows = [item(1, 0.9), item(2, 0.85), item(3, 0.7), item(4, 0.69), item(5, 0.58), item(6, 0.56)]
    sampled = stratified_sample(rows, per_band=1, seed=42, threshold=0.55, middle=0.65, high=0.80)
    assert {band for band, _ in sampled} == {"high", "middle", "threshold_near"}
    assert len(sampled) == 3


def test_summary_computes_strict_and_useful_precision():
    rows = [
        {"review_label": "relevant", "similarity_score": 0.9, "score_band": "high", "rank": 1},
        {"review_label": "partially_relevant", "similarity_score": 0.7, "score_band": "middle", "rank": 1},
        {"review_label": "not_relevant", "similarity_score": 0.56, "score_band": "threshold_near", "rank": 2},
    ]
    report = summarize(rows)
    assert report["overall"]["precision_strict"] == 0.3333
    assert report["overall"]["precision_useful"] == 0.6667
    assert report["overall"]["weighted_quality"] == 0.5
