from dataclasses import dataclass

import numpy as np

from backend.scripts_build_relationships import build_candidates


@dataclass
class FakeIndex:
    ntotal: int = 3

    def reconstruct_n(self, start: int, count: int):
        assert start == 0
        assert count == 3
        return np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def search(self, vectors, search_k):
        assert vectors.shape == (3, 2)
        assert search_k == 3
        return (
            np.array(
                [
                    [1.0, 0.90, 0.20],
                    [1.0, 0.90, 0.30],
                    [1.0, 0.30, 0.20],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 1, 2],
                    [1, 0, 2],
                    [2, 1, 0],
                ],
                dtype=np.int64,
            ),
        )


def test_build_candidates_excludes_self_and_applies_threshold():
    metadata = [
        {"section_number": "1"},
        {"section_number": "2"},
        {"section_number": "3"},
    ]

    result = build_candidates(
        index=FakeIndex(),
        metadata=metadata,
        top_k=2,
        min_score=0.55,
    )

    assert [(item.source_section, item.target_section) for item in result] == [
        ("1", "2"),
        ("2", "1"),
    ]
    assert all(item.source_section != item.target_section for item in result)
    assert all(item.rank == 1 for item in result)
