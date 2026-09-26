"""Score English pilot refuses unmatched or answer-bearing preparation inputs."""

from __future__ import annotations

import pytest

from training.data.prepare_score_en_pilot import (
    choose_control,
    r2_native_prompts,
    validate_r2_english,
)


def test_control_needs_exact_parent_only_tokens_and_score_floor() -> None:
    candidates = [
        ({"id": f"parent-{index}", "language": "en", "task_type": task_type}, length)
        for index, (task_type, length) in enumerate(
            [
                ("score", 200),
                ("choice", 201),
                ("noul", 202),
                ("choice", 300),
                ("noul", 301),
                ("score", 302),
            ]
        )
    ]
    chosen = choose_control(candidates, count=3, target_tokens=603, minimum_score=1)
    assert len({row["id"] for row, _ in chosen}) == 3
    assert sum(length for _, length in chosen) == 603
    assert any(row["task_type"] == "score" for row, _ in chosen)
    with pytest.raises(ValueError, match="No exact parent-only matched control"):
        choose_control(candidates, count=3, target_tokens=605, minimum_score=2)


def test_r2_packet_never_accepts_answer_fields() -> None:
    packet = []
    operations = (
        "allocation_caps",
        "inclusive_coverage",
        "independent_quorum",
        "waiver_precedence",
    )
    for operation in operations:
        for group in range(20):
            for level in range(3):
                packet.append(
                    {
                        "group_id": f"opaque-{operation}-{group}",
                        "review_id": f"row-{operation}-{group}-{level}",
                        "language": "en" if group < 16 else "zh",
                        "operation": operation,
                        "state": "A synthetic gold-free state",
                        "instructions": "Choose a score",
                        "options": [
                            {"key": str(score), "description": f"Band {score}"}
                            for score in range(3)
                        ],
                    }
                )
    # The fake packet has the correct language, operation and group counts.
    english = validate_r2_english(packet)
    assert len(english) == 192
    native = r2_native_prompts(english)
    assert native[0]["id"] == english[0]["review_id"]
    assert native[0]["questions"]["decision"]["criteria"] == [
        "Band 0",
        "Band 1",
        "Band 2",
    ]
    packet[0]["label"] = 0
    with pytest.raises(ValueError, match="answer fields"):
        validate_r2_english(packet)
