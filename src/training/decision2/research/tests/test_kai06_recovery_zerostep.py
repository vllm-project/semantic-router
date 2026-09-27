"""Native Kai prediction fields differ by question type."""

import pytest

from research.kai06_recovery_zerostep import compare


def test_zero_step_compare_accepts_all_native_type_schemas():
    base = {
        "question_id": "decision",
        "candidate_ids": ["no", "yes"],
        "input_tokens": 16,
        "probabilities": [0.25, 0.75],
        "logits": [0.0, 1.0],
        "confidence": 0.75,
    }
    examples = [
        (
            {**base, "id": str(i), "type": "Noul", "probability": 0.75}
            if i % 3 == 0
            else (
                {**base, "id": str(i), "type": "Choice", "choice_id": "yes"}
                if i % 3 == 1
                else {
                    **base,
                    "id": str(i),
                    "type": "Score",
                    "choice_id": "yes",
                    "score": 1.0,
                    "expected_value": 0.75,
                }
            )
        )
        for i in range(700)
    ]
    assert compare(examples, examples) == {
        "items": 700,
        "argmax_changes": 0,
        "max_abs_drift": 0.0,
    }
    changed = [dict(row) for row in examples]
    del changed[1]["choice_id"]
    with pytest.raises(ValueError, match="fields changed"):
        compare(examples, changed)
