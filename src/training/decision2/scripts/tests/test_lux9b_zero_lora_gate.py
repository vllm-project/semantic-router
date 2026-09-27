"""A zero-update adapter must preserve the selected answer distributions."""

import pytest
from scripts.lux9b_zero_lora_gate import compare


def _rows() -> list[dict]:
    return [
        {
            "id": str(index),
            "prompt_sha256": str(index).zfill(64),
            "token_ids_sha256": str(index + 1).zfill(64),
            "probabilities": {"a": 0.8, "b": 0.2},
        }
        for index in range(32)
    ]


def test_exact_source_to_zero_lora_passes() -> None:
    assert compare(_rows(), _rows())["status"] == "PASS"


def test_changed_choice_blocks_optimizer_start() -> None:
    fresh = _rows()
    fresh[0]["probabilities"] = {"a": 0.2, "b": 0.8}
    assert compare(_rows(), fresh)["status"] == "FAIL"


def test_changed_native_input_is_rejected() -> None:
    fresh = _rows()
    fresh[0]["token_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="inputs differ"):
        compare(_rows(), fresh)
