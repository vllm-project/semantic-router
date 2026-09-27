"""Technical parity checks for the native checkpoint reload comparator."""

from scripts.verify_native_select_reload import compare


def _row(name: str, kind: str, answer: dict, prediction: str) -> dict:
    return {
        "id": name,
        "task_type": kind,
        "prompt_sha256": name + "-prompt",
        "token_ids_sha256": name + "-tokens",
        "answer": {"type": kind, **answer},
        "prediction_key": prediction,
    }


def test_compare_three_native_types() -> None:
    original = [
        _row("c", "choice", {"probabilities": {"a": 0.7, "b": 0.3}}, "a"),
        _row("n", "noul", {"noul": 0.8}, "true"),
        _row("s", "score", {"probabilities": {"0": 0.2, "1": 0.8}}, "1"),
    ]
    result = compare(original, [dict(row) for row in original], 3)
    assert result["status"] == "PASS"
    assert result["categorical_changes"] == 0
    assert result["by_type"] == {"choice": 1, "noul": 1, "score": 1}


def test_compare_detects_probability_and_identity_drift() -> None:
    original = [_row("c", "choice", {"probabilities": {"a": 0.7, "b": 0.3}}, "a")]
    shifted = [_row("c", "choice", {"probabilities": {"a": 0.6, "b": 0.4}}, "a")]
    assert compare(original, shifted, 1)["status"] == "FAIL"
    shifted[0]["token_ids_sha256"] = "different"
    try:
        compare(original, shifted, 1)
    except ValueError as exc:
        assert "native input changed" in str(exc)
    else:
        raise AssertionError("Token identity drift was accepted")
