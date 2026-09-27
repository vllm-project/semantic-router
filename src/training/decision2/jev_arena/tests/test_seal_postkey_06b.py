"""Prediction sealing refuses partial and stale native output."""

import json

import pytest

from inference.run import digest
from jev_arena.seal_postkey_06b import _predictions


def test_prediction_seal_checks_prompt_and_model_identity(tmp_path):
    prompts = [
        {
            "id": "one",
            "state": "A case",
            "questions": {
                "a": {"type": "noul", "criteria": {"true": "yes", "false": "no"}}
            },
        }
    ]
    model = {
        "model_id": "example/model",
        "weight_revision": "revision-1",
        "native_adapter": "native-1",
    }
    valid = {
        "id": "one",
        "model_id": model["model_id"],
        "model_revision": model["weight_revision"],
        "adapter_version": model["native_adapter"],
        "source_input_sha256": digest(
            {"state": prompts[0]["state"], "questions": prompts[0]["questions"]}
        ),
        "answers": {"a": {"type": "noul", "noul": 0.75}},
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    receipt = _predictions(path, prompts, model)
    assert receipt["items"] == receipt["answer_slots"] == 1

    path.write_text(json.dumps({**valid, "model_revision": "stale"}) + "\n")
    with pytest.raises(ValueError, match="provenance"):
        _predictions(path, prompts, model)

    path.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n")
    with pytest.raises(ValueError, match="repeated"):
        _predictions(path, prompts, model)
