"""Hard-CAL transport leaves Choice content and the original fit intact."""

from __future__ import annotations

import json
import math
from pathlib import Path

from research import hard_calibration_screen as screen


def test_transport_applies_only_noul_and_score(tmp_path: Path, monkeypatch) -> None:
    original_fit = tmp_path / "original-fit.json"
    original_fit.write_text(
        json.dumps(
            {
                "model_sha256": "model",
                "temperature_by_type": {"choice": 1, "noul": 2, "score": 0.5},
            }
        ),
        encoding="utf-8",
    )
    hard_fit = tmp_path / "hard-fit.json"
    hard_fit.write_text(
        json.dumps(
            {
                "version": screen.VERSION,
                "model_sha256": "model",
                "original_calibration_sha256": screen.sha256(original_fit),
                "temperature_by_type": {"choice": 1, "noul": 1, "score": 1},
            }
        ),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "one",
                "model_sha256": "model",
                "calibration_sha256": screen.sha256(original_fit),
                "answers": {
                    "choice": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.7, "b": 0.3},
                    },
                    "noul": {"type": "noul", "noul": 0.8},
                    "score": {
                        "type": "score",
                        "score": 0.8,
                        "probabilities": {"0": 0.2, "1": 0.8},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(screen, "MODEL_SHA", "model")
    monkeypatch.setattr(screen, "ORIGINAL_CAL_SHA", screen.sha256(original_fit))
    monkeypatch.setattr(screen, "DEV_PREDICTIONS_SHA", screen.sha256(predictions))
    monkeypatch.setattr(screen, "DEV_COUNTS", {"choice": 1, "noul": 1, "score": 1})
    output = tmp_path / "new.jsonl"
    report = screen.diagnostic_transform(predictions, original_fit, hard_fit, output)
    answers = json.loads(output.read_text(encoding="utf-8"))["answers"]
    assert report["counts"] == {"choice": 1, "noul": 1, "score": 1}
    assert answers["choice"] == {
        "type": "choice",
        "choice": "a",
        "probabilities": {"a": 0.7, "b": 0.3},
    }
    assert math.isclose(answers["noul"]["noul"], 16 / 17)
    score = answers["score"]
    assert math.isclose(score["probabilities"]["1"], 2 / 3)
    assert math.isclose(score["score"], 2 / 3)
    assert "calibration_sha256" not in json.loads(output.read_text(encoding="utf-8"))
