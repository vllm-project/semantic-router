"""Gold-free CPU checks for AutoJev package qualification inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import attest_autojev27_v3 as attest


def test_gpu_count_receipt_must_bind_pinned_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, models, external = (
        tmp_path / name for name in ("source", "models", "external")
    )
    for path in (source, models, external):
        path.mkdir()
    count = tmp_path / "native-count.json"
    row = {
        "model_id": attest.MODEL_ID,
        "model_revision": attest.MODEL_REVISION,
        "native_model_sha256": "a" * 64,
        "runtime_source_sha256": "b" * 64,
        "loaded_parameters": 25_000_000_000,
        "input_items": 32,
    }
    count.write_text(json.dumps(row))
    args = [
        "attest_autojev27_v3",
        "--source-root",
        str(source),
        "--model-root",
        str(models),
        "--external-root",
        str(external),
        "--native-count-receipt",
        str(count),
        "--receipt-output",
        str(tmp_path / "receipt.json"),
        "--attestation-output",
        str(tmp_path / "attestation.json"),
    ]
    release = {
        "native_model_sha256": "a" * 64,
        "runtime_source_sha256": "b" * 64,
        "loaded_parameters": 25_000_000_000,
    }
    with (
        patch.object(sys, "argv", args),
        patch.object(attest, "verify_release", return_value=release),
        patch.object(attest, "build_attestation") as build,
    ):
        attest.main()
        build.assert_called_once()
        assert "attested" in capsys.readouterr().out
        row["native_model_sha256"] = "c" * 64
        count.write_text(json.dumps(row))
        with pytest.raises(ValueError, match="GPU count receipt"):
            attest.main()
        build.assert_called_once()
