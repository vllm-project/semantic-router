"""Bind an existing SemIf source/merged parity run to exact release bytes.

This is a gold-free, read-only verifier. It does not infer, score, or declare a
model releasable. The original parity files and their historical model ID stay
unchanged; the new receipt explicitly records that ID alongside the requested
release ID. Use only with the complete DEV and CSS-pilot runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from publication import bundle_arena as bundle

PANEL_ITEMS = {"dev": 1600, "css_pilot": 1430}
RUN_STEMS = {"dev": "dev", "css_pilot": "css"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Evidence is absent or linked")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Evidence must be an object")
    return value


def _rows(path: Path, count: int) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prediction file is absent or linked")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != count or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Prediction count or shape differs from panel")
    return rows


def _panel(
    run: Path,
    name: str,
    old: dict[str, Any],
    native_sha: str,
    calibration_sha: str,
) -> dict[str, Any]:
    stem, count = RUN_STEMS[name], PANEL_ITEMS[name]
    report_path = run / f"parity-{stem}.report.json"
    report = _json(report_path)
    originals = {
        "selected": run / f"parity-{stem}.selected.predictions.jsonl",
        "merged": run / f"parity-{stem}.merged.predictions.jsonl",
        "fresh": run / f"package-{stem}.predictions.jsonl",
    }
    manifest_path = originals["fresh"].with_name(
        originals["fresh"].name + ".manifest.json"
    )
    manifest = _json(manifest_path)
    if (
        old["panels"][name].get("report_sha256") != _sha(report_path)
        or report.get("candidate_manifest_sha256") != native_sha
        or report.get("calibration_sha256") != calibration_sha
        or report.get("prompt_sha256") != old["panels"][name].get("prompt_sha256")
        or report.get("items") != count
        or report.get("answers") != count
        or report.get("selected_predictions_sha256") != _sha(originals["selected"])
        or report.get("merged_predictions_sha256") != _sha(originals["merged"])
        or report.get("choice_mismatch_n") != 0
        or report.get("probability_drift_p99") != 0
        or report.get("probability_drift_max") != 0
        or report.get("predeclared_gate", {}).get("pass") is not True
        or manifest.get("predictions_sha256") != _sha(originals["fresh"])
        or manifest.get("model_sha256") != native_sha
        or manifest.get("calibration_sha256") != calibration_sha
        or manifest.get("input_sha256") != report.get("prompt_sha256")
        or manifest.get("input_items") != count
        or manifest.get("evaluated_items") != count
    ):
        raise ValueError(
            f"{name}: parity report, original receipt or fresh run differs"
        )
    selected, merged, fresh = (
        _rows(originals[key], count) for key in ("selected", "merged", "fresh")
    )
    seen: set[str] = set()
    for source, packaged, rerun in zip(selected, merged, fresh, strict=True):
        identity = (source.get("id"), source.get("source_input_sha256"))
        if (
            not isinstance(identity[0], str)
            or not identity[0]
            or identity[0] in seen
            or any(
                (row.get("id"), row.get("source_input_sha256")) != identity
                or row.get("model_sha256") != native_sha
                or row.get("calibration_sha256") != calibration_sha
                for row in (source, packaged, rerun)
            )
            or not isinstance(source.get("answers"), dict)
            or source["answers"] != packaged.get("answers")
            or source["answers"] != rerun.get("answers")
        ):
            raise ValueError(
                f"{name}: native answer or input changed across package runs"
            )
        seen.add(identity[0])
    return {
        "items": count,
        "answers": count,
        "categorical_mismatch_n": 0,
        "probability_drift_p99": 0.0,
        "probability_drift_max": 0.0,
        "gate_pass": True,
        "prompt_sha256": report["prompt_sha256"],
        "report_sha256": _sha(report_path),
        "selected_predictions_sha256": _sha(originals["selected"]),
        "merged_predictions_sha256": _sha(originals["merged"]),
        "fresh_package_predictions_sha256": _sha(originals["fresh"]),
        "fresh_package_manifest_sha256": _sha(manifest_path),
    }


def attest(model_dir: Path, run: Path, model_id: str, revision: str) -> dict[str, Any]:
    if bundle.MODEL_ID.fullmatch(model_id) is None:
        raise ValueError("Release model ID is not in the Decision 2.0 family")
    model_dir, run = model_dir.resolve(strict=True), run.resolve(strict=True)
    # The historical native serve.py documents a local-only HTTP default.
    # Preserve its frozen bytes while screening every non-loopback address.
    original_ip_pattern = bundle.IP_ADDRESS
    bundle.IP_ADDRESS = re.compile(
        r"\b(?!127\.0\.0\.1\b)(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
    )
    try:
        files = bundle._inventory(model_dir)
    finally:
        bundle.IP_ADDRESS = original_ip_pattern
    bundle._profile(model_dir, "qwen3.5-semif", files)
    native_sha = files["SHA256SUMS"]
    bundle._native_identity(
        model_dir,
        {
            "native_identity": {
                "scheme": "sha256-file",
                "file": "SHA256SUMS",
                "sha256": native_sha,
            }
        },
        files,
    )
    calibration_sha = files["calib.json"]
    provenance = _json(model_dir / "decision2_provenance.json")
    old_path = run / "parity-dev-css.receipt.json"
    old = _json(old_path)
    if (
        provenance.get("selected_checkpoint") != revision
        or provenance.get("calibration_sha256") != calibration_sha
        or old.get("model_sha256") != native_sha
        or old.get("calibration_sha256") != calibration_sha
        or old.get("selected_checkpoint") != revision
        or old.get("total_items") != sum(PANEL_ITEMS.values())
        or old.get("total_categorical_mismatches") != 0
        or old.get("predeclared_gate_pass") is not True
        or set(old.get("panels", {})) != set(PANEL_ITEMS)
    ):
        raise ValueError("Frozen native package and original parity receipt differ")
    panels = {
        name: _panel(run, name, old, native_sha, calibration_sha)
        for name in PANEL_ITEMS
    }
    historical_manifest = _json(run / "package-dev.predictions.jsonl.manifest.json")
    if historical_manifest.get("model_id") != "llm-semantic-router/dev-2.0-4b":
        raise ValueError("Unexpected historical package model ID")
    inventory = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": bundle.PARITY_VERSION,
        "status": "passed",
        "model_id": model_id,
        "model_revision": revision,
        "native_model_sha256": native_sha,
        "model_files_sha256": inventory,
        "calibration_sha256": calibration_sha,
        "panels": panels,
        "historical_model_id": historical_manifest["model_id"],
        "original_parity_receipt_sha256": _sha(old_path),
        "verification_source_sha256": _sha(Path(__file__)),
        "public_text_exception": "Only literal 127.0.0.1 loopback addresses are allowed in the frozen native runtime; other IPs, private paths and credential patterns remain blocked.",
        "derivation": "Gold-free exact native answer/probability equality: selected source, merged package, and independent fresh package process; original reports and checkpoint bytes unchanged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--parity-run", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = attest(args.model_dir, args.parity_run, args.model_id, args.revision)
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
