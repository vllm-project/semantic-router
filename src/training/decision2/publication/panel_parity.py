"""Compare gold-free native source and package outputs on complete panels.

This consumes previously sealed predictions. It does not run a model, read
labels, select a checkpoint, or turn development scores into release evidence.
The resulting receipt has the schema accepted by ``bundle_arena``; release
review must separately verify the actual inference runs and their chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from .adapter_parity import _native_values
from .bundle_arena import PARITY_VERSION
from .package_native_arena import (
    _no_duplicate_keys,
    _package_manifest,
    _reject_constant,
    _sha_file,
    input_digest,
    load_gold_free,
)

P99_LIMIT = 0.005
MAX_LIMIT = 0.02
PANEL_ITEMS = {"dev": 1600, "css_pilot": 1430}


def _object(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Evidence must be a regular file")
    payload = path.read_bytes()
    value = json.loads(
        payload,
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("Evidence must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _rows_bytes(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in payload.splitlines():
        row = json.loads(
            line,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in seen
        ):
            raise ValueError("Predictions contain a malformed or duplicate ID")
        seen.add(row["id"])
        rows.append(row)
    return rows


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Predictions must be a regular file")
    return _rows_bytes(path.read_bytes())


def _bound_predictions(
    path: Path,
    *,
    prompts_sha256: str,
    items: int,
    model_sha256: str,
    calibration_sha256: str,
    package_sha256: str | None,
) -> tuple[list[dict[str, Any]], str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Predictions must be a regular file")
    manifest_path = path.with_name(path.name + ".manifest.json")
    prediction_bytes = path.read_bytes()
    prediction_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    manifest, manifest_sha256 = _object(manifest_path)
    if (
        manifest.get("predictions_sha256") != prediction_sha256
        or manifest.get("input_sha256") != prompts_sha256
        or manifest.get("input_items") != items
        or manifest.get("model_sha256") != model_sha256
        or manifest.get("calibration_sha256") != calibration_sha256
    ):
        raise ValueError("Prediction manifest does not bind this panel/model/CAL")
    if (
        package_sha256 is not None
        and manifest.get("package_manifest_sha256") != package_sha256
    ):
        raise ValueError("Package predictions do not bind frozen package bytes")
    rows = _rows_bytes(prediction_bytes)
    if len(rows) != items:
        raise ValueError("Prediction count differs from frozen prompt count")
    return rows, manifest_sha256, prediction_sha256


def _point_and_values(
    question: dict[str, Any], answer: Any
) -> tuple[tuple[str, str | bool | None], list[float]]:
    kind = question.get("type")
    if not isinstance(answer, dict) or answer.get("type") != kind:
        return ("invalid", None), []
    if "error" in answer:
        reason = answer["error"]
        if not isinstance(reason, str) or not reason:
            raise ValueError("Invalid answer has no error reason")
        return (f"error:{reason}", None), []
    native = _native_values(question, answer)
    if native is None:
        return ("invalid", None), []
    return ("ok", native[1]), native[0]


def compare_panel(
    *,
    name: str,
    prompt_path: Path,
    source_path: Path,
    package_path: Path,
    model_sha256: str,
    calibration_sha256: str,
    package_sha256: str,
    expected_items: int,
) -> dict[str, Any]:
    """Compare scorer point decisions and every returned probability/Score value."""
    prompts_sha256 = _sha_file(prompt_path)
    prompts = load_gold_free(prompt_path)
    if len(prompts) != expected_items:
        raise ValueError("Prompt count differs from frozen panel")
    source, source_manifest_sha, source_prediction_sha = _bound_predictions(
        source_path,
        prompts_sha256=prompts_sha256,
        items=expected_items,
        model_sha256=model_sha256,
        calibration_sha256=calibration_sha256,
        package_sha256=None,
    )
    package, package_manifest_sha, package_prediction_sha = _bound_predictions(
        package_path,
        prompts_sha256=prompts_sha256,
        items=expected_items,
        model_sha256=model_sha256,
        calibration_sha256=calibration_sha256,
        package_sha256=package_sha256,
    )
    mismatches: list[str] = []
    drifts: list[float] = []
    invalid_pairs = malformed_pairs = answers = 0
    for prompt, left, right in zip(prompts, source, package, strict=True):
        item_id = prompt["id"]
        digest = input_digest(prompt)
        if any(
            row.get("id") != item_id or row.get("source_input_sha256") != digest
            for row in (left, right)
        ):
            raise ValueError("Prediction order or serialized prompt identity changed")
        questions = prompt["questions"]
        la, ra = left.get("answers"), right.get("answers")
        if (
            not isinstance(la, dict)
            or not isinstance(ra, dict)
            or set(la) != set(questions)
            or set(ra) != set(questions)
        ):
            raise ValueError("Prediction lacks complete native question answers")
        for qid, question in questions.items():
            if not isinstance(question, dict):
                raise ValueError("Malformed frozen question")
            answers += 1
            left_point, left_values = _point_and_values(question, la[qid])
            right_point, right_values = _point_and_values(question, ra[qid])
            if left_point[0] != "ok" or right_point[0] != "ok":
                invalid_pairs += 1
            if left_point[0] == "invalid" or right_point[0] == "invalid":
                malformed_pairs += 1
            changed = left_point != right_point
            if left_values or right_values:
                if len(left_values) != len(right_values):
                    changed = True
                else:
                    drifts.extend(
                        abs(a - b)
                        for a, b in zip(left_values, right_values, strict=True)
                    )
            if changed:
                mismatches.append(f"{item_id}:{qid}")
    if _sha_file(prompt_path) != prompts_sha256:
        raise ValueError("Frozen prompt bytes changed during comparison")
    ordered = sorted(drifts)
    p99 = (
        ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]
        if ordered
        else 0.0
    )
    maximum = max(drifts, default=0.0)
    gate = (
        not mismatches
        and malformed_pairs == 0
        and p99 <= P99_LIMIT
        and maximum <= MAX_LIMIT
    )
    return {
        "panel": name,
        "prompt_sha256": prompts_sha256,
        "source_predictions_sha256": source_prediction_sha,
        "package_predictions_sha256": package_prediction_sha,
        "source_manifest_sha256": source_manifest_sha,
        "package_manifest_sha256": package_manifest_sha,
        "items": expected_items,
        "answers": answers,
        "categorical_mismatch_n": len(mismatches),
        "mismatch_ids": mismatches,
        "invalid_pair_n": invalid_pairs,
        "malformed_pair_n": malformed_pairs,
        "numeric_comparisons": len(drifts),
        "probability_drift_p99": p99,
        "probability_drift_max": maximum,
        "gate_pass": gate,
    }


def _write_private(path: Path, value: dict[str, Any]) -> str:
    if (
        path.exists()
        or path.is_symlink()
        or path.parent.is_symlink()
        or not path.parent.is_dir()
        or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
    ):
        raise ValueError("Output requires a new file in a private 0700 directory")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_file(path)


def build_receipt(
    *,
    package_manifest_path: Path,
    panels: dict[str, tuple[Path, Path, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    """Write private detailed comparisons and one publication-gate receipt."""
    if set(panels) != set(PANEL_ITEMS):
        raise ValueError("Complete DEV and CSS pilot panels are required")
    package, package_sha256 = _object(package_manifest_path)
    package, _ = _package_manifest(
        package_manifest_path.parent,
        package_sha256,
        package["model_id"],
        f"package-sha256:{package_sha256}",
    )
    base = package.get("base")
    if not isinstance(base, dict):
        raise ValueError("Package has no external base identity")
    files = {
        **{
            f"checkpoint/{name}": digest
            for name, digest in package["model_files_sha256"].items()
        },
        **{f"source/{name}": digest for name, digest in base["files_sha256"].items()},
    }
    files_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_panels: dict[str, Any] = {}
    for name, (prompts, source, published) in panels.items():
        report = compare_panel(
            name=name,
            prompt_path=prompts,
            source_path=source,
            package_path=published,
            model_sha256=package["model_sha256"],
            calibration_sha256=package["calibration_sha256"],
            package_sha256=package_sha256,
            expected_items=PANEL_ITEMS[name],
        )
        report_sha = _write_private(output_dir / f"{name}-parity-detail.json", report)
        receipt_panels[name] = {
            key: report[key]
            for key in (
                "prompt_sha256",
                "items",
                "answers",
                "categorical_mismatch_n",
                "invalid_pair_n",
                "probability_drift_p99",
                "probability_drift_max",
                "gate_pass",
            )
        }
        receipt_panels[name]["report_sha256"] = report_sha
    passed = all(panel["gate_pass"] for panel in receipt_panels.values())
    receipt = {
        "schema_version": PARITY_VERSION,
        "status": "passed" if passed else "failed",
        "model_id": package["model_id"],
        "model_revision": f"package-sha256:{package_sha256}",
        "native_model_sha256": package["model_sha256"],
        "model_files_sha256": files_digest,
        "calibration_sha256": package["calibration_sha256"],
        "panels": receipt_panels,
    }
    _write_private(output_dir / "native-parity.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--dev-package", type=Path, required=True)
    parser.add_argument("--css-prompts", type=Path, required=True)
    parser.add_argument("--css-source", type=Path, required=True)
    parser.add_argument("--css-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        package_manifest_path=args.package_manifest,
        panels={
            "dev": (args.dev_prompts, args.dev_source, args.dev_package),
            "css_pilot": (args.css_prompts, args.css_source, args.css_package),
        },
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": receipt["status"], "panels": receipt["panels"]}))


if __name__ == "__main__":
    main()
