"""Verify the prospectively selected Eikos runtime for v3 candidate locking.

This reads only gold-free development receipts. The historical v2 freeze
checker remains responsible for package, CAL, selection and parity lineage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from scripts.plan_final_eval import EIKOS_ARCHITECTURE, EIKOS_PARITY_PANELS, sha_file

LOCK_FIELD = "v3_eikos_stable_runtime"
FLA_BACKEND = "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule"
TORCH_BACKEND = (
    "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule"
)
REPEAT_SCHEMA = "decision2-eikos-torch-reference-execution/1"
FULL_SCHEMA = "decision2-eikos-torch-reference-full-execution/1"


def stable_backend(runtime: Any) -> bool:
    return isinstance(runtime, dict) and all(
        (
            runtime.get("torch_deterministic_algorithms") is True,
            runtime.get("gated_delta_backend_before") == FLA_BACKEND,
            runtime.get("gated_delta_backend") == TORCH_BACKEND,
        )
    )


def _receipt(spec: Any, name: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError(f"{name}: expected a frozen path and SHA-256")
    if not isinstance(spec["path"], str):
        raise ValueError(f"{name}: expected an absolute path string")
    path = Path(spec["path"])
    digest = spec["sha256"]
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name}: expected an absolute regular file")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or sha_file(path) != digest
    ):
        raise ValueError(f"{name}: frozen SHA-256 differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return path, value


def _artifact(receipt: dict[str, Any], directory: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("Receipt artifact name must be a basename")
    artifacts = receipt.get("artifact_sha256")
    digest = artifacts.get(name) if isinstance(artifacts, dict) else None
    path = directory / name
    if (
        not isinstance(digest, str)
        or path.is_symlink()
        or not path.is_file()
        or sha_file(path) != digest
    ):
        raise ValueError(f"Stable-runtime artifact changed: {name}")
    return path


def _all_artifacts(receipt: dict[str, Any], directory: Path) -> None:
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Stable-runtime execution has no artifact hashes")
    for name in artifacts:
        if not isinstance(name, str):
            raise ValueError("Stable-runtime artifact name must be a string")
        _artifact(receipt, directory, name)


def _manifest(
    receipt: dict[str, Any],
    directory: Path,
    name: str,
    candidate: dict[str, Any],
    collector_sha: str,
    count: int,
    prompt_sha: str | None = None,
) -> None:
    value = json.loads(_artifact(receipt, directory, name).read_text(encoding="utf-8"))
    if (
        value.get("model_sha256") != candidate["model_sha256"]
        or value.get("calibration_sha256") != candidate["calibration_sha256"]
        or value.get("model_revision") != candidate["selected_checkpoint"]
        or value.get("collector_source_sha256") != collector_sha
        or value.get("input_items") != count
        or value.get("evaluated_items") != count
        or value.get("counts", {}).get("items") != count
        or (prompt_sha is not None and value.get("input_sha256") != prompt_sha)
        or not stable_backend(value.get("runtime"))
    ):
        raise ValueError(f"Stable-runtime manifest identity/backend differs: {name}")


def verified_stable_runtime(
    lock_path: Path, candidates: list[dict[str, Any]], source_root: Path
) -> dict[str, Any]:
    """Require the two independent CSS runs and complete fixed-backend parity."""
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    eikos = {
        row["key"]: row
        for row in candidates
        if row.get("architecture") == EIKOS_ARCHITECTURE
    }
    declared = lock.get(LOCK_FIELD, {})
    if not isinstance(declared, dict) or set(declared) != set(eikos):
        raise ValueError("v3 Eikos stable-runtime receipts must match candidate keys")
    collector_sha = sha_file(source_root / "training/eikos/published_infer.py")
    for key, candidate in eikos.items():
        specs = declared[key]
        if not isinstance(specs, dict) or set(specs) != {
            "repeat_execution",
            "full_execution",
            "full_parity",
        }:
            raise ValueError(f"{key}: require three stable-runtime receipt locks")
        repeat_path, repeat = _receipt(specs["repeat_execution"], "repeat_execution")
        full_path, full = _receipt(specs["full_execution"], "full_execution")
        parity_path, parity = _receipt(specs["full_parity"], "full_parity")
        _all_artifacts(repeat, repeat_path.parent)
        _all_artifacts(full, full_path.parent)
        comparison = repeat.get("comparison", {})
        drift = comparison.get("max_option_probability_drift")
        if (
            repeat.get("schema_version") != REPEAT_SCHEMA
            or repeat.get("package_sha256") != candidate["model_sha256"]
            or repeat.get("calibration_sha256") != candidate["calibration_sha256"]
            or repeat.get("collector_sha256") != collector_sha
            or repeat.get("prompt_sha256") != EIKOS_PARITY_PANELS["css_pilot"][0]
            or repeat.get("predeclared_numeric_repeat_gate_pass") is not True
            or repeat.get("original_order_input_digest_and_token_parity") is not True
            or comparison.get("items") != 1430
            or comparison.get("categorical_mismatch_n") != 0
            or type(drift) not in (int, float)
            or not math.isfinite(drift)
            or drift > 1e-6
            or full.get("schema_version") != FULL_SCHEMA
            or full.get("all_gates_pass") is not True
            or full.get("combined_parity_pass") is not True
            or full.get("runtime_image_id") != repeat.get("runtime_image_id")
            or full.get("physical_gpu") != repeat.get("physical_gpu")
            or parity_path != full_path.parent / "parity-dev-css.receipt.json"
            or full.get("artifact_sha256", {}).get(parity_path.name)
            != specs["full_parity"]["sha256"]
            or parity.get("model_sha256") != candidate["model_sha256"]
            or parity.get("calibration_sha256") != candidate["calibration_sha256"]
            or parity.get("selected_checkpoint") != candidate["selected_checkpoint"]
            or parity.get("predeclared_gate_pass") is not True
            or parity.get("total_items") != 3030
            or parity.get("total_categorical_mismatches") != 0
        ):
            raise ValueError(f"{key}: stable-runtime receipt identity/gate differs")
        _artifact(full, full_path.parent, parity_path.name)
        for run in ("r1", "r2"):
            item = repeat.get(run, {})
            if (
                item.get("exit_code") != 0
                or item.get("items") != 1430
                or item.get("valid_questions") != 1430
                or item.get("model_load_warnings") is not False
                or item.get("manifest_sha256")
                != repeat["artifact_sha256"].get(
                    f"{run}.predictions.jsonl.manifest.json"
                )
                or item.get("predictions_sha256")
                != repeat["artifact_sha256"].get(f"{run}.predictions.jsonl")
            ):
                raise ValueError(f"{key}: independent {run} receipt failed")
            _manifest(
                repeat,
                repeat_path.parent,
                f"{run}.predictions.jsonl.manifest.json",
                candidate,
                collector_sha,
                1430,
                EIKOS_PARITY_PANELS["css_pilot"][0],
            )
        for panel, (prompt_sha, count) in EIKOS_PARITY_PANELS.items():
            name = f"parity-{'css' if panel == 'css_pilot' else panel}.report.json"
            report_path = Path(candidate["parity_reports"][panel])
            if report_path != full_path.parent / name:
                raise ValueError(
                    f"{key}: candidate parity path is not stable-runtime evidence"
                )
            report = json.loads(
                _artifact(full, full_path.parent, name).read_text(encoding="utf-8")
            )
            panel_receipt = parity.get("panels", {}).get(panel, {})
            if (
                panel_receipt.get("report_sha256") != sha_file(report_path)
                or panel_receipt.get("prompt_sha256") != prompt_sha
                or panel_receipt.get("items") != count
                or panel_receipt.get("gate_pass") is not True
                or report.get("candidate_manifest_sha256") != candidate["model_sha256"]
                or report.get("calibration_sha256") != candidate["calibration_sha256"]
                or report.get("selected_checkpoint") != candidate["selected_checkpoint"]
                or report.get("prompt_sha256") != prompt_sha
                or report.get("items") != count
                or report.get("predeclared_gate", {}).get("pass") is not True
                or not stable_backend(report.get("runtime"))
            ):
                raise ValueError(f"{key}: {panel} stable-runtime parity differs")
        for panel, count in (("dev", 1600), ("css", 1430), ("public", 231)):
            process = full.get("processes", {}).get(f"package-{panel}", {})
            if (
                process.get("exit_code") != 0
                or process.get("model_load_warnings") is not False
            ):
                raise ValueError(f"{key}: package-{panel} process failed")
            _manifest(
                full,
                full_path.parent,
                f"package-{panel}.predictions.jsonl.manifest.json",
                candidate,
                collector_sha,
                count,
                (
                    EIKOS_PARITY_PANELS["css_pilot" if panel == "css" else panel][0]
                    if panel != "public"
                    else None
                ),
            )
        if (
            full.get("scores", {}).get("dev", {}).get("valid") != 1600
            or full.get("scores", {}).get("css_pilot", {}).get("valid") != 1430
            or full.get("scores", {}).get("public231", {}).get("strict_valid") != 231
            or full.get("scores", {}).get("public231", {}).get("renormalized") != 0
        ):
            raise ValueError(f"{key}: fixed-backend designated-panel validity failed")
    return declared
