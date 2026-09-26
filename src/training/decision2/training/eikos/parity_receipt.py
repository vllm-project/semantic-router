"""Bind DEV and CSS pilot same-process Eikos parity checks to one receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.eikos.io import atomic_json
from training.model.data import file_sha256


def combine(
    *,
    dev_report: Path,
    css_report: Path,
    dev_prompts: Path,
    css_prompts: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    panels = {}
    for name, report_path, prompts, expected in (
        ("dev", dev_report, dev_prompts, 1600),
        ("css_pilot", css_report, css_prompts, 1430),
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("role")
            != "gold-free direct selected-LoRA versus merged package parity"
            or report.get("inference_variant")
            != "selected_lora_and_merged_same_process"
            or report.get("items") != expected
            or report.get("answers") != expected
            or report.get("prompt_sha256") != file_sha256(prompts)
        ):
            raise ValueError(f"{name}: missing or mismatched direct parity panel")
        panels[name] = {"report_sha256": file_sha256(report_path), "report": report}
    dev, css = (panels[name]["report"] for name in ("dev", "css_pilot"))
    for field in (
        "candidate_manifest_sha256",
        "adapter_weights_sha256",
        "calibration_sha256",
        "selected_checkpoint",
    ):
        if dev[field] != css[field]:
            raise ValueError(f"Direct parity panels disagree on {field}")
    combined = {
        "role": "gold-free package parity across independent DEV and CSS pilot prompts",
        "model_sha256": dev["candidate_manifest_sha256"],
        "selected_adapter_sha256": dev["adapter_weights_sha256"],
        "calibration_sha256": dev["calibration_sha256"],
        "selected_checkpoint": dev["selected_checkpoint"],
        "panels": {
            name: {
                "prompt_sha256": item["report"]["prompt_sha256"],
                "report_sha256": item["report_sha256"],
                "items": item["report"]["items"],
                "categorical_mismatch_n": item["report"]["choice_mismatch_n"],
                "pmax_abs_drift_max": item["report"]["pmax_abs_drift_max"],
                "max_option_probability_drift": item["report"]["probability_drift_max"],
                "gate_pass": item["report"]["predeclared_gate"]["pass"],
            }
            for name, item in panels.items()
        },
        "total_items": dev["items"] + css["items"],
        "total_categorical_mismatches": dev["choice_mismatch_n"]
        + css["choice_mismatch_n"],
        "predeclared_gate_pass": all(
            item["report"]["predeclared_gate"]["pass"] for item in panels.values()
        ),
        "derivation": "two same-process original selected LoRA versus standalone merged candidate native SemIf runs; no gold labels",
    }
    atomic_json(output, combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-report", type=Path, required=True)
    parser.add_argument("--css-report", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            combine(
                dev_report=args.dev_report,
                css_report=args.css_report,
                dev_prompts=args.dev_prompts,
                css_prompts=args.css_prompts,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
