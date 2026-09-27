"""Prospective, fail-closed requalification of the unchanged clean-v2 Eikos 4B.

Run inside a pinned, private ROCm image on one reserved physical GPU. This
program only touches the already open DEV, CSS *pilot* and public 231 panels.
It never reads the typed FINAL or CSS evaluation labels, trains, or publishes.
An owner-private JSON config supplies paths and image/GPU provenance; no
credentials, host names, or private paths are part of this source file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from inference.run import digest, load_prompts
from scripts.eikos_stable_runtime_v3 import (
    FULL_SCHEMA,
    REPEAT_SCHEMA,
    stable_backend,
)
from scripts.plan_final_eval import EIKOS_PARITY_PANELS, sha_file
from training.eikos.native import selected_checkpoint
from training.eikos.published_infer import MODEL_ID, package_identity

CHECKPOINT = "checkpoint-0232"
PINNED_SHA256 = {
    "package/SHA256SUMS": "7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39",
    "package/calib.json": "6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60",
    "source_release_manifest": "8978a143290508976d8ddffb735416954cca1de779d5524ca4c59d528539d3dd",
    "run/NATIVE_BEST.json": "d46a2d8ff09126a22a851fcc988261dd6b14252ac9eaa00046e52fda10abb459",
    "run/checkpoint-0232/adapter/adapter_model.safetensors": "31e783d97e70715e67e39b935ed662e6bbc7d185ce848649be9c004c6d51cad7",
    "run/checkpoint-0232/adapter/adapter_config.json": "82ad5297859fdd2c1d0372aadf47d28da6f662d46f59bb509475012a9710ac71",
    "dev_prompts": EIKOS_PARITY_PANELS["dev"][0],
    "css_pilot_prompts": EIKOS_PARITY_PANELS["css_pilot"][0],
    "public_panel/prompts.jsonl": "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
    "dev_gold": "c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc",
    "css_pilot_gold": "9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391",
    "public_panel/targets.jsonl": "abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f",
    "public_panel/manifest.json": "e0e7c67701cf05f996d3b4eac09abf1bb3e6bb363cfe007089acd3644a38ed35",
}
SOURCE_MODULES = (
    "training/eikos/published_infer.py",
    "training/eikos/verify_export.py",
    "training/eikos/deterministic_repeat.py",
    "training/eikos/parity_receipt.py",
    "benchmark/score.py",
    "transfer/score.py",
    "jev_arena/jevbench_public.py",
    "scripts/eikos_stable_runtime_v3.py",
    "scripts/requalify_eikos4b_v3.py",
)
EXPECTED_COUNTS = {"dev": 1600, "css": 1430, "public": 231}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")


def _path(config: dict[str, Any], key: str) -> Path:
    raw = config.get(key)
    if not isinstance(raw, str):
        raise ValueError(f"Missing private path: {key}")
    path = Path(raw)
    if not path.is_absolute() or not path.exists():
        raise ValueError(f"Private path must exist and be absolute: {key}")
    return path


def _pinned_files(config: dict[str, Any]) -> dict[str, Path]:
    package, run, public = (
        _path(config, key) for key in ("package", "run", "public_panel")
    )
    paths = {
        "package/SHA256SUMS": package / "SHA256SUMS",
        "package/calib.json": package / "calib.json",
        "source_release_manifest": _path(config, "source_release_manifest"),
        "run/NATIVE_BEST.json": run / "NATIVE_BEST.json",
        "run/checkpoint-0232/adapter/adapter_model.safetensors": run
        / CHECKPOINT
        / "adapter/adapter_model.safetensors",
        "run/checkpoint-0232/adapter/adapter_config.json": run
        / CHECKPOINT
        / "adapter/adapter_config.json",
        "dev_prompts": _path(config, "dev_prompts"),
        "css_pilot_prompts": _path(config, "css_pilot_prompts"),
        "public_panel/prompts.jsonl": public / "prompts.jsonl",
        "dev_gold": _path(config, "dev_gold"),
        "css_pilot_gold": _path(config, "css_pilot_gold"),
        "public_panel/targets.jsonl": public / "targets.jsonl",
        "public_panel/manifest.json": public / "manifest.json",
    }
    for key, expected in PINNED_SHA256.items():
        path = paths[key]
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f"Preregistered input bytes differ: {key}")
    return paths


def _prompt_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows = load_prompts(path)
    if len(rows) != count or len({row["id"] for row in rows}) != count:
        raise ValueError(f"Prompt IDs/count differ from preregistration: {path.name}")
    return rows


def preflight(config_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _json(config_path)
    if config.get("model_id") != MODEL_ID:
        raise ValueError(f"Public model ID must be {MODEL_ID}")
    if config.get("checkpoint") != CHECKPOINT:
        raise ValueError("Selected checkpoint changed")
    image_id = config.get("runtime_image_id")
    if not isinstance(image_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image_id
    ):
        raise ValueError("Require immutable runtime image ID")
    physical_gpu = config.get("physical_gpu")
    if (
        not isinstance(physical_gpu, dict)
        or type(physical_gpu.get("index")) is not int
        or not isinstance(physical_gpu.get("guid"), str)
        or not physical_gpu["guid"]
    ):
        raise ValueError("Require reserved physical GPU identity")
    if not isinstance(config.get("source_commit"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", config["source_commit"]
    ):
        raise ValueError("Require exact local source commit")
    source = _path(config, "source_root")
    if not source.is_dir() or source.resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError("Runner must execute from the exact mirrored source tree")
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    if config.get("device") != "cuda:0" or not visible or "," in visible:
        raise ValueError("Use cuda:0 with one explicitly reserved ROCm GPU")
    paths = _pinned_files(config)
    for key, count in (
        ("dev_prompts", 1600),
        ("css_pilot_prompts", 1430),
        ("public_panel/prompts.jsonl", 231),
    ):
        _prompt_rows(paths[key], count)
    panel = _json(paths["public_panel/manifest.json"])
    if (
        panel.get("items") != 231
        or panel.get("prompts_sha256") != PINNED_SHA256["public_panel/prompts.jsonl"]
    ):
        raise ValueError("Public panel manifest differs")
    package = _path(config, "package")
    rights = (
        _path(config, "rights_attestation")
        if config.get("rights_attestation")
        else None
    )
    identity = package_identity(package, rights_attestation=rights, require_rights=True)
    selection = selected_checkpoint(_path(config, "run"), _path(config, "source_model"))
    if (
        identity["model_sha256"] != PINNED_SHA256["package/SHA256SUMS"]
        or identity["calibration_sha256"] != PINNED_SHA256["package/calib.json"]
        or identity["selected_checkpoint"] != CHECKPOINT
        or selection["name"] != CHECKPOINT
        or selection["adapter_weights_sha256"]
        != PINNED_SHA256["run/checkpoint-0232/adapter/adapter_model.safetensors"]
    ):
        raise ValueError("Package and selected-source identities differ")
    return config, paths


def _source_sha(source: Path) -> dict[str, str]:
    return {name: sha_file(source / name) for name in SOURCE_MODULES}


def _verify_rows(
    predictions: Path, prompts: Path, count: int, package_sha: str
) -> None:
    expected = _prompt_rows(prompts, count)
    with predictions.open(encoding="utf-8") as stream:
        actual = [json.loads(line) for line in stream if line.strip()]
    if len(actual) != count:
        raise ValueError("Package did not answer every original-order prompt")
    for prompt, row in zip(expected, actual):
        visible = {"state": prompt["state"], "questions": prompt["questions"]}
        if (
            row.get("id") != prompt["id"]
            or row.get("source_input_sha256") != digest(visible)
            or row.get("model_id") != MODEL_ID
            or row.get("model_revision") != CHECKPOINT
            or row.get("model_sha256") != package_sha
            or row.get("invalid_reason") is not None
            or set(row.get("answers", {})) != set(prompt["questions"])
            or type(row.get("usage", {}).get("input_tokens")) is not int
            or row["usage"]["input_tokens"] <= 0
        ):
            raise ValueError(f"Invalid/stale native package answer: {prompt['id']}")


def _verify_manifest(
    predictions: Path, prompts: Path, count: int, source: Path
) -> dict[str, Any]:
    value = _json(Path(str(predictions) + ".manifest.json"))
    counts = value.get("counts", {})
    runtime = value.get("runtime", {})
    if (
        value.get("model_id") != MODEL_ID
        or value.get("model_revision") != CHECKPOINT
        or value.get("model_sha256") != PINNED_SHA256["package/SHA256SUMS"]
        or value.get("calibration_sha256") != PINNED_SHA256["package/calib.json"]
        or value.get("collector_source_sha256")
        != sha_file(source / "training/eikos/published_infer.py")
        or value.get("input_sha256") != sha_file(prompts)
        or value.get("predictions_sha256") != sha_file(predictions)
        or value.get("input_items") != count
        or value.get("evaluated_items") != count
        or value.get("max_items") is not None
        or counts.get("items") != count
        or counts.get("valid_questions") != count
        or counts.get("invalid_questions") != 0
        or not stable_backend(runtime)
        or runtime.get("torch") != "2.12.0+git6bbd260"
        or runtime.get("hip") != "7.2.53211"
        or runtime.get("transformers") != "5.17.0"
        or runtime.get("flash_linear_attention") != "0.5.2"
    ):
        raise ValueError("Native package manifest/runtime differs from preregistration")
    _verify_rows(predictions, prompts, count, PINNED_SHA256["package/SHA256SUMS"])
    return value


def _artifacts(directory: Path) -> dict[str, str]:
    return {
        path.name: sha_file(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "execution.receipt.json"
    }


class StageRunner:
    def __init__(self, source: Path, env: dict[str, str]):
        self.source, self.env = source, env

    def __call__(
        self, name: str, module: str, arguments: list[str], directory: Path
    ) -> dict[str, Any]:
        log = directory / f"{name}.process.log"
        if log.exists():
            raise FileExistsError(log)
        start = _now()
        began = time.perf_counter()
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(
                [sys.executable, "-m", module, *arguments],
                cwd=self.source,
                env=self.env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        stage = {
            "exit_code": result.returncode,
            "started_at": start,
            "ended_at": _now(),
            "elapsed_seconds": time.perf_counter() - began,
            "log_sha256": sha_file(log),
            "model_load_warnings": any(
                marker in log.read_text(encoding="utf-8", errors="replace").lower()
                for marker in (
                    "missing key",
                    "unexpected key",
                    "size mismatch",
                    "not initialized",
                )
            ),
        }
        if result.returncode != 0 or stage["model_load_warnings"]:
            raise RuntimeError(f"{name}: native process failed; inspect private log")
        return stage


def _infer_args(config: dict[str, Any], prompts: Path, output: Path) -> list[str]:
    args = [
        "--model-path",
        config["package"],
        "--input",
        str(prompts),
        "--output",
        str(output),
        "--model-id",
        MODEL_ID,
        "--model-revision",
        CHECKPOINT,
        "--device",
        "cuda:0",
        "--deterministic-algorithms",
        "--torch-reference-gated-delta",
    ]
    if config.get("rights_attestation"):
        args += ["--rights-attestation", config["rights_attestation"]]
    return args


def _verify_parity(report_path: Path, prompts: Path, count: int) -> None:
    report = _json(report_path)
    gate = report.get("predeclared_gate", {})
    selected = report_path.with_name(
        report_path.name.replace(".report.json", ".selected.jsonl")
    )
    merged = report_path.with_name(
        report_path.name.replace(".report.json", ".merged.jsonl")
    )
    expected = _prompt_rows(prompts, count)
    for prediction_path in (selected, merged):
        with prediction_path.open(encoding="utf-8") as stream:
            actual = [json.loads(line) for line in stream if line.strip()]
        if len(actual) != count or any(
            row.get("id") != prompt["id"]
            or row.get("source_input_sha256")
            != digest({"state": prompt["state"], "questions": prompt["questions"]})
            or set(row.get("answers", {})) != set(prompt["questions"])
            for prompt, row in zip(expected, actual)
        ):
            raise ValueError(
                "Selected/package parity predictions changed original inputs"
            )
    if (
        report.get("candidate_manifest_sha256") != PINNED_SHA256["package/SHA256SUMS"]
        or report.get("calibration_sha256") != PINNED_SHA256["package/calib.json"]
        or report.get("selected_checkpoint") != CHECKPOINT
        or report.get("prompt_sha256") != sha_file(prompts)
        or report.get("selected_predictions_sha256") != sha_file(selected)
        or report.get("merged_predictions_sha256") != sha_file(merged)
        or report.get("items") != count
        or report.get("answers") != count
        or report.get("choice_mismatch_n") != 0
        or type(report.get("probability_drift_p99")) not in (int, float)
        or not math.isfinite(report["probability_drift_p99"])
        or report["probability_drift_p99"] > 0.005
        or type(report.get("probability_drift_max")) not in (int, float)
        or not math.isfinite(report["probability_drift_max"])
        or report["probability_drift_max"] > 0.02
        or gate.get("pass") is not True
        or not stable_backend(report.get("runtime"))
    ):
        raise ValueError("Same-process selected/package parity failed")


def _receipt(directory: Path, payload: dict[str, Any]) -> None:
    payload["artifact_sha256"] = _artifacts(directory)
    _write_new(directory / "execution.receipt.json", payload)


def execute(
    config_path: Path, output_root: Path, runner: StageRunner | None = None
) -> dict[str, Path]:
    """Execute once into an absent private directory; never retry/overwrite."""
    config, paths = preflight(config_path)
    if output_root.exists() or not output_root.is_absolute():
        raise ValueError("Choose a fresh absolute private output directory")
    source = _path(config, "source_root")
    # The source mirror is immutable for this execution. A caller can compare
    # these hashes with a local checkout without relying on remote Git metadata.
    source_sha = _source_sha(source)
    output_root.mkdir(mode=0o700, parents=True)
    repeat, full = output_root / "repeat", output_root / "full"
    repeat.mkdir(mode=0o700)
    full.mkdir(mode=0o700)
    _write_new(
        output_root / "frozen.inputs.json",
        {
            "schema_version": "decision2-eikos-v3-requalification-inputs/1",
            "model_id": MODEL_ID,
            "checkpoint": CHECKPOINT,
            "source_commit": config["source_commit"],
            "config_sha256": sha_file(config_path),
            "pinned_input_sha256": PINNED_SHA256,
            "source_module_sha256": source_sha,
            "runtime_image_id": config["runtime_image_id"],
            "physical_gpu": config["physical_gpu"],
            "started_at": _now(),
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    run = runner or StageRunner(source, env)
    repeat_receipt: dict[str, Any] = {
        "schema_version": REPEAT_SCHEMA,
        "model_id": MODEL_ID,
        "package_sha256": PINNED_SHA256["package/SHA256SUMS"],
        "calibration_sha256": PINNED_SHA256["package/calib.json"],
        "collector_sha256": source_sha["training/eikos/published_infer.py"],
        "prompt_sha256": PINNED_SHA256["css_pilot_prompts"],
        "runtime_image_id": config["runtime_image_id"],
        "physical_gpu": config["physical_gpu"],
        "original_order_input_digest_and_token_parity": False,
        "predeclared_numeric_repeat_gate_pass": False,
        "gpu_hours": 0.0,
    }
    try:
        for name in ("r1", "r2"):
            output = repeat / f"{name}.predictions.jsonl"
            stage = run(
                name,
                "training.eikos.published_infer",
                _infer_args(config, paths["css_pilot_prompts"], output),
                repeat,
            )
            manifest = _verify_manifest(
                output, paths["css_pilot_prompts"], 1430, source
            )
            repeat_receipt[name] = {
                **stage,
                "items": 1430,
                "valid_questions": manifest["counts"]["valid_questions"],
                "manifest_sha256": sha_file(Path(str(output) + ".manifest.json")),
                "predictions_sha256": sha_file(output),
            }
            repeat_receipt["gpu_hours"] += stage["elapsed_seconds"] / 3600
        result = run(
            "repeat-audit",
            "training.eikos.deterministic_repeat",
            [
                "--predictions-a",
                str(repeat / "r1.predictions.jsonl"),
                "--predictions-b",
                str(repeat / "r2.predictions.jsonl"),
                "--prompts",
                str(paths["css_pilot_prompts"]),
                "--package",
                config["package"],
                "--output",
                str(repeat / "comparison.json"),
                "--require-torch-reference-gated-delta",
            ],
            repeat,
        )
        comparison = _json(repeat / "comparison.json")
        if comparison.get("predeclared_numeric_repeat_gate_pass") is not True:
            raise ValueError(
                "Cross-process categorical/probability repeatability failed"
            )
        repeat_receipt["audit_process"] = result
        repeat_receipt["comparison"] = comparison["comparison"]
        repeat_receipt["original_order_input_digest_and_token_parity"] = True
        repeat_receipt["predeclared_numeric_repeat_gate_pass"] = True
    except Exception as exc:
        repeat_receipt["failure"] = type(exc).__name__ + ": " + str(exc)
        _receipt(repeat, repeat_receipt)
        raise
    _receipt(repeat, repeat_receipt)

    full_receipt: dict[str, Any] = {
        "schema_version": FULL_SCHEMA,
        "model_id": MODEL_ID,
        "runtime_image_id": config["runtime_image_id"],
        "physical_gpu": config["physical_gpu"],
        "all_gates_pass": False,
        "combined_parity_pass": False,
        "processes": {},
        "scores": {},
        "gpu_hours": 0.0,
    }
    try:
        for panel, prompt_key, count in (
            ("dev", "dev_prompts", 1600),
            ("css", "css_pilot_prompts", 1430),
        ):
            output = full / f"parity-{panel}.report.json"
            stage = run(
                f"parity-{panel}",
                "training.eikos.verify_export",
                [
                    "--model-path",
                    config["source_model"],
                    "--run",
                    config["run"],
                    "--merged",
                    config["package"],
                    "--prompts",
                    str(paths[prompt_key]),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                    "--direct-selected",
                    "--selected-predictions",
                    str(full / f"parity-{panel}.selected.jsonl"),
                    "--merged-predictions",
                    str(full / f"parity-{panel}.merged.jsonl"),
                    "--deterministic-algorithms",
                    "--torch-reference-gated-delta",
                ],
                full,
            )
            _verify_parity(output, paths[prompt_key], count)
            full_receipt["processes"][f"parity-{panel}"] = stage
            full_receipt["gpu_hours"] += stage["elapsed_seconds"] / 3600
        stage = run(
            "parity-combine",
            "training.eikos.parity_receipt",
            [
                "--dev-report",
                str(full / "parity-dev.report.json"),
                "--css-report",
                str(full / "parity-css.report.json"),
                "--dev-prompts",
                str(paths["dev_prompts"]),
                "--css-prompts",
                str(paths["css_pilot_prompts"]),
                "--model-id",
                MODEL_ID,
                "--output",
                str(full / "parity-dev-css.receipt.json"),
            ],
            full,
        )
        parity = _json(full / "parity-dev-css.receipt.json")
        if (
            parity.get("model_id") != MODEL_ID
            or parity.get("predeclared_gate_pass") is not True
            or parity.get("total_items") != 3030
            or parity.get("total_categorical_mismatches") != 0
        ):
            raise ValueError("Combined parity gate failed")
        full_receipt["processes"]["parity-combine"] = stage
        full_receipt["combined_parity_pass"] = True

        for panel, prompt_key, gold_key, count in (
            ("dev", "dev_prompts", "dev_gold", 1600),
            ("css", "css_pilot_prompts", "css_pilot_gold", 1430),
            ("public", "public_panel/prompts.jsonl", "public_panel/targets.jsonl", 231),
        ):
            output = full / f"package-{panel}.predictions.jsonl"
            stage = run(
                f"package-{panel}",
                "training.eikos.published_infer",
                _infer_args(config, paths[prompt_key], output),
                full,
            )
            _verify_manifest(output, paths[prompt_key], count, source)
            full_receipt["processes"][f"package-{panel}"] = stage
            full_receipt["gpu_hours"] += stage["elapsed_seconds"] / 3600
            score_path = full / f"package-{panel}.score.json"
            if panel == "dev":
                module = "benchmark.score"
                score_args = [
                    "--gold",
                    str(paths[gold_key]),
                    "--predictions",
                    str(output),
                    "--model-id",
                    MODEL_ID,
                    "--model-revision",
                    CHECKPOINT,
                    "--backend",
                    "eikos-semif-native",
                    "--output",
                    str(score_path),
                ]
            elif panel == "css":
                module = "transfer.score"
                score_args = [
                    "--gold",
                    str(paths[gold_key]),
                    "--predictions",
                    str(output),
                    "--output",
                    str(score_path),
                ]
            else:
                module = "jev_arena.jevbench_public"
                score_args = [
                    "score",
                    "--panel-dir",
                    config["public_panel"],
                    "--predictions",
                    str(output),
                    "--model-id",
                    MODEL_ID,
                    "--model-revision",
                    CHECKPOINT,
                    "--prediction-manifest",
                    str(Path(str(output) + ".manifest.json")),
                    "--output",
                    str(score_path),
                ]
            full_receipt["processes"][f"score-{panel}"] = run(
                f"score-{panel}", module, score_args, full
            )
            score = _json(score_path)
            if score.get("predictions_sha256") != sha_file(output):
                raise ValueError(f"{panel}: score is not bound to prediction bytes")
            valid = (
                score.get("overall", {}).get("valid_n")
                if panel == "dev"
                else (
                    score.get("roles", {}).get("pilot", {}).get("valid_items")
                    if panel == "css"
                    else score.get("strict_valid")
                )
            )
            if valid != count or (panel == "public" and score.get("renormalized") != 0):
                raise ValueError(f"{panel}: designated-panel validity gate failed")
            full_receipt["scores"][
                {"dev": "dev", "css": "css_pilot", "public": "public231"}[panel]
            ] = (
                {"strict_valid": valid, "renormalized": score["renormalized"]}
                if panel == "public"
                else {"valid": valid}
            )
        if _source_sha(source) != source_sha:
            raise ValueError("Source mirror changed during requalification")
        full_receipt["all_gates_pass"] = True
    except Exception as exc:
        full_receipt["failure"] = type(exc).__name__ + ": " + str(exc)
        _receipt(full, full_receipt)
        raise
    _receipt(full, full_receipt)
    return {
        "frozen_inputs": output_root / "frozen.inputs.json",
        "repeat_execution": repeat / "execution.receipt.json",
        "full_execution": full / "execution.receipt.json",
        "full_parity": full / "parity-dev-css.receipt.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Owner-private config JSON"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="New owner-private output directory",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    # Do not print private paths. The caller can inspect the private receipts.
    if args.preflight_only:
        preflight(args.config)
        print(json.dumps({"status": "preflight_pass", "model_id": MODEL_ID}))
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --preflight-only is set")
    execute(args.config, args.output_root)
    print(
        json.dumps({"status": "technical_requalification_pass", "model_id": MODEL_ID})
    )


if __name__ == "__main__":
    main()
