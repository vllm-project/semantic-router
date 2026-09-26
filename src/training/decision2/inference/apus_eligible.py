"""Plan and audit a separate APUS final-evaluation eligibility appendix.

Planning reads only the saved main evaluation plan, its pre-test freeze, and
gold-free prompt files. It emits commands but runs no model or scorer. After
the final evaluation is independently executed, ``summarize`` audits frozen
prediction/report bytes and reads no gold itself.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from transfer.build import EVALUATION_TASKS

from .apus import ADAPTER_VERSION, MODELS
from .run import digest, file_digest, load_prompts

PLAN_VERSION = "apus-native-eligible-final-plan/1"
RESULT_VERSION = "apus-native-eligible-final-report/1"
MAIN_PLAN_VERSION = "decision2-final-evaluation-plan/2"
FREEZE_VERSION = "decision2-pretest-freeze/1"
CSS_SHA256 = "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6"
APUS_ADAPTER_SHA256 = "9e2e178d210c42d407bec14b7521cc8554e5dde0dac630ce5669b3488833c52e"
RELEASES = {
    "4b": {
        "release_manifest_sha256": "9e8263363fe13fa815c0acf0bdd7d5dd5d7da4dcf430dba8b96c7ae1d57c5f6f",
        "model_config_sha256": "4b53bdb886bb68d228841409c2db2f5d7eb670c28ae4376849f58d3bd014968d",
    },
    "9b": {
        "release_manifest_sha256": "45806c9cc88a4782e0fcb0b493b94bbdac184038ecc6ba5400688f7c8f7df3a2",
        "model_config_sha256": "b1d02fb6b40aeba43105ca558b2532f31d062e0bd95bce564d9996c0a34b4211",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sh(*parts: Any) -> str:
    return shlex.join(str(part) for part in parts)


def source_command(source_root: Path, *parts: Any) -> str:
    return f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={shlex.quote(str(source_root))} {sh(*parts)}"


def check_freeze_binding(main: dict[str, Any], freeze_path: Path) -> str:
    freeze_sha = file_digest(freeze_path)
    if (
        main.get("freeze_manifest_path") != str(freeze_path)
        or main.get("freeze_manifest_sha256") != freeze_sha
    ):
        raise ValueError(
            "Main plan does not bind the supplied pre-test freeze bytes/path"
        )
    freeze = read_json(freeze_path)
    if freeze.get("freeze_version") != FREEZE_VERSION:
        raise ValueError("Unexpected pre-test freeze schema")
    sources = freeze.get("selection_sources")
    allowed = {"train", "select", "cal", "synthetic_dev", "css_pilot"}
    if (
        not isinstance(sources, list)
        or len(sources) != len(set(sources))
        or not {"select", "cal"} <= set(sources)
        or not set(sources) <= allowed
    ):
        raise ValueError("Freeze selection includes final data or lacks SELECT/CAL")
    entries = freeze.get("candidates")
    declared = main.get("frozen_candidates")
    if not isinstance(entries, list) or not entries or not isinstance(declared, list):
        raise ValueError("Missing frozen Decision 2.0 candidate inventory")
    by_key = {entry.get("key"): entry for entry in entries if isinstance(entry, dict)}
    if (
        len(by_key) != len(entries)
        or len(declared) != len(entries)
        or {item.get("key") for item in declared} != set(by_key)
    ):
        raise ValueError("Freeze candidate set differs from main plan")
    for item in declared:
        entry = by_key[item["key"]]
        for key in (
            "model_id",
            "selected_checkpoint",
            "model_sha256",
            "calibration_sha256",
            "best_sha256",
            "complete_sha256",
            "provenance_sha256",
        ):
            if entry.get(key) != item.get(key):
                raise ValueError(
                    f"{item['key']}: main plan candidate differs from freeze"
                )
    return freeze_sha


def panel_info(path: Path, *, final: bool) -> dict[str, Any]:
    rows = load_prompts(path)  # Explicitly rejects gold-bearing input.
    expected = 1600 if final else 6547
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} gold-free {'final' if final else 'CSS'} items"
        )
    counts: Counter[str] = Counter()
    tasks: set[str] = set()
    for row in rows:
        for question in row["questions"].values():
            qtype = question.get("type")
            if qtype not in {"choice", "noul", "score"} or (
                not final and qtype != "choice"
            ):
                raise ValueError("Unexpected frozen panel question type")
            if qtype == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
                    raise ValueError(
                        "Frozen panel exceeds APUS's 2..16 Choice candidates"
                    )
            if qtype == "noul" and set(question.get("criteria", {})) != {
                "true",
                "false",
            }:
                raise ValueError(
                    "Frozen panel Noul criteria differ from native binary admission"
                )
            counts[qtype] += 1
        if not final:
            parts = row["id"].split("/")
            if len(parts) != 3 or parts[0] != "css":
                raise ValueError("Unexpected CSS evaluation item ID")
            tasks.add(parts[1])
    if final and not ({"choice", "noul", "score"} <= set(counts)):
        raise ValueError("Final panel lacks an expected typed slice")
    if not final and tasks != set(EVALUATION_TASKS):
        raise ValueError("CSS prompt task inventory differs from frozen 15 tasks")
    return {
        "path": str(path),
        "sha256": file_digest(path),
        "items": len(rows),
        "questions": sum(counts.values()),
        "question_types": dict(sorted(counts.items())),
        "eligible_questions": counts["choice"] + counts["noul"],
        "unsupported_score_questions": counts["score"],
        **({"task_count": len(tasks)} if not final else {}),
    }


def source_hashes(source_root: Path, main: dict[str, Any]) -> dict[str, str]:
    protocol = main.get("protocol_source_sha256")
    if not isinstance(protocol, dict):
        raise ValueError("Main plan lacks pinned scorer/protocol sources")
    expected = {"inference/apus.py": APUS_ADAPTER_SHA256}
    for name in (
        "inference/run.py",
        "benchmark/score.py",
        "transfer/score.py",
        "transfer/build.py",
    ):
        if not sha(protocol.get(name)):
            raise ValueError(f"Main plan lacks pinned {name}")
        expected[name] = protocol[name]
    found = {name: file_digest(source_root / name) for name in expected}
    if found != expected:
        raise ValueError("APUS adapter or main protocol source changed after freeze")
    found["inference/apus_eligible.py"] = file_digest(
        source_root / "inference/apus_eligible.py"
    )
    return found


def validate_inputs(
    *,
    main_plan: Path,
    main_plan_sha256: str,
    freeze: Path,
    final_prompts: Path,
    css_prompts: Path,
    source_root: Path,
    output_root: Path,
    plan_path: Path,
) -> dict[str, Any]:
    paths = (
        main_plan,
        freeze,
        final_prompts,
        css_prompts,
        source_root,
        output_root,
        plan_path,
    )
    if any(not path.is_absolute() for path in paths):
        raise ValueError("APUS final appendix paths must be absolute")
    if not sha(main_plan_sha256) or file_digest(main_plan) != main_plan_sha256:
        raise ValueError("Main final plan SHA-256 does not match the saved plan")
    main = read_json(main_plan)
    if (
        main.get("plan_version") != MAIN_PLAN_VERSION
        or main.get("status") != "commands_only_not_executed"
    ):
        raise ValueError("Unexpected main final plan version/status")
    freeze_sha = check_freeze_binding(main, freeze)
    evaluation_root = Path(main["raw_prediction_hashes_path"]).parent
    if final_prompts != evaluation_root / "final.prompts.jsonl":
        raise ValueError(
            "Synthetic final prompt path differs from the main frozen plan"
        )
    css_info = main.get("css_evaluation_prompts", {})
    if css_info.get("path") != str(css_prompts) or css_info.get("sha256") != CSS_SHA256:
        raise ValueError("CSS prompt identity differs from the main frozen plan")
    if plan_path.is_relative_to(output_root) or not plan_path.parent.is_dir():
        raise ValueError("Save the APUS plan outside the new appendix output directory")
    if output_root.exists() or plan_path.exists():
        raise FileExistsError("Use fresh APUS appendix output and saved-plan paths")
    final_panel = panel_info(final_prompts, final=True)
    css_panel = panel_info(css_prompts, final=False)
    if css_panel["sha256"] != CSS_SHA256 or css_info.get("items") != css_panel["items"]:
        raise ValueError("CSS evaluation prompt bytes/count differ from freeze")
    return {
        "main_plan_sha256": main_plan_sha256,
        "freeze_manifest_sha256": freeze_sha,
        "evaluation_root": str(evaluation_root),
        "final_panel": final_panel,
        "css_panel": css_panel,
        "source_sha256": source_hashes(source_root, main),
    }


def build_plan(
    *,
    inputs: dict[str, Any],
    main_plan: Path,
    freeze: Path,
    source_root: Path,
    model_root: Path,
    output_root: Path,
    plan_path: Path,
    python: str,
) -> dict[str, Any]:
    if any(not path.is_absolute() for path in (model_root,)) or not python:
        raise ValueError("Model root must be absolute and Python executable nonempty")
    models: list[dict[str, Any]] = []
    preflight: list[str] = []
    inference: list[str] = []
    scoring: list[str] = []
    prediction_paths: list[Path] = []
    for size in ("4b", "9b"):
        model_id, revision, _ = MODELS[size]
        model_path = model_root / f"APUS-OpenJev-v1-{size.upper()}"
        preflight.append(
            source_command(
                source_root,
                python,
                "-m",
                "inference.apus",
                "--size",
                size,
                "--model-path",
                model_path,
                "--model-revision",
                revision,
                "--verify-only",
            )
        )
        predictions: dict[str, str] = {}
        reports: dict[str, str] = {}
        for panel, prompts in (
            ("final", Path(inputs["final_panel"]["path"])),
            ("css-evaluation", Path(inputs["css_panel"]["path"])),
        ):
            pred = output_root / "predictions" / f"apus{size}.{panel}.predictions.jsonl"
            report = output_root / "reports" / f"apus{size}.{panel}.score.v2.json"
            predictions[panel], reports[panel] = str(pred), str(report)
            prediction_paths.append(pred)
            env = (
                'ROCR_VISIBLE_DEVICES="${GPU_ID}" HF_HUB_OFFLINE=1 '
                "TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
            )
            inference.append(
                env
                + source_command(
                    source_root,
                    python,
                    "-m",
                    "inference.apus",
                    "--size",
                    size,
                    "--model-path",
                    model_path,
                    "--model-revision",
                    revision,
                    "--effort",
                    "high",
                    "--input",
                    prompts,
                    "--output",
                    pred,
                )
            )
            if panel == "final":
                score_args = (
                    "-m",
                    "benchmark.score",
                    "--gold",
                    Path(inputs["evaluation_root"]) / "final.gold.jsonl",
                    "--predictions",
                    pred,
                    "--model-id",
                    model_id,
                    "--model-revision",
                    revision,
                    "--backend",
                    "native-openjet-high",
                    "--output",
                    report,
                )
            else:
                score_args = (
                    "-m",
                    "transfer.score",
                    "--gold",
                    Path(inputs["css_panel"]["path"]).with_name(
                        "css-evaluation.gold.jsonl"
                    ),
                    "--predictions",
                    pred,
                    "--output",
                    report,
                )
            scoring.append(
                sh("test", "!", "-e", report)
                + " && "
                + source_command(source_root, python, *score_args)
            )
        models.append(
            {
                "size": size,
                "model_id": model_id,
                "revision": revision,
                "model_path": str(model_path),
                "effort": "high",
                "adapter_version": ADAPTER_VERSION,
                **RELEASES[size],
                "predictions": predictions,
                "reports": reports,
            }
        )
    raw_hashes = output_root / "RAW_PREDICTIONS.sha256"
    return {
        "plan_version": PLAN_VERSION,
        "status": "commands_only_not_executed",
        "scope": "APUS native Choice+Noul eligible appendix and CSS Choice; excluded from all-type rank",
        "main_plan_path": str(main_plan),
        "freeze_manifest_path": str(freeze),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "plan_path": str(plan_path),
        **inputs,
        "models": models,
        "required_report_versions": {
            "final": "typed-decision-report/2",
            "css_evaluation": "css-transfer-score/2",
        },
        "preparation_command": sh(
            "install",
            "-d",
            "-m",
            "700",
            output_root,
            output_root / "predictions",
            output_root / "reports",
        ),
        "preflight_commands": preflight,
        "inference_commands": inference,
        "raw_prediction_hash_command": sh("test", "!", "-e", raw_hashes)
        + " && "
        + sh("sha256sum", *prediction_paths)
        + " > "
        + shlex.quote(str(raw_hashes)),
        "raw_prediction_hashes_path": str(raw_hashes),
        "scoring_commands": scoring,
        "summary_command_template": source_command(
            source_root,
            python,
            "-m",
            "inference.apus_eligible",
            "summarize",
            "--plan",
            plan_path,
        )
        + ' --expected-plan-sha256 "${APUS_PLAN_SHA256}"',
        "separation_rule": "Never add APUS rows to the all-type rank/matrix; Score has no native ordinal API. Report full denominator and Choice+Noul eligible denominator separately.",
    }


def validate_prediction_file(path: Path, prompts: Path, model: dict[str, Any]) -> str:
    expected = {row["id"]: row for row in load_prompts(prompts)}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            receipt = json.loads(line)
            item_id = receipt.get("id")
            if item_id not in expected or item_id in seen:
                raise ValueError(f"{path}:{number}: unknown or duplicate ID")
            row = expected[item_id]
            identity = {
                "model_id": model["model_id"],
                "model_revision": model["revision"],
                "adapter_version": model["adapter_version"],
                "effort": model["effort"],
                "release_manifest_sha256": model["release_manifest_sha256"],
                "model_config_sha256": model["model_config_sha256"],
            }
            if any(receipt.get(key) != value for key, value in identity.items()):
                raise ValueError(f"{path}:{number}: APUS model identity changed")
            if receipt.get("source_input_sha256") != digest(
                {"state": row["state"], "questions": row["questions"]}
            ):
                raise ValueError(f"{path}:{number}: prediction input hash changed")
            answers = receipt.get("answers")
            if not isinstance(answers, dict) or set(answers) != set(row["questions"]):
                raise ValueError(f"{path}:{number}: prediction answer map changed")
            for key, question in row["questions"].items():
                if question["type"] == "score" and answers[key] != {
                    "type": "score",
                    "error": "unsupported_native_ordinal_score",
                }:
                    raise ValueError(
                        f"{path}:{number}: native Score was falsely presented as supported"
                    )
            seen.add(item_id)
    if seen != set(expected):
        raise ValueError(
            f"{path}: missing {len(set(expected) - seen)} APUS predictions"
        )
    return file_digest(path)


def model_summary(model: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    final_prompt = Path(plan["final_panel"]["path"])
    css_prompt = Path(plan["css_panel"]["path"])
    final_pred = Path(model["predictions"]["final"])
    css_pred = Path(model["predictions"]["css-evaluation"])
    final_pred_sha = validate_prediction_file(final_pred, final_prompt, model)
    css_pred_sha = validate_prediction_file(css_pred, css_prompt, model)
    final_report = read_json(Path(model["reports"]["final"]))
    css_report = read_json(Path(model["reports"]["css-evaluation"]))
    if (
        final_report.get("schema_version") != "typed-decision-report/2"
        or final_report.get("split") != "final"
        or final_report.get("model")
        != {
            "id": model["model_id"],
            "revision": model["revision"],
            "backend": "native-openjet-high",
        }
        or final_report.get("predictions_sha256") != final_pred_sha
        or final_report.get("items") != plan["final_panel"]["items"]
        or final_report.get("predicted_items") != plan["final_panel"]["items"]
    ):
        raise ValueError("APUS synthetic-final v2 score report identity/count differs")
    counts = plan["final_panel"]["question_types"]
    by_type = final_report.get("by_type", {})
    if set(by_type) != set(counts) or any(
        by_type[key].get("n") != value for key, value in counts.items()
    ):
        raise ValueError("APUS final typed question counts differ from frozen prompts")
    all_metrics = final_report.get("overall", {})
    score = by_type["score"]
    if (
        all_metrics.get("n") != plan["final_panel"]["questions"]
        or score.get("valid_n") != 0
        or score.get("invalid_or_missing_n") != counts["score"]
        or score.get("correct_n") != 0
    ):
        raise ValueError("APUS final full denominator/unsupported Score audit failed")
    eligible_n = plan["final_panel"]["eligible_questions"]
    eligible_correct = sum(by_type[key]["correct_n"] for key in ("choice", "noul"))
    eligible_valid = sum(by_type[key]["valid_n"] for key in ("choice", "noul"))
    if (
        eligible_n != sum(counts[key] for key in ("choice", "noul"))
        or all_metrics.get("correct_n") != eligible_correct
        or all_metrics.get("valid_n") != eligible_valid
    ):
        raise ValueError("APUS final eligible accuracy/count differs from v2 report")
    role = css_report.get("roles", {}).get("evaluation", {})
    if (
        css_report.get("score_schema_version") != "css-transfer-score/2"
        or css_report.get("predictions_sha256") != css_pred_sha
        or role.get("items") != plan["css_panel"]["items"]
        or role.get("tasks") != len(EVALUATION_TASKS)
        or set(css_report.get("tasks", {})) != set(EVALUATION_TASKS)
    ):
        raise ValueError("APUS CSS Choice-only v2 score report identity/count differs")
    return {
        "model_id": model["model_id"],
        "revision": model["revision"],
        "final_full": {
            "correct_n": all_metrics["correct_n"],
            "n": all_metrics["n"],
            "accuracy_all": all_metrics["accuracy_all"],
            "valid_n": all_metrics["valid_n"],
            "unsupported_score_n": counts["score"],
        },
        "final_choice_noul_eligible": {
            "correct_n": eligible_correct,
            "n": eligible_n,
            "accuracy_all": eligible_correct / eligible_n,
            "valid_n": eligible_valid,
            "invalid_or_missing_n": eligible_n - eligible_valid,
            "choice": by_type["choice"],
            "noul": by_type["noul"],
        },
        "css_choice_only": role,
        "predictions_sha256": {"final": final_pred_sha, "css_evaluation": css_pred_sha},
        "reports_sha256": {
            "final": file_digest(Path(model["reports"]["final"])),
            "css_evaluation": file_digest(Path(model["reports"]["css-evaluation"])),
        },
        "gold_sha256_from_reports": {
            "final": final_report["gold_sha256"],
            "css_evaluation": css_report["gold_sha256"],
        },
    }


def summarize(plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    if not sha(expected_plan_sha256) or file_digest(plan_path) != expected_plan_sha256:
        raise ValueError("APUS saved plan differs from its recorded SHA-256")
    plan = read_json(plan_path)
    if plan.get("plan_version") != PLAN_VERSION or plan.get("plan_path") != str(
        plan_path
    ):
        raise ValueError("Unexpected APUS appendix plan")
    if (
        file_digest(Path(plan["main_plan_path"])) != plan["main_plan_sha256"]
        or check_freeze_binding(
            read_json(Path(plan["main_plan_path"])), Path(plan["freeze_manifest_path"])
        )
        != plan["freeze_manifest_sha256"]
    ):
        raise ValueError("Main final freeze changed after APUS appendix planning")
    if (
        panel_info(Path(plan["final_panel"]["path"]), final=True) != plan["final_panel"]
        or panel_info(Path(plan["css_panel"]["path"]), final=False) != plan["css_panel"]
    ):
        raise ValueError(
            "Frozen gold-free prompts changed after APUS appendix planning"
        )
    if (
        source_hashes(
            Path(plan["source_root"]), read_json(Path(plan["main_plan_path"]))
        )
        != plan["source_sha256"]
    ):
        raise ValueError("APUS appendix or scorers changed after planning")
    models = plan.get("models")
    if not isinstance(models, list) or [model.get("size") for model in models] != [
        "4b",
        "9b",
    ]:
        raise ValueError("APUS appendix model roster changed")
    raw_paths = {
        Path(model["predictions"][panel])
        for model in models
        for panel in ("final", "css-evaluation")
    }
    ledger: dict[Path, str] = {}
    for line in (
        Path(plan["raw_prediction_hashes_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        digest_value, separator, name = line.partition("  ")
        if not separator or not sha(digest_value) or Path(name) in ledger:
            raise ValueError(
                "Malformed or duplicate raw APUS prediction hash ledger entry"
            )
        ledger[Path(name)] = digest_value
    if set(ledger) != raw_paths or any(
        file_digest(path) != value for path, value in ledger.items()
    ):
        raise ValueError(
            "Raw APUS prediction ledger differs from final prediction bytes"
        )
    results = [model_summary(model, plan) for model in models]
    for panel in ("final", "css_evaluation"):
        if len({model["gold_sha256_from_reports"][panel] for model in results}) != 1:
            raise ValueError("APUS models were scored against different gold bytes")
    return {
        "report_version": RESULT_VERSION,
        "scope": plan["scope"],
        "plan_sha256": expected_plan_sha256,
        "main_plan_sha256": plan["main_plan_sha256"],
        "freeze_manifest_sha256": plan["freeze_manifest_sha256"],
        "final_panel": plan["final_panel"],
        "css_panel": plan["css_panel"],
        "models": results,
        "ranking_policy": "APUS excluded from all-type rank and matrix; compare only native eligible Choice+Noul and CSS Choice slices.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser(
        "plan", help="Emit commands after main freeze and gold-free panels exist"
    )
    for name in (
        "main-plan",
        "freeze-manifest",
        "final-prompts",
        "css-prompts",
        "source-root",
        "model-root",
        "output-root",
        "plan-path",
    ):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--expected-main-plan-sha256", required=True)
    create.add_argument("--python", default="python3")
    result = sub.add_parser(
        "summarize", help="Audit completed v2 reports without opening gold"
    )
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        inputs = validate_inputs(
            main_plan=args.main_plan,
            main_plan_sha256=args.expected_main_plan_sha256,
            freeze=args.freeze_manifest,
            final_prompts=args.final_prompts,
            css_prompts=args.css_prompts,
            source_root=args.source_root,
            output_root=args.output_root,
            plan_path=args.plan_path,
        )
        result = build_plan(
            inputs=inputs,
            main_plan=args.main_plan,
            freeze=args.freeze_manifest,
            source_root=args.source_root,
            model_root=args.model_root,
            output_root=args.output_root,
            plan_path=args.plan_path,
            python=args.python,
        )
        with args.plan_path.open("x", encoding="utf-8") as target:
            target.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            )
        print(
            json.dumps(
                {
                    "plan_path": str(args.plan_path),
                    "plan_sha256": file_digest(args.plan_path),
                    "status": result["status"],
                },
                sort_keys=True,
            )
        )
    else:
        result = summarize(args.plan, args.expected_plan_sha256)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
