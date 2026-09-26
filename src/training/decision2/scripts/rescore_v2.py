"""Recompute the frozen synthetic DEV and CSS pilot baseline panel on CPU.

All outputs are new files in one versioned directory. Raw predictions, API
receipts, gold, and earlier reports remain untouched. Jev DEV is rebound from
gold-free prompts after verifying its original API receipts and predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.score import load_jsonl as load_benchmark_jsonl
from benchmark.score import score_suite
from clients.normalize_jev_v2 import normalize as normalize_jev
from transfer.build import PILOT_TASKS
from transfer.score import read_jsonl as load_css_jsonl
from transfer.score import score as score_css

SUMMARY_VERSION = "decision2-dev-css-pilot-v2/1"
DEV_GOLD_SHA256 = "c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc"
CSS_PILOT_GOLD_SHA256 = (
    "9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391"
)
LUX_STEP25_MODEL_SHA256 = (
    "edecdb7aba86e1d47dfbd44c39df51593a2a08a8af13007351eb814c83efcebc"
)

# IDs and revisions are pinned to the source releases and original inference
# manifests. CSS uses the same published model identity as synthetic DEV.
SPECS = (
    (
        "jev",
        "Jev 1.13",
        "hosted",
        "dev.jev-1.13",
        "jev-1.13",
        "TypeSafe/jev-1.13.0",
        "jev-1.13.0",
        "official-api",
    ),
    (
        "decider",
        "Decider 4B",
        "open",
        "decider4b-dev",
        "decider4b",
        "Mapika/decider-4b",
        "eb5fbdfc9448473ec25e399882912863afbdb70e",
        "native-eager",
    ),
    (
        "kev",
        "Kev 4B",
        "open",
        "kev4b-dev",
        "kev4b",
        "jaredpalmer/kev-4b",
        "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        "kev",
    ),
    (
        "this-that",
        "This-That 1.0",
        "open",
        "this-that10-dev",
        "this-that10",
        "flock-io/this-that-model-1.0",
        "3d927195c4f9845efe66c5715883a7a0f42b1239",
        "this-that",
    ),
    (
        "laya",
        "Laya Typed",
        "open",
        "laya-typed-dev",
        "laya",
        "convaiinnovations/laya-typed-decisions",
        "1a793eb568e6718f15941d08f85432581df534e3",
        "laya",
    ),
    (
        "eos",
        "Decision 1.0 Eos",
        "decision1",
        "eos08b-dev",
        "eos08b",
        "llm-semantic-router/Decision-1.0-Eos-0.8B",
        "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd",
        "eos",
    ),
    (
        "kai",
        "Decision 1.0 Kai",
        "decision1",
        "kai06b-dev",
        "kai06b",
        "llm-semantic-router/Decision-1.0-Kai-0.6B",
        "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
        "kai",
    ),
    (
        "lex",
        "Decision 1.0 Lex",
        "decision1",
        "lex06b-dev",
        "lex06b",
        "llm-semantic-router/Decision-1.0-Lex-0.6B",
        "ee8e74d912fca8328a353c11d174b44da3f91781",
        "lex",
    ),
    (
        "lux",
        "Decision 1.0 Lux",
        "decision1",
        "lux9b-dev",
        "lux9b",
        "llm-semantic-router/Decision-1.0-Lux-9B",
        "bd45a30aee8c84032791c245c70f86dee5389cc8",
        "native-qualified-rocm",
    ),
    (
        "nox",
        "Decision 1.0 Nox",
        "decision1",
        "nox4b-dev",
        "nox4b",
        "llm-semantic-router/Decision-1.0-Nox-4B",
        "0bb833504965c0eabdb9630b7bbd385cb2fe5cd4",
        "nox",
    ),
    (
        "sol",
        "Decision 1.0 Sol",
        "decision1",
        "sol2b-dev",
        "sol2b",
        "llm-semantic-router/Decision-1.0-Sol-2B",
        "0665a41108e8f0b33a9515c98311c45947b99399",
        "sol",
    ),
    (
        "lux-step25",
        "Lux 9B combined6k checkpoint25",
        "decision2-candidate",
        "lux9b-c6k-step25-dev",
        "lux9b-c6k-step25",
        "decision2-lux9b-c6k-r1-step25",
        "checkpoint-0000025",
        "decision2-peft-lora-native",
    ),
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as target:
        target.write(
            json.dumps(
                value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )


def require_complete(path: Path, expected_ids: set[str], loader: Any) -> None:
    observed = set(loader(path))
    if observed != expected_ids:
        raise ValueError(
            f"{path.name}: {len(observed)} prediction IDs differ from {len(expected_ids)} gold IDs"
        )


def verify_lux_step25(run_root: Path) -> dict[str, Any]:
    found = []
    for file, expected_items in (
        (run_root / "lux9b-c6k-step25-dev.predictions.jsonl", 1600),
        (
            run_root / "css-transfer-v1/css-pilot.lux9b-c6k-step25.predictions.jsonl",
            1430,
        ),
    ):
        manifest_path = Path(str(file) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model_id") != "decision2-lux9b-c6k-r1-step25"
            or manifest.get("model_revision") != "checkpoint-0000025"
            or manifest.get("model_sha256") != LUX_STEP25_MODEL_SHA256
            or manifest.get("predictions_sha256") != sha_file(file)
            or manifest.get("counts", {}).get("items") != expected_items
        ):
            raise ValueError(
                f"{manifest_path.name}: unexpected checkpoint or prediction digest"
            )
        found.append(
            {
                "file": file.name,
                "manifest_sha256": sha_file(manifest_path),
                "model_sha256": manifest["model_sha256"],
                "predictions_sha256": manifest["predictions_sha256"],
            }
        )
    return {"model_sha256": LUX_STEP25_MODEL_SHA256, "prediction_manifests": found}


def dev_metrics(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    return {
        "n": overall["n"],
        "valid_n": overall["valid_n"],
        "invalid_or_missing_n": overall["invalid_or_missing_n"],
        "accuracy_all": overall["accuracy_all"],
        "macro_family_accuracy": report["macro_family_accuracy"],
        "brier": overall["brier"],
        "nll": overall["nll"],
        "ece_10": overall["ece_10"],
        "probability_n": overall["probability_n"],
        "option_probability_sum_abs_delta": overall["option_probability_sum_abs_delta"],
        "invalid_reasons": report["invalid_reasons"],
        "by_type": {
            name: {
                field: value[field]
                for field in ("n", "valid_n", "accuracy_all", "brier", "ece_10")
            }
            for name, value in report["by_type"].items()
        },
    }


def css_metrics(report: dict[str, Any]) -> dict[str, Any]:
    pilot = report["roles"]["pilot"]
    invalid_reasons: Counter[str] = Counter()
    for task in report["tasks"].values():
        invalid_reasons.update(task["invalid_reasons"])
    return {
        "n": pilot["items"],
        "valid_n": pilot["valid_items"],
        "invalid_or_missing_n": pilot["items"] - pilot["valid_items"],
        "micro_accuracy_all": pilot["micro_accuracy_all"],
        "median_task_accuracy_all": pilot["median_task_accuracy_all"],
        "median_task_macro_f1_all": pilot["median_task_macro_f1_all"],
        "median_task_brier_sum": pilot["median_task_brier_sum"],
        "median_task_ece_pmax_15": pilot["median_task_ece_pmax_15"],
        "option_probability_sum_abs_delta": pilot["option_probability_sum_abs_delta"],
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "by_task": {
            name: {
                field: value[field]
                for field in (
                    "n",
                    "valid_n",
                    "accuracy_all",
                    "macro_f1_all",
                    "brier_sum",
                    "ece_pmax_15",
                    "ece_native_confidence_15",
                )
            }
            for name, value in report["tasks"].items()
        },
    }


def comparisons(models: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {model["key"]: model for model in models}
    candidate = by_key["lux-step25"]
    result = {}
    for baseline in ("lux", "jev"):
        other = by_key[baseline]
        result[f"lux-step25_minus_{baseline}"] = {
            "dev_accuracy_all": candidate["dev"]["accuracy_all"]
            - other["dev"]["accuracy_all"],
            "dev_brier": candidate["dev"]["brier"] - other["dev"]["brier"],
            "dev_ece_10": candidate["dev"]["ece_10"] - other["dev"]["ece_10"],
            "css_pilot_micro_accuracy_all": candidate["css_pilot"]["micro_accuracy_all"]
            - other["css_pilot"]["micro_accuracy_all"],
            "css_pilot_median_task_macro_f1_all": candidate["css_pilot"][
                "median_task_macro_f1_all"
            ]
            - other["css_pilot"]["median_task_macro_f1_all"],
            "css_pilot_median_task_brier_sum": candidate["css_pilot"][
                "median_task_brier_sum"
            ]
            - other["css_pilot"]["median_task_brier_sum"],
            "css_pilot_median_task_ece_pmax_15": candidate["css_pilot"][
                "median_task_ece_pmax_15"
            ]
            - other["css_pilot"]["median_task_ece_pmax_15"],
        }
    return result


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen DEV and CSS pilot v2 baseline panel",
        "",
        "Both panels are development evidence. CSS pilot contains three human-label tasks; its median is not the 15-task evaluation headline. The Decision 2.0 checkpoint is a training candidate, not a released model.",
        "",
        "Accepted Choice/Score maps retain the original answer and validity thresholds, then divide option probabilities by their original sum for Brier, NLL, pmax ECE, and selective confidence. Native scalar confidence remains unchanged. Invalid or missing predictions remain in point-metric denominators.",
        "",
        "## Synthetic DEV (1,600 items)",
        "",
        "| Model | Valid / n | Accuracy | Brier | ECE10 | Invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        value = model["dev"]
        lines.append(
            f"| {model['label']} | {value['valid_n']}/{value['n']} | {value['accuracy_all']:.2%} | {value['brier']:.4f} | {value['ece_10']:.4f} | {value['invalid_or_missing_n']} |"
        )
    lines += [
        "",
        "## CSS pilot (1,430 human-label items, three tasks)",
        "",
        "| Model | Valid / n | Micro accuracy | Median task macro-F1 | Median task Brier sum | Median task ECE pmax | Invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in summary["models"]:
        value = model["css_pilot"]
        lines.append(
            f"| {model['label']} | {value['valid_n']}/{value['n']} | {value['micro_accuracy_all']:.2%} | {value['median_task_macro_f1_all']:.4f} | {value['median_task_brier_sum']:.4f} | {value['median_task_ece_pmax_15']:.4f} | {value['invalid_or_missing_n']} |"
        )
    lines += [
        "",
        "## Lux checkpoint25 differences (candidate minus baseline)",
        "",
        "| Baseline | DEV accuracy (pp) | DEV Brier | DEV ECE10 | CSS pilot accuracy (pp) | CSS pilot macro-F1 | CSS pilot Brier sum | CSS pilot ECE pmax |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline in ("lux", "jev"):
        delta = summary["comparisons"][f"lux-step25_minus_{baseline}"]
        lines.append(
            f"| {baseline} | {delta['dev_accuracy_all'] * 100:+.3f} | {delta['dev_brier']:+.4f} | {delta['dev_ece_10']:+.4f} | {delta['css_pilot_micro_accuracy_all'] * 100:+.3f} | {delta['css_pilot_median_task_macro_f1_all']:+.4f} | {delta['css_pilot_median_task_brier_sum']:+.4f} | {delta['css_pilot_median_task_ece_pmax_15']:+.4f} |"
        )
    lines += [
        "",
        "## SHA-256 provenance",
        "",
        f"Synthetic DEV gold: `{summary['gold']['dev_sha256']}`  ",
        f"CSS pilot gold: `{summary['gold']['css_pilot_sha256']}`",
        "",
        "| Model | Panel | Original prediction SHA-256 | Scored prediction SHA-256 | v2 report SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in summary["models"]:
        for panel in ("dev", "css_pilot"):
            value = model[panel]
            lines.append(
                f"| {model['label']} | {panel} | `{value['original_predictions_sha256']}` | `{value['scored_predictions_sha256']}` | `{value['report_sha256']}` |"
            )
    lines += [
        "",
        "Jev DEV uses a separate verified v2 prediction copy because the legacy normalizer put the full API-body digest in `source_input_sha256`. All 1,600 receipt hashes, model IDs, responses, and old predictions were checked against the gold-free prompt file before computing the state/questions payload digest. Original files remain unchanged. Details and file SHA-256 values are in `summary.json`.",
        "",
        "`summary.json` also records exact model revisions, scorer source hashes, invalid reasons, per-type DEV slices, CSS pilot per-task metrics, and original probability-sum deviation statistics.",
        "",
    ]
    return "\n".join(lines)


def run(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    dev_gold = run_root / "dev.gold.jsonl"
    css_gold = run_root / "css-transfer-v1/css-pilot.gold.jsonl"
    if (
        sha_file(dev_gold) != DEV_GOLD_SHA256
        or sha_file(css_gold) != CSS_PILOT_GOLD_SHA256
    ):
        raise ValueError("Frozen DEV or CSS pilot gold SHA-256 changed")
    dev_ids = set(load_benchmark_jsonl(dev_gold))
    css_ids = set(load_css_jsonl(css_gold))
    if len(dev_ids) != 1600 or len(css_ids) != 1430:
        raise ValueError("Frozen panel item count changed")
    checkpoint = verify_lux_step25(run_root)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        legacy_jev = run_root / "dev.jev-1.13.predictions.jsonl"
        bound_jev = temporary / "jev-dev.predictions.v2-bound.jsonl"
        jev_binding = normalize_jev(
            run_root / "dev.prompts.jsonl",
            run_root / "dev.jev-1.13.receipts.jsonl",
            legacy_jev,
            bound_jev,
        )
        models = []
        for key, label, group, dev_stem, css_stem, model_id, revision, backend in SPECS:
            raw_dev = run_root / f"{dev_stem}.predictions.jsonl"
            dev_pred = bound_jev if key == "jev" else raw_dev
            css_pred = (
                run_root / "css-transfer-v1" / f"css-pilot.{css_stem}.predictions.jsonl"
            )
            require_complete(dev_pred, dev_ids, load_benchmark_jsonl)
            require_complete(css_pred, css_ids, load_css_jsonl)
            dev_report = score_suite(dev_gold, dev_pred, model_id, revision, backend)
            css_report = score_css(css_gold, css_pred)
            if (
                dev_report["schema_version"] != "typed-decision-report/2"
                or css_report["score_schema_version"] != "css-transfer-score/2"
                or set(css_report["tasks"]) != set(PILOT_TASKS)
            ):
                raise ValueError("Unexpected scorer version or CSS pilot task set")
            dev_file = temporary / f"{key}.dev.score.v2.json"
            css_file = temporary / f"{key}.css-pilot.score.v2.json"
            write_json(dev_file, dev_report)
            write_json(css_file, css_report)
            models.append(
                {
                    "key": key,
                    "label": label,
                    "group": group,
                    "model": {"id": model_id, "revision": revision, "backend": backend},
                    "dev": {
                        **dev_metrics(dev_report),
                        "original_predictions_file": raw_dev.name,
                        "original_predictions_sha256": sha_file(raw_dev),
                        "scored_predictions_file": dev_pred.name,
                        "scored_predictions_sha256": sha_file(dev_pred),
                        "report_file": dev_file.name,
                        "report_sha256": sha_file(dev_file),
                        "gold_sha256": DEV_GOLD_SHA256,
                    },
                    "css_pilot": {
                        **css_metrics(css_report),
                        "original_predictions_file": css_pred.name,
                        "original_predictions_sha256": sha_file(css_pred),
                        "scored_predictions_file": css_pred.name,
                        "scored_predictions_sha256": sha_file(css_pred),
                        "report_file": css_file.name,
                        "report_sha256": sha_file(css_file),
                        "gold_sha256": CSS_PILOT_GOLD_SHA256,
                    },
                }
            )
        summary = {
            "summary_version": SUMMARY_VERSION,
            "scope": "Frozen synthetic DEV and three-task CSS pilot; development evidence only",
            "gold": {
                "dev_sha256": DEV_GOLD_SHA256,
                "css_pilot_sha256": CSS_PILOT_GOLD_SHA256,
                "dev_items": len(dev_ids),
                "css_pilot_items": len(css_ids),
            },
            "score_policy": {
                "benchmark_report": "typed-decision-report/2",
                "css_report": "css-transfer-score/2",
                "option_probabilities": "Normalize accepted Choice/Score maps by original sum for probability metrics; point answers and validity thresholds unchanged",
            },
            "source_sha256": {
                name: sha_file(path)
                for name, path in (
                    ("rescore_v2.py", Path(__file__)),
                    ("benchmark.score.py", Path(score_suite.__code__.co_filename)),
                    ("transfer.score.py", Path(score_css.__code__.co_filename)),
                    (
                        "clients.normalize_jev_v2.py",
                        Path(normalize_jev.__code__.co_filename),
                    ),
                )
            },
            "jev_dev_binding": jev_binding,
            "lux_step25": checkpoint,
            "models": models,
            "comparisons": comparisons(models),
        }
        (temporary / "summary.md").write_text(markdown(summary), encoding="utf-8")
        write_json(temporary / "summary.json", summary)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        temporary.rename(output_dir)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.runs_root, args.output_dir)
    print(
        json.dumps(
            {
                "summary_version": result["summary_version"],
                "models": len(result["models"]),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
