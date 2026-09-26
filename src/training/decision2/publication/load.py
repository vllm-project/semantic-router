"""Load and validate frozen benchmark and optional CSS score reports."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from benchmark.generate import FINAL_FAMILIES
from transfer.build import EVALUATION_TASKS, PANEL_VERSION, PILOT_TASKS, sha_file

GROUPS = {"decision2", "decision1", "open", "hosted", "other"}
TYPES = {"choice", "noul", "score"}
SECRET = re.compile(
    r"(?i)(?:\bhf_[A-Za-z0-9]{20,}|\bjv_live_[A-Za-z0-9_-]{16,}"
    r"|\bapikey_[A-Za-z0-9_-]{16,}|\bsk-[A-Za-z0-9_-]{16,}"
    r"|Authorization\s*:\s*Bearer\s+\S+)"
)
HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:home|root|data|work|mnt|tmp|private|Users|var|opt)/[^\s\"'<>]+"
    r"|\b[A-Za-z]:[\\/](?:Users|Documents|ProgramData|Windows)[\\/][^\s\"'<>]+"
)


def _public_text(value: str, description: str) -> None:
    if value.startswith("/") or SECRET.search(value) or HOST_PATH.search(value):
        raise ValueError(f"{description} contains a credential or absolute host path")


def _sha(value: Any, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _number(value: Any, description: str, low: float = 0.0, high: float = 1.0) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{description} must be finite in [{low},{high}]")
    return float(value)


def _count(value: Any, description: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError(
            f"{description} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return value


def _summary(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    n = _count(value.get("n"), f"{name}.n", positive=True)
    correct = _count(value.get("correct_n"), f"{name}.correct_n")
    valid = _count(value.get("valid_n"), f"{name}.valid_n")
    if correct > valid or valid > n:
        raise ValueError(f"{name}: impossible correct/valid/n counts")
    accuracy = _number(value.get("accuracy_all"), f"{name}.accuracy_all")
    if not math.isclose(accuracy, correct / n, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{name}: accuracy_all disagrees with correct_n/n")
    if value.get("invalid_or_missing_n") != n - valid:
        raise ValueError(f"{name}: invalid_or_missing_n disagrees with valid_n")
    brier = value.get("brier")
    if brier is not None:
        _number(brier, f"{name}.brier", high=1.0 + 1e-12)
    return value


def benchmark_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: benchmark report must be an object")
    if (
        report.get("schema_version") != "typed-decision-report/2"
        or report.get("split") != "final"
    ):
        raise ValueError(
            f"{path}: publication requires a frozen final typed-decision report"
        )
    _sha(report.get("gold_sha256"), "benchmark gold_sha256")
    _sha(report.get("predictions_sha256"), "benchmark predictions_sha256")
    _count(report.get("items"), "items", positive=True)
    _count(report.get("predicted_items"), "predicted_items")
    if report["predicted_items"] > report["items"]:
        raise ValueError("predicted_items exceeds the frozen item count")
    model = report.get("model")
    if not isinstance(model, dict) or any(
        not isinstance(model.get(key), str) or not model[key]
        for key in ("id", "revision", "backend")
    ):
        raise ValueError(f"{path}: model id/revision/backend is missing")
    for field in ("id", "revision", "backend"):
        _public_text(model[field], f"benchmark model {field}")
    overall = _summary(report.get("overall"), "overall")
    by_family = report.get("by_family")
    by_type = report.get("by_type")
    if not isinstance(by_family, dict) or set(by_family) != set(FINAL_FAMILIES):
        raise ValueError(f"{path}: final family set differs from the frozen protocol")
    if not isinstance(by_type, dict) or set(by_type) != TYPES:
        raise ValueError(f"{path}: native Choice/Noul/Score slices are required")
    families = [
        _summary(by_family[name], f"by_family.{name}") for name in FINAL_FAMILIES
    ]
    types = [_summary(by_type[name], f"by_type.{name}") for name in TYPES]
    if (
        sum(value["n"] for value in families) != overall["n"]
        or sum(value["n"] for value in types) != overall["n"]
    ):
        raise ValueError(f"{path}: slice counts disagree with overall")
    if sum(value["correct_n"] for value in families) != overall["correct_n"]:
        raise ValueError(f"{path}: family correct counts disagree with overall")
    if sum(value["correct_n"] for value in types) != overall["correct_n"]:
        raise ValueError(f"{path}: type correct counts disagree with overall")
    if (
        sum(value["valid_n"] for value in families) != overall["valid_n"]
        or sum(value["valid_n"] for value in types) != overall["valid_n"]
    ):
        raise ValueError(f"{path}: slice valid counts disagree with overall")
    macro = _number(report.get("macro_family_accuracy"), "macro_family_accuracy")
    expected_macro = statistics.mean(value["accuracy_all"] for value in families)
    if not math.isclose(macro, expected_macro, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{path}: macro_family_accuracy disagrees with family scores")
    return report


def css_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: CSS report must be an object")
    if report.get("score_schema_version") != "css-transfer-score/2":
        raise ValueError(f"{path}: wrong CSS score schema version")
    if report.get("panel_version") != PANEL_VERSION:
        raise ValueError(f"{path}: wrong CSS panel version")
    _sha(report.get("gold_sha256"), "CSS gold_sha256")
    _sha(report.get("predictions_sha256"), "CSS predictions_sha256")
    tasks = report.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"{path}: CSS task details missing")
    if any(not isinstance(value, dict) for value in tasks.values()):
        raise ValueError(f"{path}: CSS task details must be objects")
    if set(tasks) - set(EVALUATION_TASKS) - set(PILOT_TASKS):
        raise ValueError(f"{path}: CSS report contains an unknown task")
    for name, value in tasks.items():
        expected_role = "evaluation" if name in EVALUATION_TASKS else "pilot"
        if value.get("role") != expected_role:
            raise ValueError(f"{path}: CSS {name} has the wrong task role")
        n = _count(value.get("n"), f"CSS {name}.n", positive=True)
        valid = _count(value.get("valid_n"), f"CSS {name}.valid_n")
        correct = _count(value.get("correct_n"), f"CSS {name}.correct_n")
        if (
            correct > valid
            or valid > n
            or value.get("invalid_or_missing_n") != n - valid
            or not math.isclose(
                _number(value.get("accuracy_all"), f"CSS {name}.accuracy_all"),
                correct / n,
                rel_tol=0,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(f"{path}: CSS {name} has inconsistent task counts")
        _number(value.get("macro_f1_all"), f"CSS {name}.macro_f1_all")
    evaluation = {
        name: value
        for name, value in tasks.items()
        if value.get("role") == "evaluation"
    }
    if set(evaluation) != set(EVALUATION_TASKS):
        raise ValueError(f"{path}: CSS report needs all 15 evaluation tasks")
    median_f1 = statistics.median(
        _number(value.get("macro_f1_all"), f"CSS {name}.macro_f1_all")
        for name, value in evaluation.items()
    )
    median_accuracy = statistics.median(
        _number(value.get("accuracy_all"), f"CSS {name}.accuracy_all")
        for name, value in evaluation.items()
    )
    roles = report.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get("evaluation"), dict):
        raise ValueError(f"{path}: CSS evaluation role missing")
    role = roles["evaluation"]
    if role.get("tasks") != 15 or not math.isclose(
        _number(role.get("median_task_macro_f1_all"), "CSS median F1"),
        median_f1,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{path}: CSS median F1 disagrees with task scores")
    if not math.isclose(
        _number(role.get("median_task_accuracy_all"), "CSS median accuracy"),
        median_accuracy,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{path}: CSS median accuracy disagrees with task scores")
    items = sum(value["n"] for value in evaluation.values())
    valid = sum(value["valid_n"] for value in evaluation.values())
    correct = sum(value["correct_n"] for value in evaluation.values())
    if (
        role.get("items") != items
        or role.get("valid_items") != valid
        or not math.isclose(
            _number(role.get("micro_accuracy_all"), "CSS evaluation micro accuracy"),
            correct / items,
            rel_tol=0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError(
            f"{path}: CSS evaluation role counts disagree with task scores"
        )
    return report


def comparison_report(
    path: Path, css_new: dict[str, Any], css_old: dict[str, Any]
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{path}: CSS comparison must be an object")
    if report.get("comparison_version") != "css-paired-item-bootstrap/1":
        raise ValueError(f"{path}: wrong paired comparison version")
    if (
        report.get("gold_sha256") != css_new["gold_sha256"]
        or report.get("gold_sha256") != css_old["gold_sha256"]
        or report.get("predictions_a_sha256") != css_new["predictions_sha256"]
        or report.get("predictions_b_sha256") != css_old["predictions_sha256"]
    ):
        raise ValueError(
            f"{path}: paired comparison does not match CSS prediction files"
        )
    headline = report.get("evaluation_median_over_15_tasks", {})
    if headline.get("task_count") != 15:
        raise ValueError(f"{path}: comparison has incomplete evaluation tasks")
    for metric, css_key in (
        ("macro_f1_all", "median_task_macro_f1_all"),
        ("accuracy_all", "median_task_accuracy_all"),
    ):
        summary = headline.get(metric)
        if not isinstance(summary, dict):
            raise ValueError(f"{path}: missing paired {metric}")
        a = css_new["roles"]["evaluation"][css_key]
        b = css_old["roles"]["evaluation"][css_key]
        if not all(
            math.isclose(
                _number(summary.get(key), key, -1, 1), target, rel_tol=0, abs_tol=1e-10
            )
            for key, target in (
                ("median_a", a),
                ("median_b", b),
                ("difference_a_minus_b", a - b),
            )
        ):
            raise ValueError(
                f"{path}: paired {metric} point estimate disagrees with CSS score reports"
            )
        ci = summary.get("difference_interval95")
        if not isinstance(ci, dict):
            raise ValueError(f"{path}: paired {metric} interval missing")
        low, high = _number(ci.get("low"), "paired CI low", -1, 1), _number(
            ci.get("high"), "paired CI high", -1, 1
        )
        if low > high:
            raise ValueError(f"{path}: inverted paired interval")
    return report


def resolve_report(config_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("Report path must be a nonempty string")
    path = Path(raw)
    _public_text(path.name, "report filename")
    return path if path.is_absolute() else config_path.parent / path


def load_inputs(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("models"), list)
        or not config["models"]
    ):
        raise ValueError("Config needs a nonempty models list")
    if set(config) - {"models", "comparison_pairs", "title"}:
        raise ValueError("Unknown publication config keys")
    models = []
    keys = set()
    labels = set()
    for entry in config["models"]:
        if not isinstance(entry, dict) or set(entry) - {
            "key",
            "label",
            "group",
            "size",
            "benchmark_report",
            "css_report",
        }:
            raise ValueError("Invalid model metadata fields")
        for name in ("key", "label", "group", "benchmark_report"):
            if not isinstance(entry.get(name), str) or not entry[name]:
                raise ValueError(f"Model needs nonempty {name}")
        for name in ("key", "label"):
            _public_text(entry[name], f"model {name}")
        if entry["group"] not in GROUPS:
            raise ValueError(f"Unknown model group {entry['group']}")
        if entry["key"] in keys or entry["label"] in labels:
            raise ValueError("Duplicate model key or display label")
        if "size" in entry and (
            not isinstance(entry["size"], str) or not entry["size"]
        ):
            raise ValueError("Model size must be a nonempty display string")
        if "size" in entry:
            _public_text(entry["size"], "model size")
        keys.add(entry["key"])
        labels.add(entry["label"])
        bench_path = resolve_report(config_path, entry["benchmark_report"])
        bench = benchmark_report(bench_path)
        css_path = (
            resolve_report(config_path, entry["css_report"])
            if entry.get("css_report")
            else None
        )
        css = css_report(css_path) if css_path else None
        models.append(
            {
                **entry,
                "benchmark": bench,
                "css": css,
                "benchmark_path": bench_path,
                "css_path": css_path,
                "benchmark_file_sha256": sha_file(bench_path),
                "css_file_sha256": sha_file(css_path) if css_path else None,
            }
        )
    gold = {model["benchmark"]["gold_sha256"] for model in models}
    items = {model["benchmark"]["items"] for model in models}
    questions = {model["benchmark"]["overall"]["n"] for model in models}
    if len(gold) != 1 or len(items) != 1 or len(questions) != 1:
        raise ValueError(
            "All benchmark reports must use the same frozen gold and item/question counts"
        )
    identities = [
        (model["benchmark"]["model"]["id"], model["benchmark"]["model"]["revision"])
        for model in models
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("The same benchmark model revision appears twice")
    css_gold = {
        model["css"]["gold_sha256"] for model in models if model["css"] is not None
    }
    if len(css_gold) > 1:
        raise ValueError("CSS reports use different human-label gold panels")
    by_key = {model["key"]: model for model in models}
    pair_entries = config.get("comparison_pairs", [])
    if not isinstance(pair_entries, list):
        raise ValueError("comparison_pairs must be a list")
    pairs = []
    seen_pairs = set()
    for entry in pair_entries:
        if not isinstance(entry, dict) or set(entry) - {
            "new",
            "old",
            "css_comparison_report",
        }:
            raise ValueError("Invalid comparison pair")
        if (
            entry.get("new") not in by_key
            or entry.get("old") not in by_key
            or entry["new"] == entry["old"]
        ):
            raise ValueError("Comparison pair must name two distinct configured models")
        pair_key = (entry["new"], entry["old"])
        if pair_key in seen_pairs:
            raise ValueError("Duplicate comparison pair")
        seen_pairs.add(pair_key)
        new, old = by_key[entry["new"]], by_key[entry["old"]]
        comparison_path = (
            resolve_report(config_path, entry["css_comparison_report"])
            if entry.get("css_comparison_report")
            else None
        )
        if comparison_path and (new["css"] is None or old["css"] is None):
            raise ValueError("A CSS paired comparison needs both CSS score reports")
        comparison = (
            comparison_report(comparison_path, new["css"], old["css"])
            if comparison_path
            else None
        )
        pairs.append(
            {
                "new": new,
                "old": old,
                "css_comparison": comparison,
                "css_comparison_path": comparison_path,
                "css_comparison_file_sha256": (
                    sha_file(comparison_path) if comparison_path else None
                ),
            }
        )
    title = config.get("title", "Decision 2.0 — frozen typed-decision benchmark")
    if not isinstance(title, str) or not title:
        raise ValueError("title must be a nonempty string")
    _public_text(title, "publication title")
    return {
        "title": title,
        "config_sha256": sha_file(config_path),
        "models": models,
        "pairs": pairs,
        "gold_sha256": next(iter(gold)),
        "items": next(iter(items)),
        "questions": next(iter(questions)),
        "css_gold_sha256": next(iter(css_gold)) if css_gold else None,
    }
