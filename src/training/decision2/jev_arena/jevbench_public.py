"""Build and score the pinned public JevBench panel without test-text leakage.

JevBench's sealed tasks are unavailable. Reports from this module are an
independent *public-only* view, never an official JevBench composite/rank.
The source task set and scoring semantics are attributed to the MIT-licensed
fstandhartinger/jevbench project, revision pinned below.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any

SOURCE_URL = "https://github.com/fstandhartinger/jevbench"
SOURCE_REVISION = "1bcc55eb6c8cffde2306b3db03ede39b61c6152a"
FILES = {
    "easy": (
        "datasets/public/easy.jsonl",
        "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
        48,
    ),
    "standard": (
        "datasets/public/original.jsonl",
        "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
        72,
    ),
    "hard": (
        "datasets/public/hard.jsonl",
        "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
        111,
    ),
}
BUILD_VERSION = "jevarena-jevbench-public-build/1"
SCORE_VERSION = "jevarena-jevbench-public-score/1"
QUESTION_ID = "decision"
ABSENT_MODEL_ID = "__absent__"  # Native Decider receipts omit model_id.
RENORM_TOL = 0.02  # JevBench jevbench/scoring.py public headline tolerance.
STRICT_TOL = 0.001


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def input_digest(state: Any, questions: dict[str, Any]) -> str:
    # Matches inference.run.digest over the exact model-visible object.
    encoded = json.dumps(
        {"state": state, "questions": questions},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def build(upstream: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    prompts, targets, seen = [], [], set()
    for tier, (relative, expected_sha, count) in FILES.items():
        path = upstream / relative
        if sha_file(path) != expected_sha:
            raise ValueError(f"JevBench {tier} source bytes changed")
        rows = _read_jsonl(path)
        if len(rows) != count:
            raise ValueError(f"JevBench {tier} cardinality changed")
        for row in rows:
            item_id, question = row["id"], row["question"]
            if (
                item_id in seen
                or row.get("split") != "public"
                or question.get("type") not in ("choice", "noul", "score")
                or row.get("expected") is None
            ):
                raise ValueError(f"Invalid or duplicate JevBench public item {item_id}")
            labels = row["labels"]
            if (
                not isinstance(labels, list)
                or len(labels) < 2
                or len(labels) != len(set(labels))
            ):
                raise ValueError(f"Invalid JevBench labels in {item_id}")
            qtype, criteria = question["type"], question["criteria"]
            if qtype == "choice" and (
                not isinstance(criteria, dict) or set(criteria) != set(labels)
            ):
                raise ValueError(f"Choice label/criteria mismatch in {item_id}")
            if qtype == "noul" and (
                labels != ["no", "yes"] or set(criteria) != {"false", "true"}
            ):
                raise ValueError(f"Noul label/criteria mismatch in {item_id}")
            if qtype == "score" and (
                not isinstance(criteria, list)
                or labels != [str(i) for i in range(len(criteria))]
            ):
                raise ValueError(f"Score level/criteria mismatch in {item_id}")
            questions = {QUESTION_ID: question}
            prompt = {"id": item_id, "state": row["state"], "questions": questions}
            target = {
                "id": item_id,
                "tier": tier,
                "family": row["family"],
                "task_type": qtype,
                "labels": labels,
                "expected": row["expected"],
                "group": row.get("group"),
                "source_input_sha256": input_digest(row["state"], questions),
            }
            if any(
                key in prompt for key in ("expected", "gold", "provenance", "labels")
            ):
                raise AssertionError("Gold or provenance in model-visible prompt")
            prompts.append(prompt)
            targets.append(target)
            seen.add(item_id)
    if len(prompts) != 231:
        raise AssertionError("Unexpected JevBench public panel size")
    prompt_bytes = b"".join(map(json_bytes, prompts))
    target_bytes = b"".join(map(json_bytes, targets))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary)
        (stage / "prompts.jsonl").write_bytes(prompt_bytes)
        (stage / "targets.jsonl").write_bytes(target_bytes)
        manifest = {
            "build_version": BUILD_VERSION,
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "source_sha256": {tier: digest for tier, (_, digest, _) in FILES.items()},
            "prompts_sha256": sha_file(stage / "prompts.jsonl"),
            "targets_sha256": sha_file(stage / "targets.jsonl"),
            "items": len(prompts),
            "tier_counts": dict(collections.Counter(t["tier"] for t in targets)),
            "scope": "JevBench public items only; no sealed item or official composite score",
        }
        (stage / "manifest.json").write_bytes(json_bytes(manifest))
        stage.rename(output_dir)
    return manifest


def _valid_probs(
    raw: Any, labels: list[str]
) -> tuple[dict[str, float], bool, bool] | None:
    if not isinstance(raw, dict) or set(raw) != set(labels):
        return None
    values = {}
    for label in labels:
        value = raw[label]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return None
        values[label] = float(value)
    total = sum(values.values())
    if total <= 0 or abs(total - 1) > RENORM_TOL:
        return None
    strict = abs(total - 1) <= STRICT_TOL
    return (
        {label: value / total for label, value in values.items()},
        strict,
        not strict,
    )


def _evaluate(answer: Any, target: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(answer, dict):
        return {"valid": False, "correct": False, "reason": "missing answer"}
    qtype, labels = target["task_type"], target["labels"]
    if answer.get("type", qtype) != qtype:
        return {"valid": False, "correct": False, "reason": "type mismatch"}
    if qtype == "noul":
        p = answer.get("noul")
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            return {
                "valid": False,
                "correct": False,
                "reason": "invalid Noul probability",
            }
        probs = {"no": 1 - float(p), "yes": float(p)}
        strict_valid, renormalized = True, False
    else:
        validated = _valid_probs(answer.get("probabilities"), labels)
        if validated is None:
            return {
                "valid": False,
                "correct": False,
                "reason": "missing or invalid option probabilities",
            }
        probs, strict_valid, renormalized = validated
    # JevBench score_task uses argmax and breaks exact ties alphabetically.
    predicted = min(labels, key=lambda label: (-probs[label], label))
    expected = str(target["expected"])
    brier = (
        sum((value - float(label == expected)) ** 2 for label, value in probs.items())
        / 2
    )
    return {
        "valid": True,
        "strict_valid": strict_valid,
        "renormalized": renormalized,
        "correct": predicted == expected,
        "predicted": predicted,
        "brier": brier,
        "confidence": max(probs.values()),
        "point_disagrees_with_argmax": (
            answer.get("choice") != predicted if qtype == "choice" else False
        ),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _ece_15(evaluated: list[dict[str, Any]]) -> float | None:
    valid = [row for row in evaluated if row["valid"]]
    if not valid:
        return None
    bins: list[list[dict[str, Any]]] = [[] for _ in range(15)]
    for row in valid:
        bins[min(int(row["confidence"] * 15), 14)].append(row)
    return sum(
        len(bucket)
        / len(valid)
        * abs(
            statistics.mean(float(row["correct"]) for row in bucket)
            - statistics.mean(row["confidence"] for row in bucket)
        )
        for bucket in bins
        if bucket
    )


def _native_manifest(
    path: Path,
    predictions_path: Path,
    prompts_path: Path,
    model_id: str,
    model_revision: str,
    count: int,
) -> dict[str, Any]:
    """Bind a Decision2 collector's row-external model identity to exact bytes."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("model_id") != model_id
        or receipt.get("model_revision") != model_revision
        or receipt.get("predictions_sha256") != sha_file(predictions_path)
        or receipt.get("input_sha256") != sha_file(prompts_path)
        or receipt.get("input_items") != count
        or receipt.get("counts", {}).get("items") != count
    ):
        raise ValueError(
            "Native prediction manifest model/input/output binding differs"
        )
    for field in ("model_sha256", "adapter_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Native prediction manifest is missing {field}")
    return receipt


def score(
    panel_dir: Path,
    predictions_path: Path,
    model_id: str,
    model_revision: str,
    output: Path,
    prediction_manifest: Path | None = None,
) -> dict[str, Any]:
    manifest_path = panel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("build_version") != BUILD_VERSION or manifest.get("items") != 231:
        raise ValueError("Unknown JevBench public build")
    prompts_path, targets_path = (
        panel_dir / "prompts.jsonl",
        panel_dir / "targets.jsonl",
    )
    if (
        sha_file(prompts_path) != manifest["prompts_sha256"]
        or sha_file(targets_path) != manifest["targets_sha256"]
    ):
        raise ValueError("JevBench public panel changed")
    prompts, targets = _read_jsonl(prompts_path), _read_jsonl(targets_path)
    if len(prompts) != 231 or len(targets) != 231:
        raise ValueError("JevBench public panel count changed")
    expected = {}
    for prompt, target in zip(prompts, targets):
        digest = input_digest(prompt["state"], prompt["questions"])
        if prompt["id"] != target["id"] or target["source_input_sha256"] != digest:
            raise ValueError("Prompt/target pairing changed")
        expected[prompt["id"]] = target
    native = (
        _native_manifest(
            prediction_manifest,
            predictions_path,
            prompts_path,
            model_id,
            model_revision,
            len(prompts),
        )
        if prediction_manifest is not None
        else None
    )
    predictions = {}
    for row in _read_jsonl(predictions_path):
        item_id = row.get("id")
        native_identity = (
            (
                native is not None
                and row.get("model_id") in (None, model_id)
                and row.get("model_revision") in (None, model_revision)
                and row.get("model_sha256") == native["model_sha256"]
                and row.get("adapter_sha256") == native["adapter_sha256"]
                and row.get("input_sha256") == expected[item_id]["source_input_sha256"]
                and (
                    "calibration" not in native
                    or row.get("calibration_sha256")
                    == native["calibration"]["file_sha256"]
                )
            )
            if item_id in expected
            else False
        )
        inline_identity = (
            row.get("model_id") == (None if model_id == ABSENT_MODEL_ID else model_id)
            and row.get("model_revision") == model_revision
        )
        if (
            item_id not in expected
            or item_id in predictions
            or row.get("source_input_sha256")
            != expected[item_id]["source_input_sha256"]
            or not (inline_identity or native_identity)
        ):
            raise ValueError(
                "Unknown, duplicate, stale-input or wrong-model JevBench prediction"
            )
        predictions[item_id] = row
    evaluated = []
    for target in targets:
        row = predictions.get(target["id"])
        answer = row.get("answers", {}).get(QUESTION_ID) if row else None
        evaluated.append(
            {
                "id": target["id"],
                "tier": target["tier"],
                "family": target["family"],
                "type": target["task_type"],
                **_evaluate(answer, target),
            }
        )
    tiers = {}
    for tier in FILES:
        records = [entry for entry in evaluated if entry["tier"] == tier]
        tiers[tier] = {
            "items": len(records),
            "correct": sum(item["correct"] for item in records),
            "valid": sum(item["valid"] for item in records),
            "strict_valid": sum(item.get("strict_valid", False) for item in records),
            "renormalized": sum(item.get("renormalized", False) for item in records),
            "accuracy_all": statistics.mean(item["correct"] for item in records),
            "brier_valid": (
                statistics.mean(item["brier"] for item in records if item["valid"])
                if any(item["valid"] for item in records)
                else None
            ),
            "ece_pmax_15": _ece_15(records),
        }
    latency = [
        float(row["latency_ms"])
        for row in predictions.values()
        if type(row.get("latency_ms")) in (int, float)
        and math.isfinite(row["latency_ms"])
    ]
    report = {
        "score_version": SCORE_VERSION,
        "model_id": None if model_id == ABSENT_MODEL_ID else model_id,
        "model_revision": model_revision,
        "panel_manifest_sha256": sha_file(manifest_path),
        "prompts_sha256": manifest["prompts_sha256"],
        "targets_sha256": manifest["targets_sha256"],
        "predictions_sha256": sha_file(predictions_path),
        "prediction_manifest_sha256": (
            sha_file(prediction_manifest) if prediction_manifest is not None else None
        ),
        "items": len(targets),
        "answered": len(predictions),
        "valid": sum(entry["valid"] for entry in evaluated),
        "strict_valid": sum(entry.get("strict_valid", False) for entry in evaluated),
        "renormalized": sum(entry.get("renormalized", False) for entry in evaluated),
        "correct": sum(entry["correct"] for entry in evaluated),
        "accuracy_all": statistics.mean(entry["correct"] for entry in evaluated),
        "tier_macro_accuracy": statistics.mean(
            tier["accuracy_all"] for tier in tiers.values()
        ),
        "brier_valid": (
            statistics.mean(entry["brier"] for entry in evaluated if entry["valid"])
            if any(entry["valid"] for entry in evaluated)
            else None
        ),
        "ece_pmax_15": _ece_15(evaluated),
        "p50_latency_ms": _percentile(latency, 0.5),
        "p95_latency_ms": _percentile(latency, 0.95),
        "tiers": tiers,
        "point_argmax_disagreements": sum(
            entry.get("point_disagrees_with_argmax", False) for entry in evaluated
        ),
        "per_item": evaluated,
        "scope": "Independent public-only rerun; no sealed JevBench score or official rank",
        "source": {"url": SOURCE_URL, "revision": SOURCE_REVISION},
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        destination.write(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--upstream-root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--panel-dir", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--model-id", required=True)
    score_parser.add_argument("--model-revision", required=True)
    score_parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help="Explicit native collector receipt for row-external model identity",
    )
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.upstream_root, args.output_dir)
        print(json.dumps(result, sort_keys=True))
    else:
        result = score(
            args.panel_dir,
            args.predictions,
            args.model_id,
            args.model_revision,
            args.output,
            args.prediction_manifest,
        )
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("items", "answered", "valid", "accuracy_all")
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
