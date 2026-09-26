"""Score standardized typed-decision predictions against a private gold JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .generate import FAMILIES, SCHEMA_VERSION, digest


class DataError(ValueError):
    pass


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                raise DataError(f"{path}:{line_number}: blank line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise DataError(f"{path}:{line_number}: row needs string id")
            if row["id"] in result:
                raise DataError(f"{path}:{line_number}: duplicate id {row['id']}")
            result[row["id"]] = row
    if not result:
        raise DataError(f"{path}: empty JSONL")
    return result


def validate_suite(
    items: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, list[tuple[str, str, str]]]]:
    splits = set()
    pair_rows: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    group_rows: dict[str, list[str]] = defaultdict(list)
    for item_id, item in items.items():
        if item.get("schema_version") != SCHEMA_VERSION:
            raise DataError(f"{item_id}: unexpected schema version")
        split, family = item.get("split"), item.get("family")
        if split not in FAMILIES or family not in FAMILIES[split]:
            raise DataError(f"{item_id}: invalid family/split")
        splits.add(split)
        group_id = item.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise DataError(f"{item_id}: missing group_id")
        group_rows[group_id].append(item_id)
        payload = {"state": item.get("state"), "questions": item.get("questions")}
        provenance = item.get("provenance", {})
        if (
            provenance.get("source") != "programmatic_synthetic"
            or provenance.get("generator_family") != family
        ):
            raise DataError(f"{item_id}: invalid provenance")
        if provenance.get("payload_sha256") != digest(payload):
            raise DataError(f"{item_id}: payload digest mismatch")
        if (
            not isinstance(provenance.get("seed_commitment_sha256"), str)
            or len(provenance["seed_commitment_sha256"]) != 64
        ):
            raise DataError(f"{item_id}: missing seed commitment")
        questions, gold = item.get("questions"), item.get("gold")
        if (
            not isinstance(questions, dict)
            or not questions
            or not isinstance(gold, dict)
            or questions.keys() != gold.keys()
        ):
            raise DataError(f"{item_id}: questions/gold mismatch")
        for key, question in questions.items():
            answer = gold[key]
            if not isinstance(answer, dict) or answer.get("type") != question.get(
                "type"
            ):
                raise DataError(f"{item_id}/{key}: mismatched gold type")
            qtype, value = question["type"], answer.get("value")
            if qtype == "choice":
                labels = question.get("criteria")
                if (
                    not isinstance(labels, dict)
                    or len(labels) < 2
                    or value not in labels
                ):
                    raise DataError(f"{item_id}/{key}: invalid choice labels/gold")
                mapping = answer.get("label_to_semantic")
                if (
                    not isinstance(mapping, dict)
                    or set(mapping) != set(labels)
                    or mapping[value] != answer.get("semantic_value")
                ):
                    raise DataError(f"{item_id}/{key}: invalid semantic map")
            elif qtype == "score":
                criteria = question.get("criteria")
                if (
                    not isinstance(criteria, list)
                    or not 2 <= len(criteria) <= 10
                    or type(value) is not int
                    or not 0 <= value < len(criteria)
                ):
                    raise DataError(f"{item_id}/{key}: invalid score criteria/gold")
            elif qtype == "noul":
                if type(value) is not bool:
                    raise DataError(f"{item_id}/{key}: noul gold must be Boolean")
            else:
                raise DataError(f"{item_id}/{key}: unsupported question type")
        pairs = item.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise DataError(f"{item_id}: missing pair membership")
        for pair in pairs:
            pair_id, relation, role = (
                pair.get("id"),
                pair.get("relation"),
                pair.get("role"),
            )
            if (
                not isinstance(pair_id, str)
                or relation
                not in ("counterfactual", "order_invariance", "label_invariance")
                or role not in ("anchor", "variant")
            ):
                raise DataError(f"{item_id}: invalid pair metadata")
            pair_rows[pair_id].append((item_id, relation, role))
    if len(splits) != 1:
        raise DataError("one gold JSONL must contain exactly one split")
    for group_id, members in group_rows.items():
        if len(members) != 4:
            raise DataError(f"{group_id}: expected four variants, got {len(members)}")
    for pair_id, members in pair_rows.items():
        if (
            len(members) != 2
            or {m[2] for m in members} != {"anchor", "variant"}
            or len({m[1] for m in members}) != 1
        ):
            raise DataError(
                f"{pair_id}: pair must have one anchor and one variant of the same relation"
            )
        first, second = (items[m[0]] for m in members)
        if (
            first["group_id"] != second["group_id"]
            or first["questions"].keys() != second["questions"].keys()
        ):
            raise DataError(f"{pair_id}: pair crosses group/question set")
        relation = members[0][1]
        for key in first["questions"]:
            left, right = (
                first["gold"][key]["semantic_value"],
                second["gold"][key]["semantic_value"],
            )
            if relation == "counterfactual" and left == right:
                raise DataError(f"{pair_id}/{key}: counterfactual does not change gold")
            if relation != "counterfactual" and left != right:
                raise DataError(f"{pair_id}/{key}: invariance changes gold")
    return next(iter(splits)), pair_rows


def number(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def probability_map(raw: Any, labels: list[str]) -> dict[str, float] | None:
    if not isinstance(raw, dict) or not raw or not set(raw) <= set(labels):
        return None
    values = {label: number(raw[label]) if label in raw else 0.0 for label in labels}
    if any(value is None or not 0 <= value <= 1 for value in values.values()):
        return None
    if abs(sum(values.values()) - 1.0) > 0.02:
        return None
    return values  # type: ignore[return-value]


def unique_argmax(values: dict[str, float]) -> str | None:
    best = max(values.values())
    labels = [key for key, value in values.items() if abs(value - best) <= 1e-8]
    return labels[0] if len(labels) == 1 else None


def evaluate_answer(
    question: dict[str, Any], gold: dict[str, Any], answer: Any
) -> dict[str, Any]:
    if not isinstance(answer, dict):
        return {"status": "invalid", "reason": "answer must be object"}
    qtype = question["type"]
    if "type" in answer and answer["type"] != qtype:
        return {"status": "invalid", "reason": "answer type mismatch"}
    point: str | int | bool | None
    probs: dict[str, float] | None = None
    raw_score = None
    if qtype == "noul":
        p = number(answer.get("noul"))
        if p is None or not 0 <= p <= 1:
            return {"status": "invalid", "reason": "noul must be probability in [0,1]"}
        point = None if p == 0.5 else p > 0.5
        probs = {"false": 1 - p, "true": p}
    elif qtype == "choice":
        labels = list(question["criteria"])
        point = answer.get("choice")
        if not isinstance(point, str) or point not in labels:
            return {"status": "invalid", "reason": "choice must be one offered label"}
        if "probabilities" in answer:
            probs = probability_map(answer["probabilities"], labels)
            if probs is None:
                return {
                    "status": "invalid",
                    "reason": "choice probabilities need all labels and sum to one",
                }
            if probs[point] < max(probs.values()) - 0.02:
                return {
                    "status": "invalid",
                    "reason": "choice disagrees with highest probability",
                }
    else:
        raw_score = number(answer.get("score"))
        levels = list(range(len(question["criteria"])))
        if raw_score is None or not 0 <= raw_score <= levels[-1]:
            return {"status": "invalid", "reason": "score outside ordered level range"}
        if "probabilities" in answer:
            probs = probability_map(
                answer["probabilities"], [str(level) for level in levels]
            )
            if probs is None:
                return {
                    "status": "invalid",
                    "reason": "score probabilities need all levels and sum to one",
                }
            expected = sum(level * probs[str(level)] for level in levels)
            if abs(raw_score - expected) > 0.06:
                return {
                    "status": "invalid",
                    "reason": "score disagrees with probability-weighted mean",
                }
            argmax = unique_argmax(probs)
            point = int(argmax) if argmax is not None else None
        else:
            nearest = round(raw_score)
            point = nearest if abs(raw_score - nearest) < 0.5 else None
    correct = point == gold["value"]
    if qtype == "choice":
        semantic = gold["label_to_semantic"].get(point) if point is not None else None
    else:
        semantic = point
    result: dict[str, Any] = {
        "status": "ok",
        "point": point,
        "semantic_point": semantic,
        "correct": correct,
    }
    if raw_score is not None:
        result["absolute_error"] = abs(raw_score - gold["value"])
    if probs is not None:
        option_sum_abs_delta = None
        if qtype in ("choice", "score"):
            total = sum(probs.values())
            option_sum_abs_delta = abs(total - 1.0)
            # The native answer and validity checks above use the original map.
            # Probability metrics use a proper distribution after acceptance.
            probs = {label: value / total for label, value in probs.items()}
        truth = str(gold["value"]).lower() if qtype == "noul" else str(gold["value"])
        # Divide multiclass squared error by two so one-hot wrong predictions
        # have Brier 1, matching the usual binary (p - y)^2 range.
        brier = sum((p - float(label == truth)) ** 2 for label, p in probs.items()) / 2
        confidence = max(probs.values())
        result["probability"] = {
            "brier": brier,
            "nll": -math.log(max(probs[truth], 1e-12)),
            "confidence": confidence,
        }
        if option_sum_abs_delta is not None:
            result["probability"]["option_sum_abs_delta"] = option_sum_abs_delta
    return result


def option_sum_diagnostics(
    probabilities: list[dict[str, float]],
) -> dict[str, int | float | None]:
    deltas = [
        p["option_sum_abs_delta"] for p in probabilities if "option_sum_abs_delta" in p
    ]
    return {
        "n": len(deltas),
        "mean": statistics.mean(deltas) if deltas else None,
        "p50": percentile(deltas, 0.5),
        "p95": percentile(deltas, 0.95),
        "p99": percentile(deltas, 0.99),
        "max": max(deltas) if deltas else None,
        "over_0_001_n": sum(delta > 0.001 for delta in deltas),
        "over_0_01_n": sum(delta > 0.01 for delta in deltas),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    valid = [r for r in results if r["status"] == "ok"]
    correct = sum(bool(r["correct"]) for r in valid)
    probability = [r["probability"] for r in valid if "probability" in r]
    errors = [r["absolute_error"] for r in valid if "absolute_error" in r]
    output: dict[str, Any] = {
        "n": count,
        "valid_n": len(valid),
        "invalid_or_missing_n": count - len(valid),
        "correct_n": correct,
        "accuracy_all": correct / count if count else None,
        "accuracy_valid": correct / len(valid) if valid else None,
        "probability_n": len(probability),
        "option_probability_sum_abs_delta": option_sum_diagnostics(probability),
        "brier": (
            statistics.mean(p["brier"] for p in probability) if probability else None
        ),
        "nll": statistics.mean(p["nll"] for p in probability) if probability else None,
        "ece_10": ece(valid) if probability else None,
        "score_mae": statistics.mean(errors) if errors else None,
    }
    output["selective"] = {}
    for threshold in (0.5, 0.8, 0.95):
        subset = [
            r
            for r in valid
            if "probability" in r and r["probability"]["confidence"] >= threshold
        ]
        output["selective"][str(threshold)] = {
            "n": len(subset),
            "coverage_of_probability_answers": (
                len(subset) / len(probability) if probability else None
            ),
            "accuracy": (
                sum(bool(r["correct"]) for r in subset) / len(subset)
                if subset
                else None
            ),
        }
    return output


def ece(results: list[dict[str, Any]]) -> float | None:
    observed = [r for r in results if "probability" in r]
    if not observed:
        return None
    bins: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for result in observed:
        confidence = result["probability"]["confidence"]
        bins[min(9, int(confidence * 10))].append(result)
    return sum(
        len(bucket)
        / len(observed)
        * abs(
            statistics.mean(r["probability"]["confidence"] for r in bucket)
            - statistics.mean(float(r["correct"]) for r in bucket)
        )
        for bucket in bins
        if bucket
    )


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percentile_value
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def abstention_summary(
    items: dict[str, dict[str, Any]], per_answer: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, int | float | None]:
    total = true_positive = false_positive = false_negative = 0
    for item_id, item in items.items():
        for key, question in item["questions"].items():
            if (
                question["type"] != "choice"
                or "insufficient_evidence" not in question["criteria"]
            ):
                continue
            total += 1
            gold_abstain = item["gold"][key]["value"] == "insufficient_evidence"
            result = per_answer[(item_id, key)]
            predicted_abstain = (
                result["status"] == "ok" and result["point"] == "insufficient_evidence"
            )
            true_positive += bool(predicted_abstain and gold_abstain)
            false_positive += bool(predicted_abstain and not gold_abstain)
            false_negative += bool(not predicted_abstain and gold_abstain)
    return {
        "n": total,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
    }


def evidence_cross_question_consistency(
    items: dict[str, dict[str, Any]], per_answer: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, int | float | None]:
    count = valid = consistent = 0
    for item_id, item in items.items():
        if item["family"] != "evidence_join":
            continue
        count += 1
        decision, determinate = (
            per_answer[(item_id, "decision")],
            per_answer[(item_id, "determinate")],
        )
        if (
            decision["status"] == determinate["status"] == "ok"
            and type(determinate["point"]) is bool
        ):
            valid += 1
            consistent += bool(
                (decision["point"] == "insufficient_evidence")
                == (determinate["point"] is False)
            )
    return {
        "n": count,
        "valid_n": valid,
        "consistent_n": consistent,
        "consistency_all": consistent / count if count else None,
    }


def score_suite(
    gold_path: Path, predictions_path: Path, model_id: str, revision: str, backend: str
) -> dict[str, Any]:
    items = load_jsonl(gold_path)
    split, pairs = validate_suite(items)
    predictions = load_jsonl(predictions_path)
    unexpected = sorted(set(predictions) - set(items))
    if unexpected:
        raise DataError(
            f"predictions contain {len(unexpected)} unknown IDs; first: {unexpected[0]}"
        )
    for item_id, prediction in predictions.items():
        expected_hash = items[item_id]["provenance"]["payload_sha256"]
        if prediction.get("source_input_sha256") != expected_hash:
            raise DataError(
                f"{item_id}: prediction source_input_sha256 is missing or differs from gold payload"
            )
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_answer: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_reasons: dict[str, int] = defaultdict(int)
    latency = []
    cost = []
    input_tokens = []
    output_tokens = []
    for item_id, item in items.items():
        prediction = predictions.get(item_id)
        answers = prediction.get("answers") if isinstance(prediction, dict) else None
        if prediction is not None:
            elapsed = number(prediction.get("latency_ms"))
            if elapsed is not None and elapsed >= 0:
                latency.append(elapsed)
            price = number(prediction.get("cost_usd"))
            if price is not None and price >= 0:
                cost.append(price)
            usage = prediction.get("usage")
            if isinstance(usage, dict):
                for field, dest in (
                    ("input_tokens", input_tokens),
                    ("output_tokens", output_tokens),
                ):
                    value = number(usage.get(field))
                    if value is not None and value >= 0:
                        dest.append(value)
        for key, question in item["questions"].items():
            if prediction is None:
                result = {"status": "missing", "reason": "missing item prediction"}
            elif not isinstance(answers, dict) or set(answers) != set(
                item["questions"]
            ):
                result = {
                    "status": "invalid",
                    "reason": "answers need exactly the requested question IDs",
                }
            else:
                result = evaluate_answer(question, item["gold"][key], answers[key])
            if result["status"] == "invalid":
                invalid_reasons[result["reason"]] += 1
            by_type[question["type"]].append(result)
            by_family[item["family"]].append(result)
            per_answer[(item_id, key)] = result
    pair_metrics: dict[str, dict[str, int | float | None]] = {}
    for relation in ("counterfactual", "order_invariance", "label_invariance"):
        selected = [members for members in pairs.values() if members[0][1] == relation]
        total = valid_count = consistent = joint_correct = 0
        for members in selected:
            anchor_id = next(
                item_id for item_id, _, role in members if role == "anchor"
            )
            variant_id = next(
                item_id for item_id, _, role in members if role == "variant"
            )
            for key in items[anchor_id]["questions"]:
                total += 1
                left, right = (
                    per_answer[(anchor_id, key)],
                    per_answer[(variant_id, key)],
                )
                if left["status"] == right["status"] == "ok":
                    valid_count += 1
                    same = left["semantic_point"] == right["semantic_point"]
                    consistent += (not same) if relation == "counterfactual" else same
                    joint_correct += bool(left["correct"] and right["correct"])
        pair_metrics[relation] = {
            "n": total,
            "valid_n": valid_count,
            "relation_consistent_n": consistent,
            "relation_consistency_all": consistent / total if total else None,
            "joint_correct_n": joint_correct,
            "joint_accuracy_all": joint_correct / total if total else None,
        }
    all_results = [result for group in by_type.values() for result in group]
    family_summary = {
        family: summarize(by_family[family]) for family in sorted(by_family)
    }
    report = {
        "schema_version": "typed-decision-report/2",
        "metric_policy": {
            "point_accuracy": "Uses the accepted native answer; invalid and missing answers count as misses.",
            "probability_acceptance": "Choice and Score option probabilities must each be finite in [0,1], with the original sum within 0.02 of one; existing answer consistency checks use original probabilities.",
            "probability_metrics": "For accepted Choice and Score maps, divide each option by the original sum before Brier, NLL, ECE, and selective confidence. Noul uses its supplied scalar probability.",
            "option_probability_sum_abs_delta": "Absolute original option-probability sum minus one, for valid Choice and Score maps only.",
        },
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "split": split,
        "model": {"id": model_id, "revision": revision, "backend": backend},
        "items": len(items),
        "predicted_items": len(predictions),
        "overall": summarize(all_results),
        "macro_family_accuracy": statistics.mean(
            s["accuracy_all"] for s in family_summary.values()
        ),
        "by_type": {
            qtype: summarize(results) for qtype, results in sorted(by_type.items())
        },
        "by_family": family_summary,
        "pairs": pair_metrics,
        "abstention": abstention_summary(items, per_answer),
        "evidence_cross_question_consistency": evidence_cross_question_consistency(
            items, per_answer
        ),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "runtime": {
            "latency_n": len(latency),
            "median_latency_ms": percentile(latency, 0.5),
            "p90_latency_ms": percentile(latency, 0.9),
            "p95_latency_ms": percentile(latency, 0.95),
            "cost_n": len(cost),
            "total_cost_usd": sum(cost) if cost else None,
            "input_tokens_n": len(input_tokens),
            "total_input_tokens": int(sum(input_tokens)) if input_tokens else None,
            "output_tokens_n": len(output_tokens),
            "total_output_tokens": int(sum(output_tokens)) if output_tokens else None,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score_suite(
        args.gold, args.predictions, args.model_id, args.model_revision, args.backend
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
