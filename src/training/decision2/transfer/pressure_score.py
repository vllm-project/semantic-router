"""Score the public pressure panel as a development diagnostic only.

Every row, including missing, malformed, and context-overflow responses, stays
in the accuracy denominator. Confidence metrics require a valid option map.
Repeated requests and multiple K values are clustered by source UID.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .pressure_build import (
    INSTRUCTIONS,
    PANEL_VERSION,
    SLICES,
    SOURCE_REVISION,
    SOURCE_SHA256,
    digest,
    file_sha256,
)

SCORE_VERSION = "decision2-pressure-score/2"


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            item_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(item_id, str) or not item_id or item_id in rows:
                raise ValueError(f"{path}:{line_number}: missing or duplicate id")
            rows[item_id] = row
    return rows


def load_panel(
    manifest_path: Path,
    slice_name: str,
    prompts_path: Path,
    gold_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("panel_version") != PANEL_VERSION
        or manifest.get("role") != "public_development_diagnostic_only"
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("source_sha256") != SOURCE_SHA256
        or manifest.get("quality_gate", {}).get("source_gates_passed") is not False
    ):
        raise ValueError("wrong pressure panel or missing failed-G3 provenance")
    if slice_name not in SLICES or slice_name not in manifest.get("slices", {}):
        raise ValueError("unknown pressure slice")
    slice_manifest = manifest["slices"][slice_name]
    for kind, path in (("prompts", prompts_path), ("gold", gold_path)):
        receipt = slice_manifest["files"][kind]
        if path.name != receipt["name"] or file_sha256(path) != receipt["sha256"]:
            raise ValueError(f"{kind} file does not match frozen panel manifest")
    prompts, gold = read_jsonl(prompts_path), read_jsonl(gold_path)
    expected = slice_manifest["main_requests"] + slice_manifest["control_requests"]
    if len(prompts) != expected or set(prompts) != set(gold):
        raise ValueError("prompt/gold IDs or expected slice count differ")
    seen_keys: set[tuple[str, str, int, int, bool]] = set()
    counts = Counter()
    for item_id, row in gold.items():
        prompt = prompts[item_id]
        if set(prompt) != {"id", "state", "questions"}:
            raise ValueError(f"{item_id}: prompt leaks metadata")
        question = (
            prompt["questions"].get("decision")
            if isinstance(prompt["questions"], dict)
            else None
        )
        if (
            not isinstance(question, dict)
            or set(prompt["questions"]) != {"decision"}
            or question.get("type") != "choice"
            or question.get("instructions") != INSTRUCTIONS
            or not isinstance(question.get("criteria"), dict)
        ):
            raise ValueError(f"{item_id}: wrong typed prompt")
        payload = {"state": prompt["state"], "questions": prompt["questions"]}
        labels = list(question["criteria"])
        if (
            row.get("panel_version") != PANEL_VERSION
            or row.get("slice") != slice_name
            or row.get("id") != item_id
            or row.get("input_sha256") != digest(payload)
            or not isinstance(row.get("uid"), str)
            or not row["uid"].startswith(row.get("domain", "") + ":")
            or row.get("text_sha256") != digest(prompt["state"].encode("utf-8"))
            or row.get("labels") != labels
            or len(labels) != row.get("k")
            or len(set(labels)) != len(labels)
            or any(question["criteria"][label] != label for label in labels)
            or row.get("gold") not in labels
            or row.get("option_order_sha256") != digest(labels)
            or row.get("option_set_sha256") != digest(sorted(labels))
        ):
            raise ValueError(f"{item_id}: malformed gold/provenance/payload")
        if (
            row.get("pool") not in SLICES[slice_name]["pools"]
            or row.get("k") not in SLICES[slice_name]["k"]
            or row.get("domain") not in SLICES[slice_name]["domains"]
            or type(row.get("leaked")) is not bool
            or type(row.get("repeat")) is not bool
            or type(row.get("permutation")) is not int
            or not 0 <= row["permutation"] < SLICES[slice_name]["permutations"]
            or (
                row["repeat"]
                and (slice_name != "rq2_shared_order" or row["permutation"] != 0)
            )
        ):
            raise ValueError(f"{item_id}: outside slice contract")
        key = (row["uid"], row["pool"], row["k"], row["permutation"], row["repeat"])
        if key in seen_keys:
            raise ValueError(f"{item_id}: duplicate variant key")
        seen_keys.add(key)
        counts["control" if row["repeat"] else "main"] += 1
    if (
        counts["main"] != slice_manifest["main_requests"]
        or counts["control"] != slice_manifest["control_requests"]
    ):
        raise ValueError("main/control request counts differ")
    if len({row["uid"] for row in gold.values()}) != SLICES[slice_name]["uids"]:
        raise ValueError("source UID count differs from frozen split")
    return manifest, gold


def _probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def evaluate(gold: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    if prediction is None:
        return {"valid": False, "reason": "missing_prediction", "choice": None}
    answer_map = prediction.get("answers")
    if not isinstance(answer_map, dict) or set(answer_map) != {"decision"}:
        return {"valid": False, "reason": "answer_map", "choice": None}
    answer = answer_map["decision"]
    if not isinstance(answer, dict) or answer.get("type") not in (None, "choice"):
        return {"valid": False, "reason": "answer_type", "choice": None}
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in gold["labels"]:
        reason = answer.get("invalid_reason") or answer.get("error") or "choice"
        # Adapters can record details elsewhere; keep the metric keys bounded.
        if reason not in {
            "context_overflow",
            "candidate_limit",
            "truncated",
            "admission_error",
        }:
            reason = "choice"
        return {"valid": False, "reason": reason, "choice": None}
    result = {
        "valid": True,
        "choice": choice,
        "correct": choice == gold["gold"],
        "probability_valid": False,
        "pmax": None,
        "gold_probability": None,
        "brier_sum": None,
        "nll": None,
        "option_sum_abs_delta": None,
    }
    if "probabilities" not in answer:
        return result
    probabilities = answer["probabilities"]
    if (
        not isinstance(probabilities, dict)
        or set(probabilities) != set(gold["labels"])
        or any(not _probability(value) for value in probabilities.values())
        or abs(sum(probabilities.values()) - 1.0) > 0.02
        or probabilities[choice] < max(probabilities.values()) - 0.02
    ):
        return {"valid": False, "reason": "probabilities", "choice": None}
    total = sum(probabilities.values())
    normalized = {label: probabilities[label] / total for label in gold["labels"]}
    gold_probability = normalized[gold["gold"]]
    result.update(
        {
            "probability_valid": True,
            "pmax": max(normalized.values()),
            "gold_probability": gold_probability,
            "brier_sum": sum(
                (normalized[label] - (label == gold["gold"])) ** 2
                for label in gold["labels"]
            ),
            "nll": -math.log(max(gold_probability, 1e-12)),
            "option_sum_abs_delta": abs(total - 1.0),
        }
    )
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    valid = [results[row["id"]] for row in rows if results[row["id"]]["valid"]]
    probability = [result for result in valid if result["probability_valid"]]
    wrong = [result["pmax"] for result in probability if not result["correct"]]
    latency = [
        predictions[row["id"]]["latency_ms"]
        for row in rows
        if row["id"] in predictions
        and type(predictions[row["id"]].get("latency_ms")) in (int, float)
        and math.isfinite(predictions[row["id"]]["latency_ms"])
        and predictions[row["id"]]["latency_ms"] >= 0
    ]
    return {
        "n": len(rows),
        "valid_n": len(valid),
        "probability_valid_n": len(probability),
        "correct_n": sum(result["correct"] for result in valid),
        "invalid_or_missing_n": len(rows) - len(valid),
        "invalid_reasons": dict(
            sorted(
                Counter(
                    results[row["id"]]["reason"]
                    for row in rows
                    if not results[row["id"]]["valid"]
                ).items()
            )
        ),
        "accuracy_all": (
            sum(result["correct"] for result in valid) / len(rows) if rows else None
        ),
        "accuracy_valid": _mean([float(result["correct"]) for result in valid]),
        "brier_sum": _mean([result["brier_sum"] for result in probability]),
        "nll": _mean([result["nll"] for result in probability]),
        "wrong_answer_pmax": _mean(wrong),
        "wrong_answer_pmax_n": len(wrong),
        "option_probability_sum_abs_delta_mean": _mean(
            [result["option_sum_abs_delta"] for result in probability]
        ),
        "latency_ms": {
            "n": len(latency),
            "median": statistics.median(latency) if latency else None,
            "p95": _percentile(latency, 0.95) if latency else None,
        },
    }


def _cluster_interval(
    pairs: list[dict[str, Any]],
    *,
    numerator: Callable[[dict[str, Any]], float],
    denominator: Callable[[dict[str, Any]], float],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_uid[pair["uid"]].append(pair)
    units = sorted(by_uid)
    uid_values = [
        (
            sum(numerator(row) for row in by_uid[uid]),
            sum(denominator(row) for row in by_uid[uid]),
        )
        for uid in units
    ]
    total_numerator = sum(value[0] for value in uid_values)
    total_denominator = sum(value[1] for value in uid_values)
    if not total_denominator:
        return {
            "estimate": None,
            "interval95": None,
            "uid_clusters": len(units),
            "denominator": 0,
        }
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        selected = [uid_values[rng.randrange(len(units))] for _ in units]
        sample_denominator = sum(value[1] for value in selected)
        if sample_denominator:
            draws.append(sum(value[0] for value in selected) / sample_denominator)
    return {
        "estimate": total_numerator / total_denominator,
        "interval95": (
            {"low": _percentile(draws, 0.025), "high": _percentile(draws, 0.975)}
            if draws
            else None
        ),
        "uid_clusters": len(units),
        "denominator": total_denominator,
    }


def near_far(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    index = {(row["uid"], row["k"], row["pool"]): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for uid, k, pool in sorted(index):
        if pool != "near":
            continue
        near = index[uid, k, "near"]
        far = index.get((uid, k, "far"))
        if (
            far is None
            or near["gold"] != far["gold"]
            or near["text_sha256"] != far["text_sha256"]
        ):
            raise ValueError("near/far pairing changed source text or gold")
        left, right = results[near["id"]], results[far["id"]]
        pairs.append(
            {
                "uid": uid,
                "domain": near["domain"],
                "leaked": near["leaked"],
                "k": k,
                "delta_far_minus_near": int(right.get("correct", False))
                - int(left.get("correct", False)),
                "joint_correct": int(
                    left.get("correct", False) and right.get("correct", False)
                ),
                "both_wrong": int(
                    not left.get("correct", False) and not right.get("correct", False)
                ),
                "gold_probability_delta": (
                    right["gold_probability"] - left["gold_probability"]
                    if left.get("probability_valid") and right.get("probability_valid")
                    else None
                ),
            }
        )
    expected = SLICES["rq3_shared_hardness"]["uids"] * len(
        SLICES["rq3_shared_hardness"]["k"]
    )
    if len(pairs) != expected:
        raise ValueError("near/far pair count differs from protocol")

    def measures(selected: list[dict[str, Any]], local_seed: int) -> dict[str, Any]:
        return {
            "pairs": len(selected),
            "accuracy_delta_far_minus_near": _cluster_interval(
                selected,
                numerator=lambda row: row["delta_far_minus_near"],
                denominator=lambda row: 1,
                iterations=iterations,
                seed=local_seed,
            ),
            "joint_correct_fraction": _cluster_interval(
                selected,
                numerator=lambda row: row["joint_correct"],
                denominator=lambda row: 1,
                iterations=iterations,
                seed=local_seed + 1,
            ),
            "both_wrong_fraction": _mean([row["both_wrong"] for row in selected]),
            "gold_probability_delta_far_minus_near": _cluster_interval(
                selected,
                numerator=lambda row: row["gold_probability_delta"] or 0,
                denominator=lambda row: row["gold_probability_delta"] is not None,
                iterations=iterations,
                seed=local_seed + 2,
            ),
        }

    return {
        "overall": measures(pairs, seed),
        "by_domain": {
            domain: measures(
                [row for row in pairs if row["domain"] == domain], seed + i + 1
            )
            for i, domain in enumerate(sorted({row["domain"] for row in pairs}))
        },
        "by_k": {
            str(k): measures([row for row in pairs if row["k"] == k], seed + k)
            for k in sorted({row["k"] for row in pairs})
        },
        "by_leaked": {
            str(leaked).lower(): measures(
                [row for row in pairs if row["leaked"] == leaked],
                seed + int(leaked) + 100,
            )
            for leaked in (False, True)
            if any(row["leaked"] == leaked for row in pairs)
        },
    }


def order_stability(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    index = {
        (row["uid"], row["pool"], row["permutation"], row["repeat"]): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for uid, pool, permutation, repeated in sorted(index):
        if permutation != 0 or repeated:
            continue
        base = index[uid, pool, 0, False]
        for variant in range(1, 5):
            candidate = index.get((uid, pool, variant, False))
            if (
                candidate is None
                or candidate["option_set_sha256"] != base["option_set_sha256"]
                or candidate["gold"] != base["gold"]
                or candidate["text_sha256"] != base["text_sha256"]
            ):
                raise ValueError("order variant changes candidate set")
            left, right = results[base["id"]], results[candidate["id"]]
            valid = left["valid"] and right["valid"]
            comparisons.append(
                {
                    "uid": uid,
                    "domain": base["domain"],
                    "leaked": base["leaked"],
                    "pool": pool,
                    "valid": valid,
                    "changed": bool(valid and left["choice"] != right["choice"]),
                    "changed_or_invalid": bool(
                        not valid or left["choice"] != right["choice"]
                    ),
                }
            )
        repeat = index.get((uid, pool, 0, True))
        if (
            repeat is None
            or repeat["input_sha256"] != base["input_sha256"]
            or repeat["option_order_sha256"] != base["option_order_sha256"]
        ):
            raise ValueError("identical-order control payload changed")
        left, right = results[base["id"]], results[repeat["id"]]
        valid = left["valid"] and right["valid"]
        controls.append(
            {
                "uid": uid,
                "domain": base["domain"],
                "leaked": base["leaked"],
                "pool": pool,
                "valid": valid,
                "changed": bool(valid and left["choice"] != right["choice"]),
                "changed_or_invalid": bool(
                    not valid or left["choice"] != right["choice"]
                ),
            }
        )
    expected_groups = SLICES["rq2_shared_order"]["uids"] * len(
        SLICES["rq2_shared_order"]["pools"]
    )
    if len(comparisons) != expected_groups * 4 or len(controls) != expected_groups:
        raise ValueError("order and repeat comparison counts differ from protocol")

    def measure(selected: list[dict[str, Any]], local_seed: int) -> dict[str, Any]:
        return {
            "pairs": len(selected),
            "valid_pairs": sum(row["valid"] for row in selected),
            "choice_flip_fraction_on_valid_pairs": _cluster_interval(
                selected,
                numerator=lambda row: row["changed"],
                denominator=lambda row: row["valid"],
                iterations=iterations,
                seed=local_seed,
            ),
            "flip_or_invalid_fraction_all": _cluster_interval(
                selected,
                numerator=lambda row: row["changed_or_invalid"],
                denominator=lambda row: 1,
                iterations=iterations,
                seed=local_seed + 1,
            ),
        }

    return {
        "order_changes": measure(comparisons, seed),
        "identical_order_control": measure(controls, seed + 1000),
        "order_by_domain": {
            domain: measure(
                [row for row in comparisons if row["domain"] == domain], seed + i + 2000
            )
            for i, domain in enumerate(sorted({row["domain"] for row in comparisons}))
        },
        "order_by_pool": {
            pool: measure(
                [row for row in comparisons if row["pool"] == pool], seed + i + 3000
            )
            for i, pool in enumerate(("near", "far"))
        },
        "order_by_leaked": {
            str(leaked).lower(): measure(
                [row for row in comparisons if row["leaked"] == leaked],
                seed + int(leaked) + 4000,
            )
            for leaked in (False, True)
            if any(row["leaked"] == leaked for row in comparisons)
        },
    }


def capacity_cluster_bootstrap(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Paired K-curve intervals by resampling complete source-UID vectors."""
    ks = tuple(SLICES["rq1_extended_capacity"]["k"])
    by_uid: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["k"] in by_uid[row["uid"]]:
            raise ValueError("RQ1 source UID has duplicate K")
        by_uid[row["uid"]][row["k"]] = results[row["id"]]
    units = sorted(by_uid)
    if len(units) != SLICES["rq1_extended_capacity"]["uids"] or any(
        set(by_uid[uid]) != set(ks) for uid in units
    ):
        raise ValueError("RQ1 source UID/K grid is incomplete")
    n = len(units)
    correct = {
        k: [int(by_uid[uid][k].get("correct", False)) for uid in units] for k in ks
    }
    valid = {k: [int(by_uid[uid][k]["valid"]) for uid in units] for k in ks}
    estimates = {k: sum(correct[k]) / n for k in ks}
    valid_estimates = {k: sum(valid[k]) / n for k in ks}
    rng = random.Random(seed)
    draws = {k: {"accuracy": [], "validity": []} for k in ks}
    for _ in range(iterations):
        chosen = [rng.randrange(n) for _ in units]
        for k in ks:
            draws[k]["accuracy"].append(sum(correct[k][i] for i in chosen) / n)
            draws[k]["validity"].append(sum(valid[k][i] for i in chosen) / n)

    def interval(values: list[float]) -> dict[str, float]:
        return {"low": _percentile(values, 0.025), "high": _percentile(values, 0.975)}

    curve = {}
    for index, k in enumerate(ks):
        previous = ks[max(0, index - 1)]
        baseline = ks[0]
        curve[str(k)] = {
            "accuracy_all": {
                "estimate": estimates[k],
                "interval95": interval(draws[k]["accuracy"]),
            },
            "validity_fraction": {
                "estimate": valid_estimates[k],
                "interval95": interval(draws[k]["validity"]),
            },
            "delta_accuracy_vs_k2": {
                "estimate": estimates[k] - estimates[baseline],
                "interval95": interval(
                    [
                        a - b
                        for a, b in zip(
                            draws[k]["accuracy"], draws[baseline]["accuracy"]
                        )
                    ]
                ),
            },
            "delta_accuracy_vs_previous_k": {
                "previous_k": previous,
                "estimate": estimates[k] - estimates[previous],
                "interval95": interval(
                    [
                        a - b
                        for a, b in zip(
                            draws[k]["accuracy"], draws[previous]["accuracy"]
                        )
                    ]
                ),
            },
        }
    return {
        "method": "paired percentile cluster bootstrap by source UID",
        "iterations": iterations,
        "seed": seed,
        "uid_clusters": n,
        "requests_per_k": n,
        "by_k": curve,
    }


def score(
    manifest_path: Path,
    slice_name: str,
    prompts_path: Path,
    gold_path: Path,
    predictions_path: Path,
    *,
    model: str,
    revision: str,
    adapter_version: str,
    deployment_profile: str = "unmatched",
    source_overlap: str = "not_audited",
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260926,
) -> dict[str, Any]:
    if (
        not model
        or not revision
        or not adapter_version
        or bootstrap_iterations < 100
        or type(bootstrap_seed) is not int
    ):
        raise ValueError(
            "model/revision/adapter and >=100 integer-seeded bootstrap iterations required"
        )
    manifest, gold = load_panel(manifest_path, slice_name, prompts_path, gold_path)
    predictions = read_jsonl(predictions_path)
    if set(predictions) - set(gold):
        raise ValueError("prediction file contains an ID outside frozen panel")
    for item_id, prediction in predictions.items():
        for receipt_key, expected in (
            ("model_id", model),
            ("model_revision", revision),
            ("adapter_version", adapter_version),
        ):
            if prediction.get(receipt_key) != expected:
                raise ValueError(
                    f"{item_id}: prediction {receipt_key} differs from score identity"
                )
        if prediction.get("source_input_sha256") != gold[item_id]["input_sha256"]:
            raise ValueError(
                f"{item_id}: prediction source_input_sha256 differs from frozen input"
            )
    results = {
        item_id: evaluate(row, predictions.get(item_id))
        for item_id, row in gold.items()
    }
    main = [row for row in gold.values() if not row["repeat"]]
    controls = [row for row in gold.values() if row["repeat"]]
    strata: dict[str, Any] = {
        "all_main_requests": summarize(main, results, predictions),
        "by_domain": {
            domain: summarize(
                [row for row in main if row["domain"] == domain], results, predictions
            )
            for domain in sorted({row["domain"] for row in main})
        },
        "by_domain_and_leaked": {
            f"{domain}/{str(leaked).lower()}": summarize(
                [
                    row
                    for row in main
                    if row["domain"] == domain and row["leaked"] == leaked
                ],
                results,
                predictions,
            )
            for domain in sorted({row["domain"] for row in main})
            for leaked in (False, True)
            if any(row["domain"] == domain and row["leaked"] == leaked for row in main)
        },
        "by_k": {
            str(k): summarize(
                [row for row in main if row["k"] == k], results, predictions
            )
            for k in sorted({row["k"] for row in main})
        },
        "controls": summarize(controls, results, predictions) if controls else None,
    }
    if slice_name == "rq3_shared_hardness":
        strata["by_pool"] = {
            pool: summarize(
                [row for row in main if row["pool"] == pool], results, predictions
            )
            for pool in ("near", "far")
        }
        strata["paired_near_far"] = near_far(
            main, results, iterations=bootstrap_iterations, seed=bootstrap_seed
        )
    if slice_name == "rq2_shared_order":
        strata["by_pool"] = {
            pool: summarize(
                [row for row in main if row["pool"] == pool], results, predictions
            )
            for pool in ("near", "far")
        }
        strata["order_stability"] = order_stability(
            list(gold.values()),
            results,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
    if slice_name == "rq1_extended_capacity":
        strata["rq1_uid_cluster_bootstrap"] = capacity_cluster_bootstrap(
            main, results, iterations=bootstrap_iterations, seed=bootstrap_seed
        )
    return {
        "score_version": SCORE_VERSION,
        "panel_version": PANEL_VERSION,
        "role": "public_development_diagnostic_only",
        "slice": slice_name,
        "model": model,
        "revision": revision,
        "adapter_version": adapter_version,
        "deployment_profile": deployment_profile,
        "source_overlap": source_overlap,
        "input_sha256": {
            "manifest": file_sha256(manifest_path),
            "prompts": file_sha256(prompts_path),
            "gold": file_sha256(gold_path),
            "predictions": file_sha256(predictions_path),
        },
        "score_code_sha256": file_sha256(Path(__file__)),
        "source_revision": manifest["source_revision"],
        "source_data_license": manifest["source_data_license"],
        "quality_gate": manifest["quality_gate"],
        "metric_policy": {
            "main_denominator": "All main requests; missing, invalid, and truncated responses are misses. Order repeats are separate controls.",
            "receipt_identity": "Every present prediction must attest the exact model_id, model_revision and adapter_version named in this score report.",
            "choice": "Returned semantic option string must occur in the frozen candidate set.",
            "probabilities": "Optional; if supplied, finite complete option map within 0.02 of unit sum and chosen option within 0.02 of max. Normalize valid map to unit sum for multiclass Brier sum, NLL and confidence.",
            "paired_near_far": "Same source UID and K; far accuracy minus near accuracy. CI resamples source UIDs.",
            "order_flip": "Changed semantic answer vs permutation zero on same UID, pool and candidate set. Report valid-pair and all-pair-with-invalid rates separately.",
            "identical_order_control": "Permutation zero repeated verbatim with a new ID; diagnostic of nondeterminism/cache effects, never subtracted blindly.",
            "bootstrap": {
                "method": "percentile cluster bootstrap by source UID",
                "iterations": bootstrap_iterations,
                "seed": bootstrap_seed,
                "confidence_level": 0.95,
            },
            "rq1_k_curve": "RQ1 resamples complete K vectors for each source UID, sharing draws across K for paired accuracy deltas and validity intervals.",
            "interpretation": "Public items and failed text-blind-picker gate make absolute accuracy unsuitable as a final release claim.",
        },
        "metrics": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slice", choices=tuple(SLICES), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter-version", required=True)
    parser.add_argument("--deployment-profile", default="unmatched")
    parser.add_argument("--source-overlap", default="not_audited")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = score(
        args.manifest,
        args.slice,
        args.prompts,
        args.gold,
        args.predictions,
        model=args.model,
        revision=args.revision,
        adapter_version=args.adapter_version,
        deployment_profile=args.deployment_profile,
        source_overlap=args.source_overlap,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accuracy_all": report["metrics"]["all_main_requests"]["accuracy_all"],
            }
        )
    )


if __name__ == "__main__":
    main()
