"""One-use blinded r2 English selector scorer for the frozen Score v6 pair.

``seal`` has no key argument. ``score`` first revalidates the complete seal,
then marks the one permitted unblinding before opening the key. This is a
selection diagnostic, not a release evaluation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from training.model.data import file_sha256, load_partition
from training.model.infer import (
    ADAPTER_VERSION,
    checkpoint_fingerprint,
    load_prompts,
    prompt_input_sha256,
)

VERSION = "decision2-score-en-r2-blind-scorer/1"
SOURCE_SHA = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
PREP_SHA = "b10eae10192096c2dcf52af60ad6805c5180abd00f7ff2a5af52a4f5dadc736c"
PROMPT_SHA = "3ad89c37170de17ccda63f149aaf14a7d4112e75db01741a3ef1b4b2c161b726"
SOURCE_PRED_SHA = "8d8bff54cedfdd17257cb7328006b1e86af6588cedcab7448005ce26301b0548"
SOURCE_PRED_MANIFEST_SHA = (
    "b1e58549ce168ee27119baab8444ab900453c5a2851916ec0a276f89545de50e"
)
KEY_SHA = "8e9d1232c1d0db75d7fc1e583d07753e5db6716c85aca39040744a2ec834eda1"
PARITY_SHA = "f09802ee3da2921d66949796ab7efdab9da6c6b7dfb5fb48bb472a99e10441fa"
ARM_SHA = {
    "A": "6a6ef7d3f2eac2a63cdd61cd806275e67c0aa772e78b45bf0953200f2a776235",
    "B": "1c705c9a8271ce2e526b6bc91affe18d463a52bb86ce99b6b1bcb5007d467b41",
}
SELECT_SHA = "e41774cfc6f1dcea58fa0baa3c8a0cef3940af36fcd464f2b0d5ca01598a95ce"
CAL_SHA = "ff6d80d04a07427939ac1ec4fe807e57fdaa921ff90e0abb50a033094cc485a8"
OPS = (
    "allocation_caps",
    "inclusive_coverage",
    "independent_quorum",
    "waiver_precedence",
)
BOOTSTRAP_SEED_TEXT = "decision2-score-v6-en-r2-stratified-bootstrap-20260927-v1"
BOOTSTRAP_DRAWS = 10_000
FINAL_CHECKPOINT = "checkpoint-0000174"
RAW_TOKENS = 474_124


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path.name}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            require(bool(line.strip()), f"Blank JSONL line: {path.name}:{index}")
            row = json.loads(line)
            require(isinstance(row, dict), f"Non-object JSONL: {path.name}:{index}")
            rows.append(row)
    require(bool(rows), f"Empty JSONL: {path.name}")
    return rows


def _private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _prompt_roster(path: Path) -> list[dict[str, Any]]:
    require(file_sha256(path) == PROMPT_SHA, "Frozen r2 English prompt SHA differs")
    rows = load_prompts(path)
    require(len(rows) == 192, "r2 English native prompt count differs")
    for item in rows:
        questions = item["questions"]
        require(
            set(questions) == {"decision"}
            and questions["decision"].get("type") == "score"
            and isinstance(questions["decision"].get("criteria"), list)
            and len(questions["decision"]["criteria"]) == 3,
            "r2 English native prompt is not three-level Score",
        )
    return rows


def _prepared(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    require(
        file_sha256(manifest_path) == PREP_SHA,
        "Frozen arm preparation manifest differs",
    )
    manifest = _json(manifest_path)
    outputs = manifest.get("output_sha256", {})
    expected = {
        "arm_a_train": ("arm_a.train.jsonl", ARM_SHA["A"]),
        "arm_b_train": ("arm_b.train.jsonl", ARM_SHA["B"]),
        "parent_select_en": ("parent_en.select.jsonl", SELECT_SHA),
        "parent_cal_en": ("parent_en.cal.jsonl", CAL_SHA),
        "r2_en_native_prompts": ("r2_en.native-prompts.jsonl", PROMPT_SHA),
    }
    for role, (filename, expected_sha) in expected.items():
        require(
            outputs.get(role) == expected_sha
            and file_sha256(path / filename) == expected_sha,
            f"Prepared {role} differs",
        )
    require(
        manifest.get("schema_version") == "decision2-score-v6-en-matched-arms/1"
        and manifest.get("source_model_sha256") == SOURCE_SHA
        and manifest.get("arm_rows") == 2777
        and manifest.get("english_r2_groups") == 64
        and manifest.get("v6_native_tokens") == 220_932
        and manifest.get("control_native_tokens") == 220_932
        and isinstance(manifest.get("padding_relative_delta"), (int, float))
        and manifest["padding_relative_delta"] <= 0.05,
        "Frozen data exposure or padding gate failed",
    )
    return manifest


def _run(run: Path, arm: str) -> dict[str, Any]:
    complete = _json(run / "COMPLETE.json")
    latest = _json(run / "LATEST.json")
    best = _json(run / "BEST.json")
    checkpoint = run / FINAL_CHECKPOINT
    metadata = _json(checkpoint / "checkpoint.json")
    require(
        complete.get("status") == "complete"
        and complete.get("step") == complete.get("planned_updates") == 174
        and complete.get("best") == FINAL_CHECKPOINT
        and latest.get("checkpoint") == FINAL_CHECKPOINT
        and latest.get("step") == 174
        and best.get("checkpoint") == FINAL_CHECKPOINT
        and metadata.get("step") == 174
        and metadata.get("complete") is True,
        f"Arm {arm} is not complete at fixed step 174",
    )
    require(
        [path.name for path in run.glob("checkpoint-*") if path.is_dir()]
        == [FINAL_CHECKPOINT],
        f"Arm {arm} contains an unregistered checkpoint candidate",
    )
    provenance = _json(run / "provenance.json")
    contract = provenance.get("contract", {})
    code_files = provenance.get("code_sha256")
    require(
        isinstance(code_files, dict)
        and set(code_files)
        == {
            "data.py",
            "decision_model.py",
            "loss.py",
            "plan.py",
            "source.py",
            "train.py",
            "lora.py",
            "infer.py",
        }
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in code_files.values()
        ),
        f"Arm {arm} trainer source-code manifest differs",
    )
    require(
        contract.get("init_kind") == "decision2-lora"
        and contract.get("train_mode") == "lora"
        and contract.get("direct_lora_arm") == arm
        and contract.get("initial_model_sha256") == SOURCE_SHA
        and contract.get("direct_lora_parity_sha256") == PARITY_SHA
        and contract.get("data_sha256")
        == {"train": ARM_SHA[arm], "select": SELECT_SHA, "cal": CAL_SHA}
        and contract.get("epochs") == 1
        and contract.get("max_steps") == 174
        and contract.get("planned_updates") == 174
        and contract.get("microbatch") == 1
        and contract.get("accumulation") == 16
        and contract.get("max_length") == 1024
        and contract.get("head_dim") == 256
        and contract.get("lora", {}).get("rank") == 8
        and contract.get("lora", {}).get("alpha") == 16
        and contract.get("lora", {}).get("dropout") == 0.05
        and contract.get("lora", {}).get("lr") == 2e-5
        and contract.get("head_lr") == 1e-5
        and contract.get("weight_decay") == 0.01
        and contract.get("warmup_ratio") == 0.05
        and contract.get("objective") == "ce"
        and contract.get("replay_fraction") == 0.0
        and contract.get("replay_kl_weight") == 0.0
        and contract.get("seed") == 20260927
        and contract.get("gradient_checkpointing") is True
        and provenance.get("initial_model_identity", {}).get("model_sha256")
        == SOURCE_SHA
        and provenance.get("train_examples") == 2777
        and provenance.get("replay_pool_examples") == 0
        and provenance.get("replay_examples_per_epoch") == 0
        and provenance.get("select_examples") == 588
        and provenance.get("train_tokens") == RAW_TOKENS
        and provenance.get("train_max_tokens", 1025) <= 1024,
        f"Arm {arm} provenance differs from prospective contract",
    )
    events = _jsonl(run / "train-metrics.jsonl")
    steps: dict[int, int] = {}
    for event in events:
        if event.get("event") != "train":
            continue
        step, tokens = event.get("step"), event.get("tokens")
        require(
            type(step) is int
            and 1 <= step <= 174
            and type(tokens) is int
            and tokens > 0,
            f"Arm {arm} has malformed optimizer log",
        )
        steps[step] = tokens  # Latest resumed record supersedes an interrupted attempt.
    require(
        set(steps) == set(range(1, 175)) and sum(steps.values()) == RAW_TOKENS,
        f"Arm {arm} optimizer steps or admitted raw tokens differ",
    )
    select_files = {
        tag: run / f"{tag}-predictions.jsonl"
        for tag in ("select-baseline", "select-step-0000174")
    }
    select_rows = {tag: _jsonl(path) for tag, path in select_files.items()}
    for tag, rows in select_rows.items():
        require(len(rows) == 588, f"Arm {arm} {tag} parent SELECT count differs")
        require(
            collections.Counter(row.get("task_type") for row in rows)
            == {"choice": 277, "noul": 271, "score": 40},
            f"Arm {arm} {tag} SELECT type balance differs",
        )
        ids = [row.get("id") for row in rows]
        require(
            len(set(ids)) == 588 and all(isinstance(item, str) for item in ids),
            f"Arm {arm} {tag} SELECT IDs differ",
        )
    require(
        [row.get("id") for row in select_rows["select-baseline"]]
        == [row.get("id") for row in select_rows["select-step-0000174"]]
        and [row.get("prompt_sha256") for row in select_rows["select-baseline"]]
        == [row.get("prompt_sha256") for row in select_rows["select-step-0000174"]]
        and [row.get("token_ids_sha256") for row in select_rows["select-baseline"]]
        == [row.get("token_ids_sha256") for row in select_rows["select-step-0000174"]]
        and [row.get("gold_key") for row in select_rows["select-baseline"]]
        == [row.get("gold_key") for row in select_rows["select-step-0000174"]],
        f"Arm {arm} SELECT baseline/final inputs differ",
    )
    paths = [
        run / filename
        for filename in (
            "COMPLETE.json",
            "LATEST.json",
            "BEST.json",
            "provenance.json",
            "train-metrics.jsonl",
        )
    ] + [checkpoint / "checkpoint.json", *select_files.values()]
    return {
        "checkpoint": checkpoint,
        "provenance": provenance,
        "select": select_rows,
        "artifacts": {str(path.resolve()): file_sha256(path) for path in paths},
    }


def _predictions(
    path: Path, prompts: list[dict[str, Any]], model_identity: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = _json(manifest_path)
    rows = _jsonl(path)
    counts = manifest.get("counts", {})
    adapter_files = manifest.get("adapter_files_sha256")
    require(
        manifest.get("predictions_sha256") == file_sha256(path)
        and manifest.get("input_sha256") == PROMPT_SHA
        and manifest.get("input_items") == 192
        and manifest.get("adapter_version") == ADAPTER_VERSION
        and isinstance(adapter_files, dict)
        and set(adapter_files)
        == {"infer.py", "decision_model.py", "data.py", "lora.py", "source.py"}
        and all(
            isinstance(value, str) and len(value) == 64
            for value in adapter_files.values()
        )
        and manifest.get("model_sha256") == model_identity["model_sha256"]
        and manifest.get("model_files_sha256") == model_identity["files_sha256"]
        and manifest.get("max_length") == 1024
        and manifest.get("temperature") == 1.0
        and "calibration" not in manifest
        and counts.get("items") == counts.get("questions") == 192
        and counts.get("valid_questions", -1) + counts.get("invalid_questions", -1)
        == 192
        and counts.get("truncated_questions") == 0
        and counts.get("over_budget_questions") == 0
        and len(rows) == 192,
        f"Native prediction manifest or cardinality differs: {path.name}",
    )
    require(
        sum(row.get("adapter_status") != "ok" for row in rows)
        == counts["invalid_questions"],
        f"Native prediction invalid count differs: {path.name}",
    )
    for item, record in zip(prompts, rows):
        require(
            record.get("id") == item["id"]
            and record.get("input_sha256") == prompt_input_sha256(item)
            and record.get("source_input_sha256") == prompt_input_sha256(item)
            and record.get("model_sha256") == model_identity["model_sha256"]
            and record.get("adapter_sha256") == manifest.get("adapter_sha256")
            and type(record.get("usage", {}).get("input_tokens")) is int
            and record["usage"]["input_tokens"] > 0
            and record.get("truncated_questions") == 0
            and not any(
                key in record for key in ("label", "target", "gold", "gold_key")
            ),
            f"Native prediction row/input differs: {path.name}",
        )
    return rows, manifest


def build_seal(paths: dict[str, str]) -> dict[str, Any]:
    prepared = Path(paths["prepared_dir"])
    prep = _prepared(prepared)
    prompts_path = prepared / "r2_en.native-prompts.jsonl"
    prompts = _prompt_roster(prompts_path)
    source_path = Path(paths["source_path"])
    source = checkpoint_fingerprint(Path(paths["source_checkpoint"]), source_path)
    require(source["model_sha256"] == SOURCE_SHA, "Original source fingerprint differs")
    runs = {arm: _run(Path(paths[f"arm_{arm.lower()}_run"]), arm) for arm in ("A", "B")}
    require(
        all(
            run["provenance"].get("initial_model_identity") == source
            for run in runs.values()
        ),
        "Arm initial source identities differ from frozen original",
    )
    common_a = runs["A"]["provenance"]["contract"].copy()
    common_b = runs["B"]["provenance"]["contract"].copy()
    for contract in (common_a, common_b):
        contract["data_sha256"] = {**contract["data_sha256"], "train": "ARM"}
        contract["direct_lora_arm"] = "ARM"
    require(
        common_a == common_b
        and runs["A"]["provenance"].get("code_sha256")
        == runs["B"]["provenance"].get("code_sha256")
        and runs["A"]["provenance"].get("initial_model_identity")
        == runs["B"]["provenance"].get("initial_model_identity"),
        "Matched arm contracts or source identities differ",
    )
    identities = {"source": source}
    for arm in ("A", "B"):
        identities[arm] = checkpoint_fingerprint(runs[arm]["checkpoint"], source_path)
    source_prediction = Path(paths["source_prediction"])
    require(
        file_sha256(source_prediction) == SOURCE_PRED_SHA
        and file_sha256(
            source_prediction.with_name(source_prediction.name + ".manifest.json")
        )
        == SOURCE_PRED_MANIFEST_SHA,
        "Prospectively sealed original-source r2 predictions differ",
    )
    predictions = {}
    manifests = {}
    for role, identity in identities.items():
        path = Path(paths[f"{role.lower()}_prediction"])
        rows, manifest = _predictions(path, prompts, identity)
        predictions[role] = rows
        manifests[role] = manifest
    shared = (
        "input_sha256",
        "adapter_sha256",
        "adapter_files_sha256",
        "adapter_version",
        "max_length",
        "temperature",
        "execution",
        "truncation_policy",
    )
    for field in shared:
        require(
            manifests["source"].get(field)
            == manifests["A"].get(field)
            == manifests["B"].get(field),
            f"Native prediction adapter contract differs: {field}",
        )
    for index in range(192):
        require(
            len(
                {
                    predictions[role][index].get("usage", {}).get("input_tokens")
                    for role in identities
                }
            )
            == 1,
            "Native prediction token count differs across models",
        )
    for arm in ("A", "B"):
        baseline = runs[arm]["select"]["select-baseline"]
        require(
            [row.get("id") for row in baseline]
            == [row.get("id") for row in runs["A"]["select"]["select-baseline"]]
            and [row.get("prompt_sha256") for row in baseline]
            == [
                row.get("prompt_sha256")
                for row in runs["A"]["select"]["select-baseline"]
            ],
            "Matched arm parent SELECT inputs differ",
        )
        for tag in ("select-baseline", "select-step-0000174"):
            require(
                [row.get("id") for row in runs[arm]["select"][tag]]
                == [row.get("id") for row in runs["A"]["select"][tag]]
                and [row.get("gold_key") for row in runs[arm]["select"][tag]]
                == [row.get("gold_key") for row in runs["A"]["select"][tag]],
                "Matched arm parent SELECT labels differ",
            )
    artifact_paths = [
        prepared / "manifest.json",
        *(
            prepared / filename
            for filename in (
                "arm_a.train.jsonl",
                "arm_b.train.jsonl",
                "parent_en.select.jsonl",
                "parent_en.cal.jsonl",
                "r2_en.native-prompts.jsonl",
            )
        ),
    ]
    artifact_sha = {str(path.resolve()): file_sha256(path) for path in artifact_paths}
    for arm in ("A", "B"):
        artifact_sha.update(runs[arm]["artifacts"])
    for role in identities:
        path = Path(paths[f"{role.lower()}_prediction"])
        artifact_sha[str(path.resolve())] = file_sha256(path)
        artifact_sha[str(path.with_name(path.name + ".manifest.json").resolve())] = (
            file_sha256(path.with_name(path.name + ".manifest.json"))
        )
    return {
        "schema_version": VERSION,
        "status": "SEALED_GOLD_FREE_PENDING_SINGLE_UNBLIND",
        "scorer_sha256": file_sha256(Path(__file__)),
        "paths": {name: str(Path(value).resolve()) for name, value in paths.items()},
        "artifacts_sha256": dict(sorted(artifact_sha.items())),
        "model_sha256": {
            role: value["model_sha256"] for role, value in identities.items()
        },
        "model_files_sha256": {
            role: value["files_sha256"] for role, value in identities.items()
        },
        "native_adapter_sha256": manifests["source"]["adapter_sha256"],
        "prepared_padding_relative_delta": prep["padding_relative_delta"],
        "r2_prompt_sha256": PROMPT_SHA,
        "r2_rows": 192,
        "source_rows": len(predictions["source"]),
        "arm_rows": {arm: len(predictions[arm]) for arm in ("A", "B")},
        "final_step": 174,
        "r2_key_sha256_after_unblind_only": KEY_SHA,
    }


def _number(value: Any) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def score_answer(record: dict[str, Any] | None, gold: int) -> dict[str, Any]:
    """Native three-level Score; invalid and tied answers are point failures."""
    invalid = {"valid": False, "point": None, "correct": False, "brier": None}
    if not isinstance(record, dict):
        return invalid
    if record.get("adapter_status") != "ok" or record.get("adapter_errors"):
        return invalid
    answer = record.get("answers", {}).get("decision")
    if not isinstance(answer, dict) or answer.get("type") != "score":
        return invalid
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != {"0", "1", "2"}:
        return invalid
    values = {key: _number(raw[key]) for key in ("0", "1", "2")}
    if any(value is None or not 0 <= value <= 1 for value in values.values()):
        return invalid
    total = sum(values.values())
    expected = sum(int(key) * value for key, value in values.items())
    declared = _number(answer.get("score"))
    if total <= 0 or abs(total - 1.0) > 0.02 or declared is None:
        return invalid
    if not 0 <= declared <= 2 or abs(declared - expected) > 0.06:
        return invalid
    maximum = max(values.values())
    winners = [key for key, value in values.items() if abs(value - maximum) <= 1e-8]
    if len(winners) != 1:
        return invalid
    point = int(winners[0])
    normalized = {key: value / total for key, value in values.items()}
    brier = (
        sum((value - float(int(key) == gold)) ** 2 for key, value in normalized.items())
        / 2
    )
    return {"valid": True, "point": point, "correct": point == gold, "brier": brier}


def _key_rows(path: Path, prompts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    require(file_sha256(path) == KEY_SHA, "Frozen private r2 oracle key SHA differs")
    rows = load_partition(path, "select")
    require(len(rows) == 240, "r2 oracle row count differs")
    require(
        collections.Counter(row["language"] for row in rows) == {"en": 192, "zh": 48},
        "r2 oracle language balance differs",
    )
    english = {row["id"]: row for row in rows if row["language"] == "en"}
    require(len(english) == 192, "r2 English oracle has repeated IDs")
    group_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in english.values():
        require(
            row["task_type"] == "score"
            and [option["key"] for option in row["options"]] == ["0", "1", "2"]
            and row.get("audit_metadata", {}).get("operation") in OPS,
            "r2 English oracle schema differs",
        )
        group_rows[row["group_id"]].append(row)
    require(len(group_rows) == 64, "r2 independent group count differs")
    groups_per_op = collections.Counter()
    for members in group_rows.values():
        operation = members[0]["audit_metadata"]["operation"]
        groups_per_op[operation] += 1
        require(
            len(members) == 3
            and {row["label"] for row in members} == {0, 1, 2}
            and all(row["audit_metadata"]["operation"] == operation for row in members),
            "r2 English oracle has an incomplete group",
        )
    require(groups_per_op == dict.fromkeys(OPS, 16), "r2 operation balance differs")
    for prompt in prompts:
        row = english.get(prompt["id"])
        require(row is not None, "Frozen prompt lacks oracle row")
        question = prompt["questions"]["decision"]
        require(
            prompt["state"] == row["state"]
            and question["instructions"] == row["instructions"]
            and question["criteria"]
            == [option["description"] for option in row["options"]],
            "Frozen prompt and oracle input differ",
        )
    return english


def paired_bootstrap(group_delta: dict[str, dict[str, int]]) -> tuple[float, float]:
    """Fixed 10k stratified, paired group resamples; no row independence fiction."""
    require(set(group_delta) == set(OPS), "Bootstrap operation set differs")
    seed = int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED_TEXT.encode()).digest(), "big")
    rng = random.Random(seed)
    ordered = []
    for operation in OPS:
        values = [
            group_delta[operation][group] for group in sorted(group_delta[operation])
        ]
        require(len(values) == 16, "Bootstrap needs exactly 16 groups per operation")
        ordered.append(values)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        difference = sum(
            values[rng.randrange(16)] for values in ordered for _ in range(16)
        )
        draws.append(difference / 192)
    draws.sort()
    return draws[249], draws[9749]


def _select_type(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    subset = [row for row in rows if row["task_type"] == kind]
    correct = invalid = 0
    brier_total = 0.0
    for row in subset:
        answer = row.get("answer")
        gold = row.get("gold_key")
        value = None
        point = None
        if isinstance(answer, dict) and answer.get("type") == kind:
            if kind == "noul":
                p = _number(answer.get("noul"))
                if p is not None and 0 <= p <= 1:
                    point = "true" if p > 0.5 else "false" if p < 0.5 else None
                    if gold in ("true", "false"):
                        value = (p - float(gold == "true")) ** 2
            else:
                probs = answer.get("probabilities")
                if isinstance(probs, dict) and gold in probs and len(probs) >= 2:
                    numeric = {key: _number(p) for key, p in probs.items()}
                    if all(p is not None and 0 <= p <= 1 for p in numeric.values()):
                        total = sum(numeric.values())
                        if total > 0 and abs(total - 1.0) <= 0.02:
                            highest = max(numeric.values())
                            winners = [
                                key
                                for key, p in numeric.items()
                                if abs(p - highest) <= 1e-8
                            ]
                            point = winners[0] if len(winners) == 1 else None
                            if point == answer.get("choice"):
                                normalized = {
                                    key: p / total for key, p in numeric.items()
                                }
                                value = (
                                    sum(
                                        (p - float(key == gold)) ** 2
                                        for key, p in normalized.items()
                                    )
                                    / 2
                                )
        if point is None or value is None:
            invalid += 1
            brier_total += 1.0
        else:
            correct += point == gold
            brier_total += value
            require(
                row.get("prediction_key") == point
                and row.get("correct") == (point == gold)
                and _number(row.get("brier")) is not None
                and abs(row["brier"] - value) <= 1e-5,
                "Parent SELECT native record and recomputed metric disagree",
            )
    return {
        "n": len(subset),
        "correct": correct,
        "accuracy": correct / len(subset),
        "invalid": invalid,
        "normalized_brier": brier_total / len(subset),
    }


def score_sealed(seal: dict[str, Any], key_path: Path) -> dict[str, Any]:
    paths = seal["paths"]
    prepared = Path(paths["prepared_dir"])
    prompts = _prompt_roster(prepared / "r2_en.native-prompts.jsonl")
    key = _key_rows(key_path, prompts)
    models = {}
    per_row = {}
    by_group: dict[str, list[str]] = collections.defaultdict(list)
    for prompt in prompts:
        by_group[key[prompt["id"]]["group_id"]].append(prompt["id"])
    for role in ("source", "A", "B"):
        records = _jsonl(Path(paths[f"{role.lower()}_prediction"]))
        scored = {
            prompt["id"]: score_answer(record, key[prompt["id"]]["label"])
            for prompt, record in zip(prompts, records)
        }
        per_row[role] = scored
        slices = {}
        for operation in OPS:
            ids = [
                item_id
                for item_id in scored
                if key[item_id]["audit_metadata"]["operation"] == operation
            ]
            require(len(ids) == 48, "r2 operation row count differs")
            slices[operation] = {
                "n": 48,
                "correct": sum(scored[item_id]["correct"] for item_id in ids),
                "invalid": sum(not scored[item_id]["valid"] for item_id in ids),
            }
        correct = sum(item["correct"] for item in scored.values())
        valid_brier = [item["brier"] for item in scored.values() if item["valid"]]
        models[role] = {
            "n": 192,
            "correct": correct,
            "accuracy": correct / 192,
            "invalid": sum(not item["valid"] for item in scored.values()),
            "valid_only_normalized_brier": (
                sum(valid_brier) / len(valid_brier) if valid_brier else None
            ),
            "by_operation": slices,
            "all_three_correct_groups": sum(
                all(scored[item_id]["correct"] for item_id in ids)
                for ids in by_group.values()
            ),
            "confusion": {
                str(gold): {
                    str(pred): sum(
                        key[item_id]["label"] == gold
                        and scored[item_id]["point"] == pred
                        for item_id in scored
                    )
                    for pred in (0, 1, 2, None)
                }
                for gold in (0, 1, 2)
            },
        }
    group_delta: dict[str, dict[str, int]] = {operation: {} for operation in OPS}
    for group, ids in by_group.items():
        operation = key[ids[0]]["audit_metadata"]["operation"]
        group_delta[operation][group] = sum(
            per_row["A"][item_id]["correct"] - per_row["B"][item_id]["correct"]
            for item_id in ids
        )
    interval = paired_bootstrap(group_delta)
    op_delta = {
        operation: models["A"]["by_operation"][operation]["correct"]
        - models["B"]["by_operation"][operation]["correct"]
        for operation in OPS
    }
    parent = {}
    for arm in ("A", "B"):
        run = Path(paths[f"arm_{arm.lower()}_run"])
        rows = {
            tag: _jsonl(run / f"{tag}-predictions.jsonl")
            for tag in ("select-baseline", "select-step-0000174")
        }
        parent[arm] = {
            kind: {tag: _select_type(items, kind) for tag, items in rows.items()}
            for kind in ("choice", "noul")
        }
    gates = {
        "source_data_steps_tokens_padding_and_native_outputs": True,
        "a_minus_b_at_least_12_of_192": models["A"]["correct"] - models["B"]["correct"]
        >= 12,
        "paired_group_bootstrap_lower_above_zero": interval[0] > 0,
        "at_least_two_operations_improve": sum(delta > 0 for delta in op_delta.values())
        >= 2,
        "no_operation_loses_more_than_2_of_48": all(
            delta >= -2 for delta in op_delta.values()
        ),
    }
    for kind in ("choice", "noul"):
        baseline = parent["A"][kind]["select-baseline"]
        final = parent["A"][kind]["select-step-0000174"]
        name = f"parent_{kind}"
        gates[f"{name}_accuracy_loss_at_most_0_02"] = (
            final["accuracy"] >= baseline["accuracy"] - 0.02
        )
        gates[f"{name}_brier_worsening_at_most_0_02"] = (
            final["normalized_brier"] <= baseline["normalized_brier"] + 0.02
        )
        gates[f"{name}_invalid_not_increased"] = final["invalid"] <= baseline["invalid"]
    return {
        "schema_version": VERSION,
        "status": (
            "ADVANCE_TO_SINGLE_DEV_CSS_PILOT"
            if all(gates.values())
            else "DO_NOT_ADVANCE"
        ),
        "seal_sha256": seal["seal_sha256"],
        "oracle_key_sha256": KEY_SHA,
        "bootstrap": {
            "unit": "independent three-row group, stratified 16 per operation",
            "replicates": BOOTSTRAP_DRAWS,
            "seed_sha256": hashlib.sha256(BOOTSTRAP_SEED_TEXT.encode()).hexdigest(),
            "a_minus_b_accuracy_interval_95": interval,
        },
        "a_minus_b_correct_count": models["A"]["correct"] - models["B"]["correct"],
        "a_minus_b_correct_by_operation": op_delta,
        "models": models,
        "parent_select": parent,
        "gates": gates,
        "limitations": [
            "English-only SELECT diagnostic, not JevArena FINAL or release evidence.",
            "The source-relative bootstrap describes this one frozen selector.",
            "Chinese Score transfer remains unevaluated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal", help="Gold-free prerequisite and prediction seal")
    for name in (
        "prepared_dir",
        "source_checkpoint",
        "source_path",
        "arm_a_run",
        "arm_b_run",
        "source_prediction",
        "a_prediction",
        "b_prediction",
    ):
        seal.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("score", help="Single key opening after complete seal")
    score.add_argument("--seal", type=Path, required=True)
    score.add_argument("--seal-sha256", required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        paths = {
            name: str(getattr(args, name))
            for name in (
                "prepared_dir",
                "source_checkpoint",
                "source_path",
                "arm_a_run",
                "arm_b_run",
                "source_prediction",
                "a_prediction",
                "b_prediction",
            )
        }
        value = build_seal(paths)
        _private_json(args.output, value)
        print(
            json.dumps(
                {"status": value["status"], "seal_sha256": file_sha256(args.output)}
            )
        )
        return
    require(not args.output.exists(), "Refusing to overwrite scorer result")
    require(
        file_sha256(args.seal) == args.seal_sha256,
        "Blind prediction seal SHA differs",
    )
    frozen = _json(args.seal)
    require(
        frozen.get("schema_version") == VERSION
        and frozen.get("status") == "SEALED_GOLD_FREE_PENDING_SINGLE_UNBLIND"
        and frozen.get("scorer_sha256") == file_sha256(Path(__file__)),
        "Blind scoring seal/schema/code differs",
    )
    current = build_seal(frozen["paths"])
    require(
        current == frozen, "Sealed artifacts or model identities changed before unblind"
    )
    marker = args.seal.with_name(args.seal.name + ".unblinded")
    _private_json(
        marker, {"seal_sha256": args.seal_sha256, "key_expected_sha256": KEY_SHA}
    )
    # First key read occurs only after complete revalidation and one-use marker.
    report = score_sealed({**frozen, "seal_sha256": args.seal_sha256}, args.key)
    require(
        all(
            file_sha256(Path(path)) == expected
            for path, expected in frozen["artifacts_sha256"].items()
        ),
        "A sealed artifact changed during the one-use unblind",
    )
    _private_json(args.output, report)
    print(
        json.dumps(
            {"status": report["status"], "report_sha256": file_sha256(args.output)}
        )
    )


if __name__ == "__main__":
    main()
