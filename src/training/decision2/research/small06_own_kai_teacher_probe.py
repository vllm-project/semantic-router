"""Read-only TRAIN coverage and bounded own-Kai native teacher screen.

No student weights are changed. Private source rows, selected IDs, prompts and
probability vectors are never written by this script; only aggregates are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inference.kai_lex import (
    MODELS,
    is_context_overflow,
    load_system_one,
    runtime_report,
    verify_native_bundle,
)
from research.eikos_teacher_train_pilot import probabilities, write_once
from training.data.audit_small06_train_balance import native_parts
from training.model.data import canonical, file_sha256, load_partition

TRAIN_SHA = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
NATIVE_MANIFEST_SHA = "a2591978481e019f482686df69d55f213a970f51ce465c6a45e6910bc9d3dca5"
NATIVE_TRAIN_SHA = "fc8496c76c84b37f0a9308b13a270b69aab608defb8a9e4a3f62011352b6976c"
QUARANTINE_SHA = "1bd6a566eaac72f374a5e391ee6613fed4d667870d9c79e2fac8bc152789379d"
TOKENIZER_SHA = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
KINDS = ("choice", "noul", "score")
SEED = "decision2-own-kai06-train-screen-v1"
QUOTAS = {
    ("choice", "stage4"): 16,
    ("choice", "other"): 16,
    ("noul", "stage4"): 16,
    ("noul", "other"): 16,
    ("score", "stage4"): 16,
    ("score", "other_three"): 8,
    ("score", "other_non_three"): 8,
}


def bucket(row: dict[str, Any]) -> str:
    if row["source"] == "legacy:stage4-general-composition-v2":
        return "stage4"
    if row["task_type"] != "score":
        return "other"
    return "other_three" if len(row["options"]) == 3 else "other_non_three"


def _sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Pinned input bytes differ: {path.name}")


def load_train(
    train: Path, native_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify the prior Kai conversion against the exact Qwen TRAIN bytes."""
    _sha(train, TRAIN_SHA)
    _sha(native_dir / "manifest.json", NATIVE_MANIFEST_SHA)
    _sha(native_dir / "train.jsonl", NATIVE_TRAIN_SHA)
    _sha(native_dir / "quarantine.jsonl", QUARANTINE_SHA)
    manifest = json.loads((native_dir / "manifest.json").read_text())
    if (
        manifest.get("base_model", {}).get("model_revision")
        != MODELS["kai"]["revision"]
        or manifest.get("base_model", {}).get("model_config_sha256")
        != MODELS["kai"]["manifest_sha256"]
        or manifest.get("parent_split_sha256", {}).get("train") != TRAIN_SHA
        or manifest.get("audit", {}).get("train", {}).get("native_rows") != 6262
    ):
        raise ValueError("Kai native conversion is not the pinned own-Kai TRAIN")
    rows = load_partition(train, "train")
    excluded = [json.loads(line) for line in (native_dir / "quarantine.jsonl").open()]
    if any(
        item.get("role") != "train" or item.get("reason") != "over_1024_whole_group"
        for item in excluded
    ):
        raise ValueError("Unexpected Kai quarantine role or reason")
    excluded_ids = {item["source_row_id"] for item in excluded}
    if len(rows) != 7455 or len(excluded_ids) != 1193:
        raise ValueError("Frozen TRAIN or whole-group quarantine count changed")
    eligible = [row for row in rows if row["id"] not in excluded_ids]
    native_ids = [
        json.loads(line)["id"] for line in (native_dir / "train.jsonl").open()
    ]
    # The published converter assigns new native row IDs. Its exact manifest
    # attests the conversion; source IDs are retained in the original split.
    if len(eligible) != 6262 or len(native_ids) != 6262 or len(set(native_ids)) != 6262:
        raise ValueError("Eligible original/native TRAIN counts differ")
    return eligible, excluded


def roster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prospectively choose 96 independent source groups without reading labels."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (kind, part), count in QUOTAS.items():
        candidates = sorted(
            (row for row in rows if row["task_type"] == kind and bucket(row) == part),
            key=lambda row: hashlib.sha256(
                f"{SEED}\x00{kind}\x00{part}\x00{row['group_id']}\x00{row['id']}".encode()
            ).digest(),
        )
        selected = 0
        for row in candidates:
            if row["group_id"] in seen:
                continue
            result.append(row)
            seen.add(row["group_id"])
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError(
                f"Too few independent Kai-compatible TRAIN groups: {kind}/{part}"
            )
    return result


def roster_sha(rows: list[dict[str, Any]]) -> str:
    payload = [{"id": row["id"], "input_sha256": row["input_sha256"]} for row in rows]
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def question(row: dict[str, Any]) -> dict[str, Any]:
    """Use the exact native Choice/Noul/Score request semantics of Kai conversion."""
    options = row["options"]
    result: dict[str, Any] = {
        "type": row["task_type"],
        "instructions": row["instructions"],
    }
    if row["task_type"] == "choice":
        result["criteria"] = {item["key"]: item["description"] for item in options}
    elif row["task_type"] == "noul":
        keyed = {item["key"]: item["description"] for item in options}
        if set(keyed) != {"false", "true"}:
            raise ValueError("Native Noul requires false/true option keys")
        criteria = {key: value for key, value in keyed.items() if value is not None}
        if criteria:
            result["criteria"] = criteria
    elif row["task_type"] == "score":
        if [item["key"] for item in options] != [str(i) for i in range(len(options))]:
            raise ValueError("Native Score options must be ordered integer levels")
        result["criteria"] = [
            item["description"] if item["description"] is not None else item["key"]
            for item in options
        ]
    else:
        raise ValueError("Unexpected task type")
    return result


def aggregate(
    rows: list[dict[str, Any]], decide: Any
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "rows": 0,
            "valid": 0,
            "overflow": 0,
            "ties": 0,
            "correct": 0,
            "gold_probability_sum": 0.0,
            "brier_sum": 0.0,
            "top_probability_sum": 0.0,
        }
    )
    for row in rows:
        category = f"{row['task_type']}/{bucket(row)}"
        for key in (row["task_type"], category):
            stats[key]["rows"] += 1
        try:
            response = decide(row["state"], question(row))
        except ValueError as error:
            if not is_context_overflow(error):
                raise
            for key in (row["task_type"], category):
                stats[key]["overflow"] += 1
            continue
        if response.get("model") != MODELS["kai"]["model_name"] or set(
            response.get("answers", {})
        ) != {"decision"}:
            raise ValueError("Kai returned a different native response identity")
        probs = probabilities(row, response["answers"]["decision"])
        gold = row["options"][row["label"]]["key"]
        top = max(probs.values())
        ties = sum(abs(value - top) <= 1e-8 for value in probs.values()) != 1
        correct = not ties and abs(probs[gold] - top) <= 1e-8
        brier = sum((value - int(key == gold)) ** 2 for key, value in probs.items()) / 2
        for key in (row["task_type"], category):
            item = stats[key]
            item["valid"] += 1
            item["ties"] += int(ties)
            item["correct"] += int(correct)
            item["gold_probability_sum"] += probs[gold]
            item["brier_sum"] += brier
            item["top_probability_sum"] += top
    return dict(sorted(stats.items()))


def inventory(
    train: Path,
    rows: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    tokenizer_dir: Path,
) -> dict[str, Any]:
    _sha(tokenizer_dir / "tokenizer.json", TOKENIZER_SHA)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, use_fast=True
    )
    excluded_ids = {item["source_row_id"] for item in excluded}
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in load_partition(train, "train"):
        role = "excluded" if row["id"] in excluded_ids else "eligible"
        tokens = sum(
            len(tokenizer.encode(part, add_special_tokens=False))
            for part in native_parts(row)
        )
        buckets[role]["rows"] += 1
        buckets[role]["tokens"] += tokens
        buckets[role][f"type_rows/{row['task_type']}"] += 1
        buckets[role][f"type_tokens/{row['task_type']}"] += tokens
        buckets[role][f"source_rows/{row['source']}"] += 1
        buckets[role][f"source_tokens/{row['source']}"] += tokens
        buckets[role][f"source_type_rows/{row['source']}/{row['task_type']}"] += 1
        buckets[role][
            f"source_type_tokens/{row['source']}/{row['task_type']}"
        ] += tokens
    if (
        buckets["eligible"]["rows"] != len(rows)
        or buckets["excluded"]["rows"] != len(excluded)
        or sum(part["tokens"] for part in buckets.values()) != 4_094_489
    ):
        raise ValueError("Qwen native token exposure differs from frozen audit")
    return {
        "eligible": {key: value for key, value in sorted(buckets["eligible"].items())},
        "excluded": {key: value for key, value in sorted(buckets["excluded"].items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("inventory", "screen"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows, excluded = load_train(args.train, args.native_dir)
    selected = roster(rows)
    roster_id = roster_sha(selected)
    if args.mode == "inventory":
        if args.tokenizer_dir is None:
            raise ValueError("Inventory requires pinned official-Qwen tokenizer")
        print(
            json.dumps(
                {
                    "schema": "decision2-own-kai-train-feasibility/1",
                    "train_sha256": TRAIN_SHA,
                    "native_manifest_sha256": NATIVE_MANIFEST_SHA,
                    "kai_revision": MODELS["kai"]["revision"],
                    "eligible_rows": len(rows),
                    "quarantined_rows": len(excluded),
                    "eligible_by_type": dict(
                        sorted(Counter(row["task_type"] for row in rows).items())
                    ),
                    "excluded_by_type": dict(
                        sorted(Counter(item["task_type"] for item in excluded).items())
                    ),
                    "eligible_by_source_type": dict(
                        sorted(
                            Counter(
                                f"{row['source']}/{row['task_type']}" for row in rows
                            ).items()
                        )
                    ),
                    "score_levels": dict(
                        sorted(
                            Counter(
                                len(row["options"])
                                for row in rows
                                if row["task_type"] == "score"
                            ).items()
                        )
                    ),
                    "screen_roster_sha256": roster_id,
                    "screen_by_type_bucket": dict(
                        sorted(
                            Counter(
                                f"{row['task_type']}/{bucket(row)}" for row in selected
                            ).items()
                        )
                    ),
                    "native_qwen_exposure": inventory(
                        args.train, rows, excluded, args.tokenizer_dir
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if (
        args.model_path is None
        or args.roster_sha256 != roster_id
        or args.output is None
    ):
        raise ValueError(
            "Screen needs pinned Kai release, roster hash, and private output"
        )
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Screen output must be new within a private directory")
    identity = verify_native_bundle(args.model_path, "kai", MODELS["kai"]["revision"])
    runtime = runtime_report("cuda:0", check_gpu=True)
    if not runtime["runtime_matches_validated"]:
        raise ValueError("Native Kai runtime differs from its qualified profile")
    _, client = load_system_one(args.model_path, "kai", "cuda:0")
    results = aggregate(
        selected,
        lambda state, item: client.system_one(
            state=state, questions={"decision": item}
        ),
    )
    payload = {
        "schema": "decision2-own-kai-train-screen/1",
        "source": f"{MODELS['kai']['model_id']}@{MODELS['kai']['revision']}",
        "model_config_sha256": identity["model_config_sha256"],
        "train_sha256": TRAIN_SHA,
        "native_manifest_sha256": NATIVE_MANIFEST_SHA,
        "roster_sha256": roster_id,
        "script_sha256": file_sha256(__file__),
        "runtime": runtime,
        "sampled_rows": len(selected),
        "sampled_groups": len({row["group_id"] for row in selected}),
        "by_type_and_bucket": results,
    }
    write_once(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
