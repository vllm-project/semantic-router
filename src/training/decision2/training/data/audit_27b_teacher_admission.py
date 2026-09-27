"""CPU-only, private admission gate for one matched 27B teacher contrast.

No model weights, optimizer, benchmark answer key, or public artifact is read.
The receipt contains only counts and digests; TRAIN text stays in private inputs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from training.data.audit_release4b_overlap import audit_panel
from training.model.data import digest, file_sha256, load_partition

SCHEMA = "decision2-27b-teacher-admission/1"
TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
SELECT_SHA256 = "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6"
CAL_SHA256 = "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"
RIGHTS_SHA256 = "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8"
CHOICE_TEACHER_SHA256 = (
    "8cf211e5af88920de556dc84aa0fcb8b16677fdfd5209abb04db0ffe8e8b195c"
)
SCORE_TEACHER_SHA256 = (
    "072cd519657caaa883eea1f5077789e5bacbf85f8ee20ab44cd562acc317701b"
)
TEACHER_ID = "denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015"
TEACHER_SOURCE_REVISION = "ee63c1515980491a742f0bd0685c8dc5ca1f00c3"
TEACHER_MODEL_SHA256 = (
    "d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2"
)
CHOICE_ROSTER_SHA256 = (
    "21f7b73c20107d3aa00e525d0f670963f8e8347d3ef7ca9916854a8400268d75"
)
SCORE_ROSTER_SHA256 = "a2d759bc47b0a8cc116c47d99762f73a2fde175f58bc551c4a0c58b397828d3b"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SCHEDULE_NAMESPACE = "decision2-27b-train-teacher-residual-v1"
HUMAN_SOURCES = {
    "google_goemotions_official_train",
    "legacy:cosmos_qa",
    "legacy:squad2_answerability",
    "legacy:snli",
    "css_flute_official_train",
}
REQUIRED_ROLES = {
    "typed_dev",
    "css_pilot",
    "typed_final_goldfree",
    "css15_goldfree",
    "jevbench_public231",
}
PROMPT_FIELDS = {
    "id",
    "group_id",
    "review_id",
    "family",
    "operation",
    "language",
    "state",
    "questions",
    "instructions",
    "options",
}
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)
GOLD_FIELDS = {
    "answer",
    "answers",
    "correct",
    "correct_answer",
    "gold",
    "label",
    "target",
}


class AdmissionHold(ValueError):
    """Expected fail-closed data or source gate."""


def _check(condition: bool, code: str) -> None:
    if not condition:
        raise AdmissionHold(code)


def _key(value: str) -> str:
    return hashlib.sha256(f"{SCHEDULE_NAMESPACE}/{value}".encode()).hexdigest()


def verify_rights(
    manifest: dict[str, Any], train: list[dict[str, Any]]
) -> dict[str, Any]:
    source_counts = dict(
        sorted(collections.Counter(row["source"] for row in train).items())
    )
    entries = manifest.get("source_rights", [])
    _check(
        manifest.get("schema_version") == "decision2-rights-clean-splits/1",
        "RIGHTS_SCHEMA",
    )
    _check(
        manifest.get("counts", {}).get("source") == source_counts,
        "RIGHTS_SOURCE_COUNTS",
    )
    _check(isinstance(entries, list) and bool(entries), "RIGHTS_SOURCE_LEDGER")
    scoped = [
        entry for entry in entries if entry.get("partition_scope", "TRAIN") == "TRAIN"
    ]
    _check(
        sum(entry.get("rows", -1) for entry in scoped) == len(train), "RIGHTS_ROW_TOTAL"
    )
    _check(
        all(
            isinstance(entry.get("source"), str)
            and isinstance(entry.get("license"), str)
            and isinstance(entry.get("evidence"), str)
            and entry["source"]
            and entry["license"]
            and entry["evidence"]
            for entry in scoped
        ),
        "RIGHTS_MISSING_TERMS",
    )
    _check(manifest.get("publication_eligible") is True, "RIGHTS_MANIFEST_HOLD")
    return {
        "train_rows": len(train),
        "source_count": len(source_counts),
        "ledger_entry_count": len(scoped),
        "status": "PASS_DECLARED_LEDGER_INTEGRITY",
        "limitation": "Manifest integrity is not an independent legal determination or permission to redistribute source text.",
    }


def verify_teacher(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    task: str,
) -> dict[str, dict[str, float]]:
    expected_schema = (
        "decision2-autojev-score-teacher-distributions/1"
        if task == "score"
        else "decision2-autojev-choice-noul-train-distributions/1"
    )
    expected_roster = SCORE_ROSTER_SHA256 if task == "score" else CHOICE_ROSTER_SHA256
    _check(artifact.get("schema") == expected_schema, "TEACHER_SCHEMA")
    _check(artifact.get("source") == TEACHER_ID, "TEACHER_REVISION")
    _check(
        artifact.get("source_revision") == TEACHER_SOURCE_REVISION, "TEACHER_RUNTIME"
    )
    _check(artifact.get("native_model_sha256") == TEACHER_MODEL_SHA256, "TEACHER_MODEL")
    _check(artifact.get("train_sha256") == TRAIN_SHA256, "TEACHER_TRAIN")
    _check(artifact.get("roster_sha256") == expected_roster, "TEACHER_ROSTER")
    if task != "score":
        _check(
            artifact.get("rights_manifest_sha256") == RIGHTS_SHA256, "TEACHER_RIGHTS"
        )
    entries = artifact.get("rows")
    _check(isinstance(entries, list) and len(entries) == len(rows), "TEACHER_ROW_COUNT")
    vectors: dict[str, dict[str, float]] = {}
    for row, entry in zip(rows, entries, strict=True):
        _check(isinstance(entry, dict), "TEACHER_ROW_SCHEMA")
        expected = {
            "id",
            "input_sha256",
            "group_id",
            "source",
            "family",
            "probabilities",
        }
        expected |= (
            {"level_count"}
            if task == "score"
            else {"task_type", "language", "option_count"}
        )
        _check(set(entry) == expected, "TEACHER_ROW_SCHEMA")
        for field in ("id", "input_sha256", "group_id", "source", "family"):
            _check(entry[field] == row[field], "TEACHER_ROW_IDENTITY")
        if task == "score":
            _check(entry["level_count"] == len(row["options"]), "TEACHER_LEVELS")
        else:
            for field in ("task_type", "language"):
                _check(entry[field] == row[field], "TEACHER_ROW_IDENTITY")
            _check(entry["option_count"] == len(row["options"]), "TEACHER_OPTIONS")
        keys = {option["key"] for option in row["options"]}
        probabilities = entry["probabilities"]
        _check(
            isinstance(probabilities, dict) and set(probabilities) == keys,
            "TEACHER_OPTION_KEYS",
        )
        _check(
            all(
                type(value) in (float, int) and math.isfinite(value) and value >= 0
                for value in probabilities.values()
            )
            and abs(sum(probabilities.values()) - 1) <= 1e-5,
            "TEACHER_PROBABILITY",
        )
        vectors[row["id"]] = probabilities
    _check(len(vectors) == len(rows), "TEACHER_DUPLICATE_ID")
    return vectors


def largest_remainder(counts: dict[str, int], target: int) -> dict[str, int]:
    _check(target >= 0 and sum(counts.values()) >= target, "QUOTA_CAPACITY")
    if target == 0:
        return dict.fromkeys(counts, 0)
    total = sum(counts.values())
    base = {source: target * count // total for source, count in counts.items()}
    remainder = target - sum(base.values())
    order = sorted(
        counts, key=lambda source: (-(target * counts[source] % total), source)
    )
    for source in order[:remainder]:
        base[source] += 1
    return base


def choose_whole_groups(
    groups: list[list[dict[str, Any]]], target: int
) -> list[dict[str, Any]]:
    """Find an exact row quota while preserving deterministic group preference.

    A greedy prefix can falsely fail on a later two-row group and an earlier
    one-row group. Reachable sums are never replaced, so the hash-sorted group
    order supplies the same deterministic tie break on every run.
    """
    previous: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(groups):
        size = len(group)
        for subtotal in sorted(previous, reverse=True):
            candidate = subtotal + size
            if candidate <= target and candidate not in previous:
                previous[candidate] = (subtotal, index)
        if target in previous:
            break
    _check(target in previous, "WHOLE_GROUP_QUOTA")
    chosen: list[dict[str, Any]] = []
    subtotal = target
    while subtotal:
        step = previous[subtotal]
        assert step is not None
        subtotal, index = step
        chosen.extend(groups[index])
    return chosen


def select_schedule(
    rows: list[dict[str, Any]],
    token_lengths: dict[str, int],
    *,
    total: int = 2560,
    max_length: int = 4096,
    min_score: int = 460,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    admitted = [row for row in rows if token_lengths[row["id"]] <= max_length]
    score = [row for row in admitted if row["task_type"] == "score"]
    _check(len(score) >= min_score, "SCORE_ADMITTED_COUNT")
    remaining = total - len(score)
    _check(remaining >= 0, "SCHEDULE_TOTAL")
    targets = {"choice": remaining // 2, "noul": remaining - remaining // 2}
    selected = list(score)
    source_quota: dict[str, dict[str, int]] = {}
    for task in ("choice", "noul"):
        pool = [row for row in admitted if row["task_type"] == task]
        source_counts = dict(collections.Counter(row["source"] for row in pool))
        quota = largest_remainder(source_counts, targets[task])
        source_quota[task] = dict(sorted(quota.items()))
        groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(
            list
        )
        for row in pool:
            groups[(row["source"], row["group_id"])].append(row)
        for source in sorted(quota):
            groups_here = sorted(
                (group for (name, _), group in groups.items() if name == source),
                key=lambda group: _key(f"{task}/{source}/{group[0]['group_id']}"),
            )
            selected.extend(choose_whole_groups(groups_here, quota[source]))
    _check(len(selected) == total, "SCHEDULE_TOTAL")
    selected.sort(key=lambda row: _key(f"row/{row['id']}/{row['input_sha256']}"))
    _check(len({row["id"] for row in selected}) == total, "SCHEDULE_DUPLICATE_ID")
    return selected, {
        "rows": total,
        "updates": total // 16,
        "by_type": dict(
            sorted(collections.Counter(row["task_type"] for row in selected).items())
        ),
        "by_language": dict(
            sorted(collections.Counter(row["language"] for row in selected).items())
        ),
        "source_quota": source_quota,
        "source_counts": dict(
            sorted(collections.Counter(row["source"] for row in selected).items())
        ),
        "raw_token_exposure": sum(token_lengths[row["id"]] for row in selected),
        "schedule_sha256": digest(
            [
                {
                    "id": row["id"],
                    "input_sha256": row["input_sha256"],
                    "tokens": token_lengths[row["id"]],
                }
                for row in selected
            ]
        ),
    }


def teacher_mask(
    schedule: list[dict[str, Any]],
    vectors: dict[str, dict[str, float]],
    *,
    minimum_score: int = 160,
    minimum_three_level: int = 40,
    minimum_human: int = 400,
) -> dict[str, Any]:
    selected = []
    by_type: collections.Counter[str] = collections.Counter()
    by_level: collections.Counter[str] = collections.Counter()
    for row in schedule:
        probabilities = vectors[row["id"]]
        gold = row["options"][row["label"]]["key"]
        maximum = max(probabilities.values())
        winners = [
            key for key, value in probabilities.items() if abs(value - maximum) <= 1e-8
        ]
        is_human = row["source"] in HUMAN_SOURCES
        minimum = 0.50 if row["task_type"] == "score" else 0.70
        eligible = (
            len(winners) == 1
            and winners[0] == gold
            and probabilities[gold] >= minimum
            and (row["task_type"] == "score" or is_human)
        )
        if eligible:
            by_type[row["task_type"]] += 1
            if row["task_type"] == "score":
                by_level[str(len(row["options"]))] += 1
            selected.append({"id": row["id"], "input_sha256": row["input_sha256"]})
    _check(by_type["score"] >= minimum_score, "MASK_SCORE_MINIMUM")
    _check(by_level["3"] >= minimum_three_level, "MASK_THREE_LEVEL_MINIMUM")
    _check(by_type["choice"] + by_type["noul"] >= minimum_human, "MASK_HUMAN_MINIMUM")
    return {
        "by_type": dict(sorted(by_type.items())),
        "score_by_level_count": dict(sorted(by_level.items())),
        "masked_rows": len(selected),
        "mask_sha256": digest(selected),
    }


def read_prompt_inventory(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    _check(isinstance(entries, list) and bool(entries), "PROTECTED_INVENTORY")
    roles: dict[str, list[dict[str, Any]]] = {}
    role_hashes = {}
    for entry in entries:
        _check(
            isinstance(entry, dict) and {"role", "path", "sha256"} <= set(entry),
            "PROTECTED_ENTRY",
        )
        role = entry["role"]
        _check(isinstance(role, str) and role not in roles, "PROTECTED_ROLE")
        prompt_path = Path(entry["path"])
        _check(
            prompt_path.is_file() and file_sha256(prompt_path) == entry["sha256"],
            "PROTECTED_HASH",
        )
        prompt_rows = []
        seen_ids = set()
        with prompt_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                _check(
                    isinstance(row, dict) and set(row) <= PROMPT_FIELDS,
                    "PROTECTED_GOLD_FIELD",
                )
                _check(not _contains_gold_key(row), "PROTECTED_NESTED_GOLD_FIELD")
                identifier = row.get("id", row.get("review_id"))
                _check(isinstance(identifier, str) and bool(identifier), "PROTECTED_ID")
                _check(identifier not in seen_ids, "PROTECTED_DUPLICATE_ID")
                seen_ids.add(identifier)
                if "state" in row:
                    prompt_rows.append({"id": identifier, "state": row["state"]})
        roles[role] = prompt_rows
        role_hashes[role] = entry["sha256"]
    _check(set(roles) >= REQUIRED_ROLES, "PROTECTED_REQUIRED_ROLE")
    return roles, {
        "inventory_sha256": file_sha256(path),
        "role_file_sha256": dict(sorted(role_hashes.items())),
        "role_count": len(roles),
    }


def _contains_gold_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(GOLD_FIELDS & set(value)) or any(
            _contains_gold_key(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_gold_key(child) for child in value)
    return False


def audit_overlap(
    schedule: list[dict[str, Any]], roles: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    findings = {}
    blocked = False
    for role, prompts in sorted(roles.items()):
        if role == "rights_clean_train":
            continue  # The schedule is intentionally a subset of this TRAIN.
        if not prompts:
            continue
        panel = audit_panel(schedule, prompts)
        counts = panel["counts"]
        findings[role] = counts
        blocked |= any(counts.values())
    return {
        "roles_scanned": len(findings),
        "by_role": findings,
        "status": (
            "HOLD_CANDIDATE_OVERLAP" if blocked else "PASS_BOUNDED_OBSERVABLE_SCREEN"
        ),
        "limitation": "State-level exact/SimHash near scans can miss paraphrase, partial evidence, task-family reuse and base/teacher pretraining exposure; semantic review is separate.",
    }


def tokenizer_inventory(path: Path) -> dict[str, str]:
    files = {
        name: file_sha256(path / name)
        for name in TOKENIZER_FILES
        if (path / name).is_file()
    }
    _check("tokenizer_config.json" in files, "TOKENIZER_CONFIG")
    _check(
        "tokenizer.json" in files or {"merges.txt", "vocab.json"} <= set(files),
        "TOKENIZER_FILES",
    )
    return dict(sorted(files.items()))


def audit(
    *,
    train_path: Path,
    select_path: Path,
    cal_path: Path,
    rights_path: Path,
    choice_teacher_path: Path,
    score_teacher_path: Path,
    tokenizer_path: Path,
    protected_path: Path,
    encode_length: Callable[[dict[str, Any]], int],
) -> dict[str, Any]:
    inputs = {
        "train": (train_path, TRAIN_SHA256),
        "select": (select_path, SELECT_SHA256),
        "cal": (cal_path, CAL_SHA256),
        "rights_manifest": (rights_path, RIGHTS_SHA256),
        "teacher_choice_noul": (choice_teacher_path, CHOICE_TEACHER_SHA256),
        "teacher_score": (score_teacher_path, SCORE_TEACHER_SHA256),
    }
    for name, (path, expected) in inputs.items():
        _check(
            path.is_file() and file_sha256(path) == expected,
            f"FROZEN_INPUT_HASH_{name.upper()}",
        )
    train = load_partition(train_path, "train")
    _check(len(train) == 7455, "TRAIN_ROW_COUNT")
    select = load_partition(select_path, "select")
    cal = load_partition(cal_path, "cal")
    _check(len(select) == len(cal) == 700, "MONITOR_ROW_COUNT")
    rights = verify_rights(json.loads(rights_path.read_text(encoding="utf-8")), train)
    crows = [row for row in train if row["task_type"] != "score"]
    srows = [row for row in train if row["task_type"] == "score"]
    _check(len(crows) == 6939 and len(srows) == 516, "TRAIN_TYPE_COUNT")
    cvec = verify_teacher(
        json.loads(choice_teacher_path.read_text(encoding="utf-8")),
        crows,
        task="choice_noul",
    )
    svec = verify_teacher(
        json.loads(score_teacher_path.read_text(encoding="utf-8")), srows, task="score"
    )
    vectors = cvec | svec
    _check(len(vectors) == len(train), "TEACHER_TOTAL")
    lengths = {row["id"]: encode_length(row) for row in train}
    _check(
        len(lengths) == len(train) and all(value > 0 for value in lengths.values()),
        "TOKEN_LENGTH",
    )
    schedule, profile = select_schedule(train, lengths)
    mask = teacher_mask(schedule, vectors)
    roles, inventory = read_prompt_inventory(protected_path)
    # Monitor labels are parsed by the strict partition loader but never used.
    roles["rights_clean_select"] = [
        {"id": row["id"], "state": row["state"]} for row in select
    ]
    roles["rights_clean_cal"] = [
        {"id": row["id"], "state": row["state"]} for row in cal
    ]
    overlap = audit_overlap(schedule, roles)
    status = (
        "PASS_CPU_OBSERVABLE"
        if overlap["status"].startswith("PASS")
        else "HOLD_OVERLAP"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "source_revision": SOURCE_REVISION,
        "input_sha256": {name: expected for name, (_, expected) in inputs.items()},
        "tokenizer_files_sha256": tokenizer_inventory(tokenizer_path),
        "teacher_vectors": {"choice_noul": len(cvec), "score": len(svec)},
        "rights": rights,
        "schedule": profile,
        "mask": mask,
        "protected_inventory": inventory,
        "overlap": overlap,
        "gpu_hours": 0,
        "limits": "CPU data admission only. No model/optimizer, zero-step, gradient, SELECT, CAL or formal/public inference performed. A PASS does not resolve semantic/source rights review or authorize training.",
    }


def write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train",
        "select",
        "cal",
        "rights_manifest",
        "teacher_choice_noul",
        "teacher_score",
        "tokenizer_dir",
        "protected_inventory",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    try:
        required = {
            "train": args.train,
            "select": args.select,
            "cal": args.cal,
            "rights_manifest": args.rights_manifest,
            "teacher_choice_noul": args.teacher_choice_noul,
            "teacher_score": args.teacher_score,
            "protected_inventory": args.protected_inventory,
        }
        for name, path in required.items():
            _check(path.is_file(), f"MISSING_{name.upper()}")
        _check(args.tokenizer_dir.is_dir(), "MISSING_TOKENIZER")
        from transformers import AutoTokenizer

        from training.model.decision_model import encode

        tokenizer_files = tokenizer_inventory(args.tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir, local_files_only=True
        )
        result = audit(
            train_path=args.train,
            select_path=args.select,
            cal_path=args.cal,
            rights_path=args.rights_manifest,
            choice_teacher_path=args.teacher_choice_noul,
            score_teacher_path=args.teacher_score,
            tokenizer_path=args.tokenizer_dir,
            protected_path=args.protected_inventory,
            encode_length=lambda row: len(encode(row, tokenizer, 1_000_000)["ids"]),
        )
        _check(result["tokenizer_files_sha256"] == tokenizer_files, "TOKENIZER_CHANGED")
    except AdmissionHold as error:
        result = {
            "schema": SCHEMA,
            "status": "HOLD",
            "reason_code": str(error),
            "gpu_hours": 0,
        }
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        result = {
            "schema": SCHEMA,
            "status": "HOLD",
            "reason_code": f"PRIVATE_INPUT_OR_RUNTIME_{type(error).__name__.upper()}",
            "gpu_hours": 0,
        }
    write_once(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "reason_code": result.get("reason_code"),
                "receipt_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
