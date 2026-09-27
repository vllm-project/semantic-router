"""Attach one pinned external decision teacher to existing TRAIN rows only.

The student always starts from an official Qwen source. Teacher probabilities
are optional, row-local loss targets; they never initialize model weights.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from typing import Any

from .data import digest, file_sha256

ARTIFACT_SCHEMA = "decision2-autojev-choice-noul-train-distributions/1"
ARTIFACT_SHA256 = "8cf211e5af88920de556dc84aa0fcb8b16677fdfd5209abb04db0ffe8e8b195c"
TEACHER_SOURCE = "denis-pplx/autojev-27b@6f5b557e037f5edb25c7dc92dbc6553e5a19c015"
TEACHER_SOURCE_REVISION = "ee63c1515980491a742f0bd0685c8dc5ca1f00c3"
TEACHER_MODEL_SHA256 = (
    "d0b1e161c17d60889744b6ccb5fa588bf80f9856f8535e6a04e083ffcf667ca2"
)
TEACHER_SOURCE_SHA256 = (
    "550ccd857350c6771a1de03e4bcba9fb4412247b58bcdc9e8ac03de2ab9641a5"
)
TEACHER_CONFIG_SHA256 = (
    "bacbcbb281a53af5ef5cc6c9028601097d155bf981129f18a727219517921dcd"
)
TEACHER_SCRIPT_SHA256 = (
    "4909ed107fa4dcffaaef0ba2110a73af7eed698a4f72d179c2ab855867109b3f"
)
TEACHER_ROSTER_SHA256 = (
    "21f7b73c20107d3aa00e525d0f670963f8e8347d3ef7ca9916854a8400268d75"
)
RIGHTS_MANIFEST_SHA256 = (
    "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8"
)
TRAIN_SHA256 = "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755"
ELIGIBLE_ROSTER_SHA256 = (
    "0e7aeee17f2fd78acbbfd710921fb98cebb992a8558e49a32485065eeff842c5"
)
EXCLUDED_SOURCE = "legacy:stage4-general-composition-v2"
MIN_GOLD_PROBABILITY = 0.7
EXPECTED_ELIGIBLE = {"choice": 1841, "noul": 1849}
EXPECTED_TOTAL_TRAIN = 7455
EXPECTED_TYPED_TRAIN = 6939
STUDENT_SOURCE_FILES = {
    "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def is_eligible(row: dict[str, Any], probabilities: dict[str, float]) -> bool:
    """Prospective gold-aware TRAIN-only mask; never uses SELECT or eval labels."""
    if row["task_type"] not in ("choice", "noul") or row["source"] == EXCLUDED_SOURCE:
        return False
    gold = row["options"][row["label"]]["key"]
    maximum = max(probabilities.values())
    winners = [
        key for key, value in probabilities.items() if abs(value - maximum) <= 1e-8
    ]
    return (
        len(winners) == 1
        and winners[0] == gold
        and probabilities[gold] >= MIN_GOLD_PROBABILITY
    )


def attach_external_teacher(
    path: str | Path,
    train_rows: list[dict[str, Any]],
    *,
    train_sha256: str,
    source_files_sha256: dict[str, str],
) -> dict[str, Any]:
    """Validate the entire immutable source and mask before mutating TRAIN rows."""
    path = Path(path)
    if (
        train_sha256 != TRAIN_SHA256
        or source_files_sha256 != STUDENT_SOURCE_FILES
        or file_sha256(path) != ARTIFACT_SHA256
    ):
        raise ValueError(
            "Frozen TRAIN, official student source or teacher bytes differ"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("source") != TEACHER_SOURCE
        or artifact.get("source_revision") != TEACHER_SOURCE_REVISION
        or artifact.get("native_model_sha256") != TEACHER_MODEL_SHA256
        or artifact.get("runtime_source_sha256") != TEACHER_SOURCE_SHA256
        or artifact.get("model_config_sha256") != TEACHER_CONFIG_SHA256
        or artifact.get("script_sha256") != TEACHER_SCRIPT_SHA256
        or artifact.get("loaded_parameters") != 26_086_635_760
        or artifact.get("rights_manifest_sha256") != RIGHTS_MANIFEST_SHA256
        or artifact.get("train_sha256") != TRAIN_SHA256
        or artifact.get("roster_sha256") != TEACHER_ROSTER_SHA256
    ):
        raise ValueError("External teacher source identity differs")
    typed_rows = [row for row in train_rows if row["task_type"] in ("choice", "noul")]
    entries = artifact.get("rows")
    if (
        len(train_rows) != EXPECTED_TOTAL_TRAIN
        or len(typed_rows) != EXPECTED_TYPED_TRAIN
        or not isinstance(entries, list)
        or len(entries) != len(typed_rows)
        or len({row["id"] for row in train_rows}) != len(train_rows)
    ):
        raise ValueError("External teacher needs the complete original TRAIN roster")
    selected: list[tuple[dict[str, Any], dict[str, float]]] = []
    selected_counts: collections.Counter[str] = collections.Counter()
    for row, entry in zip(typed_rows, entries, strict=True):
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "input_sha256",
            "group_id",
            "task_type",
            "source",
            "family",
            "language",
            "option_count",
            "probabilities",
        }:
            raise ValueError("Malformed external teacher row")
        for field in (
            "id",
            "input_sha256",
            "group_id",
            "task_type",
            "source",
            "family",
            "language",
        ):
            if entry[field] != row[field]:
                raise ValueError("External teacher row or native input differs")
        keys = [option["key"] for option in row["options"]]
        probabilities = entry["probabilities"]
        if (
            entry["option_count"] != len(keys)
            or not isinstance(probabilities, dict)
            or set(probabilities) != set(keys)
            or any(
                type(probabilities[key]) not in (int, float)
                or not math.isfinite(probabilities[key])
                or probabilities[key] < 0
                for key in keys
            )
            or abs(sum(probabilities.values()) - 1.0) > 1e-5
        ):
            raise ValueError("External teacher option parity or probability differs")
        if is_eligible(row, probabilities):
            selected.append((row, probabilities))
            selected_counts[row["task_type"]] += 1
    selected_roster = digest(
        [{"id": row["id"], "input_sha256": row["input_sha256"]} for row, _ in selected]
    )
    if (
        dict(selected_counts) != EXPECTED_ELIGIBLE
        or selected_roster != ELIGIBLE_ROSTER_SHA256
    ):
        raise ValueError("Frozen external teacher eligibility roster differs")
    for row, probabilities in selected:
        row["teacher_probs"] = probabilities
    return {
        "attached": len(selected),
        "by_type": dict(selected_counts),
        "eligible_roster_sha256": selected_roster,
        "teacher_artifact_sha256": ARTIFACT_SHA256,
        "teacher_model_sha256": TEACHER_MODEL_SHA256,
        "rule": "unique-gold-max; gold-p>=0.7; exclude stage4 composition; Score hard only",
    }
