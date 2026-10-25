"""Human-labelled source adapters and deterministic, source-group-disjoint pilot."""

from __future__ import annotations

import csv
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from .artifacts import digest, file_digest, read_json, write_json
from .sources import SOURCES

SPLITS = ("train", "calibration", "held_out")
LEVELS = ["negative", "neutral", "positive"]
MINIMUM_SOURCE_GROUPS = 12


def _example(
    source: str,
    original_id: str,
    group: str,
    state: object,
    questions: dict,
    labels: dict,
) -> dict:
    return {
        "id": f"{source}:{original_id}",
        "source": source,
        "source_id": original_id,
        "group_id": f"{source}:{group}",
        "cohort": "public",
        "request": {
            "state": state,
            "questions": questions,
            "options": {"return_meta": True},
        },
        "labels": labels,
    }


def _question(kind: str, instructions: str, criteria: object = None) -> dict:
    value = {"type": kind, "instructions": instructions, "require_full_input": True}
    if criteria is not None:
        value["criteria"] = criteria
    return value


def banking(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    options = {
        label: label.replace("_", " ")
        for label in sorted({row["category"] for row in rows})
    }
    question = _question(
        "choice", "Which banking intent best describes the customer's request?", options
    )
    return [
        _example(
            "banking77",
            str(index),
            digest(row["text"].strip().casefold()),
            row["text"],
            {"intent": question},
            {"intent": {"label": row["category"]}},
        )
        for index, row in enumerate(rows)
    ]


def boolq(path: Path) -> list[dict]:
    output = []
    with zipfile.ZipFile(path) as archive:
        rows = [
            json.loads(line)
            for line in archive.read("BoolQ/train.jsonl").decode().splitlines()
            if line.strip()
        ]
    for index, row in enumerate(rows):
        if type(row["label"]) is not bool:
            raise ValueError("BoolQ source answer must be a human-labelled boolean")
        output.append(
            _example(
                "boolq",
                str(index),
                digest(row.get("title") or row["passage"].split(" -- ", 1)[0]),
                row["passage"],
                {"answer": _question("noul", row["question"])},
                {"answer": {"label": str(row["label"]).lower()}},
            )
        )
    return output


def dynasent(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith("round02-dynabench-train.jsonl")
            and not name.startswith("__MACOSX/")
        ]
        if len(names) != 1:
            raise ValueError("expected exactly one DynaSent Round 2 training file")
        rows = [
            json.loads(line)
            for line in archive.read(names[0]).decode().splitlines()
            if line.strip()
        ]
    output = []
    for row in rows:
        label = row.get("gold_label")
        if label not in LEVELS:
            continue
        counts = {
            str(i): len(row["label_distribution"].get(level, []))
            for i, level in enumerate(LEVELS)
        }
        total = sum(counts.values())
        if not total:
            raise ValueError("DynaSent example has no human sentiment annotations")
        # Keep the original full annotation counts, including mixed/other votes.
        prompt = row.get("prompt_data") or {}
        group = prompt.get("review_id") or row["text_id"]
        question = _question(
            "score",
            "Rate the sentiment expressed by this sentence on the ordered negative, neutral, positive scale.",
            LEVELS,
        )
        output.append(
            _example(
                "dynasent",
                row["text_id"],
                str(group),
                row["sentence"],
                {"sentiment": question},
                {
                    "sentiment": {
                        "label": str(LEVELS.index(label)),
                        "distribution": {
                            key: count / total for key, count in counts.items()
                        },
                        "annotation_counts": {
                            key: len(votes)
                            for key, votes in row["label_distribution"].items()
                        },
                        "distribution_conditioning": "negative/neutral/positive votes; full original counts retained",
                    }
                },
            )
        )
    return output


def _add_bundle(row: dict) -> None:
    """Two deterministic relabellings of the SAME annotation, not new samples."""
    label = int(row["labels"]["sentiment"]["label"])
    row["request"]["questions"].update(
        {
            "sentiment_name": _question(
                "choice",
                "Which sentiment does this sentence express?",
                {key: key for key in LEVELS},
            ),
            "positive": _question(
                "noul", "Does this sentence express positive sentiment?"
            ),
        }
    )
    row["labels"].update(
        {
            "sentiment_name": {"label": LEVELS[label], "derived_from": "sentiment"},
            "positive": {
                "label": str(LEVELS[label] == "positive").lower(),
                "derived_from": "sentiment",
            },
        }
    )
    row["bundle_annotation"] = (
        "Three views of one human sentiment judgement; not three independent labels"
    )


def synthetic_edges() -> list[dict]:
    """Exact arithmetic predicates; held-out diagnostics, never fit/calibrate on them."""
    rows = []
    for index in range(32):
        amount = index - 16
        threshold = index % 7 - 3
        difference = amount - threshold
        label = "below" if difference < 0 else "equal" if difference == 0 else "above"
        questions = {
            "comparison": _question(
                "choice",
                "Compare amount with threshold. Use only the supplied numeric values.",
                {key: key for key in ("below", "equal", "above")},
            ),
            "exceeds": _question("noul", "Is amount strictly greater than threshold?"),
            "distance": _question(
                "score",
                "Rate absolute distance between amount and threshold: zero, one, or at least two.",
                ["zero", "one", "at least two"],
            ),
        }
        row = _example(
            "synthetic",
            str(index),
            f"comparison-{index // 4}",
            {"amount": amount, "threshold": threshold},
            questions,
            {
                "comparison": {"label": label},
                "exceeds": {"label": str(difference > 0).lower()},
                "distance": {"label": str(min(abs(difference), 2))},
            },
        )
        row.update(
            cohort="synthetic_edge",
            split="held_out",
            label_provenance="deterministic arithmetic, not model-labelled",
        )
        rows.append(row)
    return rows


def prepare(
    directory: Path, output: Path, *, per_source: int = 128, seed: int = 42
) -> dict:
    if per_source < MINIMUM_SOURCE_GROUPS or per_source % 4:
        raise ValueError("per_source must be at least 12 and divisible by four")
    source_receipt = read_json(directory / "sources.json")
    records = []
    for name, adapter in (
        ("banking77", banking),
        ("boolq", boolq),
        ("dynasent", dynasent),
    ):
        path = directory / SOURCES[name]["filename"]
        if file_digest(path) != source_receipt["sources"][name]["sha256"]:
            raise ValueError(f"{name} source checksum mismatch")
        # One representative per source group prevents article/review siblings leaking.
        groups, texts = {}, set()
        ordered = sorted(adapter(path), key=lambda row: digest([seed, row["id"]]))
        for row in ordered:
            text = digest(str(row["request"]["state"]).strip().casefold())
            if row["group_id"] not in groups and text not in texts:
                groups[row["group_id"]] = row
                texts.add(text)
        selected = list(groups.values())[:per_source]
        if len(selected) != per_source:
            raise ValueError(f"not enough unique source groups for {name}")
        for index, row in enumerate(selected):
            row["split"] = SPLITS[
                (
                    0
                    if index < per_source // 2
                    else 1 if index < per_source * 3 // 4 else 2
                )
            ]
            if name == "dynasent" and index % 2 == 0:
                _add_bundle(row)
            records.append(row)
    random.Random(seed).shuffle(records)
    records.extend(synthetic_edges())
    result = {
        "schema_version": "systemone-pilot/v1",
        "seed": seed,
        "source_receipt": source_receipt,
        "records": records,
        "split_method": "source-group-disjoint 50/25/25 research partitions of official training releases",
        "counts": {
            "public_unique_sources": 3 * per_source,
            "synthetic_edges": 32,
            "public_splits": dict(
                Counter(row["split"] for row in records if row["cohort"] == "public")
            ),
        },
        "limitations": [
            "Pilot, not a broad capability benchmark",
            "Unknown foundation-model pretraining overlap",
            "Score covers three-level ordinal sentiment only",
            "Mixed sentiment questions share one annotation",
            "Synthetic edge suite is diagnostic and excluded from policy fitting and primary aggregate",
        ],
    }
    result["records_sha256"] = digest(records)
    validate_dataset(result)
    write_json(output, result)
    return result


def validate_dataset(dataset: dict) -> None:
    if dataset.get("schema_version") != "systemone-pilot/v1" or not dataset.get(
        "records"
    ):
        raise ValueError("invalid or empty dataset manifest")
    if digest(dataset["records"]) != dataset.get("records_sha256"):
        raise ValueError("dataset record digest mismatch")
    seen, groups, texts = set(), {}, {}
    for row in dataset["records"]:
        if row["id"] in seen or row["split"] not in SPLITS:
            raise ValueError("duplicate ID or invalid split")
        seen.add(row["id"])
        for lookup, key in (
            (groups, row["group_id"]),
            (texts, digest(str(row["request"]["state"]).strip().casefold())),
        ):
            if key in lookup and lookup[key] != row["split"]:
                raise ValueError("source group or duplicate state leaks across splits")
            lookup[key] = row["split"]
        if set(row["labels"]) != set(row["request"]["questions"]):
            raise ValueError(
                "every question needs exactly one independent or declared derived label"
            )
