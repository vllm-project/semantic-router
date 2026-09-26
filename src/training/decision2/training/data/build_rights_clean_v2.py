"""Add official human GoEmotions TRAIN labels to the rights-clean control.

The public source is restricted to its official TRAIN split. Distinct comment
groups are reserved for training, selection and calibration; all selected
contexts are quarantined against the frozen Decision 2.0 panels. The existing
synthetic-only v1 holdouts remain as structured anchors.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

V1_SHA = {
    "rights_clean.train.jsonl": "4973f999b7a19c69a8d7236ab941556e788a9ab69f208c015d3768fd7c86841b",
    "select.jsonl": "1d564becab12717f7883c77131b8e8611a2e495c102a71789a1b1e5c2cf9afd4",
    "cal.jsonl": "35c27a2a16271b7295afa2d65474dbdc7c64742fdce23ba26bb2d3792a0921d6",
    "rights_clean.manifest.json": "a4ded3bc13f5dcc9cffbd98899714507f17d29b0f4084cc0319f030a32728bdf",
}
GOEMOTIONS_REVISION = "2adf640a14f11025ae5a9d0ec493b78530d276d3"
GOEMOTIONS_SHA = {
    "train.tsv": "1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5",
    "emotions.txt": "45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89",
}
REFERENCE_SHA = {
    "synthetic_dev": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "css_pilot": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    "css_evaluation": "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    "rq1": "f1584742583dacab334e0f5ca7702e7248991d7f58f41176e87e14b14446aa19",
    "rq2": "e094f222e01240f22b2cc508893ce554a3850961958f72ddb6d20551572b9df8",
    "rq3": "be73de25f7fd3202d2a7dfd8831c338c70cce344a01551313eab066670055b32",
}
SEED = "decision2-rights-clean-human-goemotions-v2"
SOURCE = "google_goemotions_official_train"
TRAIN_GROUPS = 1400
HOLDOUT_GROUPS = 200
VALENCE_GROUPS = (
    (
        "admiration",
        "amusement",
        "approval",
        "caring",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    ),
    (
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "fear",
        "grief",
        "nervousness",
        "remorse",
        "sadness",
    ),
    ("confusion", "curiosity", "desire", "realization", "surprise"),
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[field] for row in rows).items()))


def _load_source(directory: Path) -> tuple[list[dict[str, str]], list[str]]:
    for name, digest in GOEMOTIONS_SHA.items():
        if pilot.sha_file(directory / name) != digest:
            raise ValueError(f"Official GoEmotions {name} bytes changed")
    labels = (directory / "emotions.txt").read_text(encoding="utf-8").splitlines()
    if len(labels) != 28 or labels[-1] != "neutral":
        raise ValueError("Official GoEmotions label order changed")
    source = []
    seen_ids = set()
    with (directory / "train.tsv").open(encoding="utf-8", newline="") as stream:
        for text, label_ids, source_id in csv.reader(stream, delimiter="\t"):
            if source_id in seen_ids:
                raise ValueError("GoEmotions comment ID was repeated")
            seen_ids.add(source_id)
            ids = [int(value) for value in label_ids.split(",")]
            if len(ids) != 1 or ids[0] == 27:
                continue
            text = " ".join(text.split())
            if not 12 <= len(text) <= 400 or not text.isprintable():
                continue
            source.append({"id": source_id, "text": text, "emotion": labels[ids[0]]})
    if len(seen_ids) != 43410 or len(source) < 20000:
        raise ValueError("Official GoEmotions TRAIN inventory changed")
    return source, labels


def _make_pair(source: dict[str, str], role: str, seed: str) -> list[dict[str, Any]]:
    text, emotion, identifier = source["text"], source["emotion"], source["id"]
    neighbours = next(group for group in VALENCE_GROUPS if emotion in group)
    distractors = sorted(
        (value for value in neighbours if value != emotion),
        key=lambda value: _hash(f"{seed}\0distractor\0{identifier}\0{value}"),
    )[:3]
    choices = sorted(
        [emotion, *distractors],
        key=lambda value: _hash(f"{seed}\0choice\0{identifier}\0{value}"),
    )
    queried = (
        emotion
        if int(_hash(f"{seed}\0polarity\0{identifier}")[:2], 16) % 2 == 0
        else sorted(
            (
                value
                for group in VALENCE_GROUPS
                if emotion not in group
                for value in group
            ),
            key=lambda value: _hash(f"{seed}\0negative\0{identifier}\0{value}"),
        )[0]
    )
    group_id = f"goemotions-official-train:{identifier}"
    common = {
        "state": text,
        "group_id": group_id,
        "language": "en",
        "split": role,
        "source": SOURCE,
        "evaluation_role": (
            "train"
            if role == "train"
            else ("select" if role == "select" else "calibrate")
        ),
        "render_template": "goemotions_human_annotation_v1",
        "audit_metadata": {
            "original_source": {
                "dataset": "google-research/goemotions",
                "official_split": "train",
                "revision": GOEMOTIONS_REVISION,
                "source_id": identifier,
                "human_label": emotion,
            },
            "projection_version": 1,
            "label_note": "Filtered human annotation; emotion judgments are subjective and may be incomplete.",
        },
    }
    choice = {
        **common,
        "id": f"goemotions-v2:{identifier}:choice",
        "instructions": "Which emotion best describes the comment? Choose from the supplied labels.",
        "options": [{"key": value, "description": value} for value in choices],
        "label": choices.index(emotion),
        "task_type": "choice",
        "family": "human_goemotions_choice",
    }
    noul = {
        **common,
        "id": f"goemotions-v2:{identifier}:noul",
        "instructions": f"Does the comment express {queried}?",
        "options": [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ],
        "label": int(queried == emotion),
        "task_type": "noul",
        "family": "human_goemotions_noul",
    }
    for row in (choice, noul):
        row["input_sha256"] = pilot.input_sha256(row)
        pilot.validate_train_row({**row, "split": "train", "evaluation_role": "train"})
    return [choice, noul]


def _contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return targeted.context_rows(rows)


def _remove_overlaps(
    candidates: list[dict[str, str]],
    protected: list[dict[str, Any]],
    seed: str,
    role: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Quarantine complete source comment groups on exact/near state overlap."""
    protected_ids = {row["id"] for row in protected}
    protected_groups = {row.get("group_id") for row in protected}
    protected_raw = {targeted.text_hashes(row["state"]) for row in protected}
    raw = {pair[0] for pair in protected_raw}
    normalized = {pair[1] for pair in protected_raw}
    eligible, rejected = [], collections.Counter()
    for source in candidates:
        rows = _make_pair(source, role, seed)
        fingerprints = targeted.text_hashes(source["text"])
        if any(
            row["id"] in protected_ids or row["group_id"] in protected_groups
            for row in rows
        ):
            rejected["id_group"] += 1
        elif fingerprints[0] in raw or fingerprints[1] in normalized:
            rejected["exact_state"] += 1
        else:
            eligible.append(source)
    near_rows = [_make_pair(item, role, seed)[0] for item in eligible]
    near = pilot.near_duplicates(
        _contexts(near_rows), _contexts(protected), collect_left_ids=True
    )
    near_ids = set(near.pop("left_ids"))
    kept = [item for item, row in zip(eligible, near_rows) if row["id"] not in near_ids]
    rejected["near_state"] = len(eligible) - len(kept)
    return kept, {
        "input_comments": len(candidates),
        "eligible_comments": len(kept),
        "removed_comments": dict(sorted(rejected.items())),
        "near_method": near["method"],
    }


def _select_balanced(
    source: list[dict[str, str]], count: int, seed: str
) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for item in source:
        by_label[item["emotion"]].append(item)
    for label in by_label:
        by_label[label].sort(key=lambda item: _hash(f"{seed}\0balanced\0{item['id']}"))
    labels = sorted(by_label)
    selected = []
    while len(selected) < count:
        progressed = False
        for label in labels:
            if by_label[label] and len(selected) < count:
                selected.append(by_label[label].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("Insufficient isolated GoEmotions comments")
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for name, digest in V1_SHA.items():
        if pilot.sha_file(args.v1_dir / name) != digest:
            raise ValueError(f"Frozen rights-clean v1 {name} changed")
    v1_manifest = json.loads((args.v1_dir / "rights_clean.manifest.json").read_text())
    if v1_manifest.get("schema_version") != "decision2-rights-clean-splits/1":
        raise ValueError("Unrecognized rights-clean parent")
    base = {
        role: load_partition(args.v1_dir / name, role)
        for role, name in (
            ("train", "rights_clean.train.jsonl"),
            ("select", "select.jsonl"),
            ("cal", "cal.jsonl"),
        )
    }
    source, labels = _load_source(args.goemotions_dir)
    references = {}
    for role in REFERENCE_SHA:
        path = getattr(args, role)
        if pilot.sha_file(path) != REFERENCE_SHA[role]:
            raise ValueError(f"Protected {role} context SHA changed")
        references[role], _ = targeted.load_context_reference(path)
    benchmark_protected = [row for rows in references.values() for row in rows]
    protected = [*benchmark_protected, *(row for rows in base.values() for row in rows)]
    by_label: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for item in source:
        by_label[item["emotion"]].append(item)
    train_pool = [
        item
        for label in sorted(by_label)
        for item in sorted(
            by_label[label],
            key=lambda row: _hash(f"{args.seed}\0train-pool\0{row['id']}"),
        )[:200]
    ]
    train_pool, train_filter = _remove_overlaps(
        train_pool, protected, args.seed, "train"
    )
    train_groups = _select_balanced(train_pool, TRAIN_GROUPS, args.seed)
    human_train = [
        row for item in train_groups for row in _make_pair(item, "train", args.seed)
    ]
    selected_ids = {item["id"] for item in train_groups}
    holdout_pool = sorted(
        (item for item in source if item["id"] not in selected_ids),
        key=lambda item: _hash(f"{args.seed}\0holdout-pool\0{item['id']}"),
    )[:3500]
    holdout_pool, select_filter = _remove_overlaps(
        holdout_pool, [*protected, *human_train], args.seed, "select"
    )
    selected = holdout_pool[:HOLDOUT_GROUPS]
    if len(selected) != HOLDOUT_GROUPS:
        raise ValueError("Insufficient isolated human SELECT groups")
    human_select = [
        row for item in selected for row in _make_pair(item, "select", args.seed)
    ]
    remaining = holdout_pool[HOLDOUT_GROUPS:]
    remaining, cal_filter = _remove_overlaps(remaining, human_select, args.seed, "cal")
    calibrated = remaining[:HOLDOUT_GROUPS]
    if len(calibrated) != HOLDOUT_GROUPS:
        raise ValueError("Insufficient isolated human CAL groups")
    human_cal = [
        row for item in calibrated for row in _make_pair(item, "cal", args.seed)
    ]
    partitions = {
        "train": [*base["train"], *human_train],
        "select": [*base["select"], *human_select],
        "cal": [*base["cal"], *human_cal],
    }
    for role, rows in partitions.items():
        rows.sort(
            key=lambda row: (_hash(f"{args.seed}\0{role}\0{row['id']}"), row["id"])
        )
    check_partition_isolation(partitions)
    if pilot.train_consistency_audit(partitions["train"])["conflicting_gold_groups"]:
        raise ValueError("TRAIN has contradictory gold for identical typed input")
    audits = {}
    for role, rows in partitions.items():
        for reference_name, reference_rows in references.items():
            audits[f"{role}_vs_{reference_name}"] = targeted.context_overlap(
                rows, reference_rows, approximate=True
            )
    for left, right in (("train", "select"), ("train", "cal"), ("select", "cal")):
        audits[f"{left}_vs_{right}"] = targeted.context_overlap(
            partitions[left], partitions[right], approximate=True
        )
    payloads = {
        (
            "rights_clean.train.jsonl" if role == "train" else f"{role}.jsonl"
        ): pilot.jsonl_bytes(rows)
        for role, rows in partitions.items()
    }
    source_rights = [
        item
        for item in v1_manifest["source_rights"]
        if item["source"] != "oracle SELECT/CAL"
    ]
    source_rights.extend(
        [
            {
                "source": "GoEmotions official TRAIN",
                "rows": len(human_train),
                "partition_scope": "TRAIN",
                "license": "CC BY 4.0",
                "evidence": "https://github.com/google-research/google-research/ ; https://github.com/google-research/google-research/tree/master/goemotions",
            },
            {
                "source": "GoEmotions official TRAIN",
                "rows": len(human_select) + len(human_cal),
                "partition_scope": "SELECT/CAL",
                "license": "CC BY 4.0",
                "evidence": "https://github.com/google-research/google-research/",
            },
            {
                "source": "oracle SELECT/CAL",
                "rows": len(base["select"]) + len(base["cal"]),
                "partition_scope": "SELECT/CAL",
                "license": "internally generated",
                "evidence": "parent rights-clean v1 generator SHA",
            },
        ]
    )
    report = {
        "schema_version": "decision2-rights-clean-splits/1",
        "derivation_version": "decision2-goemotions-human-v2/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "inputs": {
            "rights_clean_v1": {"sha256": V1_SHA},
            "goemotions_official_train": {
                "revision": GOEMOTIONS_REVISION,
                "sha256": GOEMOTIONS_SHA,
                "official_split": "train",
            },
            "protected_context_sha256": REFERENCE_SHA,
        },
        "counts": {
            field: _counts(partitions["train"], field)
            for field in ("source", "family", "task_type", "language")
        },
        "partition_counts": {
            role: {
                field: _counts(rows, field)
                for field in ("source", "family", "task_type", "language")
            }
            for role, rows in partitions.items()
        },
        "human_source": {
            "dataset": "GoEmotions",
            "source_split": "train",
            "source_rows": 43410,
            "eligible_single_non_neutral": len(source),
            "train_groups": len(train_groups),
            "select_groups": len(selected),
            "cal_groups": len(calibrated),
            "label_names": labels,
        },
        "group_quarantine": {
            "train": train_filter,
            "select": select_filter,
            "cal": cal_filter,
        },
        "overlap_audits": audits,
        "source_rights": source_rights,
        "publication_eligible": True,
        "publication_scope": v1_manifest["publication_scope"],
        "publication_conditions": [
            *v1_manifest["publication_conditions"],
            "attribute Google Research GoEmotions and its CC BY 4.0 dataset terms",
        ],
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "rows": len(payload.splitlines()),
                "bytes": len(payload),
            }
            for name, payload in payloads.items()
        },
        "limitations": [
            "GoEmotions annotations are subjective; single-label non-neutral selection does not represent the original corpus prevalence.",
            "Choice negatives are same-valence distractors and binary negatives use a different valence; this projection changes the original annotation task.",
            "SELECT and CAL include only 200 human comment groups each, supplementing 300 synthetic oracles each; transfer calibration remains narrow.",
            "Approximate near-context screening is not a mathematical guarantee of semantic independence.",
            "The base model's pretraining and earlier fine-tuning rights require separate audit.",
        ],
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(args.output_dir / name, payload)
    pilot._atomic_write(
        args.output_dir / "rights_clean.manifest.json",
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--goemotions-dir", type=Path, required=True)
    for role in REFERENCE_SHA:
        parser.add_argument(
            "--" + role.replace("_", "-"), dest=role, type=Path, required=True
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    report = build(args)
    print(
        json.dumps(
            {
                "counts": {
                    name: value["rows"] for name, value in report["outputs"].items()
                },
                "sha256": {
                    name: value["sha256"] for name, value in report["outputs"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
