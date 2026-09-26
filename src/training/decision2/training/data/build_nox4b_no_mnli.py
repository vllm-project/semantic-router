"""Derive an audited MultiNLI-free TRAIN from the frozen Nox structured mix.

This keeps every other source group and the exact SELECT/CAL bytes.  It is a
development candidate until the remaining source terms are reviewed; the
manifest deliberately does not grant a publication approval.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from training.data import build_nox4b_structured_mix as structured
from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

PARENT_TRAIN_SHA256 = "773cd53d21663095a4208e6e35de6654bdca4af5d314e23b03582dbe70eae87f"
PARENT_MANIFEST_SHA256 = (
    "336c039e22573c6bb0396bbedc3f86c04a25d9d126a933d4c9a6f1c6fcc0931f"
)
OUTPUT_NAME = "nox4b_no_mnli.train.jsonl"


def _origin_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, dict):
        return [text for nested in value.values() for text in _origin_strings(nested)]
    if isinstance(value, list):
        return [text for nested in value for text in _origin_strings(nested)]
    return []


def is_mnli_origin(row: dict[str, Any]) -> bool:
    """Use source lineage, not sample text, to identify inherited MNLI rows."""
    provenance = [
        row.get("source", ""),
        row.get("audit_metadata", {}).get("original_source", {}),
    ]
    return any(
        "multi_nli" in text or "multinli" in text or "multi-genre nli" in text
        for text in _origin_strings(provenance)
    )


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[field] for row in rows).items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sources = {
        "parent_train": (args.parent_train, PARENT_TRAIN_SHA256),
        "parent_manifest": (args.parent_manifest, PARENT_MANIFEST_SHA256),
        "select": (args.select_file, structured.SELECT_SHA256),
        "cal": (args.cal_file, structured.CAL_SHA256),
        "synthetic_dev": (args.dev_prompts, structured.DEV_SHA256),
        "css_pilot": (args.css_pilot_prompts, structured.CSS_PILOT_SHA256),
        "css_evaluation": (
            args.css_evaluation_prompts,
            structured.CSS_EVALUATION_SHA256,
        ),
    }
    for role, (path, digest) in sources.items():
        if pilot.sha_file(path) != digest:
            raise ValueError(f"Frozen {role} SHA-256 differs")
    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    if (
        parent_manifest["outputs"]["nox4b_structured.train.jsonl"]["sha256"]
        != PARENT_TRAIN_SHA256
    ):
        raise ValueError("Parent manifest TRAIN digest differs")
    parent = load_partition(args.parent_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(parent), len(select), len(cal)) != (8522, 600, 900):
        raise ValueError("Frozen partition counts differ")

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in parent:
        groups[row["group_id"]].append(row)
    removed, retained = [], []
    for rows in groups.values():
        flags = {is_mnli_origin(row) for row in rows}
        if len(flags) != 1:
            raise ValueError("MultiNLI origin mixes with other rows inside a group")
        (removed if True in flags else retained).extend(rows)
    if len(removed) != 267 or {row["source"] for row in removed} != {
        "legacy:nyu-mll/multi_nli"
    }:
        raise ValueError("Expected exactly the 267 pinned MultiNLI-origin rows")
    if len(retained) != 8255 or any(is_mnli_origin(row) for row in retained):
        raise AssertionError("Incomplete MultiNLI origin removal")
    if sorted(row["id"] for row in [*removed, *retained]) != sorted(
        row["id"] for row in parent
    ):
        raise AssertionError("Derivation changed a non-MultiNLI row")
    check_partition_isolation({"train": retained, "select": select, "cal": cal})
    if pilot.train_consistency_audit(retained)["conflicting_gold_groups"]:
        raise ValueError("Retained TRAIN has conflicting gold labels")

    references, reference_receipts = {}, {}
    for role, path, expected_name in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
        ("css_evaluation", args.css_evaluation_prompts, "css-evaluation.prompts.jsonl"),
    ):
        rows, receipt = targeted.load_context_reference(
            path, expected_name=expected_name
        )
        references[role] = rows
        reference_receipts[role] = receipt
    overlap = {
        role: targeted.context_overlap(retained, rows, approximate=True)
        for role, rows in {
            "select": targeted.context_rows(select),
            "cal": targeted.context_rows(cal),
            **references,
        }.items()
    }
    if any(
        any(
            audit[key]
            for key in (
                "id_rows",
                "group_id_rows",
                "input_sha256_rows",
                "raw_context_rows",
                "normalized_context_rows",
            )
        )
        or audit["near_context"]["count"]
        for audit in overlap.values()
    ):
        raise ValueError("Retained TRAIN overlaps a protected partition")

    payloads = {
        OUTPUT_NAME: pilot.jsonl_bytes(retained),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    report = {
        "schema_version": "decision2-nox4b-no-mnli/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "inputs": {
            role: {"file": path.name, "sha256": digest}
            for role, (path, digest) in sources.items()
        },
        "derivation": "whole-group removal of every MultiNLI-origin row; all other TRAIN rows unchanged",
        "removed_rows": len(removed),
        "removed_groups": len({row["group_id"] for row in removed}),
        "removed_source_counts": _counts(removed, "source"),
        "retained_source_counts": _counts(retained, "source"),
        "retained_family_counts": _counts(retained, "family"),
        "unchanged_non_mnli_added_quotas": {
            key: value
            for key, value in parent_manifest["realized_source_counts"].items()
            if key != "legacy:nyu-mll/multi_nli"
        },
        "reference_receipts": reference_receipts,
        "holdout_context_audits": overlap,
        "source_rights_status": {
            "multi_nli": "excluded entirely, including 67 inherited balanced-base rows",
            "cosmos_qa": "CC BY 4.0; attribute AllenAI/CosmosQA",
            "snli": "CC BY-SA 4.0; attribute Stanford SNLI; assess share-alike obligations",
            "squad_v2": "CC BY-SA 4.0; attribute Rajpurkar et al.; assess share-alike obligations",
            "flute": "AFL 3.0 in ColumbiaNLP dataset card; attribute and preserve notice",
            "stage3_replay": "128 CLINC150 CC BY 3.0; 81 Banking77 CC BY 4.0; 435 internal synthetic",
            "tweeteval": "3,600 rows; benchmark permits reuse but individual task and Twitter terms remain unresolved",
        },
        "release_gate": "development_only_pending_tweeteval_task_platform_and_sharealike_review",
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in payloads.items()
        },
        "limitations": [
            "Filtering MultiNLI changes the source mix and shrinks TRAIN by 267 rows; comparative gains are not a causal ablation.",
            "The 120 FLUTE TRAIN rows make CSS FLUTE same-task supervised.",
            "Approximate near-context screening does not establish semantic independence.",
            "No evaluation labels or raw training rows are part of this release audit.",
        ],
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(args.output_dir / name, payload)
    pilot._atomic_write(
        args.output_dir / "nox4b_no_mnli.manifest.json",
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-train",
        "parent-manifest",
        "select-file",
        "cal-file",
        "dev-prompts",
        "css-pilot-prompts",
        "css-evaluation-prompts",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    result = build(parser.parse_args(argv))
    print(
        json.dumps(
            {
                "rows": result["outputs"][OUTPUT_NAME]["rows"],
                "sha256": result["outputs"][OUTPUT_NAME]["sha256"],
                "release_gate": result["release_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
