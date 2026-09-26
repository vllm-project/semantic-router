"""Convert audited Decision 2.0 TRAIN rows into Kev's native labelled requests.

The converter checks every record with the pinned Kev encoder before writing a
training file. Kev's own trainer filters overlong custom records, so a plain
format conversion without that check would silently change the train set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inference.kev import (
    KEV_BASE_ID,
    KEV_BASE_REVISION,
    KEV_MODEL_ID,
    KEV_MODEL_REVISION,
    KEV_SOURCE_REVISION,
    verify_provenance,
)
from inference.run import load_prompts, local_revision

from training.model.data import (
    check_partition_isolation,
    digest,
    file_sha256,
    load_partition,
)

FORMAT = "decision2-kev-native-train/1"
QUESTION_ID = "decision"


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve native type semantics while mapping an ordered flat row."""
    options = row["options"]
    selected = options[row["label"]]["key"]
    kind = row["task_type"]
    if kind == "choice":
        criteria: Any = {option["key"]: option["description"] for option in options}
        label: Any = selected
    elif kind == "noul":
        criteria = {option["key"]: option["description"] for option in options}
        label = selected == "true"
    elif kind == "score":
        # Kev's Score labels index ascending rubric levels. The flat contract
        # permits option permutations, so the flat label is an option index.
        criteria = [
            option["description"]
            for option in sorted(options, key=lambda o: int(o["key"]))
        ]
        label = int(selected)
    else:
        raise ValueError(f"{row['id']}: unsupported native type {kind}")
    question = {
        "type": kind,
        "instructions": row["instructions"],
        "criteria": criteria,
        "label": label,
        "src": f"{row['source']}/{row['family']}/{kind}",
    }
    return {
        "state": row["state"],
        "questions": {QUESTION_ID: question},
        "_meta": {
            "id": row["id"],
            "group_id": row["group_id"],
            "source": row["source"],
            "split": "train",
            "input_sha256": row["input_sha256"],
            "family": row["family"],
            "language": row["language"],
            "render_template": row["render_template"],
        },
    }


def _question_signature(
    state: Any, kind: str, instructions: Any, options: list[dict[str, Any]]
) -> str:
    if kind == "score":
        normalized = [
            (option["key"], option["description"])
            for option in sorted(options, key=lambda option: int(option["key"]))
        ]
    else:
        normalized = sorted(
            (option["key"], option["description"]) for option in options
        )
    return digest(
        {
            "state": state,
            "type": kind,
            "instructions": instructions,
            "options": normalized,
        }
    )


def _flat_signature(row: dict[str, Any]) -> str:
    return _question_signature(
        row["state"], row["task_type"], row["instructions"], row["options"]
    )


def _prompt_signatures(row: dict[str, Any]) -> list[str]:
    result = []
    for question in row["questions"].values():
        if not isinstance(question, dict) or question.get("type") not in {
            "choice",
            "noul",
            "score",
        }:
            raise ValueError(
                f"{row['id']}: audit prompt contains an invalid native question"
            )
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "score":
            if not isinstance(criteria, list):
                raise ValueError(f"{row['id']}: Score audit criteria must be a list")
            options = [
                {"key": str(index), "description": value}
                for index, value in enumerate(criteria)
            ]
        else:
            if criteria is None and kind == "noul":
                criteria = {}
            if not isinstance(criteria, dict):
                raise ValueError(f"{row['id']}: audit criteria must be an object")
            options = [
                {"key": key, "description": value} for key, value in criteria.items()
            ]
        result.append(
            _question_signature(
                row["state"], kind, question.get("instructions"), options
            )
        )
    return result


def audit_overlap(
    train: list[dict[str, Any]],
    select: list[dict[str, Any]],
    cal: list[dict[str, Any]],
    prompt_sets: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Fail closed on IDs, lineage groups, ordered inputs, states and question semantics."""
    partitions = {"train": train, "select": select, "cal": cal}
    check_partition_isolation(partitions)
    train_inputs = {row["input_sha256"] for row in train}
    if len(train_inputs) != len(train):
        raise ValueError("TRAIN contains duplicate canonical input payloads")
    seen_ids: dict[str, str] = {}
    seen_states: dict[str, str] = {}
    seen_questions: dict[str, str] = {}
    for role, rows in partitions.items():
        for row in rows:
            for name, key, seen in (
                ("ID", row["id"], seen_ids),
                ("state", digest(row["state"]), seen_states),
                ("question", _flat_signature(row), seen_questions),
            ):
                previous = seen.get(key)
                if previous is not None and previous != role:
                    raise ValueError(f"{role} {name} overlaps {previous}: {row['id']}")
                seen[key] = role
    audits = {}
    for path, rows in prompt_sets.items():
        if not rows:
            raise ValueError(f"{path}: empty prompt audit")
        for row in rows:
            if row["id"] in seen_ids or digest(row["state"]) in seen_states:
                raise ValueError(
                    f"{path}: prompt ID or state overlaps a train/select/cal partition: {row['id']}"
                )
            if any(
                signature in seen_questions for signature in _prompt_signatures(row)
            ):
                raise ValueError(
                    f"{path}: prompt question overlaps a train/select/cal partition: {row['id']}"
                )
        audits[str(path)] = {
            "sha256": file_sha256(path),
            "items": len(rows),
            "questions": sum(len(row["questions"]) for row in rows),
        }
    return {
        "policy": "Cross-role ID/group/input/state/question exact hashes; gold-free prompts only",
        "train_unique_ids": len(train),
        "train_unique_inputs": len(train_inputs),
        "select_items": len(select),
        "cal_items": len(cal),
        "gold_free_prompt_audits": audits,
    }


def verify_parent(model_path: Path, source_path: Path) -> dict[str, Any]:
    provenance = verify_provenance(model_path, source_path)
    if not local_revision(model_path, KEV_MODEL_REVISION):
        raise ValueError(
            "Kev parent local download lacks exact HF revision attestation"
        )
    measured = provenance.get("measured_checkpoint", {})
    head_sha = file_sha256(model_path / "head.pt")
    # The release fitted a temperature after the measured training checkpoint
    # was saved. That updates head.pt bytes, so its measured pre-calibration
    # digest in provenance is not the published head.pt digest. Bind warm starts
    # to the exact attested published file actually loaded by Checkpoint.
    return {
        "model_id": KEV_MODEL_ID,
        "model_revision": KEV_MODEL_REVISION,
        "source_revision": KEV_SOURCE_REVISION,
        "base_id": KEV_BASE_ID,
        "base_revision": KEV_BASE_REVISION,
        "adapter_sha256": measured["adapter_sha256"],
        "head_sha256": head_sha,
        "measured_head_pre_calibration_sha256": measured.get("head_sha256"),
        "provenance_sha256": file_sha256(model_path / "provenance.json"),
    }


def native_preflight(
    path: Path, converted: list[dict[str, Any]], source_path: Path, max_state: int
) -> dict[str, Any]:
    """Use the same pinned load/materialize/fits path as kev.train --data."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    package = sys.modules.get("kev")
    package_file = getattr(package, "__file__", None) if package is not None else None
    if package is not None and (
        package_file is None
        or Path(package_file).resolve().parent != (source_path / "kev").resolve()
    ):
        raise RuntimeError("Another Kev package is already imported")
    sys.path.insert(0, str(source_path))
    from kev.data import load_records, materialize
    from kev.model import encode, fits, load_tokenizer, training_context

    requests = load_records(path)
    if len(requests) != len(converted):
        raise ValueError("Kev native loader changed converted row count")
    tokenizer = load_tokenizer(KEV_BASE_ID, revision=KEV_BASE_REVISION)
    context = training_context(max_state)
    maximum = 0
    for expected, request in zip(converted, requests):
        if request["_meta"]["id"] != expected["_meta"]["id"]:
            raise ValueError("Kev native loader changed row order or identity")
        rec = materialize(request)
        if len(rec["questions"]) != 1:
            raise ValueError("Kev native loader changed question count")
        question = rec["questions"][0]
        original = expected["questions"][QUESTION_ID]
        if question["qtype"] != original["type"]:
            raise ValueError("Kev native loader changed question type")
        expected_keys = (
            list(original["criteria"])
            if original["type"] == "choice"
            else (
                ["false", "true"]
                if original["type"] == "noul"
                else [str(i) for i in range(len(original["criteria"]))]
            )
        )
        expected_label = (
            expected_keys.index(original["label"])
            if original["type"] == "choice"
            else int(original["label"])
        )
        if question["keys"] != expected_keys or question["label"] != expected_label:
            raise ValueError("Kev native loader changed option or gold semantics")
        if not fits(rec, tokenizer, **context):
            raise ValueError(
                f"{request['_meta']['id']}: Kev would silently drop this overlong TRAIN row"
            )
        encoded = encode(
            tokenizer,
            rec,
            max_state=context["max_state"],
            max_branch=context["max_branch"],
            strict=True,
        )
        if encoded["state_truncated"] or len(encoded["ids"]) > context["max_packed"]:
            raise ValueError(
                f"{request['_meta']['id']}: native encoding truncated or exceeded budget"
            )
        maximum = max(maximum, len(encoded["ids"]))
    return {
        "records": len(requests),
        "questions": len(requests),
        "max_encoded_tokens": maximum,
        "training_context": context,
        "native_source_revision": KEV_SOURCE_REVISION,
    }


def prepare(
    train_path: Path,
    select_path: Path,
    cal_path: Path,
    model_path: Path,
    source_path: Path,
    output: Path,
    *,
    max_state: int = 7552,
    audit_prompts: tuple[Path, ...] = (),
    native_check: Callable[
        [Path, list[dict[str, Any]], Path, int], dict[str, Any]
    ] = native_preflight,
    parent_check: Callable[[Path, Path], dict[str, Any]] = verify_parent,
) -> dict[str, Any]:
    output = output.resolve()
    manifest_path = output.with_name(output.name + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Kev training output or manifest already exists")
    train = load_partition(train_path, "train")
    select = load_partition(select_path, "select")
    cal = load_partition(cal_path, "cal")
    prompts = {path: load_prompts(path) for path in audit_prompts}
    audit = audit_overlap(train, select, cal, prompts)
    parent = parent_check(model_path, source_path)
    converted = [convert_row(row) for row in train]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="kev-prepare-", dir=output.parent
    ) as directory:
        staged = Path(directory) / output.name
        with staged.open("x", encoding="utf-8") as stream:
            for row in converted:
                # Dict insertion order is the native Choice option order.
                # Canonical JSON sorts keys and would silently reorder it.
                stream.write(
                    json.dumps(
                        row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        native = native_check(staged, converted, source_path, max_state)
        if native.get("records") != len(train) or native.get("questions") != len(train):
            raise ValueError(
                "Native preflight did not cover every train row and question"
            )
        manifest = {
            "format": FORMAT,
            "train_sha256": file_sha256(train_path),
            "select_sha256": file_sha256(select_path),
            "cal_sha256": file_sha256(cal_path),
            "kev_train_sha256": file_sha256(staged),
            "rows": len(train),
            "family_counts": dict(
                sorted(Counter(row["family"] for row in train).items())
            ),
            "type_counts": dict(
                sorted(Counter(row["task_type"] for row in train).items())
            ),
            "source_counts": dict(
                sorted(Counter(row["source"] for row in train).items())
            ),
            "overlap_audit": audit,
            "parent": parent,
            "native_preflight": native,
            "conversion_code_sha256": file_sha256(Path(__file__)),
        }
        staged_manifest = Path(directory) / manifest_path.name
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with staged_manifest.open("rb") as stream:
            os.fsync(stream.fileno())
        if output.exists() or manifest_path.exists():
            raise FileExistsError("Kev training output appeared during preflight")
        os.replace(staged, output)
        os.replace(staged_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "select", "cal", "model-path", "source-path", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--audit-prompts",
        type=Path,
        action="append",
        default=[],
        help="Gold-free benchmark or transfer prompt JSONL; repeat as needed",
    )
    parser.add_argument("--max-state", type=int, default=7552)
    args = parser.parse_args()
    report = prepare(
        args.train,
        args.select,
        args.cal,
        args.model_path,
        args.source_path,
        args.output,
        max_state=args.max_state,
        audit_prompts=tuple(args.audit_prompts),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": report["rows"],
                "kev_train_sha256": report["kev_train_sha256"],
                "manifest": str(args.output) + ".manifest.json",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
